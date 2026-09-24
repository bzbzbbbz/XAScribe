"""Reproduce the headline numbers of the XAScribe paper from the bundled data.

    python paper/reproduce_numbers.py            -> paper/outputs/numbers.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xascribe import model, preprocess as pp, report  # noqa: E402

OUT = ROOT / "paper" / "outputs"; OUT.mkdir(parents=True, exist_ok=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def tallman_series():
    st = pd.read_csv(ROOT / "data/tallman2021/states.csv", header=None, names=["colour", "x", "V"])
    sp = [(f"x = {r.x:.2f}, {r.V:.2f} V", *pp.read_spectrum(ROOT / f"data/tallman2021/spectrum_{r.colour.replace(' ', '_')}.csv"))
          for r in st.itertuples()]
    ref_ox = pd.read_csv(ROOT / "data/tallman2021/reference_ni_valence_redigitized.csv").sort_values("x")
    ref_bl = pd.read_csv(ROOT / "data/tallman2021/reference_ni_o_exafs.csv", header=None, names=["x", "d"]).sort_values("x")
    rox = np.interp(st.x, ref_ox.x, ref_ox.valence_com)
    rbl = np.interp(st.x, ref_bl.x, ref_bl.d)
    return st, sp, rox, rbl


def main():
    pars = model.load()
    sim = np.load(ROOT / "data/simulation/featurexas_nmc_structure_avg_v1.npz")
    E = pp.E_MODEL
    I = gaussian_filter1d(sim["I"], 2 / (E[1] - E[0]), axis=1); I /= I.max(1, keepdims=True)
    Y = sim["Y"]; P = model.predict(pars, I)
    out = {"simulation": {}}
    for s in ("tr", "va", "te"):
        ix = sim[s]
        out["simulation"][s] = {t: dict(rmse=rmse(Y[ix, j], P[ix, j]), mae=float(np.abs(Y[ix, j] - P[ix, j]).mean()),
                                        r2=r2(Y[ix, j], P[ix, j])) for j, t in enumerate(("oxidation", "bond_A"))}
    out["label_pearson_train"] = float(np.corrcoef(Y[sim["tr"]].T)[0, 1])

    st, sp, rox, rbl = tallman_series()
    a = report.analyse_series(sp, reference_index=0, reference_valence=2.67, x=st.x.values, pars=pars)
    ox, bl = np.array(a["oxidation"]), np.array(a["bond_A"])
    out["tallman"] = dict(calibrated_oxidation=ox.tolist(), oxidation_sd=a["oxidation_sd"], bond_A=bl.tolist(), bond_sd_A=a["bond_sd_A"],
                          reference_oxidation=rox.tolist(), reference_bond_A=rbl.tolist(),
                          oxidation_rmse_5=rmse(ox[1:], rox[1:]), oxidation_mae_5=float(np.abs(ox[1:] - rox[1:]).mean()),
                          oxidation_max_abs_5=float(np.abs(ox[1:] - rox[1:]).max()), bond_rmse_6=rmse(bl, rbl),
                          bond_mae_6=float(np.abs(bl - rbl).mean()), pearson_ox=float(np.corrcoef(ox, rox)[0, 1]),
                          total_change_pred=float(ox[-1] - ox[0]), total_change_ref=float(rox[-1] - rox[0]),
                          contrast_end_of_charge=a["contrast"][-1], observables=a["observables"])
    # dataset-level expected-gradient shares (100 training spectra as background, rng 42, 64 nodes)
    rng = np.random.default_rng(42)
    bg = I[sim["tr"]][np.sort(rng.choice(len(sim["tr"]), 100, replace=False))]
    phi = np.zeros((len(I), 100, 2))
    for s in range(0, len(I), 50):
        phi[s:s + 50] = model.expected_gradients(pars, I[s:s + 50], bg, nodes=64)
    m = np.abs(phi).mean(0)
    eexp = E + pp.REGISTRATION_EV
    shares = {r: (v / m.sum(0)).tolist() for r, v in model.region_sums(m, eexp).items()}
    out["dataset_attribution_shares"] = shares
    out["eg_completeness_max_error"] = float(np.abs(phi.sum(1) - (P - model.predict(pars, bg).mean(0))).max())
    (OUT / "numbers.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    t = out["tallman"]
    print(f"test RMSE {out['simulation']['te']['oxidation']['rmse']:.4f} / {out['simulation']['te']['bond_A']['rmse']:.5f} A; "
          f"R2 {out['simulation']['te']['oxidation']['r2']:.3f} / {out['simulation']['te']['bond_A']['r2']:.3f}")
    print(f"Tallman calibrated valence RMSE (5) {t['oxidation_rmse_5']:.4f}; Ni-O RMSE (6) {t['bond_rmse_6']:.4f} A")
    print("dataset shares (ox, Ni-O):", {k: [round(100 * x, 1) for x in v] for k, v in shares.items()})
    print("contrast at x=0.31:", {k: [round(x, 3) for x in v] for k, v in t["contrast_end_of_charge"].items()})


if __name__ == "__main__":
    main()
