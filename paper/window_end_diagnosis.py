"""Why the attribution concentrates at the high-energy end of the window (paper ESI Section S6).

    python paper/window_end_diagnosis.py        -> paper/outputs/window_end_diagnosis.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xascribe import model, preprocess as pp  # noqa: E402

pars = model.load()
sim = np.load(ROOT / "data/simulation/featurexas_nmc_structure_avg_v1.npz")
E = pp.E_MODEL
S = gaussian_filter1d(sim["I"], 2 / (E[1] - E[0]), axis=1); S /= S.max(1, keepdims=True)
Y, tr, te = sim["Y"], sim["tr"], sim["te"]
sd = pars["scale0"].copy(); sd[-1] = np.nan                      # C_100 = 1 is constant
W = np.abs(pars["W0_0"]).sum(1); sens = W / pars["scale0"]; sens[-1] = np.nan
rho = np.array([spearmanr(S[:, k], Y[:, 0]).statistic for k in range(100)])
G = np.abs(model.gradient(pars, S)).mean(0)
rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
P = model.predict(pars, S[te])
occ = []
for s in range(93):
    X = S[te].copy(); X[:, s:s + 8] = S[tr][:, s:s + 8].mean(0)
    occ.append(rm(model.predict(pars, X)[:, 0], Y[te, 0]))
Xt = np.c_[S[:, 92:].mean(1), np.ones(len(S))]; c = np.linalg.lstsq(Xt[tr], Y[tr, 0], rcond=None)[0]
out = dict(sd_C_second_to_last=float(sd[98]), sd_C_median_mid=float(np.nanmedian(sd[10:90])),
           sensitivity_ratio_tail_vs_mid=float(np.nanmax(sens[92:99]) / np.nanmedian(sens[10:90])),
           mean_abs_grad_last3eV=float(G[92:, 0].mean()), mean_abs_grad_median_elsewhere=float(np.median(G[10:90, 0])),
           spearman_intensity_ox_last10=[float(rho[90:].min()), float(rho[90:].max())],
           test_rmse_intact=rm(P[:, 0], Y[te, 0]), test_rmse_occlude_last3eV=occ[-1], test_rmse_occlude_max_elsewhere=float(max(occ[:-8])),
           linear_tail_intensity_rmse=rm(Xt[te] @ c, Y[te, 0]), label_sd_test=float(Y[te, 0].std()))
(ROOT / "paper/outputs").mkdir(parents=True, exist_ok=True)
(ROOT / "paper/outputs/window_end_diagnosis.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
