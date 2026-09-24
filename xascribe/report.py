"""Analysis of a measured series, the facts block handed to the LLM, retrieval, prompt,
generation and a claim-level audit of the generated paragraph."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from . import llm, model, preprocess as pp
from .corpus.plan import parse_json


# ------------------------------------------------------------------ analysis
def series_observables(spectra):
    """Model-independent observables on the measured (raw) spectra of a series: rising-edge energy at
    half height above a common pre-edge baseline, and white-line energy/height from a 7-point parabolic fit."""
    base = min(mu[: max(3, int(np.argmax(mu)) // 4)].min() for _, _, mu in spectra)
    out = []
    for _, e, mu in spectra:
        k = int(np.argmax(mu))
        idx = np.argsort(np.abs(np.arange(len(e)) - k))[:7]
        c = np.polyfit(e[idx] - e[k], mu[idx], 2)
        ewl = e[k] - c[1] / (2 * c[0]) if c[0] < 0 else e[k]
        hwl = float(np.polyval(c, ewl - e[k])) if c[0] < 0 else float(mu[k])
        lev = base + 0.5 * (mu[k] - base)
        j = int(np.argmax(mu[:k + 1] >= lev))
        eh = e[j - 1] + (lev - mu[j - 1]) * (e[j] - e[j - 1]) / (mu[j] - mu[j - 1]) if j > 0 else e[j]
        out.append(dict(half_height_eV=float(eh), white_line_eV=float(ewl), white_line_height=hwl))
    return out


def analyse_series(spectra, reference_index=None, reference_valence=None, x=None,
                   registration_ev=pp.REGISTRATION_EV, draws=2048, seed=260906, pars=None, prior=None):
    """spectra: list of (name, energy, mu).  Returns a JSON-serialisable dict."""
    pars = pars or model.load(); prior = prior or pp.MissingPointPrior()
    names = [s[0] for s in spectra]

    def run(shift):
        bags, obs = [], []
        for k, (_, e, mu) in enumerate(spectra):
            y, eexp = pp.to_model_grid(e, mu, shift)
            bags.append(prior.complete(y, draws, seed + k)); obs.append(y)
        return bags, obs, eexp

    bags, obs, eexp = run(registration_ev)
    P = np.array([model.predict(pars, b) for b in bags])          # (n, draws, 2)
    mean, sd = P.mean(1), P.std(1)
    ref = reference_index if reference_index is not None else 0
    if reference_valence is not None:
        ox, delta = pp.calibrate_valence(mean[:, 0], ref, reference_valence)
    else:
        ox, delta = mean[:, 0], None
    # registration sensitivity (+-0.25, +-0.5 eV), recalibrating at each shift
    shifted = []
    for d in (-0.5, -0.25, 0.25, 0.5):
        b2, _, _ = run(registration_ev + d)
        m2 = np.array([model.predict(pars, b).mean(0) for b in b2[:]])
        o2 = pp.calibrate_valence(m2[:, 0], ref, reference_valence)[0] if reference_valence is not None else m2[:, 0]
        shifted.append(np.column_stack([o2, m2[:, 1]]))
    shifted = np.array(shifted + [np.column_stack([ox, mean[:, 1]])])
    reg_sd = shifted.std(0)
    ox_sd = np.sqrt(sd[:, 0] ** 2 + (sd[ref, 0] ** 2 if reference_valence is not None else 0) + reg_sd[:, 0] ** 2)
    ox_sd[ref] = 0.0 if reference_valence is not None else ox_sd[ref]
    bl_sd = np.sqrt(sd[:, 1] ** 2 + reg_sd[:, 1] ** 2)
    # contrast attribution vs reference spectrum (128 paired draws)
    B = bags[ref][:128]
    contrast = []
    for k, b in enumerate(bags):
        A = model.integrated_gradients(pars, b[:128], B).mean(0)
        contrast.append({r: v.tolist() for r, v in model.region_sums(A, eexp).items()})
    obsv = series_observables(spectra)
    out = dict(names=names, registration_ev=registration_ev, calibrated=reference_valence is not None,
               reference_index=ref, reference_valence=reference_valence, delta=delta,
               oxidation=ox.tolist(), oxidation_sd=ox_sd.tolist(), bond_A=mean[:, 1].tolist(), bond_sd_A=bl_sd.tolist(),
               missing_points=[int(np.isnan(o).sum()) for o in obs], contrast=contrast, observables=obsv,
               energy_exp=eexp.tolist(), mean_inputs=[b.mean(0).tolist() for b in bags])
    if x is not None:
        x = np.asarray(x, float)
        out["x"] = x.tolist()
        out["slope_ox_per_x"] = float(np.polyfit(x, ox, 1)[0])
        order = np.argsort(-x)
        out["monotonic_ox"] = bool(np.all(np.diff(ox[order]) >= 0) or np.all(np.diff(ox[order]) <= 0))
        out["monotonic_bond"] = bool(np.all(np.diff(mean[order, 1]) >= 0) or np.all(np.diff(mean[order, 1]) <= 0))
    return out


# ------------------------------------------------------------------ facts / prompt
def facts_block(a, dataset_shares=None, model_summary=None):
    """Only numbers produced by the regression layer; never reference labels."""
    L = ["QUANTITATIVE RESULTS (from the XAScribe regression layer; these are the only numbers you may use):"]
    if model_summary:
        L.append(model_summary)
    if a["calibrated"]:
        L.append(f"Valence calibration: the oxidation-state scale was aligned with a single reference, {a['names'][a['reference_index']]}, "
                 f"whose Ni valence is known to be {a['reference_valence']:.2f}; the same constant offset was applied to every spectrum. "
                 "Bond lengths were not adjusted.")
    else:
        L.append("No reference spectrum was supplied: oxidation states are on the uncalibrated model scale (relative trends only).")
    L.append("Per-spectrum predictions (uncertainty = missing-data reconstruction and energy-registration sensitivity):")
    for k, n in enumerate(a["names"]):
        lab = f"x = {a['x'][k]:.2f} ({n})" if "x" in a else n
        ox = (f"{a['oxidation'][k]:.2f} (calibration reference)" if a["calibrated"] and k == a["reference_index"]
              else f"{a['oxidation'][k]:.2f} +/- {a['oxidation_sd'][k]:.2f}")
        L.append(f"  {lab}: Ni oxidation state {ox}; Ni-O bond length {a['bond_A'][k]:.3f} +/- {a['bond_sd_A'][k]:.3f} A")
    if "x" in a:
        i0, i1 = int(np.argmax(a["x"])), int(np.argmin(a["x"]))
        L.append(f"Trend analysis (computed by the quantitative layer): the predicted Ni oxidation state "
                 f"{'changes monotonically' if a['monotonic_ox'] else 'is not monotonic'} with Li content, "
                 f"by {a['oxidation'][i1] - a['oxidation'][i0]:+.2f} from x = {a['x'][i0]:.2f} to x = {a['x'][i1]:.2f} "
                 f"(least-squares slope {a['slope_ox_per_x']:.2f} valence units per unit x); the predicted Ni-O bond length changes "
                 f"{'monotonically' if a['monotonic_bond'] else 'non-monotonically'} by {a['bond_A'][i1] - a['bond_A'][i0]:+.3f} A.")
        o0, o1 = a["observables"][i0], a["observables"][i1]
        L.append(f"Measured spectral changes from x = {a['x'][i0]:.2f} to x = {a['x'][i1]:.2f}: rising-edge half-height "
                 f"{o1['half_height_eV'] - o0['half_height_eV']:+.2f} eV, white-line maximum {o1['white_line_eV'] - o0['white_line_eV']:+.2f} eV, "
                 f"white-line height change {o1['white_line_height'] - o0['white_line_height']:+.3f}.")
        c = a["contrast"][i1]
        L.append("Attribution of the predicted change at x = %.2f relative to the reference spectrum (integrated gradients, energy regions): "
                 "oxidation-state change = %s; Ni-O change = %s." % (
                     a["x"][i1], ", ".join(f"{r} {v[0]:+.2f}" for r, v in c.items()),
                     ", ".join(f"{r} {v[1]:+.3f} A" for r, v in c.items())))
    if dataset_shares:
        L.append("Attribution across the training spectra (mean |attribution| share): " + dataset_shares)
    return "\n".join(L)


def retrieval_query(description, a):
    rng = f"predicted Ni valence from {min(a['oxidation']):.2f} to {max(a['oxidation']):.2f}; Ni-O bond length from {max(a['bond_A']):.3f} to {min(a['bond_A']):.3f} A"
    return (f"{description} Ni K-edge XANES edge shift and Ni oxidation state during delithiation; Ni-O bond contraction from EXAFS; "
            f"Ni2+/Ni3+/Ni4+ redox; charge compensation in layered Ni-rich oxides; {rng}")


def build_prompt(description, facts, hits):
    src, passages = {}, []
    for h in hits:
        if h["doc_id"] not in src:
            src[h["doc_id"]] = (len(src) + 1, h["citation"])
        passages.append(f"[{src[h['doc_id']][0]}] (similarity {h['score']:.2f}) {h['text']}")
    refs = "\n".join(f"[{n}] {c}" for n, c in src.values())
    prompt = f"""You are a spectroscopist drafting the Results paragraph of a manuscript.

EXPERIMENT DESCRIPTION: {description}

{facts}

RETRIEVED LITERATURE PASSAGES (each tagged with a numbered source; cite with the bracketed number):
{chr(10).join(passages)}

SOURCE LIST:
{refs}

INSTRUCTIONS:
- Write one Results-style paragraph (250-400 words) interpreting the quantitative results.
- State the quantitative predictions with their uncertainties exactly as given; do not introduce, round differently, or infer any other numbers.
- Explain which spectral changes drive the predictions, using the measured spectral changes and the attribution summary.
- Interpret the trends against the retrieved passages and mark every literature-derived statement with the corresponding bracketed source number(s). Cite only the numbered sources listed above.
- Do not speculate beyond the provided evidence; if the literature passages do not support a claim, do not make it.
- Output the paragraph only, followed by a line 'References:' and the numbered source list you actually cited."""
    return prompt, {c: n for n, c in src.values()}


def generate(prompt, temperature=0.2):
    return llm.chat(prompt, system="You write precise, citation-grounded scientific prose.", temperature=temperature, max_tokens=1600)


# ------------------------------------------------------------------ audit
_NUM = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?")


def numeric_check(paragraph, facts):
    """Every number in the paragraph body must occur in the facts block (deterministic, no LLM)."""
    body = paragraph.split("References:")[0]
    body = re.sub(r"\[[\d,\s–-]+\]", " ", body)               # drop citation markers
    fact_nums = {n.lstrip("+").replace("−", "-") for n in _NUM.findall(facts)}
    fact_abs = {n.lstrip("-") for n in fact_nums}
    found = [n.lstrip("+").replace("−", "-") for n in _NUM.findall(body)]
    missing = [n for n in found if n not in fact_nums and n.lstrip("-") not in fact_abs]
    return dict(n_numbers=len(found), not_in_facts=sorted(set(missing)))


def audit(paragraph, prompt, temperature=0.0):
    """LLM claim audit (run with a different model from the generator if possible)."""
    q = ("Audit the generated paragraph against the prompt it was written from. Split it into atomic claims. For each claim "
         "give: claim (<=25 words), type (numeric | literature | interpretive), verdict (numeric: exact | wrong; literature: "
         "supported | partial | unsupported | miscited; interpretive: sound | overreaching), and evidence (<=30 words, quote the "
         "supporting passage or facts line). Be strict: a statement about one composition supported only by a passage about "
         "another is 'partial'. Return ONLY a JSON array.\n\n=== PROMPT ===\n" + prompt + "\n\n=== GENERATED PARAGRAPH ===\n" + paragraph)
    text, m = llm.chat(q, system="You are a strict scientific fact-checker.", temperature=temperature, max_tokens=4000)
    claims = parse_json(text)
    return (claims if isinstance(claims, list) else []), m, text


def summarise_audit(claims):
    from collections import Counter
    c = Counter((str(x.get("type", "")).lower(), str(x.get("verdict", "")).lower()) for x in claims)
    return {f"{t}:{v}": n for (t, v), n in sorted(c.items())} | {"n_claims": len(claims)}


def save_run(out_dir, **items):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    for k, v in items.items():
        p = out / (k + (".txt" if isinstance(v, str) else ".json"))
        p.write_text(v if isinstance(v, str) else json.dumps(v, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps({"saved": stamp, "files": list(items)}, indent=2), encoding="utf-8")
    return out
