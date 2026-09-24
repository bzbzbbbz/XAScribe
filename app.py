"""XAScribe Streamlit app:  streamlit run app.py"""
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from xascribe import llm, preprocess as pp, report
from xascribe.corpus.index import ChunkIndex

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="XAScribe", layout="wide")
st.title("XAScribe")
st.caption("Quantitative Ni K-edge XANES inference with literature-grounded, auditable reports")

# ------------------------------------------------------------------ inputs
with st.sidebar:
    st.header("Spectra")
    use_example = st.checkbox("Load example: operando NMC622 (Tallman et al. 2021)", value=True)
    files = st.file_uploader("Or upload spectra (two columns: energy eV, normalised absorption)", type=["csv", "txt", "dat"],
                             accept_multiple_files=True)
    reg = st.number_input("Energy registration E_exp - E_model (eV)", value=float(pp.REGISTRATION_EV), step=0.25)
    st.header("Calibration")
    calibrate = st.checkbox("Calibrate valence with a reference spectrum of known valence", value=True)
    ref_val = st.number_input("Known Ni valence of the reference", value=2.67, step=0.01, format="%.2f")
    st.header("Language model")
    user_key = st.text_input("Your free API key (OpenRouter by default; used only for this session, never stored)",
                             type="password", help="Get one free at https://openrouter.ai/keys - no credit card needed.")
    if user_key:
        llm.use_key(user_key)
    provider, _, _, models = llm.config()
    st.write(f"Provider: **{provider}**; models: {', '.join(models[:2]) or '-'}")
    if not llm.available():
        st.warning("No free LLM key found. Get one at openrouter.ai/keys and set OPENROUTER_API_KEY (see README).")

spectra, xs = [], []
if use_example and not files:
    stt = pd.read_csv(ROOT / "data/tallman2021/states.csv", header=None, names=["colour", "x", "V"])
    for r in stt.itertuples():
        e, mu = pp.read_spectrum(ROOT / f"data/tallman2021/spectrum_{r.colour.replace(' ', '_')}.csv")
        spectra.append((f"x = {r.x:.2f} ({r.V:.2f} V)", e, mu)); xs.append(r.x)
    default_desc = ("Six Ni K-edge XANES spectra of a LiNi0.6Mn0.2Co0.2O2 (NMC622) cathode recorded operando during the first "
                    "charge of a pouch cell, from the pristine state (x = 1.00, 3.27 V) to x = 0.31 (4.26 V).")
else:
    for f in files or []:
        df = pd.read_csv(io.BytesIO(f.getvalue()), sep=None, engine="python", comment="#")
        e, mu = pp.read_spectrum(df); spectra.append((Path(f.name).stem, e, mu))
    default_desc = ""
if not spectra:
    st.info("Load the example or upload spectra to start."); st.stop()

names = [s[0] for s in spectra]
ref_name = st.selectbox("Reference spectrum (for calibration and attribution contrast)", names, index=0)
x_text = st.text_input("Optional Li content (or other coordinate) per spectrum, comma separated",
                       value=", ".join(f"{x:.3f}" for x in xs) if xs else "")
description = st.text_area("Experiment description", value=default_desc, height=80)
x = None
try:
    vals = [float(v) for v in x_text.split(",") if v.strip()]
    x = vals if len(vals) == len(spectra) else None
except ValueError:
    st.warning("Could not parse the coordinate list; trend statistics are skipped.")


@st.cache_data(show_spinner="Running the quantitative layer ...")
def run_analysis(_spectra, key, ref_index, ref_valence, x, reg):
    return report.analyse_series(_spectra, reference_index=ref_index, reference_valence=ref_valence, x=x, registration_ev=reg)


ri = names.index(ref_name)
a = run_analysis(spectra, tuple(names), ri, ref_val if calibrate else None, tuple(x) if x else None, reg)

# ------------------------------------------------------------------ results
tab = pd.DataFrame({"spectrum": names, "missing low-energy points": a["missing_points"],
                    "Ni oxidation state": np.round(a["oxidation"], 3), "± (1σ)": np.round(a["oxidation_sd"], 3),
                    "Ni–O (Å)": np.round(a["bond_A"], 4), "± (Å)": np.round(a["bond_sd_A"], 4)})
if not a["calibrated"]:
    st.warning("Oxidation states are on the uncalibrated model scale; supply a reference of known valence for absolute values.")
st.dataframe(tab, use_container_width=True)
c1, c2, c3 = st.columns(3)
E = np.array(a["energy_exp"])
with c1:
    fig, ax = plt.subplots(figsize=(4, 3))
    for n, y in zip(names, a["mean_inputs"]): ax.plot(E, y, lw=1, label=n)
    ax.set_xlabel("Energy (eV)"); ax.set_ylabel("Normalised absorption"); ax.legend(fontsize=6); st.pyplot(fig)
with c2:
    fig, ax = plt.subplots(figsize=(4, 3)); xx = x if x else list(range(len(names)))
    ax.errorbar(xx, a["oxidation"], yerr=a["oxidation_sd"], fmt="o-", ms=4)
    ax.set_xlabel("Li content x" if x else "spectrum"); ax.set_ylabel("Ni oxidation state")
    if x: ax.invert_xaxis()
    st.pyplot(fig)
with c3:
    fig, ax = plt.subplots(figsize=(4, 3)); last = int(np.argmin(x)) if x else len(names) - 1
    regs = list(a["contrast"][last]); vals = [a["contrast"][last][r][0] for r in regs]
    ax.bar(regs, vals, color=["#2a78d6" if v >= 0 else "#e34948" for v in vals])
    ax.set_ylabel(f"Δ ox. state vs {ref_name}"); ax.set_title(names[last], fontsize=8); ax.tick_params(axis="x", labelsize=7)
    st.pyplot(fig)

# ------------------------------------------------------------------ literature layer
st.header("Literature-grounded analysis")
corpora = sorted(p.parent for p in (ROOT / "corpora").glob("*/corpus_manifest.json"))
col1, col2 = st.columns([2, 1])
with col1:
    corpus = st.selectbox("Corpus", corpora, format_func=lambda p: p.name) if corpora else None
with col2:
    with st.expander("Build a new corpus with the LLM agent"):
        el = st.text_input("Element", "Ni"); edge = st.text_input("Edge", "K")
        mat = st.text_input("Material class", "layered NMC lithium-ion cathode")
        excl = st.text_input("Exclude DOIs (comma separated), e.g. the paper whose data you analyse", "")
        target = st.number_input("Target number of papers", 5, 60, 25)
        name = st.text_input("Corpus name", "my_corpus")
        if st.button("Build corpus"):
            from xascribe.corpus.build import build_corpus
            from xascribe.corpus.plan import CorpusSpec
            log = st.empty(); lines = []
            def progress(m):
                lines.append(str(m)); log.code("\n".join(lines[-15:]))
            build_corpus(CorpusSpec(el, edge, mat, target_papers=int(target),
                                    exclude_dois=[d.strip() for d in excl.split(",") if d.strip()]),
                         ROOT / "corpora" / name, log=progress)
            st.success("Corpus built; reload the page to select it.")
if corpus and st.button("Generate analysis paragraph", disabled=not llm.available()):
    facts = report.facts_block(a)
    hits = ChunkIndex.load(corpus).search(report.retrieval_query(description, a), k=20, min_score=0.30)[:15]
    prompt, sources = report.build_prompt(description, facts, hits)
    with st.spinner("Generating ..."):
        paragraph, model_id = report.generate(prompt)
    st.markdown(paragraph)
    st.caption(f"Generated by {model_id}; {len(hits)} passages from {len(sources)} sources.")
    chk = report.numeric_check(paragraph, facts)
    (st.success if not chk["not_in_facts"] else st.error)(f"Numeric check: {chk}")
    with st.spinner("Auditing claims ..."):
        try:
            claims, aud_model, _ = report.audit(paragraph, prompt)
            st.dataframe(pd.DataFrame(claims), use_container_width=True); st.caption(f"Audit by {aud_model}")
        except llm.LLMUnavailable as exc:
            st.warning(f"Audit unavailable: {exc}")
    st.download_button("Download paragraph", paragraph, "xascribe_paragraph.txt")
    st.download_button("Download full run (JSON)", json.dumps(dict(analysis=a, prompt=prompt, paragraph=paragraph,
                       model=model_id), default=float, indent=2), "xascribe_run.json")
st.download_button("Download predictions (CSV)", tab.to_csv(index=False), "xascribe_predictions.csv")
