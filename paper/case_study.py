"""End-to-end XAScribe case study of the paper (operando NMC622, Tallman et al. 2021).

1. quantitative layer: calibrated predictions, uncertainties, attribution, observables
2. literature layer: the LLM agent builds the corpus from open scholarly APIs (the source paper of the
   spectra is excluded by DOI and title), then retrieval over the new index
3. generation with a free open-weight model; claim audit with a *different* free model

    set OPENROUTER_API_KEY=...            (free key, https://openrouter.ai/keys)
    python paper/case_study.py [--corpus corpora/ni_k_nmc] [--rebuild]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xascribe import llm, report  # noqa: E402
from xascribe.corpus.build import build_corpus  # noqa: E402
from xascribe.corpus.index import ChunkIndex  # noqa: E402
from xascribe.corpus.plan import CorpusSpec  # noqa: E402
from reproduce_numbers import tallman_series  # noqa: E402

SOURCE_DOI = "10.1021/acs.jpcc.0c08095"
SOURCE_TITLE = "Cathode Lithiation Mechanism and Extended Cycling Effects Using Operando X-ray Absorption Spectroscopy"
DESCRIPTION = ("Six Ni K-edge XANES spectra of a LiNi0.6Mn0.2Co0.2O2 (NMC622) cathode recorded operando during the first charge "
               "of a pouch cell, from the pristine state (x = 1.00 in LixNi0.6Mn0.2Co0.2O2, 3.27 V) to x = 0.31 (4.26 V).")
GENERATOR = "z-ai/glm-5.2:free"
AUDITOR = "nvidia/nemotron-3-ultra-550b-a55b:free,qwen/qwen3.8-27b:free"


def with_models(models, fn, *a, **k):
    old = os.environ.get("XASCRIBE_LLM_MODEL")
    os.environ["XASCRIBE_LLM_MODEL"] = models
    try:
        return fn(*a, **k)
    finally:
        if old is None:
            os.environ.pop("XASCRIBE_LLM_MODEL", None)
        else:
            os.environ["XASCRIBE_LLM_MODEL"] = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "corpora" / "ni_k_nmc"))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--target", type=int, default=25)
    args = ap.parse_args()
    if not llm.available():
        sys.exit("No free LLM key configured (set OPENROUTER_API_KEY, see README).")
    out = ROOT / "paper" / "outputs" / time.strftime("case_study_%Y%m%d_%H%M%S")
    st, sp, _, _ = tallman_series()
    a = report.analyse_series(sp, reference_index=0, reference_valence=2.67, x=st.x.values)
    numbers = json.loads((ROOT / "paper/outputs/numbers.json").read_text()) if (ROOT / "paper/outputs/numbers.json").exists() else None
    shares = None
    if numbers:
        s = numbers["dataset_attribution_shares"]
        shares = ("oxidation-state model - " + ", ".join(f"{r} {100 * v[0]:.0f}%" for r, v in s.items()) +
                  "; bond-length model - " + ", ".join(f"{r} {100 * v[1]:.0f}%" for r, v in s.items()) + ".")
    facts = report.facts_block(a, dataset_shares=shares, model_summary=(
        "Model: two neural-network regressors on cumulative-distribution (CDF) features of Ni K-edge spectra, trained on "
        "DFT spectra of NMC cathodes (held-out R^2 = 0.981 for Ni oxidation state and 0.986 for Ni-O bond length on computed spectra)."))

    corpus = Path(args.corpus)
    if args.rebuild or not (corpus / "corpus_manifest.json").exists():
        spec = CorpusSpec(element="Ni", edge="K", material="layered NMC lithium-ion cathode",
                          properties=["oxidation state", "bond length"], target_papers=args.target,
                          exclude_dois=[SOURCE_DOI], exclude_title_patterns=[SOURCE_TITLE])
        with_models(GENERATOR, build_corpus, spec, corpus)
    manifest = json.loads((corpus / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert all(SOURCE_DOI not in (p.get("doi") or "") for p in manifest["papers"]), "source paper leaked into corpus"
    idx = ChunkIndex.load(corpus)
    query = report.retrieval_query(DESCRIPTION, a)
    hits = idx.search(query, k=20, min_score=0.30)[:15]
    prompt, sources = report.build_prompt(DESCRIPTION, facts, hits)
    paragraph, gen_model = with_models(GENERATOR, report.generate, prompt)
    claims, aud_model, aud_raw = with_models(AUDITOR, report.audit, paragraph, prompt)
    run = dict(generator=gen_model, auditor=aud_model, corpus=str(corpus), n_hits=len(hits), n_sources=len(sources),
               hit_scores=[h["score"] for h in hits], numeric_check=report.numeric_check(paragraph, facts),
               audit_summary=report.summarise_audit(claims), corpus_summary={k: manifest[k] for k in
               ("planner", "screener", "n_records", "n_passed_screening", "n_papers", "n_full_text", "n_chunks", "excluded")})
    report.save_run(out, prompt=prompt, paragraph=paragraph, facts=facts, hits=hits, sources=sources, analysis=a,
                    audit=claims, audit_raw=aud_raw, run=run)
    print(json.dumps(run, indent=2)[:3000])
    print("\n" + paragraph)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
