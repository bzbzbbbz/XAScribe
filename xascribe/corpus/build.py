"""End-to-end agentic corpus builder: spec -> LLM-planned queries -> open scholarly search ->
exclusion -> LLM relevance screening -> open-access full text -> chunk index + manifest.

    python -m xascribe.corpus.build --element Ni --edge K --material "layered NMC lithium-ion cathode" \
        --properties "oxidation state" "bond length" --target 25 \
        --exclude-doi 10.1021/acs.jpcc.0c08095 --out corpora/ni_k_nmc
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import fetch, plan, sources
from .index import ChunkIndex


def build_corpus(spec: plan.CorpusSpec, out_dir, threshold: float = 0.6, per_query: int = 25,
                 allow_abstract_only: bool = True, backend: str = "auto", log=print) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S")
    errors: list = []
    queries, planner = plan.plan_queries(spec, log=log)
    log(f"{len(queries)} queries (planner: {planner})")
    records, counts = sources.search_all(queries, per_query=per_query, from_year=spec.from_year,
                                         progress=log, errors=errors)
    excluded = []
    kept_records = []
    for r in records:
        why = plan.is_excluded(r, spec)
        (excluded.append({"doi": r.get("doi"), "title": r.get("title"), "reason": why}) if why else kept_records.append(r))
    log(f"{len(records)} unique records; {len(excluded)} excluded by rule")
    kept, screener = plan.screen(kept_records, spec, threshold=threshold, log=log)
    log(f"{len(kept)} records pass screening (threshold {threshold}, screener {screener})")

    chunks, meta, papers = [], [], []
    for rec in kept:
        if sum(p["n_chunks"] > 0 for p in papers) >= spec.target_papers:
            break
        text, prov = fetch.fetch_text(rec, out_dir / "text")
        abstract_only = not text
        if abstract_only:
            if not (allow_abstract_only and len(rec.get("abstract", "")) > 300):
                continue
            text, prov = rec["abstract"], "abstract"
        cs = fetch.chunk_text(fetch.clean_text(text))
        citation = sources.format_citation(rec)
        for c in cs:
            chunks.append(c)
            meta.append(dict(doc_id=rec["doc_id"], citation=citation, doi=rec.get("doi", ""), title=rec.get("title", ""),
                             year=rec.get("year"), abstract_only=abstract_only))
        papers.append(dict(doc_id=rec["doc_id"], citation=citation, doi=rec.get("doi", ""), title=rec.get("title", ""),
                           year=rec.get("year"), relevance=rec["relevance"], screen_reason=rec.get("screen_reason", ""),
                           screened_by=rec.get("screened_by", ""), full_text_source=prov, abstract_only=abstract_only,
                           n_chunks=len(cs)))
        log(f"   [{len(papers):2d}] {'abstract' if abstract_only else 'full text'} {rec.get('title', '')[:70]}")
    idx = ChunkIndex(backend=backend).build(chunks, meta)
    idx.save(out_dir / "index")
    manifest = dict(spec=spec.to_json(), started=t0, finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    planner=planner, screener=screener, queries=queries, search_counts=counts,
                    excluded=excluded, n_records=len(records), n_passed_screening=len(kept),
                    n_papers=len(papers), n_full_text=sum(not p["abstract_only"] for p in papers),
                    n_chunks=len(chunks), index_backend=idx.backend, threshold=threshold, papers=papers,
                    screening=[dict(doi=r.get("doi"), title=r.get("title"), year=r.get("year"),
                                    relevance=r["relevance"], reason=r.get("screen_reason", ""),
                                    screened_by=r.get("screened_by", "")) for r in kept_records],
                    errors=errors[-50:])
    (out_dir / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "citations.json").write_text(json.dumps({p["doc_id"]: p["citation"] for p in papers}, indent=2,
                                                       ensure_ascii=False), encoding="utf-8")
    log(f"corpus: {manifest['n_papers']} papers ({manifest['n_full_text']} full text), {len(chunks)} chunks -> {out_dir}")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build an XAScribe literature corpus with an LLM agent")
    ap.add_argument("--element", default="Ni"); ap.add_argument("--edge", default="K")
    ap.add_argument("--material", default="layered NMC lithium-ion cathode")
    ap.add_argument("--properties", nargs="*", default=["oxidation state", "bond length"])
    ap.add_argument("--extra", nargs="*", default=[]); ap.add_argument("--from-year", type=int, default=1995)
    ap.add_argument("--target", type=int, default=25); ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--exclude-doi", nargs="*", default=[]); ap.add_argument("--exclude-title", nargs="*", default=[])
    ap.add_argument("--no-abstract-only", action="store_true"); ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = plan.CorpusSpec(a.element, a.edge, a.material, a.properties, a.extra, a.from_year, a.target,
                           a.exclude_doi, a.exclude_title)
    build_corpus(spec, a.out, threshold=a.threshold, allow_abstract_only=not a.no_abstract_only, backend=a.backend,
                 log=sources.safe_print)


if __name__ == "__main__":
    main()
