"""Agentic query planning, exclusion and relevance screening.

With a configured LLM (see xascribe.llm) the agent writes the search queries and screens
titles/abstracts with an explicit rubric; without one, transparent heuristics are used so the
builder always runs.  Every decision (score, reason, model) is recorded in the manifest.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .. import llm
from .sources import normalize_doi, normalize_title


@dataclass
class CorpusSpec:
    """What the corpus should cover; the only input needed to retarget XAScribe."""
    element: str = "Ni"
    edge: str = "K"
    material: str = "layered NMC lithium-ion cathode"
    properties: list = field(default_factory=lambda: ["oxidation state", "bond length"])
    extra_terms: list = field(default_factory=list)
    from_year: int | None = 1995
    target_papers: int = 25
    exclude_dois: list = field(default_factory=list)
    exclude_title_patterns: list = field(default_factory=list)

    def to_json(self):
        return asdict(self)


def parse_json(text, want=None):
    """Extract a JSON array from an LLM reply.  Reasoning models often think aloud (with brackets) before
    answering, so the LAST bracket-balanced, parseable array is returned; `want` = 'dicts' or 'strings'
    additionally requires the element type."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    for end in reversed([i for i, ch in enumerate(text) if ch == "]"]):
        depth = 0
        for start in range(end, -1, -1):
            depth += text[start] == "]"
            depth -= text[start] == "["
            if depth == 0:
                try:
                    got = json.loads(text[start:end + 1])
                except ValueError:
                    break
                if isinstance(got, list) and got and (
                        want is None or (want == "dicts" and all(isinstance(g, dict) for g in got))
                        or (want == "strings" and all(isinstance(g, str) for g in got))):
                    return got
                break
    return None


def is_excluded(rec: dict, spec: CorpusSpec) -> str:
    """Return the reason a record is excluded ('' if it is not)."""
    doi = normalize_doi(rec.get("doi", ""))
    if doi and doi in {normalize_doi(d) for d in spec.exclude_dois}:
        return f"excluded DOI {doi}"
    title = normalize_title(rec.get("title", ""))
    for pat in spec.exclude_title_patterns:
        if normalize_title(pat) and normalize_title(pat) in title:
            return f"excluded title pattern '{pat}'"
    return ""


# ------------------------------------------------------------------- planning
def plan_queries(spec: CorpusSpec, n: int = 10, log=None) -> tuple[list[str], str]:
    """Return (queries, planner) where planner is the model id or 'heuristic'."""
    if llm.available():
        prompt = (
            f"Write {n} complementary bibliographic search queries for building a literature corpus on "
            f"{spec.element} {spec.edge}-edge X-ray absorption spectroscopy (XANES/EXAFS) of {spec.material}. "
            f"The corpus will ground the interpretation of: {', '.join(spec.properties)}. Cover the core "
            "edge/material combination, in situ/operando measurements across states of charge, charge "
            "compensation, edge-position/valence calibration with references, EXAFS local structure and "
            "bond lengths, and reviews. Each query under 14 words, no boolean operators. "
            "Return ONLY a JSON array of strings.")
        try:
            text, model = llm.chat(prompt, system="You are a literature-search agent for X-ray spectroscopy. "
                                   "Reply with the JSON array only.", temperature=0.3, max_tokens=6000)
            got = parse_json(text, want="strings")
            if isinstance(got, list):
                qs = [str(q).strip() for q in got if str(q).strip()][:n]
                if len(qs) >= 4:
                    return qs, model
        except llm.LLMUnavailable as exc:
            if log:
                log(f"LLM planning failed ({exc}); using heuristic queries")
    el, edge, mat = spec.element, spec.edge, spec.material
    base = [f"{el} {edge}-edge XANES {mat}", f"{el} {edge}-edge X-ray absorption spectroscopy {mat}",
            f"operando {el} {edge}-edge XAS {mat} state of charge", f"in situ {el} {edge}-edge XANES charge compensation {mat}",
            f"{el} oxidation state {edge}-edge XANES reference compounds", f"EXAFS {el}-O bond length {mat}",
            f"{el} {edge}-edge XANES delithiation local structure", f"charge compensation mechanism {mat} X-ray absorption",
            f"X-ray absorption spectroscopy review battery cathode {el}", f"{el} {edge}-edge XAS {mat} cycling"]
    base += [f"{t} {el} {edge}-edge XAS" for t in spec.extra_terms]
    return base[:n], "heuristic"


# ------------------------------------------------------------------ screening
_XAS = ("xanes", "x-ray absorption", "xas", "exafs", "near-edge", "near edge", "xafs", "absorption spectroscopy")


def heuristic_score(rec: dict, spec: CorpusSpec) -> float:
    """Strict transparent rule: XAS term AND element AND a material-class term are required."""
    text = f"{rec.get('title', '')} {rec.get('abstract', '')}".lower()
    if not text.strip():
        return 0.0
    el = spec.element.lower()
    has_xas = any(t in text for t in _XAS)
    has_el = re.search(rf"\b{re.escape(el)}\b", text) is not None
    mat_tokens = [t for t in re.split(r"\W+", spec.material.lower()) if len(t) > 3]
    mat_hits = sum(t in text for t in mat_tokens)
    if not (has_xas and has_el and mat_hits):
        return 0.0
    s = 0.45 + 0.15 * min(1.0, mat_hits / 2)
    if re.search(rf"{re.escape(el)}\s*{re.escape(spec.edge.lower())}[- ]edge", text):
        s += 0.2
    s += 0.05 * sum(t in text for t in ("operando", "in situ", "state of charge", "delithiat", "charge compensation"))
    return min(1.0, s)


def screen(records: list[dict], spec: CorpusSpec, threshold: float = 0.6, batch: int = 15, log=None):
    """Attach relevance/screening fields to every record; return (kept_sorted, screener)."""
    for r in records:
        r["relevance_heuristic"] = heuristic_score(r, spec)
        r["relevance"], r["screen_reason"], r["screened_by"] = r["relevance_heuristic"], "heuristic rule", "heuristic"
    screener = "heuristic"
    if llm.available():
        # only records mentioning X-ray absorption at all are sent to the LLM (keeps free-tier usage low)
        pool = [r for r in records if any(t in f"{r.get('title', '')} {r.get('abstract', '')}".lower() for t in _XAS)]
        rubric = (
            f"Target corpus: {spec.element} {spec.edge}-edge X-ray absorption spectroscopy of {spec.material}, "
            f"used to interpret {', '.join(spec.properties)}.\n"
            "Score each record 0-1: 1.0 = measures or reviews the target edge in the target material class; "
            "0.7 = the target edge in a closely analogous material (e.g. related layered oxide cathodes); "
            "0.4 = relevant method or reference-compound work on the same edge; 0.0-0.2 = other materials or "
            "applications (catalysis, other elements, unrelated techniques). Give a reason of at most 15 words.\n"
            "Return ONLY a JSON array of objects {\"id\": <int>, \"score\": <float>, \"reason\": <string>}.\n\n")
        for s in range(0, len(pool), batch):
            chunk = pool[s:s + batch]
            listing = "\n".join(f"{i}. {r['title']} ({r.get('year') or ''}) :: {(r.get('abstract') or '')[:600]}"
                                for i, r in enumerate(chunk))
            try:
                text, model = llm.chat(rubric + listing, system="You screen papers for a spectroscopy literature corpus. "
                                       "Reply with the JSON array only.", temperature=0.0, max_tokens=8000)
            except llm.LLMUnavailable as exc:
                if log:
                    log(f"LLM screening FAILED for batch {s // batch}: {exc}")
                continue
            got = parse_json(text, want="dicts")
            if not isinstance(got, list):
                if log:
                    log(f"LLM screening FAILED for batch {s // batch}: no JSON array in reply ({len(text)} chars)")
                continue
            screener = model
            n_scored = 0
            for item in got:
                try:
                    i, sc = int(item["id"]), float(item["score"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= i < len(chunk):
                    chunk[i].update(relevance=max(0.0, min(1.0, sc)), screen_reason=str(item.get("reason", ""))[:200],
                                    screened_by=model)
                    n_scored += 1
            if log:
                log(f"   screened {min(s + batch, len(pool))}/{len(pool)} with {model}: {n_scored}/{len(chunk)} scored")
        unscored = [r for r in pool if r["screened_by"] == "heuristic"]
        for r in unscored:
            r["screened_by"] = "heuristic (LLM batch failed)"
        if log and unscored:
            log(f"   WARNING: {len(unscored)} of {len(pool)} records kept their heuristic score (LLM batch failures)")
    keep = [r for r in records if r["relevance"] >= threshold]
    keep.sort(key=lambda r: (-r["relevance"], -(r.get("year") or 0)))
    return keep, screener
