import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper"))
from xascribe import model, report  # noqa: E402
from xascribe.corpus import fetch, plan, sources  # noqa: E402
from xascribe.corpus.index import ChunkIndex  # noqa: E402
from reproduce_numbers import tallman_series  # noqa: E402


def test_tallman_calibrated_rmse():
    st, sp, rox, rbl = tallman_series()
    a = report.analyse_series(sp, reference_index=0, reference_valence=2.67, x=st.x.values, draws=2048)
    ox, bl = np.array(a["oxidation"]), np.array(a["bond_A"])
    assert abs(np.sqrt(np.mean((ox[1:] - rox[1:]) ** 2)) - 0.0474) < 2e-3
    assert abs(np.sqrt(np.mean((bl - rbl) ** 2)) - 0.0133) < 1e-3
    facts = report.facts_block(a)
    for v in rox[1:]:
        assert f"{v:.3f}" not in facts           # reference labels never reach the LLM


def test_ig_completeness():
    pars = model.load()
    X = np.random.default_rng(0).random((3, 100)) + 0.1; B = np.ones((3, 100))
    A = model.integrated_gradients(pars, X, B, nodes=64)
    assert np.abs(A.sum(1) - (model.predict(pars, X) - model.predict(pars, B))).max() < 1e-8


def test_numeric_check():
    facts = "Ni oxidation state 2.95 +/- 0.08; Ni-O 1.965 A"
    assert report.numeric_check("It is 2.95 ± 0.08 [1].", facts)["not_in_facts"] == []
    assert report.numeric_check("It is 3.10.", facts)["not_in_facts"] == ["3.10"]


def test_exclusion_and_citation():
    spec = plan.CorpusSpec(exclude_dois=["https://doi.org/10.1021/ACS.JPCC.0C08095"], exclude_title_patterns=["Lithiation Mechanism"])
    assert plan.is_excluded({"doi": "10.1021/acs.jpcc.0c08095", "title": "x"}, spec)
    assert plan.is_excluded({"doi": "", "title": "NMC622 Cathode Lithiation Mechanism and ..."}, spec)
    assert not plan.is_excluded({"doi": "10.1/abc", "title": "Other"}, spec)
    rec = sources.new_record(title="Ni K-edge &lt;i&gt;XANES&lt;/i&gt;", journal="J. Test", year=2020, volume="1", pages="2–3",
                             authors=[{"given": "Ann B.", "family": "Smith"}, {"given": "C.", "family": "Doe"}])
    c = sources.format_citation(rec, include_doi=False)
    assert c.startswith("A. B. Smith and C. Doe, Ni K-edge XANES")


def test_llm_json_parsing_and_index():
    assert plan.parse_json('```json\n[{"id": 0, "score": 0.9, "reason": "ok"}]\n```')[0]["score"] == 0.9
    ch = fetch.chunk_text("para one.\n\n" + "x" * 2500)
    assert all(len(c) <= 1300 for c in ch) and len(ch) >= 3
    idx = ChunkIndex(backend="sparse").build(["nickel k-edge shift on charge", "cobalt oxide catalysis"],
                                             [{"doc_id": "a", "citation": "A"}, {"doc_id": "b", "citation": "B"}])
    assert idx.search("nickel K-edge", k=2, min_score=0.0)[0]["doc_id"] == "a"
