"""Phase 14.14 Q3 -- Pure-Python TF-IDF tool retrieval tests."""

from __future__ import annotations

import asyncio

import pytest

from a2a_agent import tool_retrieval_tfidf as tr
from a2a_agent.planner import plan_tool_use


@pytest.fixture(autouse=True)
def _reset():
    tr._reset_index()
    yield
    tr._reset_index()


# ─────────────────────────────────────────────────────────────────────
# Index lifecycle
# ─────────────────────────────────────────────────────────────────────

def test_index_covers_at_least_140_tools():
    res = tr.retrieve_tools_by_query("readmission risk")
    assert res.n_indexed >= 140


def test_top_k_caps_returned_hits():
    res = tr.retrieve_tools_by_query("medication", top_k=2)
    assert len(res.hits) <= 2


def test_top_k_must_be_positive():
    with pytest.raises(ValueError):
        tr.retrieve_tools_by_query("anything", top_k=0)


def test_method_label_is_tfidf():
    res = tr.retrieve_tools_by_query("readmission")
    assert res.method == "tfidf"


# ─────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────

def test_repeated_query_returns_identical_ranking():
    a = tr.retrieve_tools_by_query("stroke nihss thrombolysis")
    tr._reset_index()
    b = tr.retrieve_tools_by_query("stroke nihss thrombolysis")
    assert [h.tool_name for h in a.hits] == [h.tool_name for h in b.hits]
    assert [h.score for h in a.hits] == [h.score for h in b.hits]


def test_ties_break_on_tool_name():
    """Two records with identical TF-IDF score must order alphabetically
    by tool name. We construct a tiny synthetic index to guarantee tie."""
    records = [
        {"name": "compute_zeta", "module": "x",
         "docstring": "alpha bravo", "bundle_memberships": []},
        {"name": "compute_alpha", "module": "x",
         "docstring": "alpha bravo", "bundle_memberships": []},
    ]
    idx = tr._TfIdfIndex(records)
    hits = idx.query("alpha bravo", top_k=2)
    names = [idx.records[i]["name"] for i, _ in hits]
    assert names == ["compute_alpha", "compute_zeta"]


# ─────────────────────────────────────────────────────────────────────
# Semantic relevance
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query, expected_in_topk",
    [
        ("readmission risk after discharge", "compute_readmission_risk"),
        ("HEART score for chest pain", "compute_heart_score"),
        ("stroke thrombolysis tPA eligibility",
         "compute_stroke_thrombolysis_eligibility"),
        ("DKA severity diabetic ketoacidosis",
         "compute_dka_severity"),
        ("KDIGO acute kidney injury staging",
         "compute_aki_kdigo_stage"),
        ("HL7 v2 message parser",
         "compute_hl7v2_message_parse"),
        ("C-CDA document parser",
         "compute_ccda_document_parse"),
        ("Cox proportional hazards survival",
         "compute_cox_proportional_hazards"),
    ],
)
def test_semantic_query_lands_relevant_tool_in_topk(
    query: str, expected_in_topk: str,
):
    res = tr.retrieve_tools_by_query(query, top_k=5)
    names = [h.tool_name for h in res.hits]
    assert expected_in_topk in names, (
        f"expected {expected_in_topk!r} in top-5 for {query!r}, "
        f"got {names}"
    )


def test_empty_query_returns_no_hits():
    res = tr.retrieve_tools_by_query("the", top_k=5)
    assert res.hits == []


def test_score_decreases_monotonically():
    res = tr.retrieve_tools_by_query("readmission risk discharge", top_k=10)
    scores = [h.score for h in res.hits]
    assert scores == sorted(scores, reverse=True)


def test_hit_score_within_unit_interval():
    res = tr.retrieve_tools_by_query("stroke thrombolysis", top_k=10)
    for h in res.hits:
        assert 0.0 <= h.score <= 1.0


# ─────────────────────────────────────────────────────────────────────
# Tokeniser
# ─────────────────────────────────────────────────────────────────────

def test_tokenize_drops_stop_words_and_short():
    toks = tr._tokenize("The patient is a 60-year-old with chest pain")
    assert "the" not in toks and "is" not in toks and "a" not in toks
    # length-1 tokens dropped
    assert "60" in toks
    assert "chest" in toks and "pain" in toks


def test_tokenize_keeps_underscored_compound_tokens():
    toks = tr._tokenize("compute_readmission_risk")
    assert "compute_readmission_risk" in toks


# ─────────────────────────────────────────────────────────────────────
# Planner integration
# ─────────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


def test_planner_does_not_use_tfidf_when_intent_matches():
    plan = _run(plan_tool_use(
        "Compute the readmission risk for this patient",
        enable_tfidf_retrieval=True,
        tfidf_top_k=3,
    ))
    # Intent rule fires -> no tfidf retrieval steps
    assert not any(
        s.step_id.startswith("floor-tfidf-") for s in plan.steps
    )


def test_planner_uses_tfidf_when_no_intent_matches():
    """A query the curated regex panel does not cover should trigger
    TF-IDF retrieval and surface advisory candidate tools."""
    plan = _run(plan_tool_use(
        "epworth sleepiness scale obstructive sleep apnea",
        enable_tfidf_retrieval=True,
        tfidf_top_k=3,
    ))
    tfidf_steps = [
        s for s in plan.steps if s.step_id.startswith("floor-tfidf-")
    ]
    assert len(tfidf_steps) >= 1
    # Advisory only -- not mandatory
    assert all(not s.mandatory for s in tfidf_steps)


def test_planner_disabled_by_default():
    """Without `enable_tfidf_retrieval=True` the planner stays
    purely intent-rule-driven."""
    plan = _run(plan_tool_use(
        "epworth sleepiness scale obstructive sleep apnea",
    ))
    assert not any(
        s.step_id.startswith("floor-tfidf-") for s in plan.steps
    )


def test_planner_tfidf_steps_share_clinical_intent_bundle():
    plan = _run(plan_tool_use(
        "DAS28 rheumatoid arthritis disease activity score",
        enable_tfidf_retrieval=True,
        tfidf_top_k=3,
    ))
    tfidf_steps = [
        s for s in plan.steps if s.step_id.startswith("floor-tfidf-")
    ]
    assert len(tfidf_steps) >= 1
    bundles = {s.bundle for s in tfidf_steps}
    # At least one of the candidate bundles must be a real bundle
    from mcp_server.tools import BUNDLES
    assert bundles & set(BUNDLES.keys())
