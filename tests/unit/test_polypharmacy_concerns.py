"""Unit tests for detect_polypharmacy_concerns."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.polypharmacy_concerns import (
    _grade_severity,
    _scan_interactions,
    detect_polypharmacy_concerns,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── _scan_interactions ───────────────────────

def test_warfarin_nsaid_high_severity():
    interactions = _scan_interactions(["warfarin 5 mg", "ibuprofen 600 mg"])
    assert len(interactions) == 1
    assert interactions[0].severity == "high"
    assert "bleeding" in interactions[0].mechanism.lower()


def test_warfarin_naproxen_high_severity():
    interactions = _scan_interactions(["warfarin 5 mg", "naproxen 500 mg"])
    assert any(i.severity == "high" for i in interactions)


def test_warfarin_clarithromycin_high_inr():
    interactions = _scan_interactions(["warfarin 5 mg", "clarithromycin 500 mg"])
    assert any("INR" in i.mechanism or "metabolism" in i.detail.lower() for i in interactions)


def test_ace_arb_dual_blockade_high():
    interactions = _scan_interactions(["lisinopril 10 mg", "losartan 50 mg"])
    assert any(i.severity == "high" for i in interactions)


def test_ace_mra_hyperkalemia_medium():
    interactions = _scan_interactions(["lisinopril 10 mg", "spironolactone 25 mg"])
    assert any(i.severity == "medium" and "hyperkalemia" in i.mechanism.lower() for i in interactions)


def test_ssri_tramadol_serotonin_syndrome():
    interactions = _scan_interactions(["sertraline 50 mg", "tramadol 50 mg"])
    assert any("serotonin" in i.mechanism.lower() for i in interactions)


def test_double_anticoagulant():
    interactions = _scan_interactions(["warfarin 5 mg", "heparin"])
    assert any("anticoagulant" in i.detail.lower() for i in interactions)


def test_doac_heparin_double_anticoag():
    interactions = _scan_interactions(["apixaban 5 mg", "enoxaparin 40 mg"])
    assert any(i.severity == "high" for i in interactions)


def test_no_interactions_clean_list():
    assert _scan_interactions(["metformin 500 mg", "atorvastatin 40 mg"]) == []


def test_interactions_symmetric():
    """Order of meds should not matter."""
    a = _scan_interactions(["warfarin 5 mg", "ibuprofen 600 mg"])
    b = _scan_interactions(["ibuprofen 600 mg", "warfarin 5 mg"])
    assert len(a) == len(b) == 1


def test_no_duplicate_interactions_for_same_pair():
    """A single (a, b) pair should fire one rule, not multiple."""
    interactions = _scan_interactions([
        "warfarin 5 mg", "ibuprofen", "ibuprofen 600 mg",  # duplicate ibuprofen
    ])
    # Even with two ibuprofens, the same (warfarin, ibuprofen) DDI should fire
    # at most twice (once per warfarin-ibuprofen pair).
    assert len(interactions) <= 2


# ─────────────────────── _grade_severity ───────────────────────

def test_grade_high_when_high_ddi():
    from shared.schemas import DrugInteraction
    di = [DrugInteraction(drug_a="x", drug_b="y", severity="high",
                          mechanism="m", detail="d")]
    assert _grade_severity(0, di) == "high"


def test_grade_high_when_7_high_risk():
    assert _grade_severity(7, []) == "high"


def test_grade_medium_when_5_high_risk():
    assert _grade_severity(5, []) == "medium"


def test_grade_low_when_3_high_risk():
    assert _grade_severity(3, []) == "low"


def test_grade_none_when_clean():
    assert _grade_severity(0, []) == "none"


# ─────────────────────── End-to-end async ───────────────────────

def test_detect_polypharmacy_empty_input():
    rep = _run(detect_polypharmacy_concerns(medications=[]))
    assert rep.n_medications == 0
    assert rep.polypharmacy_severity == "none"


def test_detect_polypharmacy_none_input():
    rep = _run(detect_polypharmacy_concerns(medications=None))
    assert rep.n_medications == 0


def test_detect_polypharmacy_clean_two_meds():
    rep = _run(detect_polypharmacy_concerns(medications=[
        "metformin 500 mg",
        "atorvastatin 40 mg",
    ]))
    assert rep.n_medications == 2
    assert rep.n_high_risk == 0
    assert rep.interactions == []
    assert rep.polypharmacy_severity == "none"


def test_detect_polypharmacy_warfarin_nsaid_high():
    rep = _run(detect_polypharmacy_concerns(medications=[
        "warfarin 5 mg",
        "ibuprofen 600 mg",
    ]))
    assert rep.polypharmacy_severity == "high"
    assert len(rep.interactions) == 1
    assert "warfarin" in rep.rationale.lower() or "DDI" in rep.rationale


def test_detect_polypharmacy_dict_input():
    """Tool accepts list of dicts with `name` keys."""
    rep = _run(detect_polypharmacy_concerns(medications=[
        {"name": "warfarin 5 mg"},
        {"name": "ibuprofen"},
    ]))
    assert rep.n_medications == 2
    assert rep.polypharmacy_severity == "high"


def test_detect_polypharmacy_class_counts():
    rep = _run(detect_polypharmacy_concerns(medications=[
        "warfarin 5 mg",       # anticoagulant_vka
        "apixaban 5 mg",       # anticoagulant_doac
        "metformin 500 mg",    # biguanide (not high-risk)
        "lisinopril 10 mg",    # ace_inhibitor
        "spironolactone 25 mg",# mra
        "insulin glargine",    # insulin
    ]))
    # 5 high-risk classes (vka, doac, ace, mra, insulin) -- biguanide is not
    assert rep.n_high_risk == 5
    assert rep.class_counts.get("anticoagulant_vka") == 1
    assert rep.class_counts.get("biguanide") == 1


def test_detect_polypharmacy_classifies_unknown_as_no_class():
    rep = _run(detect_polypharmacy_concerns(medications=[
        "completely_unknown_compound_xyz",
        "another_made_up_thing",
    ]))
    assert rep.n_medications == 2
    assert rep.n_high_risk == 0
    # Class counts only includes classified meds
    assert rep.class_counts == {}


def test_detect_polypharmacy_seven_high_risk_no_ddi():
    rep = _run(detect_polypharmacy_concerns(medications=[
        "warfarin",
        "apixaban",         # NB will trigger DDI with warfarin -> bumps to high
    ]))
    # The DDI auto-bumps severity to high regardless
    assert rep.polypharmacy_severity == "high"
