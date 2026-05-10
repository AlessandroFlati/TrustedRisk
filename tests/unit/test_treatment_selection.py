"""Unit tests for compute_treatment_selection."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.treatment_selection import compute_treatment_selection


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Atrial fibrillation anticoagulation ───────────────────────

def test_afib_high_chads_picks_doac():
    sel = _run(compute_treatment_selection(
        condition="atrial_fibrillation_anticoagulation",
        patient_factors={
            "cha2ds2_vasc_score": 4,
            "hasbled_score": 2,
            "egfr_ml_min": 65,
            "age": 72,
            "mechanical_valve": False,
        },
    ))
    assert sel.top_pick_id == "doac_apixaban"
    assert not sel.abstain_recommended
    # Aspirin alone should be ranked very low
    asa = next(o for o in sel.options if o.option_id == "aspirin_only")
    assert asa.rank >= 3


def test_afib_mechanical_valve_picks_warfarin():
    sel = _run(compute_treatment_selection(
        condition="atrial_fibrillation_anticoagulation",
        patient_factors={
            "cha2ds2_vasc_score": 5,
            "hasbled_score": 2,
            "egfr_ml_min": 60,
            "mechanical_valve": True,
        },
    ))
    assert sel.top_pick_id == "warfarin"
    doac = next(o for o in sel.options if o.option_id == "doac_apixaban")
    assert doac.score == 0.0  # contraindicated
    assert any("mechanical" in c.lower() for c in doac.contraindications)


def test_afib_severe_renal_blocks_doac():
    sel = _run(compute_treatment_selection(
        condition="atrial_fibrillation_anticoagulation",
        patient_factors={
            "cha2ds2_vasc_score": 4,
            "hasbled_score": 3,
            "egfr_ml_min": 12,
        },
    ))
    doac = next(o for o in sel.options if o.option_id == "doac_apixaban")
    assert any("eGFR" in c and "15" in c for c in doac.contraindications)


def test_afib_aspirin_carries_caution():
    sel = _run(compute_treatment_selection(
        condition="atrial_fibrillation_anticoagulation",
        patient_factors={"cha2ds2_vasc_score": 3, "hasbled_score": 2},
    ))
    asa = next(o for o in sel.options if o.option_id == "aspirin_only")
    assert any("not recommended" in c.lower() for c in asa.cautions)


# ─────────────────────── Pre-op anticoagulation bridging ───────────────────────

def test_preop_low_risk_picks_no_bridging():
    sel = _run(compute_treatment_selection(
        condition="anticoagulation_pre_op_bridging",
        patient_factors={
            "ac_indication": "atrial_fibrillation",
            "cha2ds2_vasc_score": 3,
            "mechanical_valve": False,
            "surgery_bleeding_risk": "low",
        },
    ))
    assert sel.top_pick_id == "no_bridging"


def test_preop_mechanical_valve_picks_bridging():
    sel = _run(compute_treatment_selection(
        condition="anticoagulation_pre_op_bridging",
        patient_factors={
            "ac_indication": "atrial_fibrillation",
            "cha2ds2_vasc_score": 4,
            "mechanical_valve": True,
            "surgery_bleeding_risk": "low",
        },
    ))
    assert sel.top_pick_id == "bridge_lmwh"


def test_preop_recent_vte_picks_bridging():
    sel = _run(compute_treatment_selection(
        condition="anticoagulation_pre_op_bridging",
        patient_factors={
            "ac_indication": "vte",
            "cha2ds2_vasc_score": 0,
            "recent_vte_weeks": 6,
            "surgery_bleeding_risk": "low",
        },
    ))
    assert sel.top_pick_id == "bridge_lmwh"


# ─────────────────────── HFrEF GDMT ───────────────────────

def test_hf_normal_hemodynamics_picks_arni():
    sel = _run(compute_treatment_selection(
        condition="hf_reduced_ef_initial_therapy",
        patient_factors={
            "egfr_ml_min": 65, "systolic_bp": 120,
            "potassium_meq_l": 4.0, "heart_rate": 75,
        },
    ))
    assert sel.top_pick_id in ("arni_sacubitril_valsartan",
                                 "beta_blocker_carvedilol",
                                 "mra_spironolactone",
                                 "sglt2i_dapagliflozin")
    arni = next(o for o in sel.options if o.option_id == "arni_sacubitril_valsartan")
    assert arni.score >= 0.90


def test_hf_hyperkalemia_blocks_mra():
    sel = _run(compute_treatment_selection(
        condition="hf_reduced_ef_initial_therapy",
        patient_factors={
            "egfr_ml_min": 50, "systolic_bp": 120,
            "potassium_meq_l": 5.6, "heart_rate": 75,
        },
    ))
    mra = next(o for o in sel.options if o.option_id == "mra_spironolactone")
    assert mra.score == 0.0
    assert any("K+" in c for c in mra.contraindications)


def test_hf_hypotension_lowers_arni():
    sel = _run(compute_treatment_selection(
        condition="hf_reduced_ef_initial_therapy",
        patient_factors={
            "egfr_ml_min": 60, "systolic_bp": 88,
            "potassium_meq_l": 4.0, "heart_rate": 80,
        },
    ))
    arni = next(o for o in sel.options if o.option_id == "arni_sacubitril_valsartan")
    assert arni.score < 0.95
    bb = next(o for o in sel.options if o.option_id == "beta_blocker_carvedilol")
    # SBP < 90 also defers BB
    assert bb.score < 0.5


# ─────────────────────── DM2 second-line ───────────────────────

def test_dm2_with_hf_picks_sglt2i():
    sel = _run(compute_treatment_selection(
        condition="dm2_second_line_after_metformin",
        patient_factors={
            "ascvd": False, "heart_failure": True, "ckd": False,
            "bmi": 28, "egfr_ml_min": 65, "a1c_pct": 8.5,
        },
    ))
    assert sel.top_pick_id == "sglt2i"


def test_dm2_with_ascvd_picks_glp1():
    sel = _run(compute_treatment_selection(
        condition="dm2_second_line_after_metformin",
        patient_factors={
            "ascvd": True, "heart_failure": False, "ckd": False,
            "bmi": 30, "egfr_ml_min": 70, "a1c_pct": 8.0,
        },
    ))
    assert sel.top_pick_id == "glp1_ra"


def test_dm2_high_a1c_promotes_insulin():
    sel = _run(compute_treatment_selection(
        condition="dm2_second_line_after_metformin",
        patient_factors={
            "ascvd": False, "heart_failure": False, "ckd": False,
            "bmi": 25, "egfr_ml_min": 80, "a1c_pct": 11.0,
        },
    ))
    insulin = next(o for o in sel.options if o.option_id == "basal_insulin")
    assert insulin.score >= 0.50


def test_dm2_severe_renal_blocks_sglt2():
    sel = _run(compute_treatment_selection(
        condition="dm2_second_line_after_metformin",
        patient_factors={
            "ascvd": False, "heart_failure": True, "ckd": True,
            "bmi": 28, "egfr_ml_min": 18, "a1c_pct": 8.5,
        },
    ))
    sglt = next(o for o in sel.options if o.option_id == "sglt2i")
    assert sglt.score == 0.0
    assert any("eGFR" in c for c in sglt.contraindications)


# ─────────────────────── Unsupported condition ───────────────────────

def test_unsupported_condition_abstains():
    sel = _run(compute_treatment_selection(
        condition="acute_pulmonary_embolism",
        patient_factors={},
    ))
    assert sel.abstain_recommended is True
    assert sel.abstain_reason and sel.abstain_reason.startswith("condition_not_supported")
    assert sel.options == []


# ─────────────────────── Output structure ───────────────────────

def test_output_has_references():
    sel = _run(compute_treatment_selection(
        condition="atrial_fibrillation_anticoagulation",
        patient_factors={"cha2ds2_vasc_score": 3},
    ))
    assert len(sel.references) >= 1
    assert any("AHA" in r or "ACC" in r or "ESC" in r for r in sel.references)


def test_options_ranked_by_score_desc():
    sel = _run(compute_treatment_selection(
        condition="atrial_fibrillation_anticoagulation",
        patient_factors={"cha2ds2_vasc_score": 4, "egfr_ml_min": 60},
    ))
    scores = [o.score for o in sel.options]
    assert scores == sorted(scores, reverse=True)
    ranks = [o.rank for o in sel.options]
    assert ranks == list(range(1, len(sel.options) + 1))


def test_grounding_hook_does_not_break_when_corpus_missing(monkeypatch):
    """If grounding fails, treatment selection still returns a valid result."""
    async def _fail(*a, **kw):
        raise RuntimeError("corpus unavailable")

    from mcp_server.tools import treatment_selection as ts
    monkeypatch.setattr(ts, "_ground", _fail)

    sel = _run(compute_treatment_selection(
        condition="atrial_fibrillation_anticoagulation",
        patient_factors={"cha2ds2_vasc_score": 3, "egfr_ml_min": 60},
    ))
    assert sel.top_pick_id is not None  # still produced
