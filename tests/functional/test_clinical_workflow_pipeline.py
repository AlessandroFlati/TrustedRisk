"""Functional tests for the LIB clinical-workflow pipeline.

Chains: order_set -> adherence_predictor -> care_gap_detector ->
prom_influence. Validates the full discharge-execution toolset.
"""
from __future__ import annotations

import asyncio


# ─────────────────────── Order-set generator ───────────────────────

def test_order_set_chf_high_risk_yields_short_pcp_window(chf_risk_estimate):
    from mcp_server.tools.order_set_generator import compute_order_set
    r = asyncio.run(compute_order_set(
        discharge_action="home_with_care",
        risk_estimate={**chf_risk_estimate, "probability_mean": 0.45},
        chief_complaint="acute decompensated CHF",
        medications=[
            {"name": "warfarin 5 mg", "drug_class": "anticoagulant_vka"},
            {"name": "lisinopril 10 mg", "drug_class": "ace_inhibitor"},
            {"name": "spironolactone 25 mg", "drug_class": "mra"},
        ],
    ))
    # High risk -> 3-7d PCP window
    assert r.risk_tier == "high"
    assert r.next_visit_window_days == (3, 7)
    # Drug-class monitoring orders for warfarin (INR) + ACE-I (BMP) + MRA
    lab_orders = [o.order_text for o in r.orders
                     if o.category == "lab_monitoring"]
    assert any("INR" in o for o in lab_orders)
    assert any("Basic metabolic panel" in o for o in lab_orders)


def test_order_set_snf_disposition_includes_transition_visit(chf_risk_estimate):
    from mcp_server.tools.order_set_generator import compute_order_set
    r = asyncio.run(compute_order_set(
        discharge_action="snf",
        risk_estimate=chf_risk_estimate))
    assert any("SNF" in o.order_text for o in r.orders)


def test_order_set_specialty_routing_aki():
    from mcp_server.tools.order_set_generator import compute_order_set
    r = asyncio.run(compute_order_set(
        discharge_action="home_with_care",
        risk_estimate={"probability_mean": 0.20},
        chief_complaint="acute kidney injury post-contrast",
    ))
    assert any("Nephrology" in o.order_text for o in r.orders)


# ─────────────────────── Adherence predictor ───────────────────────

def test_adherence_complex_chf_regimen_high_risk_tier():
    from mcp_server.tools.medication_adherence import (
        compute_medication_adherence_predictor,
    )
    r = asyncio.run(compute_medication_adherence_predictor(
        medications=[
            {"name": f"drug_{i}", "drug_class": "anticoagulant_vka"
              if i == 0 else "ace_inhibitor",
              "frequency": "BID" if i % 2 else "daily"}
            for i in range(10)
        ],
        patient_age=82, insurance_type="medicaid",
        prior_adherence_known=False,
    ))
    assert r.risk_tier == "high"
    assert r.adherence_30d_probability < 0.4
    blob = " ".join(r.interventions_recommended).lower()
    assert "blister-pack" in blob or "pill organizer" in blob


def test_adherence_low_complexity_high_probability():
    from mcp_server.tools.medication_adherence import (
        compute_medication_adherence_predictor,
    )
    r = asyncio.run(compute_medication_adherence_predictor(
        medications=[{"name": "lisinopril 10 mg", "drug_class":
                          "ace_inhibitor", "frequency": "daily"}],
        patient_age=55, insurance_type="commercial",
        has_caregiver=True, prior_adherence_known=True,
    ))
    assert r.risk_tier == "low"
    assert r.adherence_30d_probability > 0.7


def test_adherence_caregiver_boost_observed():
    from mcp_server.tools.medication_adherence import (
        compute_medication_adherence_predictor,
    )
    meds = [{"name": "warfarin", "drug_class": "anticoagulant_vka",
                "frequency": "daily"}]
    no_cg = asyncio.run(compute_medication_adherence_predictor(
        meds, has_caregiver=False))
    with_cg = asyncio.run(compute_medication_adherence_predictor(
        meds, has_caregiver=True))
    assert with_cg.adherence_30d_probability > no_cg.adherence_30d_probability


# ─────────────────────── Care-gap detector ───────────────────────

def test_care_gap_75yo_female_finds_canonical_gaps(chf_bundle):
    from mcp_server.tools.care_gap_detector import compute_care_gap_detector
    r = asyncio.run(compute_care_gap_detector(chf_bundle))
    titles = " ".join(g.title for g in r.high_priority_gaps
                          + r.moderate_priority_gaps).lower()
    # 75yo female -> flu vaccine (>1y), pneumococcal, mammo (>2y)
    assert "influenza" in titles or "pneumococcal" in titles


def test_care_gap_diabetic_patient_a1c_check_eligible(chf_bundle):
    """The CHF fixture has DM + a recent A1c. Force-rewind the A1c
    observation to > 90 days old -> the gap should now fire."""
    import copy
    from datetime import datetime, timedelta, timezone
    from mcp_server.tools.care_gap_detector import compute_care_gap_detector

    bundle = copy.deepcopy(chf_bundle)
    old = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
    for entry in bundle["entry"]:
        rsrc = entry["resource"]
        if rsrc.get("resourceType") != "Observation":
            continue
        for coding in rsrc.get("code", {}).get("coding", []) or []:
            if coding.get("code") == "4548-4":
                rsrc["effectiveDateTime"] = old
    r = asyncio.run(compute_care_gap_detector(bundle))
    titles = " ".join(g.title for g in r.high_priority_gaps).lower()
    assert "a1c" in titles or "hemoglobin" in titles


def test_care_gap_priority_buckets_distinct(chf_bundle):
    from mcp_server.tools.care_gap_detector import compute_care_gap_detector
    r = asyncio.run(compute_care_gap_detector(chf_bundle))
    high_ids = {g.gap_id for g in r.high_priority_gaps}
    mod_ids = {g.gap_id for g in r.moderate_priority_gaps}
    low_ids = {g.gap_id for g in r.low_priority_gaps}
    assert high_ids.isdisjoint(mod_ids)
    assert high_ids.isdisjoint(low_ids)


# ─────────────────────── PROM influence ───────────────────────

def test_prom_severe_distress_shifts_away_from_discharge_home():
    from mcp_server.tools.prom_influence import compute_prom_influence
    r = asyncio.run(compute_prom_influence({
        "instrument": "PROMIS-29",
        "pain_intensity": 9.0,
        "fatigue": 80, "depression": 75, "anxiety": 70,
        "sleep_disturbance": 65, "physical_function": 30,
    }))
    assert r.distress_burden_score >= 70
    assert r.action_dominance_shift["discharge_home"] < 0


def test_prom_eq5d_index_drives_qaly_directly():
    from mcp_server.tools.prom_influence import compute_prom_influence
    r = asyncio.run(compute_prom_influence({
        "instrument": "EQ-5D-5L", "eq5d_index_score": 0.85,
    }))
    expected = 0.85 * (30.0 / 365.0)
    assert abs(r.qaly_delta_30d - expected) < 1e-3


# ─────────────────────── End-to-end clinical-workflow chain ───────────────────────

def test_full_clinical_workflow_chain_for_chf_patient(
    chf_bundle, chf_risk_estimate,
):
    """The full clinical-workflow toolset on the canonical CHF patient.

    1. order_set composes structured orders for the discharge
    2. adherence_predictor scores the regimen
    3. care_gap_detector surfaces preventive gaps
    4. prom_influence (using a synthetic PROM input) shifts dominance
    """
    from mcp_server.tools.care_gap_detector import compute_care_gap_detector
    from mcp_server.tools.medication_adherence import (
        compute_medication_adherence_predictor,
    )
    from mcp_server.tools.order_set_generator import compute_order_set
    from mcp_server.tools.prom_influence import compute_prom_influence

    orders = asyncio.run(compute_order_set(
        discharge_action="home_with_care",
        risk_estimate=chf_risk_estimate,
        chief_complaint="acute decompensated CHF",
        medications=[
            {"name": "warfarin 5 mg", "drug_class": "anticoagulant_vka"},
            {"name": "lisinopril 10 mg", "drug_class": "ace_inhibitor"},
            {"name": "furosemide 40 mg", "drug_class": "loop_diuretic"},
            {"name": "metoprolol 50 mg", "drug_class": "beta_blocker"},
            {"name": "metformin 500 mg", "drug_class": "biguanide"},
        ],
    ))
    adherence = asyncio.run(compute_medication_adherence_predictor(
        medications=[
            {"name": "warfarin", "drug_class": "anticoagulant_vka",
              "frequency": "daily"},
            {"name": "lisinopril", "drug_class": "ace_inhibitor",
              "frequency": "daily"},
            {"name": "furosemide", "drug_class": "loop_diuretic",
              "frequency": "BID"},
            {"name": "metoprolol", "drug_class": "beta_blocker",
              "frequency": "BID"},
            {"name": "metformin", "drug_class": "biguanide",
              "frequency": "BID"},
        ],
        patient_age=75, insurance_type="medicare",
    ))
    gaps = asyncio.run(compute_care_gap_detector(chf_bundle))
    prom = asyncio.run(compute_prom_influence({
        "instrument": "PROMIS-29", "pain_intensity": 5.0,
        "fatigue": 60, "depression": 55, "physical_function": 45,
    }))

    # All four reports valid
    assert orders.n_orders >= 5
    assert adherence.adherence_30d_probability > 0
    assert gaps.n_gaps_found >= 1
    assert prom.distress_burden_score > 0
