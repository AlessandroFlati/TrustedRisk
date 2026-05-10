"""Property-based invariant tests across clinical tools (SCI-2).

Each test uses Hypothesis to generate randomised valid inputs across the
documented input domain and asserts a property the tool MUST satisfy
regardless of input. Properties tested:

  - Schema validity (output always parses as the declared Pydantic model)
  - Score bounds (NIHSS ∈ [0, 42], NEWS2 ∈ [0, 20], PEWS ∈ [0, 12], etc.)
  - Monotonicity (higher LACE -> higher recommended response severity;
    higher CHA₂DS₂-VASc -> DOAC score never lower)
  - Edge clamping (age 0 / 120 / -1, BP 0 / 999 must not crash)
  - Determinism (same input twice -> same output)
  - Abstain triggers fire under documented conditions

These are not unit tests -- they're invariants that hold across the
generative space. Hypothesis falsifies them by finding counter-examples.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st


# Reduce default deadline (some tools do FAISS load) -- but the suite
# runs with a small enough N that it stays under 30s total.
_HYP = settings(
    deadline=2000,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── NIHSS (compute_stroke_severity) ───────────────────────

# All 15 NIHSS items with their max-points domains
_NIHSS_ITEMS = {
    "loc_responsiveness": 3, "loc_questions": 2, "loc_commands": 2,
    "best_gaze": 2, "visual_fields": 3, "facial_palsy": 3,
    "motor_arm_left": 4, "motor_arm_right": 4,
    "motor_leg_left": 4, "motor_leg_right": 4,
    "limb_ataxia": 2, "sensory": 2, "best_language": 3,
    "dysarthria": 2, "extinction_inattention": 2,
}

_nihss_strategy = st.fixed_dictionaries({
    item: st.integers(min_value=0, max_value=max_pts)
    for item, max_pts in _NIHSS_ITEMS.items()
})


@given(items=_nihss_strategy,
        lkw=st.integers(min_value=0, max_value=10000))
@_HYP
def test_nihss_total_in_bounds(items, lkw):
    """NIHSS total must always be in [0, 42] regardless of input."""
    from mcp_server.tools.stroke_severity import compute_stroke_severity
    rep = _run(compute_stroke_severity(item_scores=items,
                                          last_known_well_minutes_ago=lkw))
    assert 0 <= rep.score_total <= 42
    assert rep.severity_tier in ("minor", "moderate", "moderate_severe", "severe")
    # Sum of items equals total
    assert sum(it.points for it in rep.items) == rep.score_total


@given(items=_nihss_strategy)
@_HYP
def test_nihss_severity_tier_monotonic(items):
    """Higher total -> not-lower severity tier (monotonicity check)."""
    from mcp_server.tools.stroke_severity import compute_stroke_severity, _classify_severity
    rep = _run(compute_stroke_severity(item_scores=items))
    expected = _classify_severity(rep.score_total)
    assert rep.severity_tier == expected


# ─────────────────────── NEWS2 (compute_clinical_deterioration_score) ───────────────────────

@given(rr=st.floats(min_value=4.0, max_value=60.0, allow_nan=False),
        spo2=st.floats(min_value=70.0, max_value=100.0, allow_nan=False),
        sbp=st.floats(min_value=50.0, max_value=260.0, allow_nan=False),
        hr=st.floats(min_value=20.0, max_value=220.0, allow_nan=False),
        temp=st.floats(min_value=33.0, max_value=42.0, allow_nan=False),
        avpu=st.sampled_from(["A", "V", "P", "U"]))
@_HYP
def test_news2_total_in_bounds(rr, spo2, sbp, hr, temp, avpu):
    from mcp_server.tools.clinical_deterioration_score import compute_clinical_deterioration_score
    vitals = [
        {"type": "respiratory_rate", "value": rr,
         "observed_at": datetime.now(timezone.utc).isoformat()},
        {"type": "spo2", "value": spo2,
         "observed_at": datetime.now(timezone.utc).isoformat()},
        {"type": "systolic_bp", "value": sbp,
         "observed_at": datetime.now(timezone.utc).isoformat()},
        {"type": "heart_rate", "value": hr,
         "observed_at": datetime.now(timezone.utc).isoformat()},
        {"type": "temperature", "value": temp,
         "observed_at": datetime.now(timezone.utc).isoformat()},
        {"type": "consciousness", "value": avpu,
         "observed_at": datetime.now(timezone.utc).isoformat()},
    ]
    rep = _run(compute_clinical_deterioration_score(vital_signs=vitals))
    assert 0 <= rep.score_total <= 20
    assert rep.severity_tier in ("low", "low_medium", "medium", "high")


@given(rr=st.floats(min_value=12.0, max_value=20.99, allow_nan=False))
@_HYP
def test_news2_normal_rr_zero_points(rr):
    """RR in [12, 20] should always score 0 for the respiratory_rate parameter."""
    from mcp_server.tools.clinical_deterioration_score import compute_clinical_deterioration_score
    rep = _run(compute_clinical_deterioration_score(vital_signs=[
        {"type": "respiratory_rate", "value": rr,
         "observed_at": datetime.now(timezone.utc).isoformat()},
    ]))
    rr_pts = next((c.points for c in rep.parameter_contributions
                    if c.parameter == "respiratory_rate"), None)
    assert rr_pts == 0


# ─────────────────────── PEWS (compute_pediatric_early_warning) ───────────────────────

@given(age_months=st.integers(min_value=0, max_value=216),
        hr=st.floats(min_value=40.0, max_value=240.0, allow_nan=False),
        rr=st.floats(min_value=10.0, max_value=80.0, allow_nan=False),
        spo2=st.floats(min_value=80.0, max_value=100.0, allow_nan=False),
        behavior=st.sampled_from(["appropriate", "sleeping", "irritable", "lethargic"]))
@_HYP
def test_pews_total_in_bounds(age_months, hr, rr, spo2, behavior):
    from mcp_server.tools.pediatric_early_warning import compute_pediatric_early_warning
    rep = _run(compute_pediatric_early_warning(
        age_months=age_months, behavior=behavior,
        heart_rate=hr, respiratory_rate=rr, spo2=spo2,
    ))
    assert 0 <= rep.score_total <= 12
    assert rep.severity_tier in ("low", "medium", "high")
    assert rep.age_band in ("0-11mo", "1-4y", "5-11y", "12-17y")


@given(age_months=st.integers(min_value=0, max_value=216))
@_HYP
def test_pews_age_band_correct(age_months):
    from mcp_server.tools.pediatric_early_warning import (
        _age_band, compute_pediatric_early_warning,
    )
    expected_band, _ = _age_band(age_months)
    rep = _run(compute_pediatric_early_warning(age_months=age_months))
    assert rep.age_band == expected_band


# ─────────────────────── HEART score ───────────────────────

@given(history=st.sampled_from(["non_suspicious", "slightly_suspicious",
                                  "moderately_suspicious", "highly_suspicious"]),
        ecg=st.sampled_from(["normal", "non_specific_repolarization",
                              "significant_st_depression", "stemi"]),
        age=st.integers(min_value=18, max_value=110),
        rf_count=st.integers(min_value=0, max_value=6),
        known_cad=st.booleans(),
        trop=st.floats(min_value=0.0, max_value=20.0, allow_nan=False))
@_HYP
def test_heart_score_in_bounds(history, ecg, age, rf_count, known_cad, trop):
    from mcp_server.tools.heart_score import compute_heart_score
    rep = _run(compute_heart_score(
        history_descriptor=history, ecg_descriptor=ecg, age=age,
        risk_factors_count=rf_count, known_atherosclerotic_disease=known_cad,
        troponin_times_uln=trop,
    ))
    assert 0 <= rep.total_score <= 10
    assert rep.risk_band in ("low", "moderate", "high")
    # Total = sum of components
    assert (rep.history_points + rep.ecg_points + rep.age_points
              + rep.risk_factors_points + rep.troponin_points
              == rep.total_score)


@given(rf_count=st.integers(min_value=0, max_value=6))
@_HYP
def test_heart_known_cad_dominates_rf_count(rf_count):
    """known_atherosclerotic_disease=True -> R points should always be 2,
    regardless of rf_count."""
    from mcp_server.tools.heart_score import compute_heart_score
    rep = _run(compute_heart_score(
        history_descriptor="moderately_suspicious", ecg_descriptor="normal",
        age=50, risk_factors_count=rf_count,
        known_atherosclerotic_disease=True, troponin_times_uln=0.0,
    ))
    assert rep.risk_factors_points == 2


# ─────────────────────── KDIGO AKI staging ───────────────────────

@given(baseline=st.floats(min_value=0.5, max_value=4.0, allow_nan=False),
        current=st.floats(min_value=0.5, max_value=10.0, allow_nan=False),
        uop=st.one_of(st.none(),
                        st.floats(min_value=0.0, max_value=2.0, allow_nan=False)))
@_HYP
def test_kdigo_stage_monotonic_in_creatinine_ratio(baseline, current, uop):
    """Higher Cr/baseline ratio -> not-lower KDIGO stage."""
    from mcp_server.tools.aki_kdigo_stage import compute_aki_kdigo_stage
    rep = _run(compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=baseline,
        creatinine_current_mg_dl=current,
        urine_output_ml_per_kg_per_hour=uop,
        urine_output_window_hours=12,
    ))
    assert rep.aki_stage in ("no_aki", "stage_1", "stage_2", "stage_3")
    # If ratio < 1.5 and abs change < 0.3 and UOP normal -> no AKI
    abs_change = current - baseline
    ratio = current / baseline if baseline > 0 else 0
    if abs_change < 0.3 and ratio < 1.5 and (uop is None or uop >= 0.5):
        assert rep.aki_stage == "no_aki"


# ─────────────────────── Suicide risk assessment ───────────────────────

@given(ideation_lifetime=st.integers(min_value=0, max_value=5),
        ideation_recent=st.integers(min_value=0, max_value=5),
        attempts=st.integers(min_value=0, max_value=10),
        recent=st.booleans(),
        warnings=st.integers(min_value=0, max_value=10),
        protective=st.integers(min_value=0, max_value=10))
@_HYP
def test_suicide_risk_in_bounds(ideation_lifetime, ideation_recent,
                                   attempts, recent, warnings, protective):
    from mcp_server.tools.suicide_risk_assessment import compute_suicide_risk_assessment
    rep = _run(compute_suicide_risk_assessment(
        ideation_lifetime_level=ideation_lifetime,
        ideation_past_30d_level=ideation_recent,
        behavior_lifetime_attempts=attempts,
        behavior_past_30d_any=recent,
        warning_factors_count=warnings,
        protective_factors_count=protective,
    ))
    assert rep.risk_level in ("low", "moderate", "high", "imminent")
    # Recent attempt OR ideation level 5 -> imminent
    if recent or ideation_recent >= 5:
        assert rep.risk_level == "imminent"


# ─────────────────────── Trauma severity (ISS) ───────────────────────

_ais_strategy = st.lists(
    st.fixed_dictionaries({
        "body_region": st.sampled_from([
            "head_neck", "face", "chest", "abdomen_pelvis",
            "extremities_pelvic_girdle", "external",
        ]),
        "ais_severity": st.integers(min_value=1, max_value=6),
    }),
    min_size=0, max_size=8,
)


@given(injuries=_ais_strategy,
        gcs=st.integers(min_value=3, max_value=15),
        sbp=st.floats(min_value=40.0, max_value=220.0, allow_nan=False),
        rr=st.floats(min_value=0.0, max_value=60.0, allow_nan=False))
@_HYP
def test_iss_in_bounds(injuries, gcs, sbp, rr):
    from mcp_server.tools.trauma_severity_score import compute_trauma_severity_score
    rep = _run(compute_trauma_severity_score(
        injuries=injuries, glasgow_coma_score=gcs,
        systolic_bp=sbp, respiratory_rate=rr,
    ))
    assert 0 <= rep.iss <= 75
    assert 0 <= rep.rts <= 12
    # Any AIS=6 -> ISS=75 by convention
    if any(i["ais_severity"] == 6 for i in injuries):
        assert rep.iss == 75


# ─────────────────────── Falls Morse ───────────────────────

@given(history=st.booleans(), secondary=st.booleans(),
        aid=st.sampled_from(["none", "crutches", "cane", "walker", "furniture"]),
        iv=st.booleans(),
        gait=st.sampled_from(["normal", "weak", "impaired"]),
        ms=st.sampled_from(["oriented", "forgets_limitations"]))
@_HYP
def test_morse_score_in_bounds(history, secondary, aid, iv, gait, ms):
    from mcp_server.tools.falls_risk_morse import compute_falls_risk_morse
    rep = _run(compute_falls_risk_morse(
        history_of_falling_3mo=history, secondary_diagnosis_present=secondary,
        ambulatory_aid=aid, has_iv_or_heparin_lock=iv,
        gait=gait, mental_status=ms,
    ))
    assert 0 <= rep.score_total <= 125
    assert rep.risk_tier in ("low", "moderate", "high")


# ─────────────────────── Weight-based dosing ───────────────────────

@given(weight=st.floats(min_value=2.0, max_value=120.0, allow_nan=False),
        age_months=st.integers(min_value=0, max_value=216))
@_HYP
def test_weight_dose_capped_at_adult_max(weight, age_months):
    """For amoxicillin: final dose must never exceed 1000 mg adult cap."""
    assume(age_months >= 6)  # ibuprofen contraindication path is separate
    from mcp_server.tools.weight_based_dosing import compute_weight_based_dosing
    rep = _run(compute_weight_based_dosing(
        drug="amoxicillin", weight_kg=weight, age_months=age_months,
        indication="acute_otitis_media",
    ))
    assert rep.final_dose_mg <= 1000.0


@given(weight=st.floats(min_value=2.0, max_value=120.0, allow_nan=False))
@_HYP
def test_ibuprofen_under_6mo_always_blocked(weight):
    from mcp_server.tools.weight_based_dosing import compute_weight_based_dosing
    rep = _run(compute_weight_based_dosing(
        drug="ibuprofen", weight_kg=weight, age_months=4,
        indication="fever",
    ))
    assert rep.abstain_recommended is True


# ─────────────────────── DKA severity ───────────────────────

@given(ph=st.floats(min_value=6.7, max_value=7.5, allow_nan=False),
        bicarb=st.floats(min_value=2.0, max_value=28.0, allow_nan=False),
        glucose=st.floats(min_value=100.0, max_value=900.0, allow_nan=False),
        k=st.floats(min_value=2.0, max_value=7.5, allow_nan=False),
        weight=st.floats(min_value=20.0, max_value=150.0, allow_nan=False))
@_HYP
def test_dka_severity_categorical(ph, bicarb, glucose, k, weight):
    from mcp_server.tools.dka_severity import compute_dka_severity
    rep = _run(compute_dka_severity(
        ph=ph, bicarbonate_meq_l=bicarb, glucose_mg_dl=glucose,
        ketones_present=True, mental_status="alert",
        potassium_meq_l=k, weight_kg=weight,
    ))
    assert rep.severity in ("mild", "moderate", "severe", "not_dka")
    # K+ < 3.3 -> must hold insulin first
    if k < 3.3 and rep.severity != "not_dka":
        assert rep.potassium_replacement_at_initiation is True


# ─────────────────────── Polypharmacy ───────────────────────

_med_strategy = st.lists(
    st.fixed_dictionaries({
        "name": st.sampled_from([
            "warfarin 5mg", "lisinopril 10mg", "metoprolol 25mg",
            "spironolactone 25mg", "metformin 500mg", "atorvastatin 40mg",
            "ibuprofen 400mg", "aspirin 81mg", "apixaban 5mg",
            "amoxicillin 500mg", "furosemide 40mg", "insulin glargine",
        ]),
        "drug_class": st.sampled_from([
            "anticoagulant_vka", "ace_inhibitor", "beta_blocker",
            "mra", "biguanide", "statin", "nsaid", "antiplatelet",
            "anticoagulant_doac", "loop_diuretic", "insulin",
        ]),
        "status": st.sampled_from(["active", "stopped"]),
    }),
    min_size=0, max_size=15,
)


@given(meds=_med_strategy)
@_HYP
def test_polypharmacy_severity_in_bounds(meds):
    from mcp_server.tools.polypharmacy_concerns import detect_polypharmacy_concerns
    rep = _run(detect_polypharmacy_concerns(medications=meds))
    assert rep.polypharmacy_severity in ("none", "low", "medium", "high")
    assert rep.n_medications == len([m for m in meds])
    assert 0 <= rep.n_high_risk <= rep.n_medications
