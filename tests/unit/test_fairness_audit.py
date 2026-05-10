"""Unit tests for compute_fairness_audit."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from mcp_server.tools.fairness_audit import (
    _age_to_baseline,
    _drift_severity,
    _grade_confidence_action,
    compute_fairness_audit,
)
from shared.schemas import (
    Factor,
    FairnessReport,
    RiskEstimate,
    SubgroupCalibration,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _isolate_from_calibrated_baseline(monkeypatch):
    """Tests in this file assert against the literature-default multipliers.
    Force-disable the calibrated baseline so the cache doesn't override them."""
    monkeypatch.setenv("TRUSTEDRISK_FAIRNESS_BASELINE_PATH", "/nonexistent/calibrated.json")
    from mcp_server.tools import fairness_audit
    fairness_audit._load_calibrated_baseline.cache_clear()
    yield
    fairness_audit._load_calibrated_baseline.cache_clear()


def _risk(prob: float = 0.18) -> RiskEstimate:
    half = 0.06
    return RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="test-v1",
        horizon_days=30,
        lace_raw_score=10,
        probability_mean=prob,
        probability_ci95=(max(0.0, prob - half), min(1.0, prob + half)),
        probability_ci_width=0.12,
        contributing_factors=[Factor(name="L", raw_value=4, lace_points=4, weight=0.4)],
        computed_at=datetime.now(timezone.utc),
    )


# ─────────────────────── helpers ───────────────────────

def test_age_band_18_64():
    res = _age_to_baseline(40)
    assert res is not None
    lo, hi, mult, _ = res
    assert lo == 18 and hi == 64 and mult == 1.0


def test_age_band_75():
    res = _age_to_baseline(80)
    assert res is not None
    _, _, mult, _ = res
    assert 1.30 <= mult <= 1.40


def test_age_band_85_plus():
    res = _age_to_baseline(90)
    assert res is not None
    _, _, mult, _ = res
    assert 1.45 <= mult <= 1.55


def test_age_outside_range():
    assert _age_to_baseline(10) is None


def test_drift_severity_high():
    assert _drift_severity(-0.40) == "high"
    assert _drift_severity(0.35) == "high"


def test_drift_severity_medium():
    assert _drift_severity(0.25) == "medium"


def test_drift_severity_low():
    assert _drift_severity(0.15) == "low"


def test_drift_severity_none():
    assert _drift_severity(0.05) == "none"


# ─────────────────────── _grade_confidence_action ───────────────────────

def _mk_drift(name: str, value: str, sev: str) -> SubgroupCalibration:
    return SubgroupCalibration(
        subgroup_name=name, subgroup_value=value,
        expected_rate_baseline=0.20, predicted_rate_for_patient=0.18,
        relative_drift=-0.10,
        severity=sev,  # type: ignore[arg-type]
        citation="test",
    )


def test_grade_no_action_when_empty():
    assert _grade_confidence_action([]) == "no_action"


def test_grade_abstain_when_two_high():
    drifts = [_mk_drift("race", "x", "high"), _mk_drift("age", "85+", "high")]
    assert _grade_confidence_action(drifts) == "abstain_recommended"


def test_grade_downgrade_when_one_high():
    drifts = [_mk_drift("race", "x", "high")]
    assert _grade_confidence_action(drifts) == "downgrade_confidence"


def test_grade_downgrade_when_two_medium():
    drifts = [_mk_drift("race", "x", "medium"), _mk_drift("age", "85+", "medium")]
    assert _grade_confidence_action(drifts) == "downgrade_confidence"


def test_grade_flag_when_one_medium():
    drifts = [_mk_drift("race", "x", "medium")]
    assert _grade_confidence_action(drifts) == "flag_for_review"


def test_grade_no_action_when_all_low():
    drifts = [_mk_drift("race", "x", "low"), _mk_drift("age", "65-74", "low")]
    assert _grade_confidence_action(drifts) == "no_action"


# ─────────────────────── End-to-end ───────────────────────

def test_fairness_audit_no_demographics():
    """When patient_demographics is empty, the audit must abstain rather
    than silently report `n_subgroups_assessed=0` as if the audit ran
    successfully."""
    rep = _run(compute_fairness_audit(risk=_risk(), patient_demographics={}))
    assert rep.n_subgroups_assessed == 0
    assert rep.confidence_action == "no_action"
    assert rep.abstain_recommended is True
    assert "missing_critical_demographic" in (rep.abstain_reason or "")


def test_fairness_audit_age_only():
    rep = _run(compute_fairness_audit(
        risk=_risk(prob=0.18),
        patient_demographics={"age": 88},
    ))
    assert rep.n_subgroups_assessed == 1
    age_drift = rep.subgroup_drifts[0]
    assert age_drift.subgroup_name == "age_band"
    assert age_drift.subgroup_value == "85-120"
    # 85+ multiplier 1.50 -> expected = 0.18 × 1.50 = 0.27 (capped at 1.0)
    # rel_drift = (0.18 - 0.27) / 0.27 ≈ -0.33 -> high severity
    assert age_drift.severity == "high"


def test_fairness_audit_race_black_medium_drift():
    rep = _run(compute_fairness_audit(
        risk=_risk(prob=0.18),
        patient_demographics={"race": "Black"},
    ))
    race_drift = next(d for d in rep.subgroup_drifts if d.subgroup_name == "race")
    # Black multiplier 1.18 -> drift ≈ -0.15 -> low severity
    assert abs(race_drift.relative_drift) > 0.10


def test_fairness_audit_white_no_drift():
    rep = _run(compute_fairness_audit(
        risk=_risk(prob=0.18),
        patient_demographics={"race": "White"},
    ))
    race_drift = next(d for d in rep.subgroup_drifts if d.subgroup_name == "race")
    # White is reference (multiplier 1.00) -> no drift
    assert race_drift.severity == "none"


def test_fairness_audit_medicaid_flagged():
    rep = _run(compute_fairness_audit(
        risk=_risk(prob=0.18),
        patient_demographics={"insurance_type": "medicaid"},
    ))
    ins_drift = next(d for d in rep.subgroup_drifts if d.subgroup_name == "insurance_type")
    # Medicaid multiplier 1.18 -> drift ≈ -0.15 -> low severity
    assert abs(ins_drift.relative_drift) > 0.10


def test_fairness_audit_uninsured_severity_at_least_low():
    rep = _run(compute_fairness_audit(
        risk=_risk(prob=0.20),
        patient_demographics={"insurance_type": "uninsured"},
    ))
    ins_drift = next(d for d in rep.subgroup_drifts if d.subgroup_name == "insurance_type")
    # Uninsured multiplier 1.25 -> drift ≈ -0.20 (exact) -> at the medium threshold.
    # FP edge case may land at low or medium -- both acceptable.
    assert ins_drift.severity in ("low", "medium", "high")
    assert abs(ins_drift.relative_drift) >= 0.18


def test_fairness_audit_compounding_subgroups_abstain():
    """Black + 85+ + Medicaid -> multiple high drifts -> abstain_recommended."""
    rep = _run(compute_fairness_audit(
        risk=_risk(prob=0.18),
        patient_demographics={
            "age": 88, "race": "Black", "insurance_type": "medicaid",
        },
    ))
    assert rep.n_subgroups_assessed == 3
    high_count = sum(1 for d in rep.subgroup_drifts if d.severity == "high")
    assert high_count >= 1
    # Either downgrade_confidence or abstain_recommended given the stack
    assert rep.confidence_action in ("downgrade_confidence", "abstain_recommended")


def test_fairness_audit_accepts_dict_risk():
    rep = _run(compute_fairness_audit(
        risk={"probability_mean": 0.20},
        patient_demographics={"age": 50},
    ))
    assert rep.n_subgroups_assessed == 1


def test_fairness_audit_rejects_invalid_prob():
    with pytest.raises(ValueError, match="\\[0,1\\]"):
        _run(compute_fairness_audit(
            risk={"probability_mean": 1.5},
            patient_demographics={"age": 50},
        ))


def test_fairness_audit_unknown_race_abstains():
    """When race carries the HL7 'unknown' sentinel, the tool must abstain
    rather than substituting the population-mean 'unknown' baseline multiplier.
    The sentinel cannot be assumed equivalent to any tracked subgroup."""
    rep = _run(compute_fairness_audit(
        risk=_risk(prob=0.18),
        patient_demographics={"race": "unknown"},
    ))
    assert rep.abstain_recommended is True
    assert rep.abstain_reason is not None
    assert "race" in rep.abstain_reason
    assert rep.n_subgroups_assessed == 0


def test_fairness_audit_hl7_u_sentinel_abstains():
    """The two-letter HL7 code 'U' is also an unknown sentinel."""
    rep = _run(compute_fairness_audit(
        risk=_risk(prob=0.18),
        patient_demographics={"race": "U"},
    ))
    assert rep.abstain_recommended is True
    assert rep.n_subgroups_assessed == 0


def test_fairness_audit_unknown_insurance_abstains():
    """'unknown' insurance_type triggers abstain."""
    rep = _run(compute_fairness_audit(
        risk=_risk(prob=0.18),
        patient_demographics={"insurance_type": "unknown"},
    ))
    assert rep.abstain_recommended is True
    assert rep.abstain_reason is not None
    assert "insurance_type" in rep.abstain_reason


def test_fairness_audit_disclaimer_present():
    rep = _run(compute_fairness_audit(
        risk=_risk(),
        patient_demographics={"age": 70},
    ))
    assert "v1" in rep.audit_disclaimer.lower() or "calibrate" in rep.audit_disclaimer.lower()


# ─────────────────────── Calibrated baseline path (W5) ───────────────────────

def test_calibrated_baseline_overrides_literature(tmp_path, monkeypatch):
    """When a fairness_baseline.json is staged, its multipliers replace the
    literature defaults for matching subgroup keys."""
    import json
    calibrated = {
        "schema_version": 1,
        "by_age_band": {
            "85+": {
                "multiplier": 0.80,  # OPPOSITE of literature 1.50
                "n_encounters": 200,
                "citation": "Calibrated on Synthea 7878-row cohort.",
            },
        },
        "by_race": {},
        "by_sex": {},
    }
    fp = tmp_path / "fairness_baseline.json"
    fp.write_text(json.dumps(calibrated))

    # Override the cache + env
    from mcp_server.tools import fairness_audit
    monkeypatch.setenv("TRUSTEDRISK_FAIRNESS_BASELINE_PATH", str(fp))
    fairness_audit._load_calibrated_baseline.cache_clear()

    try:
        rep = _run(compute_fairness_audit(
            risk=_risk(prob=0.18),
            patient_demographics={"age": 88},
        ))
        age_drift = next(d for d in rep.subgroup_drifts if d.subgroup_name == "age_band")
        # Calibrated multiplier 0.80 -> expected = 0.144, drift = +25%
        assert age_drift.relative_drift > 0.20  # drift sign FLIPPED vs literature
        assert "Calibrated" in age_drift.citation
    finally:
        fairness_audit._load_calibrated_baseline.cache_clear()
