"""Phase 15.C3 -- Tests for the synthetic prospective evaluation."""

from __future__ import annotations

import pytest

from a2a_agent.prospective_eval import (
    DISPOSITION_THRESHOLD, ProspectiveSummary,
    _baseline_lace_only, _evaluate_encounter,
    _propose_initial_action, run_prospective_eval,
)


# ─────────────────────────────────────────────────────────────────────
# Initial-action proposer
# ─────────────────────────────────────────────────────────────────────

def test_propose_initial_action_low_risk_is_discharge_home():
    assert _propose_initial_action(0.04) == "discharge_home"


def test_propose_initial_action_moderate_risk_is_homecare():
    # threshold=0.20, half-threshold=0.10 -> 0.15 sits in homecare band
    assert _propose_initial_action(0.15) == "discharge_with_homecare"


def test_propose_initial_action_high_risk_is_continued_admission():
    assert _propose_initial_action(0.45) == "continued_admission"


def test_propose_initial_action_at_threshold_is_continued_admission():
    """The risk == threshold case is conservative: continue admission."""
    assert (
        _propose_initial_action(DISPOSITION_THRESHOLD)
        == "continued_admission"
    )


# ─────────────────────────────────────────────────────────────────────
# Per-encounter pipeline
# ─────────────────────────────────────────────────────────────────────

def _dem(**overrides):
    base = dict(
        age=70, age_band="65-74", age_mult=1.05,
        sex="male", race="white", ethnicity="non_hispanic",
        insurance_type="commercial", language="english",
    )
    base.update(overrides)
    return base


def test_unflagged_low_risk_encounter_yields_approved():
    res = _evaluate_encounter(
        enc_id=0, lace=2, dem=_dem(),
        risk_estimate=0.05, ci_width=0.03, outcome=0,
        fairness_audit_present=True,
    )
    assert res.debate_verdict == "approved"
    assert res.abstained is False
    assert res.downgraded is False
    assert res.proposed_action == "discharge_home"
    assert res.final_action == "discharge_home"


def test_flagged_subgroup_without_audit_forces_abstain():
    res = _evaluate_encounter(
        enc_id=0, lace=4, dem=_dem(race="black"),
        risk_estimate=0.12, ci_width=0.045, outcome=1,
        fairness_audit_present=False,
    )
    assert res.fairness_flagged is True
    assert res.abstained is True
    assert res.final_action == "abstain"


def test_flagged_subgroup_with_audit_revises_to_homecare():
    res = _evaluate_encounter(
        enc_id=0, lace=2, dem=_dem(insurance_type="medicaid"),
        risk_estimate=0.06, ci_width=0.045, outcome=0,
        fairness_audit_present=True,
    )
    assert res.fairness_flagged is True
    assert res.abstained is False
    assert res.downgraded is True
    # discharge_home initial -> downgraded to discharge_with_homecare
    assert res.final_action == "discharge_with_homecare"


# ─────────────────────────────────────────────────────────────────────
# run_prospective_eval -- invariants
# ─────────────────────────────────────────────────────────────────────

def test_run_prospective_eval_n_must_be_positive():
    with pytest.raises(ValueError):
        run_prospective_eval(0)


def test_run_prospective_eval_yields_summary_with_n():
    summary = run_prospective_eval(200, seed=1)
    assert isinstance(summary, ProspectiveSummary)
    assert summary.cohort_n == 200
    assert summary.overall["n"] == 200


def test_run_prospective_eval_is_deterministic_for_seed():
    a = run_prospective_eval(150, seed=42)
    b = run_prospective_eval(150, seed=42)
    assert a.model_dump() == b.model_dump()


def test_run_prospective_eval_six_subgroup_axes():
    summary = run_prospective_eval(300, seed=2)
    assert set(summary.per_subgroup) == {
        "age_band", "sex", "race", "ethnicity",
        "insurance_type", "language",
    }


def test_run_prospective_eval_fairness_audit_off_increases_abstain():
    """Toggling fairness_audit_present=False must not lower the abstain
    rate -- it removes the 'present' override that revises instead of
    abstains for flagged subgroups."""
    on = run_prospective_eval(2000, seed=3, fairness_audit_present=True)
    off = run_prospective_eval(2000, seed=3, fairness_audit_present=False)
    assert (
        off.overall["abstain_rate"] >= on.overall["abstain_rate"]
    )


def test_run_prospective_eval_calibration_gap_under_5pct():
    """The calibrated risk model + Beta-Binomial should track the
    simulator's true generative outcome rate within ~5 percentage
    points on a moderately sized cohort."""
    summary = run_prospective_eval(5000, seed=4)
    assert summary.overall["calibration_gap_abs"] < 0.05


def test_run_prospective_eval_baseline_lace_only_has_no_abstain():
    summary = run_prospective_eval(500, seed=5)
    assert summary.baseline_lace_only["abstain_rate"] == 0.0
    assert summary.baseline_lace_only["downgrade_rate"] == 0.0


# ─────────────────────────────────────────────────────────────────────
# Schema invariants
# ─────────────────────────────────────────────────────────────────────

def test_summary_round_trips_through_pydantic():
    summary = run_prospective_eval(200, seed=6)
    payload = summary.model_dump(mode="json")
    rebuilt = ProspectiveSummary.model_validate(payload)
    assert rebuilt.cohort_n == summary.cohort_n
    assert (
        rebuilt.overall["abstain_rate"]
        == summary.overall["abstain_rate"]
    )


def test_action_distribution_counts_sum_to_n():
    summary = run_prospective_eval(800, seed=7)
    total = sum(summary.overall["action_distribution"].values())
    assert total == summary.cohort_n
