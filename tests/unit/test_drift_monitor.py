"""Unit tests for SCI-3 calibration drift monitor."""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from a2a_agent.drift_monitor import (
    _ece,
    _ks_distance,
    _lace_to_bucket,
    compute_drift_report,
)
from a2a_agent.memory import MemoryStore
from shared.schemas import (
    Action, AuditBlock, ClaimGrounding, DecisionCard, DecisionReasoning,
    DecisionValidation, Factor, PHIReport, Recommendation, RiskEstimate,
    UtilityAnalysis, AbstainTrigger,
)


def _make_card(*, prob: float = 0.10, lace: int = 5, ci_width: float = 0.06,
                abstain: bool = False, request_id: str = "rid-1",
                patient_id: str = "pt-1") -> DecisionCard:
    risk = RiskEstimate(
        model_name="lace-plus-bayesian-v1", model_version="t",
        outcome_id="readmission_30d",
        horizon_days=30, lace_raw_score=lace,
        probability_mean=prob,
        probability_ci95=(max(0.0, prob - ci_width / 2),
                            min(1.0, prob + ci_width / 2)),
        probability_ci_width=ci_width,
        contributing_factors=[Factor(name="x", raw_value=0.0,
                                        lace_points=0, weight=0.0)],
        fhir_observations_used=[],
        computed_at=datetime.now(timezone.utc),
    )
    util = UtilityAnalysis(
        action_scores_qaly_weeks={Action.DISCHARGE_HOME: 0.8},
        action_scores_ci95={Action.DISCHARGE_HOME: (0.6, 1.0)},
        action_costs_usd={Action.DISCHARGE_HOME: 0.0},
        dominant_action=Action.DISCHARGE_HOME, dominance_confidence=0.95,
        reasoning_trace="x",
    )
    grounding = ClaimGrounding(claim_text="x", sub_claims=[],
                                 overall_verdict="supported",
                                 context_fingerprint=patient_id,
                                 grounded_at=datetime.now(timezone.utc))
    phi = PHIReport(entities_found=[], entity_count_by_type={},
                     redaction_map={}, risk_level="none")
    audit = AuditBlock(request_id=request_id, context_fingerprint=patient_id,
                         tool_trace=[], server_version="t", agent_version="t",
                         model_coefficients_version="t",
                         timestamp=datetime.now(timezone.utc))
    return DecisionCard(
        recommendation=Recommendation(action=Action.DISCHARGE_HOME,
                                          confidence="high"),
        reasoning=DecisionReasoning(risk_estimate=risk, utility_analysis=util),
        validation=DecisionValidation(grounding=grounding, phi_check=phi),
        abstain=([AbstainTrigger(type="evidence_insufficient",
                                    detail="test")] if abstain else None),
        audit=audit,
    )


# ─────────────────────── Helpers ───────────────────────

def test_lace_buckets():
    assert _lace_to_bucket(0) == "0-2"
    assert _lace_to_bucket(2) == "0-2"
    assert _lace_to_bucket(3) == "3-5"
    assert _lace_to_bucket(9) == "6-9"
    assert _lace_to_bucket(12) == "10-12"
    assert _lace_to_bucket(15) == "13-19"


def test_ks_distance_zero_when_identical():
    h = {"0-2": 0.2, "3-5": 0.3, "6-9": 0.5}
    assert _ks_distance(h, h) == 0.0


def test_ks_distance_positive_when_shifted():
    a = {"0-2": 1.0}
    b = {"13-19": 1.0}
    assert _ks_distance(a, b) > 0.5


def test_ece_zero_for_perfect_calibration():
    """If predictions == observed rate per bin -> ECE = 0."""
    preds = [0.05] * 10 + [0.5] * 10 + [0.95] * 10
    outs = [0] * 10 + [0] * 5 + [1] * 5 + [1] * 10
    ece = _ece(preds, outs)
    # Bin 0.0-0.1 has mean_pred=0.05 obs=0; bin 0.4-0.5 mean=0.5 obs=0.5; bin 0.9-1.0 mean=0.95 obs=1.0
    # Total |gap| weighted: ~ 1/3 * 0.05 + 1/3 * 0 + 1/3 * 0.05 = ~0.033
    assert ece < 0.05


def test_ece_high_when_miscalibrated():
    preds = [0.10] * 20
    outs = [1] * 20  # all observed; predicted 10% -> big gap
    assert _ece(preds, outs) > 0.5


# ─────────────────────── End-to-end ───────────────────────

def test_drift_report_empty_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(db_path=Path(tmp) / "mem.sqlite")
        rep = compute_drift_report(store=store, window_hours=24)
    assert rep.window_n_cards == 0
    assert rep.overall_tier == "ok"
    assert rep.signals == []


def test_drift_report_baseline_distribution_no_drift():
    """Cards mirroring training baseline -> no drift signals fire."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(db_path=Path(tmp) / "mem.sqlite")
        # 10 cards spread across the LACE histogram with mean prob ~0.11
        for i in range(10):
            store.store_card(_make_card(
                prob=0.10 + i * 0.005,  # mean ~0.13 -- within 0.05 drift
                lace=[5, 5, 7, 7, 8, 9, 9, 11, 11, 14][i],
                request_id=f"rid-{i}",
            ))
        rep = compute_drift_report(store=store, window_hours=24)
    assert rep.window_n_cards == 10
    assert rep.overall_tier in ("ok", "warn")  # tolerance


def test_drift_report_alert_on_high_abstain_rate():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(db_path=Path(tmp) / "mem.sqlite")
        for i in range(10):
            # 6 of 10 cards have abstain -> 60% rate
            abstain = i < 6
            store.store_card(_make_card(
                prob=0.10, lace=7, abstain=abstain,
                request_id=f"rid-{i}",
            ))
        rep = compute_drift_report(store=store, window_hours=24)
    abstain_signal = next(s for s in rep.signals if s.name == "abstain_rate")
    assert abstain_signal.tier == "alert"
    assert rep.overall_tier == "alert"


def test_drift_report_alert_on_mean_prob_shift():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(db_path=Path(tmp) / "mem.sqlite")
        # All cards predict 0.40 (training baseline = 0.11) -> ~30pp shift
        for i in range(10):
            store.store_card(_make_card(prob=0.40, lace=15,
                                            request_id=f"rid-{i}"))
        rep = compute_drift_report(store=store, window_hours=24)
    mean_signal = next(s for s in rep.signals
                        if s.name == "mean_predicted_probability")
    assert mean_signal.tier == "alert"


def test_drift_report_post_hoc_ece_when_outcomes_provided():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(db_path=Path(tmp) / "mem.sqlite")
        # 10 cards predicting 0.10; suppose actual outcome is 50% positive
        # -> big calibration miss
        for i in range(10):
            store.store_card(_make_card(prob=0.10, lace=5,
                                            request_id=f"rid-{i}"))
        outcomes = {f"rid-{i}": (1 if i < 5 else 0) for i in range(10)}
        rep = compute_drift_report(store=store, window_hours=24,
                                       outcomes_observed=outcomes)
    assert rep.n_outcomes_observed == 10
    assert rep.ece_post_hoc is not None
    assert rep.ece_post_hoc > 0.30  # 0.10 predicted vs 0.50 observed


def test_drift_report_window_filters_old_cards():
    """Cards outside the window must be excluded."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(db_path=Path(tmp) / "mem.sqlite")
        # Insert 5 cards
        for i in range(5):
            store.store_card(_make_card(prob=0.40, lace=15,
                                            request_id=f"rid-old-{i}"))
        # Manipulate written_at_iso so they look like 30 days old
        with store._connect() as conn:
            old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            conn.execute("UPDATE decision_cards SET written_at_iso = ?", (old,))
        rep = compute_drift_report(store=store, window_hours=24)
    assert rep.window_n_cards == 0
    assert rep.overall_tier == "ok"
