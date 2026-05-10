"""Unit tests for AMB-5.3 longitudinal patient timeline + drift detection."""
from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from a2a_agent.memory import (
    MemoryStore,
    _compute_longitudinal_drift,
    _infer_encounter_type,
)
from shared.schemas import (
    Action,
    AuditBlock,
    ClaimGrounding,
    DecisionCard,
    DecisionReasoning,
    DecisionValidation,
    Factor,
    PatientTimelineEntry,
    PHIReport,
    Recommendation,
    RiskEstimate,
    UtilityAnalysis,
)


def _entry(idx: int, action: str, prob: float, conf: str = "high",
            abstain_count: int = 0) -> PatientTimelineEntry:
    return PatientTimelineEntry(
        encounter_id=f"req-{idx}",
        encounter_at=datetime(2026, 1, idx + 1, tzinfo=timezone.utc),
        encounter_type="discharge",  # type: ignore[arg-type]
        recommendation_action=action,
        recommendation_confidence=conf,
        risk_probability_mean=prob,
        risk_outcome_id="readmission_30d",
        abstain_triggers_count=abstain_count,
    )


# ─────────────────────── Drift detection ───────────────────────

def test_no_drift_for_stable_timeline():
    entries = [
        _entry(0, "discharge_home", 0.10),
        _entry(1, "discharge_home", 0.11),
        _entry(2, "discharge_home", 0.09),
    ]
    flags = _compute_longitudinal_drift(entries)
    assert flags == []


def test_drift_action_escalation():
    entries = [
        _entry(0, "discharge_home", 0.10),
        _entry(1, "continued_admission", 0.40),
    ]
    flags = _compute_longitudinal_drift(entries)
    assert any("action escalation" in f for f in flags)


def test_drift_probability_worsening():
    entries = [
        _entry(0, "discharge_home", 0.10),
        _entry(1, "discharge_home", 0.30),
    ]
    flags = _compute_longitudinal_drift(entries)
    assert any("probability_mean worsening" in f for f in flags)


def test_drift_confidence_drop_two_tiers():
    entries = [
        _entry(0, "discharge_home", 0.10, conf="high"),
        _entry(1, "discharge_home", 0.10, conf="low"),
    ]
    flags = _compute_longitudinal_drift(entries)
    assert any("confidence drop" in f for f in flags)


def test_drift_abstain_emerging():
    entries = [
        _entry(0, "discharge_home", 0.10, abstain_count=0),
        _entry(1, "discharge_home", 0.10, abstain_count=2),
    ]
    flags = _compute_longitudinal_drift(entries)
    assert any("abstain emerged" in f for f in flags)


def test_drift_single_entry_no_flags():
    flags = _compute_longitudinal_drift([_entry(0, "discharge_home", 0.10)])
    assert flags == []


# ─────────────────────── Encounter type inference ───────────────────────

def _make_card(action: str | None = "discharge_home",
                  abstain: bool = False) -> DecisionCard:
    risk = RiskEstimate(
        model_name="lace-plus-bayesian-v1", model_version="test",
        outcome_id="readmission_30d",
        horizon_days=30, lace_raw_score=5,
        probability_mean=0.10, probability_ci95=(0.07, 0.13),
        probability_ci_width=0.06,
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
    grounding = ClaimGrounding(
        claim_text="x", sub_claims=[], overall_verdict="supported",
        context_fingerprint="abc",
        grounded_at=datetime.now(timezone.utc),
    )
    phi = PHIReport(entities_found=[], entity_count_by_type={},
                     redaction_map={}, risk_level="none")
    audit = AuditBlock(
        request_id="req-test", context_fingerprint="abc",
        tool_trace=[], server_version="test", agent_version="test",
        model_coefficients_version="test",
        timestamp=datetime.now(timezone.utc),
    )
    rec = (Recommendation(action=Action(action), confidence="high")  # type: ignore[arg-type]
            if action and not abstain else None)
    abstain_list = (
        [{"type": "evidence_insufficient", "detail": "test"}]
        if abstain else None
    )
    return DecisionCard(
        recommendation=rec,
        reasoning=DecisionReasoning(risk_estimate=risk,
                                       utility_analysis=util),
        validation=DecisionValidation(grounding=grounding, phi_check=phi),
        abstain=abstain_list,  # type: ignore[arg-type]
        audit=audit,
    )


def test_encounter_type_discharge():
    assert _infer_encounter_type(_make_card("discharge_home")) == "discharge"
    assert _infer_encounter_type(_make_card("home_with_care")) == "discharge"
    assert _infer_encounter_type(_make_card("snf")) == "discharge"


def test_encounter_type_inpatient_review():
    assert _infer_encounter_type(_make_card("continued_admission")) == "inpatient_review"


def test_encounter_type_ed_visit_when_abstain():
    assert _infer_encounter_type(_make_card(action=None, abstain=True)) == "ed_visit"


# ─────────────────────── Memory.build_patient_timeline ───────────────────────

def test_build_patient_timeline_empty():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(db_path=Path(tmp) / "mem.sqlite")
        timeline = store.build_patient_timeline("pt-empty")
        assert timeline.n_encounters == 0
        assert timeline.entries == []
        assert timeline.drift_flags == []


def test_build_patient_timeline_chronological():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(db_path=Path(tmp) / "mem.sqlite")
        # Store 3 cards for the same patient
        for i in range(3):
            card = _make_card("discharge_home")
            # Override request_id + context_fingerprint so they cluster
            card = card.model_copy(update={
                "audit": card.audit.model_copy(update={
                    "request_id": f"req-pt-X-{i}",
                    "context_fingerprint": "pt-X",
                }),
            })
            store.store_card(card)
        timeline = store.build_patient_timeline("pt-X")
        assert timeline.n_encounters == 3
        assert timeline.first_encounter_at is not None
        assert timeline.last_encounter_at is not None
        assert (timeline.last_encounter_at >= timeline.first_encounter_at)


def test_build_patient_timeline_detects_escalation_drift():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(db_path=Path(tmp) / "mem.sqlite")
        # First a discharge, then a continued_admission -> escalation drift
        card1 = _make_card("discharge_home")
        card1 = card1.model_copy(update={
            "audit": card1.audit.model_copy(update={
                "request_id": "req-pt-Y-1",
                "context_fingerprint": "pt-Y",
            }),
        })
        store.store_card(card1)

        card2 = _make_card("continued_admission")
        card2 = card2.model_copy(update={
            "audit": card2.audit.model_copy(update={
                "request_id": "req-pt-Y-2",
                "context_fingerprint": "pt-Y",
            }),
        })
        store.store_card(card2)

        timeline = store.build_patient_timeline("pt-Y")
        assert timeline.n_encounters == 2
        assert any("action escalation" in f for f in timeline.drift_flags)
