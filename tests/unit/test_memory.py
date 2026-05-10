"""Unit tests for the memory layer (SQLite-backed prior-card store)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from a2a_agent.memory import MemoryStore, detect_recommendation_drift
from shared.schemas import (
    AbstainTrigger,
    Action,
    AuditBlock,
    ClaimGrounding,
    DecisionCard,
    DecisionReasoning,
    DecisionValidation,
    Factor,
    PHIReport,
    Recommendation,
    RiskEstimate,
    ToolInvocationTrace,
    UtilityAnalysis,
)


def _now():
    return datetime.now(timezone.utc)


def _card(*, request_id="req-1", context_fingerprint="patient-X",
          action=Action.HOME_WITH_CARE, confidence="high",
          prob=0.18, ts=None, valid_for_min=720,
          model_version="readmission-test",
          abstain=None) -> DecisionCard:
    ts = ts or _now()
    risk = RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version=model_version,
        horizon_days=30,
        lace_raw_score=10,
        probability_mean=prob,
        probability_ci95=(max(0.0, prob - 0.05), min(1.0, prob + 0.05)),
        probability_ci_width=0.10,
        contributing_factors=[Factor(name="L", raw_value=4, lace_points=4, weight=0.4)],
        computed_at=ts,
        valid_for_minutes=valid_for_min,
        valid_until=ts + timedelta(minutes=valid_for_min),
    )
    util = UtilityAnalysis(
        action_scores_qaly_weeks={Action.HOME_WITH_CARE: 4.5},
        action_scores_ci95={Action.HOME_WITH_CARE: (4.0, 5.0)},
        action_costs_usd={Action.HOME_WITH_CARE: 1500.0},
        dominant_action=Action.HOME_WITH_CARE,
        dominance_confidence=0.8,
        reasoning_trace="ok",
    )
    grounding = ClaimGrounding(
        claim_text="ok", sub_claims=[], overall_verdict="supported",
        context_fingerprint=context_fingerprint, grounded_at=ts,
    )
    phi = PHIReport(entities_found=[], entity_count_by_type={}, redaction_map={}, risk_level="none")
    audit = AuditBlock(
        request_id=request_id, context_fingerprint=context_fingerprint,
        tool_trace=[ToolInvocationTrace(tool="x", invoked_at=ts, duration_ms=10, status="ok")],
        server_version="0.1", agent_version="0.1",
        model_coefficients_version=model_version, timestamp=ts,
    )
    return DecisionCard(
        recommendation=Recommendation(action=action, confidence=confidence) if action else None,  # type: ignore[arg-type]
        reasoning=DecisionReasoning(risk_estimate=risk, utility_analysis=util),
        validation=DecisionValidation(grounding=grounding, phi_check=phi),
        abstain=abstain, audit=audit,
    )


# ─────────────────────── store / retrieve roundtrip ───────────────────────

def test_store_and_get_latest(tmp_path: Path):
    store = MemoryStore(db_path=tmp_path / "mem.db")
    card = _card(request_id="r1", context_fingerprint="patient-A")
    store.store_card(card)

    retrieved = store.get_latest_for_patient("patient-A")
    assert retrieved is not None
    assert retrieved.audit.request_id == "r1"
    assert retrieved.recommendation.action == Action.HOME_WITH_CARE


def test_get_latest_missing_patient(tmp_path: Path):
    store = MemoryStore(db_path=tmp_path / "mem.db")
    assert store.get_latest_for_patient("nobody-here") is None


def test_replace_on_same_request_id(tmp_path: Path):
    store = MemoryStore(db_path=tmp_path / "mem.db")
    store.store_card(_card(request_id="r1", context_fingerprint="patient-A", prob=0.10))
    store.store_card(_card(request_id="r1", context_fingerprint="patient-A", prob=0.30))

    assert store.count() == 1
    retrieved = store.get_latest_for_patient("patient-A")
    assert retrieved.reasoning.risk_estimate.probability_mean == 0.30


def test_get_latest_picks_most_recent(tmp_path: Path):
    store = MemoryStore(db_path=tmp_path / "mem.db")
    older = _card(request_id="r1", context_fingerprint="patient-A",
                  ts=_now() - timedelta(hours=2))
    newer = _card(request_id="r2", context_fingerprint="patient-A",
                  ts=_now())
    store.store_card(older)
    store.store_card(newer)

    retrieved = store.get_latest_for_patient("patient-A")
    assert retrieved.audit.request_id == "r2"


# ─────────────────────── validity gating ───────────────────────

def test_get_latest_skips_expired_when_require_valid(tmp_path: Path):
    store = MemoryStore(db_path=tmp_path / "mem.db")
    # 5h ago with 60-min validity -> expired
    expired = _card(request_id="r1", context_fingerprint="patient-A",
                    ts=_now() - timedelta(hours=5), valid_for_min=60)
    store.store_card(expired)

    assert store.get_latest_for_patient("patient-A", require_valid=True) is None
    # But returns it when validity check is disabled
    assert store.get_latest_for_patient("patient-A", require_valid=False) is not None


def test_get_latest_filters_by_model_version(tmp_path: Path):
    store = MemoryStore(db_path=tmp_path / "mem.db")
    store.store_card(_card(request_id="r1", context_fingerprint="patient-A",
                           model_version="v1"))

    # Asking for v2 returns nothing
    assert store.get_latest_for_patient("patient-A",
                                         require_model_version="v2") is None
    # Asking for v1 returns it
    assert store.get_latest_for_patient("patient-A",
                                         require_model_version="v1") is not None


# ─────────────────────── history retrieval ───────────────────────

def test_get_history_returns_in_descending_order(tmp_path: Path):
    store = MemoryStore(db_path=tmp_path / "mem.db")
    base = _now()
    for i in range(5):
        store.store_card(_card(
            request_id=f"r{i}",
            context_fingerprint="patient-A",
            ts=base + timedelta(minutes=i),
        ))
    history = store.get_history_for_patient("patient-A", limit=10)
    assert len(history) == 5
    # Most recent first
    assert history[0].audit.request_id == "r4"
    assert history[-1].audit.request_id == "r0"


def test_get_history_respects_limit(tmp_path: Path):
    store = MemoryStore(db_path=tmp_path / "mem.db")
    for i in range(10):
        store.store_card(_card(request_id=f"r{i}", context_fingerprint="patient-A"))
    history = store.get_history_for_patient("patient-A", limit=3)
    assert len(history) == 3


def test_isolated_per_patient(tmp_path: Path):
    store = MemoryStore(db_path=tmp_path / "mem.db")
    store.store_card(_card(request_id="r1", context_fingerprint="patient-A"))
    store.store_card(_card(request_id="r2", context_fingerprint="patient-B"))

    a = store.get_latest_for_patient("patient-A")
    b = store.get_latest_for_patient("patient-B")
    assert a.audit.request_id == "r1"
    assert b.audit.request_id == "r2"


# ─────────────────────── eviction ───────────────────────

def test_evict_expired_removes_only_expired(tmp_path: Path):
    store = MemoryStore(db_path=tmp_path / "mem.db")
    # One expired + one fresh
    store.store_card(_card(request_id="old", context_fingerprint="A",
                           ts=_now() - timedelta(hours=10), valid_for_min=60))
    store.store_card(_card(request_id="fresh", context_fingerprint="B",
                           valid_for_min=720))
    n_removed = store.evict_expired()
    assert n_removed == 1
    assert store.count() == 1


# ─────────────────────── drift detection ───────────────────────

def test_detect_drift_no_change_returns_none():
    a = _card(request_id="r1", context_fingerprint="patient-A", prob=0.20)
    b = _card(request_id="r2", context_fingerprint="patient-A", prob=0.21)
    assert detect_recommendation_drift(a, b) is None


def test_detect_drift_action_flipped():
    cur = _card(request_id="cur", context_fingerprint="patient-A",
                action=Action.SNF, prob=0.30)
    prior = _card(request_id="prior", context_fingerprint="patient-A",
                  action=Action.HOME_WITH_CARE, prob=0.20)
    drift = detect_recommendation_drift(cur, prior)
    assert drift is not None
    assert any("action" in s for s in drift["drift_signals"])


def test_detect_drift_prob_jumped_significantly():
    cur = _card(request_id="cur", context_fingerprint="patient-A", prob=0.40)
    prior = _card(request_id="prior", context_fingerprint="patient-A", prob=0.15)
    drift = detect_recommendation_drift(cur, prior)
    assert drift is not None
    assert any("prob_mean" in s for s in drift["drift_signals"])


def test_detect_drift_abstain_status_changed():
    abstain = AbstainTrigger(type="evidence_insufficient", detail="x")
    cur = _card(request_id="cur", context_fingerprint="patient-A", abstain=[abstain])
    cur = cur.model_copy(update={"recommendation": None, "reasoning": cur.reasoning})
    prior = _card(request_id="prior", context_fingerprint="patient-A")  # has recommendation
    drift = detect_recommendation_drift(cur, prior)
    assert drift is not None
    # action change OR abstain change should be flagged
    assert any("abstain" in s or "action" in s for s in drift["drift_signals"])


# ─────────────────────── env-var DB path ───────────────────────

def test_default_db_path_uses_env_var(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "subdir" / "memory.db"
    monkeypatch.setenv("TRUSTEDRISK_MEMORY_DB_PATH", str(db_path))
    store = MemoryStore()
    assert store.db_path == db_path
    assert db_path.parent.exists()
