"""Unit tests for shared.abstain -- 3-trigger abstain composition."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from shared.abstain import (
    check_ci_width,
    check_evidence,
    check_ood,
    compose_abstain_triggers,
)
from shared.schemas import (
    ClaimGrounding,
    EvidenceSource,
    Factor,
    RiskEstimate,
    SubClaim,
)


def _now():
    return datetime.now(timezone.utc)


def _risk(*, ci_width: float = 0.10) -> RiskEstimate:
    half = ci_width / 2
    return RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="test",
        horizon_days=30,
        lace_raw_score=10,
        probability_mean=0.18,
        probability_ci95=(max(0.0, 0.18 - half), min(1.0, 0.18 + half)),
        probability_ci_width=ci_width,
        contributing_factors=[Factor(name="L", raw_value=4, lace_points=4, weight=0.4)],
        computed_at=_now(),
    )


def _grounding(verdict: str = "supported", sub_claims: list | None = None) -> ClaimGrounding:
    return ClaimGrounding(
        claim_text="discharge plan ok",
        sub_claims=sub_claims or [],
        overall_verdict=verdict,
        context_fingerprint="fp",
        grounded_at=_now(),
    )


# ─────────────────────── Trigger 1 -- CI width ───────────────────────

def test_ci_width_below_threshold_no_trigger():
    assert check_ci_width(_risk(ci_width=0.10), threshold=0.30) is None


def test_ci_width_above_threshold_fires():
    t = check_ci_width(_risk(ci_width=0.40), threshold=0.30)
    assert t is not None
    assert t.type == "confidence_interval_too_wide"
    assert t.threshold_exceeded["ci_width"] == 0.40
    assert t.threshold_exceeded["threshold"] == 0.30


def test_ci_width_at_threshold_no_trigger():
    # Strictly greater-than required
    assert check_ci_width(_risk(ci_width=0.30), threshold=0.30) is None


def test_ci_width_uses_env_default(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_CI_WIDTH_THRESHOLD", "0.05")
    t = check_ci_width(_risk(ci_width=0.10))
    assert t is not None
    assert t.threshold_exceeded["threshold"] == 0.05


# ─────────────────────── Trigger 2 -- evidence insufficient ───────────────────────

def test_evidence_supported_no_trigger():
    assert check_evidence(_grounding(verdict="supported")) is None


def test_evidence_overall_unsupported_fires():
    t = check_evidence(_grounding(verdict="unsupported"))
    assert t is not None
    assert t.type == "evidence_insufficient"
    assert "overall_verdict" in t.detail


def test_evidence_critical_subclaim_unsupported_fires():
    sc = SubClaim(
        text="Patient on warfarin", atomic_kind="medication",
        verdict="unsupported", confidence=0.4,
        reason_if_unsupported="No INR observation in last 48h",
    )
    t = check_evidence(_grounding(verdict="partially_supported", sub_claims=[sc]))
    assert t is not None
    assert t.type == "evidence_insufficient"
    assert "warfarin" in t.detail or "INR" in t.detail


def test_evidence_non_critical_subclaim_unsupported_no_trigger():
    sc = SubClaim(
        text="Patient lives alone", atomic_kind="general",
        verdict="unsupported", confidence=0.4,
    )
    assert check_evidence(_grounding(verdict="partially_supported", sub_claims=[sc])) is None


def test_evidence_partially_supported_no_critical_no_trigger():
    sc = SubClaim(
        text="Patient improved", atomic_kind="general",
        verdict="partially_supported", confidence=0.6,
    )
    assert check_evidence(_grounding(verdict="partially_supported", sub_claims=[sc])) is None


# ─────────────────────── Trigger 3 -- OOD ───────────────────────

@pytest.fixture
def lace_percentile_policy(tmp_path: Path, monkeypatch):
    """Stage a lace_percentile policy file and point env vars at it."""
    policy = {
        "policy_type": "lace_percentile_fallback",
        "threshold_lace": 13,
        "method": {"cohort_percentile_cutoff": 90},
    }
    fp = tmp_path / "policy.json"
    fp.write_text(json.dumps(policy))
    monkeypatch.setenv("TRUSTEDRISK_OOD_POLICY_PATH", str(fp))
    monkeypatch.setenv("TRUSTEDRISK_OOD_POLICY_TYPE", "lace_percentile")
    # Bust the lru_cache by reaching through the wrapped function
    from shared import abstain
    abstain._load_ood_policy.cache_clear()
    return policy


def test_ood_lace_below_threshold_no_trigger(lace_percentile_policy):
    assert check_ood(lace_score=10) is None


def test_ood_lace_at_threshold_fires(lace_percentile_policy):
    t = check_ood(lace_score=13)
    assert t is not None
    assert t.type == "out_of_distribution"
    assert t.threshold_exceeded["lace_score"] == 13
    assert t.threshold_exceeded["threshold_lace"] == 13


def test_ood_lace_above_threshold_fires(lace_percentile_policy):
    t = check_ood(lace_score=18)
    assert t is not None
    assert t.threshold_exceeded["lace_score"] == 18


def test_ood_no_policy_path_no_trigger(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_OOD_POLICY_PATH", raising=False)
    from shared import abstain
    abstain._load_ood_policy.cache_clear()
    assert check_ood(lace_score=18) is None


def test_ood_topological_no_embedding_no_trigger(tmp_path: Path, monkeypatch):
    policy = {
        "policy_type": "topological",
        "ood_distance_threshold": 0.5,
    }
    fp = tmp_path / "policy.json"
    fp.write_text(json.dumps(policy))
    monkeypatch.setenv("TRUSTEDRISK_OOD_POLICY_PATH", str(fp))
    monkeypatch.setenv("TRUSTEDRISK_OOD_POLICY_TYPE", "topological")
    from shared import abstain
    abstain._load_ood_policy.cache_clear()
    # encounter_embedding=None -> topological check skipped
    assert check_ood(lace_score=10, encounter_embedding=None) is None


# ─────────────────────── compose_abstain_triggers ───────────────────────

def test_compose_no_triggers():
    triggers = compose_abstain_triggers(
        risk=_risk(ci_width=0.05),
        grounding=_grounding(verdict="supported"),
        lace_score=5,
    )
    assert triggers == []


def test_compose_ci_trigger_only(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_OOD_POLICY_PATH", raising=False)
    from shared import abstain
    abstain._load_ood_policy.cache_clear()
    triggers = compose_abstain_triggers(
        risk=_risk(ci_width=0.40),
        grounding=_grounding(verdict="supported"),
        lace_score=5,
    )
    assert len(triggers) == 1
    assert triggers[0].type == "confidence_interval_too_wide"


def test_compose_multiple_triggers(lace_percentile_policy):
    triggers = compose_abstain_triggers(
        risk=_risk(ci_width=0.40),  # CI trigger
        grounding=_grounding(verdict="unsupported"),  # evidence trigger
        lace_score=18,  # OOD trigger
    )
    types = {t.type for t in triggers}
    assert types == {
        "confidence_interval_too_wide",
        "evidence_insufficient",
        "out_of_distribution",
    }
