"""Unit tests for compute_decision_utility -- utility analysis + dominance."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from mcp_server.tools.decision_utility import (
    _build_reasoning_trace,
    _stochastic_dominance,
    compute_decision_utility,
)
from shared.schemas import Action


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── _stochastic_dominance ───────────────────────

def test_stochastic_dominance_empty():
    action, conf = _stochastic_dominance({})
    assert action == "INCONCLUSIVE"
    assert conf == 0.0


def test_stochastic_dominance_single_action():
    samples = {Action.HOME_WITH_CARE: np.array([1.0, 1.5, 2.0])}
    action, conf = _stochastic_dominance(samples)
    assert action == Action.HOME_WITH_CARE
    assert conf == 1.0


def test_stochastic_dominance_clear_winner():
    rng = np.random.default_rng(42)
    samples = {
        Action.HOME_WITH_CARE: rng.normal(5.0, 0.5, 1000),  # higher
        Action.SNF: rng.normal(2.0, 0.5, 1000),
    }
    action, conf = _stochastic_dominance(samples)
    assert action == Action.HOME_WITH_CARE
    assert conf > 0.9


def test_stochastic_dominance_inconclusive():
    rng = np.random.default_rng(42)
    samples = {
        Action.HOME_WITH_CARE: rng.normal(3.0, 1.0, 1000),
        Action.SNF: rng.normal(3.0, 1.0, 1000),  # nearly identical
    }
    action, conf = _stochastic_dominance(samples)
    # Likely INCONCLUSIVE due to overlap
    assert conf < 0.6


# ─────────────────────── _build_reasoning_trace ───────────────────────

def test_reasoning_trace_includes_dominant():
    trace = _build_reasoning_trace(
        scores={Action.HOME_WITH_CARE: 4.5, Action.SNF: 3.2},
        costs={Action.HOME_WITH_CARE: 1500.0, Action.SNF: 18000.0},
        dominant=Action.HOME_WITH_CARE,
        confidence=0.85,
    )
    assert "home_with_care" in trace.lower() or "HOME_WITH_CARE" in trace
    assert "QALY" in trace


def test_reasoning_trace_inconclusive():
    trace = _build_reasoning_trace(
        scores={Action.HOME_WITH_CARE: 3.0, Action.SNF: 3.1},
        costs={Action.HOME_WITH_CARE: 1500.0, Action.SNF: 18000.0},
        dominant="INCONCLUSIVE",
        confidence=0.45,
    )
    assert "no stochastic dominance" in trace.lower() or "INCONCLUSIVE" in trace.upper()


# ─────────────────────── End-to-end with patched utility_fn ───────────────────────

def _patch_utility_fn(monkeypatch):
    """Inject a minimal utility_fn that the test can rely on."""
    from mcp_server.tools import decision_utility as du
    fake_utility_fn = {
        "qaly_weights": {
            "discharge_home_well": 1.0,
            "discharge_home_readmit": -2.0,
            "home_with_care_well": 0.95,
            "home_with_care_readmit": -1.8,
            "snf_well": 0.6,
            "snf_readmit": -0.8,
            "continued_admission_well": 0.3,
            "continued_admission_readmit": -0.4,
        },
        "cost_usd_per_day": {
            "discharge_home": 50.0,
            "home_with_care": 200.0,
            "snf": 800.0,
            "continued_admission": 3000.0,
        },
        "time_horizon_days": 90,
    }
    monkeypatch.setattr(du, "_UTILITY_FN", None)  # bust cache
    monkeypatch.setattr(du, "_load_utility_fn", lambda: fake_utility_fn)


def test_compute_decision_utility_returns_analysis(monkeypatch):
    _patch_utility_fn(monkeypatch)

    outcome_probs = {
        "home_with_care": {
            "well": {"mean": 0.78, "ci95": [0.72, 0.84]},
            "readmit": {"mean": 0.22, "ci95": [0.16, 0.28]},
        },
        "snf": {
            "well": {"mean": 0.85, "ci95": [0.80, 0.90]},
            "readmit": {"mean": 0.15, "ci95": [0.10, 0.20]},
        },
    }

    result = _run(compute_decision_utility(outcome_probs=outcome_probs, n_monte_carlo=500))
    assert Action.HOME_WITH_CARE in result.action_scores_qaly_weeks
    assert Action.SNF in result.action_scores_qaly_weeks
    assert result.dominance_confidence >= 0.0
    assert result.dominance_confidence <= 1.0
    assert isinstance(result.reasoning_trace, str)
    assert len(result.reasoning_trace) > 10


def test_compute_decision_utility_skips_unknown_action(monkeypatch):
    _patch_utility_fn(monkeypatch)

    outcome_probs = {
        "home_with_care": {"well": {"mean": 0.8, "ci95": [0.7, 0.9]}},
        "FAKE_ACTION": {"well": {"mean": 0.5, "ci95": [0.4, 0.6]}},
    }
    result = _run(compute_decision_utility(outcome_probs=outcome_probs, n_monte_carlo=200))
    assert Action.HOME_WITH_CARE in result.action_scores_qaly_weeks
    # Unknown action filtered
    assert "FAKE_ACTION" not in {a.value if hasattr(a, "value") else str(a)
                                  for a in result.action_scores_qaly_weeks}
