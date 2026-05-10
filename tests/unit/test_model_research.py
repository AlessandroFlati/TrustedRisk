"""Phase 14.7/14.8/14.9 -- model_research bundle tests."""

from __future__ import annotations

import asyncio
import math
import random

import pytest

from mcp_server.tools.model_research import (
    compute_cox_proportional_hazards,
    compute_ensemble_stacking, compute_shap_attribution,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# Cox PH
# ─────────────────────────────────────────────────────────────────────

def _synth_cox_data(n=200, true_beta=0.5, seed=42):
    rng = random.Random(seed)
    durations: list[float] = []
    events: list[int] = []
    covariates: list[list[float]] = []
    for _ in range(n):
        x = rng.gauss(0, 1)
        # Hazard λ(t) = exp(true_beta * x); duration ~ Exp(λ)
        lam = math.exp(true_beta * x)
        t = rng.expovariate(lam)
        # 70 % observed, 30 % censored at t = 5
        censored_at = 5.0
        if t > censored_at:
            durations.append(censored_at)
            events.append(0)
        else:
            durations.append(t)
            events.append(1)
        covariates.append([x])
    return durations, events, covariates


def test_cox_recovers_positive_beta():
    d, e, x = _synth_cox_data(n=300, true_beta=0.5, seed=1)
    out = _run(compute_cox_proportional_hazards(
        durations=d, events=e, covariates=x,
        feature_names=["risk_factor"],
    ))
    beta = out.coefficients["risk_factor"]
    # Should recover a positive coefficient ≈ 0.5 within ± 0.25
    assert beta > 0.2
    assert abs(beta - 0.5) < 0.4


def test_cox_concordance_above_random():
    d, e, x = _synth_cox_data(n=200, true_beta=0.7, seed=2)
    out = _run(compute_cox_proportional_hazards(
        durations=d, events=e, covariates=x,
    ))
    assert out.concordance_index > 0.55


def test_cox_too_few_observations_raises():
    with pytest.raises(ValueError):
        _run(compute_cox_proportional_hazards(
            durations=[1.0, 2.0], events=[1, 0],
            covariates=[[0.1], [0.2]],
        ))


def test_cox_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        _run(compute_cox_proportional_hazards(
            durations=[1, 2, 3, 4, 5, 6],
            events=[1, 0, 1, 0, 1],   # one short
            covariates=[[0.1]] * 6,
        ))


# ─────────────────────────────────────────────────────────────────────
# SHAP attribution
# ─────────────────────────────────────────────────────────────────────

def test_shap_sum_plus_base_equals_prediction():
    out = _run(compute_shap_attribution(
        feature_values={"L": 4, "A": 3, "C": 3, "E": 1},
        coefficients={"L": 0.10, "A": 0.05, "C": 0.20, "E": 0.04},
        intercept=0.5,
    ))
    assert out.actual_prediction == pytest.approx(
        out.sum_shap_plus_base, abs=1e-6,
    )


def test_shap_zero_value_yields_zero_attribution():
    out = _run(compute_shap_attribution(
        feature_values={"L": 0, "A": 0, "C": 0, "E": 0},
        coefficients={"L": 0.1, "A": 0.1, "C": 0.1, "E": 0.1},
        intercept=0.2,
    ))
    for v in out.shap_values.values():
        assert v == 0.0
    assert out.actual_prediction == 0.2


def test_shap_keys_must_match_coefficients():
    with pytest.raises(ValueError):
        _run(compute_shap_attribution(
            feature_values={"L": 1, "A": 2},
            coefficients={"L": 0.1, "C": 0.3},
        ))


def test_shap_empty_features_rejected():
    with pytest.raises(ValueError):
        _run(compute_shap_attribution(
            feature_values={}, coefficients={},
        ))


# ─────────────────────────────────────────────────────────────────────
# Ensemble stacking
# ─────────────────────────────────────────────────────────────────────

def test_ensemble_stacking_runs_on_50_obs_3_models():
    rng = random.Random(123)
    n = 50
    labels = [int(rng.random() < 0.4) for _ in range(n)]
    base = {
        "lookup": [
            (0.4 + 0.3 if y else 0.4 - 0.3) + rng.gauss(0, 0.1)
            for y in labels
        ],
        "logistic": [
            (0.5 + 0.2 if y else 0.5 - 0.2) + rng.gauss(0, 0.1)
            for y in labels
        ],
        "tree": [
            (0.3 + 0.4 if y else 0.3 - 0.2) + rng.gauss(0, 0.15)
            for y in labels
        ],
    }
    out = _run(compute_ensemble_stacking(
        base_predictions=base, labels=labels,
    ))
    assert 0.0 <= out.stacked_auroc <= 1.0
    assert 0.0 <= sum(out.meta_learner_weights.values()) <= 1.0001


def test_ensemble_stacking_reports_per_base_aurocs():
    rng = random.Random(456)
    n = 30
    labels = [i % 2 for i in range(n)]
    base = {
        "good": [0.7 if y else 0.3 for y in labels],
        "noisy": [0.5 + rng.gauss(0, 0.1) for _ in range(n)],
    }
    out = _run(compute_ensemble_stacking(
        base_predictions=base, labels=labels,
    ))
    # The "good" base should have AUROC > "noisy"
    assert out.base_model_aurocs["good"] >= out.base_model_aurocs["noisy"]


def test_ensemble_too_few_observations_rejected():
    with pytest.raises(ValueError):
        _run(compute_ensemble_stacking(
            base_predictions={"a": [0.5]}, labels=[1],
        ))


def test_ensemble_misaligned_predictions_rejected():
    with pytest.raises(ValueError):
        _run(compute_ensemble_stacking(
            base_predictions={
                "a": [0.5] * 25, "b": [0.5] * 24,    # mismatch
            },
            labels=[i % 2 for i in range(25)],
        ))


# ─────────────────────────────────────────────────────────────────────
# Bundle wiring
# ─────────────────────────────────────────────────────────────────────

def test_model_research_bundle_registered():
    from mcp_server.tools import BUNDLES
    assert "model_research" in BUNDLES
    assert set(BUNDLES["model_research"]) == {
        "compute_cox_proportional_hazards",
        "compute_shap_attribution",
        "compute_ensemble_stacking",
    }
