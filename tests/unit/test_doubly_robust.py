"""Phase 17.AU - Doubly-robust estimators tests."""

from __future__ import annotations

import math
import random

import pytest

from a2a_agent.doubly_robust import (
    DoublyRobustReport, estimate_ate_doubly_robust,
)


def _synthetic(n: int = 1000, seed: int = 7,
               true_ate: float = 0.5):
    rng = random.Random(seed)
    features = []
    treatments = []
    outcomes = []
    for _ in range(n):
        x = rng.gauss(0, 1)
        # Propensity depends on x
        true_p = 1 / (1 + math.exp(-0.5 * x))
        t = 1 if rng.random() < true_p else 0
        # True outcome model
        y_base = 0.3 * x + rng.gauss(0, 0.5)
        y = y_base + true_ate * t
        features.append({"x": x})
        treatments.append(t)
        outcomes.append(y)
    return features, treatments, outcomes, true_ate


def _correct_propensity(features):
    return 1 / (1 + math.exp(-0.5 * features["x"]))


def _correct_outcome_treated(features):
    return 0.3 * features["x"] + 0.5


def _correct_outcome_control(features):
    return 0.3 * features["x"]


def _wrong_propensity(features):
    return 0.5  # constant - misspecified


def _wrong_outcome_treated(features):
    return 0.0  # misspecified


def _wrong_outcome_control(features):
    return 0.0


# ─────────────────────────────────────────────────────────────────────
# Estimator behaviour
# ─────────────────────────────────────────────────────────────────────

def test_aipw_recovers_true_ate_with_correct_models():
    f, t, y, true_ate = _synthetic(n=2000, seed=1)
    rep = estimate_ate_doubly_robust(
        features=f, treatments=t, outcomes=y,
        propensity_fn=_correct_propensity,
        outcome_fn_treated=_correct_outcome_treated,
        outcome_fn_control=_correct_outcome_control,
    )
    assert abs(rep.ate_aipw - true_ate) < 0.1


def test_aipw_robust_to_wrong_outcome_when_propensity_correct():
    """Double-robust property: AIPW should still recover the true ATE
    when the outcome model is wrong but the propensity is correct."""
    f, t, y, true_ate = _synthetic(n=2000, seed=2)
    rep = estimate_ate_doubly_robust(
        features=f, treatments=t, outcomes=y,
        propensity_fn=_correct_propensity,
        outcome_fn_treated=_wrong_outcome_treated,
        outcome_fn_control=_wrong_outcome_control,
    )
    assert abs(rep.ate_aipw - true_ate) < 0.15


def test_aipw_robust_to_wrong_propensity_when_outcome_correct():
    """Symmetric: AIPW should still recover when propensity is wrong
    but outcome model is correct."""
    f, t, y, true_ate = _synthetic(n=2000, seed=3)
    rep = estimate_ate_doubly_robust(
        features=f, treatments=t, outcomes=y,
        propensity_fn=_wrong_propensity,
        outcome_fn_treated=_correct_outcome_treated,
        outcome_fn_control=_correct_outcome_control,
    )
    assert abs(rep.ate_aipw - true_ate) < 0.1


def test_g_formula_recovers_true_ate_with_correct_outcome():
    f, t, y, true_ate = _synthetic(n=2000, seed=4)
    rep = estimate_ate_doubly_robust(
        features=f, treatments=t, outcomes=y,
        propensity_fn=_wrong_propensity,
        outcome_fn_treated=_correct_outcome_treated,
        outcome_fn_control=_correct_outcome_control,
    )
    assert abs(rep.ate_g_formula - true_ate) < 0.05


def test_aipw_se_non_negative_and_finite():
    f, t, y, _ = _synthetic(n=500, seed=5)
    rep = estimate_ate_doubly_robust(
        features=f, treatments=t, outcomes=y,
        propensity_fn=_correct_propensity,
        outcome_fn_treated=_correct_outcome_treated,
        outcome_fn_control=_correct_outcome_control,
    )
    assert rep.se_aipw >= 0.0
    assert rep.se_aipw < 10.0


def test_estimator_rejects_misaligned():
    with pytest.raises(ValueError):
        estimate_ate_doubly_robust(
            features=[{"x": 1.0}], treatments=[0, 1], outcomes=[1.0],
            propensity_fn=_correct_propensity,
            outcome_fn_treated=_correct_outcome_treated,
            outcome_fn_control=_correct_outcome_control,
        )


def test_estimator_rejects_invalid_treatment():
    with pytest.raises(ValueError):
        estimate_ate_doubly_robust(
            features=[{"x": 1.0}],
            treatments=[2], outcomes=[1.0],
            propensity_fn=_correct_propensity,
            outcome_fn_treated=_correct_outcome_treated,
            outcome_fn_control=_correct_outcome_control,
        )


def test_estimator_rejects_empty():
    with pytest.raises(ValueError):
        estimate_ate_doubly_robust(
            features=[], treatments=[], outcomes=[],
            propensity_fn=_correct_propensity,
            outcome_fn_treated=_correct_outcome_treated,
            outcome_fn_control=_correct_outcome_control,
        )


def test_estimator_clips_propensity():
    """When the propensity function returns extreme values, the
    clipped min/max stays within [0.01, 0.99] by default."""
    f, t, y, _ = _synthetic(n=200, seed=6)
    rep = estimate_ate_doubly_robust(
        features=f, treatments=t, outcomes=y,
        propensity_fn=lambda x: 0.999,   # extreme
        outcome_fn_treated=_correct_outcome_treated,
        outcome_fn_control=_correct_outcome_control,
    )
    assert rep.propensity_min >= 0.01
    assert rep.propensity_max <= 0.99


def test_round_trip_through_pydantic():
    f, t, y, _ = _synthetic(n=200, seed=8)
    rep = estimate_ate_doubly_robust(
        features=f, treatments=t, outcomes=y,
        propensity_fn=_correct_propensity,
        outcome_fn_treated=_correct_outcome_treated,
        outcome_fn_control=_correct_outcome_control,
    )
    payload = rep.model_dump(mode="json")
    rebuilt = DoublyRobustReport.model_validate(payload)
    assert rebuilt.n == rep.n
