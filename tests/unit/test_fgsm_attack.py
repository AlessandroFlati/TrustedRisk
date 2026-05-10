"""Phase 17.AK - FGSM-style adversarial attack tests."""

from __future__ import annotations

import math

import pytest

from a2a_agent.fgsm_attack import (
    FGSMReport, RobustnessReport,
    _numeric_gradient,
    adversarial_robustness_sweep, fgsm_attack,
)


def _logistic(features: dict[str, float]) -> float:
    z = -3.0 + 0.5 * features.get("L", 0.0) \
        + 0.3 * features.get("A", 0.0) \
        + 0.2 * features.get("C", 0.0) \
        + 0.1 * features.get("E", 0.0)
    return 1.0 / (1.0 + math.exp(-z))


def _bucket_map(features: dict[str, float]) -> float:
    """Mimics the W1 bucket map - non-differentiable but locally
    flat between bucket boundaries."""
    lace = sum(features.get(k, 0.0) for k in ("L", "A", "C", "E"))
    if lace <= 2:
        return 0.072
    if lace <= 5:
        return 0.103
    if lace <= 9:
        return 0.158
    if lace <= 12:
        return 0.234
    return 0.327


# ─────────────────────────────────────────────────────────────────────
# Numeric gradient
# ─────────────────────────────────────────────────────────────────────

def test_numeric_gradient_recovers_signs():
    g = _numeric_gradient(
        {"L": 1.0, "A": 1.0, "C": 1.0, "E": 1.0},
        _logistic,
    )
    # All true coefficients positive -> gradients positive
    for v in g.values():
        assert v >= -1e-6


def test_numeric_gradient_largest_for_dominant_feature():
    g = _numeric_gradient(
        {"L": 1.0, "A": 1.0, "C": 1.0, "E": 1.0},
        _logistic,
    )
    assert g["L"] == max(g.values())


# ─────────────────────────────────────────────────────────────────────
# FGSM
# ─────────────────────────────────────────────────────────────────────

def test_fgsm_flips_low_risk_instance_with_enough_perturbation():
    """Smooth logistic predictor flips with enough perturbation."""
    rep = fgsm_attack(
        features={"L": 0.0, "A": 0.0, "C": 0.0, "E": 0.0},
        predictor=_logistic,
        decision_threshold=0.20,
        epsilon_max=10.0, epsilon_step=1.0,
        feature_max={"L": 7.0, "A": 3.0, "C": 5.0, "E": 4.0},
    )
    assert rep.flipped is True
    assert rep.epsilon_min_to_flip is not None
    assert rep.adversarial_class != rep.original_class


def test_fgsm_no_flip_when_epsilon_too_small():
    rep = fgsm_attack(
        features={"L": 0.0, "A": 0.0, "C": 0.0, "E": 0.0},
        predictor=_logistic,
        decision_threshold=0.20,
        epsilon_max=0.5, epsilon_step=0.5,
        feature_max={"L": 7.0},
    )
    assert rep.flipped is False
    assert rep.epsilon_min_to_flip is None


def test_fgsm_targeted_decrease_finds_lower_score():
    """Start from a high-score instance and push it lower."""
    high = {"L": 7.0, "A": 3.0, "C": 5.0, "E": 4.0}
    rep = fgsm_attack(
        features=high, predictor=_logistic,
        decision_threshold=0.20,
        epsilon_max=10.0, epsilon_step=1.0,
        feature_min={"L": 0.0, "A": 0.0, "C": 0.0, "E": 0.0},
        targeted_increase=False,
    )
    assert rep.adversarial_score <= rep.original_score


def test_fgsm_respects_feature_min_max_clipping():
    rep = fgsm_attack(
        features={"L": 0.0}, predictor=_logistic,
        decision_threshold=0.20,
        epsilon_max=20.0, epsilon_step=2.0,
        feature_max={"L": 5.0},
    )
    # The adversarial L must not exceed the cap
    assert rep.adversarial_features["L"] <= 5.0 + 1e-6


def test_fgsm_rejects_empty_features():
    with pytest.raises(ValueError):
        fgsm_attack(features={}, predictor=_logistic)


def test_fgsm_rejects_non_positive_epsilon():
    with pytest.raises(ValueError):
        fgsm_attack(
            features={"L": 1.0}, predictor=_logistic,
            epsilon_max=0.0,
        )


def test_fgsm_round_trip_through_pydantic():
    rep = fgsm_attack(
        features={"L": 1.0, "A": 0.0, "C": 0.0, "E": 0.0},
        predictor=_logistic,
    )
    payload = rep.model_dump(mode="json")
    rebuilt = FGSMReport.model_validate(payload)
    assert rebuilt.flipped == rep.flipped


def test_fgsm_on_bucket_map_constant_locally_no_flip_at_zero():
    """Demonstrates a real-world robustness property: the W1 bucket
    map is locally constant inside [0, 2] so FGSM with h=1.0 cannot
    detect a useful gradient at L=A=C=E=0 - the model is robust to
    sub-bucket-width perturbations by construction."""
    rep = fgsm_attack(
        features={"L": 0.0, "A": 0.0, "C": 0.0, "E": 0.0},
        predictor=_bucket_map,
        decision_threshold=0.20, epsilon_max=2.0, epsilon_step=0.5,
        feature_max={"L": 7.0, "A": 3.0, "C": 5.0, "E": 4.0},
    )
    # Bucket map flat in [0,2] -> gradient is zero -> no flip
    assert rep.flipped is False


# ─────────────────────────────────────────────────────────────────────
# Robustness sweep
# ─────────────────────────────────────────────────────────────────────

def test_robustness_sweep_reports_three_thresholds():
    instances = [
        {"L": 1.0, "A": 0.0, "C": 0.0, "E": 0.0},
        {"L": 4.0, "A": 1.0, "C": 1.0, "E": 0.0},
        {"L": 6.0, "A": 2.0, "C": 1.0, "E": 1.0},
        {"L": 5.0, "A": 1.0, "C": 1.0, "E": 1.0},
    ]
    rep = adversarial_robustness_sweep(
        instances=instances, predictor=_logistic,
        decision_threshold=0.20,
        epsilon_max=10.0, epsilon_step=1.0,
        feature_max={"L": 7.0, "A": 3.0, "C": 5.0, "E": 4.0},
        feature_min={"L": 0.0, "A": 0.0, "C": 0.0, "E": 0.0},
    )
    assert rep.n_instances == 4
    assert 0.0 <= rep.pct_robust_at_eps_1 <= 1.0
    # Robust-set monotonicity: instances robust to a larger
    # perturbation are a subset of those robust to a smaller one.
    assert rep.pct_robust_at_eps_3 <= rep.pct_robust_at_eps_2 + 1e-9
    assert rep.pct_robust_at_eps_2 <= rep.pct_robust_at_eps_1 + 1e-9


def test_robustness_sweep_round_trip_through_pydantic():
    rep = adversarial_robustness_sweep(
        instances=[{"L": 1.0, "A": 0.0, "C": 0.0, "E": 0.0}],
        predictor=_logistic,
        epsilon_max=2.0, epsilon_step=1.0,
    )
    payload = rep.model_dump(mode="json")
    rebuilt = RobustnessReport.model_validate(payload)
    assert rebuilt.n_instances == rep.n_instances


def test_robustness_sweep_rejects_empty():
    with pytest.raises(ValueError):
        adversarial_robustness_sweep(
            instances=[], predictor=_logistic,
        )
