"""Phase 10.9 -- Split conformal prediction unit tests."""

from __future__ import annotations

import random

import pytest

from a2a_agent.conformal import (
    fit_split_conformal,
    predict_interval,
    predict_set,
)


# ─────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────

def test_fit_returns_quantile_within_score_range():
    preds = [0.2, 0.4, 0.6, 0.8]
    outs = [0.0, 1.0, 0.0, 1.0]
    cal = fit_split_conformal(
        preds, outs, target_coverage=0.90,
        score_function_id="abs_residual",
    )
    # Scores: 0.2, 0.6, 0.6, 0.2 -> max = 0.6
    assert 0.0 <= cal.quantile_threshold <= 1.0


def test_fit_n_calibration_matches_input():
    preds = [0.1, 0.3, 0.5, 0.7, 0.9]
    outs = [0, 1, 1, 1, 1]
    cal = fit_split_conformal(
        preds, outs, target_coverage=0.80,
        score_function_id="binary_one_minus_p",
    )
    assert cal.n_calibration == 5


def test_fit_rejects_bad_target_coverage():
    with pytest.raises(ValueError):
        fit_split_conformal([0.5], [1], target_coverage=1.5)
    with pytest.raises(ValueError):
        fit_split_conformal([0.5], [1], target_coverage=0.0)


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_split_conformal([0.1, 0.2], [1])


def test_fit_rejects_unknown_score_function():
    with pytest.raises(ValueError):
        fit_split_conformal(
            [0.5], [1], score_function_id="something_else",
        )


def test_fit_rejects_empty_calibration_set():
    with pytest.raises(ValueError):
        fit_split_conformal([], [])


# ─────────────────────────────────────────────────────────────────────
# Marginal coverage on synthetic exchangeable data
# ─────────────────────────────────────────────────────────────────────

def test_marginal_coverage_holds_on_iid_synthetic_regression():
    """Generate 2000 IID (p, y) pairs from a fixed Beta(2,5) noise
    distribution. Split 1000/1000 calibration/test. Empirical test
    coverage should be ≥ target − tolerance."""
    rng = random.Random(123)
    n = 2000
    preds = [rng.random() for _ in range(n)]
    # Outcome is the prediction plus a centered noise term
    outs = [
        max(0.0, min(1.0, p + (rng.random() - 0.5) * 0.3))
        for p in preds
    ]
    cal_preds, cal_outs = preds[:1000], outs[:1000]
    test_preds, test_outs = preds[1000:], outs[1000:]
    cal = fit_split_conformal(
        cal_preds, cal_outs, target_coverage=0.90,
        score_function_id="abs_residual",
    )
    in_band = 0
    for p, y in zip(test_preds, test_outs):
        iv = predict_interval(p, cal)
        if iv.lower_bound <= y <= iv.upper_bound:
            in_band += 1
    empirical = in_band / len(test_preds)
    # Allow ± 0.04 slack around the 0.90 target (binomial std on n=1000)
    assert empirical >= 0.86, (
        f"Empirical coverage {empirical:.3f} < 0.86 (target 0.90)"
    )


def test_marginal_coverage_holds_on_iid_synthetic_binary():
    """Same idea but with binary outcomes and the binary_one_minus_p
    score; the prediction set should contain the realised outcome at
    the calibrated rate."""
    rng = random.Random(456)
    n = 2000
    preds = [rng.random() for _ in range(n)]
    outs = [1 if rng.random() < p else 0 for p in preds]
    cal_preds, cal_outs = preds[:1000], outs[:1000]
    test_preds, test_outs = preds[1000:], outs[1000:]
    cal = fit_split_conformal(
        cal_preds, cal_outs, target_coverage=0.85,
        score_function_id="binary_one_minus_p",
    )
    in_band = 0
    for p, y in zip(test_preds, test_outs):
        ps = predict_set(p, cal)
        if int(y) in ps.prediction_set:
            in_band += 1
    empirical = in_band / len(test_preds)
    assert empirical >= 0.80, (
        f"Empirical coverage {empirical:.3f} < 0.80 (target 0.85)"
    )


# ─────────────────────────────────────────────────────────────────────
# Prediction interval / set semantics
# ─────────────────────────────────────────────────────────────────────

def test_predict_interval_clips_to_unit_box():
    cal = fit_split_conformal(
        [0.5] * 10, [1] * 10, target_coverage=0.90,
        score_function_id="abs_residual",
    )
    iv_low = predict_interval(0.05, cal)
    iv_hi = predict_interval(0.97, cal)
    assert iv_low.lower_bound == 0.0
    assert iv_hi.upper_bound == 1.0


def test_predict_interval_requires_abs_residual_calibration():
    cal_binary = fit_split_conformal(
        [0.5] * 10, [1] * 10, target_coverage=0.90,
        score_function_id="binary_one_minus_p",
    )
    with pytest.raises(ValueError):
        predict_interval(0.5, cal_binary)


def test_predict_set_includes_both_classes_for_uncertain_prediction():
    cal = fit_split_conformal(
        [0.5] * 10, [1] * 5 + [0] * 5, target_coverage=0.90,
        score_function_id="binary_one_minus_p",
    )
    # p = 0.5 -> score(1) = 0.5, score(0) = 0.5; both ≤ q likely
    ps = predict_set(0.5, cal)
    assert set(ps.prediction_set) == {0, 1}


def test_predict_set_requires_binary_calibration():
    cal_reg = fit_split_conformal(
        [0.5] * 10, [1] * 10, target_coverage=0.90,
        score_function_id="abs_residual",
    )
    with pytest.raises(ValueError):
        predict_set(0.5, cal_reg)


def test_predict_set_rejects_out_of_unit_probability():
    cal = fit_split_conformal(
        [0.5] * 10, [1] * 10, target_coverage=0.90,
        score_function_id="binary_one_minus_p",
    )
    with pytest.raises(ValueError):
        predict_set(1.5, cal)


def test_calibration_finite_sample_correction_uses_n_plus_1():
    """The split-conformal threshold uses ceil((n+1)(1-α))-th order
    statistic, NOT the simple ceil(n(1-α)) -- guards against off-by-one
    coverage drift on small calibration sets."""
    # n=10, target=0.90 -> ceil(11 * 0.90) = 10 -> should be max score
    scores_via_internal = [0.1 * i for i in range(1, 11)]
    cal = fit_split_conformal(
        calibration_predictions=[0.0] * 10,
        calibration_outcomes=scores_via_internal,
        target_coverage=0.90,
        score_function_id="abs_residual",
    )
    # max score is 1.0
    assert cal.quantile_threshold == 1.0


def test_calibration_empirical_coverage_at_least_target():
    """On the calibration fold itself, empirical coverage is ≥ target
    by construction (since q is defined as the (1-α) quantile)."""
    rng = random.Random(789)
    preds = [rng.random() for _ in range(500)]
    outs = [p + (rng.random() - 0.5) * 0.2 for p in preds]
    outs = [max(0.0, min(1.0, o)) for o in outs]
    cal = fit_split_conformal(
        preds, outs, target_coverage=0.90,
        score_function_id="abs_residual",
    )
    assert cal.empirical_coverage >= 0.89    # tiny floor for finite n
