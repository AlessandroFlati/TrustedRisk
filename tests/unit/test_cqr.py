"""Phase 17.AY - Conformalised quantile regression tests."""

from __future__ import annotations

import random

import pytest

from a2a_agent.cqr import (
    CQRReport, LinearQuantileModel, fit_cqr,
    fit_linear_quantile, predict_quantile,
)


def _split_synth(
    n: int = 600, seed: int = 7,
):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)
        # Heteroscedastic: noise grows with |x1|
        sigma = 0.3 + 0.3 * abs(x1)
        target = 2.0 * x1 + 0.5 * x2 + rng.gauss(0, sigma)
        X.append([x1, x2])
        y.append(target)
    cut1 = int(0.5 * n)
    cut2 = int(0.75 * n)
    return (
        X[:cut1], y[:cut1],
        X[cut1:cut2], y[cut1:cut2],
        X[cut2:], y[cut2:],
    )


# ─────────────────────────────────────────────────────────────────────
# Linear quantile regression
# ─────────────────────────────────────────────────────────────────────

def test_linear_quantile_fits_finite_coefficients():
    X = [[1.0, 2.0], [2.0, 1.0], [-1.0, 0.5]]
    y = [1.0, 2.0, -1.0]
    m = fit_linear_quantile(
        X=X, y=y, quantile=0.5, n_iterations=200,
    )
    for c in m.coefficients:
        assert abs(c) < 100


def test_linear_quantile_lower_below_upper_on_average():
    rng = random.Random(11)
    X = [[rng.gauss(0, 1)] for _ in range(200)]
    y = [2 * x[0] + rng.gauss(0, 1) for x in X]
    lo = fit_linear_quantile(
        X=X, y=y, quantile=0.1, n_iterations=400,
    )
    hi = fit_linear_quantile(
        X=X, y=y, quantile=0.9, n_iterations=400, seed=18,
    )
    n_lo_lt_hi = sum(
        1 for x in X
        if predict_quantile(lo, x) < predict_quantile(hi, x)
    )
    assert n_lo_lt_hi >= 0.8 * len(X)


def test_linear_quantile_rejects_invalid_quantile():
    with pytest.raises(ValueError):
        fit_linear_quantile(
            X=[[1.0]], y=[1.0], quantile=0.0,
        )


def test_linear_quantile_rejects_misaligned():
    with pytest.raises(ValueError):
        fit_linear_quantile(
            X=[[1.0]], y=[1.0, 2.0], quantile=0.5,
        )


def test_linear_quantile_rejects_empty():
    with pytest.raises(ValueError):
        fit_linear_quantile(
            X=[], y=[], quantile=0.5,
        )


def test_linear_quantile_round_trip_through_pydantic():
    m = fit_linear_quantile(
        X=[[1.0]], y=[2.0], quantile=0.5, n_iterations=10,
    )
    payload = m.model_dump(mode="json")
    rebuilt = LinearQuantileModel.model_validate(payload)
    assert rebuilt.quantile == m.quantile


# ─────────────────────────────────────────────────────────────────────
# CQR end-to-end
# ─────────────────────────────────────────────────────────────────────

def test_cqr_returns_intervals_with_lo_le_hi():
    X_tr, y_tr, X_cal, y_cal, X_te, y_te = _split_synth(n=400, seed=1)
    rep = fit_cqr(
        X_train=X_tr, y_train=y_tr,
        X_calibration=X_cal, y_calibration=y_cal,
        X_test=X_te, y_test=y_te,
        target_coverage=0.90,
    )
    for lo, hi in rep.intervals:
        assert lo <= hi


def test_cqr_empirical_coverage_meets_or_exceeds_target():
    X_tr, y_tr, X_cal, y_cal, X_te, y_te = _split_synth(
        n=600, seed=2)
    rep = fit_cqr(
        X_train=X_tr, y_train=y_tr,
        X_calibration=X_cal, y_calibration=y_cal,
        X_test=X_te, y_test=y_te,
        target_coverage=0.90, n_iterations=500,
    )
    # Conformal guarantees E[coverage] >= 0.90 - 1/(n_cal+1).
    # Our test gives a single sample; allow a small downward
    # tolerance vs target.
    assert rep.empirical_coverage_test >= 0.80


def test_cqr_higher_target_coverage_yields_wider_intervals():
    X_tr, y_tr, X_cal, y_cal, X_te, y_te = _split_synth(
        n=400, seed=3)
    rep_50 = fit_cqr(
        X_train=X_tr, y_train=y_tr,
        X_calibration=X_cal, y_calibration=y_cal,
        X_test=X_te, target_coverage=0.50, n_iterations=300,
    )
    rep_90 = fit_cqr(
        X_train=X_tr, y_train=y_tr,
        X_calibration=X_cal, y_calibration=y_cal,
        X_test=X_te, target_coverage=0.90, n_iterations=300,
    )
    assert (
        rep_90.average_interval_width
        > rep_50.average_interval_width
    )


def test_cqr_rejects_invalid_coverage():
    with pytest.raises(ValueError):
        fit_cqr(
            X_train=[[1.0]], y_train=[1.0],
            X_calibration=[[1.0]], y_calibration=[1.0],
            X_test=[[1.0]],
            target_coverage=1.0,
        )


def test_cqr_rejects_empty_training_set():
    with pytest.raises(ValueError):
        fit_cqr(
            X_train=[], y_train=[],
            X_calibration=[[1.0]], y_calibration=[1.0],
            X_test=[[1.0]],
        )


def test_cqr_round_trip_through_pydantic():
    X_tr, y_tr, X_cal, y_cal, X_te, _ = _split_synth(n=200, seed=4)
    rep = fit_cqr(
        X_train=X_tr, y_train=y_tr,
        X_calibration=X_cal, y_calibration=y_cal,
        X_test=X_te, n_iterations=100,
    )
    payload = rep.model_dump(mode="json")
    rebuilt = CQRReport.model_validate(payload)
    assert rebuilt.n_test == rep.n_test
