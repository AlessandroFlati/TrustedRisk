"""Phase 17.AF - Kalman filter for vitals tests."""

from __future__ import annotations

import math
import random

import pytest

from a2a_agent.kalman_vitals import (
    Kalman1DReport, MultiKalmanReport,
    kalman_filter_1d, kalman_filter_multivariate,
)


# ─────────────────────────────────────────────────────────────────────
# 1-D Kalman
# ─────────────────────────────────────────────────────────────────────

def test_kalman_smooths_noisy_constant_signal():
    rng = random.Random(7)
    truth = 80.0
    n = 100
    z = [truth + rng.gauss(0, 4) for _ in range(n)]
    t = list(range(n))
    rep = kalman_filter_1d(
        timestamps=t, measurements=z,
        process_noise_q=0.05, measurement_noise_r=16.0,
    )
    final = rep.steps[-1].filtered_value
    # Filter should converge well within the observation noise
    assert abs(final - truth) < 2.0


def test_kalman_tracks_a_step_change_with_change_point_flag():
    timestamps = list(range(20))
    # 10 stable then a 30-unit jump
    z = [80.0] * 10 + [110.0] * 10
    rep = kalman_filter_1d(
        timestamps=timestamps, measurements=z,
        process_noise_q=0.05, measurement_noise_r=4.0,
        change_point_threshold_z=2.0,
    )
    # Some change-point flag must fire after the jump
    assert any(s.is_change_point for s in rep.steps[10:14])


def test_kalman_n_change_points_count_matches_flags():
    timestamps = list(range(30))
    z = [80.0] * 30
    z[15] = 200.0   # extreme outlier
    rep = kalman_filter_1d(
        timestamps=timestamps, measurements=z,
    )
    assert rep.n_change_points == sum(
        1 for s in rep.steps if s.is_change_point
    )


def test_kalman_rejects_misaligned():
    with pytest.raises(ValueError):
        kalman_filter_1d(
            timestamps=[1.0, 2.0],
            measurements=[10.0],
        )


def test_kalman_rejects_empty():
    with pytest.raises(ValueError):
        kalman_filter_1d(timestamps=[], measurements=[])


def test_kalman_round_trip_through_pydantic():
    rep = kalman_filter_1d(
        timestamps=[0.0, 1.0, 2.0],
        measurements=[80.0, 81.0, 82.0],
    )
    payload = rep.model_dump(mode="json")
    rebuilt = Kalman1DReport.model_validate(payload)
    assert rebuilt.n == rep.n


def test_kalman_filtered_velocity_positive_for_rising_signal():
    timestamps = list(range(10))
    z = [80.0 + i * 2.0 for i in range(10)]
    rep = kalman_filter_1d(
        timestamps=timestamps, measurements=z,
        process_noise_q=0.5, measurement_noise_r=1.0,
    )
    assert rep.steps[-1].filtered_velocity > 0.5


# ─────────────────────────────────────────────────────────────────────
# Multi-D Kalman
# ─────────────────────────────────────────────────────────────────────

def test_multivariate_filters_each_feature_independently():
    rng = random.Random(8)
    feature_names = ["HR", "SpO2", "RR"]
    timestamps = list(range(30))
    measurements = [
        {"HR": 80.0 + rng.gauss(0, 3),
         "SpO2": 96.0 + rng.gauss(0, 1),
         "RR": 16.0 + rng.gauss(0, 1)}
        for _ in range(30)
    ]
    rep = kalman_filter_multivariate(
        feature_names=feature_names,
        timestamps=timestamps, measurements=measurements,
    )
    # Filtered HR final value should be close to 80
    assert abs(rep.steps[-1].filtered["HR"] - 80.0) < 3.0


def test_multivariate_change_point_fires_when_any_feature_jumps():
    feature_names = ["HR", "SpO2"]
    timestamps = list(range(20))
    measurements = []
    for i in range(20):
        if i < 10:
            measurements.append({"HR": 80.0, "SpO2": 96.0})
        else:
            # SpO2 desaturation
            measurements.append({"HR": 80.0, "SpO2": 86.0})
    rep = kalman_filter_multivariate(
        feature_names=feature_names,
        timestamps=timestamps, measurements=measurements,
    )
    assert rep.n_change_points >= 1
    assert any(s.is_change_point for s in rep.steps[10:14])


def test_multivariate_rejects_missing_feature_in_measurement():
    with pytest.raises(ValueError):
        kalman_filter_multivariate(
            feature_names=["HR", "SpO2"],
            timestamps=[1.0],
            measurements=[{"HR": 80.0}],   # missing SpO2
        )


def test_multivariate_rejects_empty_features():
    with pytest.raises(ValueError):
        kalman_filter_multivariate(
            feature_names=[], timestamps=[1.0],
            measurements=[{}],
        )


def test_multivariate_round_trip_through_pydantic():
    rep = kalman_filter_multivariate(
        feature_names=["HR"],
        timestamps=[0.0, 1.0],
        measurements=[{"HR": 80.0}, {"HR": 82.0}],
    )
    payload = rep.model_dump(mode="json")
    rebuilt = MultiKalmanReport.model_validate(payload)
    assert rebuilt.n == rep.n
    assert rebuilt.feature_names == ["HR"]
