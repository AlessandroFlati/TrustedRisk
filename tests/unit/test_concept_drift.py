"""Phase 17.AT - Concept-drift online detection tests."""

from __future__ import annotations

import random

import pytest

from a2a_agent.concept_drift import (
    ADWIN, ADWINReport, DDM, DDMReport,
    detect_drifts_adwin, detect_drifts_ddm,
)


# ─────────────────────────────────────────────────────────────────────
# ADWIN
# ─────────────────────────────────────────────────────────────────────

def test_adwin_no_drift_on_stationary_stream():
    rng = random.Random(7)
    a = ADWIN(delta=0.001)
    for _ in range(500):
        a.add(rng.gauss(0.5, 0.05))
    rep = a.report()
    assert rep.drift_detected_at == []


def test_adwin_detects_drift_on_step_change():
    rng = random.Random(7)
    a = ADWIN(delta=0.05)
    for _ in range(120):
        a.add(rng.gauss(0.2, 0.02))
    drift_after = []
    for _ in range(120):
        if a.add(rng.gauss(0.8, 0.02)):
            drift_after.append(a._n_seen)
    assert len(a.report().drift_detected_at) >= 1


def test_adwin_rejects_invalid_delta():
    with pytest.raises(ValueError):
        ADWIN(delta=0.0)
    with pytest.raises(ValueError):
        ADWIN(delta=1.0)


def test_adwin_rejects_tiny_window():
    with pytest.raises(ValueError):
        ADWIN(max_window=2)


def test_adwin_report_round_trip():
    a = ADWIN()
    for v in (0.1, 0.2, 0.1, 0.3):
        a.add(v)
    rep = a.report()
    payload = rep.model_dump(mode="json")
    rebuilt = ADWINReport.model_validate(payload)
    assert rebuilt.n_samples_seen == rep.n_samples_seen


def test_detect_drifts_adwin_helper_returns_list():
    rng = random.Random(11)
    stream = (
        [rng.gauss(0.2, 0.02) for _ in range(100)]
        + [rng.gauss(0.8, 0.02) for _ in range(100)]
    )
    drifts = detect_drifts_adwin(stream=stream, delta=0.05)
    assert isinstance(drifts, list)
    assert all(isinstance(d, int) for d in drifts)


# ─────────────────────────────────────────────────────────────────────
# DDM
# ─────────────────────────────────────────────────────────────────────

def test_ddm_starts_in_control():
    d = DDM()
    rep = d.report()
    assert rep.state == "in_control"


def test_ddm_rejects_invalid_min_n():
    with pytest.raises(ValueError):
        DDM(min_n_before_check=2)


def test_ddm_rejects_invalid_error_value():
    d = DDM()
    with pytest.raises(ValueError):
        d.add(2)


def test_ddm_in_control_for_low_error_rate_stream():
    rng = random.Random(7)
    d = DDM()
    for _ in range(200):
        d.add(1 if rng.random() < 0.05 else 0)
    assert d.report().state in ("in_control", "warning")


def test_ddm_drifts_on_jump_in_error_rate():
    rng = random.Random(7)
    d = DDM(min_n_before_check=20)
    # Establish a low-error baseline
    for _ in range(150):
        d.add(1 if rng.random() < 0.05 else 0)
    # Now jump to 70% error
    drifted = False
    for _ in range(200):
        state = d.add(1 if rng.random() < 0.7 else 0)
        if state == "drift":
            drifted = True
            break
    assert drifted


def test_ddm_round_trip_through_pydantic():
    d = DDM()
    for e in (0, 0, 1, 0, 1, 0, 0, 0):
        d.add(e)
    rep = d.report()
    payload = rep.model_dump(mode="json")
    rebuilt = DDMReport.model_validate(payload)
    assert rebuilt.n_samples_seen == rep.n_samples_seen


def test_detect_drifts_ddm_helper():
    rng = random.Random(7)
    errors = (
        [1 if rng.random() < 0.05 else 0 for _ in range(100)]
        + [1 if rng.random() < 0.7 else 0 for _ in range(100)]
    )
    drifts = detect_drifts_ddm(errors=errors)
    assert isinstance(drifts, list)
    assert all(isinstance(d, int) for d in drifts)
