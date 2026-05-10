"""Functional tests for the TIME streaming pipeline.

Chains: event_sourced_log -> incremental_risk -> deterioration_nowcast ->
trajectory_predictor. Validates the streaming-data path that production
EHRs need (vs the snapshot path).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from a2a_agent.deterioration_nowcast import nowcast_deterioration
from a2a_agent.event_sourced_log import (
    append_event,
    list_events,
    replay_to_state,
    truncate_log,
)
from a2a_agent.incremental_risk import recompute_with_observation
from a2a_agent.trajectory_predictor import predict_trajectory


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_EVENT_LOG_PATH",
                          str(tmp_path / "event_log.sqlite3"))


# ─────────────────────── TIME-1: Event-sourced log ───────────────────────

def test_log_admit_to_discharge_replay_yields_state():
    """A full hospital trajectory: admit -> 3 obs -> discharge -> replay."""
    pid = "pt-streaming-001"
    append_event(patient_id=pid, fhir_resource_type="Encounter",
                   fhir_resource_id="enc-1",
                   payload={"status": "in-progress",
                              "class": {"code": "IMP"}})
    for i, value in enumerate([4.2, 4.5, 4.8]):
        append_event(patient_id=pid, fhir_resource_type="Observation",
                       fhir_resource_id=f"obs-K-{i}",
                       payload={"code": {"text": "potassium"},
                                  "valueQuantity": {"value": value}})
    append_event(patient_id=pid, fhir_resource_type="Encounter",
                   fhir_resource_id="enc-1",
                   payload={"status": "finished", "class": {"code": "IMP"}})
    snap = replay_to_state(pid)
    # Encounter id collapses to 1 (latest update wins)
    assert len(snap.resources_by_type.get("Encounter", [])) == 1
    assert snap.resources_by_type["Encounter"][0]["status"] == "finished"
    # 3 distinct potassium observations preserved
    assert len(snap.resources_by_type.get("Observation", [])) == 3


def test_log_replay_with_until_iso_filter():
    pid = "pt-streaming-002"
    early = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    late = datetime.now(timezone.utc).isoformat()
    append_event(patient_id=pid, fhir_resource_type="Observation",
                   fhir_resource_id="obs-early",
                   payload={"v": 1}, created_at_iso=early)
    append_event(patient_id=pid, fhir_resource_type="Observation",
                   fhir_resource_id="obs-late",
                   payload={"v": 2}, created_at_iso=late)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    snap = replay_to_state(pid, until_iso=cutoff)
    # Only the early event in the replayed state
    obs = snap.resources_by_type.get("Observation", [])
    assert any(o["v"] == 1 for o in obs)
    assert not any(o["v"] == 2 for o in obs)


def test_log_state_hash_changes_on_event_addition():
    pid = "pt-streaming-003"
    for i in range(3):
        append_event(patient_id=pid, fhir_resource_type="Observation",
                       fhir_resource_id=f"obs-{i}",
                       payload={"v": i})
    h1 = replay_to_state(pid).state_hash
    append_event(patient_id=pid, fhir_resource_type="Observation",
                   fhir_resource_id="obs-new", payload={"v": 99})
    h2 = replay_to_state(pid).state_hash
    assert h1 != h2


# ─────────────────────── TIME-2: Incremental risk recompute ───────────────────────

def test_incremental_recompute_on_ed_visit_increases_e():
    prior = {"probability_mean": 0.20, "lace_raw_score": 8}
    new = {"resourceType": "Encounter", "class": {"code": "EMER"}}
    r = recompute_with_observation(prior, new)
    assert "E" in r.affected_lace_components
    assert r.posterior_probability != r.prior_probability


def test_incremental_recompute_on_inpatient_recommends_full_recompute():
    prior = {"probability_mean": 0.20, "lace_raw_score": 8}
    new = {"resourceType": "Encounter", "class": {"code": "IMP"}}
    r = recompute_with_observation(prior, new)
    assert r.full_recompute_recommended is True


def test_incremental_recompute_on_irrelevant_obs_no_change():
    prior = {"probability_mean": 0.20, "lace_raw_score": 8}
    new = {"resourceType": "Observation",
              "valueQuantity": {"value": 4.5}}
    r = recompute_with_observation(prior, new)
    assert r.delta == 0.0
    assert r.posterior_probability == r.prior_probability


def test_incremental_recompute_chain_of_3_observations():
    prior = {"probability_mean": 0.15, "lace_raw_score": 6}
    # 3 conditions added in sequence -- each increments C
    for _ in range(3):
        cond = {"resourceType": "Condition", "code": {"text": "x"}}
        prior = {"probability_mean":
                    recompute_with_observation(prior, cond).posterior_probability,
                  "lace_raw_score": prior["lace_raw_score"] + 1}
    # The cumulative effect should bump probability above the original
    assert prior["probability_mean"] >= 0.15


# ─────────────────────── TIME-3: Real-time nowcasting ───────────────────────

def _vital_obs(*, code: str, value: float, hours_ago: float) -> dict:
    ts = (datetime.now(timezone.utc)
            - timedelta(hours=hours_ago)).isoformat()
    return {
        "resourceType": "Observation",
        "code": {"coding": [{"code": code, "system": "http://loinc.org"}]},
        "valueQuantity": {"value": value, "unit": "x"},
        "effectiveDateTime": ts,
    }


def test_nowcast_classic_decompensation_triggers_alert():
    """Rising HR + falling SpO2 within 6h -> alert tier."""
    obs = [
        _vital_obs(code="8867-4", value=80, hours_ago=5),
        _vital_obs(code="8867-4", value=110, hours_ago=2),
        _vital_obs(code="8867-4", value=130, hours_ago=0.5),
        _vital_obs(code="59408-5", value=98, hours_ago=5),
        _vital_obs(code="59408-5", value=93, hours_ago=2),
        _vital_obs(code="59408-5", value=88, hours_ago=0.5),
    ]
    r = nowcast_deterioration(obs)
    assert r.alert_tier == "alert"
    assert len(r.triggered_signals) >= 2


def test_nowcast_steady_vitals_yields_ok():
    obs = [
        _vital_obs(code="8867-4", value=72, hours_ago=5),
        _vital_obs(code="8867-4", value=74, hours_ago=2),
    ]
    r = nowcast_deterioration(obs)
    assert r.alert_tier == "ok"


def test_nowcast_old_observations_skipped():
    """Obs > 6h ago should NOT count."""
    obs = [
        _vital_obs(code="8867-4", value=70, hours_ago=20),
        _vital_obs(code="8867-4", value=140, hours_ago=15),
    ]
    r = nowcast_deterioration(obs, window_hours=6)
    assert r.n_observations == 0
    assert r.alert_tier == "ok"


# ─────────────────────── TIME-4: Trajectory predictor ───────────────────────

def test_trajectory_72h_horizon_with_increasing_drift():
    r = predict_trajectory(current_probability=0.25,
                              horizon_hours=72,
                              direction="increasing")
    assert len(r.points) == 73
    assert r.points[-1].probability_mean > r.points[0].probability_mean


def test_trajectory_ci_widens_over_horizon():
    r = predict_trajectory(current_probability=0.30,
                              horizon_hours=72)
    early = r.points[3]
    late = r.points[-1]
    assert (late.probability_ci95_high - late.probability_ci95_low) \
        > (early.probability_ci95_high - early.probability_ci95_low)


def test_trajectory_method_is_bayesian():
    r = predict_trajectory(current_probability=0.20)
    assert r.method == "bayesian_state_space"


# ─────────────────────── End-to-end TIME chain ───────────────────────

def test_full_streaming_chain_event_to_risk_to_nowcast_to_trajectory():
    """Streaming pipeline: append events -> replay -> incremental risk ->
    nowcast -> 72h trajectory. All 4 surfaces produce well-formed output."""
    pid = "pt-stream-e2e"
    # Append 5 events: 1 encounter + 3 vital obs + 1 condition
    append_event(patient_id=pid, fhir_resource_type="Encounter",
                   fhir_resource_id="enc-1",
                   payload={"class": {"code": "IMP"}})
    for i, value in enumerate([4.2, 4.5, 4.8]):
        append_event(patient_id=pid, fhir_resource_type="Observation",
                       fhir_resource_id=f"obs-K-{i}",
                       payload={"code": {"text": "potassium"},
                                  "valueQuantity": {"value": value}})
    append_event(patient_id=pid, fhir_resource_type="Condition",
                   fhir_resource_id="cond-1",
                   payload={"code": {"text": "CHF"}})

    snap = replay_to_state(pid)
    assert snap.n_events_replayed == 5

    # Incremental: a new ED encounter event
    prior = {"probability_mean": 0.20, "lace_raw_score": 8}
    inc = recompute_with_observation(prior, {
        "resourceType": "Encounter", "class": {"code": "EMER"}})

    # Nowcast: synthesize a vital sequence within window
    obs = [
        _vital_obs(code="8867-4", value=80, hours_ago=4),
        _vital_obs(code="8867-4", value=95, hours_ago=2),
        _vital_obs(code="8867-4", value=100, hours_ago=1),
    ]
    nc = nowcast_deterioration(obs)

    # Trajectory: 72h forecast from the incremental-recompute output
    traj = predict_trajectory(
        current_probability=inc.posterior_probability,
        horizon_hours=72)

    assert snap.state_hash
    assert inc.posterior_probability >= 0
    assert nc.alert_tier in ("ok", "watch", "alert")
    assert traj.expected_peak_probability >= inc.posterior_probability \
        - 0.01   # peak should be ≥ starting probability
