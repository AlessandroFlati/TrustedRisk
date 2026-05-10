"""Unit tests for trauma_severity_score + massive_transfusion_protocol."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.trauma_severity_score import compute_trauma_severity_score
from mcp_server.tools.massive_transfusion_protocol import compute_massive_transfusion_protocol


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── ISS / RTS ───────────────────────

def test_iss_minor_trauma():
    rep = _run(compute_trauma_severity_score(
        injuries=[{"body_region": "extremities_pelvic_girdle", "ais_severity": 2}],
        glasgow_coma_score=15, systolic_bp=125, respiratory_rate=16,
    ))
    assert rep.iss == 4
    assert rep.iss_band == "minor"
    assert rep.rts >= 11.0
    assert rep.triage_priority == "minor_injury_outpatient"


def test_iss_major_polytrauma():
    rep = _run(compute_trauma_severity_score(
        injuries=[
            {"body_region": "head_neck", "ais_severity": 4},
            {"body_region": "chest", "ais_severity": 3},
            {"body_region": "abdomen_pelvis", "ais_severity": 4},
        ], glasgow_coma_score=10, systolic_bp=85, respiratory_rate=24,
        has_active_hemorrhage=True,
    ))
    # ISS = 16+9+16 = 41
    assert rep.iss == 41
    assert rep.iss_band == "severe"
    assert rep.triage_priority == "operating_room_immediate"


def test_iss_75_when_unsurvivable():
    rep = _run(compute_trauma_severity_score(
        injuries=[{"body_region": "head_neck", "ais_severity": 6}],
    ))
    assert rep.iss == 75
    assert rep.iss_band == "unsurvivable"


def test_rts_normal_physiology():
    rep = _run(compute_trauma_severity_score(
        injuries=[], glasgow_coma_score=15, systolic_bp=120, respiratory_rate=16,
    ))
    assert rep.rts >= 11.5
    assert rep.rts_band == "normal_physiology"


def test_rts_critical_with_severe_hypotension():
    rep = _run(compute_trauma_severity_score(
        injuries=[{"body_region": "abdomen_pelvis", "ais_severity": 4}],
        glasgow_coma_score=6, systolic_bp=60, respiratory_rate=8,
    ))
    assert rep.rts < 8
    assert rep.rts_band == "critical"


# ─────────────────────── ABC / MTP ───────────────────────

def test_mtp_full_score():
    rep = _run(compute_massive_transfusion_protocol(
        penetrating_mechanism=True, field_or_arrival_sbp_le_90=True,
        heart_rate_ge_120=True, positive_fast_exam=True,
        minutes_since_injury=45,
    ))
    assert rep.abc_score == 4
    assert rep.mtp_activated is True
    assert rep.txa_indicated is True
    assert rep.estimated_initial_request["rbc_units"] >= 4


def test_mtp_below_threshold():
    rep = _run(compute_massive_transfusion_protocol(
        penetrating_mechanism=False, field_or_arrival_sbp_le_90=False,
        heart_rate_ge_120=True, positive_fast_exam=False,
    ))
    assert rep.abc_score == 1
    assert rep.mtp_activated is False
    assert rep.estimated_initial_request["rbc_units"] == 0


def test_txa_not_indicated_after_3h():
    rep = _run(compute_massive_transfusion_protocol(
        penetrating_mechanism=True, field_or_arrival_sbp_le_90=True,
        heart_rate_ge_120=True, positive_fast_exam=False,
        minutes_since_injury=240,  # > 180 min
    ))
    assert rep.mtp_activated is True
    assert rep.txa_indicated is False
    assert rep.txa_window_open is False
    assert any("180 min" in r for r in rep.additional_recommendations)


def test_pediatric_request_scales_by_weight():
    rep = _run(compute_massive_transfusion_protocol(
        penetrating_mechanism=True, field_or_arrival_sbp_le_90=True,
        heart_rate_ge_120=True, positive_fast_exam=False,
        pediatric_weight_kg=30,
    ))
    # Pediatric path uses scaled units
    assert rep.estimated_initial_request["rbc_units"] >= 1
    assert rep.estimated_initial_request["platelet_packs"] >= 1


def test_large_blood_loss_triggers_second_round():
    rep = _run(compute_massive_transfusion_protocol(
        penetrating_mechanism=True, field_or_arrival_sbp_le_90=True,
        heart_rate_ge_120=True, positive_fast_exam=True,
        estimated_blood_loss_ml=2500,
    ))
    assert rep.estimated_initial_request["rbc_units"] == 6
    assert rep.estimated_initial_request["ffp_units"] == 6


def test_mtp_recommendations_include_calcium():
    rep = _run(compute_massive_transfusion_protocol(
        penetrating_mechanism=True, field_or_arrival_sbp_le_90=True,
        heart_rate_ge_120=True, positive_fast_exam=True,
    ))
    text = " ".join(rep.additional_recommendations).lower()
    assert "calcium" in text
    assert "fibrinogen" in text or "cryoprecipitate" in text.lower()
