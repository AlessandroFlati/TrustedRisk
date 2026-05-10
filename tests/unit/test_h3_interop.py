"""Phase 16.H3 - SMART-on-FHIR + DICOM SR + CDS Hooks card tests."""

from __future__ import annotations

import pytest

from a2a_agent.cds_hooks_card import (
    build_decision_card, build_response,
)
from a2a_agent.dicom_sr import (
    build_minimal_dicom_sr_bytes, parse_dicom_sr_bytes,
)
from a2a_agent.smart_on_fhir import (
    begin_smart_launch, code_challenge_for, decode_id_token_unsafe,
    generate_code_verifier, load_smart_configuration,
    simulate_ehr_token_exchange,
)


# ─────────────────────────────────────────────────────────────────────
# SMART-on-FHIR
# ─────────────────────────────────────────────────────────────────────

def test_smart_configuration_carries_required_endpoints():
    cfg = load_smart_configuration("https://hapi.fhir.org/baseR4")
    assert "authorization_endpoint" in cfg
    assert "token_endpoint" in cfg
    assert "S256" in cfg["code_challenge_methods_supported"]
    assert "launch-ehr" in cfg["capabilities"]


def test_pkce_challenge_matches_verifier_round_trip():
    v = generate_code_verifier()
    challenge_a = code_challenge_for(v)
    challenge_b = code_challenge_for(v)
    assert challenge_a == challenge_b
    assert challenge_a != v   # must hash, not echo


def test_begin_smart_launch_rejects_empty_iss():
    with pytest.raises(ValueError):
        begin_smart_launch(iss="", launch_token="x")


def test_begin_smart_launch_rejects_empty_launch():
    with pytest.raises(ValueError):
        begin_smart_launch(iss="https://x", launch_token="")


def test_full_smart_launch_round_trip_succeeds():
    ctx = begin_smart_launch(
        iss="https://hapi.fhir.org/baseR4",
        launch_token="opaque-launch-1",
        rng_seed=7,
    )
    assert ctx.code_challenge_method == "S256"
    trace = simulate_ehr_token_exchange(
        ctx=ctx, code="auth-code-1",
        returned_state=ctx.state,
        code_verifier=ctx.code_verifier,
        patient_id="p-001", encounter_id="e-7",
    )
    assert trace.pkce_validated is True
    assert trace.token_response.patient == "p-001"
    decoded = decode_id_token_unsafe(trace.token_response.id_token)
    assert decoded["nonce"] == ctx.nonce
    assert decoded["sub"] == "p-001"


def test_smart_launch_rejects_state_mismatch():
    ctx = begin_smart_launch(
        iss="https://x", launch_token="lt", rng_seed=11,
    )
    with pytest.raises(ValueError):
        simulate_ehr_token_exchange(
            ctx=ctx, code="c", returned_state="ATTACKER",
            code_verifier=ctx.code_verifier,
            patient_id="p", encounter_id="e",
        )


def test_smart_launch_rejects_pkce_verifier_mismatch():
    ctx = begin_smart_launch(
        iss="https://x", launch_token="lt", rng_seed=12,
    )
    with pytest.raises(ValueError):
        simulate_ehr_token_exchange(
            ctx=ctx, code="c", returned_state=ctx.state,
            code_verifier="wrong-verifier",
            patient_id="p", encounter_id="e",
        )


# ─────────────────────────────────────────────────────────────────────
# DICOM SR
# ─────────────────────────────────────────────────────────────────────

def test_dicom_sr_round_trip():
    blob = build_minimal_dicom_sr_bytes(
        items=[
            ("Imaging Procedure Description", "TEXT",
             "CT chest with contrast"),
            ("Impressions", "TEXT",
             "No acute findings."),
        ],
    )
    rep = parse_dicom_sr_bytes(blob)
    assert rep.modality == "SR"
    assert rep.n_items == 2
    assert any("CT chest" in i.concept_value for i in rep.items)
    assert any("No acute" in i.concept_value for i in rep.items)


def test_dicom_sr_rejects_short_blob():
    with pytest.raises(ValueError):
        parse_dicom_sr_bytes(b"too short")


def test_dicom_sr_rejects_missing_dicm_magic():
    blob = b"\x00" * 128 + b"NOPE" + b"\x00" * 100
    with pytest.raises(ValueError):
        parse_dicom_sr_bytes(blob)


def test_dicom_sr_completion_flag_propagates():
    blob = build_minimal_dicom_sr_bytes()
    rep = parse_dicom_sr_bytes(blob)
    assert rep.completion_flag == "PARTIAL"
    assert rep.verification_flag == "UNVERIFIED"


# ─────────────────────────────────────────────────────────────────────
# CDS Hooks card
# ─────────────────────────────────────────────────────────────────────

def test_cds_card_indicator_is_warning_for_high_risk():
    card = build_decision_card(
        patient_id="p1", encounter_id="e1",
        recommended_action="continued_admission",
        risk_point_estimate=0.45,
    )
    assert card.indicator == "warning"


def test_cds_card_indicator_is_info_for_low_risk():
    card = build_decision_card(
        patient_id="p1", encounter_id="e1",
        recommended_action="discharge_home",
        risk_point_estimate=0.05,
    )
    assert card.indicator == "info"


def test_cds_card_summary_includes_action_and_risk_pct():
    card = build_decision_card(
        patient_id="p1", encounter_id="e1",
        recommended_action="discharge_with_homecare",
        risk_point_estimate=0.18,
    )
    assert "discharge_with_homecare" in card.summary
    assert "18.0%" in card.summary


def test_cds_card_includes_smart_app_link():
    card = build_decision_card(
        patient_id="p1", encounter_id="e1",
        recommended_action="discharge_home",
        risk_point_estimate=0.05,
    )
    smart_links = [l for l in card.links if l.type == "smart"]
    assert smart_links
    assert "patient=p1" in smart_links[0].url


def test_cds_card_includes_default_override_reasons():
    card = build_decision_card(
        patient_id="p1", encounter_id=None,
        recommended_action="abstain",
        risk_point_estimate=0.5,
    )
    assert len(card.overrideReasons) >= 4


def test_cds_card_rejects_out_of_unit_risk():
    with pytest.raises(ValueError):
        build_decision_card(
            patient_id="p", encounter_id="e",
            recommended_action="discharge_home",
            risk_point_estimate=1.5,
        )


def test_cds_response_carries_card():
    card = build_decision_card(
        patient_id="p", encounter_id="e",
        recommended_action="discharge_home",
        risk_point_estimate=0.05,
    )
    resp = build_response([card])
    assert len(resp.cards) == 1
    assert resp.systemActions == []


def test_cds_card_suggestion_uuid_is_deterministic():
    a = build_decision_card(
        patient_id="p", encounter_id="e",
        recommended_action="discharge_home",
        risk_point_estimate=0.1,
    )
    b = build_decision_card(
        patient_id="p", encounter_id="e",
        recommended_action="discharge_home",
        risk_point_estimate=0.1,
    )
    assert (
        a.suggestions[0].uuid == b.suggestions[0].uuid
    )
