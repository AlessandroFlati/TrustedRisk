"""Phase 7.5 -- pre-arrival triage tests."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.preadmit_triage import (
    compute_symptom_followup_questions,
    compute_symptom_red_flag_check,
    compute_when_to_seek_care,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Red-flag check ───────────────────────


def test_chest_pain_radiation_critical():
    out = _run(compute_symptom_red_flag_check(
        "I have chest pain shooting into my left arm",
    ))
    assert out.recommendation == "call_911"
    assert any(f.flag_id == "chest_pain_with_radiation" for f in out.flags)


def test_stroke_focal_deficit_critical():
    out = _run(compute_symptom_red_flag_check(
        "I have weakness on one side and trouble speaking",
    ))
    assert out.recommendation == "call_911"
    assert any(f.flag_id == "stroke_focal_deficit" for f in out.flags)


def test_thunderclap_headache_critical():
    out = _run(compute_symptom_red_flag_check(
        "Sudden severe headache, worst headache of my life",
    ))
    assert out.recommendation == "call_911"


def test_loss_of_consciousness_high():
    out = _run(compute_symptom_red_flag_check(
        "Patient passed out for 30 seconds",
    ))
    assert out.recommendation == "go_to_ed"


def test_persistent_fever_medium():
    out = _run(compute_symptom_red_flag_check(
        "fever for 5 days that won't go away",
    ))
    assert out.recommendation in ("seek_urgent_care", "go_to_ed")


def test_no_red_flag_returns_clean():
    out = _run(compute_symptom_red_flag_check(
        "I have a stuffy nose and feel a bit under the weather",
    ))
    assert out.recommendation == "no_red_flag"
    assert out.n_flags == 0


def test_suicidal_intent_critical():
    out = _run(compute_symptom_red_flag_check(
        "I want to kill myself",
    ))
    assert out.recommendation == "call_911"


def test_pain_score_high_medium():
    out = _run(compute_symptom_red_flag_check(
        "the pain is 9 out of 10 right now",
    ))
    assert out.recommendation in ("seek_urgent_care", "go_to_ed")


# ─────────────────────── When to seek care ───────────────────────


def test_when_to_seek_care_911_for_critical_red_flag():
    out = _run(compute_when_to_seek_care(
        "chest pain radiating to my arm",
    ))
    assert out.recommended_level == "call_911"


def test_when_to_seek_care_self_care_for_mild():
    out = _run(compute_when_to_seek_care(
        "mild headache",
        duration_hours=2, severity_1_to_10=2,
    ))
    assert out.recommended_level == "self_care"


def test_when_to_seek_care_telehealth_for_moderate():
    out = _run(compute_when_to_seek_care(
        "mild congestion + sore throat",
        duration_hours=24, severity_1_to_10=5,
    ))
    assert out.recommended_level == "telehealth"


def test_when_to_seek_care_urgent_for_high_severity():
    out = _run(compute_when_to_seek_care(
        "really bad pain in my back",
        severity_1_to_10=9,
    ))
    assert out.recommended_level == "urgent_care"


def test_when_to_seek_care_pcp_24h_for_persistent_moderate():
    out = _run(compute_when_to_seek_care(
        "this back ache won't go away",
        duration_hours=120, severity_1_to_10=5,
    ))
    assert out.recommended_level == "primary_care_within_24h"


def test_when_to_seek_care_abstains_on_empty_input():
    out = _run(compute_when_to_seek_care(""))
    assert out.abstain_recommended is True


def test_when_to_seek_care_carries_watchlist():
    out = _run(compute_when_to_seek_care(
        "stuffy nose", severity_1_to_10=2,
    ))
    assert out.things_to_watch_for
    assert any("breath" in w.lower() for w in out.things_to_watch_for)


# ─────────────────────── Follow-up questions ───────────────────────


def test_followup_returns_3_generic_questions_baseline():
    out = _run(compute_symptom_followup_questions("I feel weird"))
    ids = [q.question_id for q in out.questions]
    assert "duration" in ids
    assert "severity" in ids


def test_followup_adds_chest_pain_targeted_questions():
    out = _run(compute_symptom_followup_questions("I have chest pain"))
    ids = [q.question_id for q in out.questions]
    assert "chest_radiation" in ids
    assert "chest_diaphoresis" in ids


def test_followup_adds_abdominal_targeted_question():
    out = _run(compute_symptom_followup_questions("my belly hurts"))
    ids = [q.question_id for q in out.questions]
    assert "abd_location" in ids


def test_followup_adds_headache_targeted_question():
    out = _run(compute_symptom_followup_questions("really bad headache"))
    ids = [q.question_id for q in out.questions]
    assert "hd_thunderclap" in ids


def test_followup_caps_at_max_questions():
    out = _run(compute_symptom_followup_questions(
        "chest pain and bad headache and belly hurts and short of breath",
        max_questions=4,
    ))
    assert out.n_questions <= 4


# ─────────────────────── Bundle wiring ───────────────────────


def test_preadmit_bundle_present():
    from mcp_server.tools import BUNDLES
    assert "preadmit_triage" in BUNDLES
    bundle = BUNDLES["preadmit_triage"]
    assert "compute_symptom_red_flag_check" in bundle
    assert "compute_when_to_seek_care" in bundle
    assert "compute_symptom_followup_questions" in bundle
