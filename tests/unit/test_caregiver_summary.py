"""PATIENT-3 unit tests for the caregiver summarizer."""
from __future__ import annotations

import asyncio
import json

import pytest

from mcp_server.tools import caregiver_summary as mod
from mcp_server.tools.caregiver_summary import (
    _build_daily_checklist,
    _split_red_flags,
    compute_caregiver_summary,
)
from shared.schemas import (
    CounselingSection,
    DischargeCounseling,
)


def _run(coro):
    return asyncio.run(coro)


def _sample_counseling() -> DischargeCounseling:
    return DischargeCounseling(
        patient_id="pt-1",
        locale="en", reading_level_grade=6,
        sections=[
            CounselingSection(
                section_id="your_medications",
                title="Your medications",
                plain_text="Warfarin to thin your blood.",
                bullets=["Warfarin 5 mg daily."],
            ),
            CounselingSection(
                section_id="follow_up",
                title="Follow up",
                plain_text="See your doctor in 7 days.",
                bullets=["Call PCP within 7 days."],
            ),
            CounselingSection(
                section_id="warning_signs",
                title="Warning signs",
                plain_text="Watch for these.",
                bullets=[
                    "Chest pain that does not go away.",   # 911
                    "Severe headache.",                       # 911
                    "Mild bruising.",                          # PCP
                    "Mild nausea.",                            # PCP
                ],
            ),
        ],
        follow_up_window_days=(7, 14),
        n_medications_explained=1, n_red_flags=4,
    )


@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")


# ─────────────────────── Validation ───────────────────────

def test_invalid_audience_raises():
    with pytest.raises(ValueError, match="target_audience"):
        _run(compute_caregiver_summary(_sample_counseling(),
                                              target_audience="random_role"))


def test_dict_input_coerced():
    payload = _sample_counseling().model_dump(mode="json")
    r = _run(compute_caregiver_summary(payload))
    assert r.target_audience == "caregiver"


# ─────────────────────── Red-flag splitting ───────────────────────

def test_red_flag_splitter_separates_911_and_pcp():
    bullets = [
        "Chest pain that does not go away.",    # 911
        "Severe headache.",                        # 911
        "Mild bruising.",                           # PCP
        "Slight nausea.",                           # PCP
    ]
    call_911, call_pcp = _split_red_flags(bullets)
    assert "Chest pain that does not go away." in call_911
    assert "Severe headache." in call_911
    assert "Mild bruising." in call_pcp
    assert "Slight nausea." in call_pcp


def test_red_flag_splitter_empty_input():
    a, b = _split_red_flags([])
    assert a == []
    assert b == []


# ─────────────────────── Daily checklist builder ───────────────────────

def test_daily_checklist_includes_medications():
    c = _sample_counseling()
    items = _build_daily_checklist(c)
    assert any("Warfarin" in i for i in items)


def test_daily_checklist_includes_baseline_observations():
    c = _sample_counseling()
    items = _build_daily_checklist(c)
    blob = " ".join(items).lower()
    assert "weight" in blob
    assert "symptoms" in blob


def test_daily_checklist_capped_at_10_items():
    c = DischargeCounseling(
        patient_id="x", locale="en", reading_level_grade=6,
        sections=[CounselingSection(
            section_id="your_medications",
            title="Meds",
            plain_text="t",
            bullets=[f"Drug{i} 5 mg daily." for i in range(20)],
        )],
        follow_up_window_days=(7, 14),
        n_medications_explained=20, n_red_flags=0,
    )
    assert len(_build_daily_checklist(c)) <= 10


# ─────────────────────── Deterministic floor ───────────────────────

def test_deterministic_floor_returns_template():
    c = _sample_counseling()
    r = _run(compute_caregiver_summary(c))
    assert r.method == "deterministic_template"
    assert r.reading_level_grade == 10  # higher than patient default


def test_deterministic_floor_separates_911_and_pcp_lists():
    c = _sample_counseling()
    r = _run(compute_caregiver_summary(c))
    blob_911 = " ".join(r.when_to_call_911).lower()
    blob_pcp = " ".join(r.when_to_call_pcp).lower()
    assert "chest pain" in blob_911
    assert ("bruising" in blob_pcp or "nausea" in blob_pcp)


def test_deterministic_floor_red_flag_actions_have_decision_rules():
    c = _sample_counseling()
    r = _run(compute_caregiver_summary(c))
    blob = " ".join(r.red_flag_actions).lower()
    assert "911" in blob
    assert "pcp" in blob or "primary care" in blob.replace("->", "")


# ─────────────────────── LLM happy path ───────────────────────

def test_llm_overrides_deterministic_when_valid_json(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    response = json.dumps({
        "summary": "Caregiver summary written by the LLM at 10th grade.",
        "red_flag_actions": [
            "If chest pain -> call 911",
            "If mild bruising -> call PCP within 4 hours",
            "If patient cannot speak -> call 911",
        ],
        "daily_observation_checklist": [
            "Confirm warfarin taken at 8pm.",
            "Record weight.",
            "Check for new bruises.",
        ],
        "when_to_call_pcp": ["Persistent mild bruising."],
        "when_to_call_911": ["Chest pain that does not go away."],
    })
    monkeypatch.setattr(mod, "_call_ollama", lambda _: response)
    r = _run(compute_caregiver_summary(_sample_counseling()))
    assert r.method == "llm"
    assert "10th grade" in r.summary
    assert any("warfarin" in i.lower() for i in r.daily_observation_checklist)


# ─────────────────────── LLM hardening ───────────────────────

def test_invalid_json_falls_back_to_deterministic(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama", lambda _: "I am not JSON.")
    r = _run(compute_caregiver_summary(_sample_counseling()))
    assert r.method == "deterministic_template"


def test_truncated_json_falls_back(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama", lambda _: '{"summary": "ok",')
    r = _run(compute_caregiver_summary(_sample_counseling()))
    assert r.method == "deterministic_template"


def test_empty_response_falls_back(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama", lambda _: "")
    r = _run(compute_caregiver_summary(_sample_counseling()))
    assert r.method == "deterministic_template"


def test_oversized_lists_capped(monkeypatch):
    """LLM hallucinating 100-item lists must get truncated."""
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    huge = json.dumps({
        "summary": "ok",
        "red_flag_actions": [f"action {i}" for i in range(50)],
        "daily_observation_checklist": [f"obs {i}" for i in range(50)],
        "when_to_call_pcp": [f"pcp {i}" for i in range(50)],
        "when_to_call_911": [f"911 {i}" for i in range(50)],
    })
    monkeypatch.setattr(mod, "_call_ollama", lambda _: huge)
    r = _run(compute_caregiver_summary(_sample_counseling()))
    assert len(r.red_flag_actions) <= 8
    assert len(r.daily_observation_checklist) <= 12
    assert len(r.when_to_call_pcp) <= 10
    assert len(r.when_to_call_911) <= 10


# ─────────────────────── Audience variants ───────────────────────

def test_guardian_audience():
    r = _run(compute_caregiver_summary(_sample_counseling(),
                                              target_audience="guardian"))
    assert r.target_audience == "guardian"


def test_family_proxy_audience():
    r = _run(compute_caregiver_summary(_sample_counseling(),
                                              target_audience="family_proxy"))
    assert r.target_audience == "family_proxy"


# ─────────────────────── Bundle registration ───────────────────────

def test_caregiver_in_patient_facing_bundle():
    from mcp_server.tools import BUNDLES
    assert "compute_caregiver_summary" in BUNDLES["patient_facing"]
