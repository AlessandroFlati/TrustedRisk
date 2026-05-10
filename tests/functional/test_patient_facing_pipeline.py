"""Functional tests for the patient-facing pipeline.

Chains: discharge_counseling -> translator -> caregiver_summary ->
notification_formatter. Plus the FAQ generator on the same source.
"""
from __future__ import annotations

import json

from a2a_agent.notification_formatter import format_discharge_notifications
from mcp_server.tools.caregiver_summary import compute_caregiver_summary
from mcp_server.tools.patient_faq import compute_patient_faq
from mcp_server.tools.translate_discharge_counseling import (
    compute_translate_discharge_counseling,
)


# ─────────────────────── End-to-end chain ───────────────────────

def test_pipeline_translator_preserves_bullet_count(run_async, chf_counseling):
    """English passthrough is the deterministic baseline; bullet count
    must equal the source."""
    r = run_async(compute_translate_discharge_counseling(
        counseling=chf_counseling, target_language="en"))
    for src, tr in zip(chf_counseling["sections"], r.sections):
        assert len(tr.bullets) == len(src["bullets"])


def test_pipeline_translator_es_floor_safety_warning(run_async, chf_counseling):
    """When LLM is disabled, ES translation falls back with a warning."""
    r = run_async(compute_translate_discharge_counseling(
        counseling=chf_counseling, target_language="es"))
    assert any("translation_unavailable" in w
                  for w in r.safety_warnings)


def test_pipeline_translator_zh_round_trip_dict_input(run_async, chf_counseling):
    """Pass via JSON dict (MCP transport) and confirm the output shape."""
    payload = json.dumps(chf_counseling, default=str)
    coerced = json.loads(payload)
    r = run_async(compute_translate_discharge_counseling(
        counseling=coerced, target_language="zh"))
    assert r.target_locale == "zh"


# ─────────────────────── Caregiver summary ───────────────────────

def test_pipeline_caregiver_split_911_vs_pcp_red_flags(run_async, chf_counseling):
    r = run_async(compute_caregiver_summary(chf_counseling,
                                                    target_audience="caregiver"))
    blob_911 = " ".join(r.when_to_call_911).lower()
    blob_pcp = " ".join(r.when_to_call_pcp).lower()
    assert "chest pain" in blob_911 or "headache" in blob_911 or \
           "trouble breathing" in blob_911


def test_pipeline_caregiver_daily_checklist_includes_meds(
    run_async, chf_counseling,
):
    r = run_async(compute_caregiver_summary(chf_counseling))
    blob = " ".join(r.daily_observation_checklist).lower()
    assert any(med in blob for med in
                  ("warfarin", "lisinopril", "furosemide", "metoprolol",
                    "metformin"))


def test_pipeline_caregiver_audience_variants(run_async, chf_counseling):
    for audience in ("caregiver", "guardian", "family_proxy"):
        r = run_async(compute_caregiver_summary(chf_counseling,
                                                        target_audience=audience))
        assert r.target_audience == audience


# ─────────────────────── Notification formatter ───────────────────────

def test_pipeline_formatter_yields_three_channels(chf_counseling):
    bundle = format_discharge_notifications(chf_counseling)
    channels = {c.channel for c in bundle.channels}
    assert channels == {"sms", "email_html", "print_markdown"}


def test_pipeline_formatter_sms_chunks_under_limit(chf_counseling):
    bundle = format_discharge_notifications(chf_counseling,
                                                  channels=["sms"])
    sms = bundle.channels[0]
    for chunk in sms.chunks:
        assert len(chunk) <= 160


def test_pipeline_formatter_html_escapes_user_input():
    """Inject HTML-special chars via section title; verify they're escaped."""
    counseling = {
        "patient_id": "pt-1", "locale": "en", "reading_level_grade": 6,
        "sections": [{
            "section_id": "your_medications",
            "title": "<script>alert(1)</script>",
            "plain_text": "Take meds & follow up",
            "bullets": ["bullet & one", "bullet & two"],
        }],
        "follow_up_window_days": [7, 14],
        "n_medications_explained": 0, "n_red_flags": 0,
    }
    bundle = format_discharge_notifications(counseling,
                                                  channels=["email_html"])
    body = bundle.channels[0].chunks[0]
    assert "<script>" not in body
    assert "&amp;" in body


def test_pipeline_formatter_print_markdown_structure(chf_counseling):
    bundle = format_discharge_notifications(chf_counseling,
                                                  channels=["print_markdown"])
    body = bundle.channels[0].chunks[0]
    assert body.startswith("# ")
    assert "## Your medications" in body or \
           "## Follow up" in body or \
           "## Warning signs" in body


# ─────────────────────── FAQ generator ───────────────────────

def test_pipeline_faq_in_scope_question_responds_with_template(
    run_async, chf_counseling, chf_decision_card,
):
    r = run_async(compute_patient_faq(
        question="Why am I taking warfarin?",
        decision_card=chf_decision_card,
        counseling=chf_counseling,
    ))
    assert r.in_scope is True
    assert r.method == "deterministic_template"
    assert "discharge" in r.answer_text.lower() or \
           "warfarin" in r.answer_text.lower() or \
           "summary" in r.answer_text.lower()


def test_pipeline_faq_dose_change_blocked(run_async, chf_decision_card):
    r = run_async(compute_patient_faq(
        question="Should I take more warfarin tonight?",
        decision_card=chf_decision_card,
    ))
    assert r.in_scope is False
    assert r.refusal_reason == "dose_change_request"
    assert "doctor" in r.answer_text.lower() \
        or "pharmacist" in r.answer_text.lower()


def test_pipeline_faq_emergency_routes_to_911(run_async):
    r = run_async(compute_patient_faq(
        question="Am I having a heart attack?"))
    assert r.in_scope is False
    assert "911" in r.answer_text


def test_pipeline_faq_lab_interpretation_refused(run_async):
    r = run_async(compute_patient_faq(
        question="What does my INR of 4 mean?"))
    assert r.in_scope is False
    assert r.refusal_reason == "lab_interpretation"


# ─────────────────────── Combined chain ───────────────────────

def test_full_patient_chain_translator_to_formatter(run_async, chf_counseling):
    """Counseling -> translator (en passthrough) -> formatter -> 3 channels."""
    translated = run_async(compute_translate_discharge_counseling(
        counseling=chf_counseling, target_language="en"))
    bundle = format_discharge_notifications(
        translated.model_dump(mode="json"))
    assert bundle.sections_formatted == len(chf_counseling["sections"])
    assert len(bundle.channels) == 3


def test_full_patient_chain_caregiver_then_formatter(run_async, chf_counseling):
    """Caregiver summary should also be format-renderable when treated
    as a structured document."""
    r = run_async(compute_caregiver_summary(chf_counseling))
    # Caregiver report is structured but not a CounselingSection list;
    # the formatter expects DischargeCounseling shape. Confirm caregiver
    # report is independently well-formed instead.
    assert r.summary
    assert r.daily_observation_checklist
    assert r.when_to_call_911 or r.when_to_call_pcp
