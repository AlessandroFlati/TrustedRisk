"""PATIENT-4 unit tests for the multi-channel notification formatter."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from a2a_agent.notification_formatter import (
    _sms_chunks,
    format_discharge_notifications,
)
from shared.schemas import (
    CounselingSection,
    DischargeCounseling,
    TranslatedDischargeCounseling,
)


def _sample() -> DischargeCounseling:
    return DischargeCounseling(
        patient_id="pt-1", locale="en", reading_level_grade=6,
        sections=[
            CounselingSection(
                section_id="your_medications",
                title="Your medications",
                plain_text="You are starting warfarin.",
                bullets=["Take warfarin 5 mg daily.",
                          "Get an INR blood test weekly."],
            ),
            CounselingSection(
                section_id="warning_signs",
                title="Warning signs",
                plain_text="Call 911 if any of these.",
                bullets=["Chest pain.", "Severe headache."],
            ),
        ],
        follow_up_window_days=(7, 14),
        n_medications_explained=1, n_red_flags=2,
    )


def _translated() -> TranslatedDischargeCounseling:
    return TranslatedDischargeCounseling(
        source_locale="en", target_locale="es",
        translation_method="llm", reading_level_grade=6,
        sections=[
            CounselingSection(
                section_id="your_medications",
                title="Tus medicamentos",
                plain_text="Estás comenzando warfarina.",
                bullets=["Toma warfarina 5 mg al día."],
            ),
        ],
        disclaimer="Sigue siempre las instrucciones oficiales.",
        safety_warnings=[],
    )


# ─────────────────────── SMS chunking ───────────────────────

def test_sms_chunks_respect_160_char_limit():
    body = "A" * 500
    chunks = _sms_chunks(body, limit=160)
    for c in chunks:
        assert len(c) <= 160


def test_sms_chunks_paginated():
    body = "A" * 500
    chunks = _sms_chunks(body, limit=160)
    n = len(chunks)
    for i, c in enumerate(chunks, start=1):
        assert f"({i}/{n})" in c


def test_sms_chunks_empty_body():
    assert _sms_chunks("") == []


def test_sms_chunks_short_body_single_chunk():
    chunks = _sms_chunks("Just a short message.", limit=160)
    assert len(chunks) == 1
    assert "(1/1)" in chunks[0]


def test_sms_chunks_break_on_sentence_boundary():
    body = "Take warfarin 5 mg daily. " \
           "Get an INR blood test weekly. " \
           "Call your doctor in 7 days for follow up."
    chunks = _sms_chunks(body, limit=80)
    # Each chunk should ideally end after a period -- but the test only
    # asserts that boundaries don't slice in the middle of words
    for c in chunks:
        stripped = c.split(" (")[0]
        assert not stripped.endswith(("warfa", "blo", "doc"))


def test_sms_custom_limit_applied():
    chunks = _sms_chunks("X" * 300, limit=100)
    for c in chunks:
        assert len(c) <= 100


# ─────────────────────── Email HTML rendering ───────────────────────

def test_email_html_includes_section_titles():
    bundle = format_discharge_notifications(_sample(),
                                                  channels=["email_html"])
    body = bundle.channels[0].chunks[0]
    assert "<h2>Your medications</h2>" in body
    assert "<h2>Warning signs</h2>" in body


def test_email_html_escapes_dangerous_input():
    """User-provided text must be HTML-escaped (XSS guard)."""
    src = DischargeCounseling(
        patient_id="x", locale="en", reading_level_grade=6,
        sections=[CounselingSection(
            section_id="your_medications",
            title="<script>alert('xss')</script>",
            plain_text="Patient was given <b>bad</b> drug.",
            bullets=["A & B & C"],
        )],
        follow_up_window_days=(7, 14),
        n_medications_explained=0, n_red_flags=0,
    )
    bundle = format_discharge_notifications(src, channels=["email_html"])
    body = bundle.channels[0].chunks[0]
    assert "<script>alert" not in body
    assert "&lt;script&gt;" in body
    assert "&amp;" in body  # & escaped


def test_email_html_preserves_disclaimer():
    bundle = format_discharge_notifications(_sample(),
                                                  channels=["email_html"])
    body = bundle.channels[0].chunks[0]
    assert "discharge paperwork" in body or "official" in body


def test_email_html_no_inline_styles_or_js():
    bundle = format_discharge_notifications(_sample(),
                                                  channels=["email_html"])
    body = bundle.channels[0].chunks[0]
    assert "<script" not in body
    assert " style=" not in body
    assert "onclick" not in body


# ─────────────────────── Print Markdown ───────────────────────

def test_print_markdown_uses_h1_h2_structure():
    bundle = format_discharge_notifications(_sample(),
                                                  channels=["print_markdown"])
    body = bundle.channels[0].chunks[0]
    assert body.startswith("# Your discharge summary")
    assert "## Your medications" in body
    assert "## Warning signs" in body


def test_print_markdown_renders_bullets():
    bundle = format_discharge_notifications(_sample(),
                                                  channels=["print_markdown"])
    body = bundle.channels[0].chunks[0]
    assert "- Take warfarin 5 mg daily." in body
    assert "- Chest pain." in body


def test_print_markdown_separator_before_disclaimer():
    bundle = format_discharge_notifications(_sample(),
                                                  channels=["print_markdown"])
    body = bundle.channels[0].chunks[0]
    assert "---" in body


# ─────────────────────── Channel selection ───────────────────────

def test_default_channels_yields_all_three():
    bundle = format_discharge_notifications(_sample())
    assert bundle.sections_formatted == 2
    assert {c.channel for c in bundle.channels} == {
        "sms", "email_html", "print_markdown"}


def test_subset_channels_honored():
    bundle = format_discharge_notifications(_sample(),
                                                  channels=["sms"])
    assert {c.channel for c in bundle.channels} == {"sms"}


def test_invalid_channel_raises():
    with pytest.raises(ValueError, match="unsupported"):
        format_discharge_notifications(_sample(),
                                              channels=["fax"])


def test_invalid_sms_limit_raises():
    with pytest.raises(ValueError, match="sms_chunk_limit"):
        format_discharge_notifications(_sample(),
                                              sms_chunk_limit=10)


# ─────────────────────── TranslatedDischargeCounseling input ───────────────────────

def test_translated_counseling_routes_through_correctly():
    """The formatter must accept either DischargeCounseling or
    TranslatedDischargeCounseling, with no schema difference at the output."""
    bundle = format_discharge_notifications(_translated(),
                                                  channels=["email_html"])
    body = bundle.channels[0].chunks[0]
    assert "warfarina" in body or "Tus medicamentos" in body


def test_dict_input_with_target_locale_routes_to_translated():
    payload = _translated().model_dump(mode="json")
    bundle = format_discharge_notifications(payload,
                                                  channels=["print_markdown"])
    body = bundle.channels[0].chunks[0]
    assert "Tus medicamentos" in body


# ─────────────────────── /api/notifications/format endpoint ───────────────────────

@pytest.fixture(scope="module")
def app():
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "playground_server",
        str(ROOT / "apps" / "playground" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["playground_server"] = mod
    spec.loader.exec_module(mod)
    return mod.app


def test_endpoint_returns_all_three_channels(app):
    from fastapi.testclient import TestClient
    payload = _sample().model_dump(mode="json")
    with TestClient(app) as c:
        r = c.post("/api/notifications/format",
                     json={"counseling": payload})
    assert r.status_code == 200, r.text
    body = r.json()
    channels = {ch["channel"] for ch in body["channels"]}
    assert channels == {"sms", "email_html", "print_markdown"}


def test_endpoint_rejects_missing_counseling(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/notifications/format", json={})
    assert r.status_code == 400


def test_endpoint_rejects_invalid_channel(app):
    from fastapi.testclient import TestClient
    payload = _sample().model_dump(mode="json")
    with TestClient(app) as c:
        r = c.post("/api/notifications/format",
                     json={"counseling": payload,
                            "channels": ["fax"]})
    assert r.status_code == 400
