"""PATIENT-2 unit tests for the patient FAQ generator."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools import patient_faq as mod
from mcp_server.tools.patient_faq import (
    _classify_scope,
    compute_patient_faq,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")


# ─────────────────────── Scope classification ───────────────────────

@pytest.mark.parametrize("question", [
    "Should I take more warfarin tonight?",
    "Should I stop my lisinopril?",
    "Can I stop the metformin?",
    "What does my INR of 4 mean?",
    "Diagnose me.",
    "What's my prognosis?",
    "Am I having a heart attack?",
    "Should I go to the ER?",
])
def test_out_of_scope_classified_correctly(question):
    in_scope, reason, _ = _classify_scope(question)
    assert in_scope is False
    assert reason is not None


@pytest.mark.parametrize("question", [
    "Why am I taking warfarin?",
    "When should I take my pills?",
    "What side effects should I watch for?",
    "Can I drink coffee with this medication?",
    "When should I see my doctor next?",
    "What foods should I avoid?",
])
def test_in_scope_classified_correctly(question):
    in_scope, reason, _ = _classify_scope(question)
    assert in_scope is True
    assert reason is None


# ─────────────────────── Refusal pathway ───────────────────────

def test_dose_change_question_refused():
    r = _run(compute_patient_faq("Should I take more warfarin?"))
    assert r.in_scope is False
    assert r.refusal_reason == "dose_change_request"
    assert "doctor" in r.answer_text.lower() or \
           "pharmacist" in r.answer_text.lower()


def test_acute_symptom_routes_to_911():
    r = _run(compute_patient_faq("Am I having a heart attack?"))
    assert r.in_scope is False
    assert "911" in r.answer_text


def test_lab_interpretation_refused():
    r = _run(compute_patient_faq("What does my INR of 4 mean?"))
    assert r.in_scope is False
    assert r.refusal_reason == "lab_interpretation"


def test_diagnosis_request_refused():
    r = _run(compute_patient_faq("Diagnose me."))
    assert r.in_scope is False
    assert r.refusal_reason == "diagnosis_request"


def test_empty_question_returns_friendly_error():
    r = _run(compute_patient_faq(""))
    assert r.in_scope is False
    assert r.refusal_reason == "empty_question"


# ─────────────────────── In-scope deterministic templates ───────────────────────

def test_in_scope_question_gets_template_when_llm_disabled():
    r = _run(compute_patient_faq("Why am I taking warfarin?"))
    assert r.in_scope is True
    assert r.method == "deterministic_template"
    assert r.refusal_reason is None
    assert "discharge" in r.answer_text.lower() or \
           "summary" in r.answer_text.lower()


def test_lifestyle_question_routes_to_activities_template():
    r = _run(compute_patient_faq("Can I drive after this?"))
    assert r.in_scope is True
    assert "activit" in r.answer_text.lower() or \
           "self-care" in r.answer_text.lower()


def test_followup_question_routes_to_followup_template():
    r = _run(compute_patient_faq("When should I see my doctor next?"))
    assert r.in_scope is True
    assert "follow up" in r.answer_text.lower() or \
           "follow-up" in r.answer_text.lower() or \
           "primary care" in r.answer_text.lower()


# ─────────────────────── LLM happy path ───────────────────────

def test_llm_in_scope_answer_returned(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(
        mod, "_call_ollama",
        lambda _: "Warfarin helps prevent dangerous blood clots. Always "
                    "follow your written discharge plan; this does not replace it.",
    )
    r = _run(compute_patient_faq("Why am I taking warfarin?"))
    assert r.in_scope is True
    assert r.method == "llm"
    assert "blood clot" in r.answer_text.lower()


# ─────────────────────── LLM hardening: dose-change phrase blocking ───────────────────────

@pytest.mark.parametrize("malicious", [
    "You should take more warfarin tonight.",
    "Stop taking lisinopril if you feel okay.",
    "Skip the dose if you forget.",
    "Increase your metformin to 1000 mg.",
])
def test_llm_dose_change_phrases_blocked(monkeypatch, malicious):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama", lambda _: malicious)
    r = _run(compute_patient_faq("Why am I taking warfarin?"))
    assert r.method == "deterministic_template"
    assert any("dose_change_phrase_blocked" in w
                  for w in r.safety_warnings)


def test_llm_unavailable_records_safety_warning(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama", lambda _: None)
    r = _run(compute_patient_faq("Why am I taking warfarin?"))
    assert r.method == "deterministic_template"
    assert any("template_used" in w for w in r.safety_warnings)


# ─────────────────────── DecisionCard / counseling integration ───────────────────────

def test_decision_card_summary_propagated_to_prompt(monkeypatch):
    """When card is provided, the summary feeding the LLM contains
    risk + recommendation."""
    captured = {}

    def _capturing_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Generic answer respecting your plan."

    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama", _capturing_llm)
    card_dict = {
        "recommendation": {"action": "home_with_care", "confidence": "medium"},
        "reasoning": {"risk_estimate": {"probability_mean": 0.30}},
        "validation": {}, "audit": {},
    }
    _run(compute_patient_faq("When should I see my doctor next?",
                                  decision_card=card_dict))
    assert "home_with_care" in captured["prompt"]
    assert "0.3" in captured["prompt"] or "0.30" in captured["prompt"]


def test_counseling_summary_propagated_to_prompt(monkeypatch):
    captured = {}

    def _capturing_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(mod, "_call_ollama", _capturing_llm)
    counseling_dict = {
        "patient_id": "pt-1", "locale": "en", "reading_level_grade": 6,
        "sections": [{
            "section_id": "your_medications",
            "title": "Your medications",
            "plain_text": "You are starting warfarin to thin your blood.",
            "bullets": ["Take warfarin 5 mg daily."],
        }],
        "follow_up_window_days": (7, 14),
        "n_medications_explained": 1, "n_red_flags": 0,
    }
    _run(compute_patient_faq("When should I take warfarin?",
                                  counseling=counseling_dict))
    assert "warfarin" in captured["prompt"].lower()


# ─────────────────────── Citations ───────────────────────

def test_citations_attached_when_corpus_returns(monkeypatch):
    from shared.schemas import EvidenceSource
    fake_evidence = [
        EvidenceSource(
            source_type="guideline_passage",
            source_id="ACC-2014-1",
            excerpt="Warfarin requires regular INR monitoring.",
            relevance_score=0.85,
            recency_days=None,
        )
    ]
    monkeypatch.setattr(mod, "_retrieve_citations",
                          lambda q, k=2: fake_evidence)
    r = _run(compute_patient_faq("Why am I taking warfarin?"))
    assert len(r.citations) == 1
    assert r.citations[0].source_id == "ACC-2014-1"


def test_citations_optional_when_corpus_empty(monkeypatch):
    monkeypatch.setattr(mod, "_retrieve_citations", lambda q, k=2: [])
    r = _run(compute_patient_faq("Why am I taking warfarin?"))
    assert r.citations == []


# ─────────────────────── Bundle registration ───────────────────────

def test_faq_in_patient_facing_bundle():
    from mcp_server.tools import BUNDLES
    assert "compute_patient_faq" in BUNDLES["patient_facing"]
