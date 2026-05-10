"""Phase 6.2 -- LLM polish client unit tests.

Verifies the resolution chain (Ollama -> Gemini -> Null), the
hallucination post-check (cite-back / drug / ICD / dose preservation),
and the integration with the scribe `llm_polish_sections` async hook.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from a2a_agent.llm_polish import (
    GeminiPolishClient,
    LLMPolishClient,
    NullPolishClient,
    OllamaPolishClient,
    PolishResult,
    find_preserved_tokens,
    post_check_preserved_tokens,
    reset_client_cache,
    resolve_polish_client,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Token extraction ───────────────────────


def test_finds_cite_back_tokens():
    text = "Continue furosemide 40 mg PO daily (cite med-furo)."
    tokens = find_preserved_tokens(text)
    assert any("cite med-furo" in t for t in tokens)


def test_finds_icd10_codes():
    text = "Patient has I50.21 acute on chronic systolic CHF."
    tokens = find_preserved_tokens(text)
    assert any("I50.21" == t for t in tokens)


def test_finds_drug_names():
    text = "Continue warfarin 5 mg and aspirin 81 mg daily."
    tokens = find_preserved_tokens(text)
    assert any("warfarin" in t.lower() for t in tokens)
    assert any("aspirin" in t.lower() for t in tokens)


def test_finds_dose_values():
    text = "Furosemide 40 mg PO daily."
    tokens = find_preserved_tokens(text)
    assert any("40 mg" in t.lower() for t in tokens)


def test_post_check_passes_when_polished_preserves_everything():
    src = "Continue furosemide 40 mg PO daily (cite med-furo) for I50.21."
    polished = (
        "Maintain furosemide 40 mg PO daily (cite med-furo) for "
        "I50.21 acute on chronic systolic CHF."
    )
    passed, missing = post_check_preserved_tokens(src, polished)
    assert passed is True
    assert missing == []


def test_post_check_fails_when_polished_drops_drug_name():
    src = "Continue warfarin 5 mg daily (cite med-warf) for AFib."
    polished = "Continue blood thinner daily (cite med-warf) for AFib."
    passed, missing = post_check_preserved_tokens(src, polished)
    assert passed is False
    assert any("warfarin" in t.lower() for t in missing)


def test_post_check_fails_when_polished_drops_icd10():
    src = "Patient with I50.21 admitted today."
    polished = "Patient with heart failure admitted today."
    passed, missing = post_check_preserved_tokens(src, polished)
    assert passed is False
    assert "I50.21" in missing


def test_post_check_fails_when_polished_drops_cite_back():
    src = "Furosemide (cite med-furo) for diuresis."
    polished = "Furosemide for diuresis."
    passed, missing = post_check_preserved_tokens(src, polished)
    assert passed is False


# ─────────────────────── Null client ───────────────────────


def test_null_client_returns_input_unchanged():
    c = NullPolishClient()
    res = _run(c.polish("hello world"))
    assert isinstance(res, PolishResult)
    assert res.polished_text == "hello world"
    assert res.is_polished is False
    assert res.model_id is None


# ─────────────────────── Resolution ───────────────────────


def test_resolver_returns_null_when_disabled(monkeypatch):
    reset_client_cache()
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")
    c = resolve_polish_client()
    assert isinstance(c, NullPolishClient)


def test_resolver_falls_back_to_null_without_ollama_or_gemini(monkeypatch):
    reset_client_cache()
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    # Force OllamaPolishClient._probe to fail by patching its import path
    monkeypatch.setattr(
        "a2a_agent.llm_polish.OllamaPolishClient._probe",
        lambda self: False,
    )
    monkeypatch.setattr(
        "a2a_agent.llm_polish.GeminiPolishClient._probe",
        lambda self: False,
    )
    c = resolve_polish_client()
    assert isinstance(c, NullPolishClient)


def test_resolver_uses_ollama_when_available(monkeypatch):
    reset_client_cache()
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)
    monkeypatch.setattr(
        "a2a_agent.llm_polish.OllamaPolishClient._probe",
        lambda self: True,
    )
    c = resolve_polish_client()
    assert isinstance(c, OllamaPolishClient)


# ─────────────────────── Mocked polish hook ───────────────────────


class _FakeClient(LLMPolishClient):
    """Test double that returns the configured polished text."""
    model_id = "fake:1.0"

    def __init__(self, polished: str):
        self._polished = polished

    async def polish(
        self, text: str, *, system_prompt: str | None = None,
        timeout_s: float = 30.0,
    ) -> PolishResult:
        passed, _ = post_check_preserved_tokens(text, self._polished)
        if not passed:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason="hallucination_check_failed",
            )
        return PolishResult(
            polished_text=self._polished, model_id=self.model_id,
            is_polished=True,
        )


def test_scribe_polish_uses_async_client_and_preserves_cite_backs(monkeypatch):
    reset_client_cache()
    src_section = "Continue furosemide 40 mg (cite med-furo)."
    polished_text = "Maintain furosemide 40 mg (cite med-furo) for diuresis."
    monkeypatch.setattr(
        "a2a_agent.llm_polish.resolve_polish_client",
        lambda: _FakeClient(polished_text),
    )

    from mcp_server.tools._scribe_helpers import llm_polish_sections
    from shared.schemas import ClinicalNoteSection

    sections = [
        ClinicalNoteSection(
            section_id="medications", title="Medications",
            body=src_section,
            cited_evidence_ids=["med-furo"],
        ),
    ]
    polished, model_id = _run(llm_polish_sections(sections))
    assert model_id == "fake:1.0"
    assert polished[0].body == polished_text
    assert polished[0].is_llm_polished is True
    # Cite-back IDs preserved verbatim
    assert polished[0].cited_evidence_ids == ["med-furo"]


def test_scribe_polish_rejects_when_hallucination_check_fails(monkeypatch):
    reset_client_cache()
    src_section = "Continue furosemide 40 mg (cite med-furo)."
    bad_polished = "Maintain water pill daily."   # drops drug name + cite
    monkeypatch.setattr(
        "a2a_agent.llm_polish.resolve_polish_client",
        lambda: _FakeClient(bad_polished),
    )

    from mcp_server.tools._scribe_helpers import llm_polish_sections
    from shared.schemas import ClinicalNoteSection

    sections = [
        ClinicalNoteSection(
            section_id="medications", title="Medications",
            body=src_section,
            cited_evidence_ids=["med-furo"],
        ),
    ]
    polished, model_id = _run(llm_polish_sections(sections))
    # Polish was REJECTED -> deterministic body preserved
    assert polished[0].body == src_section
    assert polished[0].is_llm_polished is False
