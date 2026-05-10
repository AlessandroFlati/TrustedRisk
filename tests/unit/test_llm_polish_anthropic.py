"""Phase 11.3 -- Anthropic polish client + cite-back enforcement."""

from __future__ import annotations

import asyncio

import pytest

from a2a_agent.llm_polish import (
    AnthropicPolishClient,
    PolishResult,
    find_preserved_tokens,
    post_check_preserved_tokens,
    reset_client_cache,
    resolve_polish_client,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# AnthropicPolishClient probe behaviour
# ─────────────────────────────────────────────────────────────────────

def test_anthropic_client_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicPolishClient()
    assert client._probe() is False


def test_anthropic_client_returns_null_polish_when_unavailable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicPolishClient()
    res = _run(client.polish("warfarin 5 mg daily"))
    assert res.is_polished is False
    assert res.polish_rejected_reason == "anthropic_unavailable"
    assert res.polished_text == "warfarin 5 mg daily"


def test_anthropic_default_model_id_present(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_ANTHROPIC_MODEL", raising=False)
    client = AnthropicPolishClient()
    assert client.model_id and "claude" in client.model_id


def test_anthropic_model_id_overridable_via_env(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_ANTHROPIC_MODEL", "claude-test-x")
    client = AnthropicPolishClient()
    assert client.model_id == "claude-test-x"


# ─────────────────────────────────────────────────────────────────────
# Resolver promotion to Anthropic when ANTHROPIC_API_KEY is set
# ─────────────────────────────────────────────────────────────────────

def test_resolver_picks_anthropic_when_key_set_and_sdk_available(
    monkeypatch,
):
    """When the SDK can't be imported the probe falls through; the test
    monkeypatches the probe to simulate a usable SDK."""
    reset_client_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.delenv("TRUSTEDRISK_DISABLE_LLM", raising=False)

    # Force the probe to return True without actually loading anthropic
    from a2a_agent import llm_polish as _llm
    monkeypatch.setattr(_llm.AnthropicPolishClient, "_probe",
                            lambda self: True)

    client = resolve_polish_client()
    assert isinstance(client, _llm.AnthropicPolishClient)
    reset_client_cache()


def test_disable_env_overrides_anthropic(monkeypatch):
    reset_client_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")

    from a2a_agent import llm_polish as _llm
    monkeypatch.setattr(_llm.AnthropicPolishClient, "_probe",
                            lambda self: True)

    client = resolve_polish_client()
    assert isinstance(client, _llm.NullPolishClient)
    reset_client_cache()


# ─────────────────────────────────────────────────────────────────────
# Cite-back / preserved-token post-check
# ─────────────────────────────────────────────────────────────────────

def test_preserved_tokens_capture_drug_dose_and_icd():
    src = (
        "Patient on warfarin 5 mg daily. Discharge dx I50.21 (cite cond-1)."
    )
    tokens = find_preserved_tokens(src)
    assert any("warfarin" in t for t in tokens)
    assert any("5 mg" in t for t in tokens)
    assert "I50.21" in tokens


def test_post_check_passes_when_all_tokens_preserved():
    src = "Take warfarin 5 mg daily. Diagnosis I50.21 (cite cond-1)."
    polished = (
        "Take warfarin 5 mg daily. Your diagnosis is I50.21 (cite cond-1)."
    )
    passed, missing = post_check_preserved_tokens(src, polished)
    assert passed
    assert missing == []


def test_post_check_rejects_when_drug_dropped():
    src = "Take warfarin 5 mg daily."
    polished = "Take your blood thinner daily."  # drug name gone
    passed, missing = post_check_preserved_tokens(src, polished)
    assert not passed
    assert any("warfarin" in m.lower() for m in missing)


def test_post_check_rejects_when_dose_dropped():
    src = "Take warfarin 5 mg daily."
    polished = "Take warfarin daily."  # dose missing
    passed, missing = post_check_preserved_tokens(src, polished)
    assert not passed
    assert any("5 mg" in m for m in missing)


def test_post_check_rejects_when_icd_dropped():
    src = "Discharge diagnosis I50.21."
    polished = "Discharge diagnosis: heart failure."
    passed, missing = post_check_preserved_tokens(src, polished)
    assert not passed
    assert "I50.21" in missing


# ─────────────────────────────────────────────────────────────────────
# discharge_counseling integration with the audited polish path
# ─────────────────────────────────────────────────────────────────────

def test_discharge_counseling_polish_falls_back_when_polish_rejected(
    monkeypatch,
):
    """A stub polish that drops the drug name must trigger the post-check
    rejection -- discharge_counseling falls back to the deterministic
    floor for every section."""
    monkeypatch.setenv("TRUSTEDRISK_COUNSELING_LLM_POLISH", "1")
    reset_client_cache()

    class _RejectingClient:
        model_id = "stub:rejecting"

        async def polish(self, text, *, system_prompt=None,
                                timeout_s=30.0):
            # Drop every preserved token by returning bland filler
            return PolishResult(
                polished_text="Take your medications as told.",
                model_id=None, is_polished=False,
                polish_rejected_reason="hallucination_check_failed",
            )

    from a2a_agent import llm_polish as _llm
    monkeypatch.setattr(_llm, "resolve_polish_client",
                            lambda: _RejectingClient())

    from mcp_server.tools.discharge_counseling import (
        compute_discharge_counseling,
    )

    out = _run(compute_discharge_counseling(
        medications=[
            {"name": "warfarin", "dose_mg": 5.0, "status": "active"},
        ],
        lace_score=8,
        recommendation_action="discharge_home",
    ))
    # Floor preserves the structured drug name in the bullets
    med_section = next(
        s for s in out.sections if s.section_id == "your_medications"
    )
    assert any("warfarin" in b.lower() for b in med_section.bullets)
    reset_client_cache()


def test_discharge_counseling_polish_applied_when_post_check_passes(
    monkeypatch,
):
    """A stub polish that keeps the structured tokens passes the
    post-check; the polished prose replaces the floor's plain_text."""
    monkeypatch.setenv("TRUSTEDRISK_COUNSELING_LLM_POLISH", "1")
    reset_client_cache()

    class _PassingClient:
        model_id = "stub:passing"

        async def polish(self, text, *, system_prompt=None,
                                timeout_s=30.0):
            # Append a benign phrase but keep ALL preserved tokens
            polished_text = text + " Stay safe."
            return PolishResult(
                polished_text=polished_text,
                model_id="stub:passing", is_polished=True,
                polish_rejected_reason=None,
            )

    from a2a_agent import llm_polish as _llm
    monkeypatch.setattr(_llm, "resolve_polish_client",
                            lambda: _PassingClient())

    from mcp_server.tools.discharge_counseling import (
        compute_discharge_counseling,
    )

    out = _run(compute_discharge_counseling(
        medications=[
            {"name": "warfarin", "dose_mg": 5.0, "status": "active"},
        ],
        lace_score=8,
        recommendation_action="discharge_home",
    ))
    # The polished prose must contain the appended marker; bullets stay
    assert any("Stay safe" in s.plain_text for s in out.sections)
    reset_client_cache()


# ─────────────────────────────────────────────────────────────────────
# Sanity: existing polish-disabled path stays untouched
# ─────────────────────────────────────────────────────────────────────

def test_discharge_counseling_no_polish_when_flag_off(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_COUNSELING_LLM_POLISH", raising=False)
    reset_client_cache()

    from mcp_server.tools.discharge_counseling import (
        compute_discharge_counseling,
    )
    out = _run(compute_discharge_counseling(
        medications=[
            {"name": "warfarin", "dose_mg": 5.0, "status": "active"},
        ],
        lace_score=8,
        recommendation_action="discharge_home",
    ))
    # Floor produces deterministic prose without any "Stay safe" suffix
    assert all("Stay safe" not in s.plain_text for s in out.sections)
    reset_client_cache()
