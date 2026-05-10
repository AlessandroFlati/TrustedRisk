"""Unit tests for A2A agent model resolution (Gemini vs LiteLlm/Ollama)."""
from __future__ import annotations

import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")

# Ensure src is on path before importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from a2a_agent.agent import _resolve_model


def test_resolve_model_gemini_string_passthrough():
    """Gemini IDs are passed as plain strings (ADK consumes them directly)."""
    assert _resolve_model("gemini-2.5-flash") == "gemini-2.5-flash"


def test_resolve_model_models_prefix_passthrough():
    assert _resolve_model("models/gemini-1.5-pro") == "models/gemini-1.5-pro"


def test_resolve_model_no_slash_passthrough():
    assert _resolve_model("custom-managed-model") == "custom-managed-model"


def test_resolve_model_ollama_wraps_in_litellm():
    from google.adk.models.lite_llm import LiteLlm

    result = _resolve_model("ollama_chat/qwen2.5:7b-instruct")
    assert isinstance(result, LiteLlm)


def test_resolve_model_openai_wraps_in_litellm():
    from google.adk.models.lite_llm import LiteLlm

    result = _resolve_model("openai/gpt-4o-mini")
    assert isinstance(result, LiteLlm)


def test_resolve_model_anthropic_wraps_in_litellm():
    from google.adk.models.lite_llm import LiteLlm

    result = _resolve_model("anthropic/claude-3-5-haiku-latest")
    assert isinstance(result, LiteLlm)
