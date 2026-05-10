"""Phase 6.2 -- Real LLM polish client.

Unified entry point for the LLM polish hooks scattered across the
patient-facing + scribe + PA tools.

Resolution order (first available wins):

  1. Ollama at `localhost:11434` (free local inference; default for
     dev with RTX 5090 / gpt-oss).
  2. Gemini API via `google-genai` when `GOOGLE_API_KEY` is set.
  3. Null client -- returns input unchanged with `model_id = None`. The
     deterministic floor remains the published output; nothing else
     downstream changes.

Every client honours `TRUSTEDRISK_DISABLE_LLM=1` (forces null) and
respects per-call timeouts.

The contract is intentionally narrow: each polish call takes the
deterministic source text + a system prompt, returns the polished text.
The hallucination harness (`tests/llm_integration/test_hallucination_harness.py`)
asserts that cite-back IDs + drug names + ICD codes + dose values
present in the source are preserved verbatim in the output. When the
preserved-token check fails, the polish is REJECTED and the
deterministic floor is used instead -- the LLM cannot remove
provenance information from the structured spine.
"""

from __future__ import annotations

import asyncio
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PolishResult:
    """Outcome of one polish call."""

    polished_text: str
    model_id: str | None
    is_polished: bool   # False when null client (or post-check rejection)
    polish_rejected_reason: str | None = None


def _disabled_via_env() -> bool:
    return os.environ.get("TRUSTEDRISK_DISABLE_LLM", "0").lower() in (
        "1", "true", "yes", "on",
    )


# ─────────────────────────────────────────────────────────────────────
# Cite-back / fact-preservation post-check
# ─────────────────────────────────────────────────────────────────────


_CITATION_PATTERNS = [
    # cite IDs that look like FHIR resource refs (e.g. cond-x, obs-y, med-z)
    re.compile(r"\((?:cite|cit|ref) [A-Za-z0-9_/\-]+\)", re.IGNORECASE),
    re.compile(r"cite [A-Za-z0-9_/\-]{2,}", re.IGNORECASE),
]
# ICD-10-CM patterns
_ICD10_PATTERN = re.compile(r"\b[A-TV-Z]\d{2}(?:\.\d{1,4})?\b")
# Drug-name preservation: extract any token that looks like a med (basic)
_DRUG_PATTERN = re.compile(
    r"\b(?:warfarin|apixaban|rivaroxaban|dabigatran|aspirin|clopidogrel|"
    r"furosemide|torsemide|spironolactone|lisinopril|enalapril|losartan|"
    r"metoprolol|atenolol|amlodipine|atorvastatin|rosuvastatin|simvastatin|"
    r"metformin|insulin|amiodarone|adalimumab|methotrexate|azathioprine|"
    r"ibuprofen|naproxen|tramadol|oxycodone|morphine|tylenol|acetaminophen|"
    r"penicillin|ceftriaxone|amoxicillin|piperacillin|vancomycin|"
    r"alteplase|aspirin)"
    r"(?:\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|mL|units|U))?", re.IGNORECASE,
)
# Dose preservation: numeric values + units
_DOSE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|mL|units|U|kg|cm)\b", re.IGNORECASE,
)


def find_preserved_tokens(source_text: str) -> set[str]:
    """Extract the union of tokens that the post-check considers
    'must-preserve': cite-backs, ICD codes, drug names, dose values."""
    tokens: set[str] = set()
    for pat in _CITATION_PATTERNS:
        tokens.update(m.group(0).lower() for m in pat.finditer(source_text))
    tokens.update(m.group(0).upper() for m in _ICD10_PATTERN.finditer(source_text))
    tokens.update(m.group(0).lower() for m in _DRUG_PATTERN.finditer(source_text))
    tokens.update(m.group(0).lower() for m in _DOSE_PATTERN.finditer(source_text))
    return tokens


def post_check_preserved_tokens(
    source: str, polished: str,
) -> tuple[bool, list[str]]:
    """Return (passed, missing_tokens). The polish is REJECTED if any
    must-preserve token from source is absent from polished."""
    expected = find_preserved_tokens(source)
    polished_lower = polished.lower()
    polished_upper = polished.upper()
    missing: list[str] = []
    for tok in expected:
        if tok.isupper():
            if tok not in polished_upper:
                missing.append(tok)
        else:
            if tok.lower() not in polished_lower:
                missing.append(tok)
    return (not missing, missing)


# ─────────────────────────────────────────────────────────────────────
# Client interface
# ─────────────────────────────────────────────────────────────────────


class LLMPolishClient(ABC):
    """Polish an existing deterministic text without changing facts."""

    model_id: str | None

    @abstractmethod
    async def polish(
        self,
        text: str,
        *,
        system_prompt: str | None = None,
        timeout_s: float = 30.0,
    ) -> PolishResult:
        ...


# ─────────────────────── Null client (default fallback) ───────────────────────


class NullPolishClient(LLMPolishClient):
    """Returns input unchanged. The hooks treat this as 'no polish'."""

    model_id: str | None = None

    async def polish(
        self, text: str, *, system_prompt: str | None = None,
        timeout_s: float = 30.0,
    ) -> PolishResult:
        return PolishResult(
            polished_text=text, model_id=None, is_polished=False,
            polish_rejected_reason="no_llm_configured",
        )


# ─────────────────────── Ollama client ───────────────────────


class OllamaPolishClient(LLMPolishClient):
    """Local Ollama inference via the `ollama` Python client.

    Tries `ollama.generate()`; on any failure falls back to None and
    the polish hook treats it as null.
    """

    def __init__(self, model: str = "llama3.1:8b"):
        self.model_id = model
        self._available: bool | None = None

    def _probe(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import ollama   # type: ignore
            # cheap availability check
            try:
                ollama.list()
                self._available = True
            except Exception:
                self._available = False
        except ImportError:
            self._available = False
        return self._available

    async def polish(
        self, text: str, *, system_prompt: str | None = None,
        timeout_s: float = 30.0,
    ) -> PolishResult:
        if not self._probe():
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason="ollama_unavailable",
            )
        try:
            import ollama   # type: ignore
            prompt = (
                (system_prompt or "")
                + "\n\nText to polish (preserve every clinical fact, "
                  "cite-back, drug name, dose, ICD code, exactly):\n\n"
                + text
            )
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: ollama.generate(
                        model=self.model_id, prompt=prompt,
                        options={"temperature": 0.2}),
                ),
                timeout=timeout_s,
            )
            polished = resp.get("response", "").strip() if isinstance(
                resp, dict) else getattr(resp, "response", "").strip()
        except asyncio.TimeoutError:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason="ollama_timeout",
            )
        except Exception as exc:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason=f"ollama_error: {type(exc).__name__}",
            )
        if not polished:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason="ollama_empty_response",
            )
        # Post-check: preserve all must-preserve tokens
        passed, missing = post_check_preserved_tokens(text, polished)
        if not passed:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason=(
                    f"hallucination_check_failed: missing tokens "
                    f"{missing[:5]!r}"
                ),
            )
        return PolishResult(
            polished_text=polished, model_id=f"ollama:{self.model_id}",
            is_polished=True, polish_rejected_reason=None,
        )


# ─────────────────────── Gemini client ───────────────────────


class GeminiPolishClient(LLMPolishClient):
    """Google Gemini via `google-genai`.

    Requires `GOOGLE_API_KEY`. Defaults to `gemini-2.0-flash-exp` for
    cost -- overridable via `TRUSTEDRISK_GEMINI_MODEL`.
    """

    def __init__(self, model: str | None = None):
        self.model_id = model or os.environ.get(
            "TRUSTEDRISK_GEMINI_MODEL", "gemini-2.0-flash-exp")
        self._available: bool | None = None

    def _probe(self) -> bool:
        if self._available is not None:
            return self._available
        if not os.environ.get("GOOGLE_API_KEY"):
            self._available = False
            return False
        try:
            from google import genai   # type: ignore
            self._client = genai.Client()
            self._available = True
        except Exception:
            self._available = False
        return self._available

    async def polish(
        self, text: str, *, system_prompt: str | None = None,
        timeout_s: float = 30.0,
    ) -> PolishResult:
        if not self._probe():
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason="gemini_unavailable",
            )
        try:
            from google import genai   # noqa: F401
            prompt = (
                (system_prompt or "")
                + "\n\nText to polish (preserve every clinical fact, "
                  "cite-back, drug name, dose, ICD code exactly):\n\n"
                + text
            )
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._client.models.generate_content(
                        model=self.model_id, contents=prompt),
                ),
                timeout=timeout_s,
            )
            polished = (getattr(resp, "text", "") or "").strip()
        except asyncio.TimeoutError:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason="gemini_timeout",
            )
        except Exception as exc:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason=f"gemini_error: {type(exc).__name__}",
            )
        if not polished:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason="gemini_empty_response",
            )
        passed, missing = post_check_preserved_tokens(text, polished)
        if not passed:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason=(
                    f"hallucination_check_failed: missing tokens "
                    f"{missing[:5]!r}"
                ),
            )
        return PolishResult(
            polished_text=polished, model_id=f"gemini:{self.model_id}",
            is_polished=True, polish_rejected_reason=None,
        )


# ─────────────────────── Anthropic client ───────────────────────


class AnthropicPolishClient(LLMPolishClient):
    """Anthropic Claude via the `anthropic` Python SDK.

    Requires `ANTHROPIC_API_KEY`. Defaults to a fast/cheap model id --
    overridable via `TRUSTEDRISK_ANTHROPIC_MODEL`. Temperature is
    forced to 0.0 to maximise deterministic paraphrase behaviour.
    """

    def __init__(self, model: str | None = None):
        self.model_id = model or os.environ.get(
            "TRUSTEDRISK_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001",
        )
        self._available: bool | None = None
        self._client = None

    def _probe(self) -> bool:
        if self._available is not None:
            return self._available
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self._available = False
            return False
        try:
            import anthropic    # type: ignore
            self._client = anthropic.Anthropic()
            self._available = True
        except Exception:
            self._available = False
        return self._available

    async def polish(
        self, text: str, *, system_prompt: str | None = None,
        timeout_s: float = 30.0,
    ) -> PolishResult:
        if not self._probe():
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason="anthropic_unavailable",
            )
        sys_prompt = (system_prompt or "").strip() or (
            "You are a clinical-documentation editor. Paraphrase to "
            "improve readability while preserving every clinical fact, "
            "cite-back, drug name, dose, and ICD code EXACTLY."
        )
        try:
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._client.messages.create(    # type: ignore[attr-defined]
                        model=self.model_id,
                        max_tokens=2048,
                        temperature=0.0,
                        system=sys_prompt,
                        messages=[{"role": "user", "content": text}],
                    ),
                ),
                timeout=timeout_s,
            )
            # SDK content is a list of blocks; extract text
            polished = ""
            for block in getattr(resp, "content", []) or []:
                t = getattr(block, "text", None)
                if isinstance(t, str):
                    polished += t
            polished = polished.strip()
        except asyncio.TimeoutError:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason="anthropic_timeout",
            )
        except Exception as exc:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason=(
                    f"anthropic_error: {type(exc).__name__}"
                ),
            )
        if not polished:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason="anthropic_empty_response",
            )
        passed, missing = post_check_preserved_tokens(text, polished)
        if not passed:
            return PolishResult(
                polished_text=text, model_id=None, is_polished=False,
                polish_rejected_reason=(
                    f"hallucination_check_failed: missing tokens "
                    f"{missing[:5]!r}"
                ),
            )
        return PolishResult(
            polished_text=polished, model_id=f"anthropic:{self.model_id}",
            is_polished=True, polish_rejected_reason=None,
        )


# ─────────────────────────────────────────────────────────────────────
# Resolver (cached)
# ─────────────────────────────────────────────────────────────────────


_CLIENT_CACHE: LLMPolishClient | None = None


def reset_client_cache() -> None:
    """For tests -- re-resolve on the next call."""
    global _CLIENT_CACHE
    _CLIENT_CACHE = None


def resolve_polish_client() -> LLMPolishClient:
    """Return the highest-priority available client. Cached after the
    first call."""
    global _CLIENT_CACHE
    if _CLIENT_CACHE is not None:
        return _CLIENT_CACHE
    if _disabled_via_env():
        _CLIENT_CACHE = NullPolishClient()
        return _CLIENT_CACHE

    # Priority 1: Anthropic (Claude). When ANTHROPIC_API_KEY is set this
    # is the production polisher -- its temperature-0 paraphrase pairs
    # well with the cite-back post-check.
    anthropic = AnthropicPolishClient()
    if anthropic._probe():
        _CLIENT_CACHE = anthropic
        return _CLIENT_CACHE

    # Priority 2: Ollama (local)
    ollama_model = os.environ.get(
        "TRUSTEDRISK_OLLAMA_MODEL", "llama3.1:8b")
    ollama = OllamaPolishClient(model=ollama_model)
    if ollama._probe():
        _CLIENT_CACHE = ollama
        return _CLIENT_CACHE

    # Priority 3: Gemini
    gemini = GeminiPolishClient()
    if gemini._probe():
        _CLIENT_CACHE = gemini
        return _CLIENT_CACHE

    # Fallback
    _CLIENT_CACHE = NullPolishClient()
    return _CLIENT_CACHE


# ─────────────────────────────────────────────────────────────────────
# Convenience: legacy resolver shim used by Phase 2 tools
# ─────────────────────────────────────────────────────────────────────


class _LegacyModelHandle:
    """Compat shim -- Phase 2 polish hooks call
    `from a2a_agent.llm_critic import _resolve_llm_model`. We expose a
    matching surface here so the hooks can be migrated incrementally
    without breaking older import paths.
    """
    def __init__(self, client: LLMPolishClient):
        self._client = client
        self.model_name = client.model_id


def _resolve_llm_model() -> _LegacyModelHandle | None:
    client = resolve_polish_client()
    if isinstance(client, NullPolishClient):
        return None
    return _LegacyModelHandle(client)
