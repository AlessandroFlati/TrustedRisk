"""TrustedRisk A2A agent composer -- Option C hybrid.

Structure:
  root_agent (LlmAgent)
    ├─ decide_then_validate (SequentialAgent)
    │    ├─ decide_agent (LlmAgent: readmission_risk + decision_utility)
    │    └─ validate_agent (LlmAgent: ground_claim + detect_phi)
    └─ validate_agent (LlmAgent, same sub-agent as above for PHI-only path)

Routing rules (encoded in root_agent instructions.md):
  - Clinical decision request  -> decide_then_validate (both tools ALWAYS run)
  - PHI-only check             -> validate_agent directly
  - Out-of-scope request       -> refuse with explanation
"""

from __future__ import annotations

import os
from pathlib import Path


_INSTRUCTIONS_PATH = Path(__file__).parent / "instructions.md"


def _load_instructions() -> str:
    if not _INSTRUCTIONS_PATH.exists():
        return ""
    return _INSTRUCTIONS_PATH.read_text(encoding="utf-8")


def _resolve_model(model_id: str):
    """Resolve `ADK_MODEL` to either a Gemini string ID or a LiteLlm wrapper.

    Plain strings like "gemini-2.5-flash" go to Vertex/Gemini directly.
    Strings prefixed with "ollama_chat/" or "openai/" or "anthropic/" etc.
    (any LiteLLM-recognized prefix) get wrapped in `LiteLlm`. This lets the
    deployer pick a local Ollama model without code changes -- e.g.
    `ADK_MODEL=ollama_chat/qwen2.5:32b-instruct`.
    """
    if "/" not in model_id or model_id.startswith("gemini") or model_id.startswith("models/"):
        return model_id
    try:
        from google.adk.models.lite_llm import LiteLlm  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "ADK_MODEL is set to a non-Gemini provider but LiteLLM extras are "
            "not installed. Run `pip install google-adk[extensions]`."
        ) from e
    return LiteLlm(model=model_id)


def _build_agent():
    """Build the hybrid agent. Lazy-imports ADK to avoid hard dependency at import time."""
    try:
        from google.adk.agents import LlmAgent, SequentialAgent  # type: ignore
        from google.adk.tools.mcp_tool import MCPToolset  # type: ignore
        from google.adk.tools.mcp_tool.mcp_session_manager import (  # type: ignore
            StreamableHTTPConnectionParams,
        )
    except ImportError as e:
        raise RuntimeError(
            "google-adk not installed. Install with `pip install google-adk>=1.31`."
        ) from e

    model_id = os.environ.get("ADK_MODEL", "gemini-2.5-flash")
    model = _resolve_model(model_id)

    # MCPToolset -- points at our own MCP server. ADK >= 1.31 uses
    # connection_params (StreamableHTTPConnectionParams) instead of from_url().
    mcp_url = os.environ.get(
        "TRUSTEDRISK_MCP_URL",
        f"http://localhost:{os.environ.get('TRUSTEDRISK_PORT', '8080')}/mcp",
    )
    mcp_headers = {
        "X-FHIR-Server-URL": os.environ.get("DEFAULT_FHIR_URL", ""),
        "X-FHIR-Access-Token": os.environ.get("DEFAULT_FHIR_TOKEN", ""),
    }
    decide_toolset = MCPToolset(
        connection_params=StreamableHTTPConnectionParams(url=mcp_url, headers=mcp_headers),
        tool_filter=[
            "compute_readmission_risk",
            "compute_decision_utility",
            "compute_medication_reconciliation",
            "detect_polypharmacy_concerns",
            "compute_fairness_audit",
        ],
    )
    validate_toolset = MCPToolset(
        connection_params=StreamableHTTPConnectionParams(url=mcp_url, headers=mcp_headers),
        tool_filter=["ground_claim", "detect_phi"],
    )

    validate_instruction = (
        "You validate a clinical claim or text. For clinical claims, call "
        "ground_claim. For arbitrary text (e.g., clinical notes), call "
        "detect_phi. Return the raw tool outputs."
    )

    decide_agent = LlmAgent(
        name="decide",
        model=model,
        instruction=(
            "You produce the clinical core of a discharge recommendation. "
            "Call compute_readmission_risk first to get the probability estimate, "
            "then compute_decision_utility to score discharge actions. When the "
            "patient has any structured medications listed, also call "
            "compute_medication_reconciliation. When the patient has demographics "
            "(age, race, insurance), call compute_fairness_audit. Return the "
            "raw tool outputs -- do not editorialize."
        ),
        tools=[decide_toolset],
    )

    # ADK >= 1.31 enforces single-parent for sub_agents, so the validate role
    # needs two independent instances: one inside decide_then_validate, and
    # one as a peer of decide_then_validate under root.
    validate_agent_in_seq = LlmAgent(
        name="validate_in_seq",
        model=model,
        instruction=validate_instruction,
        tools=[validate_toolset],
    )
    validate_agent_standalone = LlmAgent(
        name="validate",
        model=model,
        instruction=validate_instruction,
        tools=[validate_toolset],
    )

    # Self-critique agent -- no tools, only structural review of the candidate
    # DecisionCard. Emits a CritiqueDecision JSON. The root agent then calls
    # apply_critique() (in critique.py) to translate the verdict into edits.
    from .critique import CRITIC_INSTRUCTION
    critic_agent = LlmAgent(
        name="self_critique",
        model=model,
        instruction=CRITIC_INSTRUCTION,
        tools=[],
    )

    decide_then_validate_then_critique = SequentialAgent(
        name="decide_then_validate_then_critique",
        sub_agents=[decide_agent, validate_agent_in_seq, critic_agent],
    )

    root = LlmAgent(
        name="trustedrisk_root",
        model=model,
        instruction=_load_instructions(),
        sub_agents=[decide_then_validate_then_critique, validate_agent_standalone],
    )
    return root


# Lazy construction: the agent is only built when accessed
_ROOT_AGENT = None


def root_agent():
    """Return the root LlmAgent, building it lazily on first access."""
    global _ROOT_AGENT
    if _ROOT_AGENT is None:
        _ROOT_AGENT = _build_agent()
    return _ROOT_AGENT


def to_a2a_app():
    """Wrap the root agent as an A2A app (exposable via Uvicorn)."""
    try:
        from google.adk.a2a import to_a2a  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-adk A2A helper not available; update to adk>=2.0."
        ) from e
    return to_a2a(root_agent())
