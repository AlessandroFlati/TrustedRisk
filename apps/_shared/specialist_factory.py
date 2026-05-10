"""Specialist factory — Phase 1 federation.

Each TrustedRisk specialist (`trustedrisk-discharge`, `…-acute`,
`…-evidence`, `…-population`, `…-pediatric`) is a marketplace-publishable
A2A agent that exposes a focused skill catalog while sharing the same
underlying MCP backend (all 58 tools + the SHARP middleware).

Architecture:

    user query
        │
        ▼
    BYO orchestrator (A2A v1 chat client)
        │  (consult dropdown)
        ▼
    Specialist agent-card
        │  (focused skills)
        ▼
    Shared MCP backend
        │  (full 58 tools, SHARP middleware, OAuth, audit)
        ▼
    FHIR + LLM

The factory parameterises only what differs across specialists: the
`agent-card.json` file (focused skill catalog), the listening port,
and the human-readable agent name. Everything else (tool registration,
middleware order, audit, multi-tenant, refresh token) is shared.

Usage:

    from apps._shared import build_specialist_app, SpecialistConfig

    app = build_specialist_app(SpecialistConfig(
        name="trustedrisk-discharge",
        port=8770,
        agent_card_path="apps/specialist_discharge/agent_card.json",
    ))

The returned ASGI app is consumable by uvicorn / gunicorn / Cloud Run
exactly like the main MCP server.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mcp_server.server import build_http_app


@dataclass(frozen=True)
class SpecialistConfig:
    """Configuration for a single TrustedRisk specialist.

    Attributes:
        name: Human-readable agent name (also used in logs / audit).
        port: Default listening port (overridable by uvicorn args).
        agent_card_path: Path to the specialist's `agent_card.json`
            (relative to the repo root or absolute). The file must be
            an A2A v1 card declaring the FHIR-context extension URI.
        marketplace_path: Optional override for the marketplace.json
            file. Defaults to the main one if absent.
    """
    name: str
    port: int
    agent_card_path: str
    marketplace_path: str | None = None


def build_specialist_app(cfg: SpecialistConfig):
    """Build a Starlette ASGI app for a specialist.

    The app shares the full MCP tool surface + middleware stack with
    the main `trustedrisk-agent`; only the published agent-card differs.

    The agent-card path must exist — failure is loud, not silent: a
    specialist with a broken card cannot register on the marketplace.
    """
    card_path = Path(cfg.agent_card_path)
    if not card_path.is_absolute():
        # Resolve relative to repo root (parent of the apps directory)
        repo_root = Path(__file__).resolve().parent.parent.parent
        card_path = repo_root / cfg.agent_card_path
    if not card_path.exists():
        raise FileNotFoundError(
            f"Specialist {cfg.name!r}: agent-card file not found at "
            f"{card_path!s}. Every specialist MUST publish a valid "
            f"A2A v1 agent card."
        )

    marketplace_arg: str | None = None
    if cfg.marketplace_path:
        mp_path = Path(cfg.marketplace_path)
        if not mp_path.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent.parent
            mp_path = repo_root / cfg.marketplace_path
        marketplace_arg = str(mp_path)

    return build_http_app(
        agent_card_path=str(card_path),
        marketplace_path=marketplace_arg,
    )
