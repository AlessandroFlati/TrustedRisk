"""COMPOSE-3 -- A2A agent registry.

A static-ish catalog of the 4 agents that compose the TrustedRisk
healthcare-AI stack:

  trustedrisk-agent        -- the upstream decision agent (38 tools, 13 bundles)
  darena-data-agent        -- federated data partner (FHIR ingest + handoff)
  trustedrisk-alert-agent  -- calibration drift webhook router
  trustedrisk-scheduler-agent -- discharge follow-up planner

The registry exposes each agent's logical id, role, configured URL, and the
path to its agent-card.json. The playground `/api/registry` endpoint
serializes this catalog and the `/api/orchestrate/...` endpoints use it to
locate the partner agents at runtime.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class AgentEntry:
    """One agent in the registry."""
    id: str
    role: str             # decision / data / alerting / scheduling
    description: str
    default_url: str
    env_var: str          # env var that overrides default_url
    agent_card_path: Path
    skills: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return os.environ.get(self.env_var) or self.default_url

    def load_agent_card(self) -> dict[str, Any]:
        if not self.agent_card_path.exists():
            return {}
        return json.loads(self.agent_card_path.read_text(encoding="utf-8"))


# Built-in registry -- extend by adding entries here.
_REGISTRY: list[AgentEntry] = [
    AgentEntry(
        id="trustedrisk-agent",
        role="decision",
        description=(
            "Upstream calibrated risk + safe-discharge agent. Owns the 38 "
            "MCP tools and 13 bundles, the 3-critic ensemble, and the "
            "deterministic safety gate. Other agents in this registry are "
            "downstream consumers."
        ),
        default_url="http://localhost:8765",
        env_var="TRUSTEDRISK_A2A_URL",
        agent_card_path=ROOT / "src" / "a2a_agent" / "agent-card.json",
        skills=["safe_discharge_review", "ed_acute_review",
                  "pediatric_review", "mental_health_review"],
    ),
    AgentEntry(
        id="darena-data-agent",
        role="data",
        description=(
            "Federated data partner that ingests FHIR Bundles from upstream "
            "EHRs and hands off decision questions to TrustedRisk via A2A "
            "capability negotiation."
        ),
        default_url="http://localhost:8766",
        env_var="DARENA_A2A_URL",
        agent_card_path=ROOT / "apps" / "federation_partner" / "agent_card.json",
        skills=["evaluate_patient_via_trustedrisk"],
    ),
    AgentEntry(
        id="trustedrisk-alert-agent",
        role="alerting",
        description=(
            "Subscribes to TrustedRisk's drift_monitor and emits webhook "
            "alerts when calibration tier escalates to warn/alert."
        ),
        default_url="http://localhost:8768",
        env_var="ALERT_A2A_URL",
        agent_card_path=ROOT / "apps" / "alert_agent" / "agent_card.json",
        skills=["subscribe_to_alerts", "check_drift_now",
                  "list_recent_alerts"],
    ),
    AgentEntry(
        id="trustedrisk-scheduler-agent",
        role="scheduling",
        description=(
            "Turns a TrustedRisk DecisionCard into a structured follow-up "
            "visit plan (PCP + specialist + modality + timing tier)."
        ),
        default_url="http://localhost:8769",
        env_var="SCHEDULER_A2A_URL",
        agent_card_path=ROOT / "apps" / "scheduler_agent" / "agent_card.json",
        skills=["propose_followup_visits", "list_recent_proposals"],
    ),
]


def list_agents() -> list[AgentEntry]:
    """Return the registry entries (read-only -- copies, not the live list)."""
    return list(_REGISTRY)


def find_agent(agent_id: str) -> AgentEntry | None:
    for entry in _REGISTRY:
        if entry.id == agent_id:
            return entry
    return None


def serialize_registry() -> dict[str, Any]:
    """Produce a JSON-friendly snapshot of the registry, including the
    embedded agent-card for each entry (when the file exists)."""
    items: list[dict[str, Any]] = []
    for entry in _REGISTRY:
        items.append({
            "id": entry.id,
            "role": entry.role,
            "description": entry.description,
            "url": entry.url,
            "default_url": entry.default_url,
            "env_var": entry.env_var,
            "skills": list(entry.skills),
            "agent_card": entry.load_agent_card(),
        })
    return {"agents": items, "n": len(items)}
