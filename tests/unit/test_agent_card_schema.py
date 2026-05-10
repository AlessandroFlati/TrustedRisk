"""INT-3 -- A2A AgentCard schema validation.

Validates every agent-card.json in the project against a structural
schema that mirrors the public A2A AgentCard v1.0.0 spec
(github.com/google/A2A -- schema is intentionally minimal: name + version +
url + skills are the must-have fields, with skills carrying id/name/
description). Custom extensions (`_bundles`, `_role`) are tolerated as
underscore-prefixed conventions.

We don't pull the schema over the network during tests -- the contract
is encoded as Pydantic models below so any drift in our cards (or in
upstream spec) flags here first.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────── Pydantic mirror of A2A AgentCard v1.0.0 ───────────────────────

class A2ASkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def id_is_safe(cls, v: str) -> str:
        if not v or " " in v:
            raise ValueError(f"skill.id must be non-empty + space-free: {v!r}")
        return v


class A2AExtensionDeclaration(BaseModel):
    uri: str
    description: str | None = None
    required: bool = False
    params: dict | None = None

    class Config:
        extra = "allow"


class A2ACapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False
    extendedAgentCard: bool = False
    extensions: list[A2AExtensionDeclaration] = Field(default_factory=list)

    class Config:
        extra = "allow"


class A2ASupportedInterface(BaseModel):
    url: str
    protocolBinding: str = "A2A"
    protocolVersion: str = "1.0"

    class Config:
        extra = "allow"

    @field_validator("url")
    @classmethod
    def url_resolvable(cls, v: str) -> str:
        if not (v.startswith(("http://", "https://")) or v.startswith("<")):
            raise ValueError(f"supportedInterfaces[].url must be http(s) "
                              f"or placeholder: {v!r}")
        return v


class A2AAgentCard(BaseModel):
    """A2A AgentCard -- accepts both pre-v1 (`url`) and v1
    (`supportedInterfaces[]`) shapes. At least one of the two MUST be
    present; partner cards may still be on the legacy `url`-only shape
    while the main TrustedRisk card carries both during the v0->v1
    transition.
    """
    schemaVersion: str
    name: str
    description: str
    version: str
    # Either `url` (legacy, deprecated in v1) or `supportedInterfaces[]`
    # (v1) -- at least one must be present.
    url: str | None = None
    supportedInterfaces: list[A2ASupportedInterface] | None = None
    defaultInputModes: list[str]
    defaultOutputModes: list[str]
    skills: list[A2ASkill]
    capabilities: A2ACapabilities | None = None
    securitySchemes: dict | None = None

    class Config:
        # Permit custom underscore-prefixed extensions (e.g. _bundles)
        extra = "allow"

    @field_validator("schemaVersion")
    @classmethod
    def schema_v1_or_later(cls, v: str) -> str:
        if not v.startswith(("1.", "2.")):
            raise ValueError(f"schemaVersion must be 1.x or 2.x, got {v!r}")
        return v

    @field_validator("url")
    @classmethod
    def url_must_be_resolvable(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not (v.startswith(("http://", "https://"))
                or v.startswith("<")):  # placeholder URLs accepted in dev
            raise ValueError(f"url must be http(s) or placeholder: {v!r}")
        return v

    @field_validator("defaultInputModes", "defaultOutputModes")
    @classmethod
    def modes_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("default*Modes must be non-empty")
        return v

    @field_validator("skills")
    @classmethod
    def at_least_one_skill(cls, v: list[A2ASkill]) -> list[A2ASkill]:
        if not v:
            raise ValueError("skills must not be empty")
        return v

    @model_validator(mode="after")
    def _at_least_one_endpoint(self) -> "A2AAgentCard":
        if not self.url and not self.supportedInterfaces:
            raise ValueError(
                "Agent card must declare either `url` (pre-v1) or "
                "`supportedInterfaces[]` (v1) -- both are absent."
            )
        return self


# ─────────────────────── The agent cards we ship ───────────────────────

_CARD_PATHS = [
    PROJECT_ROOT / "src" / "a2a_agent" / "agent-card.json",
    PROJECT_ROOT / "apps" / "federation_partner" / "agent_card.json",
    PROJECT_ROOT / "apps" / "alert_agent" / "agent_card.json",
    PROJECT_ROOT / "apps" / "scheduler_agent" / "agent_card.json",
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ─────────────────────── Per-card validation ───────────────────────

@pytest.mark.parametrize("card_path", _CARD_PATHS,
                            ids=lambda p: p.parent.name)
def test_agent_card_validates_against_a2a_schema(card_path: Path):
    """Every shipped agent-card must validate against the A2A v1.0.0 schema."""
    if not card_path.exists():
        pytest.skip(f"Agent card not found at {card_path}")
    raw = _load(card_path)
    A2AAgentCard.model_validate(raw)


@pytest.mark.parametrize("card_path", _CARD_PATHS,
                            ids=lambda p: p.parent.name)
def test_agent_card_skill_ids_unique(card_path: Path):
    if not card_path.exists():
        pytest.skip(f"Agent card not found at {card_path}")
    raw = _load(card_path)
    skill_ids = [s["id"] for s in raw["skills"]]
    assert len(skill_ids) == len(set(skill_ids)), \
        f"Duplicate skill ids in {card_path.name}: {skill_ids}"


@pytest.mark.parametrize("card_path", _CARD_PATHS,
                            ids=lambda p: p.parent.name)
def test_agent_card_has_at_least_one_example_per_skill(card_path: Path):
    if not card_path.exists():
        pytest.skip(f"Agent card not found at {card_path}")
    raw = _load(card_path)
    for skill in raw["skills"]:
        assert isinstance(skill.get("examples", []), list)
        # Examples are highly recommended for marketplace discoverability
        # -- assert ≥ 1 for non-trivial agents.
        if skill["id"] != "phi_check":  # phi_check is a single-shot utility
            assert len(skill.get("examples", [])) >= 1, \
                f"{card_path.parent.name}/skill {skill['id']!r} has no examples"


# ─────────────────────── Adversarial: malformed inputs rejected ───────────────────────

def test_missing_required_field_rejected():
    bad = {"name": "x", "version": "0.1.0", "url": "http://x",
              "defaultInputModes": ["text"],
              "defaultOutputModes": ["json"], "skills": []}
    # Missing schemaVersion + description
    with pytest.raises(ValidationError):
        A2AAgentCard.model_validate(bad)


def test_empty_skills_rejected():
    bad = {
        "schemaVersion": "1.0.0", "name": "x", "description": "y",
        "version": "0.1.0", "url": "http://x",
        "defaultInputModes": ["text"], "defaultOutputModes": ["json"],
        "skills": [],
    }
    with pytest.raises(ValidationError):
        A2AAgentCard.model_validate(bad)


def test_old_schema_version_rejected():
    bad = {
        "schemaVersion": "0.4.0", "name": "x", "description": "y",
        "version": "0.1.0", "url": "http://x",
        "defaultInputModes": ["text"], "defaultOutputModes": ["json"],
        "skills": [{"id": "x", "name": "y", "description": "z"}],
    }
    with pytest.raises(ValidationError):
        A2AAgentCard.model_validate(bad)


def test_skill_id_with_spaces_rejected():
    bad = {
        "schemaVersion": "1.0.0", "name": "x", "description": "y",
        "version": "0.1.0", "url": "http://x",
        "defaultInputModes": ["text"], "defaultOutputModes": ["json"],
        "skills": [{"id": "has space", "name": "y", "description": "z"}],
    }
    with pytest.raises(ValidationError):
        A2AAgentCard.model_validate(bad)


# ─────────────────────── TrustedRisk-specific assertions ───────────────────────

def test_trustedrisk_card_has_safe_discharge_skill():
    raw = _load(PROJECT_ROOT / "src" / "a2a_agent" / "agent-card.json")
    skill_ids = {s["id"] for s in raw["skills"]}
    assert "safe_discharge_review" in skill_ids


def test_trustedrisk_card_lists_bundles_extension():
    """Our `_bundles` extension is preserved on round-trip parse."""
    raw = _load(PROJECT_ROOT / "src" / "a2a_agent" / "agent-card.json")
    parsed = A2AAgentCard.model_validate(raw)
    payload = parsed.model_dump()
    assert "_bundles" in payload
    assert "core_discharge" in payload["_bundles"]


def test_partner_agents_distinct_urls():
    """All 4 cards must declare distinct URLs (otherwise discovery collides)."""
    urls = []
    for p in _CARD_PATHS:
        if p.exists():
            raw = _load(p)
            # v1 supportedInterfaces[] preferred; pre-v1 `url` fallback
            interfaces = raw.get("supportedInterfaces") or []
            if interfaces:
                urls.append(interfaces[0]["url"])
            elif raw.get("url"):
                urls.append(raw["url"])
    assert len(urls) == len(set(urls)), \
        f"Agent cards collide on URL: {urls}"


# ─────────────────────── A2A v1 + Prompt Opinion extension ───────────────────────

def test_v1_supportedInterfaces_only_card_validates():
    """A pure-v1 card (no `url`, only `supportedInterfaces[]`) is valid."""
    raw = {
        "schemaVersion": "1.0.0", "name": "x", "description": "y",
        "version": "0.1.0",
        "supportedInterfaces": [
            {"url": "https://example.com/a2a/x",
             "protocolBinding": "A2A", "protocolVersion": "1.0"}
        ],
        "defaultInputModes": ["text"], "defaultOutputModes": ["json"],
        "skills": [{"id": "x", "name": "y", "description": "z"}],
    }
    A2AAgentCard.model_validate(raw)


def test_v0_url_only_card_validates():
    """A pre-v1 card (only `url`, no `supportedInterfaces`) is still valid
    during the transition window -- partner cards may not have migrated yet."""
    raw = {
        "schemaVersion": "1.0.0", "name": "x", "description": "y",
        "version": "0.1.0", "url": "http://x.example/a2a",
        "defaultInputModes": ["text"], "defaultOutputModes": ["json"],
        "skills": [{"id": "x", "name": "y", "description": "z"}],
    }
    A2AAgentCard.model_validate(raw)


def test_card_without_url_and_supportedInterfaces_rejected():
    """At least one of the two must be present."""
    raw = {
        "schemaVersion": "1.0.0", "name": "x", "description": "y",
        "version": "0.1.0",
        "defaultInputModes": ["text"], "defaultOutputModes": ["json"],
        "skills": [{"id": "x", "name": "y", "description": "z"}],
    }
    with pytest.raises(ValidationError):
        A2AAgentCard.model_validate(raw)


def test_trustedrisk_card_declares_prompt_opinion_fhir_extension():
    """The main agent card must declare the PO FHIR-context extension URI
    so the marketplace knows TrustedRisk consumes FHIR via A2A metadata."""
    raw = _load(PROJECT_ROOT / "src" / "a2a_agent" / "agent-card.json")
    parsed = A2AAgentCard.model_validate(raw)
    extensions = (parsed.capabilities.extensions
                       if parsed.capabilities else [])
    uris = {e.uri for e in extensions}
    assert ("https://app.promptopinion.ai/schemas/a2a/v1/fhir-context"
            in uris), f"PO FHIR-context extension URI missing; got {uris}"


def test_trustedrisk_card_declares_supportedInterfaces():
    """The main agent card must carry the A2A v1 `supportedInterfaces[]`
    field -- otherwise marketplace discovery falls back to the deprecated
    `url`."""
    raw = _load(PROJECT_ROOT / "src" / "a2a_agent" / "agent-card.json")
    parsed = A2AAgentCard.model_validate(raw)
    assert parsed.supportedInterfaces, \
        "Main card must declare `supportedInterfaces[]` (A2A v1)"


def test_trustedrisk_card_lists_all_bundles():
    """The `_bundles` extension on the main card must enumerate every
    bundle in `mcp_server.tools.BUNDLES`. Currently 21 (12 clinical-
    vertical + 8 cross-cutting + 1 Phase-2 pain-point)."""
    raw = _load(PROJECT_ROOT / "src" / "a2a_agent" / "agent-card.json")
    bundles = raw.get("_bundles", {})
    bundle_keys = {k for k in bundles if not k.startswith("_")}

    # Sync against the runtime BUNDLES dict (single source of truth)
    import sys
    SRC = PROJECT_ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mcp_server.tools import BUNDLES   # type: ignore
    runtime_keys = set(BUNDLES.keys())

    assert bundle_keys == runtime_keys, (
        f"agent-card._bundles drift: card={bundle_keys}, "
        f"runtime BUNDLES={runtime_keys}"
    )


def test_trustedrisk_card_security_schemes_v1_typed_keys():
    """A2A v1 `securitySchemes` uses nested typed-key shapes:
    `{<name>: {oauth2SecurityScheme | apiKeySecurityScheme | ...: {...}}}`.
    """
    raw = _load(PROJECT_ROOT / "src" / "a2a_agent" / "agent-card.json")
    schemes = raw.get("securitySchemes", {})
    assert schemes, "securitySchemes must be present on the main card"
    for name, decl in schemes.items():
        assert isinstance(decl, dict)
        # Exactly one *SecurityScheme key must wrap the type
        typed_keys = [k for k in decl if k.endswith("SecurityScheme")]
        assert len(typed_keys) == 1, \
            f"securitySchemes[{name!r}] must wrap exactly one *SecurityScheme typed key; got {list(decl.keys())}"
