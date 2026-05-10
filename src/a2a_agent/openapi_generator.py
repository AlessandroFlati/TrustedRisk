"""Phase 17.V - OpenAPI 3.1 spec generator.

Auto-generates a single OpenAPI 3.1 spec covering the canonical
TrustedRisk surface:
  - `/.well-known/agent-card` (A2A v1)
  - `/.well-known/marketplace`
  - `/.well-known/sharp-capabilities`
  - `/healthz`
  - `/api/batch/decision-cards`
  - `/oauth/token`
  - `/cds-services` + `/cds-services/{hookId}` (CDS Hooks v1.1)
  - `/a2a/trustedrisk` (A2A message endpoint)

Plus references to the bundle catalogue (BUNDLES) so that an
external client can discover the full tool list without crawling
each endpoint individually.

Pure-data, deterministic, stdlib-only.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OpenAPISpec(BaseModel):
    openapi: str = "3.1.0"
    info: dict[str, Any]
    servers: list[dict[str, Any]]
    paths: dict[str, Any]
    components: dict[str, Any]
    tags: list[dict[str, Any]]


def _path_get_well_known_agent_card() -> dict[str, Any]:
    return {
        "get": {
            "tags": ["A2A v1"],
            "summary": "Return the A2A agent-card",
            "operationId": "getAgentCard",
            "responses": {
                "200": {
                    "description": "A2A v1 agent card with skills, "
                                   "extensions, security schemes.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AgentCard"},
                        },
                    },
                },
            },
        },
    }


def _path_get_well_known_marketplace() -> dict[str, Any]:
    return {
        "get": {
            "tags": ["Marketplace"],
            "summary": "Return the federation marketplace manifest",
            "operationId": "getMarketplace",
            "responses": {
                "200": {
                    "description": "Marketplace manifest JSON.",
                    "content": {"application/json": {"schema": {
                        "type": "object", "additionalProperties": True,
                    }}},
                },
                "404": {"description": "Marketplace manifest not "
                                       "configured."},
            },
        },
    }


def _path_get_well_known_sharp() -> dict[str, Any]:
    return {
        "get": {
            "tags": ["SHARP-on-MCP"],
            "summary": "Return SHARP-on-MCP capability declaration",
            "operationId": "getSHARPCapabilities",
            "responses": {
                "200": {"description": "SHARP capability JSON.",
                        "content": {"application/json": {"schema": {
                            "type": "object",
                            "additionalProperties": True}}}},
            },
        },
    }


def _path_get_healthz(n_tools: int, n_bundles: int) -> dict[str, Any]:
    return {
        "get": {
            "tags": ["Operations"],
            "summary": "Liveness + tool/bundle counts",
            "operationId": "getHealthz",
            "responses": {
                "200": {
                    "description": (
                        "Returns ok + tool/bundle counts. Current "
                        f"federation surface: {n_tools} tools / "
                        f"{n_bundles} bundles."
                    ),
                    "content": {"application/json": {"schema": {
                        "$ref": "#/components/schemas/Healthz",
                    }}},
                },
            },
        },
    }


def _path_post_batch() -> dict[str, Any]:
    return {
        "post": {
            "tags": ["Batch"],
            "summary": "Run a batch of DecisionCards",
            "operationId": "postBatchDecisionCards",
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {
                    "type": "object",
                    "properties": {
                        "patients": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["patients"],
                }}},
            },
            "responses": {
                "200": {"description": "Per-patient DecisionCard "
                                       "list.",
                        "content": {"application/json": {"schema": {
                            "$ref": "#/components/schemas/BatchResponse",
                        }}}},
                "400": {"description": "Bad request."},
            },
        },
    }


def _path_post_oauth_token() -> dict[str, Any]:
    return {
        "post": {
            "tags": ["OAuth"],
            "summary": "Client-credentials grant (RFC 6749 §4.4)",
            "operationId": "postOAuthToken",
            "requestBody": {
                "required": True,
                "content": {
                    "application/x-www-form-urlencoded": {"schema": {
                        "type": "object",
                        "properties": {
                            "grant_type": {"type": "string"},
                            "client_id": {"type": "string"},
                            "client_secret": {"type": "string"},
                        },
                        "required": ["grant_type", "client_id"],
                    }},
                },
            },
            "responses": {
                "200": {
                    "description": "Bearer token (HS256 JWT).",
                    "content": {"application/json": {"schema": {
                        "$ref": "#/components/schemas/OAuthTokenResponse",
                    }}},
                },
                "401": {"description": "Invalid client credentials."},
            },
        },
    }


def _path_get_cds_services() -> dict[str, Any]:
    return {
        "get": {
            "tags": ["CDS Hooks"],
            "summary": "List CDS Hooks v1.1 services",
            "operationId": "getCDSServices",
            "responses": {
                "200": {
                    "description": "Service catalogue.",
                    "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {"services": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/CDSService",
                            },
                        }},
                        "required": ["services"],
                    }}},
                },
            },
        },
    }


def _path_post_cds_hook() -> dict[str, Any]:
    return {
        "post": {
            "tags": ["CDS Hooks"],
            "summary": "Invoke a CDS Hooks service",
            "operationId": "postCDSHook",
            "parameters": [
                {"name": "hookId", "in": "path", "required": True,
                 "schema": {"type": "string"}},
            ],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {
                    "$ref": "#/components/schemas/CDSHookRequest",
                }}},
            },
            "responses": {
                "200": {
                    "description": "Cards response.",
                    "content": {"application/json": {"schema": {
                        "$ref": "#/components/schemas/CDSHooksResponse",
                    }}},
                },
            },
        },
    }


def _path_post_a2a() -> dict[str, Any]:
    return {
        "post": {
            "tags": ["A2A v1"],
            "summary": "Send an A2A v1 message to TrustedRisk",
            "operationId": "postA2AMessage",
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {
                    "type": "object", "additionalProperties": True,
                }}},
            },
            "responses": {
                "200": {
                    "description": "A2A v1 reply Message + extensions.",
                    "content": {"application/json": {"schema": {
                        "type": "object", "additionalProperties": True,
                    }}},
                },
            },
        },
    }


def build_openapi(
    *, server_url: str = "https://trustedrisk.local",
) -> OpenAPISpec:
    """Assemble the OpenAPI 3.1 spec from the registered bundle map."""
    from mcp_server.tools import BUNDLES   # type: ignore
    n_bundles = len(BUNDLES)
    unique_tools: set[str] = set()
    for tools in BUNDLES.values():
        unique_tools.update(tools)
    n_tools = len(unique_tools)

    paths = {
        "/.well-known/agent-card": _path_get_well_known_agent_card(),
        "/.well-known/marketplace":
            _path_get_well_known_marketplace(),
        "/.well-known/sharp-capabilities":
            _path_get_well_known_sharp(),
        "/healthz": _path_get_healthz(n_tools, n_bundles),
        "/api/batch/decision-cards": _path_post_batch(),
        "/oauth/token": _path_post_oauth_token(),
        "/cds-services": _path_get_cds_services(),
        "/cds-services/{hookId}": _path_post_cds_hook(),
        "/a2a/trustedrisk": _path_post_a2a(),
    }
    components = {
        "schemas": {
            "AgentCard": {
                "type": "object",
                "additionalProperties": True,
                "description": (
                    "A2A v1 agent card with `_bundles` extension "
                    "listing the registered tool surface."
                ),
            },
            "Healthz": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["ok"]},
                    "version": {"type": "string"},
                    "tools_registered": {"type": "integer"},
                    "bundles_registered": {"type": "integer"},
                },
                "required": ["status", "tools_registered",
                             "bundles_registered"],
            },
            "BatchResponse": {
                "type": "object",
                "properties": {
                    "n_patients": {"type": "integer"},
                    "decision_cards": {
                        "type": "array",
                        "items": {"type": "object",
                                  "additionalProperties": True},
                    },
                },
            },
            "OAuthTokenResponse": {
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "token_type": {"type": "string",
                                    "enum": ["Bearer"]},
                    "expires_in": {"type": "integer"},
                    "scope": {"type": "string"},
                },
                "required": ["access_token", "token_type",
                              "expires_in"],
            },
            "CDSService": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "hook": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["id", "hook"],
            },
            "CDSHookRequest": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "hook": {"type": "string"},
                    "hookInstance": {"type": "string"},
                    "context": {"type": "object"},
                    "fhirAuthorization": {"type": "object"},
                    "prefetch": {"type": "object"},
                },
            },
            "CDSHooksResponse": {
                "type": "object",
                "properties": {
                    "cards": {"type": "array",
                                "items": {"type": "object"}},
                    "systemActions": {"type": "array",
                                       "items": {"type": "object"}},
                },
                "required": ["cards"],
            },
        },
        "securitySchemes": {
            "oauth_client_credentials": {
                "type": "oauth2",
                "flows": {
                    "clientCredentials": {
                        "tokenUrl": f"{server_url}/oauth/token",
                        "scopes": {
                            "discharge.read": "read-only discharge",
                            "discharge.execute": "execute discharge "
                                                  "decisioning",
                            "phi.scan": "PHI scanning only",
                            "grounding.read": "evidence grounding "
                                              "only",
                        },
                    },
                },
            },
            "api_key": {
                "type": "apiKey", "in": "header",
                "name": "X-API-Key",
            },
        },
    }
    info = {
        "title": "TrustedRisk Federation API",
        "version": "1.0.0",
        "description": (
            "OpenAPI 3.1 specification covering the TrustedRisk A2A "
            "federation surface (agent-card + marketplace + SHARP "
            "capabilities + batch decisioning + OAuth + CDS Hooks "
            f"v1.1). Federation backs {n_tools} MCP tools across "
            f"{n_bundles} thematic bundles."
        ),
        "license": {"name": "Apache-2.0",
                     "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    }
    tags = [
        {"name": "A2A v1", "description": "A2A v1 protocol surface."},
        {"name": "Marketplace",
         "description": "Federation marketplace metadata."},
        {"name": "SHARP-on-MCP",
         "description": "FHIR-context capability declaration."},
        {"name": "Operations",
         "description": "Liveness + readiness."},
        {"name": "Batch",
         "description": "Population-level batch decisioning."},
        {"name": "OAuth",
         "description": "Multi-tenant OAuth 2.0 client credentials."},
        {"name": "CDS Hooks",
         "description": "HL7 CDS Hooks v1.1 service surface."},
    ]
    return OpenAPISpec(
        info=info,
        servers=[{"url": server_url,
                   "description": "Production marketplace ingress"}],
        paths=paths, components=components, tags=tags,
    )
