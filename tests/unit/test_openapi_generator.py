"""Phase 17.V - OpenAPI 3.1 generator tests."""

from __future__ import annotations

import pytest

from a2a_agent.openapi_generator import OpenAPISpec, build_openapi


def test_spec_declares_openapi_3_1():
    spec = build_openapi()
    assert spec.openapi == "3.1.0"


def test_spec_includes_all_canonical_paths():
    spec = build_openapi()
    expected = {
        "/.well-known/agent-card",
        "/.well-known/marketplace",
        "/.well-known/sharp-capabilities",
        "/healthz",
        "/api/batch/decision-cards",
        "/oauth/token",
        "/cds-services",
        "/cds-services/{hookId}",
        "/a2a/trustedrisk",
    }
    assert set(spec.paths.keys()) == expected


def test_healthz_description_includes_tool_count():
    spec = build_openapi()
    description = (
        spec.paths["/healthz"]["get"]["responses"]["200"]
        ["description"]
    )
    assert "tools" in description
    assert "bundles" in description


def test_oauth_security_scheme_advertises_scopes():
    spec = build_openapi()
    scheme = spec.components["securitySchemes"]["oauth_client_credentials"]
    scopes = scheme["flows"]["clientCredentials"]["scopes"]
    for s in ("discharge.read", "discharge.execute", "phi.scan"):
        assert s in scopes


def test_api_key_security_scheme_present():
    spec = build_openapi()
    scheme = spec.components["securitySchemes"]["api_key"]
    assert scheme["type"] == "apiKey"
    assert scheme["name"] == "X-API-Key"


def test_round_trip_through_pydantic():
    spec = build_openapi()
    payload = spec.model_dump(mode="json")
    rebuilt = OpenAPISpec.model_validate(payload)
    assert rebuilt.openapi == spec.openapi
    assert set(rebuilt.paths.keys()) == set(spec.paths.keys())


def test_tags_cover_each_endpoint_group():
    spec = build_openapi()
    tag_names = {t["name"] for t in spec.tags}
    for required in ("A2A v1", "Marketplace", "SHARP-on-MCP",
                      "Operations", "Batch", "OAuth", "CDS Hooks"):
        assert required in tag_names


def test_schema_definitions_referenceable():
    spec = build_openapi()
    schemas = spec.components["schemas"]
    for required in ("AgentCard", "Healthz", "BatchResponse",
                      "OAuthTokenResponse", "CDSHookRequest",
                      "CDSHooksResponse"):
        assert required in schemas
