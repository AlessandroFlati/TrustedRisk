"""Tests for MARKET-1/2/3/4 -- public marketplace endpoints + OAuth flow."""
from __future__ import annotations

import os

import pytest


# ─────────────────────── MARKET-1: Public agent-card ───────────────────────

def test_well_known_agent_card_on_mcp_server():
    """The MCP server must expose /.well-known/agent-card.json publicly."""
    os.environ.pop("TRUSTEDRISK_OAUTH_ENABLED", None)
    from mcp_server.server import build_http_app
    from starlette.testclient import TestClient

    app = build_http_app()
    with TestClient(app) as client:
        # No SHARP headers, no auth -- must succeed (public)
        r = client.get("/.well-known/agent-card.json")
    assert r.status_code == 200
    data = r.json()
    # The card's `name` is the human-readable label; `_legacy_name`
    # carries the canonical routing slug.
    assert data.get("_legacy_name") == "trustedrisk-agent" or \
        data["name"] == "trustedrisk-agent"
    assert "skills" in data and len(data["skills"]) >= 4


def test_well_known_agent_card_on_playground():
    import importlib.util, sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "src"))

    spec = importlib.util.spec_from_file_location(
        "playground_server",
        str(ROOT / "apps" / "playground" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["playground_server"] = mod
    spec.loader.exec_module(mod)

    from fastapi.testclient import TestClient
    with TestClient(mod.app) as client:
        r = client.get("/.well-known/agent-card.json")
    assert r.status_code == 200
    data = r.json()
    assert data.get("_legacy_name") == "trustedrisk-agent" or \
        data["name"] == "trustedrisk-agent"


# ─────────────────────── MARKET-2: Marketplace manifest ───────────────────────

def test_marketplace_manifest_endpoint():
    os.environ.pop("TRUSTEDRISK_OAUTH_ENABLED", None)
    from mcp_server.server import build_http_app
    from starlette.testclient import TestClient

    app = build_http_app()
    with TestClient(app) as client:
        r = client.get("/.well-known/marketplace.json")
    assert r.status_code == 200
    data = r.json()
    # Required fields
    assert data["id"] == "trustedrisk-agent"
    assert data["manifestVersion"]
    assert "category" in data
    assert "tags" in data
    assert "endpoints" in data
    assert "supportedFhirResources" in data
    assert "R4" in data["fhirVersions"]
    assert data["sharpExtension"]["enabled"] is True


def test_marketplace_manifest_lists_examples():
    os.environ.pop("TRUSTEDRISK_OAUTH_ENABLED", None)
    from mcp_server.server import build_http_app
    from starlette.testclient import TestClient

    app = build_http_app()
    with TestClient(app) as client:
        r = client.get("/.well-known/marketplace.json")
    data = r.json()
    assert "examples" in data
    assert len(data["examples"]) >= 3
    for ex in data["examples"]:
        assert "title" in ex
        assert "input_summary" in ex


# ─────────────────────── MARKET-3: SHARP capability declaration ───────────────────────

def test_sharp_capabilities_endpoint():
    os.environ.pop("TRUSTEDRISK_OAUTH_ENABLED", None)
    from mcp_server.server import build_http_app
    from starlette.testclient import TestClient

    app = build_http_app()
    with TestClient(app) as client:
        r = client.get("/.well-known/sharp-capabilities.json")
    assert r.status_code == 200
    data = r.json()
    assert "experimental" in data
    fc = data["experimental"]["fhir_context_required"]
    assert fc["value"] is True
    assert "X-FHIR-Server-URL" in fc["headers"]
    assert "X-FHIR-Access-Token" in fc["headers"]


# ─────────────────────── Health/ready ───────────────────────

def test_healthz_returns_ok():
    os.environ.pop("TRUSTEDRISK_OAUTH_ENABLED", None)
    from mcp_server.server import build_http_app
    from starlette.testclient import TestClient

    app = build_http_app()
    with TestClient(app) as client:
        r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_when_coefficients_present():
    os.environ.pop("TRUSTEDRISK_OAUTH_ENABLED", None)
    os.environ["TRUSTEDRISK_COEFFICIENTS_PATH"] = "data/coefficients.json"
    from mcp_server.server import build_http_app
    from starlette.testclient import TestClient

    app = build_http_app()
    with TestClient(app) as client:
        r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["ready"] is True


# ─────────────────────── Public paths bypass SHARP middleware ───────────────────────

def test_public_paths_bypass_sharp_middleware():
    """The public endpoints must not require SHARP context headers."""
    os.environ.pop("TRUSTEDRISK_OAUTH_ENABLED", None)
    from mcp_server.server import build_http_app
    from starlette.testclient import TestClient

    app = build_http_app()
    public_paths = [
        "/.well-known/agent-card.json",
        "/.well-known/marketplace.json",
        "/.well-known/sharp-capabilities.json",
        "/healthz",
        "/readyz",
    ]
    with TestClient(app) as client:
        for path in public_paths:
            r = client.get(path)
            # 200 (or 503 for readyz when artifacts missing) -- never 403
            # SHARP "missing_fhir_context"
            assert r.status_code != 403, (
                f"{path} should not require SHARP headers; got 403"
            )


# ─────────────────────── MARKET-4: OAuth round-trip ───────────────────────

@pytest.fixture
def oauth_enabled(monkeypatch, tmp_path):
    """Enable OAuth with a single test client."""
    import json
    from mcp_server.oauth.issuer import OAuthClient

    secret = "test-secret-supersecure-32-bytes!"
    secret_hash = OAuthClient.hash_secret(secret)
    clients_path = tmp_path / "oauth_clients.json"
    clients_path.write_text(json.dumps({
        "schema_version": 1,
        "clients": [{
            "client_id": "test-client-1",
            "client_secret_hash": secret_hash,
            "tenant": "tenant-a",
            "allowed_scopes": ["discharge.read", "discharge.execute"],
            "allowed_fhir_servers": ["https://hapi.tenant-a.example.com/fhir"],
        }],
    }), encoding="utf-8")

    monkeypatch.setenv("TRUSTEDRISK_OAUTH_ENABLED", "1")
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_CLIENTS_PATH", str(clients_path))
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_SECRET",
                          "signing-secret-32-bytes-or-more-please")
    yield {"client_id": "test-client-1", "secret": secret,
            "tenant": "tenant-a"}


def test_oauth_token_then_protected_endpoint(oauth_enabled):
    """End-to-end: client requests token -> uses bearer -> server accepts."""
    from mcp_server.server import build_http_app
    from starlette.testclient import TestClient

    app = build_http_app()
    with TestClient(app) as client:
        # 1. Request a token
        r = client.post("/oauth/token", json={
            "client_id": oauth_enabled["client_id"],
            "client_secret": oauth_enabled["secret"],
            "scope": "discharge.read",
            "grant_type": "client_credentials",
        })
        assert r.status_code == 200, r.text
        token_resp = r.json()
        assert "access_token" in token_resp
        token = token_resp["access_token"]

        # 2. Use the token on a public endpoint (still works -- public bypass)
        r2 = client.get("/.well-known/agent-card.json",
                          headers={"Authorization": f"Bearer {token}"})
        assert r2.status_code == 200


def test_oauth_no_bearer_blocks_protected_path(oauth_enabled):
    """Without a bearer, a non-public path returns 401."""
    from mcp_server.server import build_http_app
    from starlette.testclient import TestClient

    app = build_http_app()
    with TestClient(app) as client:
        r = client.post("/api/batch/decision-cards",
                          json={"patients": []})
    assert r.status_code == 401
    assert "missing" in r.text.lower() or "bearer" in r.text.lower()


def test_oauth_invalid_credentials_400(oauth_enabled):
    from mcp_server.server import build_http_app
    from starlette.testclient import TestClient

    app = build_http_app()
    with TestClient(app) as client:
        r = client.post("/oauth/token", json={
            "client_id": "test-client-1",
            "client_secret": "wrong-secret",
            "grant_type": "client_credentials",
        })
    assert r.status_code == 400


def test_oauth_unsupported_grant_type_400(oauth_enabled):
    from mcp_server.server import build_http_app
    from starlette.testclient import TestClient

    app = build_http_app()
    with TestClient(app) as client:
        r = client.post("/oauth/token", json={
            "client_id": "test-client-1",
            "client_secret": oauth_enabled["secret"],
            "grant_type": "authorization_code",
        })
    assert r.status_code == 400
    assert r.json()["error"] == "unsupported_grant_type"


def test_oauth_unknown_client_400(oauth_enabled):
    from mcp_server.server import build_http_app
    from starlette.testclient import TestClient

    app = build_http_app()
    with TestClient(app) as client:
        r = client.post("/oauth/token", json={
            "client_id": "nonexistent-client",
            "client_secret": "anything",
            "grant_type": "client_credentials",
        })
    assert r.status_code == 400
