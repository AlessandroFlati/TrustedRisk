"""Unit tests for OAuth client_credentials grant + bearer-token middleware."""
from __future__ import annotations

import asyncio
import json
import time

import jwt
import pytest

from mcp_server.oauth import (
    ClientContext,
    OAuthClient,
    OAuthValidationError,
    TokenIssuer,
    TokenValidator,
    get_client_context,
    oauth_bearer_middleware,
)
from mcp_server.oauth.middleware import _client_ctx
from mcp_server.oauth.validator import ValidatedToken


_SECRET = "test-secret-for-hs256-please-rotate"


# ─────────────────────── OAuthClient ───────────────────────

def test_hash_secret_format():
    h = OAuthClient.hash_secret("supersecret")
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64


def test_verify_secret_correct():
    h = OAuthClient.hash_secret("supersecret")
    c = OAuthClient(client_id="c1", client_secret_hash=h, tenant="t1")
    assert c.verify_secret("supersecret") is True


def test_verify_secret_wrong():
    h = OAuthClient.hash_secret("supersecret")
    c = OAuthClient(client_id="c1", client_secret_hash=h, tenant="t1")
    assert c.verify_secret("notthat") is False


def test_verify_secret_legacy_unhashed_rejected():
    """A non-sha256-prefixed hash must be rejected outright (defensive)."""
    c = OAuthClient(client_id="c1", client_secret_hash="plaintext", tenant="t1")
    assert c.verify_secret("plaintext") is False


# ─────────────────────── TokenIssuer ───────────────────────

def _client(*, scopes=None, tenant="tenant-a", fhir_servers=None) -> OAuthClient:
    return OAuthClient(
        client_id="orch-1",
        client_secret_hash=OAuthClient.hash_secret("supersecret"),
        tenant=tenant,
        allowed_scopes=scopes or ["discharge.read", "phi.scan"],
        allowed_fhir_servers=fhir_servers or [],
    )


def test_issuer_rejects_empty_secret():
    with pytest.raises(ValueError):
        TokenIssuer([_client()], signing_secret="")


def test_issue_returns_well_formed_token():
    issuer = TokenIssuer([_client()], signing_secret=_SECRET)
    resp = issuer.issue("orch-1", "supersecret")
    assert resp["token_type"] == "Bearer"
    assert resp["expires_in"] >= 60
    assert resp["tenant"] == "tenant-a"
    assert "discharge.read" in resp["scope"]
    # Payload decodable + has expected claims
    payload = jwt.decode(resp["access_token"], _SECRET, algorithms=["HS256"])
    assert payload["iss"] == "trustedrisk"
    assert payload["sub"] == "orch-1"
    assert payload["tenant"] == "tenant-a"


def test_issue_invalid_client():
    issuer = TokenIssuer([_client()], signing_secret=_SECRET)
    with pytest.raises(ValueError, match="invalid_client"):
        issuer.issue("unknown-client", "supersecret")


def test_issue_wrong_secret():
    issuer = TokenIssuer([_client()], signing_secret=_SECRET)
    with pytest.raises(ValueError, match="invalid_client"):
        issuer.issue("orch-1", "wrong-password")


def test_issue_scope_narrowing():
    """Requested scopes are intersected with allowed_scopes."""
    issuer = TokenIssuer([_client(scopes=["discharge.read", "phi.scan"])],
                         signing_secret=_SECRET)
    resp = issuer.issue("orch-1", "supersecret",
                         requested_scopes=["discharge.read", "discharge.execute"])
    # Only discharge.read survives (not in client.allowed_scopes)
    assert "discharge.read" in resp["scope"]
    assert "discharge.execute" not in resp["scope"]


def test_issue_no_scopes_returns_all_allowed():
    issuer = TokenIssuer([_client(scopes=["a", "b"])], signing_secret=_SECRET)
    resp = issuer.issue("orch-1", "supersecret")
    assert "a" in resp["scope"] and "b" in resp["scope"]


def test_issue_empty_scope_intersection_raises():
    issuer = TokenIssuer([_client(scopes=["a"])], signing_secret=_SECRET)
    with pytest.raises(ValueError, match="invalid_scope"):
        issuer.issue("orch-1", "supersecret", requested_scopes=["x", "y"])


def test_issue_minimum_ttl_enforced():
    """TTL is clamped at >= 60s to avoid one-shot tokens nobody can use."""
    issuer = TokenIssuer([_client()], signing_secret=_SECRET, token_ttl_seconds=10)
    assert issuer.token_ttl_seconds == 60


# ─────────────────────── TokenValidator ───────────────────────

def test_validate_round_trip():
    issuer = TokenIssuer([_client()], signing_secret=_SECRET)
    validator = TokenValidator(_SECRET, clients=[_client()])
    resp = issuer.issue("orch-1", "supersecret")
    validated = validator.validate(resp["access_token"])
    assert validated.client_id == "orch-1"
    assert validated.tenant == "tenant-a"
    assert "discharge.read" in validated.scopes


def test_validate_rejects_bad_signature():
    """Token signed with a different secret must fail."""
    issuer_other = TokenIssuer([_client()], signing_secret="another-secret-xx")
    resp = issuer_other.issue("orch-1", "supersecret")
    validator = TokenValidator(_SECRET, clients=[_client()])
    with pytest.raises(OAuthValidationError, match="invalid_token"):
        validator.validate(resp["access_token"])


def test_validate_rejects_empty_token():
    validator = TokenValidator(_SECRET, clients=[_client()])
    with pytest.raises(OAuthValidationError, match="missing_token"):
        validator.validate("")


def test_validate_rejects_expired():
    """Generate a token with exp in the past."""
    payload = {
        "iss": "trustedrisk", "sub": "orch-1", "tenant": "t",
        "scope": "x", "iat": int(time.time()) - 7200,
        "exp": int(time.time()) - 3600,
    }
    expired = jwt.encode(payload, _SECRET, algorithm="HS256")
    validator = TokenValidator(_SECRET, clients=[_client()])
    with pytest.raises(OAuthValidationError, match="token_expired"):
        validator.validate(expired)


def test_validate_rejects_wrong_issuer():
    payload = {"iss": "evil-corp", "sub": "x", "exp": int(time.time()) + 3600}
    bad = jwt.encode(payload, _SECRET, algorithm="HS256")
    validator = TokenValidator(_SECRET, clients=[_client()])
    with pytest.raises(OAuthValidationError, match="invalid_issuer"):
        validator.validate(bad)


def test_validate_rejects_missing_subject():
    payload = {"iss": "trustedrisk", "exp": int(time.time()) + 3600}
    bad = jwt.encode(payload, _SECRET, algorithm="HS256")
    validator = TokenValidator(_SECRET, clients=[_client()])
    with pytest.raises(OAuthValidationError):
        validator.validate(bad)


# ─────────────────────── FHIR allowlist check ───────────────────────

def test_fhir_allowlist_match():
    client = _client(fhir_servers=["https://tenant-a.fhir.example.com"])
    validator = TokenValidator(_SECRET, clients=[client])
    assert validator.check_fhir_server_allowed(
        "orch-1", "https://tenant-a.fhir.example.com/baseR4/Patient/1"
    ) is True


def test_fhir_allowlist_reject():
    client = _client(fhir_servers=["https://tenant-a.fhir.example.com"])
    validator = TokenValidator(_SECRET, clients=[client])
    assert validator.check_fhir_server_allowed(
        "orch-1", "https://tenant-b.fhir.example.com/baseR4"
    ) is False


def test_fhir_allowlist_unknown_client():
    validator = TokenValidator(_SECRET, clients=[])
    assert validator.check_fhir_server_allowed("orch-X", "anything") is False


def test_fhir_allowlist_empty_means_wildcard():
    """Empty allowed_fhir_servers list = wildcard (used for trusted clients)."""
    client = _client(fhir_servers=[])
    validator = TokenValidator(_SECRET, clients=[client])
    assert validator.check_fhir_server_allowed(
        "orch-1", "https://anywhere.example.com"
    ) is True


# ─────────────────────── Middleware ───────────────────────

class _StubRequest:
    def __init__(self, headers: dict[str, str], path: str = "/mcp"):
        self.headers = headers
        from urllib.parse import urlsplit
        self.url = urlsplit(f"http://localhost{path}")


def _run(coro):
    return asyncio.run(coro)


def test_middleware_passthrough_when_oauth_disabled(monkeypatch):
    monkeypatch.delenv("TRUSTEDRISK_OAUTH_ENABLED", raising=False)
    req = _StubRequest({}, path="/mcp")

    async def handler(_):
        return "ok"

    assert _run(oauth_bearer_middleware(req, handler)) == "ok"


def test_middleware_401_when_no_bearer(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_ENABLED", "1")
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_SECRET", _SECRET)
    req = _StubRequest({}, path="/mcp")

    async def handler(_):
        raise AssertionError("call_next must NOT run when bearer missing")

    resp = _run(oauth_bearer_middleware(req, handler))
    assert resp.status_code == 401
    body = json.loads(resp.body)
    assert body["error_code"] == "missing_bearer"
    assert "WWW-Authenticate" in resp.headers


def test_middleware_401_when_token_expired(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_ENABLED", "1")
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_SECRET", _SECRET)
    payload = {"iss": "trustedrisk", "sub": "x", "tenant": "t",
               "scope": "discharge.read",
               "iat": int(time.time()) - 7200,
               "exp": int(time.time()) - 60}
    token = jwt.encode(payload, _SECRET, algorithm="HS256")
    req = _StubRequest({"Authorization": f"Bearer {token}"}, path="/mcp")

    async def handler(_):
        raise AssertionError("call_next must NOT run on expired token")

    resp = _run(oauth_bearer_middleware(req, handler))
    assert resp.status_code == 401
    body = json.loads(resp.body)
    assert body["error_code"] == "token_expired"


def test_middleware_oauth_token_endpoint_passes_through(monkeypatch):
    """/oauth/* must be reachable WITHOUT a bearer (it's the bearer endpoint)."""
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_ENABLED", "1")
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_SECRET", _SECRET)
    req = _StubRequest({}, path="/oauth/token")

    async def handler(_):
        return "issued"

    assert _run(oauth_bearer_middleware(req, handler)) == "issued"


def test_middleware_binds_client_context(monkeypatch, tmp_path):
    """A valid bearer token sets the ClientContext for downstream handlers."""
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_ENABLED", "1")
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_SECRET", _SECRET)
    # Stage a clients.json so the middleware can verify allowlist
    clients_path = tmp_path / "clients.json"
    clients_path.write_text(json.dumps({
        "clients": [{
            "client_id": "orch-1",
            "client_secret_hash": OAuthClient.hash_secret("s"),
            "tenant": "tenant-a",
            "allowed_scopes": ["discharge.read"],
            "allowed_fhir_servers": ["https://tenant-a.fhir.example.com"],
        }],
    }))
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_CLIENTS_PATH", str(clients_path))

    issuer = TokenIssuer([_client()], signing_secret=_SECRET)
    resp = issuer.issue("orch-1", "supersecret")
    req = _StubRequest({
        "Authorization": f"Bearer {resp['access_token']}",
        "X-FHIR-Server-URL": "https://tenant-a.fhir.example.com/baseR4",
    }, path="/mcp")

    seen: list[ClientContext] = []

    async def handler(_):
        seen.append(get_client_context())
        return "ok"

    out = _run(oauth_bearer_middleware(req, handler))
    assert out == "ok"
    assert len(seen) == 1
    assert seen[0].client_id == "orch-1"
    assert seen[0].tenant == "tenant-a"


def test_middleware_blocks_cross_tenant_fhir(monkeypatch, tmp_path):
    """A token for tenant A used with X-FHIR-Server-URL pointing to tenant B
    must be rejected (multi-tenant isolation)."""
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_ENABLED", "1")
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_SECRET", _SECRET)
    clients_path = tmp_path / "clients.json"
    clients_path.write_text(json.dumps({
        "clients": [{
            "client_id": "orch-1",
            "client_secret_hash": OAuthClient.hash_secret("s"),
            "tenant": "tenant-a",
            "allowed_scopes": ["discharge.read"],
            "allowed_fhir_servers": ["https://tenant-a.fhir.example.com"],
        }],
    }))
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_CLIENTS_PATH", str(clients_path))

    issuer = TokenIssuer([_client(fhir_servers=["https://tenant-a.fhir.example.com"])],
                         signing_secret=_SECRET)
    resp = issuer.issue("orch-1", "supersecret")
    req = _StubRequest({
        "Authorization": f"Bearer {resp['access_token']}",
        "X-FHIR-Server-URL": "https://tenant-b.fhir.example.com/baseR4",  # ← cross-tenant
    }, path="/mcp")

    async def handler(_):
        raise AssertionError("call_next must NOT run on cross-tenant attempt")

    response = _run(oauth_bearer_middleware(req, handler))
    assert response.status_code == 401
    body = json.loads(response.body)
    assert body["error_code"] == "fhir_server_not_allowed"


def test_middleware_resets_client_context_after_request(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_ENABLED", "1")
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_SECRET", _SECRET)
    clients_path = tmp_path / "clients.json"
    clients_path.write_text(json.dumps({
        "clients": [{
            "client_id": "orch-1",
            "client_secret_hash": OAuthClient.hash_secret("s"),
            "tenant": "tenant-a",
            "allowed_scopes": ["discharge.read"],
            "allowed_fhir_servers": [],
        }],
    }))
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_CLIENTS_PATH", str(clients_path))
    issuer = TokenIssuer([_client(fhir_servers=[])], signing_secret=_SECRET)
    resp = issuer.issue("orch-1", "supersecret")
    req = _StubRequest({"Authorization": f"Bearer {resp['access_token']}"})

    async def handler(_):
        return "ok"

    _run(oauth_bearer_middleware(req, handler))
    # Outside the request, ContextVar must raise
    with pytest.raises(LookupError):
        get_client_context()


# ─────────────────────── Client registry loader ───────────────────────

def test_load_clients_from_env_missing_file_returns_empty(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_CLIENTS_PATH", "/nonexistent/clients.json")
    from mcp_server.oauth.issuer import load_clients_from_env
    assert load_clients_from_env() == []


def test_load_clients_from_env_parses_well(tmp_path, monkeypatch):
    fp = tmp_path / "clients.json"
    fp.write_text(json.dumps({
        "clients": [
            {"client_id": "c1",
             "client_secret_hash": "sha256:abc",
             "tenant": "t1",
             "allowed_scopes": ["a", "b"],
             "allowed_fhir_servers": ["https://x.example.com"]},
        ],
    }))
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_CLIENTS_PATH", str(fp))
    from mcp_server.oauth.issuer import load_clients_from_env
    out = load_clients_from_env()
    assert len(out) == 1
    assert out[0].client_id == "c1"
    assert "a" in out[0].allowed_scopes


def test_load_clients_from_env_skips_malformed(tmp_path, monkeypatch):
    fp = tmp_path / "clients.json"
    fp.write_text(json.dumps({
        "clients": [
            {"client_id": "c1", "client_secret_hash": "sha256:x", "tenant": "t1"},
            {"BAD": "no client_id"},  # malformed
            "not a dict",  # malformed
        ],
    }))
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_CLIENTS_PATH", str(fp))
    from mcp_server.oauth.issuer import load_clients_from_env
    out = load_clients_from_env()
    assert len(out) == 1
