"""INT-2 -- SMART on FHIR launch handshake tests."""
from __future__ import annotations

import pytest

from mcp_server.smart import launch as smart_mod


@pytest.fixture(autouse=True)
def _reset_state():
    smart_mod._reset_state()
    yield
    smart_mod._reset_state()


@pytest.fixture
def app(monkeypatch):
    """Build the MCP HTTP app with SMART routes mounted."""
    monkeypatch.delenv("TRUSTEDRISK_OAUTH_ENABLED", raising=False)
    from mcp_server.server import build_http_app
    return build_http_app()


def _smart_config_factory(authorize_endpoint="https://ehr.example.com/oauth2/authorize",
                              token_endpoint="https://ehr.example.com/oauth2/token"):
    async def _fetch(_iss):
        return {
            "authorization_endpoint": authorize_endpoint,
            "token_endpoint": token_endpoint,
            "capabilities": ["launch-ehr", "context-ehr-patient",
                                "client-public", "permission-patient"],
        }
    return _fetch


def _token_factory(*, access_token="ehr-access-token-xyz",
                       expires_in=3600, scope="patient/*.read",
                       patient_id="pt-001",
                       fhir_context_patient: str | None = None):
    async def _exchange(token_endpoint, *, code, redirect_uri,
                            client_id, code_verifier=None):
        resp = {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": expires_in,
            "scope": scope,
        }
        if patient_id is not None:
            resp["patient"] = patient_id
        if fhir_context_patient is not None:
            resp["fhirContext"] = [
                {"reference": f"Patient/{fhir_context_patient}"}
            ]
        return resp
    return _exchange


# ─────────────────────── /smart/launch ───────────────────────

def test_launch_records_pending_session_and_returns_authorize_url(
        app, monkeypatch):
    from starlette.testclient import TestClient

    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())

    body = {"iss": "https://ehr.example.com/fhir",
              "launch": "ehr-launch-abc",
              "client_id": "trustedrisk-app",
              "scope": "launch patient/*.read"}
    with TestClient(app) as c:
        r = c.post("/smart/launch", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["session_id"]
    assert payload["state"]
    auth_url = payload["authorize_url"]
    assert "https://ehr.example.com/oauth2/authorize" in auth_url
    assert "response_type=code" in auth_url
    assert "client_id=trustedrisk-app" in auth_url
    assert f"aud=https" in auth_url   # iss propagated as aud
    assert "launch=ehr-launch-abc" in auth_url


def test_launch_missing_iss_returns_400(app, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())
    with TestClient(app) as c:
        r = c.post("/smart/launch", json={"launch": "x"})
    assert r.status_code == 400
    assert r.json()["error"] == "missing_iss"


def test_launch_missing_launch_returns_400(app, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())
    with TestClient(app) as c:
        r = c.post("/smart/launch",
                     json={"iss": "https://ehr.example.com/fhir"})
    assert r.status_code == 400


def test_launch_smart_config_fetch_failure_502(app, monkeypatch):
    from starlette.testclient import TestClient

    async def _fail(_iss):
        raise RuntimeError("SMART config endpoint unreachable")
    monkeypatch.setattr(smart_mod, "_fetch_smart_config", _fail)

    with TestClient(app) as c:
        r = c.post("/smart/launch",
                     json={"iss": "https://ehr.example.com/fhir",
                            "launch": "x"})
    assert r.status_code == 502
    assert r.json()["error"] == "smart_config_fetch_failed"


# ─────────────────────── /smart/callback ───────────────────────

def test_callback_completes_handshake(app, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())
    monkeypatch.setattr(smart_mod, "_token_exchange",
                          _token_factory(patient_id="pt-001"))

    with TestClient(app) as c:
        launch_resp = c.post("/smart/launch", json={
            "iss": "https://ehr.example.com/fhir",
            "launch": "ehr-launch-abc"}).json()
        state = launch_resp["state"]
        cb = c.get(f"/smart/callback?code=auth-code-1&state={state}")
    assert cb.status_code == 200, cb.text
    body = cb.json()
    assert body["session_id"] == launch_resp["session_id"]
    assert body["server_url"] == "https://ehr.example.com/fhir"
    assert body["patient_id"] == "pt-001"


def test_callback_missing_state_returns_400(app, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())
    with TestClient(app) as c:
        r = c.get("/smart/callback?code=x")
    assert r.status_code == 400


def test_callback_unknown_state_returns_400(app, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())
    with TestClient(app) as c:
        r = c.get("/smart/callback?code=x&state=never-issued")
    assert r.status_code == 400


def test_callback_token_exchange_failure_502(app, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())

    async def _failing_exchange(*args, **kwargs):
        raise RuntimeError("EHR token endpoint refused connection")
    monkeypatch.setattr(smart_mod, "_token_exchange", _failing_exchange)

    with TestClient(app) as c:
        launch = c.post("/smart/launch", json={
            "iss": "https://ehr.example.com/fhir",
            "launch": "x"}).json()
        cb = c.get(f"/smart/callback?code=z&state={launch['state']}")
    assert cb.status_code == 502
    assert cb.json()["error"] == "token_exchange_failed"


def test_callback_token_response_missing_access_token(app, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())

    async def _no_access_token(*args, **kwargs):
        return {"token_type": "Bearer"}
    monkeypatch.setattr(smart_mod, "_token_exchange", _no_access_token)

    with TestClient(app) as c:
        launch = c.post("/smart/launch", json={
            "iss": "https://ehr.example.com/fhir", "launch": "x"}).json()
        cb = c.get(f"/smart/callback?code=z&state={launch['state']}")
    assert cb.status_code == 502
    assert cb.json()["error"] == "token_response_missing_access_token"


def test_callback_resolves_patient_from_fhir_context(app, monkeypatch):
    """If the token response uses fhirContext instead of `patient`, parse it."""
    from starlette.testclient import TestClient
    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())
    monkeypatch.setattr(smart_mod, "_token_exchange",
                          _token_factory(patient_id=None,
                                            fhir_context_patient="pt-zzz"))

    with TestClient(app) as c:
        launch = c.post("/smart/launch", json={
            "iss": "https://ehr.example.com/fhir", "launch": "x"}).json()
        cb = c.get(f"/smart/callback?code=z&state={launch['state']}")
    assert cb.status_code == 200
    assert cb.json()["patient_id"] == "pt-zzz"


# ─────────────────────── /smart/session ───────────────────────

def test_session_returns_redacted_token(app, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())
    monkeypatch.setattr(smart_mod, "_token_exchange",
                          _token_factory(access_token="LONG-EHR-ACCESS-TOKEN"))

    with TestClient(app) as c:
        launch = c.post("/smart/launch", json={
            "iss": "https://ehr.example.com/fhir",
            "launch": "x"}).json()
        c.get(f"/smart/callback?code=z&state={launch['state']}")
        sess = c.get(f"/smart/session/{launch['session_id']}")
    assert sess.status_code == 200
    body = sess.json()
    # Never expose the raw token
    assert "access_token" not in body
    assert "token_redacted" in body
    assert body["token_redacted"].endswith("OKEN")  # last 4 chars preserved
    assert body["token_redacted"].startswith("LONG-E")


def test_session_unknown_id_404(app, monkeypatch):
    from starlette.testclient import TestClient
    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())
    with TestClient(app) as c:
        r = c.get("/smart/session/nope-not-issued")
    assert r.status_code == 404


# ─────────────────────── Integration: state transitions ───────────────────────

def test_session_only_resolved_after_callback(app, monkeypatch):
    """Before /smart/callback, /smart/session/{id} must 404."""
    from starlette.testclient import TestClient
    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())
    monkeypatch.setattr(smart_mod, "_token_exchange", _token_factory())

    with TestClient(app) as c:
        launch = c.post("/smart/launch", json={
            "iss": "https://ehr.example.com/fhir",
            "launch": "x"}).json()
        sid = launch["session_id"]
        before = c.get(f"/smart/session/{sid}")
        c.get(f"/smart/callback?code=z&state={launch['state']}")
        after = c.get(f"/smart/session/{sid}")
    assert before.status_code == 404
    assert after.status_code == 200


def test_smart_launch_path_bypasses_oauth_when_enabled(app, monkeypatch,
                                                            tmp_path):
    """SMART endpoints must remain public when OAuth middleware is enabled."""
    import json as _json
    from mcp_server.oauth.issuer import OAuthClient

    # Set up OAuth -- requires real client registry + signing secret
    secret = "test-oauth-secret-supersecure-32-bytes!"
    cp = tmp_path / "clients.json"
    cp.write_text(_json.dumps({
        "schema_version": 1, "clients": [{
            "client_id": "tc",
            "client_secret_hash": OAuthClient.hash_secret(secret),
            "tenant": "t", "allowed_scopes": ["smart.launch"],
            "allowed_fhir_servers": [],
        }],
    }), encoding="utf-8")
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_ENABLED", "1")
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_CLIENTS_PATH", str(cp))
    monkeypatch.setenv("TRUSTEDRISK_OAUTH_SECRET",
                          "signing-secret-32-bytes-or-more-please")
    monkeypatch.setattr(smart_mod, "_fetch_smart_config",
                          _smart_config_factory())

    from mcp_server.server import build_http_app
    app2 = build_http_app()
    from starlette.testclient import TestClient
    with TestClient(app2) as c:
        # No bearer token -- SMART launch should still work
        r = c.post("/smart/launch", json={
            "iss": "https://ehr.example.com/fhir",
            "launch": "x"})
    assert r.status_code == 200
