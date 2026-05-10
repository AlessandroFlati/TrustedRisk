"""Phase 3.4 -- A2A push notifications: registry + dispatch + HTTP endpoints."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from a2a_agent.push_notifications import (
    PushNotificationRegistry,
    default_registry,
    dispatch_task_artifact_update,
    dispatch_task_status_update,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean_default_registry():
    """Each test starts with a clean module-level registry."""
    default_registry().clear()
    yield
    default_registry().clear()


# ─────────────────────── Registry ───────────────────────

def test_registry_create_returns_unique_config():
    reg = PushNotificationRegistry()
    a = reg.create("task-1", "https://hook.example/1")
    b = reg.create("task-1", "https://hook.example/2")
    assert a.config_id != b.config_id


def test_registry_list_for_task():
    reg = PushNotificationRegistry()
    a = reg.create("task-1", "https://hook.example/1")
    b = reg.create("task-2", "https://hook.example/2")
    c = reg.create("task-1", "https://hook.example/3")
    by_task = reg.list_for_task("task-1")
    assert {x.config_id for x in by_task} == {a.config_id, c.config_id}


def test_registry_delete_returns_bool():
    reg = PushNotificationRegistry()
    cfg = reg.create("task-1", "https://hook.example/1")
    assert reg.delete(cfg.config_id) is True
    assert reg.delete(cfg.config_id) is False


# ─────────────────────── Dispatch (mocked HTTP) ───────────────────────

def _patched_async_client(monkeypatch, handler):
    """Replace httpx.AsyncClient with a MockTransport-backed one."""
    transport = httpx.MockTransport(handler)

    class _Patched(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, transport=transport, **kw)

    monkeypatch.setattr("a2a_agent.push_notifications.httpx.AsyncClient",
                            _Patched)


def test_dispatch_status_update_posts_to_each_webhook(monkeypatch):
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200)

    _patched_async_client(monkeypatch, handler)

    reg = default_registry()
    reg.create("t-1", "https://hook.example/a")
    reg.create("t-1", "https://hook.example/b")

    reports = _run(dispatch_task_status_update(
        "t-1", state="WORKING", progress_pct=0.5,
        message="halfway",
    ))
    assert len(reports) == 2
    assert all(r.get("ok") for r in reports)
    urls = {url for url, _ in seen}
    assert urls == {"https://hook.example/a", "https://hook.example/b"}
    body = seen[0][1]
    assert body["kind"] == "TaskStatusUpdateEvent"
    assert body["taskId"] == "t-1"


def test_dispatch_no_configs_returns_empty():
    out = _run(dispatch_task_status_update("nobody", state="COMPLETED"))
    assert out == []


def test_dispatch_severity_threshold_filters(monkeypatch):
    seen: list[str] = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200)

    _patched_async_client(monkeypatch, handler)

    reg = default_registry()
    reg.create("t-1", "https://hook.example/info",
                  severity_threshold="info")
    reg.create("t-1", "https://hook.example/warn",
                  severity_threshold="warn")
    reg.create("t-1", "https://hook.example/critical",
                  severity_threshold="critical")

    # severity=warn ⇒ only info + warn deliver, critical skips
    reports = _run(dispatch_task_status_update(
        "t-1", state="WORKING", severity="warn",
    ))
    delivered = [r for r in reports if not r.get("skipped")]
    skipped = [r for r in reports if r.get("skipped")]
    assert len(delivered) == 2
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "below_severity_threshold"


def test_dispatch_artifact_update_carries_payload(monkeypatch):
    captured: list[dict] = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200)

    _patched_async_client(monkeypatch, handler)

    reg = default_registry()
    reg.create("t-1", "https://hook.example/a")
    _run(dispatch_task_artifact_update(
        "t-1", artifact_payload={"events_avoided_mean": 12.5},
        artifact_id="art-99",
    ))
    body = captured[0]
    assert body["kind"] == "TaskArtifactUpdateEvent"
    assert body["artifact"]["artifactId"] == "art-99"
    assert body["artifact"]["parts"][0]["data"] == {"events_avoided_mean": 12.5}


def test_dispatch_handles_5xx_without_raising(monkeypatch):
    def handler(request):
        return httpx.Response(503, content=b"upstream down")
    _patched_async_client(monkeypatch, handler)

    reg = default_registry()
    reg.create("t-1", "https://hook.example/down")
    reports = _run(dispatch_task_status_update("t-1", state="WORKING"))
    assert reports[0]["ok"] is False
    assert reports[0]["status_code"] == 503


def test_dispatch_handles_network_error_without_raising(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("network down")
    _patched_async_client(monkeypatch, handler)

    reg = default_registry()
    reg.create("t-1", "https://hook.example/dead")
    reports = _run(dispatch_task_status_update("t-1", state="WORKING"))
    assert reports[0]["ok"] is False
    assert "ConnectError" in (reports[0].get("error") or "")


# ─────────────────────── HTTP endpoints (TestClient) ───────────────────────

def test_create_push_config_endpoint_round_trip():
    from starlette.testclient import TestClient
    from mcp_server.server import build_http_app

    app = build_http_app()
    client = TestClient(app)
    headers = {
        "X-FHIR-Server-URL": "http://hapi/fhir",
        "X-FHIR-Access-Token": "tok",
    }
    resp = client.post("/api/push/configs", headers=headers, json={
        "taskId": "t-99",
        "webhookUrl": "https://example.com/wh",
        "severityThreshold": "warn",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "t-99"
    assert body["webhook_url"] == "https://example.com/wh"
    cfg_id = body["config_id"]

    list_resp = client.get("/api/push/configs", headers=headers)
    assert any(c["config_id"] == cfg_id for c in list_resp.json()["configs"])

    del_resp = client.delete(f"/api/push/configs/{cfg_id}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] == cfg_id


def test_create_push_config_rejects_missing_fields():
    from starlette.testclient import TestClient
    from mcp_server.server import build_http_app

    app = build_http_app()
    client = TestClient(app)
    headers = {
        "X-FHIR-Server-URL": "http://hapi/fhir",
        "X-FHIR-Access-Token": "tok",
    }
    resp = client.post("/api/push/configs", headers=headers, json={})
    assert resp.status_code == 400
    assert resp.json()["error"] == "missing_required_field"


def test_main_agent_card_declares_push_notifications():
    """Phase 3.4 -- main agent-card now advertises pushNotifications=true."""
    import json as _json
    from pathlib import Path as _P
    p = _P(__file__).resolve().parents[2] / "src" / "a2a_agent" / "agent-card.json"
    raw = _json.loads(p.read_text(encoding="utf-8"))
    assert raw["capabilities"]["pushNotifications"] is True
