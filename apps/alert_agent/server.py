"""COMPOSE-1 — TrustedRisk alert agent (federated A2A partner).

The alert agent subscribes to TrustedRisk's calibration drift monitor and
emits webhook alerts when the overall tier escalates to `warn` or `alert`.
It plays the role of a specialist monitoring agent in the multi-agent
composition pattern, so TrustedRisk itself doesn't have to own webhook
delivery + subscriber management.

Flow:
  1. Operator → POST /a2a/skill/subscribe_to_alerts {webhook_url, severity}
  2. Operator (or a cron) → POST /a2a/skill/check_drift_now
        → fetches TrustedRisk /api/calibration/status
        → classifies tier
        → POSTs an AlertEvent to every subscriber whose threshold matches
        → returns the dispatched payload (or null when tier=ok)
  3. Operator → GET /a2a/alerts/recent — list of recent alert events.

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \
        .venv/Scripts/python.exe -m uvicorn \
        apps.alert_agent.server:app --port 8768
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


app = FastAPI(title="trustedrisk-alert-agent (COMPOSE-1)", version="0.1.0")

_AGENT_CARD_PATH = Path(__file__).parent / "agent_card.json"

TRUSTEDRISK_BASE_URL = os.environ.get(
    "TRUSTEDRISK_A2A_URL", "http://localhost:8765",
)


# ─────────────────────── In-process state ───────────────────────

# {webhook_id: {webhook_url, severity_threshold, registered_at}}
_SUBSCRIPTIONS: dict[str, dict[str, Any]] = {}

# Bounded ring buffer of recent alerts (most-recent-first when serialized)
_RECENT_ALERTS: Deque[dict[str, Any]] = deque(maxlen=200)


def _reset_state() -> None:
    """Test helper — wipe the subscription + alert history."""
    _SUBSCRIPTIONS.clear()
    _RECENT_ALERTS.clear()


# ─────────────────────── Drift → tier classification ───────────────────────

_SEVERITY_RANK = {"ok": 0, "warn": 1, "alert": 2}


def _tier_meets_threshold(tier: str, threshold: str) -> bool:
    return _SEVERITY_RANK.get(tier, 0) >= _SEVERITY_RANK.get(threshold, 1)


def _build_alert_event(report: dict[str, Any]) -> dict[str, Any]:
    """Compose the AlertEvent body delivered to subscribers."""
    signals = report.get("signals", []) or []
    top_signal = next(
        (s for s in signals if s.get("tier") in ("alert", "warn")), None,
    )
    return {
        "alert_id": uuid.uuid4().hex,
        "source_agent": "trustedrisk-alert-agent",
        "alert_type": "calibration_drift",
        "severity": report.get("overall_tier", "warn"),
        "kpi": top_signal.get("name") if top_signal else None,
        "kpi_value": top_signal.get("value") if top_signal else None,
        "kpi_baseline": top_signal.get("baseline") if top_signal else None,
        "rationale": report.get("rationale", ""),
        "window_label": report.get("window_label"),
        "window_n_cards": report.get("window_n_cards", 0),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────── Hooks (overridable for tests) ───────────────────────

async def _fetch_drift_report() -> dict[str, Any]:
    """Fetch the calibration drift report from TrustedRisk.

    Tests monkeypatch this function to inject canned reports.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{TRUSTEDRISK_BASE_URL}/api/calibration/status",
        )
        r.raise_for_status()
        return r.json()


async def _deliver_webhook(webhook_url: str,
                              payload: dict[str, Any]) -> dict[str, Any]:
    """Deliver an alert event to a single webhook URL.

    Returns a delivery record (status, latency, error). Failures are
    captured — this is best-effort delivery, not a transactional contract.
    """
    started = datetime.now(timezone.utc)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(webhook_url, json=payload)
        return {
            "webhook_url": webhook_url,
            "status_code": r.status_code,
            "delivered": 200 <= r.status_code < 300,
            "error": None,
            "started_at": started.isoformat(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "webhook_url": webhook_url,
            "status_code": None,
            "delivered": False,
            "error": str(exc),
            "started_at": started.isoformat(),
        }


# ─────────────────────── Routes ───────────────────────

@app.get("/.well-known/agent-card.json")
async def agent_card() -> JSONResponse:
    return JSONResponse(content=json.loads(
        _AGENT_CARD_PATH.read_text(encoding="utf-8")))


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse(content={
        "status": "ok",
        "subscriptions": len(_SUBSCRIPTIONS),
        "recent_alerts": len(_RECENT_ALERTS),
    })


@app.post("/a2a/skill/subscribe_to_alerts")
async def subscribe_to_alerts(req: Request) -> JSONResponse:
    body = await req.json()
    webhook_url = body.get("webhook_url")
    if not isinstance(webhook_url, str) or not webhook_url.startswith(
            ("http://", "https://")):
        raise HTTPException(status_code=400,
                              detail="webhook_url must be an http(s) URL")

    threshold = (body.get("severity_threshold") or "warn").lower()
    if threshold not in _SEVERITY_RANK:
        raise HTTPException(status_code=400,
                              detail=f"severity_threshold must be one of "
                                       f"{list(_SEVERITY_RANK)}")
    if threshold == "ok":
        raise HTTPException(status_code=400,
                              detail="severity_threshold=ok would deliver every "
                                       "poll; use warn or alert")

    sub_id = uuid.uuid4().hex
    _SUBSCRIPTIONS[sub_id] = {
        "webhook_url": webhook_url,
        "severity_threshold": threshold,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    return JSONResponse(content={"subscription_id": sub_id,
                                    "webhook_url": webhook_url,
                                    "severity_threshold": threshold})


@app.delete("/a2a/skill/subscribe_to_alerts/{sub_id}")
async def unsubscribe(sub_id: str) -> JSONResponse:
    if sub_id not in _SUBSCRIPTIONS:
        raise HTTPException(status_code=404, detail="unknown subscription_id")
    _SUBSCRIPTIONS.pop(sub_id)
    return JSONResponse(content={"deleted": sub_id})


@app.get("/a2a/skill/subscribe_to_alerts")
async def list_subscriptions() -> JSONResponse:
    return JSONResponse(content={
        "subscriptions": [
            {"subscription_id": sid, **sub}
            for sid, sub in _SUBSCRIPTIONS.items()
        ],
        "n": len(_SUBSCRIPTIONS),
    })


@app.post("/a2a/skill/check_drift_now")
async def check_drift_now(req: Request) -> JSONResponse:
    """Synchronously polls TrustedRisk for drift + dispatches webhooks."""
    body: dict[str, Any] = {}
    if req.headers.get("content-length") and \
            int(req.headers["content-length"]) > 0:
        body = await req.json()

    report = body.get("drift_report")  # allow tests / orchestrators to inject
    if report is None:
        report = await _fetch_drift_report()

    tier = (report.get("overall_tier") or "ok").lower()
    if tier == "ok":
        return JSONResponse(content={"tier": "ok", "alert": None,
                                        "deliveries": []})

    alert = _build_alert_event(report)
    deliveries = []
    for sub_id, sub in list(_SUBSCRIPTIONS.items()):
        if not _tier_meets_threshold(tier, sub["severity_threshold"]):
            continue
        delivery = await _deliver_webhook(sub["webhook_url"], alert)
        delivery["subscription_id"] = sub_id
        deliveries.append(delivery)

    record = {**alert, "deliveries": deliveries}
    _RECENT_ALERTS.appendleft(record)
    return JSONResponse(content=record)


@app.get("/a2a/alerts/recent")
async def list_recent_alerts(limit: int = 20) -> JSONResponse:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400,
                              detail="limit must be in [1, 200]")
    items = list(_RECENT_ALERTS)[:limit]
    return JSONResponse(content={"alerts": items, "n": len(items)})


# ─────────────────────── UI (minimal) ───────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    n_sub = len(_SUBSCRIPTIONS)
    n_alerts = len(_RECENT_ALERTS)
    return f"""<!DOCTYPE html>
<html><head><title>trustedrisk-alert-agent</title>
<style>body{{font-family:system-ui;margin:32px;max-width:800px;color:#222}}
code{{background:#f3f4f6;padding:2px 6px;border-radius:4px}}</style></head>
<body>
<h1>trustedrisk-alert-agent</h1>
<p>COMPOSE-1 federated A2A partner. Subscriptions: <code>{n_sub}</code> ·
   Recent alerts buffered: <code>{n_alerts}</code></p>
<ul>
  <li>POST <code>/a2a/skill/subscribe_to_alerts</code> {{webhook_url, severity_threshold?}}</li>
  <li>POST <code>/a2a/skill/check_drift_now</code> [{{drift_report?}}]</li>
  <li>GET <code>/a2a/alerts/recent</code></li>
  <li>GET <code>/.well-known/agent-card.json</code></li>
</ul>
</body></html>"""
