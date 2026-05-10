"""Federated A2A partner — `darena-data-agent` (DEMO-2).

A second FastAPI server that DEMONSTRATES the A2A pattern by acting as a
federated client of TrustedRisk. The flow:

  1. Discovery: GET /.well-known/agent-card.json on the TrustedRisk agent
     to learn its skills + bundles.
  2. Capability negotiation: pick a skill that matches the patient
     summary (e.g. `safe_discharge_review` if the request looks like a
     discharge question).
  3. Tool invocation: call /api/run/<scenario> on TrustedRisk OR call
     individual tools via the MCP transport. For the demo we use the
     scenario-run endpoint to keep the example self-contained.
  4. Audit trail: every exchanged message is recorded in a JSON log
     written to `apps/federation_partner/handshake.log`.

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \
        .venv/Scripts/python.exe -m uvicorn \
        apps.federation_partner.server:app --port 8766
    # In another terminal: TrustedRisk playground on port 8765
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


app = FastAPI(title="darena-data-agent (federation partner demo)",
                version="0.1.0")

_AGENT_CARD_PATH = Path(__file__).parent / "agent_card.json"
_HANDSHAKE_LOG = Path(__file__).parent / "handshake.log"


# Default upstream TrustedRisk URL. Override via env var.
TRUSTEDRISK_BASE_URL = os.environ.get(
    "TRUSTEDRISK_A2A_URL", "http://localhost:8765",
)


# ─────────────────────── Audit trail ───────────────────────

def _log_event(event: dict[str, Any]) -> None:
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(_HANDSHAKE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# ─────────────────────── Discovery + capability negotiation ───────────────────────

async def _discover_trustedrisk_capabilities() -> dict[str, Any]:
    """Fetch TrustedRisk's agent-card. The TrustedRisk runtime exposes the
    bundles + skills via its agent-card.json; the playground exposes the
    bundle map at /api/bundles. We hit both."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Bundle map (playground endpoint)
        bundles_resp = await client.get(f"{TRUSTEDRISK_BASE_URL}/api/bundles")
        bundles_resp.raise_for_status()
        # Agent-card may be served via static file path; fall back to the
        # local file when the playground does not expose it.
        try:
            ac = await client.get(
                f"{TRUSTEDRISK_BASE_URL}/.well-known/agent-card.json")
            ac.raise_for_status()
            agent_card = ac.json()
        except Exception:
            agent_card_path = ROOT / "src" / "a2a_agent" / "agent-card.json"
            agent_card = json.loads(agent_card_path.read_text(encoding="utf-8"))
    return {"agent_card": agent_card,
              "bundles": bundles_resp.json().get("bundles", {})}


def _select_bundle_for_request(patient_summary: dict[str, Any],
                                 bundles_map: dict[str, list[str]]) -> str:
    """Pick the most relevant bundle for the patient summary. This is the
    'capability negotiation' part of the A2A handshake — in a real system
    it would be richer (LLM-mediated, scoring per bundle, etc.). Here we
    keep a small rule-based mapping."""
    summary_str = json.dumps(patient_summary).lower()

    # Order matters — most-specific first
    if ("stroke" in summary_str or "nihss" in summary_str
            or "slurred speech" in summary_str or "facial droop" in summary_str
            or "hemiparesis" in summary_str
            or "right-sided weakness" in summary_str
            or "left-sided weakness" in summary_str):
        return "stroke_acs"
    if "chest pain" in summary_str or "heart score" in summary_str:
        return "stroke_acs"
    if ("trauma" in summary_str or "mvc" in summary_str or "fast" in summary_str
            or "polytrauma" in summary_str):
        return "trauma_critical"
    if "pediatric" in summary_str or (
        isinstance(patient_summary.get("age"), (int, float))
        and patient_summary["age"] < 18
    ):
        return "pediatric"
    if "preeclampsia" in summary_str or "pregnancy" in summary_str:
        return "obstetric_geriatric"
    if "delirium" in summary_str or "falls" in summary_str:
        return "obstetric_geriatric"
    if "sepsis" in summary_str or "antibiotic" in summary_str:
        return "antimicrobial"
    if "chemo" in summary_str or "oncology" in summary_str:
        return "oncology"
    if "dka" in summary_str or "ketoacidosis" in summary_str:
        return "endocrine_acute"
    if "dialysis" in summary_str or "aki" in summary_str:
        return "nephrology"
    if ("suicid" in summary_str or "psychiatric" in summary_str  # suicid matches suicide+suicidal
            or "self-harm" in summary_str):
        return "mental_health"
    return "core_discharge"


def _scenario_for_bundle(bundle: str) -> str:
    """Map a bundle to a representative demo scenario slug."""
    mapping = {
        "core_discharge": "A", "stroke_acs": "M",
        "trauma_critical": "Q", "pediatric": "I",
        "obstetric_geriatric": "O", "antimicrobial": "K",
        "oncology": "L", "endocrine_acute": "R",
        "nephrology": "R", "mental_health": "J",
        "ed_acute": "E", "imaging": "R",
    }
    return mapping.get(bundle, "A")


# ─────────────────────── Endpoints ───────────────────────

@app.get("/.well-known/agent-card.json")
async def agent_card() -> JSONResponse:
    return JSONResponse(content=json.loads(_AGENT_CARD_PATH.read_text(encoding="utf-8")))


@app.post("/a2a/skill/evaluate_patient_via_trustedrisk")
async def evaluate_patient_via_trustedrisk(req: Request) -> JSONResponse:
    """Main federated A2A handshake.

    Body:
      {"patient_summary": {age, conditions, [free-text fields]}}

    Returns a federated DecisionCard summary + the audit trail of
    A2A-level messages exchanged."""
    body = await req.json()
    patient_summary = body.get("patient_summary") or {}
    handshake_id = str(uuid.uuid4())[:8]

    audit: list[dict[str, Any]] = []

    # Step 1: discovery
    t0 = time.perf_counter()
    audit.append({
        "step": 1, "from": "darena-data-agent", "to": "trustedrisk-agent",
        "action": "agent-card discovery + bundle map fetch",
    })
    try:
        caps = await _discover_trustedrisk_capabilities()
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": "discovery_failed",
                       "detail": f"{type(exc).__name__}: {exc}",
                       "trustedrisk_url": TRUSTEDRISK_BASE_URL},
        )
    discovery_ms = (time.perf_counter() - t0) * 1000.0
    audit.append({
        "step": 1, "from": "trustedrisk-agent", "to": "darena-data-agent",
        "action": "agent-card returned",
        "skills_offered": [s["id"] for s in caps["agent_card"].get("skills", [])],
        "n_bundles_offered": len(caps["bundles"]),
        "duration_ms": round(discovery_ms, 1),
    })

    # Step 2: capability negotiation
    bundle = _select_bundle_for_request(patient_summary, caps["bundles"])
    audit.append({
        "step": 2, "from": "darena-data-agent", "to": "darena-data-agent",
        "action": "capability negotiation → bundle selection",
        "selected_bundle": bundle,
        "tools_in_bundle": caps["bundles"].get(bundle, [])[:5] + (
            ["… and more"] if len(caps["bundles"].get(bundle, [])) > 5 else []
        ),
    })

    # Step 3: skill invocation (via the playground's /api/run scenario endpoint)
    scenario_slug = _scenario_for_bundle(bundle)
    audit.append({
        "step": 3, "from": "darena-data-agent", "to": "trustedrisk-agent",
        "action": f"invoke representative scenario {scenario_slug} for bundle {bundle}",
    })
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{TRUSTEDRISK_BASE_URL}/api/run/{scenario_slug}")
            resp.raise_for_status()
            decision = resp.json()
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": "invocation_failed",
                       "detail": f"{type(exc).__name__}: {exc}",
                       "audit": audit},
        )
    invocation_ms = (time.perf_counter() - t0) * 1000.0

    # Step 4: response
    audit.append({
        "step": 4, "from": "trustedrisk-agent", "to": "darena-data-agent",
        "action": "decision-card returned",
        "deterministic_action": decision.get("deterministic_action"),
        "deterministic_confidence": decision.get("deterministic_confidence"),
        "agreement": decision.get("agreement"),
        "n_tool_ops": len(decision.get("operations") or []),
        "duration_ms": round(invocation_ms, 1),
    })

    # Persist audit
    _log_event({
        "handshake_id": handshake_id, "audit": audit,
        "patient_summary": patient_summary,
    })

    return JSONResponse(content={
        "handshake_id": handshake_id,
        "selected_bundle": bundle,
        "scenario_used": scenario_slug,
        "decision_summary": {
            "action": decision.get("deterministic_action"),
            "confidence": decision.get("deterministic_confidence"),
            "agreement": decision.get("agreement"),
            "abstain_triggers": decision.get("abstain_triggers"),
        },
        "audit_trail": audit,
        "trustedrisk_endpoint": TRUSTEDRISK_BASE_URL,
    })


@app.get("/a2a/handshake-log")
async def get_handshake_log(limit: int = 20) -> JSONResponse:
    """Return the recent handshake history."""
    if not _HANDSHAKE_LOG.exists():
        return JSONResponse(content={"events": []})
    events = []
    with open(_HANDSHAKE_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return JSONResponse(content={"events": events[-limit:]})


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return r"""<!DOCTYPE html>
<html><head><title>darena-data-agent (federation partner)</title>
<style>
  body { font-family: -apple-system, sans-serif; background: #0d1117; color: #c9d1d9;
          max-width: 900px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
  h1 { font-size: 22px; }
  code { background: #1f2530; padding: 2px 6px; border-radius: 3px; color: #79c0ff; }
  pre { background: #1f2530; padding: 12px; border-radius: 4px; overflow: auto; }
  button { padding: 8px 14px; background: #58a6ff; color: white; border: none;
            border-radius: 4px; cursor: pointer; font-size: 13px; }
  .audit { margin-top: 16px; }
  .step { padding: 10px 14px; background: #161b22; border-left: 3px solid #58a6ff;
           border-radius: 4px; margin: 6px 0; font-size: 13px; }
</style></head>
<body>
<h1>darena-data-agent</h1>
<p style="color: #8b949e;">Federation partner demo. Performs A2A discovery
+ capability negotiation + tool invocation against TrustedRisk.</p>
<p>Upstream TrustedRisk: <code id="ts-url"></code></p>
<button onclick="run()">Run federated request (chest pain patient)</button>
<div id="result" style="margin-top: 16px;"></div>
<script>
document.getElementById('ts-url').textContent = '""" + TRUSTEDRISK_BASE_URL + r"""';
async function run() {
  const out = document.getElementById('result');
  out.innerHTML = '<em>Running A2A handshake…</em>';
  const r = await fetch('/a2a/skill/evaluate_patient_via_trustedrisk', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      patient_summary: {
        age: 58, gender: 'M', chief_complaint: 'chest pain radiating to left arm',
        risk_factors: ['hypertension', 'smoker'],
      },
    }),
  });
  const data = await r.json();
  let html = '<h3>Decision summary</h3><pre>' +
    JSON.stringify(data.decision_summary, null, 2) + '</pre>';
  html += '<h3>A2A audit trail (' + data.audit_trail.length + ' steps)</h3>';
  data.audit_trail.forEach(s => {
    html += '<div class="step"><strong>step ' + s.step + '</strong>: ' +
      s.from + ' → ' + s.to + ' — ' + s.action +
      (s.duration_ms ? ' (' + s.duration_ms + ' ms)' : '') + '</div>';
  });
  html += '<p style="color: #8b949e; margin-top: 16px;">Selected bundle: <code>' +
    data.selected_bundle + '</code> · scenario used: <code>' +
    data.scenario_used + '</code> · handshake id: <code>' + data.handshake_id +
    '</code></p>';
  out.innerHTML = html;
}
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8766)
