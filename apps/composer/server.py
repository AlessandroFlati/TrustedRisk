"""trustedrisk-composer — Phase 6.1 BYO orchestrator (port 8780).

Demonstrates the A2A v1 *Agent Composition* pattern. Configures
end-to-end clinical workflows that consult multiple TrustedRisk
federation specialists in sequence.

Endpoints:
  GET  /.well-known/agent-card.json   — A2A v1 agent card
  GET  /api/workflows                 — list available workflow templates
  POST /api/run/{workflow_id}         — execute a workflow with inputs
  GET  /healthz / /readyz             — Cloud-Run-compatible probes

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.composer.server:app --port 8780
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .orchestrator import execute_workflow
from .workflows import REGISTRY, list_workflows


_AGENT_CARD_PATH = Path(__file__).parent / "agent_card.json"


app = FastAPI(
    title="TrustedRisk composer",
    description=(
        "Phase 6.1 BYO orchestrator. Runs realistic clinical workflows "
        "by consulting multiple TrustedRisk federation specialists "
        "(discharge / acute / evidence / population / pediatric / "
        "mental-health / pa / scribe / patient / ...)."
    ),
    version="1.0.0",
    # Disable trailing-slash redirects: A2A clients POST to the bare
    # agent URL and a 307 strips the Authorization header (HTTP redirect
    # security rule), causing every chat invocation to land unauthenticated.
    redirect_slashes=False,
)


from a2a_agent.agent_card_normalizer import normalize_agent_card
from mcp_server.oauth.middleware import oauth_bearer_middleware
from mcp_server.sharp.headers import sharp_context_middleware


# Paths that bypass SHARP enforcement (discoverability + probes + the
# A2A v1 messaging endpoint, which chat clients call without FHIR
# headers). The `/api/` workflow endpoints take FHIR context inline via
# the request body (`inputs.fhir_bundle`), so they don't need the
# header-driven SHARP ContextVar — downstream tools enforce their own
# FHIR-context requirements at invocation time. OAuth is still enforced
# separately on the messaging endpoint.
_SHARP_PUBLIC_PREFIXES = (
    "/.well-known/", "/healthz", "/readyz", "/openapi.json", "/docs",
    "/api/",
)
_SHARP_PUBLIC_EXACT = {"/"}


# Middleware registration order: FastAPI / Starlette executes the LAST
# `@app.middleware("http")` decorator FIRST (outermost). We want OAuth
# to gate the request before SHARP consumes the body, so SHARP is
# declared first (innermost) and OAuth second (outermost).
@app.middleware("http")
async def _composer_sharp(request, call_next):
    path = request.url.path or ""
    if path in _SHARP_PUBLIC_EXACT or any(
        path.startswith(p) for p in _SHARP_PUBLIC_PREFIXES
    ):
        return await call_next(request)
    return await sharp_context_middleware(request, call_next)


@app.middleware("http")
async def _composer_oauth(request, call_next):
    return await oauth_bearer_middleware(request, call_next)


@app.get("/.well-known/agent-card.json")
async def well_known_agent_card():
    if not _AGENT_CARD_PATH.exists():
        return JSONResponse(status_code=404,
                              content={"error": "agent_card_missing"})
    raw = json.loads(_AGENT_CARD_PATH.read_text(encoding="utf-8"))
    return JSONResponse(content=normalize_agent_card(raw))


# A2A v1 JSON-RPC messaging endpoint. Routes the user's prompt to one
# of the 5 workflow templates (chf_admission, sepsis_workup, ...) via
# keyword match, runs it, and returns the per-step trace as a Task
# artifact.
from a2a_agent.a2a_messaging import build_composer_handler, handle_jsonrpc

_inner_composer_handler = build_composer_handler()


async def _composer_handler(prompt: str, msg: dict):
    """Wrap the composer handler in a PO FHIR-context bind.

    The chat client delivers FHIR creds in `message.metadata` via the
    PO extension URI; downstream tools (compute_readmission_risk and
    friends) read them off the SHARP FHIRContext ContextVar. Bridge
    the two for the duration of the request.
    """
    from a2a_agent.po_fhir_context import (
        bind_po_fhir_context,
        extract_po_fhir_context,
        release_po_fhir_context,
    )
    raw_meta = msg.get("metadata") or {}
    try:
        po_ctx = extract_po_fhir_context(raw_meta if isinstance(raw_meta, dict) else None)
    except ValueError:
        po_ctx = None
    ctx_token = bind_po_fhir_context(po_ctx) if po_ctx is not None else None
    try:
        return await _inner_composer_handler(prompt, msg)
    finally:
        if ctx_token is not None:
            release_po_fhir_context(ctx_token)


@app.post("/")
async def a2a_messaging(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid_json: {exc}") from exc
    return await handle_jsonrpc(body, _composer_handler)


@app.get("/api/workflows")
async def list_workflows_endpoint():
    return {"workflows": list_workflows()}


@app.post("/api/run/{workflow_id}")
async def run_workflow_endpoint(workflow_id: str, request: Request):
    if workflow_id not in REGISTRY:
        raise HTTPException(
            status_code=404, detail=f"Unknown workflow id {workflow_id!r}; "
                                          f"see /api/workflows"
        )
    workflow = REGISTRY[workflow_id]
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400,
                                detail=f"invalid_json: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400,
                                detail="body must be a JSON object")
    inputs = body.get("inputs") or body
    try:
        execution = await execute_workflow(workflow, inputs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Serialise the WorkflowExecution dataclass (and its nested
    # WorkflowStepResult outputs) into JSON-friendly form.
    out = asdict(execution)
    # Pydantic outputs are stored under `output_dump`; clear the raw
    # `output` attribute (which can carry non-serialisable objects).
    for step in out["steps"]:
        step.pop("output", None)
    return out


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "workflows": len(REGISTRY),
        "workflow_ids": sorted(REGISTRY.keys()),
    }


@app.get("/readyz")
async def readyz():
    coef_path = os.environ.get(
        "TRUSTEDRISK_COEFFICIENTS_PATH", "data/coefficients.json")
    ready = Path(coef_path).exists()
    return JSONResponse(
        status_code=(200 if ready else 503),
        content={"ready": ready, "coefficients_path": coef_path},
    )


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_COMPOSER_PORT", "8780"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
