"""trustedrisk-orchestrator -- single-entry clinical dispatcher.

Mirrors the apps/composer FastAPI shape (OAuth + SHARP middleware,
A2A v1 messaging, redirect_slashes=False) but routes the inbound
prompt through apps.orchestrator.dispatcher instead of the prebuilt
workflow registry.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import threading as _threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


_AGENT_CARD_PATH = Path(__file__).parent / "agent_card.json"

# JSONL trace of every JSON-RPC round-trip (request + response). Used by
# scripts/watch_a2a_trace.py for live debugging in PO. Path overridable
# via TRUSTEDRISK_A2A_TRACE_PATH; set to empty string to disable.
_TRACE_PATH = Path(os.environ.get(
    "TRUSTEDRISK_A2A_TRACE_PATH",
    str(ROOT / "logs" / "a2a_trace.jsonl"),
))
_TRACE_LOCK = _threading.Lock()


def _append_trace(req_body: dict, resp_body: dict) -> None:
    """Best-effort append of a {ts, request, response} entry. Never raises."""
    try:
        if not str(_TRACE_PATH):
            return
        _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": _dt.datetime.now(_dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "request": req_body,
            "response": resp_body,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _TRACE_LOCK:
            with _TRACE_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        # Logger must never break the server. Swallow silently.
        return


app = FastAPI(
    title="TrustedRisk orchestrator",
    description=(
        "Phase 18 single-entry orchestrator. One A2A endpoint that picks "
        "the right specialist or workflow for a free-form clinical prompt "
        "and narrates the dispatch."
    ),
    version="1.0.0",
    redirect_slashes=False,
)


from a2a_agent.agent_card_normalizer import normalize_agent_card
from mcp_server.oauth.middleware import oauth_bearer_middleware
from mcp_server.sharp.headers import sharp_context_middleware


_SHARP_PUBLIC_PREFIXES = (
    "/.well-known/", "/healthz", "/readyz", "/openapi.json", "/docs",
)
_SHARP_PUBLIC_EXACT = {"/"}


@app.middleware("http")
async def _orchestrator_sharp(request, call_next):
    path = request.url.path or ""
    if path in _SHARP_PUBLIC_EXACT or any(
        path.startswith(p) for p in _SHARP_PUBLIC_PREFIXES
    ):
        return await call_next(request)
    return await sharp_context_middleware(request, call_next)


@app.middleware("http")
async def _orchestrator_oauth(request, call_next):
    return await oauth_bearer_middleware(request, call_next)


@app.get("/.well-known/agent-card.json")
async def well_known_agent_card():
    if not _AGENT_CARD_PATH.exists():
        return JSONResponse(status_code=404,
                              content={"error": "agent_card_missing"})
    raw = json.loads(_AGENT_CARD_PATH.read_text(encoding="utf-8"))
    return JSONResponse(content=normalize_agent_card(raw))


# ------------------------------------------------------------------ A2A endpoint

from a2a_agent.a2a_messaging import _artifact, handle_jsonrpc
from apps.orchestrator.dispatcher import dispatch


_FHIR_CTX_URI = "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context"


def _extract_fhir_metadata(msg: dict) -> dict:
    """Lift fhirUrl + patientId out of the PO FHIR-context extension.

    fhirToken is intentionally dropped before the dict lands in the
    artifact: even when an LLM later inspects the artifact, the bearer
    never escapes the handler scope.
    """
    out: dict = {}
    meta = msg.get("metadata") or {}
    if not isinstance(meta, dict):
        return out
    ctx = meta.get(_FHIR_CTX_URI)
    if isinstance(ctx, dict):
        if isinstance(ctx.get("patientId"), str):
            out["patient_id"] = f"Patient/{ctx['patientId']}"
        if isinstance(ctx.get("fhirUrl"), str):
            out["fhir_server_url"] = ctx["fhirUrl"]
    # Allow direct top-level overrides too (programmatic A2A clients).
    for k, v in meta.items():
        if k == _FHIR_CTX_URI or v is None:
            continue
        out[k] = v
    return out


def _format_abstain_banner(sub_results: list) -> str:
    """Build a top-of-message banner listing every abstained
    tool/step across all sub-results.

    The chat-LLM upstream has been observed to ignore an abstain warning
    appended to a long rationale and present the numeric output of an
    abstained tool as a clean clinical estimate. Hoisting the warning
    to the very first line of the status_message is the most reliable
    signal: chat clients display it before the artifact dump.
    """
    bits: list[str] = []
    for sub in sub_results or []:
        dump = getattr(sub, "output_dump", None)
        if not isinstance(dump, dict):
            continue
        steps = dump.get("abstained_steps") or []
        if isinstance(steps, list):
            for s in steps:
                if not isinstance(s, dict):
                    continue
                sid = str(s.get("step_id") or "?")
                reason = str(s.get("reason") or "abstain")
                bits.append(f"{sub.target_label} / step {sid}: "
                            f"{reason.split(':', 1)[0]}")
        if dump.get("abstain_recommended") and not steps:
            reason = str(dump.get("abstain_reason") or "abstain")
            bits.append(f"{sub.target_label}: "
                        f"{reason.split(':', 1)[0]}")
    if not bits:
        return ""
    return (
        "[ABSTAIN WARNING] The following tool(s) abstained and their "
        "numeric outputs are NOT calibrated for this patient -- DO NOT "
        "present them as final estimates: "
        + "; ".join(bits) + ". "
    )


def _format_bundle_provenance(sub_results: list) -> str:
    """Build a short provenance line listing real FHIR resources consumed.

    The status_message is the single piece of text the upstream chat-LLM
    reads first; explicitly naming the resource counts here keeps the
    LLM from concluding the engine fabricated data when its own UI view
    only surfaces the Patient resource.
    """
    seen: set[tuple] = set()
    counts: dict[str, int] = {}
    n_total = 0
    for sub in sub_results or []:
        dump = getattr(sub, "output_dump", None)
        prov = (dump or {}).get("_bundle_provenance") if isinstance(dump, dict) else None
        if not isinstance(prov, dict):
            continue
        key = (prov.get("patient_id"), prov.get("fetched_at_iso"))
        if key in seen:
            continue
        seen.add(key)
        rc = prov.get("resource_counts") or {}
        if isinstance(rc, dict):
            for k, v in rc.items():
                counts[k] = max(counts.get(k, 0), int(v))
        n_total = max(n_total, int(prov.get("n_total_resources") or 0))
    if not counts:
        return ""
    by_count = sorted(counts.items(), key=lambda x: -x[1])
    head = ", ".join(f"{n} {rt}" for rt, n in by_count[:6])
    return (
        f" Consumed {n_total} live FHIR resources from the workspace "
        f"server ({head}); the engine read these directly -- this "
        f"output is not synthesised."
    )


# Clinical fields to surface at the very top of the status message when a
# macro workflow yields nested sub-results. Order matters: the chat client
# truncates its own LLM output at a low max_tokens budget, so the most
# decision-relevant signals come first.
_HEADLINE_FIELDS: tuple[str, ...] = (
    "esi_level", "priority", "disposition", "recommended_unit",
    "severity_tier", "severity", "score_total", "level", "risk_level",
    "recommended_response", "recommended_action", "top_pick_id",
    "de_escalation_recommended", "target_regimen",
    "discharge_contract_satisfied", "n_admission_meds", "n_discharge_meds",
    "n_gaps_found", "n_high_priority_gaps", "n_concerns",
    "n_red_flags", "follow_up_window_days",
    "decision", "recommendation", "verdict",
)


def _scalar_str(v: Any) -> str | None:
    """Format a primitive for a compact headline line; skip non-scalars."""
    if v is None or isinstance(v, bool):
        return str(v).lower() if isinstance(v, bool) else None
    if isinstance(v, (int, float, str)):
        s = str(v)
        return s if len(s) <= 60 else s[:57] + "..."
    if isinstance(v, list) and all(isinstance(x, (int, float, str)) for x in v):
        s = ", ".join(str(x) for x in v[:4])
        return s if len(s) <= 60 else s[:57] + "..."
    return None


def _format_macro_headlines(output_dump: dict | None) -> str:
    """Build a compact bullet list of headline findings per inner step.

    Macro workflows return `steps[i].output_dump` shaped as
    `{inner_step_id: inner_dump}`. We pick the first 2 - 3 decision-
    relevant fields per inner dump and inline them. The chat-LLM in the
    upstream client truncates aggressively (observed at 153 output
    tokens), so loading headline numbers into the first lines of our
    response lets the user see disposition / severity / top antibiotic
    even when the LLM's own summary gets cut mid-sentence.
    """
    if not isinstance(output_dump, dict):
        return ""
    steps = output_dump.get("steps") or []
    if not isinstance(steps, list):
        return ""
    lines: list[str] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        outer_id = str(s.get("step_id") or "")
        # Trim the "hop_N_" prefix added by `_make_macro`.
        outer_short = outer_id
        if outer_short.startswith("hop_"):
            parts = outer_short.split("_", 2)
            if len(parts) == 3:
                outer_short = parts[2]
        od = s.get("output_dump")
        if not isinstance(od, dict):
            continue
        # Detect macro hop (dict of dicts) vs single-tool step (flat dict).
        inner_items = [
            (k, v) for k, v in od.items()
            if isinstance(v, dict) and not k.startswith("_")
        ]
        if not inner_items:
            # Single-tool: treat the outer dump itself as one inner.
            inner_items = [(outer_short, od)]
        for inner_id, inner in inner_items:
            picks: list[str] = []
            for field in _HEADLINE_FIELDS:
                if field in inner:
                    fmt = _scalar_str(inner[field])
                    if fmt is not None:
                        picks.append(f"{field}={fmt}")
                        if len(picks) >= 3:
                            break
            if not picks:
                # Fall back to first 2 non-meta scalar fields.
                for k, v in inner.items():
                    if k.startswith("_") or k in (
                        "abstain_recommended", "abstain_reason",
                        "patient_id", "rationale", "references",
                    ):
                        continue
                    fmt = _scalar_str(v)
                    if fmt is not None:
                        picks.append(f"{k}={fmt}")
                        if len(picks) >= 2:
                            break
            if picks:
                lines.append(f"- {outer_short}/{inner_id}: {', '.join(picks)}")
    if not lines:
        return ""
    return "Headline findings:\n" + "\n".join(lines) + "\n\n"


async def _orchestrator_handler(prompt: str, msg: dict) -> tuple[str, list[dict]]:
    from a2a_agent.po_fhir_context import (
        bind_po_fhir_context,
        extract_po_fhir_context,
        release_po_fhir_context,
    )
    from mcp_server.fhir.client import (
        begin_bundle_cache,
        reset_bundle_cache,
    )

    metadata = _extract_fhir_metadata(msg)
    raw_meta = msg.get("metadata") or {}
    # Bridge A2A metadata.fhir-context onto the SHARP FHIRContext
    # ContextVar so downstream tools that expect SHARP headers (e.g.
    # compute_readmission_risk) work from a chat invocation too.
    try:
        po_ctx = extract_po_fhir_context(raw_meta if isinstance(raw_meta, dict) else None)
    except ValueError:
        po_ctx = None
    ctx_token = bind_po_fhir_context(po_ctx) if po_ctx is not None else None
    # Open a request-scoped FHIR bundle cache. Every `fetch_patient_bundle`
    # call inside this handler -- whether from the dispatcher's own
    # demographics extraction or from the dozen tool invocations a macro
    # workflow runs -- shares one fetch per patient_id. On a chart with
    # zero non-Patient resources (the deliberate negative-example case)
    # the underlying probe touches ~30 endpoints, all latency-dominated,
    # so cutting the multiplier from N steps to 1 saves the worst-case
    # tail.
    cache_token = begin_bundle_cache()
    try:
        result = await dispatch(prompt, metadata=metadata)
    finally:
        reset_bundle_cache(cache_token)
        if ctx_token is not None:
            release_po_fhir_context(ctx_token)

    if result.target_kind == "discovery":
        return result.rationale, [_artifact(
            "trustedrisk-orchestrator_catalog",
            result.output_dump or {},
            description="Capability catalog of the TrustedRisk Orchestrator.",
        )]

    if result.error:
        text = (
            f"I picked {result.target_label} for your prompt but the call "
            f"failed with {result.error}. Rationale: {result.rationale}"
        )
        return text, [_artifact(
            f"trustedrisk-orchestrator_error",
            {"target_kind": result.target_kind,
             "target_label": result.target_label,
             "target_slug": result.target_slug,
             "error": result.error,
             "rationale": result.rationale},
            description="Dispatch failure detail.",
        )]

    if result.target_kind == "fanout" and result.sub_results:
        # The chat client's LLM has been observed to read narrative
        # phrasing like "Routing to N capabilities in parallel" as
        # "still in progress" and promise the user a follow-up
        # notification. Phrase the response as a single completion
        # announcement, inline every sub-result's headline, and state
        # explicitly that nothing else is pending.
        sub_lines: list[str] = []
        for sub in result.sub_results:
            verdict = sub.error or sub.rationale or "completed"
            sub_lines.append(
                f"- {sub.target_label}: {verdict.rstrip('.')}."
            )
        prov_summary = _format_bundle_provenance(result.sub_results)
        abstain_banner = _format_abstain_banner(result.sub_results)
        text = (
            f"{abstain_banner}Done. I ran {len(result.sub_results)} "
            f"TrustedRisk capabilities in parallel for this patient and "
            f"all of them have already returned a result. There is no "
            f"further work pending.{prov_summary} Sub-results below, "
            f"with the full structured payloads in the attached "
            f"artifacts:\n"
            + "\n".join(sub_lines)
        )
        artifacts: list[dict] = [_artifact(
            "trustedrisk-orchestrator_fanout_summary",
            result.output_dump or {},
            description=(
                "Final fan-out summary. All sub-calls completed "
                "synchronously; nothing further is pending."
            ),
        )]
        for sub in result.sub_results:
            artifact_name = (
                f"trustedrisk-orchestrator_{sub.target_kind}_"
                f"{(sub.target_slug or 'result').replace(':', '_')}"
            )
            artifacts.append(_artifact(
                artifact_name,
                sub.output_dump or {"value": str(sub.output)},
                description=(
                    f"Final {sub.target_kind} sub-result for "
                    f"{sub.target_label}."
                ),
            ))
        return text, artifacts

    prov_summary = _format_bundle_provenance(
        [result] if result.output_dump else []
    )
    abstain_banner = _format_abstain_banner(
        [result] if result.output_dump else []
    )
    # Headline findings front-load decision-relevant numbers (ESI level,
    # top antibiotic pick, gap counts, etc.) so they survive the chat
    # client's aggressive output truncation. Empty string when the
    # workflow has no nested sub-results worth highlighting.
    headline = _format_macro_headlines(result.output_dump)
    text = (
        f"{abstain_banner}{headline}Done. Ran {result.target_label} on this "
        f"patient and the call has already returned. "
        f"{result.rationale}{prov_summary} "
        f"The full structured result is in the attached artifact; "
        f"nothing further is pending."
    )
    artifact_name = (
        f"trustedrisk-orchestrator_{result.target_kind}_"
        f"{(result.target_slug or 'result').replace(':', '_')}"
    )
    return text, [_artifact(
        artifact_name,
        result.output_dump or {"value": str(result.output)},
        description=(
            f"Final {result.target_kind} dispatch result for the "
            f"active patient."
        ),
    )]


@app.post("/")
async def a2a_messaging(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400,
                              detail=f"invalid_json: {exc}") from exc
    resp = await handle_jsonrpc(body, _orchestrator_handler)
    _append_trace(body, resp)
    return resp


@app.get("/healthz")
async def healthz():
    from apps._shared.specialist_routes import SPECIALIST_ROUTES
    from apps.composer.workflows import REGISTRY
    return {
        "status": "ok",
        "n_workflows": len(REGISTRY),
        "n_specialists": len(SPECIALIST_ROUTES),
        "n_routes": sum(len(v) for v in SPECIALIST_ROUTES.values()),
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
    port = int(os.environ.get("TRUSTEDRISK_ORCHESTRATOR_PORT", "8787"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
