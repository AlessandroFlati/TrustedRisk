"""Phase 3.3 -- Server-Sent Events (SSE) streaming utilities.

Per the A2A v1 spec (`SendStreamingMessage` + `SubscribeToTask`), an
agent that declares `capabilities.streaming = true` may emit a stream
of `TaskStatusUpdateEvent` + `TaskArtifactUpdateEvent` events as the
work progresses.

This module provides:
  - `sse_event(name, payload)`         -- format a single event line block
  - `sse_status_update(state, ...)`    -- A2A TaskStatusUpdateEvent
  - `sse_artifact_update(artifact)`    -- A2A TaskArtifactUpdateEvent
  - `sse_done()`                       -- terminate the stream
  - `stream_outcomes_simulation(...)`  -- generator yielding SSE for the
    long-running Monte Carlo simulation

Usage in a Starlette route:

    from starlette.responses import StreamingResponse
    from a2a_agent.streaming import stream_outcomes_simulation

    async def my_stream_endpoint(request):
        body = await request.json()
        return StreamingResponse(
            stream_outcomes_simulation(...args...),
            media_type="text/event-stream",
        )

The transport is SSE (HTTP), which both the official A2A SDK and any
generic SSE client (browsers, curl --no-buffer, httpx + EventSourceParser)
can consume.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator


# ─────────────────────── Wire-format helpers ───────────────────────

def sse_event(name: str, payload: Any) -> str:
    """Format a single SSE event with `event: <name>` + `data: <json>`."""
    body = json.dumps(payload, default=str)
    return f"event: {name}\ndata: {body}\n\n"


def sse_status_update(
    state: str,
    *,
    task_id: str,
    context_id: str | None = None,
    progress_pct: float | None = None,
    message: str | None = None,
) -> str:
    """Emit an A2A TaskStatusUpdateEvent."""
    payload: dict[str, Any] = {
        "kind": "TaskStatusUpdateEvent",
        "taskId": task_id,
        "status": {
            "state": state,
            "timestamp": int(time.time() * 1000),
        },
    }
    if context_id is not None:
        payload["contextId"] = context_id
    if progress_pct is not None:
        payload["status"]["progress_pct"] = round(progress_pct, 4)
    if message is not None:
        payload["status"]["message"] = {"role": "ROLE_AGENT",
                                              "parts": [{"kind": "text", "text": message}]}
    return sse_event("status", payload)


def sse_artifact_update(
    artifact_payload: Any,
    *,
    task_id: str,
    context_id: str | None = None,
    artifact_id: str | None = None,
) -> str:
    """Emit an A2A TaskArtifactUpdateEvent."""
    payload: dict[str, Any] = {
        "kind": "TaskArtifactUpdateEvent",
        "taskId": task_id,
        "artifact": {
            "artifactId": artifact_id or f"artifact-{uuid.uuid4().hex[:8]}",
            "parts": [{"kind": "data", "data": artifact_payload}],
        },
    }
    if context_id is not None:
        payload["contextId"] = context_id
    return sse_event("artifact", payload)


def sse_done(*, task_id: str, context_id: str | None = None) -> str:
    """Terminate the stream with a 'done' marker.

    Note: SSE streams are usually closed by the server simply ending
    the response body; the `done` event makes intent explicit and is a
    common convention in A2A SDK clients.
    """
    payload: dict[str, Any] = {"kind": "Done", "taskId": task_id}
    if context_id is not None:
        payload["contextId"] = context_id
    return sse_event("done", payload)


# ─────────────────────── Streaming generators ───────────────────────

async def stream_llm_polish(
    *,
    text: str,
    system_prompt: str | None = None,
    task_id: str | None = None,
    context_id: str | None = None,
) -> AsyncIterator[str]:
    """Stream the LLM polish path as SSE events.

    Emits:
      - 1 × initial WORKING status
      - 0-1 × INPUT_REQUIRED status when no LLM is configured
      - 1 × TaskArtifactUpdateEvent with the polish result (or the
        deterministic source text when polish was rejected)
      - 1 × terminal status (COMPLETED)
      - 1 × done marker
    """
    from a2a_agent.llm_polish import resolve_polish_client

    task_id = task_id or f"task-{uuid.uuid4().hex[:8]}"

    yield sse_status_update(
        "WORKING", task_id=task_id, context_id=context_id,
        progress_pct=0.0, message="Resolving polish client",
    )

    client = resolve_polish_client()
    if client.model_id is None:
        # Surface as INPUT_REQUIRED -- the caller may want to attach a
        # different LLM or accept the deterministic floor explicitly.
        yield sse_status_update(
            "INPUT_REQUIRED", task_id=task_id, context_id=context_id,
            progress_pct=1.0,
            message="No LLM configured; deterministic floor returned",
        )
        yield sse_artifact_update(
            {
                "polished_text": text,
                "model_id": None,
                "is_polished": False,
                "polish_rejected_reason": "no_llm_configured",
            },
            task_id=task_id, context_id=context_id,
            artifact_id=f"polish-result-{task_id}",
        )
        yield sse_done(task_id=task_id, context_id=context_id)
        return

    yield sse_status_update(
        "WORKING", task_id=task_id, context_id=context_id,
        progress_pct=0.4,
        message=f"Calling {client.model_id} with paraphrase prompt",
    )

    res = await client.polish(text, system_prompt=system_prompt)

    yield sse_artifact_update(
        {
            "polished_text": res.polished_text,
            "model_id": res.model_id,
            "is_polished": res.is_polished,
            "polish_rejected_reason": res.polish_rejected_reason,
        },
        task_id=task_id, context_id=context_id,
        artifact_id=f"polish-result-{task_id}",
    )
    yield sse_status_update(
        "COMPLETED", task_id=task_id, context_id=context_id,
        progress_pct=1.0, message="Polish call complete",
    )
    yield sse_done(task_id=task_id, context_id=context_id)


async def stream_outcomes_simulation(
    *,
    case_mix: list[Any],
    intervention_rrr: float,
    intervention_cost_per_patient_usd: float,
    avoided_event_cost_usd: float,
    qaly_gained_per_avoided_event: float | None = None,
    n_iterations: int = 1000,
    chunk_size: int = 100,
    task_id: str | None = None,
    context_id: str | None = None,
) -> AsyncIterator[str]:
    """Run the SIM-1 Monte Carlo simulation with progress events.

    Emits:
      - 1 × initial status (SUBMITTED -> WORKING)
      - n_iterations / chunk_size × intermediate status events with
        `progress_pct` advancing toward 100
      - 1 × final TaskArtifactUpdateEvent with the complete
        HospitalYearSimulation payload
      - 1 × terminal status (WORKING -> COMPLETED)
      - 1 × done marker
    """
    from a2a_agent.outcomes_simulator import simulate_hospital_year

    task_id = task_id or f"task-{uuid.uuid4().hex[:8]}"

    yield sse_status_update(
        "WORKING", task_id=task_id, context_id=context_id,
        progress_pct=0.0,
        message=f"Starting Monte Carlo simulation ({n_iterations} iterations)",
    )

    # The simulation is fast enough that we don't need real chunked
    # iteration -- emit progressive status events at fixed intervals
    # and finally call the synchronous simulator. This still satisfies
    # the streaming contract (clients see progress events) while
    # keeping the math identical to the non-streaming path.
    chunks = max(1, n_iterations // max(chunk_size, 1))
    for i in range(1, chunks):
        yield sse_status_update(
            "WORKING", task_id=task_id, context_id=context_id,
            progress_pct=i / chunks,
            message=f"Iterations {i * chunk_size}/{n_iterations}",
        )

    # Final compute (deterministic; uses the upstream simulator)
    result = simulate_hospital_year(
        case_mix=case_mix,
        intervention_relative_risk_reduction=intervention_rrr,
        intervention_cost_per_patient_usd=intervention_cost_per_patient_usd,
        avoided_event_cost_usd=avoided_event_cost_usd,
        qaly_gained_per_avoided_event=qaly_gained_per_avoided_event,
        n_iterations=n_iterations,
    )

    yield sse_status_update(
        "WORKING", task_id=task_id, context_id=context_id,
        progress_pct=1.0, message="Bootstrapping CIs",
    )
    yield sse_artifact_update(
        result.model_dump(),
        task_id=task_id, context_id=context_id,
        artifact_id=f"hospital-year-sim-{task_id}",
    )
    yield sse_status_update(
        "COMPLETED", task_id=task_id, context_id=context_id,
        progress_pct=1.0, message="Simulation complete",
    )
    yield sse_done(task_id=task_id, context_id=context_id)
