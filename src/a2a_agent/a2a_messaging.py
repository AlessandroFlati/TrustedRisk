"""A2A v1 JSON-RPC messaging shared by every TrustedRisk agent endpoint.

The A2A spec maps `message/send` to a JSON-RPC 2.0 method. Clients
(Prompt Opinion's chat, Microsoft Copilot Studio, Google Gemini, ...)
POST a JSON-RPC envelope to the URL declared in the agent card's
supportedInterfaces[].url:

    {
      "jsonrpc": "2.0",
      "id": "<request-id>",
      "method": "message/send",
      "params": {"message": {"role": "user", "parts": [{"text": "..."}], ...}}
    }

This module provides:
  - handle_jsonrpc(body, handler): the dispatcher every agent reuses.
  - build_composer_handler():    keyword router over the 5 workflow templates.
  - build_specialist_handler():  agent-card-driven skill summary + best-effort
                                 tool execution from the bundle.
  - build_root_handler():        introduces the root MCP and points the user
                                 at the right specialist for their prompt.

Every handler returns (response_text, artifacts) and the dispatcher wraps
that into a Task with state="completed".
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

# Handler signature: (prompt_text, raw_message_dict) -> (response_text, artifacts)
AgentHandler = Callable[[str, dict], Awaitable[tuple[str, list[dict]]]]


# --------------------------------------------------------------------- helpers


def _extract_prompt_text(message: dict) -> str:
    parts = message.get("parts") or []
    chunks: list[str] = []
    for p in parts:
        if isinstance(p, dict) and isinstance(p.get("text"), str):
            chunks.append(p["text"])
    return "\n".join(chunks).strip()


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _make_task(
    *,
    response_text: str,
    artifacts: list[dict],
    context_id: str | None,
    proto_style: bool = False,
) -> dict:
    """Build a Task payload.

    A2A v1 has two interop dialects on the wire:

    - **SDK / spec-text dialect** (slash-style methods like `message/send`):
      Pydantic-style with explicit `kind` discriminators on Task / Message /
      Part for oneof, plus lowercase enum values (`completed`, `agent`).
    - **proto3 JSON dialect** (proto rpc names like `SendMessage`):
      strictly follows protobuf JSON mapping -- oneofs serialised as the
      selected field name only (no `kind`), enums serialised as their
      fully-qualified name (`TASK_STATE_COMPLETED`, `ROLE_AGENT`).

    Prompt Opinion uses proto3 JSON. Pick the dialect from the inbound
    method name so both client families work.
    """
    task_id = str(uuid.uuid4())
    ctx = context_id or str(uuid.uuid4())
    if proto_style:
        response_msg = {
            "messageId": str(uuid.uuid4()),
            "contextId": ctx,
            "taskId": task_id,
            "role": "ROLE_AGENT",
            "parts": [{"text": response_text}],
        }
        return {
            "id": task_id,
            "contextId": ctx,
            "status": {
                "state": "TASK_STATE_COMPLETED",
                "message": response_msg,
                "timestamp": _iso_now(),
            },
            "artifacts": [_artifact_to_proto(a) for a in artifacts],
            "history": [],
        }
    response_msg = {
        "kind": "message",
        "messageId": str(uuid.uuid4()),
        "contextId": ctx,
        "taskId": task_id,
        "role": "agent",
        "parts": [{"kind": "text", "text": response_text}],
    }
    return {
        "kind": "task",
        "id": task_id,
        "contextId": ctx,
        "status": {
            "state": "completed",
            "message": response_msg,
            "timestamp": _iso_now(),
        },
        "artifacts": artifacts,
        "history": [],
    }


def _artifact_to_proto(artifact: dict) -> dict:
    """Convert an SDK-style artifact into proto3 JSON shape.

    Strips `kind` from each Part and renames the SDK `parts` field to the
    proto Artifact field, while keeping the oneof bare (`{"text": ...}`
    or `{"data": ...}` or `{"file": ...}`).
    """
    out = {
        "artifactId": artifact.get("artifactId"),
        "name": artifact.get("name", ""),
        "description": artifact.get("description", ""),
    }
    proto_parts: list[dict] = []
    for p in artifact.get("parts") or []:
        if not isinstance(p, dict):
            continue
        clean = {k: v for k, v in p.items() if k != "kind"}
        proto_parts.append(clean)
    out["parts"] = proto_parts
    return out


def _ok(rpc_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _err(rpc_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


# --------------------------------------------------------------------- dispatch


async def handle_jsonrpc(body: Any, handler: AgentHandler) -> dict:
    """Dispatch a JSON-RPC 2.0 envelope to the agent's prompt handler."""
    response = await _dispatch_jsonrpc(body, handler)
    import os as _os
    if _os.environ.get("TRUSTEDRISK_A2A_DEBUG_TRACE"):
        try:
            tr = Path(_os.environ["TRUSTEDRISK_A2A_DEBUG_TRACE"])
            with tr.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "ts": _iso_now(),
                    "request": body,
                    "response": response,
                }, default=str))
                fh.write("\n")
        except OSError:
            pass
    return response


async def _dispatch_jsonrpc(body: Any, handler: AgentHandler) -> dict:
    if not isinstance(body, dict):
        return _err(None, -32700, "parse_error: body must be a JSON object")
    if body.get("jsonrpc") != "2.0":
        return _err(body.get("id"), -32600, "invalid_request: jsonrpc must be '2.0'")
    rpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params") or {}

    # A2A v1 has two equivalent JSON-RPC method-name conventions in the
    # wild: the slash-style `message/send` from the spec text and the
    # proto-style `SendMessage` derived from `a2a.proto`'s
    # `rpc SendMessage(SendMessageRequest)`. Prompt Opinion's chat client
    # emits the proto-style spelling; other clients (curl, the A2A
    # reference impls) emit the slash-style. Accept both.
    proto_style = method in ("SendMessage", "SendStreamingMessage")
    if method in ("message/send", "tasks/send", "SendMessage", "SendStreamingMessage"):
        msg = params.get("message") or {}
        if not isinstance(msg, dict):
            return _err(rpc_id, -32602, "invalid_params: message must be an object")
        prompt = _extract_prompt_text(msg)
        try:
            response_text, artifacts = await handler(prompt, msg)
        except Exception as exc:
            return _err(rpc_id, -32000, f"agent_error: {type(exc).__name__}: {exc}")
        task = _make_task(
            response_text=response_text,
            artifacts=artifacts,
            context_id=msg.get("contextId") or msg.get("context_id"),
            proto_style=proto_style,
        )
        if proto_style:
            # SendMessageResponse is a oneof of (task, msg). proto3 JSON
            # serialises a oneof as the selected field name only.
            return _ok(rpc_id, {"task": task})
        return _ok(rpc_id, task)

    # GetTask / ListTasks / CancelTask: we don't persist Task state across
    # requests (every SendMessage returns a `completed` Task synchronously),
    # so these queries have nothing to look up. Reply with a benign empty
    # result rather than method_not_found so polling clients don't error.
    if method in ("tasks/get", "GetTask"):
        task_id = (params.get("id") or params.get("taskId") or "")
        return _ok(rpc_id, {
            "kind": "task",
            "id": task_id,
            "contextId": params.get("contextId") or "",
            "status": {"state": "completed", "timestamp": _iso_now()},
            "artifacts": [],
            "history": [],
        })
    if method in ("tasks/list", "ListTasks"):
        return _ok(rpc_id, {"tasks": []})
    if method in ("tasks/cancel", "CancelTask"):
        return _ok(rpc_id, {
            "kind": "task",
            "id": params.get("id") or params.get("taskId") or "",
            "contextId": params.get("contextId") or "",
            "status": {"state": "canceled", "timestamp": _iso_now()},
            "artifacts": [],
            "history": [],
        })

    return _err(rpc_id, -32601, f"method_not_found: {method}")


# --------------------------------------------------------------------- handlers


def _artifact(name: str, data: dict, *, description: str | None = None) -> dict:
    return {
        "artifactId": str(uuid.uuid4()),
        "name": name,
        "description": description or "",
        "parts": [{"kind": "data", "data": data}],
    }


def build_composer_handler() -> AgentHandler:
    """Composer: keyword-route the prompt to one of the 5 workflow templates."""

    KEYWORDS = (
        ("chf_admission",       ("chf", "heart failure", "cardiac admission", "cardiology admission")),
        ("sepsis_workup",       ("sepsis", "septic", "sirs")),
        ("discharge_planning",  ("discharge planning", "discharge plan", "ready to go home", "ready for discharge")),
        ("outpatient_med_review", ("outpatient", "med review", "medication review", "polypharmacy review")),
        ("stroke_alert",        ("stroke", "tpa", "thrombolysis", "lkw")),
    )

    async def handler(prompt: str, msg: dict) -> tuple[str, list[dict]]:
        from apps.composer.workflows import REGISTRY
        from apps.composer.orchestrator import execute_workflow

        p = (prompt or "").lower()
        workflow_id = next(
            (wid for wid, kws in KEYWORDS if any(k in p for k in kws)),
            None,
        )

        if not workflow_id:
            ids = sorted(REGISTRY.keys())
            text = (
                "I'm the TrustedRisk composer. I orchestrate one of "
                f"{len(ids)} clinical workflows on the active patient: "
                f"{', '.join(ids)}. Try a prompt like 'run discharge planning', "
                "'sepsis workup for this patient', or 'chf admission'."
            )
            return text, [_artifact("available_workflows", {"workflow_ids": ids})]

        workflow = REGISTRY[workflow_id]
        # Build inputs strictly from the caller-supplied metadata. The
        # A2A handler no longer provides demo defaults; without a real
        # FHIR context the workflow abstains rather than running on
        # fabricated patient data.
        meta = msg.get("metadata") or {}
        if not isinstance(meta, dict) or not meta.get("fhir_server_url"):
            return (
                f"Workflow '{workflow_id}' requires a real FHIR context "
                f"(fhir_server_url in message metadata). No demo fixtures "
                f"are used as a fallback; please supply a live patient "
                f"context.",
                [_artifact("abstain", {
                    "workflow_id": workflow_id,
                    "abstain_recommended": True,
                    "abstain_reason": "no_fhir_context",
                })],
            )
        inputs: dict = {}
        for k, v in meta.items():
            if v is not None:
                inputs[k] = v
        # Backfill any required_input still missing with an empty value
        # of a plausible shape so the tool can abstain cleanly rather
        # than raising a KeyError on resolution.
        for k in workflow.required_inputs:
            inputs.setdefault(k, "")

        try:
            execution = await execute_workflow(workflow, inputs)
        except Exception as exc:
            return (
                f"Workflow '{workflow_id}' failed during execution: {type(exc).__name__}: {exc}.",
                [],
            )

        n_steps = len(execution.steps)
        ok_steps = sum(1 for s in execution.steps if not s.error)
        verdict = (
            f"ABSTAIN -- {execution.abstain_reason}"
            if execution.abstain_recommended
            else "no abstain triggered"
        )
        text = (
            f"Ran '{workflow.title}' ({ok_steps}/{n_steps} specialists "
            f"completed in {execution.duration_ms:.0f} ms). {verdict}. "
            f"Full per-step trace attached as an artifact."
        )

        trace_data = {
            "workflow_id": workflow_id,
            "workflow_title": workflow.title,
            "duration_ms": execution.duration_ms,
            "abstain_recommended": execution.abstain_recommended,
            "abstain_reason": execution.abstain_reason,
            "steps": [
                {
                    "step_id": s.step_id,
                    "specialist": s.specialist,
                    "tool_name": s.tool_name,
                    "duration_ms": s.duration_ms,
                    "skipped": s.skipped,
                    "error": s.error,
                    "output_dump": s.output_dump,
                }
                for s in execution.steps
            ],
        }
        return text, [_artifact(f"{workflow_id}_trace", trace_data,
                                 description=workflow.description)]

    return handler


def build_specialist_handler(agent_card_path: str | Path) -> AgentHandler:
    """Specialist: surface its skills + bundle inventory so the chat user
    knows what to ask. The actual tool dispatch lives under /mcp/* with a
    bearer token; this handler is the discovery layer.
    """
    card_path = Path(agent_card_path)

    async def handler(prompt: str, msg: dict) -> tuple[str, list[dict]]:
        if not card_path.exists():
            return (
                "Agent card not loaded; cannot describe my skills.",
                [],
            )
        card = json.loads(card_path.read_text(encoding="utf-8"))
        name = card.get("name", "specialist")
        slug = (card.get("url") or "").rstrip("/").rsplit("/", 1)[-1] or "agent"

        from apps._shared.specialist_routes import route_for
        route = route_for(slug, prompt)
        if route is not None:
            inputs = dict(route.inputs)
            meta = msg.get("metadata") or {}
            if isinstance(meta, dict):
                for k, v in meta.items():
                    if v is not None and k in inputs:
                        inputs[k] = v
            try:
                output = await route.callable(**inputs)
            except Exception as exc:
                text = (
                    f"I tried to run {route.skill_label} on the active "
                    f"patient but the call raised "
                    f"{type(exc).__name__}: {exc}. Inputs were "
                    f"{sorted(inputs.keys())!r}."
                )
                return text, [_artifact(
                    f"{slug}_{route.tool_name}_error",
                    {
                        "agent": name,
                        "tool": route.tool_name,
                        "skill": route.skill_label,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    description=(
                        f"{route.skill_label} call against the active "
                        f"patient failed before producing a result."
                    ),
                )]
            dump = output.model_dump() if hasattr(output, "model_dump") else output
            summary_bits: list[str] = []
            if isinstance(dump, dict):
                for f in route.summary_fields:
                    val = dump.get(f)
                    if val is not None:
                        summary_bits.append(f"{f}={val!r}")
            summary = "; ".join(summary_bits) or "see attached artifact for details"
            text = (
                f"Ran {route.skill_label} on the active patient. "
                f"{summary}. Full structured result attached."
            )
            return text, [_artifact(
                f"{slug}_{route.tool_name}",
                dump if isinstance(dump, dict) else {"value": str(dump)},
                description=(
                    f"{route.skill_label} structured output from "
                    f"{route.tool_name} on the active patient."
                ),
            )]

        # Discovery fallback: vague prompt or no route configured for slug.
        skills = card.get("skills") or []
        skill_names = [s.get("name") or s.get("id") or "?" for s in skills]
        bundles = [
            k for k in (card.get("_bundles") or {}) if not k.startswith("_")
        ]
        text = (
            f"I'm {name}. I cover {len(bundles)} clinical bundle(s) and "
            f"advertise {len(skills)} skill(s): {'; '.join(skill_names) or '(none)'}. "
            f"Ask me about any one of those (for example by name or "
            f"by what you want me to compute) and I'll run it on the "
            f"active patient."
        )
        return text, [_artifact(
            f"{slug}_capabilities",
            {
                "agent": name,
                "skills": [
                    {"id": s.get("id"), "name": s.get("name"),
                     "description": s.get("description")}
                    for s in skills
                ],
                "bundles": bundles,
                "mcp_endpoint": f"{card.get('url', '')}/mcp/v1",
            },
            description=f"Skill catalog and MCP tool endpoint for {name}.",
        )]

    return handler


def build_root_handler(agent_card_path: str | Path) -> AgentHandler:
    """Root MCP backend: 145 tools / 47 bundles, points to specialists."""
    card_path = Path(agent_card_path)

    async def handler(prompt: str, msg: dict) -> tuple[str, list[dict]]:
        try:
            from mcp_server.tools import BUNDLES
            n_tools = len({t for tools in BUNDLES.values() for t in tools})
            n_bundles = len(BUNDLES)
        except Exception:
            n_tools, n_bundles = 0, 0

        url = "https://flati.work/a2a/trustedrisk"
        if card_path.exists():
            try:
                card = json.loads(card_path.read_text(encoding="utf-8"))
                url = card.get("url", url)
            except (json.JSONDecodeError, OSError):
                pass

        text = (
            f"I'm trustedrisk-agent, the root MCP backend exposing {n_tools} "
            f"clinical tools across {n_bundles} thematic bundles. For task-"
            "specific help, consult one of the 15 specialists from the chat "
            "dropdown (trustedrisk-discharge / -acute / -evidence / etc.) or "
            "trustedrisk-composer for end-to-end workflows. For raw tool "
            f"access, my MCP transport is at {url}/mcp/v1 with an OAuth "
            "bearer; the surface is documented in the agent card."
        )
        return text, [_artifact(
            "root_topology",
            {
                "n_tools": n_tools,
                "n_bundles": n_bundles,
                "specialists": [
                    "trustedrisk-discharge", "trustedrisk-acute",
                    "trustedrisk-evidence", "trustedrisk-population",
                    "trustedrisk-pediatric", "trustedrisk-mental-health",
                    "trustedrisk-pa", "trustedrisk-scribe",
                    "trustedrisk-patient", "trustedrisk-coder",
                    "trustedrisk-pgx", "trustedrisk-preadmit",
                    "trustedrisk-quality", "trustedrisk-pophealth",
                    "trustedrisk-appeals", "trustedrisk-multimodal",
                ],
                "composer": "trustedrisk-composer",
            },
        )]

    return handler
