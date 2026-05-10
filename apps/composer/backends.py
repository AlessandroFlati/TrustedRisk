"""Pluggable execution backends for the Care Engine.

A workflow step has two equivalent execution paths:

1. **In-process** (default). The orchestrator imports the tool callable
   directly from `mcp_server.tools.*` and awaits it. Fast, deterministic,
   single-process. The 17 specialist agent cards exist for marketplace
   discoverability; the runtime never crosses an HTTP boundary.

2. **A2A** (real agent-composition). Each step becomes a
   `tools/call` JSON-RPC request to the specialist's MCP endpoint over
   HTTP via FastMCP's `StreamableHttpTransport`. The orchestrator and
   specialist are now genuinely separate agents communicating through
   the published A2A v1 surface; the FHIR-context propagates through
   SHARP headers exactly as a marketplace consumer would see it.

The selector is read once at workflow start from the
`TRUSTEDRISK_COMPOSER_BACKEND` env var:

  - unset / empty / `inprocess` -> InProcessBackend
  - `a2a`                       -> A2ABackend

A2ABackend reads the federation base URL from
`TRUSTEDRISK_FEDERATION_BASE_URL` (default `http://127.0.0.1:8765`,
matching the locally-running stack). Specialist slug -> URL mapping is
inherited from `a2a_agent.federation_registry._SPECIALIST_PORTS` for
single-port-per-specialist deployments and from the federation umbrella
mount path (`/a2a/<slug>/mcp`) when the umbrella is the entry point.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from a2a_agent.federation_registry import _SPECIALIST_PORTS


class WorkflowBackend(Protocol):
    """Strategy for invoking a single workflow step."""

    name: str

    async def call(
        self,
        specialist: str,
        tool_name: str,
        callable_: Callable[..., Awaitable[Any]],
        resolved_inputs: dict,
    ) -> Any:
        ...


@dataclass(frozen=True)
class InProcessBackend:
    """Default backend: import the tool function and await it.

    Inputs are filtered to the tool's actual parameter names before the
    call: the dispatcher always-propagates patient_id / encounter_id /
    other FHIR-context keys onto every step's inputs dict, but a tool
    that does not declare those keys in its signature would otherwise
    crash with `TypeError: ... got an unexpected keyword argument`.
    Filtering keeps the dispatcher contract loose (always-propagate)
    while keeping each tool's signature tight.
    """

    name: str = "inprocess"

    async def call(
        self,
        specialist: str,
        tool_name: str,
        callable_: Callable[..., Awaitable[Any]],
        resolved_inputs: dict,
    ) -> Any:
        import inspect
        try:
            sig = inspect.signature(callable_)
        except (TypeError, ValueError):
            return await callable_(**resolved_inputs)
        accepts_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if accepts_var_keyword:
            return await callable_(**resolved_inputs)
        accepted = {
            k: v for k, v in resolved_inputs.items()
            if k in sig.parameters
        }
        return await callable_(**accepted)


@dataclass(frozen=True)
class A2ABackend:
    """HTTP backend: dispatch each step as a tools/call across the A2A surface.

    Resolves specialist URL via two strategies in order:
      1. Federation umbrella (`{base}/a2a/{slug}/mcp`) when
         `TRUSTEDRISK_FEDERATION_BASE_URL` is set.
      2. Single-port deployment (`http://127.0.0.1:{port}/mcp`) using
         `_SPECIALIST_PORTS`.
    """

    name: str = "a2a"
    federation_base_url: str | None = None
    timeout_s: float = 30.0

    def _url_for(self, specialist: str) -> str:
        base = self.federation_base_url or os.environ.get(
            "TRUSTEDRISK_FEDERATION_BASE_URL"
        )
        if base:
            return f"{base.rstrip('/')}/a2a/{specialist}/mcp"
        port = _SPECIALIST_PORTS.get(specialist)
        if port is None:
            raise ValueError(
                f"A2ABackend: no port mapping for specialist {specialist!r}; "
                f"set TRUSTEDRISK_FEDERATION_BASE_URL or extend _SPECIALIST_PORTS."
            )
        return f"http://127.0.0.1:{port}/mcp"

    def _headers(self) -> dict:
        """Forward the SHARP context from the current FHIRContext ContextVar.

        When the orchestrator runs inside a SHARP-bound request its
        ContextVar carries the inbound FHIR credentials; we pass them
        through to the specialist so the downstream tool sees the same
        chart. Missing context is allowed: tools that don't need a chart
        (PHI scrub, parsers, scoring on inline inputs) succeed without
        it; tools that do will raise on their own.
        """
        try:
            from mcp_server.sharp.headers import get_fhir_context
            ctx = get_fhir_context()
        except (ImportError, LookupError):
            return {}
        h = {
            "X-FHIR-Server-URL": ctx.server_url,
            "X-FHIR-Access-Token": ctx.access_token,
        }
        if ctx.patient_id:
            h["X-Patient-ID"] = ctx.patient_id
        if ctx.refresh_token:
            h["X-FHIR-Refresh-Token"] = ctx.refresh_token
        if ctx.refresh_token_url:
            h["X-FHIR-Refresh-Url"] = ctx.refresh_token_url
        return h

    async def call(
        self,
        specialist: str,
        tool_name: str,
        callable_: Callable[..., Awaitable[Any]],
        resolved_inputs: dict,
    ) -> Any:
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport

        url = self._url_for(specialist)
        transport = StreamableHttpTransport(url=url, headers=self._headers())
        async with Client(transport=transport) as client:
            result = await client.call_tool(tool_name, arguments=resolved_inputs)
        return _unpack_result(result)


def _unpack_result(result: Any) -> Any:
    """Best-effort extraction of the tool's return payload.

    FastMCP wraps the response in a CallToolResult that contains a
    `content` list of TextContent / ImageContent items. For our tools
    (which return Pydantic models serialized as JSON text) we expect a
    single TextContent whose `.text` is the JSON dump. We return the
    parsed dict so the workflow's chain-resolver navigates it the same
    way it would a Pydantic model_dump.
    """
    content = getattr(result, "content", None)
    if isinstance(content, list) and content:
        first = content[0]
        text = getattr(first, "text", None)
        if isinstance(text, str):
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return {"_raw_text": text}
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    return result


def default_backend() -> WorkflowBackend:
    """Pick the backend for the current process based on env config."""
    selector = (os.environ.get("TRUSTEDRISK_COMPOSER_BACKEND") or "").strip().lower()
    if selector == "a2a":
        return A2ABackend()
    return InProcessBackend()
