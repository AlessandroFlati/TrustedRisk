"""SHARP-on-MCP header extraction + context propagation.

Per spec at https://www.sharponmcp.com/key-components.html: the server
REQUIRES FHIR context headers on every non-public request:

  - X-FHIR-Server-URL          (always required)
  - X-FHIR-Access-Token        (required when token-auth applies)
  - X-Patient-ID               (optional; required by individual tools)
  - X-FHIR-Refresh-Token       (optional; offline-access mode)
  - X-FHIR-Refresh-Url         (optional; offline-access mode)

Missing required headers -> 403 with structured error body pointing to
the spec.

The capability advertisement (`initialize_capabilities`) emits BOTH the
vendor-neutral SHARP draft path
(`capabilities.experimental.fhir_context_required`) AND the Prompt
Opinion namespaced path (`capabilities.ai.promptopinion/fhir-context`)
with the SMART scopes derived from `mcp_server.scopes`.
"""

from __future__ import annotations

import json as _json
from contextvars import ContextVar
from dataclasses import dataclass, field

from starlette.requests import Request
from starlette.responses import JSONResponse


# JSON-RPC method names that are part of the MCP framework lifecycle and
# do NOT require FHIR context. Capability discovery (`initialize`) is the
# precise moment the client *learns* that FHIR-context is required, so
# blocking it would create a chicken-and-egg deadlock where the client
# can never discover the requirement that prevents it from connecting.
# Only `tools/call` actually fans out to a tool handler and therefore
# needs the SHARP context populated.
_FRAMEWORK_RPC_METHODS: frozenset[str] = frozenset({
    "initialize",
    "initialized",
    "ping",
    "tools/list",
    "prompts/list",
    "prompts/get",
    "resources/list",
    "resources/read",
    "resources/templates/list",
    "resources/subscribe",
    "resources/unsubscribe",
    "logging/setLevel",
    "completion/complete",
})


@dataclass
class FHIRContext:
    """Parsed from request headers; accessible via get_fhir_context() inside tools.

    `refresh_token` and `refresh_token_url` are populated when the SHARP
    client uses the offline-access mode -- see SHARP §3.4 + Prompt
    Opinion's `mcp-fhir-context.html` for the refresh contract.
    """
    server_url: str
    access_token: str
    patient_id: str | None = None
    refresh_token: str | None = None
    refresh_token_url: str | None = None


_fhir_ctx: ContextVar[FHIRContext] = ContextVar("fhir_ctx")


def get_fhir_context() -> FHIRContext:
    """Retrieve the current request's FHIR context. Raises LookupError outside a request."""
    return _fhir_ctx.get()


def rpc_requires_fhir_context(raw: bytes) -> bool:
    """Decide whether a JSON-RPC payload's method needs SHARP enforcement.

    `tools/call` populates the ContextVar that tools read; framework
    methods (initialize, tools/list, ping, ...) don't, and blocking them
    deadlocks capability discovery (the client only learns FHIR context
    is required by reading the `initialize` response).

    Unparseable payloads -> True (fail-safe: enforce by default).
    """
    if not raw:
        return False
    try:
        msg = _json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return True

    def _needs(m: object) -> bool:
        if not isinstance(m, dict):
            return False
        method = m.get("method")
        if not isinstance(method, str):
            return False
        if method in _FRAMEWORK_RPC_METHODS:
            return False
        if method.startswith("notifications/"):
            return False
        return True

    if isinstance(msg, list):
        return any(_needs(m) for m in msg)
    return _needs(msg)


async def sharp_context_middleware(request: Request, call_next):
    """Middleware that extracts SHARP headers and populates the ContextVar.

    Returns 403 with spec_reference on missing required headers.
    Refresh-token headers are optional (offline-access mode) and never
    cause a 403 on their own.

    NOTE: this function is the simple BaseHTTPMiddleware-style entrypoint
    used by unit tests. The production HTTP stack uses a pure-ASGI
    wrapper (`SharpASGIMiddleware` in `mcp_server.server`) that also
    inspects the JSON-RPC body to exempt framework methods -- body peeking
    requires ASGI-level access that BaseHTTPMiddleware doesn't expose.
    """
    server_url = request.headers.get("X-FHIR-Server-URL", "").strip()
    access_token = request.headers.get("X-FHIR-Access-Token", "").strip()
    patient_id = request.headers.get("X-Patient-ID", "").strip() or None
    refresh_token = (
        request.headers.get("X-FHIR-Refresh-Token", "").strip() or None
    )
    refresh_token_url = (
        request.headers.get("X-FHIR-Refresh-Url", "").strip() or None
    )

    missing: list[str] = []
    if not server_url:
        missing.append("X-FHIR-Server-URL")
    if not access_token:
        missing.append("X-FHIR-Access-Token")
    # X-Patient-ID is NOT in `missing` -- per spec, it's optional (detect_phi doesn't need it).
    # Individual tools that require it raise at tool-invocation time via resolve_patient_id().

    if missing:
        body = {
            "error": "missing_fhir_context",
            "message": "SHARP-on-MCP requires FHIR context headers.",
            "missing_headers": missing,
            "spec_reference": "https://www.sharponmcp.com/key-components.html",
        }
        return JSONResponse(status_code=403, content=body)

    # Pairing rule: if one of the offline-access headers is present, both
    # SHOULD be -- log via the body but do not 403 (offline-access is opt-in).
    if (refresh_token and not refresh_token_url) or (
            refresh_token_url and not refresh_token):
        # Permissive: accept the partial offline-access declaration; the
        # server simply won't be able to refresh until the missing piece
        # arrives. A future spec revision may tighten this to 400.
        pass

    token = _fhir_ctx.set(FHIRContext(
        server_url=server_url,
        access_token=access_token,
        patient_id=patient_id,
        refresh_token=refresh_token,
        refresh_token_url=refresh_token_url,
    ))
    try:
        response = await call_next(request)
        return response
    finally:
        _fhir_ctx.reset(token)


def initialize_capabilities() -> dict:
    """Return the SHARP-on-MCP capability declaration for the MCP initialize response.

    Emits BOTH the vendor-neutral SHARP draft key
    (`experimental.fhir_context_required`) AND the Prompt-Opinion
    namespaced key (`ai.promptopinion/fhir-context`) so a workspace
    that consumes either spec sees the same intent.

    The scopes array is derived from the union of all 20 bundles --
    `mcp_server.scopes.scope_objects()` produces the
    `[{"name": "...", "required": true}, ...]` shape that PO docs
    publish.
    """
    # Lazy import -- `mcp_server.scopes` has no other dependencies but
    # keeping it lazy avoids a circular import if scopes ever needs to
    # introspect tool registration.
    from ..scopes import scope_objects

    scopes = scope_objects()

    return {
        "experimental": {
            "fhir_context_required": {
                "value": True,
                "headers": [
                    "X-FHIR-Server-URL",
                    "X-FHIR-Access-Token",
                    "X-Patient-ID",
                ],
                "headers_offline_access": [
                    "X-FHIR-Refresh-Token",
                    "X-FHIR-Refresh-Url",
                ],
                "patient_id_required": False,  # required per-tool, not globally
            }
        },
        # Prompt Opinion namespaced capability -- see
        # https://docs.promptopinion.ai/fhir-context/mcp-fhir-context.html
        "ai.promptopinion/fhir-context": {
            "scopes": scopes,
        },
    }
