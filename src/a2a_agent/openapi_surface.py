"""Phase 12.9 A2 -- OpenAPI 3.1 surface for the federation.

Auto-generates an OpenAPI 3.1 schema per specialist by walking:

  1. The specialist's `agent_card.json` (`_bundles` map -> tool list).
  2. Each tool function's Python signature.
  3. The Pydantic output schema (when the return annotation is a
     subclass of `BaseModel`).

The spec is mounted at `/openapi.json`; a minimal Swagger UI is served
at `/docs`. Both paths are under the existing `_PUBLIC_PATHS` allow-list
so they don't trigger the SHARP / OAuth middleware.

This is a *documentation* surface -- actual tool dispatch goes through
the FastMCP `/mcp` endpoint or the A2A endpoint. The OpenAPI doc
declares the synthetic `POST /tools/{tool_name}` shape so an EHR /
marketplace UI can browse the catalogue.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, get_args, get_origin

from starlette.responses import HTMLResponse, JSONResponse


# ─────────────────────────────────────────────────────────────────────
# Schema introspection
# ─────────────────────────────────────────────────────────────────────


def _python_type_to_openapi(t: Any) -> dict[str, Any]:
    """Best-effort Python annotation -> OpenAPI primitive mapping.

    Handles `int`, `float`, `bool`, `str`, `list[X]`, `dict`, optional,
    Pydantic models (delegates to `model_json_schema`)."""
    if t is None or t is type(None):
        return {"type": "null"}
    if t is bool:
        return {"type": "boolean"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is str:
        return {"type": "string"}
    if t is dict or t is Any or t is None:
        return {"type": "object"}

    origin = get_origin(t)
    args = get_args(t) or ()

    if origin is list:
        item_t = args[0] if args else Any
        return {"type": "array", "items": _python_type_to_openapi(item_t)}
    if origin is dict:
        return {"type": "object"}
    if origin is type or origin is None and hasattr(t, "model_json_schema"):
        try:
            return t.model_json_schema()
        except Exception:
            return {"type": "object"}

    # Union / Optional
    if origin is type(None) or (
        origin is None and hasattr(t, "__args__")
    ):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_openapi(non_none[0])
        return {
            "oneOf": [_python_type_to_openapi(a) for a in non_none],
        }

    # Pydantic model
    if hasattr(t, "model_json_schema"):
        try:
            return t.model_json_schema()
        except Exception:
            return {"type": "object"}

    return {"type": "string"}


def _signature_to_request_schema(fn: Any) -> dict[str, Any]:
    """Turn a Python function signature into an OpenAPI request body
    schema."""
    sig = inspect.signature(fn)
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        anno = p.annotation if p.annotation is not inspect.Parameter.empty \
            else str
        properties[name] = _python_type_to_openapi(anno)
        if p.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def _resolve_tool_function(tool_name: str) -> Any | None:
    """Look up a tool function by name across the registered modules."""
    import mcp_server.tools as tools_pkg
    for mod_name in tools_pkg.__all__:
        mod = getattr(tools_pkg, mod_name, None)
        if mod is None:
            continue
        fn = getattr(mod, tool_name, None)
        if callable(fn):
            return fn
    return None


# ─────────────────────────────────────────────────────────────────────
# Spec assembly
# ─────────────────────────────────────────────────────────────────────


def build_openapi_spec(
    agent_card_path: str | Path,
) -> dict[str, Any]:
    """Build the OpenAPI 3.1 spec for one specialist.

    The spec ships:
      - `info` block sourced from the agent-card.
      - `/healthz`, `/readyz`, `/.well-known/agent-card.json` paths
        (operational endpoints).
      - One `POST /tools/{tool_name}` operation per tool advertised in
        the specialist's `_bundles`.
    """
    card = json.loads(Path(agent_card_path).read_text(encoding="utf-8"))
    info = {
        "title": card.get("name", "trustedrisk"),
        "version": card.get("version", "1.0.0"),
        "description": card.get("description", "")[:1500],
    }

    paths: dict[str, Any] = {
        "/healthz": {
            "get": {
                "summary": "Liveness probe",
                "responses": {"200": {"description": "OK"}},
            },
        },
        "/readyz": {
            "get": {
                "summary": "Readiness probe",
                "responses": {"200": {"description": "OK"}},
            },
        },
        "/.well-known/agent-card.json": {
            "get": {
                "summary": "A2A v1 agent-card discovery",
                "responses": {"200": {"description": "Agent card JSON"}},
            },
        },
    }

    bundles = card.get("_bundles", {}) or {}
    seen_tools: set[str] = set()
    for bundle_id, tools in bundles.items():
        if bundle_id.startswith("_"):
            continue
        if not isinstance(tools, list):
            continue
        for tool_name in tools:
            if tool_name in seen_tools:
                continue
            seen_tools.add(tool_name)
            fn = _resolve_tool_function(tool_name)
            if fn is None:
                continue
            request_schema = _signature_to_request_schema(fn)
            description = (inspect.getdoc(fn) or "").split("\n\n", 1)[0]
            paths[f"/tools/{tool_name}"] = {
                "post": {
                    "tags": [bundle_id],
                    "summary": tool_name,
                    "description": description[:500],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": request_schema,
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Tool output (Pydantic model)",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"},
                                },
                            },
                        },
                    },
                },
            }

    return {
        "openapi": "3.1.0",
        "info": info,
        "servers": [
            {"url": card.get("url", "http://localhost:0"),
             "description": "Specialist base URL"},
        ],
        "paths": paths,
    }


# ─────────────────────────────────────────────────────────────────────
# Starlette route helpers
# ─────────────────────────────────────────────────────────────────────


def make_openapi_route(agent_card_path: str | Path):
    """Return a Starlette endpoint that serves the OpenAPI 3.1 spec."""
    spec_cache: dict[str, Any] | None = None

    async def _endpoint(request):
        nonlocal spec_cache
        if spec_cache is None:
            spec_cache = build_openapi_spec(agent_card_path)
        return JSONResponse(content=spec_cache)
    return _endpoint


_SWAGGER_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>{title} -- OpenAPI</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({{
      url: '/openapi.json',
      dom_id: '#swagger-ui',
      docExpansion: 'list',
    }});
  </script>
</body>
</html>"""


def make_docs_route(agent_card_path: str | Path):
    """Return a Starlette endpoint serving a minimal Swagger UI page."""
    title_cache: str | None = None

    async def _endpoint(request):
        nonlocal title_cache
        if title_cache is None:
            try:
                card = json.loads(
                    Path(agent_card_path).read_text(encoding="utf-8"),
                )
                title_cache = card.get("name", "TrustedRisk")
            except Exception:
                title_cache = "TrustedRisk"
        return HTMLResponse(content=_SWAGGER_HTML.format(title=title_cache))
    return _endpoint
