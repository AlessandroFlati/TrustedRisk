"""TrustedRisk MCP server entry point.

Starts FastMCP with Streamable HTTP transport + SHARP-on-MCP middleware
+ registers all 4 healthcare tools.

Run:
  $ python -m mcp_server.server
  $ trustedrisk-mcp    # if installed via pip
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


def _build_mcp():
    """Construct the FastMCP instance with all 4 tools registered.

    Separated from main() so that ASGI factories (uvicorn workers, gunicorn,
    Cloud Run start command) can call this directly to obtain the app.
    """
    try:
        from fastmcp import FastMCP  # type: ignore
    except ImportError as e:
        print(
            "FastMCP not installed. Install with: pip install fastmcp>=3.0",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    from .tools import register_all

    mcp = FastMCP(
        name="TrustedRisk",
        instructions=(
            "TrustedRisk provides 4 safe-discharge decision-support primitives: "
            "compute_readmission_risk, compute_decision_utility, ground_claim, detect_phi. "
            "SHARP-on-MCP context headers required: X-FHIR-Server-URL, "
            "X-FHIR-Access-Token. X-Patient-ID required for 3 of 4 tools."
        ),
    )
    register_all(mcp)
    _inject_promptopinion_fhir_extension(mcp)
    return mcp


def _inject_promptopinion_fhir_extension(mcp) -> None:
    """Advertise the Prompt Opinion FHIR-context extension on `initialize`.

    Per https://docs.promptopinion.ai/fhir-context/mcp-fhir-context the
    server signals support by adding the key
    `ai.promptopinion/fhir-context` under `capabilities.extensions` in
    the initialize result, with a `scopes` array of
    `{name, required?}` objects.

    FastMCP already overrides `LowLevelServer.get_capabilities` to inject
    `io.modelcontextprotocol/ui` the same way. We wrap that override one
    layer further so both extensions coexist after every call.
    """
    from .scopes import scope_objects

    inner = mcp._mcp_server
    original_get_capabilities = inner.get_capabilities

    def get_capabilities_with_fhir(notification_options, experimental_capabilities):
        capabilities = original_get_capabilities(
            notification_options, experimental_capabilities,
        )
        existing: dict = getattr(capabilities, "extensions", None) or {}
        capabilities.extensions = {
            **existing,
            "ai.promptopinion/fhir-context": {
                "scopes": scope_objects(),
            },
        }
        return capabilities

    inner.get_capabilities = get_capabilities_with_fhir


def build_http_app(agent_card_path: str | None = None,
                       marketplace_path: str | None = None):
    """Return a Starlette ASGI app with OAuth + SHARP middleware wrapping the
    FastMCP HTTP layer, plus a /oauth/token endpoint when OAuth is enabled.

    Args:
        agent_card_path: Optional override for the `/.well-known/agent-card.json`
            file path. Defaults to `src/a2a_agent/agent-card.json`. Specialist
            apps (Phase 1 federation) use this to publish their own focused
            skill catalog while sharing the underlying MCP backend.
        marketplace_path: Optional override for the marketplace.json file.

    Middleware order (outermost first, runs left to right on request):

        OAuth bearer middleware  -> 401 on missing/invalid token (when enabled)
                                ↓
        SHARP context middleware -> 403 on missing FHIR headers
                                ↓
        FastMCP /mcp/* dispatch  -> tools see both ContextVars

    The SHARP middleware uses Starlette's BaseHTTPMiddleware contract
    (request, call_next), so it must be attached at the http_app layer --
    `mcp.add_middleware()` would treat it as a FastMCP MiddlewareContext
    handler and crash with "no attribute 'headers'" on every request.
    """
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route, WebSocketRoute

    from .oauth import oauth_bearer_middleware
    from .oauth.middleware import is_oauth_enabled
    from .sharp.headers import (
        FHIRContext,
        _fhir_ctx,
        rpc_requires_fhir_context,
        sharp_context_middleware,
    )

    class OAuthStarletteMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            return await oauth_bearer_middleware(request, call_next)

    # Paths public to SHARP (no FHIR-context required). OAuth has its
    # own whitelist below. The A2A messaging root `POST /` is included
    # here -- chat clients (Prompt Opinion etc.) don't carry FHIR
    # context, so SHARP would 403 them; auth is still enforced via
    # OAuth so the surface is not anonymous.
    _PUBLIC_PATHS = (
        "/oauth/", "/.well-known/", "/healthz", "/readyz", "/smart/",
        "/metrics",
        # Phase 12.9 -- OpenAPI surface
        "/openapi.json", "/docs",
        # Phase 13.12 -- WebSocket chat surface
        "/ws/",
    )
    # Exact paths public to SHARP (cannot use prefix because every
    # path on Earth starts with "/").
    _PUBLIC_EXACT = {"/"}

    def _sharp_skip(path: str) -> bool:
        return path in _PUBLIC_EXACT or any(
            path.startswith(p) for p in _PUBLIC_PATHS
        )

    # Phase 6.3 -- initialise OpenTelemetry exporter when configured
    from a2a_agent.observability import (
        default_rate_limiter, maybe_init_otel, rate_key_for_request,
    )
    maybe_init_otel()

    class RateLimitMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            path = request.url.path or ""
            if _sharp_skip(path):
                return await call_next(request)
            allowed, retry_after = default_rate_limiter().acquire(
                rate_key_for_request(request))
            if not allowed:
                resp = JSONResponse(status_code=429, content={
                    "error": "rate_limited",
                    "retry_after_seconds": round(retry_after, 2),
                })
                resp.headers["Retry-After"] = str(int(retry_after) + 1)
                return resp
            return await call_next(request)

    class SharpStarletteMiddleware:
        """Pure-ASGI SHARP middleware.

        Sits below RateLimit + OAuth (which are BaseHTTPMiddleware-based)
        but above the FastMCP app. Two reasons we cannot reuse
        BaseHTTPMiddleware here:

        1. BaseHTTPMiddleware's `wrapped_receive` raises RuntimeError on
           any non-disconnect message after the response starts. SSE
           transports (FastMCP streamable_http) poll receive() to detect
           client disconnect, and any reading-then-replaying of the body
           inside a BaseHTTPMiddleware breaks that loop.
        2. We need to peek at the JSON-RPC method to exempt framework
           calls (initialize, tools/list, ping) from FHIR enforcement.
           Reading via Request.body() locks the receive stream; doing
           the same at ASGI level lets us buffer messages and replay
           them transparently via a wrapping receive callable.
        """

        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return

            path = scope.get("path") or ""
            if _sharp_skip(path):
                await self.app(scope, receive, send)
                return

            # Header lookup -- case-insensitive over scope's bytes-tuples.
            h: dict[str, str] = {}
            for name, value in scope.get("headers", []):
                h[name.decode("latin-1").lower()] = value.decode("latin-1")

            server_url = h.get("x-fhir-server-url", "").strip()
            access_token = h.get("x-fhir-access-token", "").strip()
            patient_id = h.get("x-patient-id", "").strip() or None
            refresh_token = h.get("x-fhir-refresh-token", "").strip() or None
            refresh_token_url = (
                h.get("x-fhir-refresh-url", "").strip() or None
            )

            missing: list[str] = []
            if not server_url:
                missing.append("X-FHIR-Server-URL")
            if not access_token:
                missing.append("X-FHIR-Access-Token")

            if missing:
                # The body-peek + JSON-RPC framework-method exemption is
                # ONLY relevant for the FastMCP dispatch path under
                # `/mcp/*`. Plain HTTP endpoints (`/api/batch/*`,
                # `/api/stream/*`, `/api/push/*`, the bare `/`...) carry
                # request bodies that aren't JSON-RPC envelopes and must
                # always enforce SHARP when headers are absent. Without
                # this guard the SHARP middleware silently bypasses any
                # non-RPC endpoint that lacks FHIR-context headers.
                is_rpc_dispatch_path = path.startswith("/mcp")
                buffered: list[dict] = []
                if is_rpc_dispatch_path and scope.get("method") == "POST":
                    while True:
                        msg = await receive()
                        buffered.append(msg)
                        if msg.get("type") != "http.request":
                            break
                        if not msg.get("more_body", False):
                            break

                raw = b"".join(
                    m.get("body", b"")
                    for m in buffered
                    if m.get("type") == "http.request"
                )

                if is_rpc_dispatch_path:
                    enforce = rpc_requires_fhir_context(raw) if buffered else True
                else:
                    enforce = True

                if enforce:
                    response = JSONResponse(status_code=403, content={
                        "error": "missing_fhir_context",
                        "message": "SHARP-on-MCP requires FHIR context headers.",
                        "missing_headers": missing,
                        "spec_reference":
                            "https://www.sharponmcp.com/key-components.html",
                    })
                    await response(scope, receive, send)
                    return

                # Bypass: framework method (initialize, tools/list, ...).
                # Replay buffered ASGI messages, then defer to original
                # receive for any post-body events (disconnect, more body).
                msg_iter = iter(buffered)

                async def replay_receive():
                    try:
                        return next(msg_iter)
                    except StopIteration:
                        return await receive()

                await self.app(scope, replay_receive, send)
                return

            # Headers present -- set ContextVar and proceed normally.
            from .fhir.auth_retry import FhirAuthRequired
            ctx_token = _fhir_ctx.set(FHIRContext(
                server_url=server_url,
                access_token=access_token,
                patient_id=patient_id,
                refresh_token=refresh_token,
                refresh_token_url=refresh_token_url,
            ))
            try:
                await self.app(scope, receive, send)
            except FhirAuthRequired as exc:
                # Phase 3.2 -- auto-refresh failure -> A2A AUTH_REQUIRED 401.
                # Best-effort: if the response has already started, the
                # send call will raise and propagate.
                response = JSONResponse(
                    status_code=401, content=exc.hint.model_dump(),
                )
                await response(scope, receive, send)
            finally:
                _fhir_ctx.reset(ctx_token)

    mcp = _build_mcp()

    middleware = [
        Middleware(RateLimitMiddleware),
        Middleware(OAuthStarletteMiddleware),
        Middleware(SharpStarletteMiddleware),
    ]

    # Inner FastMCP app -- middleware lives on the outer Starlette so the
    # batch endpoint also gets OAuth + SHARP enforcement.
    mcp_app = mcp.http_app(transport="streamable-http")

    # Phase 3.4 -- A2A push notification config endpoints
    async def push_create_endpoint(request):
        from a2a_agent.push_notifications import default_registry
        try:
            payload = await request.json()
        except Exception as exc:
            return JSONResponse(status_code=400, content={
                "error": "invalid_json", "error_description": str(exc)})
        task_id = payload.get("taskId") or payload.get("task_id")
        webhook = payload.get("webhookUrl") or payload.get("webhook_url")
        if not task_id or not webhook:
            return JSONResponse(status_code=400, content={
                "error": "missing_required_field",
                "error_description": "taskId + webhookUrl are required",
            })
        cfg = default_registry().create(
            task_id=task_id, webhook_url=webhook,
            auth_header=payload.get("authHeader"),
            severity_threshold=payload.get("severityThreshold"),
        )
        return JSONResponse(content={
            "config_id": cfg.config_id, "task_id": cfg.task_id,
            "webhook_url": cfg.webhook_url,
        })

    async def push_delete_endpoint(request):
        from a2a_agent.push_notifications import default_registry
        config_id = request.path_params.get("config_id", "")
        ok = default_registry().delete(config_id)
        if not ok:
            return JSONResponse(status_code=404, content={
                "error": "config_not_found"})
        return JSONResponse(content={"deleted": config_id})

    async def push_list_endpoint(request):
        from a2a_agent.push_notifications import default_registry
        configs = default_registry().list_all()
        return JSONResponse(content={
            "configs": [
                {"config_id": c.config_id, "task_id": c.task_id,
                 "webhook_url": c.webhook_url}
                for c in configs
            ],
        })

    # Phase 3.3 -- SSE streaming endpoint for the long-running outcomes
    # simulator. Demonstrates the A2A streaming surface that our
    # `capabilities.streaming = true` declaration in the agent-card
    # promises.
    from starlette.responses import StreamingResponse

    async def stream_llm_polish_endpoint(request):
        from a2a_agent.streaming import stream_llm_polish
        try:
            payload = await request.json()
        except Exception as exc:
            return JSONResponse(status_code=400, content={
                "error": "invalid_json", "error_description": str(exc)})
        text = payload.get("text", "")
        if not text:
            return JSONResponse(status_code=400, content={
                "error": "missing_text",
                "error_description": "`text` field is required",
            })
        return StreamingResponse(
            stream_llm_polish(
                text=text,
                system_prompt=payload.get("system_prompt"),
                task_id=payload.get("task_id"),
                context_id=payload.get("context_id"),
            ),
            media_type="text/event-stream",
        )

    async def stream_outcomes_endpoint(request):
        from a2a_agent.streaming import stream_outcomes_simulation
        try:
            payload = await request.json()
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_json",
                         "error_description": str(exc)},
            )
        case_mix = payload.get("case_mix", [])
        if not case_mix:
            return JSONResponse(
                status_code=400,
                content={"error": "missing_case_mix",
                         "error_description": "case_mix array required"},
            )
        return StreamingResponse(
            stream_outcomes_simulation(
                case_mix=case_mix,
                intervention_rrr=float(payload.get("intervention_rrr", 0.25)),
                intervention_cost_per_patient_usd=float(
                    payload.get("intervention_cost_per_patient_usd", 75.0)),
                avoided_event_cost_usd=float(
                    payload.get("avoided_event_cost_usd", 14000.0)),
                qaly_gained_per_avoided_event=payload.get(
                    "qaly_gained_per_avoided_event"),
                n_iterations=int(payload.get("n_iterations", 1000)),
                chunk_size=int(payload.get("chunk_size", 100)),
                task_id=payload.get("task_id"),
                context_id=payload.get("context_id"),
            ),
            media_type="text/event-stream",
        )

    # Batch endpoint -- deterministic per-patient pipeline that bypasses the
    # LlmAgent for throughput. Reuses the same OAuth + SHARP middleware stack.
    async def batch_endpoint(request):
        from a2a_agent.batch import (
            parse_batch_request,
            process_batch,
            serialize_batch_response,
        )
        try:
            payload = await request.json()
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_json",
                         "error_description": f"Body is not valid JSON: {exc}"},
            )
        try:
            batch_req = parse_batch_request(payload)
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_request",
                         "error_description": str(exc)},
            )
        try:
            batch_resp = await process_batch(batch_req)
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"error": "batch_failed",
                         "error_description": f"{type(exc).__name__}: {exc}"},
            )
        return JSONResponse(content=serialize_batch_response(batch_resp))

    # Add /oauth/token endpoint when OAuth is enabled.
    async def token_endpoint(request):
        from urllib.parse import parse_qs

        from .oauth import issue_token

        # Read the body exactly once -- under BaseHTTPMiddleware,
        # `request.form()` consumes the stream and a subsequent
        # `request.body()` raises "Stream consumed".
        content_type = (
            request.headers.get("content-type", "").split(";")[0].strip().lower()
        )
        raw = await request.body()
        data: dict[str, str] = {}
        if raw:
            if content_type == "application/x-www-form-urlencoded":
                parsed_qs = parse_qs(
                    raw.decode("utf-8", errors="ignore"), keep_blank_values=True,
                )
                data = {k: v[0] for k, v in parsed_qs.items() if v}
            elif content_type == "application/json":
                import json as _json
                try:
                    parsed = _json.loads(raw.decode("utf-8") or "{}")
                    if isinstance(parsed, dict):
                        data = {k: str(v) for k, v in parsed.items()}
                except (ValueError, UnicodeDecodeError):
                    pass
        client_id = (data.get("client_id") or "").strip()
        client_secret = (data.get("client_secret") or "").strip()

        # RFC 6749 Sec. 2.3.1: clients MAY authenticate via HTTP Basic in
        # `Authorization: Basic base64(client_id:client_secret)`. Many OAuth
        # clients (incl. Prompt Opinion) prefer this over body params.
        if not (client_id and client_secret):
            auth_header = request.headers.get("authorization", "")
            if auth_header.lower().startswith("basic "):
                import base64
                from urllib.parse import unquote
                try:
                    decoded = base64.b64decode(
                        auth_header.split(" ", 1)[1].strip(), validate=True,
                    ).decode("utf-8")
                except (ValueError, UnicodeDecodeError):
                    decoded = ""
                if ":" in decoded:
                    raw_id, raw_secret = decoded.split(":", 1)
                    # RFC 6749 requires form-urlencoded values inside Basic
                    if not client_id:
                        client_id = unquote(raw_id).strip()
                    if not client_secret:
                        client_secret = unquote(raw_secret).strip()

        scope_str = (data.get("scope") or "").strip()
        scopes = scope_str.split() if scope_str else None
        grant_type = (data.get("grant_type") or "client_credentials").strip()
        if grant_type != "client_credentials":
            return JSONResponse(
                status_code=400,
                content={"error": "unsupported_grant_type",
                         "error_description": "Only client_credentials is supported."},
            )
        try:
            resp = issue_token(client_id, client_secret, scopes)
        except RuntimeError as exc:
            return JSONResponse(
                status_code=503,
                content={"error": "server_misconfigured",
                         "error_description": str(exc)},
            )
        except ValueError as exc:
            err = str(exc)
            return JSONResponse(
                status_code=400,
                content={"error": err,
                         "error_description": "Token request rejected."},
            )
        return JSONResponse(content=resp)

    # MARKET-1: public agent-card endpoint (no auth, no SHARP).
    import json as _json
    from pathlib import Path as _Path
    _default_agent_card = (
        _Path(__file__).parent.parent / "a2a_agent" / "agent-card.json"
    )
    _default_marketplace = (
        _Path(__file__).parent.parent / "a2a_agent" / "marketplace.json"
    )
    _agent_card_path = (
        _Path(agent_card_path) if agent_card_path else _default_agent_card
    )
    _marketplace_path = (
        _Path(marketplace_path) if marketplace_path else _default_marketplace
    )

    from a2a_agent.agent_card_normalizer import normalize_agent_card

    async def well_known_agent_card(_request):
        if not _agent_card_path.exists():
            return JSONResponse(status_code=404,
                                  content={"error": "agent_card_missing"})
        raw = _json.loads(_agent_card_path.read_text(encoding="utf-8"))
        return JSONResponse(content=normalize_agent_card(raw))

    async def well_known_marketplace(_request):
        if not _marketplace_path.exists():
            return JSONResponse(status_code=404,
                                  content={"error": "marketplace_manifest_missing"})
        return JSONResponse(content=_json.loads(
            _marketplace_path.read_text(encoding="utf-8")))

    async def well_known_sharp_capabilities(_request):
        """MARKET-3: SHARP-on-MCP capability declaration."""
        from .sharp.headers import initialize_capabilities
        return JSONResponse(content=initialize_capabilities())

    async def healthz(_request):
        from .tools import BUNDLES
        unique_tools: set[str] = set()
        for tools in BUNDLES.values():
            unique_tools.update(tools)
        return JSONResponse(content={
            "status": "ok",
            "version": "0.7.0",
            "tools_registered": len(unique_tools),
            "bundles_registered": len(BUNDLES),
        })

    async def prometheus_metrics_endpoint(_request):
        """Phase 6.3 -- Prometheus metrics scrape endpoint."""
        from a2a_agent.observability import render_prometheus_metrics
        from starlette.responses import PlainTextResponse
        return PlainTextResponse(
            content=render_prometheus_metrics(),
            media_type="text/plain; version=0.0.4",
        )

    async def readyz(_request):
        # Lightweight readiness -- confirms coefficients.json is loadable.
        coef_path = os.environ.get("TRUSTEDRISK_COEFFICIENTS_PATH",
                                      "data/coefficients.json")
        ready = _Path(coef_path).exists()
        return JSONResponse(
            status_code=(200 if ready else 503),
            content={"ready": ready, "coefficients_path": coef_path},
        )

    # A2A v1 JSON-RPC messaging endpoint. Clients (Prompt Opinion chat,
    # Microsoft Copilot Studio, ...) POST to the URL declared in the
    # agent card's supportedInterfaces[].url. We pick the right handler
    # based on whether this build is the root MCP or a specialist:
    # specialists get build_specialist_handler bound to their card,
    # the root gets build_root_handler.
    async def a2a_messaging_endpoint(request):
        from a2a_agent.a2a_messaging import (
            build_root_handler, build_specialist_handler, handle_jsonrpc,
        )
        try:
            body = await request.json()
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_json", "detail": str(exc)},
            )
        # agent_card_path is set by the specialist factory; if absent
        # we are the root MCP backend.
        if agent_card_path is None:
            handler = build_root_handler(_agent_card_path)
        else:
            handler = build_specialist_handler(_agent_card_path)
        result = await handle_jsonrpc(body, handler)
        return JSONResponse(content=result)

    routes: list = [
        Route("/", a2a_messaging_endpoint, methods=["POST"]),
        Route("/api/batch/decision-cards", batch_endpoint, methods=["POST"]),
        Route("/api/stream/outcomes-simulation", stream_outcomes_endpoint,
                methods=["POST"]),
        Route("/api/stream/llm-polish", stream_llm_polish_endpoint,
                methods=["POST"]),
        Route("/api/push/configs", push_create_endpoint, methods=["POST"]),
        Route("/api/push/configs", push_list_endpoint, methods=["GET"]),
        Route("/api/push/configs/{config_id}", push_delete_endpoint,
                methods=["DELETE"]),
        Route("/.well-known/agent-card.json", well_known_agent_card,
                methods=["GET"]),
        Route("/.well-known/marketplace.json", well_known_marketplace,
                methods=["GET"]),
        Route("/.well-known/sharp-capabilities.json",
                well_known_sharp_capabilities, methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
        Route("/readyz", readyz, methods=["GET"]),
        Route("/metrics", prometheus_metrics_endpoint, methods=["GET"]),
    ]

    # Phase 12.9 -- OpenAPI/Swagger surface per specialist.
    # Mounted only when an agent_card_path is supplied (the master MCP
    # server uses a card too, so it gets the docs surface as well).
    if agent_card_path:
        try:
            from a2a_agent.openapi_surface import (
                make_docs_route, make_openapi_route,
            )
            routes.append(Route(
                "/openapi.json",
                make_openapi_route(agent_card_path),
                methods=["GET"],
            ))
            routes.append(Route(
                "/docs", make_docs_route(agent_card_path),
                methods=["GET"],
            ))
        except Exception:
            # OpenAPI is best-effort -- never block boot on a doc bug.
            pass

    # Phase 13.12 -- WebSocket A2A chat surface
    try:
        from a2a_agent.ws_chat import ws_chat_endpoint
        routes.append(WebSocketRoute("/ws/chat", ws_chat_endpoint))
    except Exception:
        # Best-effort -- never block boot on the chat surface.
        pass

    # INT-2: SMART on FHIR launch handshake
    from .smart.launch import smart_routes
    routes.extend(smart_routes())
    if is_oauth_enabled():
        routes.append(Route("/oauth/token", token_endpoint, methods=["POST"]))
    routes.append(Mount("/", app=mcp_app))

    outer = Starlette(
        routes=routes,
        middleware=middleware,
        lifespan=mcp_app.lifespan if hasattr(mcp_app, "lifespan") else None,
    )
    return outer


def main() -> None:
    # Load .env first so tool modules see env vars at import time
    load_dotenv()

    import uvicorn

    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_PORT", "8080"))

    app = build_http_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
