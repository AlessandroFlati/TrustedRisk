"""ASGI middleware that enforces OAuth bearer-token authentication.

Runs BEFORE the SHARP context middleware, so:

    OAuth bearer middleware  ──->  401 if missing / invalid token
                                  binds ClientContext to ContextVar
                                  ↓
    SHARP context middleware ──->  403 if missing FHIR headers
                                  cross-checks: client's allowlist matches
                                  the requested X-FHIR-Server-URL
                                  binds FHIRContext to ContextVar
                                  ↓
    Tool dispatch            ──->  reads both contexts via get_*_context()

The middleware is OPT-IN: when `TRUSTEDRISK_OAUTH_ENABLED` is unset or "0",
the middleware passes every request through without enforcement
(backward-compatible single-tenant mode).
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from dataclasses import dataclass

from starlette.requests import Request
from starlette.responses import JSONResponse

from .issuer import load_clients_from_env
from .validator import OAuthValidationError, TokenValidator


@dataclass
class ClientContext:
    """Authenticated client identity, bound per-request via ContextVar."""
    client_id: str
    tenant: str
    scopes: list[str]


_client_ctx: ContextVar[ClientContext] = ContextVar("oauth_client_ctx")


def get_client_context() -> ClientContext:
    """Retrieve the current request's authenticated client identity.

    Raises LookupError outside an OAuth-protected request OR when OAuth is
    disabled (single-tenant mode).
    """
    return _client_ctx.get()


def is_oauth_enabled() -> bool:
    return os.environ.get("TRUSTEDRISK_OAUTH_ENABLED", "0") == "1"


async def oauth_bearer_middleware(request: Request, call_next):
    """Enforce bearer-token auth on /mcp/* requests when OAuth is enabled.

    Allows /oauth/* paths through unauthenticated (the token endpoint itself
    must be reachable without a token).
    """
    if not is_oauth_enabled():
        return await call_next(request)

    # Whitelist OAuth endpoints + common health/static + marketplace metadata
    path = request.url.path or ""
    public_prefixes = ("/oauth/", "/.well-known/", "/smart/")
    # NOTE: `/` is intentionally NOT public -- it's the A2A v1 JSON-RPC
    # messaging endpoint, which clients are required to authenticate
    # against per the agent card's securitySchemes. Probes use /healthz
    # and /readyz instead.
    public_exact = ("/health", "/healthz", "/readyz", "/openapi.json")
    if any(path.startswith(p) for p in public_prefixes) or path in public_exact:
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "").strip()
    if not auth_header.lower().startswith("bearer "):
        return _unauthorized("missing_bearer", "Authorization: Bearer <token> required.")

    token = auth_header.split(" ", 1)[1].strip()
    secret = os.environ.get("TRUSTEDRISK_OAUTH_SECRET")
    if not secret:
        return _unauthorized("oauth_misconfigured",
                              "TRUSTEDRISK_OAUTH_SECRET is not set.")

    validator = TokenValidator(secret, clients=load_clients_from_env())
    try:
        validated = validator.validate(token)
    except OAuthValidationError as e:
        return _unauthorized(str(e), "Bearer token rejected.")

    # Cross-check: client's allowed_fhir_servers must include the requested
    # X-FHIR-Server-URL. This is the multi-tenant isolation guard.
    fhir_url = request.headers.get("X-FHIR-Server-URL", "").strip()
    if fhir_url and not validator.check_fhir_server_allowed(validated.client_id, fhir_url):
        return _unauthorized(
            "fhir_server_not_allowed",
            f"Client {validated.client_id!r} is not authorized for FHIR server "
            f"{fhir_url!r}. Check `allowed_fhir_servers` in the client registry.",
        )

    token_var = _client_ctx.set(ClientContext(
        client_id=validated.client_id,
        tenant=validated.tenant,
        scopes=validated.scopes,
    ))
    try:
        return await call_next(request)
    finally:
        _client_ctx.reset(token_var)


def _unauthorized(code: str, message: str) -> JSONResponse:
    body = {
        "error": "unauthorized",
        "error_code": code,
        "error_description": message,
        "spec_reference": "https://datatracker.ietf.org/doc/html/rfc6749#section-4.4",
    }
    headers = {"WWW-Authenticate": 'Bearer realm="trustedrisk"'}
    return JSONResponse(status_code=401, content=body, headers=headers)
