"""Auto-refresh + AUTH_REQUIRED transport for upstream FHIR calls.

When a tool's FHIR upstream returns 401 (token expired) the wrapper:

  1. Looks up the current FHIRContext (snake_case, populated by the
     SHARP middleware or the A2A FHIR-context reader).
  2. If `refresh_token` + `refresh_token_url` are both populated,
     calls `refresh_fhir_token_async()` per the Prompt Opinion contract.
  3. On success, mutates the in-flight FHIRContext to carry the new
     `access_token` and retries the wrapped call once.
  4. On failure (no refresh token, refresh endpoint error, or refreshed
     token also rejected), raises `FhirAuthRequired` -- the MCP server's
     middleware catches this and returns a structured 401 with the
     `AuthRequiredHint` body so the BYO orchestrator can transition
     the A2A task to AUTH_REQUIRED state.

Pure async; no global state.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Awaitable, Callable, TypeVar

from shared.schemas import AuthRequiredHint

from ..sharp.headers import FHIRContext, _fhir_ctx
from ..sharp.refresh import (
    RefreshTokenExchangeError,
    refresh_fhir_token_async,
)


T = TypeVar("T")


class FhirAuthRequired(RuntimeError):
    """Raised when the upstream FHIR server returns 401 and the
    auto-refresh path cannot recover. Carries an `AuthRequiredHint`
    payload that the middleware serialises into the 401 response."""
    def __init__(self, hint: AuthRequiredHint):
        super().__init__(hint.message)
        self.hint = hint


def _is_401(exc: Exception) -> bool:
    """Detect httpx-style 401 exceptions without importing httpx at
    module load time (keeps the wrapper usable when fhirpy is on a
    different HTTP client).

    Strict structural check -- never matches on a substring of the
    exception message alone (would yield false positives on
    `ValueError("error 4017")` etc.).
    """
    status = getattr(exc, "response", None)
    if status is not None:
        code = getattr(status, "status_code", None)
        if code == 401:
            return True
    if getattr(exc, "status_code", None) == 401:
        return True
    return False


async def _refresh_and_update_context(ctx: FHIRContext) -> FHIRContext:
    """Run the refresh exchange and propagate the new tokens into the
    current ContextVar. Returns the updated FHIRContext."""
    if not (ctx.refresh_token and ctx.refresh_token_url):
        raise FhirAuthRequired(AuthRequiredHint(
            message=(
                "Upstream FHIR server returned 401 and no refresh "
                "token was supplied -- the SHARP client must include "
                "X-FHIR-Refresh-Token + X-FHIR-Refresh-Url to enable "
                "auto-refresh."
            ),
            refresh_attempted=False,
            fhir_server_url=ctx.server_url or None,
        ))
    try:
        refreshed = await refresh_fhir_token_async(
            ctx.refresh_token_url, ctx.refresh_token,
        )
    except RefreshTokenExchangeError as exc:
        raise FhirAuthRequired(AuthRequiredHint(
            message=(
                "Upstream FHIR server returned 401 and the refresh "
                "exchange failed."
            ),
            refresh_attempted=True,
            refresh_failure_reason=str(exc),
            fhir_server_url=ctx.server_url or None,
        )) from exc

    # Mutate the FHIRContext in place so callers holding a reference to
    # the same object see the rotated tokens. ContextVar propagation
    # across `asyncio.run()` boundaries is intentionally one-way (the
    # outer task copy doesn't see inner-task `.set()` calls), so we use
    # the dataclass's mutability to propagate the new tokens to whoever
    # already has the reference.
    ctx.access_token = refreshed.access_token
    ctx.refresh_token = refreshed.refresh_token
    # Also re-bind the ContextVar to the same object so any downstream
    # reads inside this task see the rotated tokens via either path.
    _fhir_ctx.set(ctx)
    return ctx


async def with_auth_retry(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Call `func(*args, **kwargs)`. On a 401 from the FHIR upstream,
    attempt one refresh + retry. On any other failure or if refresh
    cannot recover, raise FhirAuthRequired with a structured hint.
    """
    try:
        return await func(*args, **kwargs)
    except Exception as exc:
        if not _is_401(exc):
            raise
        try:
            ctx = _fhir_ctx.get()
        except LookupError as inner:
            raise FhirAuthRequired(AuthRequiredHint(
                message=(
                    "Upstream FHIR server returned 401 but no SHARP "
                    "FHIR context is bound to this request -- cannot "
                    "auto-refresh."
                ),
                refresh_attempted=False,
            )) from inner
        await _refresh_and_update_context(ctx)
        # One retry; any failure here propagates raw (we don't try a
        # second refresh -- at that point the credentials are clearly
        # invalid).
        try:
            return await func(*args, **kwargs)
        except Exception as exc2:
            if _is_401(exc2):
                raise FhirAuthRequired(AuthRequiredHint(
                    message=(
                        "Refreshed access token was also rejected by "
                        "the FHIR server -- the BYO orchestrator must "
                        "re-fetch credentials before retrying."
                    ),
                    refresh_attempted=True,
                    refresh_failure_reason="refreshed_token_also_401",
                    fhir_server_url=ctx.server_url or None,
                )) from exc2
            raise
