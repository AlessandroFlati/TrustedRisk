"""SHARP-on-MCP / Prompt Opinion FHIR refresh-token exchange.

Per the contract published at
https://docs.promptopinion.ai/fhir-context/mcp-fhir-context.html (and
mirrored at the A2A path), when the server has consumed the offline-
access headers (X-FHIR-Refresh-Token + X-FHIR-Refresh-Url) it can
exchange a stale access token for a fresh one by posting JSON to the
refresh URL:

    POST <X-FHIR-Refresh-Url>
    {"refreshToken": "<refresh-token-value>"}

    ->

    {"accessToken": "<new>", "refreshToken": "<rotated-or-same>"}

This module provides a synchronous and an asynchronous helper. Tools
that detect a 401 from the upstream FHIR server can call
`refresh_fhir_token()` then retry.

Behaviour notes:
- The current ContextVar (FHIRContext) is NOT mutated by this helper --
  the caller decides whether to overwrite the access token in
  the ContextVar or attach the rotated token only to a single retry.
- Both helpers raise `RefreshTokenExchangeError` on any non-2xx
  response, malformed body, or missing required field. Tools should
  log the error and surface a structured abstain rather than retrying
  indefinitely.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class RefreshedTokens:
    access_token: str
    refresh_token: str


class RefreshTokenExchangeError(RuntimeError):
    """Raised when the refresh exchange fails or returns a malformed response."""


def _parse_payload(payload: dict) -> RefreshedTokens:
    if not isinstance(payload, dict):
        raise RefreshTokenExchangeError(
            f"Refresh response is not an object: {type(payload).__name__}"
        )
    access = payload.get("accessToken")
    refresh = payload.get("refreshToken")
    if not isinstance(access, str) or not access:
        raise RefreshTokenExchangeError(
            "Refresh response missing/empty `accessToken` field."
        )
    if not isinstance(refresh, str) or not refresh:
        raise RefreshTokenExchangeError(
            "Refresh response missing/empty `refreshToken` field."
        )
    return RefreshedTokens(access_token=access, refresh_token=refresh)


def refresh_fhir_token(
    refresh_url: str,
    refresh_token: str,
    *,
    timeout_s: float = 10.0,
) -> RefreshedTokens:
    """Exchange a refresh token synchronously. See module docstring."""
    if not refresh_url or not refresh_token:
        raise RefreshTokenExchangeError(
            "Both refresh_url and refresh_token are required."
        )
    try:
        response = httpx.post(
            refresh_url,
            json={"refreshToken": refresh_token},
            timeout=timeout_s,
        )
    except httpx.HTTPError as exc:
        raise RefreshTokenExchangeError(
            f"Refresh request failed: {type(exc).__name__}: {exc}"
        ) from exc
    if response.status_code // 100 != 2:
        raise RefreshTokenExchangeError(
            f"Refresh endpoint returned {response.status_code}: "
            f"{response.text[:200]!r}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RefreshTokenExchangeError(
            f"Refresh response not JSON: {exc}"
        ) from exc
    return _parse_payload(payload)


async def refresh_fhir_token_async(
    refresh_url: str,
    refresh_token: str,
    *,
    timeout_s: float = 10.0,
) -> RefreshedTokens:
    """Async variant of `refresh_fhir_token`."""
    if not refresh_url or not refresh_token:
        raise RefreshTokenExchangeError(
            "Both refresh_url and refresh_token are required."
        )
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                refresh_url,
                json={"refreshToken": refresh_token},
            )
    except httpx.HTTPError as exc:
        raise RefreshTokenExchangeError(
            f"Refresh request failed: {type(exc).__name__}: {exc}"
        ) from exc
    if response.status_code // 100 != 2:
        raise RefreshTokenExchangeError(
            f"Refresh endpoint returned {response.status_code}: "
            f"{response.text[:200]!r}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RefreshTokenExchangeError(
            f"Refresh response not JSON: {exc}"
        ) from exc
    return _parse_payload(payload)
