"""A2A v1 FHIR-context extension reader.

Per https://docs.promptopinion.ai/fhir-context/a2a-fhir-context.html,
the FHIR context on A2A messages travels in the message metadata
keyed by the canonical extension URI:

    https://app.promptopinion.ai/schemas/a2a/v1/fhir-context

with the camelCase payload shape:

    {
      "fhirUrl": "...",
      "fhirToken": "...",
      "patientId": "...",
      "fhirRefreshToken": "..." (optional),
      "fhirRefreshTokenUrl": "..." (optional)
    }

This is structurally distinct from the SHARP-on-MCP path used by the
MCP server (HTTP headers, snake_case in our internal `FHIRContext`):
when the agent runtime receives an A2A message it must transform the
PO metadata payload into the same shared `FHIRContext` (from
`mcp_server.sharp.headers`) so downstream tools see a single, uniform
context contract regardless of which transport delivered it.

Usage (from an ADK `before_model_callback` or any agent host):

    from a2a_agent.po_fhir_context import (
        PO_FHIR_CONTEXT_URI,
        extract_po_fhir_context,
        bind_po_fhir_context,
    )

    metadata = a2a_message.get("metadata", {})
    ctx = extract_po_fhir_context(metadata)
    if ctx is not None:
        token = bind_po_fhir_context(ctx)
        try:
            ... run the agent / tool chain ...
        finally:
            release_po_fhir_context(token)

Returns `None` when the metadata is absent or empty (no FHIR work
needed) -- the agent should still proceed for tools that don't depend
on FHIR (e.g. `detect_phi`, the patient_facing translators).
"""

from __future__ import annotations

from contextvars import Token
from dataclasses import dataclass

from mcp_server.sharp.headers import FHIRContext, _fhir_ctx


PO_FHIR_CONTEXT_URI = "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context"


@dataclass(frozen=True)
class POFhirContextPayload:
    """Strongly-typed view over the camelCase metadata payload.

    Keep this distinct from `FHIRContext` (snake_case, MCP-side) -- when
    we cross from A2A to MCP we DO map field names; we don't conflate
    the two types so a future spec drift is easy to localise.
    """
    fhir_url: str
    fhir_token: str | None = None
    patient_id: str | None = None
    fhir_refresh_token: str | None = None
    fhir_refresh_token_url: str | None = None


def extract_po_fhir_context(
    message_metadata: dict | None,
) -> POFhirContextPayload | None:
    """Extract the Prompt Opinion FHIR context from A2A message metadata.

    Returns `None` if the metadata block does not declare the PO
    extension (the message simply has no FHIR context to propagate).

    Raises `ValueError` if the extension is declared but the payload
    is malformed (missing `fhirUrl`, wrong type, etc.) -- that's a
    client contract violation we should surface, not silently ignore.
    """
    if not message_metadata or not isinstance(message_metadata, dict):
        return None

    payload = message_metadata.get(PO_FHIR_CONTEXT_URI)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(
            f"PO FHIR-context metadata at {PO_FHIR_CONTEXT_URI!r} must be an "
            f"object; got {type(payload).__name__}."
        )

    fhir_url = payload.get("fhirUrl")
    if not isinstance(fhir_url, str) or not fhir_url:
        raise ValueError(
            "PO FHIR-context payload missing required `fhirUrl` field."
        )

    def _opt_str(key: str) -> str | None:
        v = payload.get(key)
        if v is None:
            return None
        if not isinstance(v, str) or not v:
            raise ValueError(
                f"PO FHIR-context payload field {key!r} must be a non-empty "
                f"string when present; got {v!r}."
            )
        return v

    return POFhirContextPayload(
        fhir_url=fhir_url,
        fhir_token=_opt_str("fhirToken"),
        patient_id=_opt_str("patientId"),
        fhir_refresh_token=_opt_str("fhirRefreshToken"),
        fhir_refresh_token_url=_opt_str("fhirRefreshTokenUrl"),
    )


def to_fhir_context(payload: POFhirContextPayload) -> FHIRContext:
    """Map the A2A camelCase payload onto the shared MCP `FHIRContext`.

    `access_token` is set to empty string when the payload omits it --
    SHARP supports the no-token case (`fhirToken` is optional per the
    PO spec) and downstream tools must handle it. We deliberately
    don't substitute a sentinel -- the empty-string case is the
    semantic carrier for "no auth required against this FHIR server".
    """
    return FHIRContext(
        server_url=payload.fhir_url,
        access_token=payload.fhir_token or "",
        patient_id=payload.patient_id,
        refresh_token=payload.fhir_refresh_token,
        refresh_token_url=payload.fhir_refresh_token_url,
    )


def bind_po_fhir_context(payload: POFhirContextPayload) -> Token:
    """Set the shared `FHIRContext` ContextVar from a PO payload.

    Returns the reset-token for `release_po_fhir_context`. The caller
    MUST release in a finally-block or the ContextVar leaks across
    requests.
    """
    return _fhir_ctx.set(to_fhir_context(payload))


def release_po_fhir_context(token: Token) -> None:
    """Release a previously-bound PO FHIR context."""
    _fhir_ctx.reset(token)
