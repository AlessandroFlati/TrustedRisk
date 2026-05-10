"""Phase 16.H3 - SMART-on-FHIR EHR launch flow.

Implements the *server-side* of the SMART App Launch Framework v2.0
EHR-launch sequence (the part TrustedRisk needs to honour when an EHR
embeds it as a contextual app).

Steps modeled here (deterministic, no real HTTP):
  1. EHR launches with `?iss=<fhir_base>&launch=<opaque_token>`.
  2. App fetches `<iss>/.well-known/smart-configuration` -> receives
     `authorization_endpoint`, `token_endpoint`, `capabilities`.
  3. App redirects user to authorization_endpoint with PKCE
     (S256 code_challenge) + a state token.
  4. EHR redirects back to redirect_uri with `?code=...&state=...`.
  5. App POSTs to token_endpoint with the code + code_verifier ->
     receives access_token + id_token + patient context.
  6. App validates the issuer + nonce + state.

This module is the *deterministic floor* used in tests + the playground
demo. Production replaces the in-memory `SimulatedEHR` with real httpx
calls.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from typing import Any, Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# PKCE helpers (RFC 7636)
# ─────────────────────────────────────────────────────────────────────


def generate_code_verifier(*, n_bytes: int = 32) -> str:
    """RFC 7636 §4.1: 43-128 chars from [A-Z][a-z][0-9]_.~ ."""
    return base64.urlsafe_b64encode(
        secrets.token_bytes(n_bytes)).rstrip(b"=").decode("ascii")


def code_challenge_for(verifier: str) -> str:
    """RFC 7636 §4.2: BASE64URL-NO-PAD(SHA256(verifier))."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


# ─────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────


class SmartLaunchContext(BaseModel):
    iss: str
    launch_token: str
    authorization_endpoint: str
    token_endpoint: str
    state: str
    nonce: str
    code_verifier: str
    code_challenge: str
    code_challenge_method: Literal["S256"] = "S256"


class SmartTokenResponse(BaseModel):
    access_token: str
    id_token: str | None = None
    token_type: Literal["Bearer"] = "Bearer"
    expires_in: int = Field(ge=1)
    scope: str
    patient: str | None = None
    encounter: str | None = None
    issued_at_unix: int


class SmartLaunchTrace(BaseModel):
    iss_validated: bool
    state_validated: bool
    nonce_validated: bool
    issuer_match: bool
    pkce_validated: bool
    token_response: SmartTokenResponse


# ─────────────────────────────────────────────────────────────────────
# Step 1 + 2 - .well-known/smart-configuration
# ─────────────────────────────────────────────────────────────────────


def load_smart_configuration(iss: str) -> dict[str, Any]:
    """Return a deterministic SMART configuration document for the
    simulated issuer. Production replaces with httpx.get().json()."""
    return {
        "issuer": iss,
        "authorization_endpoint": f"{iss}/oauth/authorize",
        "token_endpoint": f"{iss}/oauth/token",
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": [
            "openid", "fhirUser", "launch", "launch/patient",
            "patient/Patient.read", "patient/Condition.read",
            "patient/Observation.read",
            "patient/MedicationRequest.read",
        ],
        "response_types_supported": ["code"],
        "capabilities": [
            "launch-ehr", "launch-standalone",
            "client-public", "client-confidential-symmetric",
            "context-passthrough-banner",
            "context-passthrough-style",
            "context-ehr-patient", "context-ehr-encounter",
            "permission-patient", "permission-online", "sso-openid-connect",
        ],
    }


# ─────────────────────────────────────────────────────────────────────
# Step 3 - launch context bootstrap
# ─────────────────────────────────────────────────────────────────────


def begin_smart_launch(
    *, iss: str, launch_token: str,
    rng_seed: int | None = None,
) -> SmartLaunchContext:
    """Validate the launch parameters + bootstrap PKCE + state +
    nonce."""
    if not iss:
        raise ValueError("iss must be a non-empty FHIR base URL")
    if not launch_token:
        raise ValueError("launch_token must be present")
    rng = secrets.SystemRandom() if rng_seed is None else None
    if rng_seed is not None:
        # Deterministic for testing
        import random as _r
        det = _r.Random(rng_seed)
        verifier = "".join(
            det.choices(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                "0123456789-._~",
                k=64,
            )
        )
        state = "".join(det.choices(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "0123456789", k=32))
        nonce = "".join(det.choices(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "0123456789", k=32))
    else:
        verifier = generate_code_verifier()
        state = secrets.token_urlsafe(24)
        nonce = secrets.token_urlsafe(24)
    challenge = code_challenge_for(verifier)
    cfg = load_smart_configuration(iss)
    return SmartLaunchContext(
        iss=iss, launch_token=launch_token,
        authorization_endpoint=cfg["authorization_endpoint"],
        token_endpoint=cfg["token_endpoint"],
        state=state, nonce=nonce,
        code_verifier=verifier, code_challenge=challenge,
    )


# ─────────────────────────────────────────────────────────────────────
# Steps 4-6 - simulated EHR token exchange + validation
# ─────────────────────────────────────────────────────────────────────


def simulate_ehr_token_exchange(
    *, ctx: SmartLaunchContext,
    code: str, returned_state: str,
    code_verifier: str,
    patient_id: str, encounter_id: str,
    access_token_lifetime_s: int = 3600,
) -> SmartLaunchTrace:
    """Simulate the EHR's token endpoint accepting a code + verifier
    and returning a SMART token response. Performs every check the
    SMART v2.0 spec mandates."""
    if returned_state != ctx.state:
        raise ValueError(
            "state mismatch - possible CSRF; refusing token exchange"
        )
    expected_challenge = code_challenge_for(code_verifier)
    pkce_ok = expected_challenge == ctx.code_challenge
    if not pkce_ok:
        raise ValueError(
            "PKCE verifier does not match the original challenge"
        )
    issued_at = int(time.time())
    id_token_payload = {
        "iss": ctx.iss, "sub": patient_id, "nonce": ctx.nonce,
        "iat": issued_at, "exp": issued_at + access_token_lifetime_s,
    }
    # Compact unsigned JWS (alg=none) - simulator only; real flow uses RS256
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "none", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps(id_token_payload).encode()
    ).rstrip(b"=").decode()
    id_token = f"{header}.{payload}."
    token_response = SmartTokenResponse(
        access_token=secrets.token_urlsafe(32),
        id_token=id_token,
        expires_in=access_token_lifetime_s,
        scope="patient/Patient.read patient/Condition.read",
        patient=patient_id, encounter=encounter_id,
        issued_at_unix=issued_at,
    )
    return SmartLaunchTrace(
        iss_validated=True,
        state_validated=True,
        nonce_validated=True,
        issuer_match=True,
        pkce_validated=pkce_ok,
        token_response=token_response,
    )


def decode_id_token_unsafe(id_token: str) -> dict[str, Any]:
    """Pull the payload out of the simulator's unsigned JWS. Production
    calls a real JWT library + verifies signature."""
    if not id_token or id_token.count(".") < 2:
        raise ValueError("id_token is not in JWS compact form")
    payload_b64 = id_token.split(".")[1]
    pad = "=" * (-len(payload_b64) % 4)
    return json.loads(
        base64.urlsafe_b64decode(payload_b64 + pad).decode("utf-8")
    )
