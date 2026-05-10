"""OAuth bearer-token validation."""

from __future__ import annotations

import os
from dataclasses import dataclass

import jwt

from .issuer import OAuthClient, load_clients_from_env


class OAuthValidationError(Exception):
    """Raised on invalid bearer token. The middleware turns this into a 401."""


@dataclass
class ValidatedToken:
    client_id: str
    tenant: str
    scopes: list[str]
    expires_at: int  # epoch seconds


class TokenValidator:
    """Validates HS256 JWT bearer tokens against the configured signing secret."""

    def __init__(
        self,
        signing_secret: str,
        clients: list[OAuthClient] | None = None,
    ) -> None:
        if not signing_secret:
            raise ValueError("signing_secret must be non-empty")
        self.signing_secret = signing_secret
        self._clients_by_id: dict[str, OAuthClient] = {
            c.client_id: c for c in (clients or [])
        }

    def validate(self, token: str) -> ValidatedToken:
        """Verify signature + expiration. Returns ValidatedToken or raises.

        Does NOT verify the FHIR-server-allowlist -- that's a separate step
        bound to the request's X-FHIR-Server-URL header (see
        check_fhir_server_allowed).
        """
        if not token:
            raise OAuthValidationError("missing_token")
        try:
            payload = jwt.decode(
                token,
                self.signing_secret,
                algorithms=["HS256"],
                options={"require": ["exp", "sub", "iss"]},
            )
        except jwt.ExpiredSignatureError as e:
            raise OAuthValidationError("token_expired") from e
        except jwt.InvalidTokenError as e:
            raise OAuthValidationError(f"invalid_token: {e}") from e

        if payload.get("iss") != "trustedrisk":
            raise OAuthValidationError("invalid_issuer")

        client_id = str(payload.get("sub", ""))
        if not client_id:
            raise OAuthValidationError("missing_subject")

        return ValidatedToken(
            client_id=client_id,
            tenant=str(payload.get("tenant", "")),
            scopes=str(payload.get("scope", "")).split(),
            expires_at=int(payload.get("exp", 0)),
        )

    def check_fhir_server_allowed(
        self,
        client_id: str,
        requested_fhir_url: str,
    ) -> bool:
        """Verify the request's X-FHIR-Server-URL is in the client's allowlist.

        This is the multi-tenant isolation guard: even if a token is valid,
        the client cannot use it to query a FHIR endpoint that isn't on its
        registered list. Returns False on unknown client.
        """
        client = self._clients_by_id.get(client_id)
        if client is None:
            return False
        if not client.allowed_fhir_servers:
            # Empty allowlist = wildcard allowed (used for trusted internal clients)
            return True
        # Match prefix to allow trailing path / query strings
        return any(
            requested_fhir_url.startswith(allowed)
            for allowed in client.allowed_fhir_servers
        )


def validate_token(token: str) -> ValidatedToken:
    """Convenience function: build validator from env, validate token."""
    secret = os.environ.get("TRUSTEDRISK_OAUTH_SECRET")
    if not secret:
        raise OAuthValidationError("oauth_disabled_no_secret")
    validator = TokenValidator(secret, clients=load_clients_from_env())
    return validator.validate(token)
