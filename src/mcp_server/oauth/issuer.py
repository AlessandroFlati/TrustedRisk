"""OAuth token issuance -- RFC 6749 §4.4 client_credentials grant.

The issuer maintains a small registry of trusted clients (loaded from a JSON
file at startup) and mints HS256-signed JWTs against a server-side secret.

Client registry format (`data/oauth_clients.json`):

    {
        "schema_version": 1,
        "clients": [
            {
                "client_id": "tenant-a-orchestrator",
                "client_secret_hash": "sha256:<hex>",
                "tenant": "tenant-a",
                "allowed_scopes": ["discharge.read", "discharge.execute", "phi.scan"],
                "allowed_fhir_servers": ["https://tenant-a.fhir.example.com/baseR4"]
            }
        ]
    }

Secrets are stored hashed (sha256). The token endpoint accepts the plaintext
secret and verifies against the hash. Production should use bcrypt or argon2;
sha256 is fine for this prototype since secrets are operator-set and rotated
infrequently.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import jwt


_DEFAULT_TTL = 3600  # 1 hour


@dataclass
class OAuthClient:
    """Registered client -- loaded from oauth_clients.json at startup."""
    client_id: str
    client_secret_hash: str
    tenant: str
    allowed_scopes: list[str] = field(default_factory=list)
    allowed_fhir_servers: list[str] = field(default_factory=list)

    @staticmethod
    def hash_secret(plaintext: str) -> str:
        """Hash a plaintext secret. Format `sha256:<hex>`."""
        digest = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def verify_secret(self, plaintext: str) -> bool:
        """Constant-time compare -- defends against timing oracles."""
        if not self.client_secret_hash.startswith("sha256:"):
            return False
        expected = self.client_secret_hash.split(":", 1)[1]
        actual = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
        # hmac.compare_digest is constant-time
        import hmac
        return hmac.compare_digest(expected, actual)


class TokenIssuer:
    """Issues HS256 JWT bearer tokens for registered clients."""

    def __init__(
        self,
        clients: list[OAuthClient],
        signing_secret: str,
        token_ttl_seconds: int = _DEFAULT_TTL,
    ) -> None:
        if not signing_secret:
            raise ValueError("signing_secret must be non-empty")
        self.signing_secret = signing_secret
        self.token_ttl_seconds = max(60, int(token_ttl_seconds))
        self._clients_by_id: dict[str, OAuthClient] = {c.client_id: c for c in clients}

    def issue(
        self,
        client_id: str,
        client_secret: str,
        requested_scopes: list[str] | None = None,
    ) -> dict:
        """Issue a new access token. Raises ValueError on auth failure or
        unauthorized scope request.

        Returns:
            dict shaped per OAuth 2.0 token response:
                {access_token, token_type, expires_in, scope, tenant}
        """
        client = self._clients_by_id.get(client_id)
        if client is None or not client.verify_secret(client_secret):
            raise ValueError("invalid_client")

        # Intersection of requested and allowed scopes (RFC 6749 §3.3 narrowing)
        allowed = set(client.allowed_scopes)
        if requested_scopes:
            granted = [s for s in requested_scopes if s in allowed]
        else:
            granted = list(allowed)
        if not granted:
            raise ValueError("invalid_scope")

        now = int(time.time())
        payload = {
            "iss": "trustedrisk",
            "sub": client.client_id,
            "tenant": client.tenant,
            "scope": " ".join(granted),
            "iat": now,
            "exp": now + self.token_ttl_seconds,
        }
        token = jwt.encode(payload, self.signing_secret, algorithm="HS256")

        return {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": self.token_ttl_seconds,
            "scope": " ".join(granted),
            "tenant": client.tenant,
        }


# ─────────────────────── Client registry loading ───────────────────────

def load_clients_from_env() -> list[OAuthClient]:
    """Load the OAuth client registry from `TRUSTEDRISK_OAUTH_CLIENTS_PATH`.

    Returns an empty list when the file is absent (OAuth-disabled deployment).
    """
    path = os.environ.get("TRUSTEDRISK_OAUTH_CLIENTS_PATH",
                          "data/oauth_clients.json")
    fp = Path(path)
    if not fp.exists():
        return []
    try:
        with fp.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    raw_list = data.get("clients", []) if isinstance(data, dict) else data
    out: list[OAuthClient] = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        try:
            out.append(OAuthClient(
                client_id=str(raw["client_id"]),
                client_secret_hash=str(raw["client_secret_hash"]),
                tenant=str(raw.get("tenant", raw["client_id"])),
                allowed_scopes=list(raw.get("allowed_scopes", [])),
                allowed_fhir_servers=list(raw.get("allowed_fhir_servers", [])),
            ))
        except (KeyError, TypeError):
            continue
    return out


def issue_token(
    client_id: str,
    client_secret: str,
    requested_scopes: list[str] | None = None,
) -> dict:
    """Convenience function: build issuer from env, issue token."""
    secret = os.environ.get("TRUSTEDRISK_OAUTH_SECRET")
    if not secret:
        raise RuntimeError(
            "TRUSTEDRISK_OAUTH_SECRET is not set; OAuth issuance disabled."
        )
    ttl = int(os.environ.get("TRUSTEDRISK_OAUTH_TOKEN_TTL_SECONDS",
                              str(_DEFAULT_TTL)))
    issuer = TokenIssuer(load_clients_from_env(), signing_secret=secret,
                         token_ttl_seconds=ttl)
    return issuer.issue(client_id, client_secret, requested_scopes)
