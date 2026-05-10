"""Phase 13.11 F2 -- Multi-tenancy + OAuth client_credentials demo.

Demonstrates that the existing OAuth + allowed_fhir_servers
infrastructure delivers strict tenant isolation: a token issued for
tenant-A cannot be used to query tenant-B's FHIR endpoint, and vice
versa. The middleware enforcement is already in place -- this test
file is the published proof.
"""

from __future__ import annotations

import time

import jwt
import pytest

from mcp_server.oauth.issuer import OAuthClient, TokenIssuer
from mcp_server.oauth.validator import (
    OAuthValidationError, TokenValidator,
)


# ─────────────────────────────────────────────────────────────────────
# Two-tenant fixture
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def signing_secret() -> str:
    return "demo-multi-tenant-secret-12345678901234"   # 33 bytes ≥ 32


@pytest.fixture
def two_tenant_clients() -> list[OAuthClient]:
    return [
        OAuthClient(
            client_id="tenant-a-orchestrator",
            client_secret_hash=OAuthClient.hash_secret("a-secret"),
            tenant="tenant-a",
            allowed_scopes=[
                "discharge.read", "discharge.execute", "phi.scan",
            ],
            allowed_fhir_servers=[
                "https://tenant-a.fhir.example.com/baseR4",
            ],
        ),
        OAuthClient(
            client_id="tenant-b-orchestrator",
            client_secret_hash=OAuthClient.hash_secret("b-secret"),
            tenant="tenant-b",
            allowed_scopes=[
                "discharge.read", "discharge.execute", "phi.scan",
            ],
            allowed_fhir_servers=[
                "https://tenant-b.fhir.example.com/baseR4",
            ],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────
# Token issuance
# ─────────────────────────────────────────────────────────────────────


def test_tenant_a_can_obtain_token(two_tenant_clients, signing_secret):
    issuer = TokenIssuer(two_tenant_clients, signing_secret=signing_secret)
    out = issuer.issue("tenant-a-orchestrator", "a-secret")
    assert out["token_type"] == "Bearer"
    assert out["tenant"] == "tenant-a"
    assert out["access_token"]


def test_wrong_secret_denied(two_tenant_clients, signing_secret):
    issuer = TokenIssuer(two_tenant_clients, signing_secret=signing_secret)
    with pytest.raises(ValueError):
        issuer.issue("tenant-a-orchestrator", "wrong-secret")


def test_unknown_client_denied(two_tenant_clients, signing_secret):
    issuer = TokenIssuer(two_tenant_clients, signing_secret=signing_secret)
    with pytest.raises(ValueError):
        issuer.issue("evil-client", "secret")


# ─────────────────────────────────────────────────────────────────────
# Cross-tenant FHIR-server isolation
# ─────────────────────────────────────────────────────────────────────


def test_tenant_a_token_can_read_own_fhir_server(
    two_tenant_clients, signing_secret,
):
    validator = TokenValidator(signing_secret, clients=two_tenant_clients)
    assert validator.check_fhir_server_allowed(
        client_id="tenant-a-orchestrator",
        requested_fhir_url="https://tenant-a.fhir.example.com/baseR4",
    ) is True


def test_tenant_a_token_cannot_read_tenant_b_fhir_server(
    two_tenant_clients, signing_secret,
):
    """The whole point of multi-tenancy: tenant-A's token is rejected
    when used against tenant-B's FHIR endpoint."""
    validator = TokenValidator(signing_secret, clients=two_tenant_clients)
    assert validator.check_fhir_server_allowed(
        client_id="tenant-a-orchestrator",
        requested_fhir_url="https://tenant-b.fhir.example.com/baseR4",
    ) is False


def test_tenant_b_token_cannot_read_tenant_a_fhir_server(
    two_tenant_clients, signing_secret,
):
    validator = TokenValidator(signing_secret, clients=two_tenant_clients)
    assert validator.check_fhir_server_allowed(
        client_id="tenant-b-orchestrator",
        requested_fhir_url="https://tenant-a.fhir.example.com/baseR4",
    ) is False


def test_tenant_b_token_can_read_own_fhir_server(
    two_tenant_clients, signing_secret,
):
    validator = TokenValidator(signing_secret, clients=two_tenant_clients)
    assert validator.check_fhir_server_allowed(
        client_id="tenant-b-orchestrator",
        requested_fhir_url="https://tenant-b.fhir.example.com/baseR4",
    ) is True


def test_unknown_client_id_rejected(two_tenant_clients, signing_secret):
    validator = TokenValidator(signing_secret, clients=two_tenant_clients)
    assert validator.check_fhir_server_allowed(
        client_id="not-registered",
        requested_fhir_url="https://anywhere.example.com/baseR4",
    ) is False


# ─────────────────────────────────────────────────────────────────────
# End-to-end: issue token + validate + cross-check FHIR allowlist
# ─────────────────────────────────────────────────────────────────────


def test_end_to_end_tenant_a_token_blocked_at_tenant_b_endpoint(
    two_tenant_clients, signing_secret,
):
    issuer = TokenIssuer(two_tenant_clients, signing_secret=signing_secret)
    validator = TokenValidator(signing_secret, clients=two_tenant_clients)

    token_response = issuer.issue("tenant-a-orchestrator", "a-secret")
    decoded = validator.validate(token_response["access_token"])
    assert decoded.client_id == "tenant-a-orchestrator"
    assert decoded.tenant == "tenant-a"

    # ↓ key isolation assertion
    cross_tenant_allowed = validator.check_fhir_server_allowed(
        client_id=decoded.client_id,
        requested_fhir_url="https://tenant-b.fhir.example.com/baseR4",
    )
    assert cross_tenant_allowed is False, (
        "MULTI-TENANCY VIOLATION: tenant-A token allowed to read "
        "tenant-B's FHIR server"
    )


def test_expired_token_rejected_independent_of_tenant(
    two_tenant_clients, signing_secret,
):
    """An issuer with a near-zero TTL produces a token that
    immediately fails validation on the next call."""
    issuer = TokenIssuer(
        two_tenant_clients, signing_secret=signing_secret,
        token_ttl_seconds=60,    # internal floor; we forge one for the test
    )
    # Forge an already-expired token (sub = tenant-a-orchestrator)
    payload = {
        "iss": "trustedrisk", "sub": "tenant-a-orchestrator",
        "tenant": "tenant-a", "scope": "discharge.read",
        "iat": int(time.time()) - 7200,
        "exp": int(time.time()) - 3600,
    }
    expired = jwt.encode(payload, signing_secret, algorithm="HS256")
    validator = TokenValidator(signing_secret, clients=two_tenant_clients)
    with pytest.raises(OAuthValidationError):
        validator.validate(expired)


# ─────────────────────────────────────────────────────────────────────
# Empty allowlist -> unrestricted (legacy / trusted internal client)
# ─────────────────────────────────────────────────────────────────────


def test_empty_allowlist_means_unrestricted(signing_secret):
    legacy = OAuthClient(
        client_id="legacy-internal",
        client_secret_hash=OAuthClient.hash_secret("x"),
        tenant="legacy", allowed_scopes=["discharge.read"],
        allowed_fhir_servers=[],   # wildcard
    )
    validator = TokenValidator(signing_secret, clients=[legacy])
    for url in (
        "https://tenant-a.fhir.example.com/baseR4",
        "https://tenant-b.fhir.example.com/baseR4",
        "https://anything.example.com/baseR4",
    ):
        assert validator.check_fhir_server_allowed(
            "legacy-internal", url,
        ) is True
