"""OAuth Client Credentials grant for TrustedRisk multi-tenant deployment.

Anonymous SHARP-on-MCP context-only headers are sufficient for the demo,
but production deployments serving multiple hospitals (each with its
own FHIR endpoint and its own data-use agreement) need authenticated client
identification on top. This module implements RFC 6749 §4.4 client_credentials
grant, with HS256-signed JWT bearer tokens validated on every protected
request.

Layered architecture:

    Client                                       TrustedRisk
    -------                                      -----------
    POST /oauth/token  ─────────────────────►    issuer.issue_token()
        client_id+secret                            ↓
                                                 returns JWT (HS256)
    POST /mcp ─────────────────────────────►     validator middleware
        Authorization: Bearer <jwt>                 ↓
        X-FHIR-Server-URL: ...                   parse + verify JWT
        X-FHIR-Access-Token: ...                    ↓
                                                 ClientContext ContextVar set
                                                    ↓
                                                 SHARP middleware runs next
                                                 FHIRContext ContextVar set
                                                    ↓
                                                 tool dispatch sees both contexts

The client's `tenant` claim is matched against an allowlist of FHIR server
URLs at request time, so a token for tenant A cannot be used to query
tenant B's FHIR endpoint even if the bearer token is otherwise valid.
"""

from .issuer import (
    OAuthClient,
    TokenIssuer,
    issue_token,
    load_clients_from_env,
)
from .middleware import (
    ClientContext,
    get_client_context,
    oauth_bearer_middleware,
)
from .validator import (
    OAuthValidationError,
    TokenValidator,
    validate_token,
)

__all__ = [
    "ClientContext",
    "OAuthClient",
    "OAuthValidationError",
    "TokenIssuer",
    "TokenValidator",
    "get_client_context",
    "issue_token",
    "load_clients_from_env",
    "oauth_bearer_middleware",
    "validate_token",
]
