"""Normalize on-disk agent cards to the A2A v1 proto-schema served shape.

The cards we keep on disk carry a few legacy patterns:

  - The `<deploy-url>` placeholder for the public hostname.
  - OpenAPI-style `type` discriminators inside security scheme objects.
  - `protocolBinding: "A2A"` instead of one of the canonical bindings
    listed in the proto spec (JSONRPC / GRPC / HTTP+JSON).

Every public agent endpoint runs this function on the parsed card before
returning it. The transforms are idempotent and side-effect free; the
file on disk stays deploy-agnostic so dry-run snapshots remain obvious.

Reference: https://raw.githubusercontent.com/a2aproject/A2A/main/specification/a2a.proto
"""

from __future__ import annotations

import os
from typing import Any


def _walk(obj: Any, deploy_host: str) -> Any:
    if isinstance(obj, str):
        return obj.replace("<deploy-url>", deploy_host)
    if isinstance(obj, dict):
        return {k: _walk(v, deploy_host) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v, deploy_host) for v in obj]
    return obj


def normalize_agent_card(card: dict) -> dict:
    """Return a card dict ready to serve under `/.well-known/agent-card.json`.

    Three transforms:
      1. Substitute `<deploy-url>` against `TRUSTEDRISK_DEPLOY_HOST`
         (default keeps the literal placeholder for unconfigured hosts).
      2. Strip the OpenAPI-style `type` field from every security scheme
         value -- the A2A proto SecurityScheme is a oneof keyed by the
         wrapper field name, the inner object MUST NOT carry `type`.
      3. Rewrite `protocolBinding: "A2A"` to `"JSONRPC"`. A2A is the
         protocol name; the wire format is JSON-RPC over HTTP.
    """
    deploy_host = os.environ.get("TRUSTEDRISK_DEPLOY_HOST", "<deploy-url>")
    normalized = _walk(card, deploy_host)

    for scheme in (normalized.get("securitySchemes") or {}).values():
        if not isinstance(scheme, dict):
            continue
        for inner in scheme.values():
            if isinstance(inner, dict):
                inner.pop("type", None)

    for iface in normalized.get("supportedInterfaces") or []:
        if isinstance(iface, dict) and iface.get("protocolBinding") == "A2A":
            iface["protocolBinding"] = "JSONRPC"

    return normalized
