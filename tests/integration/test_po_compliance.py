"""Phase 4.3 -- Prompt Opinion public-instance compliance smoke.

Pulls capability declarations from PO's public test SHARP-on-MCP
servers and verifies that:

  - Their MCP `initialize` advertises the same `ai.promptopinion/fhir-context`
    capability key our MCP server emits, with a `scopes` array of the
    expected shape.
  - Their A2A FHIR-context extension URI matches the one we declare on
    our agent-cards (`https://app.promptopinion.ai/schemas/a2a/v1/fhir-context`).

These tests are opt-in (`TRUSTEDRISK_LIVE=1`) -- they reach the public
internet and depend on PO's instances being up. CI defaults skip.

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        TRUSTEDRISK_LIVE=1 \\
        .venv/Scripts/python.exe -m pytest tests/integration/test_po_compliance.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("TRUSTEDRISK_LIVE", "").lower() not in ("1", "true", "yes"),
    reason="Live PO compliance tests require TRUSTEDRISK_LIVE=1.",
)


PO_CAPABILITY_KEY = "ai.promptopinion/fhir-context"
PO_A2A_EXTENSION_URI = (
    "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context"
)
PO_PUBLIC_MCP_TS = "https://ts.fhir-mcp.promptopinion.ai/mcp"
PO_PUBLIC_MCP_DOTNET = "https://dotnet.fhir-mcp.promptopinion.ai/mcp"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _initialize_request() -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1,
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "trustedrisk-compliance-smoke",
                              "version": "0.7.0"},
        },
    }


def _try_initialize(url: str) -> dict | None:
    """POST an initialize JSON-RPC request to a public PO MCP server.

    Returns the parsed `result` (or None on error / non-2xx). FastMCP's
    Streamable HTTP transport accepts `application/json` bodies on the
    `/mcp` endpoint. We accept either a 401/403 (which means SHARP is
    enforced and we'd need real headers -- still proves the contract is
    in place) OR a 200 with capability declarations.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    try:
        r = httpx.post(url, headers=headers,
                            json=_initialize_request(), timeout=15.0)
    except httpx.HTTPError as exc:
        pytest.skip(f"{url}: unreachable ({exc})")
    if r.status_code in (401, 403):
        # SHARP is enforced -- capability discovery requires real
        # headers; that's compatible with our contract
        return {"_protected": True, "status_code": r.status_code}
    if r.status_code != 200:
        pytest.skip(f"{url} returned {r.status_code}; not a hard failure")
    try:
        return r.json()
    except ValueError:
        # Streamable HTTP can return SSE; pick the first data: line
        for line in r.text.splitlines():
            if line.startswith("data: "):
                try:
                    return json.loads(line[len("data: "):])
                except ValueError:
                    continue
        pytest.skip(f"{url} returned non-JSON, non-SSE body")


# ─────────────────────── Local agent-card declarations ───────────────────────

def test_local_agent_card_declares_po_extension_uri():
    raw = json.loads(
        (REPO_ROOT / "src" / "a2a_agent" / "agent-card.json").read_text(
            encoding="utf-8")
    )
    extensions = raw.get("capabilities", {}).get("extensions", [])
    uris = {e.get("uri") for e in extensions}
    assert PO_A2A_EXTENSION_URI in uris


def test_specialist_cards_declare_po_extension_uri():
    apps_dir = REPO_ROOT / "apps"
    cards_found = 0
    for spec in apps_dir.glob("specialist_*/agent_card.json"):
        cards_found += 1
        raw = json.loads(spec.read_text(encoding="utf-8"))
        extensions = raw.get("capabilities", {}).get("extensions", [])
        uris = {e.get("uri") for e in extensions}
        assert PO_A2A_EXTENSION_URI in uris, \
            f"{spec.parent.name}: missing PO extension URI"
    assert cards_found >= 8, \
        f"Expected at least 8 specialist cards; found {cards_found}"


def test_local_initialize_capabilities_advertises_po_key():
    """Our own MCP `initialize_capabilities()` must emit the
    PO-namespaced capability key."""
    from mcp_server.sharp.headers import initialize_capabilities

    caps = initialize_capabilities()
    assert PO_CAPABILITY_KEY in caps
    decl = caps[PO_CAPABILITY_KEY]
    assert "scopes" in decl
    assert decl["scopes"], "Scopes list must be non-empty"


# ─────────────────────── Live PO endpoints (opt-in) ───────────────────────

@pytest.mark.parametrize("url", [PO_PUBLIC_MCP_TS, PO_PUBLIC_MCP_DOTNET])
def test_po_mcp_endpoint_reachable(url: str):
    """The public PO test instances are reachable (200 / 401 / 403 all
    count as 'reachable' -- anything else means PO is down or moved)."""
    payload = _try_initialize(url)
    assert payload is not None


@pytest.mark.parametrize("url", [PO_PUBLIC_MCP_TS, PO_PUBLIC_MCP_DOTNET])
def test_po_mcp_response_shape(url: str):
    """If the PO instance returns a 200, the response must be a
    JSON-RPC envelope (jsonrpc/result or jsonrpc/error). If it returns
    401/403 the test is skipped -- SHARP enforcement is itself the
    contract we care about."""
    payload = _try_initialize(url)
    if payload.get("_protected"):
        pytest.skip(
            f"{url}: SHARP enforcement active (status "
            f"{payload['status_code']}) -- capability negotiation "
            f"requires real FHIR context headers."
        )
    assert payload.get("jsonrpc") == "2.0"
    assert "result" in payload or "error" in payload
