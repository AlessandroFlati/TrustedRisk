"""INT-4 -- Cross-MCP round-trip test.

Acts as a *peer MCP client* invoking the TrustedRisk MCP server through the
standard MCP protocol surface (initialize -> tools/list -> tools/call).

Uses FastMCP's in-process Client transport, which exercises the same API
contract as any external MCP-compliant agent: `Client(server)` speaks
JSON-RPC against the FastMCP instance via in-memory transport, with no
HTTP layer needed.

This test confirms:
  1. The server's `initialize` handshake succeeds with the standard MCP
     client implementation.
  2. `tools/list` enumerates every registered tool with a JSON Schema.
  3. `tools/call` for a stateless pure-function tool (DKA severity)
     returns a structured result.
  4. `tools/call` correctly surfaces server-side validation errors.
"""
from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(scope="module")
def mcp_instance():
    """Build the TrustedRisk MCP server (in-process -- no HTTP)."""
    from mcp_server.server import _build_mcp
    mcp = _build_mcp()
    return mcp


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Initialize handshake ───────────────────────

def test_initialize_handshake_succeeds(mcp_instance):
    """A peer client can complete the MCP `initialize` handshake."""
    from fastmcp import Client

    async def go():
        async with Client(mcp_instance) as client:
            # Client auto-initializes on context entry; ping confirms session
            await client.ping()
            return client.is_connected()

    assert _run(go()) is True


# ─────────────────────── tools/list ───────────────────────

def test_tools_list_enumerates_every_registered_tool(mcp_instance):
    """tools/list returns a complete catalog with name + JSON Schema."""
    from fastmcp import Client

    async def go():
        async with Client(mcp_instance) as client:
            return await client.list_tools()

    tools = _run(go())
    names = {t.name for t in tools}

    # The registered tool surface -- any subset that's stable is fine
    expected_subset = {
        "compute_readmission_risk",
        "compute_dka_severity",
        "compute_expected_value_of_intervention",
        "compute_resolve_patient_from_query",
        "compute_differential_diagnosis_ranker",
        "ground_claim",
        "detect_phi",
    }
    assert expected_subset <= names, (
        f"Missing expected tools: {expected_subset - names}")

    # Every tool must declare a JSON Schema-shaped input contract
    for tool in tools:
        assert tool.name
        # FastMCP exposes the input schema in different attrs across versions
        schema = getattr(tool, "inputSchema", None) or \
            getattr(tool, "input_schema", None)
        assert schema is not None, f"{tool.name} missing input schema"


# ─────────────────────── tools/call -- stateless pure-function ───────────────────────

def test_tools_call_dka_severity_returns_structured_result(mcp_instance):
    """A pure-function tool (no FHIR context required) round-trips cleanly."""
    from fastmcp import Client

    async def go():
        async with Client(mcp_instance) as client:
            return await client.call_tool(
                "compute_dka_severity",
                {
                    "ph": 7.10,
                    "bicarbonate_meq_l": 12.0,
                    "glucose_mg_dl": 480.0,
                    "ketones_present": True,
                    "mental_status": "alert",
                    "potassium_meq_l": 4.2,
                    "weight_kg": 70.0,
                },
            )

    result = _run(go())
    # FastMCP returns a CallToolResult -- structured payload lives in
    # .data (or .structured_content depending on version)
    payload = (
        getattr(result, "data", None)
        or getattr(result, "structured_content", None)
        or {}
    )
    if not isinstance(payload, dict):
        # Older FastMCP wraps the dataclass -- pull via model_dump if available
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        else:
            # Fall back to scanning content blocks for a JSON dict
            for block in getattr(result, "content", []) or []:
                if hasattr(block, "text"):
                    import json
                    try:
                        payload = json.loads(block.text)
                        break
                    except Exception:
                        continue
    assert isinstance(payload, dict), \
        f"Expected dict payload, got {type(payload)}: {result!r}"
    assert payload["severity"] == "moderate"
    assert payload["icu_admission_indicated"] in (True, False)
    assert "fluid_protocol" in payload
    assert "insulin_protocol" in payload


# ─────────────────────── tools/call -- validation error path ───────────────────────

def test_tools_call_unknown_tool_errors(mcp_instance):
    """Calling an undefined tool must surface a structured error."""
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    async def go():
        async with Client(mcp_instance) as client:
            await client.call_tool("compute_does_not_exist", {})

    with pytest.raises((ToolError, Exception)):
        _run(go())


# ─────────────────────── tools/call -- chains via second tool ───────────────────────

def test_chained_tool_invocations_share_session(mcp_instance):
    """Multiple call_tool round-trips inside one session don't leak state.

    This exercises FastMCP's per-call context isolation -- important for
    server-side concurrency claims any peer MCP client makes.
    """
    from fastmcp import Client

    async def go():
        async with Client(mcp_instance) as client:
            r1 = await client.call_tool(
                "compute_dka_severity",
                {"ph": 7.10, "bicarbonate_meq_l": 12.0,
                 "glucose_mg_dl": 480.0,
                 "ketones_present": True})
            r2 = await client.call_tool(
                "compute_dka_severity",
                {"ph": 7.35, "bicarbonate_meq_l": 22.0,
                 "glucose_mg_dl": 220.0,
                 "ketones_present": False})
            return r1, r2

    r1, r2 = _run(go())

    def _payload(res):
        p = (getattr(res, "data", None)
             or getattr(res, "structured_content", None))
        if hasattr(p, "model_dump"):
            return p.model_dump()
        if isinstance(p, dict):
            return p
        for block in getattr(res, "content", []) or []:
            if hasattr(block, "text"):
                import json
                try:
                    return json.loads(block.text)
                except Exception:
                    continue
        return {}

    p1 = _payload(r1)
    p2 = _payload(r2)
    # First call -> moderate DKA, second -> not_dka
    assert p1["severity"] == "moderate"
    assert p2["severity"] == "not_dka"


# ─────────────────────── Schema completeness ───────────────────────

def test_tool_schemas_describe_all_required_args(mcp_instance):
    """Every tool's JSON Schema should declare its parameters."""
    from fastmcp import Client

    async def go():
        async with Client(mcp_instance) as client:
            return await client.list_tools()

    tools = _run(go())
    by_name = {t.name: t for t in tools}

    target = by_name["compute_dka_severity"]
    schema = (getattr(target, "inputSchema", None)
                or getattr(target, "input_schema", None))
    props = schema.get("properties", {}) if isinstance(schema, dict) \
        else schema.properties
    # Pure-function tool -- every documented arg should be in the schema
    for arg in ("ph", "bicarbonate_meq_l", "glucose_mg_dl",
                  "ketones_present", "mental_status"):
        assert arg in props, f"compute_dka_severity schema missing {arg!r}"
