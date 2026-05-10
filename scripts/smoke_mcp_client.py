"""Smoke test the live TrustedRisk MCP server with the FastMCP client.

Usage:
    .venv/Scripts/python.exe scripts/smoke_mcp_client.py

Server must be running on TRUSTEDRISK_PORT (default 8080, smoke uses 8765).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


async def main() -> int:
    try:
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport
    except ImportError as e:
        print(f"FastMCP client not installed: {e}", file=sys.stderr)
        return 1

    url = os.environ.get("TRUSTEDRISK_URL", "http://127.0.0.1:8765/mcp")
    headers = {
        "X-FHIR-Server-URL": "http://stub.fhir.local",
        "X-FHIR-Access-Token": "stub-token",
        "X-Patient-ID": "pt-smoke-001",
    }

    transport = StreamableHttpTransport(url=url, headers=headers)
    print(f"Connecting to: {url}")
    print(f"Headers: {list(headers.keys())}")
    print()

    async with Client(transport=transport) as client:
        # 1. List tools
        tools = await client.list_tools()
        print(f"=== {len(tools)} tools registered ===")
        for t in tools:
            print(f"  - {t.name}: {(t.description or '').splitlines()[0][:80]}")
        print()

        # 2. Show one tool's schema
        if tools:
            t = next((x for x in tools if "readmission" in x.name.lower()), tools[0])
            print(f"=== Schema for {t.name} ===")
            schema = t.inputSchema
            if schema:
                print(json.dumps(schema, indent=2)[:600])
            print()

        # 3. Call detect_phi (the only one that doesn't need a real FHIR server fetch)
        print("=== Calling detect_phi on stub text ===")
        try:
            result = await client.call_tool(
                "detect_phi",
                arguments={"text": "Patient John Doe (MRN 12345) discharged to home with care."},
            )
            content = result.content if hasattr(result, "content") else result
            if isinstance(content, list) and content:
                first = content[0]
                payload = getattr(first, "text", str(first))
                print(payload[:500])
            else:
                print(str(result)[:500])
        except Exception as e:
            print(f"detect_phi call failed: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
