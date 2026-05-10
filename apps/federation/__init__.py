"""Unified A2A federation server.

Mounts every trustedrisk-* specialist + the composer under
`/a2a/<agent-name>` in a single Starlette process. One uvicorn worker
serves all 16 agents; the MCP root keeps running as a separate process
on its own port.
"""
