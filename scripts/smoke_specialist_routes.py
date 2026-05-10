"""Sanity-check every keyword route in apps/_shared/specialist_routes.

Calls each route's tool with its declared demo inputs and reports:
  OK score  -> pretty-printed summary fields
  FAIL      -> the exception type + message

This catches shape-mismatch bugs (Pydantic validation errors, missing
required arg, ...) before the routes face a chat client.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from apps._shared.specialist_routes import SPECIALIST_ROUTES


async def main() -> int:
    failures: list[tuple[str, str, str]] = []
    for slug, routes in SPECIALIST_ROUTES.items():
        print(f"\n=== {slug} ({len(routes)} route(s)) ===")
        for r in routes:
            try:
                out = await r.callable(**r.inputs)
                dump = out.model_dump() if hasattr(out, "model_dump") else out
                bits = []
                if isinstance(dump, dict):
                    for f in r.summary_fields:
                        if f in dump:
                            bits.append(f"{f}={dump[f]!r}")
                summary = "; ".join(bits) or "(no summary fields matched)"
                print(f"  OK  {r.tool_name:<45s} {summary}")
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                failures.append((slug, r.tool_name, msg))
                print(f"  FAIL {r.tool_name:<45s} {msg}")
    print(f"\n{len(failures)} failure(s).")
    for slug, tool, msg in failures:
        print(f"  - {slug}.{tool}: {msg}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
