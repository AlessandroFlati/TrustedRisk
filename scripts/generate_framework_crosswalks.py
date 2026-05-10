"""Phase 16.G2 - Render the NIST AI RMF + OECD AI Principles crosswalks.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/generate_framework_crosswalks.py

Outputs:
    docs/regulatory/FRAMEWORK_CROSSWALKS.md
    docs/regulatory/framework_crosswalks.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.framework_crosswalks import (
    build_framework_crosswalks, render_crosswalks_md,
)


def main() -> int:
    out_dir = ROOT / "docs" / "regulatory"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_framework_crosswalks()

    (out_dir / "FRAMEWORK_CROSSWALKS.md").write_text(
        render_crosswalks_md(report), encoding="utf-8")
    (out_dir / "framework_crosswalks.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8")

    print(
        f"Wrote {out_dir / 'FRAMEWORK_CROSSWALKS.md'} "
        f"({report.n_rows} rows: "
        f"NIST {len(report.nist_ai_rmf_rows)}, "
        f"OECD {len(report.oecd_ai_principles_rows)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
