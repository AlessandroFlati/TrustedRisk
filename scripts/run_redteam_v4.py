"""Phase 16.I1 - Run the red-team v4 corpus + persist the report.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_redteam_v4.py

Outputs:
    docs/adversarial/red_team_v4.json
    docs/adversarial/RED_TEAM_V4.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.redteam_v4 import run_redteam_v4_sync


def _render_md(report) -> str:
    lines = ["# TrustedRisk - Red-team v4 indirect prompt injection",
             ""]
    lines.append(
        f"**n_cases**: {report.n_cases}, "
        f"**safe**: {report.n_safe}, "
        f"**partially_safe**: {report.n_partially_safe}, "
        f"**unsafe**: {report.n_unsafe}, "
        f"**overall posture**: `{report.overall_posture}`"
    )
    lines.append("")
    lines.append("## Per-attack-class posture")
    lines.append("")
    lines.append("| Attack class | safe | partially_safe | unsafe |")
    lines.append("| --- | ---: | ---: | ---: |")
    for cls, counts in sorted(report.per_class.items()):
        lines.append(
            f"| {cls} | {counts['safe']} | "
            f"{counts['partially_safe']} | {counts['unsafe']} |"
        )
    lines.append("")
    lines.append("## Per-case detail")
    lines.append("")
    lines.append(
        "| case_id | attack_class | target_tool | posture | evidence |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for r in report.results:
        lines.append(
            f"| {r.case_id} | {r.attack_class} | {r.target_tool} | "
            f"{r.posture} | {r.posture_evidence} |"
        )
    return "\n".join(lines)


def main() -> int:
    out_dir = ROOT / "docs" / "adversarial"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_redteam_v4_sync()
    (out_dir / "red_team_v4.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8")
    (out_dir / "RED_TEAM_V4.md").write_text(
        _render_md(report), encoding="utf-8")
    print(
        f"n_cases={report.n_cases}, "
        f"safe={report.n_safe}, "
        f"unsafe={report.n_unsafe}, "
        f"posture={report.overall_posture}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
