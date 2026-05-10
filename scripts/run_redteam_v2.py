"""Phase 10.4 -- driver for the v2 red-team corpus.

Runs the corpus against detect_phi (a stable, dependency-free target)
and writes:

  - docs/adversarial/RED_TEAM_RESULTS.md (markdown summary)
  - docs/adversarial/red_team_run.json   (full RedTeamReport)

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_redteam_v2.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.redteam_v2 import run_redteam_corpus
from mcp_server.tools.detect_phi import detect_phi


CORPUS_PATH = ROOT / "data" / "redteam" / "redteam_corpus_v2.json"
DOCS_DIR = ROOT / "docs" / "adversarial"


async def _phi_target(payload: str):
    return await detect_phi(text=payload)


async def _amain() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    rep = await run_redteam_corpus(
        CORPUS_PATH, _phi_target, target_label="detect_phi",
        corpus_id="redteam-v2-2026-04",
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Write structured artefact
    (DOCS_DIR / "red_team_run.json").write_text(
        json.dumps(rep.model_dump(), indent=2, default=str),
        encoding="utf-8",
    )

    md_lines: list[str] = []
    md_lines.append("# Red-Team Results -- v2 Corpus")
    md_lines.append("")
    md_lines.append(f"- Corpus: `{rep.corpus_id}`")
    md_lines.append(f"- Target: `{rep.target_label}`")
    md_lines.append(f"- Generated: {timestamp}")
    md_lines.append(f"- Cases: **{rep.n_cases}**")
    md_lines.append(f"- Passed: **{rep.n_passed}** "
                       f"(rate {rep.overall_pass_rate:.3f})")
    md_lines.append(f"- Failed: **{rep.n_failed}**")
    md_lines.append(f"- Posture: **{rep.posture.upper()}**")
    md_lines.append("")
    md_lines.append("## Pass-rate by category")
    md_lines.append("")
    md_lines.append("| Category | Pass rate |")
    md_lines.append("|---|---|")
    for cat in sorted(rep.pass_rate_by_category):
        md_lines.append(f"| {cat} | {rep.pass_rate_by_category[cat]:.3f} |")
    md_lines.append("")
    md_lines.append("## Pass-rate by severity")
    md_lines.append("")
    md_lines.append("| Severity | Pass rate |")
    md_lines.append("|---|---|")
    for sev in ("critical", "high", "moderate", "low", "informational"):
        if sev in rep.pass_rate_by_severity:
            md_lines.append(f"| {sev} | {rep.pass_rate_by_severity[sev]:.3f} |")
    md_lines.append("")
    md_lines.append("## Failures (first 25)")
    md_lines.append("")
    md_lines.append("| ID | Cat | Sev | Observed | Expected | Forbidden hits |")
    md_lines.append("|---|---|---|---|---|---|")
    failed = [c for c in rep.cases if not c.passed][:25]
    for c in failed:
        hits = ", ".join(c.forbidden_hits) if c.forbidden_hits else "-"
        md_lines.append(
            f"| {c.prompt_id} | {c.category} | {c.severity} | "
            f"{c.observed_outcome} | {c.expected_outcome} | {hits} |"
        )
    md_lines.append("")
    md_lines.append("## Rationale")
    md_lines.append("")
    md_lines.append(rep.rationale)
    md_lines.append("")
    md_lines.append("## References")
    md_lines.append("")
    for ref in rep.references:
        md_lines.append(f"- {ref}")
    md_lines.append("")

    (DOCS_DIR / "RED_TEAM_RESULTS.md").write_text(
        "\n".join(md_lines), encoding="utf-8",
    )
    print(f"Wrote {DOCS_DIR / 'RED_TEAM_RESULTS.md'}")
    print(f"Wrote {DOCS_DIR / 'red_team_run.json'}")
    print(f"Posture: {rep.posture} -- {rep.n_passed}/{rep.n_cases} pass")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_amain()))
