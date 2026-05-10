"""Phase 12.7 -- driver for the v3 multi-target red-team campaign."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.redteam_v3 import run_redteam_multi_target


CORPUS_PATH = ROOT / "data" / "redteam" / "redteam_corpus_v2.json"
DOCS_DIR = ROOT / "docs" / "adversarial"


async def _amain() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rep = await run_redteam_multi_target(CORPUS_PATH)

    artefact = {
        "captured_at_iso": timestamp,
        "corpus_id": rep.corpus_id,
        "n_tools_evaluated": rep.n_tools_evaluated,
        "overall_avg_pass_rate": rep.overall_avg_pass_rate,
        "rows": [r.__dict__ for r in rep.rows],
    }
    (DOCS_DIR / "red_team_map.json").write_text(
        json.dumps(artefact, indent=2, default=str), encoding="utf-8",
    )

    md: list[str] = []
    md.append("# Red-Team Robustness Map -- v3 multi-target campaign")
    md.append("")
    md.append(f"**Phase 12.7 -- captured {timestamp}**")
    md.append("")
    md.append(f"- Corpus: `{rep.corpus_id}` (110 prompts × 10 categories)")
    md.append(f"- Tools evaluated: **{rep.n_tools_evaluated}**")
    md.append(
        f"- **Average pass-rate across tools: "
        f"{rep.overall_avg_pass_rate * 100:.1f}%**"
    )
    md.append("")
    md.append("## Per-tool pass-rate")
    md.append("")
    md.append("| Tool | Cases | Passed | Failed | Pass-rate | Posture | Sample failures |")
    md.append("|---|---|---|---|---|---|---|")
    for row in sorted(rep.rows, key=lambda r: -r.pass_rate):
        sf = ", ".join(row.sample_failures) if row.sample_failures else "-"
        md.append(
            f"| {row.tool_name} | {row.n_cases} | {row.n_passed} | "
            f"{row.n_failed} | {row.pass_rate * 100:.1f}% | "
            f"**{row.posture}** | {sf} |"
        )
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append(
        "- Tools without a v3 adapter (signatures that don't accept a "
        "free-text payload) are not in the table; the v2 single-target "
        "campaign in `RED_TEAM_RESULTS.md` covers `detect_phi` "
        "specifically.\n"
        "- `posture = pass` means every prompt passed the cite-back-aware "
        "scorer; `warn` means a small fraction failed; `fail` indicates "
        "≥ 5% failure rate."
    )
    md.append("")
    (DOCS_DIR / "RED_TEAM_MAP.md").write_text(
        "\n".join(md), encoding="utf-8",
    )
    print(f"Wrote {DOCS_DIR / 'RED_TEAM_MAP.md'}")
    print(f"Average pass-rate: {rep.overall_avg_pass_rate:.3f} "
                f"across {rep.n_tools_evaluated} tools")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_amain()))
