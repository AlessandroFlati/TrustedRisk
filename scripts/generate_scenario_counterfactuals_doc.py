"""Phase 12.5 B3 -- generate docs/scenarios/COUNTERFACTUALS.md.

Runs the 5 per-scenario counterfactual analyzers and serialises the
result to a single human-readable markdown table. Re-run on every
build to track which factor combinations flip the recommended
disposition.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.scenario_counterfactuals import run_all_counterfactuals


DOCS_DIR = ROOT / "docs" / "scenarios"


async def _amain() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    reps = await run_all_counterfactuals()

    md: list[str] = []
    md.append("# Counterfactual Explanations -- Per-scenario right-to-explanation surface")
    md.append("")
    md.append(f"**Phase 12.5 -- captured {timestamp}**")
    md.append("")
    md.append(
        "For each of the 5 end-to-end clinical scenarios, the harness "
        "perturbs 2-4 key input factors and detects the **minimum-"
        "modification** path that changes the recommended decision tier. "
        "The table below is regenerated on every CI build by "
        "`scripts/generate_scenario_counterfactuals_doc.py`."
    )
    md.append("")
    cumulative_perts = sum(r.perturbations_evaluated for r in reps)
    cumulative_flips = sum(r.n_flips for r in reps)
    md.append(
        f"**Aggregate:** {cumulative_flips} flip(s) across "
        f"{cumulative_perts} perturbation(s) over {len(reps)} scenarios."
    )
    md.append("")

    for rep in reps:
        md.append(f"## {rep.title} (`{rep.scenario_id}`)")
        md.append("")
        md.append(f"- Baseline outcome: **{rep.baseline_outcome}**")
        md.append(
            f"- Perturbations evaluated: {rep.perturbations_evaluated} -- "
            f"flips found: **{rep.n_flips}**"
        )
        md.append("")
        if rep.flips_found:
            md.append("| Factor | Original | Modified | Original outcome | "
                          "Modified outcome | Distance |")
            md.append("|---|---|---|---|---|---|")
            for f in rep.flips_found:
                md.append(
                    f"| {f.factor_name} | {f.original_value} | "
                    f"{f.modified_value} | {f.original_outcome} | "
                    f"**{f.modified_outcome}** | {f.flip_distance:g} |"
                )
            md.append("")
        md.append(f"_Rationale_: {rep.rationale}")
        md.append("")
        md.append("**References:**")
        for ref in rep.references:
            md.append(f"- {ref}")
        md.append("")

    md.append("## Notes")
    md.append("")
    md.append(
        "- Single-factor perturbations are scored against the original "
        "deterministic-floor output. A flip means the perturbed run "
        "produced a different `decision` / `disposition` / `severity_tier`."
    )
    md.append(
        "- These reports are part of the GDPR Art. 22 / EU AI Act Art. 13 "
        "right-to-explanation surface -- every emitted DecisionCard can be "
        "paired with the corresponding scenario counterfactuals on demand."
    )
    md.append("")
    (DOCS_DIR / "COUNTERFACTUALS.md").write_text(
        "\n".join(md), encoding="utf-8",
    )
    print(f"Wrote {DOCS_DIR / 'COUNTERFACTUALS.md'}")
    print(f"Cumulative: {cumulative_flips}/{cumulative_perts} perturbations "
                f"flipped across {len(reps)} scenarios.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_amain()))
