"""Phase 17.AA - Render the cost-simulation artefact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.cost_simulator import simulate_cost_impact


def _render_md(report) -> str:
    lines = ["# TrustedRisk - Cost simulation vs LACE-only baseline",
             ""]
    o = report.overall
    lines.append(
        f"**Cohort n**: {report.cohort_n:,} - "
        f"**Readmissions averted**: "
        f"{o.readmissions_averted_vs_baseline:,} - "
        f"**Cost saved (overall)**: ${o.cost_saved_usd:,.0f} - "
        f"**QALYs gained**: {o.qalys_gained:.2f}"
    )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(
        f"| Cohort size | {o.n:,} |")
    lines.append(
        f"| Pipeline-predicted readmits | "
        f"{o.readmissions_predicted:,} |")
    lines.append(
        f"| Readmissions averted vs LACE-only baseline | "
        f"{o.readmissions_averted_vs_baseline:,} |")
    lines.append(
        f"| Abstain rate | {o.abstain_rate*100:.2f}% |")
    lines.append(
        f"| Cost saved (USD) | ${o.cost_saved_usd:,.0f} |")
    lines.append(
        f"| QALYs gained | {o.qalys_gained:.2f} |")
    lines.append(
        f"| ICER per QALY | ${o.icer_per_qaly:,.0f} |")
    lines.append(
        f"| Incremental net monetary benefit | "
        f"${o.incremental_net_monetary_benefit:,.0f} |")
    lines.append("")
    lines.append("## Per-payer breakdown")
    lines.append("")
    lines.append(
        "| Payer | n | averted | cost saved (USD) | QALYs | "
        "abstain |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for payer in sorted(report.by_payer):
        b = report.by_payer[payer]
        lines.append(
            f"| {payer} | {b.n:,} | "
            f"{b.readmissions_averted_vs_baseline:,} | "
            f"${b.cost_saved_usd:,.0f} | "
            f"{b.qalys_gained:.2f} | "
            f"{b.abstain_rate*100:.2f}% |"
        )
    lines.append("")
    lines.append("## References")
    lines.append("")
    for ref in report.references:
        lines.append(f"- {ref}")
    return "\n".join(lines)


def main() -> int:
    out_dir = ROOT / "docs" / "economics"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = simulate_cost_impact(n_encounters=10_000)
    (out_dir / "COST_SIMULATION.md").write_text(
        _render_md(report), encoding="utf-8")
    (out_dir / "cost_simulation.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8")
    o = report.overall
    print(
        f"n={report.cohort_n:,}, averted="
        f"{o.readmissions_averted_vs_baseline:,}, saved=$"
        f"{o.cost_saved_usd:,.0f}, QALYs={o.qalys_gained:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
