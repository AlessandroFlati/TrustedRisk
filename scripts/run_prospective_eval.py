"""Phase 15.C2 -- Render the prospective evaluation report.

Runs :func:`a2a_agent.prospective_eval.run_prospective_eval` and writes
both a JSON sidecar and a human-readable markdown report.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/run_prospective_eval.py

Outputs:
    docs/prospective/prospective_eval.json
    docs/prospective/PROSPECTIVE_EVAL.md

Pure-deterministic for the default seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.prospective_eval import run_prospective_eval

OUT_DIR = ROOT / "docs" / "prospective"


def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def _render_overall(summary) -> list[str]:
    o = summary.overall
    lines = [
        "## 1. Overall pipeline behaviour",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Cohort n | {o['n']:,} |",
        f"| Approve rate | {_fmt_pct(o['approve_rate'])} |",
        f"| Downgrade rate (4-critic / debate revise) | "
        f"{_fmt_pct(o['downgrade_rate'])} |",
        f"| Abstain rate (fairness-guard / safety floor) | "
        f"{_fmt_pct(o['abstain_rate'])} |",
        f"| Mean predicted readmission risk | "
        f"{_fmt_pct(o['mean_predicted_rate'])} |",
        f"| Observed outcome rate | "
        f"{_fmt_pct(o['observed_outcome_rate'])} |",
        f"| Absolute calibration gap | "
        f"{_fmt_pct(o['calibration_gap_abs'])} |",
        f"| Fairness-flagged share of cohort | "
        f"{_fmt_pct(o['fairness_flagged_share'])} |",
        "",
        f"**Action distribution (TrustedRisk pipeline)**: "
        f"{o['action_distribution']}",
        "",
    ]
    return lines


def _render_baseline_delta(summary) -> list[str]:
    b = summary.baseline_lace_only
    o = summary.overall
    lines = [
        "## 2. Counterfactual: LACE-threshold baseline",
        "",
        "Without the multi-agent debate / fairness-guard layer the "
        "system would deploy the LACE-threshold proposed action on "
        "every encounter.",
        "",
        "| Metric | Pipeline | LACE-only baseline | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Approve rate | {_fmt_pct(o['approve_rate'])} | "
        f"{_fmt_pct(b['approve_rate'])} | "
        f"{_fmt_pct(o['approve_rate'] - b['approve_rate'])} |",
        f"| Abstain rate | {_fmt_pct(o['abstain_rate'])} | "
        f"{_fmt_pct(b['abstain_rate'])} | "
        f"{_fmt_pct(o['abstain_rate'] - b['abstain_rate'])} |",
        f"| Downgrade rate | {_fmt_pct(o['downgrade_rate'])} | "
        f"{_fmt_pct(b['downgrade_rate'])} | "
        f"{_fmt_pct(o['downgrade_rate'] - b['downgrade_rate'])} |",
        "",
        f"**Baseline action distribution**: {b['proposed_distribution']}",
        "",
    ]
    return lines


def _render_subgroup(summary) -> list[str]:
    lines: list[str] = ["## 3. Per-subgroup metrics", ""]
    for name, by_value in summary.per_subgroup.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            "| Value | n | abstain | downgrade | predicted | observed |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: |"
        )
        for value, m in sorted(by_value.items()):
            if m.get("n", 0) == 0:
                continue
            lines.append(
                f"| {value} | {m['n']:,} | "
                f"{_fmt_pct(m['abstain_rate'])} | "
                f"{_fmt_pct(m['downgrade_rate'])} | "
                f"{_fmt_pct(m['mean_predicted_rate'])} | "
                f"{_fmt_pct(m['observed_outcome_rate'])} |"
            )
        lines.append("")
    return lines


def _render_md(summary) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# TrustedRisk - Synthetic Prospective Evaluation")
    lines.append("")
    lines.append(
        f"**Generated**: {timestamp} - **Cohort n**: "
        f"{summary.cohort_n:,} - **Seed**: {summary.seed} - "
        f"**Fairness audit present**: "
        f"{summary.fairness_audit_present}"
    )
    lines.append("")
    lines.append(
        "Counterfactual 'if we had deployed' analysis: every "
        "synthetic encounter is run through the calibrated readmission "
        "risk model -> 4-critic ensemble (deterministic-floor "
        "approximation) -> 3-agent debate. The cohort + per-encounter "
        "demographics are sampled from the same distribution the "
        "subgroup-audit builder uses (Phase 15.B1)."
    )
    lines.append("")
    lines.extend(_render_overall(summary))
    lines.extend(_render_baseline_delta(summary))
    lines.extend(_render_subgroup(summary))
    lines.append("## 4. References")
    lines.append("")
    for ref in summary.references:
        lines.append(f"- {ref}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000,
                        help="Cohort size (default: 10000)")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--no-fairness-audit", action="store_true",
                        help="Run with fairness_audit_present=False")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_prospective_eval(
        args.n, seed=args.seed,
        fairness_audit_present=not args.no_fairness_audit,
    )

    json_path = OUT_DIR / "prospective_eval.json"
    json_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    md_path = OUT_DIR / "PROSPECTIVE_EVAL.md"
    md_path.write_text(_render_md(summary), encoding="utf-8")

    o = summary.overall
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"n={summary.cohort_n:,}, abstain="
        f"{o['abstain_rate']*100:.2f}%, downgrade="
        f"{o['downgrade_rate']*100:.2f}%, calibration_gap="
        f"{o['calibration_gap_abs']*100:.2f}%."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
