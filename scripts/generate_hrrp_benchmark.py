"""Phase 17.S - Render the HRRP literature benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.hrrp_benchmark import (
    benchmark_against_hrrp, render_hrrp_md,
)


def main() -> int:
    out_dir = ROOT / "docs" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = ROOT / "docs" / "fairness" / "subgroup_audit.json"
    sub_rates: dict[str, dict[str, float]] = {}
    if audit.exists():
        data = json.loads(audit.read_text(encoding="utf-8"))
        sg = data.get("subgroups", {})
        for axis_label, source_key in (
            ("race", "race"),
            ("insurance", "insurance_type"),
        ):
            axis_data = sg.get(source_key, {}).get("by_value", {})
            sub_rates[axis_label] = {
                v: m.get("mean_predicted_rate", 0.0)
                for v, m in axis_data.items()
            }

    report = benchmark_against_hrrp(
        trustedrisk_subgroup_rates=sub_rates or None,
    )
    (out_dir / "HRRP_BENCHMARK.md").write_text(
        render_hrrp_md(report), encoding="utf-8")
    (out_dir / "hrrp_benchmark.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8")
    print(
        f"Wrote {out_dir / 'HRRP_BENCHMARK.md'} "
        f"(max abs gap: {report.overall_max_abs_gap*100:.2f}%, "
        f"within CI: "
        f"{'yes' if report.overall_within_published_ci else 'no'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
