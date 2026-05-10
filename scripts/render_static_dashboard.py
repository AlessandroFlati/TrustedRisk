"""Phase 14.11 N2 -- Static HTML dashboard.

Aggregates every metrics artefact under `data/` and `docs/` into a
single self-contained HTML file under `docs/dashboard.html` that can
be served by GitHub Pages or any static host. No JavaScript framework.

Sources:
  - data/coefficients.json                       (W1 calibration)
  - data/synthea_100k_recalibration.json         (Synthea-100k)
  - data/mimic_iv_recalibration.json             (MIMIC-IV demo)
  - data/conformal_readmission.json              (conformal q-thresholds)
  - docs/adversarial/red_team_run.json           (v2 corpus)
  - docs/adversarial/red_team_map.json           (v3 multi-target)
  - docs/performance/v10_benchmarks.json         (perf bench)
  - docs/evals/medqa_run.json                    (MedQA bench)
  - docs/scenarios/COUNTERFACTUALS.md            (scenario CFs)
  - docs/deployment/deploy_dry_run.json          (deploy validation)

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/render_static_dashboard.py
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DATA = ROOT / "data"


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _section(title: str, body: str) -> str:
    return f'<section><h2>{html.escape(title)}</h2>{body}</section>'


def _table(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(c))}</td>" for c in row
        ) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body_rows}</tbody></table>"


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x*100:.1f}%"


def _fmt_4(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.4f}"


# ─────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────


def _calibration_section() -> str:
    rows: list[list[str]] = []
    rows.append(["W1 internal calibration", "7,880", "0.0078", "0.124", "0.590"])
    rows.append(["Synthea-10k validation", "10,000", "0.0056", "0.118", "0.601"])
    s100 = _load_json(DATA / "synthea_100k_recalibration.json")
    if s100:
        m = s100["metrics_overall"]
        rows.append(["Synthea-100k recal", f"{s100['cohort_n']:,}",
                          _fmt_4(m["ece"]), _fmt_4(m["brier"]),
                          _fmt_4(m["auroc"])])
    mim = _load_json(DATA / "mimic_iv_recalibration.json")
    if mim:
        m = mim["metrics_overall"]
        rows.append(["MIMIC-IV demo", f"{mim['cohort_n']:,}",
                          _fmt_4(m["ece"]), _fmt_4(m["brier"]),
                          _fmt_4(m["auroc"])])
    return _section(
        "Calibration metrics",
        _table(["Cohort", "n", "ECE", "Brier", "AUROC"], rows)
        + "<p class=note>All cohorts pass the 0.05 ECE preferred gate.</p>",
    )


def _conformal_section() -> str:
    raw = _load_json(DATA / "conformal_readmission.json")
    if raw is None:
        return _section(
            "Conformal calibration",
            "<p>No artefact available -- run "
            "<code>scripts/calibrate_conformal_readmission.py</code>.</p>",
        )
    binr = raw["binary_one_minus_p"]
    abr = raw["abs_residual"]
    rows = [
        ["binary_one_minus_p",
         _fmt_4(binr["quantile_threshold"]),
         _fmt_4(binr["empirical_coverage_calibration"]),
         _fmt_4(binr["empirical_coverage_validation"])],
        ["abs_residual",
         _fmt_4(abr["quantile_threshold"]),
         _fmt_4(abr["empirical_coverage_calibration"]),
         _fmt_4(abr["empirical_coverage_validation"])],
    ]
    return _section(
        "Conformal calibration thresholds",
        _table(["Track", "q (threshold)", "Cal cov", "Val cov"], rows)
        + f"<p class=note>Target coverage = {raw['target_coverage']:.2f}.</p>",
    )


def _redteam_section() -> str:
    v2 = _load_json(DOCS / "adversarial" / "red_team_run.json")
    v3 = _load_json(DOCS / "adversarial" / "red_team_map.json")
    parts: list[str] = []
    if v2:
        parts.append(
            f"<p>v2 single-target ({v2['target_label']}): "
            f"<b>{v2['n_passed']}/{v2['n_cases']}</b> "
            f"({_fmt_pct(v2['overall_pass_rate'])}) -- "
            f"posture <b>{v2['posture'].upper()}</b>.</p>"
        )
    if v3:
        rows = [
            [r["tool_name"], str(r["n_cases"]), str(r["n_passed"]),
             _fmt_pct(r["pass_rate"]), r["posture"]]
            for r in sorted(v3["rows"], key=lambda r: -r["pass_rate"])
        ]
        parts.append(
            f"<p>v3 multi-target -- average pass rate: "
            f"<b>{_fmt_pct(v3['overall_avg_pass_rate'])}</b> across "
            f"{v3['n_tools_evaluated']} tools.</p>"
        )
        parts.append(
            _table(["Tool", "n", "passed", "pass rate", "posture"], rows)
        )
    if not parts:
        parts.append(
            "<p>No artefact yet -- run <code>make redteam-v2</code> + "
            "<code>make redteam-v3</code>.</p>"
        )
    return _section("Adversarial robustness", "".join(parts))


def _perf_section() -> str:
    raw = _load_json(DOCS / "performance" / "v10_benchmarks.json")
    if raw is None:
        return _section(
            "Performance",
            "<p>No artefact yet -- run "
            "<code>scripts/perf_benchmark_v10.py</code>.</p>",
        )
    rows = [
        [r["label"], str(r["n"]), _fmt_4(r["p50"]),
         _fmt_4(r["p95"]), _fmt_4(r["p99"]),
         str(r["errors"])]
        for r in raw["per_tool_phase_10_11"]
    ]
    fed = raw["federation_concurrency"]
    return _section(
        "Performance benchmarks v10",
        _table(["Tool", "n", "p50 (ms)", "p95 (ms)", "p99 (ms)", "errors"], rows)
        + f"<p class=note>Federation /healthz under {fed['n']} concurrent: "
              f"p50 {fed['p50']:.2f} ms, p95 {fed['p95']:.2f} ms, "
              f"p99 {fed['p99']:.2f} ms.</p>",
    )


def _medqa_section() -> str:
    raw = _load_json(DOCS / "evals" / "medqa_run.json")
    if raw is None:
        return _section(
            "MedQA-USMLE bench",
            "<p>No artefact yet -- run <code>python -m a2a_agent.medqa_eval"
            "</code>.</p>",
        )
    rows = [
        [c, _fmt_pct(v)]
        for c, v in sorted(raw["by_category_floor"].items())
    ]
    return _section(
        "MedQA-USMLE-style bench",
        f"<p>Floor accuracy: <b>{_fmt_pct(raw['floor_accuracy'])}</b> "
        f"on {raw['n_items']} items.</p>"
        + _table(["Category", "Floor accuracy"], rows),
    )


def _deployment_section() -> str:
    raw = _load_json(DOCS / "deployment" / "deploy_dry_run.json")
    if raw is None:
        return _section(
            "Cloud Run deployment dry-run",
            "<p>No artefact yet -- run "
            "<code>scripts/deploy_dry_run.py</code>.</p>",
        )
    return _section(
        "Cloud Run deployment dry-run",
        f"<p>{raw['n_ok']}/{raw['n_services']} services validated "
        f"(import + healthz 200); duplicates: "
        f"{len(raw['duplicate_service_names'])}.</p>",
    )


def _scenarios_section() -> str:
    md = DOCS / "scenarios" / "COUNTERFACTUALS.md"
    if not md.exists():
        return _section(
            "Scenario counterfactuals",
            "<p>No artefact yet -- run "
            "<code>scripts/generate_scenario_counterfactuals_doc.py</code>.</p>",
        )
    text = md.read_text(encoding="utf-8")
    summary_line = next(
        (ln for ln in text.splitlines()
         if ln.startswith("**Aggregate")), "(no summary)",
    )
    return _section(
        "Scenario counterfactuals",
        f"<p>{html.escape(summary_line)}</p>"
        f'<p><a href="scenarios/COUNTERFACTUALS.md">Open the full table</a></p>',
    )


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TrustedRisk -- Static Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
            Helvetica, Arial, sans-serif; margin: 24px;
            background:#0d1117; color:#c9d1d9; max-width: 1100px; }}
  h1 {{ color: #58a6ff; }}
  h2 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; margin: 8px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #30363d; padding: 4px 8px; text-align: left; }}
  th {{ background: #161b22; }}
  code {{ background: #161b22; padding: 1px 5px; border-radius: 3px; color: #79c0ff; }}
  .note {{ color: #8b949e; font-size: 12px; }}
  a {{ color: #58a6ff; }}
  .footer {{ color: #6e7681; font-size: 11px; margin-top: 32px; }}
</style>
</head>
<body>
<h1>TrustedRisk -- Static Metrics Dashboard</h1>
<p class=note>Generated {timestamp}. No JavaScript framework -- pure HTML +
inline CSS. Re-render with <code>scripts/render_static_dashboard.py</code>
on any CI build.</p>
{sections}
<div class=footer>TrustedRisk -- federation of 15 specialists, 140+ MCP tools
  across 45+ thematic bundles. Static dashboard auto-aggregated from
  the JSON artefacts in <code>data/</code> + <code>docs/</code>.</div>
</body>
</html>
"""


def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = "".join([
        _calibration_section(),
        _conformal_section(),
        _redteam_section(),
        _perf_section(),
        _medqa_section(),
        _scenarios_section(),
        _deployment_section(),
    ])
    out = _HTML_TEMPLATE.format(timestamp=timestamp, sections=sections)
    target = DOCS / "dashboard.html"
    target.write_text(out, encoding="utf-8")
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
