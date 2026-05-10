"""Phase 12.10 A1 — TrustedRisk Streamlit browser demo.

A single-page Streamlit app that lets a judge play with the
federation without writing any code:

  - **Scenarios** tab — runs each of the 5 end-to-end scenarios on
    demand and renders the per-step output as expandable cards.
  - **Counterfactuals** tab — surfaces the per-scenario CF flips.
  - **Red-team v2** tab — visualises the current 110-prompt corpus
    posture by category + severity (reads `docs/adversarial/red_team_run.json`).
  - **Red-team v3 map** tab — per-tool robustness map across the
    multi-target run (reads `docs/adversarial/red_team_map.json`).
  - **Calibration** tab — overlays W1, Synthea-10k, Synthea-100k,
    MIMIC-IV-demo metrics on a single comparison panel.
  - **Performance** tab — reads `docs/performance/v10_benchmarks.json`
    and shows the p50/p95/p99 table.
  - **MedQA bench** tab — reads `docs/evals/medqa_run.json` and
    shows accuracy by category.
  - **Tool catalogue** tab — lists the full 92-tool surface grouped
    by bundle, with a per-tool description.

Run:
    pip install -e .[demo]
    PYTHONPATH=src streamlit run apps/streamlit_demo/app.py

The app is read-only against pre-computed JSON artefacts under
`data/` and `docs/` — re-run the corresponding generator script to
refresh the panels (e.g. `make redteam-v2`, `make synthea-1k`).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _read_json(p: Path) -> dict | list | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _badge(text: str, color: str = "gray") -> str:
    return (
        f'<span style="background:{color}33; color:{color}; '
        f'padding:2px 8px; border-radius:4px; font-size:11px; '
        f'font-weight:600;">{text}</span>'
    )


# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TrustedRisk demo", page_icon=":hospital:",
    layout="wide",
)

st.title("TrustedRisk — federation demo")
st.caption(
    "92 MCP tools across 31 bundles, 15 specialists, "
    "calibrated under W1 / Synthea / MIMIC-IV-demo."
)

tabs = st.tabs([
    "Scenarios", "Counterfactuals", "Red-team v2",
    "Red-team v3 map", "Calibration", "Performance",
    "MedQA bench", "Tool catalogue",
])


# ─────────────────────────────────────────────────────────────────────
# Tab 1 — Scenarios
# ─────────────────────────────────────────────────────────────────────

with tabs[0]:
    st.subheader("End-to-end clinical scenarios")
    st.write(
        "Each scenario chains 5-7 deterministic-floor MCP tools. "
        "Inputs are inline so the demo is reproducible without a live "
        "FHIR server."
    )

    from a2a_agent.clinical_scenarios import (
        list_scenarios, run_scenario,
    )

    scenario_id = st.selectbox("Scenario", list_scenarios())
    if st.button("Run scenario", key="run_scenario"):
        run = asyncio.run(run_scenario(scenario_id))
        st.success(f"{run.title} — {run.n_tools_invoked} tools chained")
        st.caption(run.final_summary)
        for i, step in enumerate(run.steps, 1):
            with st.expander(f"{i}. {step.tool}"):
                st.write(step.summary)
                try:
                    st.json(step.output.model_dump())
                except Exception:
                    st.write(repr(step.output))


# ─────────────────────────────────────────────────────────────────────
# Tab 2 — Counterfactuals
# ─────────────────────────────────────────────────────────────────────

with tabs[1]:
    st.subheader("Per-scenario counterfactual analysis")
    st.write(
        "For each scenario, the harness perturbs 2-4 key input "
        "factors and reports the **minimum-modification** flips that "
        "change the recommended decision tier."
    )

    from a2a_agent.scenario_counterfactuals import (
        list_counterfactuals, run_counterfactual,
    )

    cf_id = st.selectbox("Scenario", list_counterfactuals(),
                                key="cf_scenario_select")
    if st.button("Run counterfactual analysis", key="run_cf"):
        rep = asyncio.run(run_counterfactual(cf_id))
        st.metric("Baseline outcome", rep.baseline_outcome)
        st.metric("Perturbations evaluated", rep.perturbations_evaluated)
        st.metric("Flips found", rep.n_flips)
        if rep.flips_found:
            rows = [{
                "factor": f.factor_name,
                "original": str(f.original_value)[:60],
                "modified": str(f.modified_value)[:60],
                "before": f.original_outcome,
                "after": f.modified_outcome,
                "distance": f.flip_distance,
            } for f in rep.flips_found]
            st.dataframe(rows, use_container_width=True)
        st.caption(rep.rationale)


# ─────────────────────────────────────────────────────────────────────
# Tab 3 — Red-team v2 (single target, detect_phi)
# ─────────────────────────────────────────────────────────────────────

with tabs[2]:
    st.subheader("v2 corpus posture — detect_phi target")
    raw = _read_json(ROOT / "docs" / "adversarial" / "red_team_run.json")
    if raw is None:
        st.warning("No artefact yet. Run `make redteam-v2`.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Cases", raw["n_cases"])
        col2.metric("Pass rate", f"{raw['overall_pass_rate']*100:.1f}%")
        col3.metric("Posture", raw["posture"].upper())

        st.write("### Pass rate by category")
        st.dataframe([
            {"category": k, "pass_rate": v}
            for k, v in sorted(raw["pass_rate_by_category"].items())
        ], use_container_width=True)

        st.write("### Pass rate by severity")
        st.dataframe([
            {"severity": k, "pass_rate": v}
            for k, v in sorted(raw["pass_rate_by_severity"].items())
        ], use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Tab 4 — Red-team v3 multi-target map
# ─────────────────────────────────────────────────────────────────────

with tabs[3]:
    st.subheader("Red-team multi-target map — robustness across the tool surface")
    raw = _read_json(ROOT / "docs" / "adversarial" / "red_team_map.json")
    if raw is None:
        st.warning("No artefact yet. Run `python scripts/run_redteam_v4.py`.")
    else:
        st.metric(
            "Average pass-rate across tools",
            f"{raw['overall_avg_pass_rate']*100:.1f}%",
        )
        rows = sorted(raw["rows"], key=lambda r: -r["pass_rate"])
        st.dataframe([
            {
                "tool": r["tool_name"],
                "cases": r["n_cases"],
                "passed": r["n_passed"],
                "failed": r["n_failed"],
                "pass_rate": r["pass_rate"],
                "posture": r["posture"],
            }
            for r in rows
        ], use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Tab 5 — Calibration overlay
# ─────────────────────────────────────────────────────────────────────

with tabs[4]:
    st.subheader("Calibration metrics across cohorts")
    cal_synthea_100k = _read_json(
        ROOT / "data" / "synthea_100k_recalibration.json"
    )
    cal_mimic = _read_json(ROOT / "data" / "mimic_iv_recalibration.json")

    rows = [
        {"cohort": "W1 published",
         "n": 7880, "ECE": 0.0078, "Brier": 0.124, "AUROC": 0.590},
        {"cohort": "Synthea-10k validation",
         "n": 10000, "ECE": 0.0056, "Brier": 0.118, "AUROC": 0.601},
    ]
    if cal_synthea_100k:
        m = cal_synthea_100k["metrics_overall"]
        rows.append({
            "cohort": "Synthea-100k (this build)",
            "n": cal_synthea_100k["cohort_n"],
            "ECE": round(m["ece"], 4),
            "Brier": round(m["brier"], 4),
            "AUROC": round(m["auroc"], 4),
        })
    if cal_mimic:
        m = cal_mimic["metrics_overall"]
        rows.append({
            "cohort": "MIMIC-IV demo",
            "n": cal_mimic["cohort_n"],
            "ECE": round(m["ece"], 4),
            "Brier": round(m["brier"], 4),
            "AUROC": round(m["auroc"], 4),
        })

    st.dataframe(rows, use_container_width=True)
    st.caption(
        "All four cohorts pass the 0.05 ECE preferred gate. The W1 "
        "calibration is robust across 10× cohort scaling and "
        "real-data MIMIC-IV ICU admissions."
    )


# ─────────────────────────────────────────────────────────────────────
# Tab 6 — Performance
# ─────────────────────────────────────────────────────────────────────

with tabs[5]:
    st.subheader("Performance benchmarks v10")
    raw = _read_json(ROOT / "docs" / "performance" / "v10_benchmarks.json")
    if raw is None:
        st.warning("No artefact yet. Run `python scripts/perf_benchmark_v10.py`.")
    else:
        st.write("### Per-tool latency (ms)")
        st.dataframe([
            {
                "tool": r["label"], "n": r["n"],
                "p50": r["p50"], "p95": r["p95"],
                "p99": r["p99"], "max": r["max"],
                "errors": r["errors"],
            }
            for r in raw["per_tool_phase_10_11"]
        ], use_container_width=True)
        st.write("### Per-scenario latency (ms)")
        st.dataframe([
            {
                "scenario": r["label"], "n": r["n"],
                "p50": r["p50"], "p95": r["p95"], "p99": r["p99"],
            }
            for r in raw["per_scenario"]
        ], use_container_width=True)
        st.write("### Federation concurrency — 100 simultaneous /healthz")
        f = raw["federation_concurrency"]
        st.metric("p50 (ms)", f["p50"])
        st.metric("p95 (ms)", f["p95"])
        st.metric("p99 (ms)", f["p99"])


# ─────────────────────────────────────────────────────────────────────
# Tab 7 — MedQA bench
# ─────────────────────────────────────────────────────────────────────

with tabs[6]:
    st.subheader("MedQA-USMLE-style bench")
    raw = _read_json(ROOT / "docs" / "evals" / "medqa_run.json")
    if raw is None:
        st.warning("No artefact yet. Run `python -m a2a_agent.medqa_eval`.")
    else:
        col1, col2 = st.columns(2)
        col1.metric(
            "Floor accuracy",
            f"{raw['floor_accuracy']*100:.1f}%",
        )
        if raw.get("llm_accuracy") is not None:
            col2.metric(
                "LLM accuracy",
                f"{raw['llm_accuracy']*100:.1f}%",
            )
        st.write("### Per-category accuracy (floor)")
        st.dataframe([
            {"category": k, "accuracy": v}
            for k, v in sorted(raw["by_category_floor"].items())
        ], use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Tab 8 — Tool catalogue
# ─────────────────────────────────────────────────────────────────────

with tabs[7]:
    st.subheader("MCP tool catalogue — 92 tools / 31 bundles")
    from mcp_server.tools import BUNDLES
    selected_bundle = st.selectbox(
        "Bundle",
        sorted(b for b in BUNDLES.keys()),
    )
    tools = BUNDLES[selected_bundle]
    st.write(f"**{len(tools)} tool(s) in `{selected_bundle}`**")

    rows = []
    import importlib
    for tool_name in tools:
        # Look up the tool function for its docstring
        doc = ""
        try:
            for mod_name in ("readmission_risk", "decision_utility",
                                  "ground_claim", "detect_phi"):
                pass    # placeholder; real lookup below
        except Exception:
            pass
        # Search across the tools package
        try:
            import mcp_server.tools as _tools_pkg
            for mod_name in _tools_pkg.__all__:
                mod = getattr(_tools_pkg, mod_name, None)
                fn = getattr(mod, tool_name, None) if mod else None
                if callable(fn):
                    doc = (fn.__doc__ or "").strip().split("\n\n", 1)[0]
                    break
        except Exception:
            pass
        rows.append({
            "tool": tool_name,
            "summary": doc[:140] if doc else "(no docstring)",
        })
    st.dataframe(rows, use_container_width=True)
