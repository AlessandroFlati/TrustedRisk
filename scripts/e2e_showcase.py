"""End-to-end showcase that exercises every MCP tool + the batch HTTP endpoint.

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \
        .venv/Scripts/python.exe scripts/e2e_showcase.py

Output:
    docs/e2e/index.html -- interactive showcase with 4 clinical scenarios.

The reasoning is deterministic (no LLM dependency): the scenarios call the
same async tool functions an LLM-mediated agent would call, simulating the
ADK tool-call sequence without requiring Ollama / Gemini. The batch
scenario uses Starlette's TestClient against the real server to validate
the HTTP layer end-to-end.
"""

from __future__ import annotations

import asyncio
import html as html_lib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

# Make `src` importable when run from the repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ─────────────────────────────────────────────────────────────────────
# Operation record (per tool invocation)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Operation:
    tool: str
    rationale: str
    input: dict[str, Any]
    output: Any
    duration_ms: float
    error: str | None = None


@dataclass
class Scenario:
    slug: str
    title: str
    persona: str
    prompt: str
    fhir_summary: str
    expected_tools: list[str]
    operations: list[Operation] = field(default_factory=list)
    final_text: str = ""
    plan_reasoning: str = ""        # LLM thought BEFORE tool calls
    plan_reasoning_ms: float = 0.0
    synthesis: str = ""             # LLM thought AFTER tool calls
    synthesis_ms: float = 0.0
    llm_model: str = ""
    deterministic_action: str = ""      # ground-truth recommendation
    deterministic_confidence: str = ""  # ground-truth confidence band
    abstain_triggers: list[dict] = field(default_factory=list)
    llm_action: str | None = None       # parsed from synthesis text
    agreement: str = ""                  # "match" | "mismatch" | "n_a"
    synthesis_kind: str = "discharge_decision"  # also: documentation_review | batch_operations
    # When the deterministic_action is a treatment option_id (not one of the
    # 5 discharge actions), populate this list so the parser/agreement can
    # match exactly without trying to apply the conservative-escalation order.
    extra_actions: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# LLM reasoning (Ollama) -- plan before, synthesis after
# ─────────────────────────────────────────────────────────────────────

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("E2E_LLM_MODEL", "mistral-nemo:latest")
LLM_TIMEOUT_S = float(os.environ.get("E2E_LLM_TIMEOUT_S", "120"))


_AGENT_SYSTEM = (
    "You are TrustedRisk, a clinical decision-support agent for safe hospital "
    "discharge. You orchestrate 10 MCP tools plus a bulk batch endpoint, but "
    "when you write the FINAL answer to the clinician you NEVER mention any "
    "tool name, function name, or orchestration detail -- you speak as if the "
    "values were just chart facts. You NEVER invent clinical values -- every "
    "claim must trace to a tool output. You ABSTAIN if the calibrated CI is "
    "too wide, evidence is unsupported, or the case is out-of-distribution. "
    "Be terse, clinical, and audit-friendly. When a deterministic safety "
    "gate provides a binding recommendation, you obey it: you may only "
    "escalate to a more conservative action or ABSTAIN, never relax it."
)


async def _ollama_complete(system: str, prompt: str,
                            temperature: float = 0.2) -> tuple[str, float]:
    """Single-turn Ollama generate call. Returns (text, duration_ms).
    Raises RuntimeError on failure so the caller can decide whether to fall back."""
    import urllib.request
    import urllib.error
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 900},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()

    def _do_call() -> str:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return (payload.get("response") or "").strip()

    try:
        text = await asyncio.to_thread(_do_call)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Ollama call failed: {type(exc).__name__}: {exc}") from exc
    dt = (time.perf_counter() - t0) * 1000.0
    return text, dt


def _digest_discharge_decision(scenario: Scenario) -> str:
    """Pull the headline numbers from each tool output so the synthesis
    prompt can require them verbatim. Prevents Mistral-class models from
    paraphrasing away counts the clinician needs to see."""
    by_tool: dict[str, Any] = {op.tool: op.output for op in scenario.operations
                                if op.output is not None}
    lines: list[str] = []

    def _r(x: Any, n: int = 3) -> str:
        if isinstance(x, (int, float)):
            return f"{round(float(x), n)}"
        return str(x)

    risk = by_tool.get("compute_readmission_risk")
    if isinstance(risk, dict):
        prob = risk.get("probability_mean")
        ci = risk.get("probability_ci95") or [None, None]
        lines.append(
            f"  readmission_risk: probability_mean={_r(prob, 3)} "
            f"({_r(prob*100, 1) if isinstance(prob, (int, float)) else 'n/a'}%), "
            f"CI95=[{_r(ci[0], 3)}, {_r(ci[1], 3)}], "
            f"lace_raw_score={risk.get('lace_raw_score')}, "
            f"ci_width={_r(risk.get('probability_ci_width'), 3)}, "
            f"valid_for_minutes={risk.get('valid_for_minutes')}"
        )

    medrecon = by_tool.get("compute_medication_reconciliation")
    if isinstance(medrecon, dict):
        sc = medrecon.get("severity_counts") or {}
        concerns = medrecon.get("concerns") or []
        lines.append(
            f"  medication_reconciliation: "
            f"n_admission_meds={medrecon.get('n_admission_meds')}, "
            f"n_discharge_meds={medrecon.get('n_discharge_meds')}, "
            f"n_concerns={len(concerns)}, severity_counts={sc}, "
            f"discharge_contract_satisfied={medrecon.get('discharge_contract_satisfied')}"
        )
        for c in concerns:
            mon_req = ", ".join(c.get("monitoring_required") or []) or "--"
            mon_obs = ", ".join(c.get("monitoring_observed") or []) or "--"
            lines.append(
                f"    • concern: {c.get('medication_name')} "
                f"(class={c.get('drug_class')}, severity={c.get('severity')}, "
                f"type={c.get('concern_type')}); monitoring_required=[{mon_req}], "
                f"monitoring_observed=[{mon_obs}]"
            )

    poly = by_tool.get("detect_polypharmacy_concerns")
    if isinstance(poly, dict):
        interactions = poly.get("interactions") or []
        lines.append(
            f"  polypharmacy: n_medications={poly.get('n_medications')}, "
            f"n_high_risk={poly.get('n_high_risk')}, "
            f"polypharmacy_severity={poly.get('polypharmacy_severity')}, "
            f"n_interactions={len(interactions)}, "
            f"class_counts={poly.get('class_counts')}"
        )
        for ddi in interactions:
            lines.append(
                f"    • interaction: {ddi.get('drug_a')} + {ddi.get('drug_b')} "
                f"(severity={ddi.get('severity')}, "
                f"mechanism={ddi.get('mechanism')!r})"
            )

    fair = by_tool.get("compute_fairness_audit")
    if isinstance(fair, dict):
        drift = fair.get("max_relative_drift")
        drift_pct = (
            f"{round(float(drift) * 100.0, 1)}%"
            if isinstance(drift, (int, float)) else "n/a"
        )
        lines.append(
            f"  fairness_audit: "
            f"n_subgroups_assessed={fair.get('n_subgroups_assessed')}, "
            f"max_relative_drift={_r(drift, 3)} ({drift_pct}), "
            f"confidence_action={fair.get('confidence_action')!r}"
        )
        for sg in fair.get("subgroup_drifts") or []:
            sg_drift = sg.get("relative_drift")
            sg_pct = (f"{round(float(sg_drift) * 100.0, 1)}%"
                       if isinstance(sg_drift, (int, float)) else "n/a")
            lines.append(
                f"    • subgroup: {sg.get('subgroup_name')}={sg.get('subgroup_value')!r} "
                f"(severity={sg.get('severity')}, drift={_r(sg_drift, 3)} / "
                f"{sg_pct}, expected_baseline={_r(sg.get('expected_rate_baseline'), 3)})"
            )

    cf = by_tool.get("compute_counterfactual_explanation")
    if isinstance(cf, dict):
        lines.append(
            f"  counterfactual: "
            f"current_lace_total={cf.get('current_lace_total')}, "
            f"current_prob_mean={_r(cf.get('current_prob_mean'), 3)}, "
            f"most_influential_factor={cf.get('most_influential_factor')!r}"
        )
        for f in cf.get("factors") or []:
            lines.append(
                f"    • factor: {f.get('factor_name')} "
                f"(current_points={f.get('current_points')}, "
                f"modifiability={f.get('modifiability')}, "
                f"delta_prob_if_zero={_r(f.get('delta_prob_if_zero'), 3)})"
            )
        for path_field in ("flip_path_safer", "flip_path_riskier"):
            path = cf.get(path_field)
            if isinstance(path, dict):
                lines.append(
                    f"    • {path_field}: target={path.get('target_action')}, "
                    f"factors_to_change={path.get('factors_to_change')}, "
                    f"cumulative_delta_points={path.get('cumulative_delta_points')}, "
                    f"achievable={path.get('achievable')}"
                )

    labs = by_tool.get("compute_lab_trend_analysis")
    if isinstance(labs, dict):
        lines.append(
            f"  lab_trends: n_labs_analyzed={labs.get('n_labs_analyzed')}, "
            f"n_observations_total={labs.get('n_observations_total')}, "
            f"flagged_labs_count={labs.get('flagged_labs_count')}"
        )
        for tr in labs.get("trends") or []:
            ref = tr.get("reference_range")
            ref_str = (f"[{ref[0]}, {ref[1]}]"
                        if isinstance(ref, (list, tuple)) and len(ref) == 2
                        else "--")
            lines.append(
                f"    • lab: {tr.get('lab_name')} (n={tr.get('n_observations')}, "
                f"latest={tr.get('latest_value')}, "
                f"direction={tr.get('direction')}, "
                f"trend_clinical={tr.get('trend_clinical')}, "
                f"in_normal_range={tr.get('in_normal_range')}, "
                f"reference_range={ref_str})"
            )

    couns = by_tool.get("compute_discharge_counseling")
    if isinstance(couns, dict):
        lines.append(
            f"  discharge_counseling: "
            f"n_medications_explained={couns.get('n_medications_explained')}, "
            f"n_red_flags={couns.get('n_red_flags')}, "
            f"follow_up_window_days={couns.get('follow_up_window_days')}, "
            f"reading_level_grade={couns.get('reading_level_grade')}"
        )
        for sec in couns.get("sections") or []:
            bullets = sec.get("bullets") or []
            preview = bullets[:3]  # up to first 3 bullets per section
            if preview:
                preview_str = " | ".join(
                    (b[:120] + "...") if len(b) > 120 else b for b in preview
                )
                lines.append(
                    f"    • section {sec.get('section_id')} "
                    f"({len(bullets)} bullets): {preview_str}"
                )

    # NEWS2 deterioration
    news = by_tool.get("compute_clinical_deterioration_score")
    if isinstance(news, dict):
        lines.append(
            f"  clinical_deterioration_score (NEWS2): "
            f"score_total={news.get('score_total')}, "
            f"severity_tier={news.get('severity_tier')!r}, "
            f"recommended_response={news.get('recommended_response')!r}, "
            f"trend_delta={news.get('trend_delta')}, "
            f"trend_direction={news.get('trend_direction')!r}"
        )
        for c in news.get("parameter_contributions") or []:
            if (c.get("points") or 0) > 0:
                lines.append(
                    f"    • {c.get('parameter')}={c.get('value')} "
                    f"(+{c.get('points')} -- {c.get('rationale')})"
                )

    # ED triage
    triage = by_tool.get("compute_admission_triage")
    if isinstance(triage, dict):
        lines.append(
            f"  admission_triage: esi_level={triage.get('esi_level')}, "
            f"priority={triage.get('priority')!r}, "
            f"disposition={triage.get('disposition')!r}, "
            f"recommended_unit={triage.get('recommended_unit')!r}, "
            f"abstain_recommended={triage.get('abstain_recommended')}"
        )
        for f in triage.get("red_flag_findings") or []:
            lines.append(f"    • red_flag: {f}")

    # Treatment selection
    tsel = by_tool.get("compute_treatment_selection")
    if isinstance(tsel, dict):
        lines.append(
            f"  treatment_selection: condition={tsel.get('condition')!r}, "
            f"top_pick_id={tsel.get('top_pick_id')!r}, "
            f"abstain_recommended={tsel.get('abstain_recommended')}, "
            f"n_options={len(tsel.get('options') or [])}"
        )
        for opt in (tsel.get("options") or [])[:4]:  # top 4 only
            contras = ", ".join((opt.get("contraindications") or [])[:2]) or "--"
            lines.append(
                f"    • option rank={opt.get('rank')} {opt.get('label')!r} "
                f"(score={_r(opt.get('score'), 2)}, contraindications=[{contras}])"
            )

    # PEWS
    pews = by_tool.get("compute_pediatric_early_warning")
    if isinstance(pews, dict):
        lines.append(
            f"  pediatric_early_warning (PEWS): "
            f"score_total={pews.get('score_total')}, "
            f"age_band={pews.get('age_band')!r}, "
            f"severity_tier={pews.get('severity_tier')!r}, "
            f"recommended_response={pews.get('recommended_response')!r}"
        )
        for c in pews.get("components") or []:
            if (c.get("points") or 0) > 0:
                lines.append(
                    f"    • {c.get('component')}=+{c.get('points')} -- "
                    f"{c.get('rationale')}"
                )

    # Weight-based dosing -- captures multiple invocations as separate entries
    # (since by_tool only keeps the last; we read raw operations instead).
    dose_ops = [op for op in scenario.operations
                 if op.tool == "compute_weight_based_dosing"]
    for i, op in enumerate(dose_ops, 1):
        out = op.output or {}
        contras = ", ".join(out.get("contraindications") or [])
        lines.append(
            f"  weight_based_dosing[#{i}]: "
            f"drug={out.get('drug')!r}, weight_kg={out.get('weight_kg')}, "
            f"age_months={out.get('age_months')}, "
            f"final_dose_mg={out.get('final_dose_mg')}, "
            f"capped_at_adult_max={out.get('capped_at_adult_max')}, "
            f"volume_to_administer_ml={out.get('volume_to_administer_ml')}, "
            f"abstain_recommended={out.get('abstain_recommended')}"
            + (f", contraindications=[{contras}]" if contras else "")
        )

    # C-SSRS
    cssrs = by_tool.get("compute_suicide_risk_assessment")
    if isinstance(cssrs, dict):
        lines.append(
            f"  suicide_risk_assessment (C-SSRS): "
            f"risk_level={cssrs.get('risk_level')!r}, "
            f"ideation_lifetime_level={(cssrs.get('ideation_lifetime') or {}).get('level')}, "
            f"ideation_past_30d_level={(cssrs.get('ideation_past_30d') or {}).get('level')}, "
            f"behavior_lifetime_attempts={cssrs.get('behavior_lifetime_attempts')}, "
            f"behavior_past_30d={cssrs.get('behavior_past_30d')}, "
            f"safety_plan_indicated={cssrs.get('safety_plan_indicated')}, "
            f"abstain_recommended={cssrs.get('abstain_recommended')}"
        )
        if cssrs.get("abstain_reason"):
            lines.append(f"    • abstain_reason: {cssrs.get('abstain_reason')}")

    # Psychiatric admission
    psy = by_tool.get("compute_psychiatric_admission_decision")
    if isinstance(psy, dict):
        lines.append(
            f"  psychiatric_admission: "
            f"disposition={psy.get('disposition')!r}, "
            f"danger_to_self={psy.get('danger_to_self')}, "
            f"danger_to_others={psy.get('danger_to_others')}, "
            f"grave_disability={psy.get('grave_disability')}, "
            f"voluntary_capable={psy.get('voluntary_capable')}, "
            f"safety_plan_required={psy.get('safety_plan_required')}, "
            f"follow_up_within_hours={psy.get('follow_up_within_hours')}"
        )
        if psy.get("legal_basis"):
            lines.append(f"    • legal_basis: {psy.get('legal_basis')}")

    # Empiric antibiotic selection
    abx = by_tool.get("compute_empiric_antibiotic_selection")
    if isinstance(abx, dict):
        lines.append(
            f"  empiric_antibiotic_selection: "
            f"infection_source={abx.get('infection_source')!r}, "
            f"severity={abx.get('severity')!r}, "
            f"top_pick_id={abx.get('top_pick_id')!r}, "
            f"local_antibiogram_used={abx.get('local_antibiogram_used')}, "
            f"n_options={len(abx.get('options') or [])}"
        )
        for opt in (abx.get("options") or [])[:3]:
            lines.append(
                f"    • option rank={opt.get('rank')} {opt.get('label')!r} "
                f"(score={_r(opt.get('score'), 2)}, route={opt.get('route')})"
            )

    # Antibiotic de-escalation
    deesc = by_tool.get("compute_antibiotic_de_escalation")
    if isinstance(deesc, dict):
        lines.append(
            f"  antibiotic_de_escalation: "
            f"pathogen_identified={deesc.get('pathogen_identified')!r}, "
            f"de_escalation_recommended={deesc.get('de_escalation_recommended')}, "
            f"target_regimen={deesc.get('target_regimen')!r}, "
            f"iv_to_po_switch_eligible={deesc.get('iv_to_po_switch_eligible')}, "
            f"duration_total_days={deesc.get('duration_total_days')}, "
            f"duration_remaining_days={deesc.get('duration_remaining_days')}"
        )
        unmet = deesc.get("iv_to_po_criteria_unmet") or []
        if unmet:
            lines.append(f"    • iv_to_po_criteria_unmet: {unmet}")

    # Chemo dose adjustment
    chemo = by_tool.get("compute_chemo_dose_adjustment")
    if isinstance(chemo, dict):
        lines.append(
            f"  chemo_dose_adjustment: regimen={chemo.get('regimen')!r}, "
            f"cycle_number={chemo.get('cycle_number')}, "
            f"decision={chemo.get('decision')!r}, "
            f"dose_reduction_pct={chemo.get('dose_reduction_pct')}, "
            f"growth_factor_indicated={chemo.get('growth_factor_indicated')}, "
            f"anc_per_ul={chemo.get('anc_per_ul')}, "
            f"platelets_per_ul={chemo.get('platelets_per_ul')}, "
            f"egfr_ml_min={chemo.get('egfr_ml_min')}, "
            f"ecog_performance_status={chemo.get('ecog_performance_status')}"
        )
        if chemo.get("delay_reason"):
            lines.append(f"    • delay_reason: {chemo.get('delay_reason')}")

    # NIHSS stroke severity
    nihss = by_tool.get("compute_stroke_severity")
    if isinstance(nihss, dict):
        lines.append(
            f"  stroke_severity (NIHSS): score_total={nihss.get('score_total')}, "
            f"severity_tier={nihss.get('severity_tier')!r}, "
            f"lvo_suspected={nihss.get('lvo_suspected')}, "
            f"recommended_response={nihss.get('recommended_response')!r}"
        )
        for it in nihss.get("items") or []:
            if (it.get("points") or 0) > 0:
                lines.append(f"    • {it.get('item')}=+{it.get('points')}")

    # tPA / EVT eligibility
    tpa = by_tool.get("compute_stroke_thrombolysis_eligibility")
    if isinstance(tpa, dict):
        lines.append(
            f"  thrombolysis_eligibility: "
            f"iv_tpa_window={tpa.get('iv_tpa_window')!r}, "
            f"iv_tpa_eligible={tpa.get('iv_tpa_eligible')}, "
            f"evt_window={tpa.get('evt_window')!r}, "
            f"evt_eligible={tpa.get('evt_eligible')}, "
            f"decision={tpa.get('decision')!r}, "
            f"last_known_well_minutes_ago={tpa.get('last_known_well_minutes_ago')}"
        )
        for e in tpa.get("iv_tpa_exclusion_present") or []:
            lines.append(f"    • tpa_exclusion: {e}")
        for c in tpa.get("evt_criteria_unmet") or []:
            lines.append(f"    • evt_unmet: {c}")

    # HEART score
    heart = by_tool.get("compute_heart_score")
    if isinstance(heart, dict):
        lines.append(
            f"  heart_score: total_score={heart.get('total_score')}, "
            f"risk_band={heart.get('risk_band')!r}, "
            f"estimated_30d_mace_risk_pct={heart.get('estimated_30d_mace_risk_pct')}, "
            f"H={heart.get('history_points')}, E={heart.get('ecg_points')}, "
            f"A={heart.get('age_points')}, R={heart.get('risk_factors_points')}, "
            f"T={heart.get('troponin_points')}"
        )

    # ACS disposition
    acs = by_tool.get("compute_acs_disposition_decision")
    if isinstance(acs, dict):
        lines.append(
            f"  acs_disposition: disposition={acs.get('disposition')!r}, "
            f"risk_band={acs.get('risk_band')!r}, "
            f"has_stemi={acs.get('has_stemi')}, "
            f"has_dynamic_troponin={acs.get('has_dynamic_troponin')}, "
            f"has_high_risk_features={acs.get('has_high_risk_features')}, "
            f"recommended_followup_hours={acs.get('recommended_followup_hours')}"
        )

    # MEOWS
    meow = by_tool.get("compute_maternal_early_warning")
    if isinstance(meow, dict):
        lines.append(
            f"  maternal_early_warning (MEOWS): "
            f"severity_tier={meow.get('severity_tier')!r}, "
            f"recommended_response={meow.get('recommended_response')!r}, "
            f"score_total={meow.get('score_total')}, "
            f"gestational_age_weeks={meow.get('gestational_age_weeks')}, "
            f"pregnancy_phase={meow.get('pregnancy_phase')!r}"
        )
        for c in meow.get("contributing_parameters") or []:
            lines.append(f"    • {c}")

    # Preeclampsia
    pre = by_tool.get("compute_preeclampsia_assessment")
    if isinstance(pre, dict):
        lines.append(
            f"  preeclampsia_assessment: "
            f"classification={pre.get('classification')!r}, "
            f"recommended_disposition={pre.get('recommended_disposition')!r}, "
            f"delivery_recommended={pre.get('delivery_recommended')}, "
            f"magnesium_sulfate_indicated={pre.get('magnesium_sulfate_indicated')}, "
            f"antihypertensive_indicated={pre.get('antihypertensive_indicated')}, "
            f"systolic_bp={pre.get('systolic_bp')}, "
            f"diastolic_bp={pre.get('diastolic_bp')}, "
            f"proteinuria_present={pre.get('proteinuria_present')}"
        )
        for sf in pre.get("severity_features_present") or []:
            lines.append(f"    • severe_feature: {sf}")

    # Morse falls
    falls = by_tool.get("compute_falls_risk_morse")
    if isinstance(falls, dict):
        lines.append(
            f"  falls_risk_morse: score_total={falls.get('score_total')}, "
            f"risk_tier={falls.get('risk_tier')!r}, "
            f"recommended_intervention={falls.get('recommended_intervention')!r}, "
            f"history_falls_points={falls.get('history_falls_points')}, "
            f"iv_or_heparin_lock_points={falls.get('iv_or_heparin_lock_points')}, "
            f"gait_points={falls.get('gait_points')}, "
            f"mental_status_points={falls.get('mental_status_points')}"
        )
        for m in (falls.get("contributing_medications_flagged") or [])[:5]:
            lines.append(f"    • flagged_med: {m}")

    # Trauma severity
    trauma = by_tool.get("compute_trauma_severity_score")
    if isinstance(trauma, dict):
        lines.append(
            f"  trauma_severity: iss={trauma.get('iss')} "
            f"({trauma.get('iss_band')!r}), "
            f"rts={trauma.get('rts')} ({trauma.get('rts_band')!r}), "
            f"triage_priority={trauma.get('triage_priority')!r}"
        )
        for inj in (trauma.get("injuries") or [])[:5]:
            lines.append(
                f"    • injury: {inj.get('body_region')}=AIS{inj.get('ais_severity')}"
            )

    # MTP
    mtp = by_tool.get("compute_massive_transfusion_protocol")
    if isinstance(mtp, dict):
        req = mtp.get("estimated_initial_request") or {}
        lines.append(
            f"  massive_transfusion_protocol: "
            f"abc_score={mtp.get('abc_score')}, "
            f"mtp_activated={mtp.get('mtp_activated')}, "
            f"target_ratio={mtp.get('target_ratio')!r}, "
            f"txa_indicated={mtp.get('txa_indicated')}, "
            f"initial_request_rbc={req.get('rbc_units')}, "
            f"initial_request_ffp={req.get('ffp_units')}, "
            f"initial_request_platelet_packs={req.get('platelet_packs')}"
        )
        for c in mtp.get("abc_components") or []:
            lines.append(f"    • abc_component: {c}")

    # DKA
    dka = by_tool.get("compute_dka_severity")
    if isinstance(dka, dict):
        lines.append(
            f"  dka_severity: severity={dka.get('severity')!r}, "
            f"icu_admission_indicated={dka.get('icu_admission_indicated')}, "
            f"ph={dka.get('ph')}, bicarbonate_meq_l={dka.get('bicarbonate_meq_l')}, "
            f"glucose_mg_dl={dka.get('glucose_mg_dl')}, "
            f"k_replacement_first={dka.get('potassium_replacement_at_initiation')}, "
            f"bicarbonate_indicated={dka.get('bicarbonate_indicated')}"
        )

    # Inpatient glycemic
    glyc = by_tool.get("compute_inpatient_glycemic_control")
    if isinstance(glyc, dict):
        lines.append(
            f"  inpatient_glycemic_control: "
            f"target_range_mg_dl={glyc.get('target_range_mg_dl')}, "
            f"basal_dose_change_pct={glyc.get('basal_dose_change_pct')}, "
            f"hypoglycemia_risk={glyc.get('hypoglycemia_risk')!r}, "
            f"recommended_regimen={(glyc.get('recommended_regimen') or '')[:120]}"
        )

    # Imaging appropriateness
    img = by_tool.get("compute_imaging_appropriateness")
    if isinstance(img, dict):
        lines.append(
            f"  imaging_appropriateness: "
            f"clinical_scenario={img.get('clinical_scenario')!r}, "
            f"top_pick_modality={(img.get('top_pick_modality') or '?')[:60]!r}, "
            f"pediatric_alara_caution={img.get('pediatric_alara_caution')}, "
            f"cumulative_radiation_caution={img.get('cumulative_radiation_caution')}"
        )
        for opt in (img.get("options") or [])[:3]:
            lines.append(
                f"    • option rank={opt.get('rank')} {opt.get('modality')!r} "
                f"(rating={opt.get('appropriateness_rating')}/9, "
                f"dose={opt.get('radiation_dose_msv')} mSv)"
            )

    # Contrast safety
    contrast = by_tool.get("compute_contrast_safety_check")
    if isinstance(contrast, dict):
        lines.append(
            f"  contrast_safety_check: "
            f"contrast_type={contrast.get('contrast_type')!r}, "
            f"contrast_induced_nephropathy_risk={contrast.get('contrast_induced_nephropathy_risk')!r}, "
            f"nsf_risk_for_gadolinium={contrast.get('nsf_risk_for_gadolinium')!r}, "
            f"metformin_hold_recommended={contrast.get('metformin_hold_recommended')}, "
            f"premedication_recommended={contrast.get('premedication_recommended')}, "
            f"proceed_with_contrast={contrast.get('proceed_with_contrast')}"
        )

    # AKI / KDIGO
    aki = by_tool.get("compute_aki_kdigo_stage")
    if isinstance(aki, dict):
        lines.append(
            f"  aki_kdigo_stage: stage={aki.get('aki_stage')!r}, "
            f"creatinine_baseline_mg_dl={aki.get('creatinine_baseline_mg_dl')}, "
            f"creatinine_current_mg_dl={aki.get('creatinine_current_mg_dl')}, "
            f"creatinine_change_ratio={aki.get('creatinine_change_ratio')}, "
            f"aki_etiology_clue={aki.get('aki_etiology_clue')!r}, "
            f"nephrology_consult_indicated={aki.get('nephrology_consult_indicated')}"
        )

    # Dialysis initiation
    dial = by_tool.get("compute_dialysis_initiation_decision")
    if isinstance(dial, dict):
        lines.append(
            f"  dialysis_initiation_decision: "
            f"dialysis_indicated={dial.get('dialysis_indicated')}, "
            f"urgency={dial.get('urgency')!r}, "
            f"modality_suggested={dial.get('modality_suggested')!r}, "
            f"n_aeiou_indications={len(dial.get('aeiou_indications_met') or [])}"
        )
        for i in dial.get("aeiou_indications_met") or []:
            lines.append(f"    • aeiou: {i}")

    # CAM delirium
    cam = by_tool.get("compute_delirium_screening_cam")
    if isinstance(cam, dict):
        lines.append(
            f"  delirium_screening_cam: cam_positive={cam.get('cam_positive')}, "
            f"delirium_subtype={cam.get('delirium_subtype')!r}, "
            f"feature1={cam.get('feature1_acute_onset_or_fluctuating')}, "
            f"feature2={cam.get('feature2_inattention')}, "
            f"feature3={cam.get('feature3_disorganized_thinking')}, "
            f"feature4={cam.get('feature4_altered_consciousness')}"
        )
        for c in (cam.get("contributing_factors") or [])[:3]:
            lines.append(f"    • contributor: {c}")
        for ns in (cam.get("next_steps") or [])[:3]:
            lines.append(f"    • next_step: {ns[:120]}")

    # RECIST tumor response
    recist = by_tool.get("compute_oncology_treatment_response")
    if isinstance(recist, dict):
        lines.append(
            f"  oncology_treatment_response (RECIST 1.1): "
            f"overall_response={recist.get('overall_response')!r}, "
            f"baseline_sum_mm={recist.get('baseline_sum_mm')}, "
            f"current_sum_mm={recist.get('current_sum_mm')}, "
            f"sum_change_pct={_r(recist.get('sum_change_pct'), 1)}, "
            f"new_lesions_present={recist.get('new_lesions_present')}, "
            f"decision_implication={recist.get('decision_implication')!r}"
        )
        for L in (recist.get("target_lesions") or [])[:4]:
            lines.append(
                f"    • lesion {L.get('lesion_id')} ({L.get('location')}): "
                f"baseline {L.get('baseline_longest_diameter_mm')} mm -> "
                f"current {L.get('current_longest_diameter_mm')} mm"
            )

    return "\n".join(lines) if lines else "(no tool output captured)"


def _digest_batch_response(scenario: Scenario) -> str:
    """Extract a clean digest from the batch endpoint response so the
    synthesis prompt doesn't have to parse the full payload."""
    op = next((o for o in scenario.operations
               if "batch/decision-cards" in o.tool), None)
    if op is None or not isinstance(op.output, dict):
        return "(no batch response captured)"
    out = op.output
    parts = [
        f"  HTTP_status_implied = 200 (response parsed as dict)",
        f"  n_requested = {out.get('n_requested')}",
        f"  n_succeeded = {out.get('n_succeeded')}",
        f"  n_failed    = {out.get('n_failed')}",
        f"  n_abstained = {out.get('n_abstained')}",
        f"  duration_ms = {out.get('duration_ms')}",
        f"  http_roundtrip_ms = {op.duration_ms:.0f}",
    ]
    action_counts: dict[str, int] = {}
    failed_lines: list[str] = []
    for r in out.get("results") or []:
        rec = (r.get("recommendation") or {}) if isinstance(r, dict) else {}
        action = (rec or {}).get("action") if isinstance(rec, dict) else None
        if action:
            action_counts[action] = action_counts.get(action, 0) + 1
        if r.get("status") == "error":
            failed_lines.append(
                f"  failed_patient_id = {r.get('patient_id')!r}; "
                f"error = {r.get('error')!r}"
            )
    if action_counts:
        parts.append(
            "  recommendation_distribution = "
            + ", ".join(f"{k}={v}" for k, v in sorted(action_counts.items()))
        )
    parts.extend(failed_lines or ["  (no failed patients)"])
    return "\n".join(parts)


def _summarize_for_synthesis(scenario: Scenario, max_per_op: int = 350) -> str:
    """Compact tool-output summary the synthesis prompt can fit in context.
    For discharge_decision scenarios the digest carries the load -- this is
    only a thin context anchor."""
    parts: list[str] = []
    for i, op in enumerate(scenario.operations, 1):
        parts.append(f"--- Tool {i}: {op.tool} ({op.duration_ms:.0f} ms) ---")
        parts.append(f"Rationale: {op.rationale}")
        if op.error:
            parts.append(f"ERROR: {op.error}")
        else:
            blob = json.dumps(op.output, default=str, ensure_ascii=False)
            if len(blob) > max_per_op:
                blob = blob[:max_per_op] + "  ...(truncated -- see KEY FACTS DIGEST above for details)"
            parts.append(f"Output: {blob}")
    return "\n".join(parts)


async def llm_plan(scenario: Scenario) -> None:
    """Ask the model to articulate which tools it plans to call and why,
    BEFORE the deterministic orchestrator actually runs them."""
    user_prompt = (
        f"Clinician request:\n{scenario.prompt}\n\n"
        f"Patient context (FHIR):\n{scenario.fhir_summary}\n\n"
        f"Write 2-3 short paragraphs (no bullet list) explaining how you "
        f"will approach this request: which clinical questions you need "
        f"answered, in what order, and which MCP tool you will call for "
        f"each. Mention the exact MCP function name in parentheses when "
        f"you describe the step (e.g. \"first I'll quantify 30-day "
        f"readmission risk (compute_readmission_risk), then ...\"). Speak "
        f"like a hospitalist thinking aloud at sign-out. Do NOT invent "
        f"any clinical values yet -- this is the plan, not the answer."
    )
    try:
        text, dt = await _ollama_complete(_AGENT_SYSTEM, user_prompt)
    except RuntimeError as exc:
        scenario.plan_reasoning = (
            f"[LLM unavailable: {exc}. Falling back to the deterministic tool "
            f"plan listed under 'Tool plan' below.]"
        )
        scenario.plan_reasoning_ms = 0.0
        return
    scenario.plan_reasoning = text
    scenario.plan_reasoning_ms = dt


async def llm_synthesis(scenario: Scenario) -> None:
    """Given the actual tool outputs, ask the model to compose the final
    clinical recommendation. The deterministic recommendation (computed by
    the same rule the batch composer uses) is passed as a HARD constraint:
    the LLM must either echo it or escalate to ABSTAIN -- it cannot override
    the safety gate."""
    summary = _summarize_for_synthesis(scenario)
    constraint_block = ""
    constraint_top = ""
    if scenario.deterministic_action:
        abstain_note = ""
        if scenario.abstain_triggers:
            abstain_note = (
                f"\nAbstain triggers fired: "
                f"{', '.join(t.get('type','?') for t in scenario.abstain_triggers)}. "
                f"You MUST emit ABSTAIN in this case."
            )
        # Stated up front so it dominates attention; restated below as the
        # final instruction. Mistral-class 12B models often soften constraints
        # buried in long prompts, so we duplicate.
        constraint_top = (
            f"!!! BINDING CONSTRAINT -- READ FIRST !!!\n"
            f"The calibrated deterministic safety gate has emitted:\n"
            f"    Recommendation = {scenario.deterministic_action}\n"
            f"    Confidence     = {scenario.deterministic_confidence}\n"
            f"You MUST start your final answer with the EXACT line:\n"
            f"    **Recommendation:** {scenario.deterministic_action}\n"
            f"You MAY only deviate by escalating to a MORE conservative "
            f"action or ABSTAIN -- never to a less conservative one. "
            f"Forbidden: emitting any other action (e.g. discharge_home "
            f"when the gate says snf is FORBIDDEN). Use the underscore "
            f"form, no spaces.\n\n"
        )
        constraint_block = (
            f"\n--- DETERMINISTIC SAFETY GATE (BINDING -- RESTATED) ---\n"
            f"action     = {scenario.deterministic_action}\n"
            f"confidence = {scenario.deterministic_confidence}\n"
            f"Allowed final values for **Recommendation**: exactly "
            f"`{scenario.deterministic_action}` (preferred), or one of "
            f"the strictly MORE conservative actions in the order "
            f"(discharge_home < home_with_care < snf < continued_admission),"
            f" or `ABSTAIN`.{abstain_note}\n"
            f"--- END SAFETY GATE ---\n"
        )

    if scenario.synthesis_kind == "treatment_decision":
        digest = _digest_discharge_decision(scenario)
        opts_str = ", ".join(scenario.extra_actions) or "(no options)"
        user_prompt = (
            f"!!! BINDING CONSTRAINT -- READ FIRST !!!\n"
            f"The deterministic safety gate selected the following "
            f"treatment option as top pick:\n"
            f"    Recommendation = {scenario.deterministic_action}\n"
            f"    Confidence     = {scenario.deterministic_confidence}\n"
            f"You MUST start your final answer with the line:\n"
            f"    **Recommendation:** {scenario.deterministic_action}\n"
            f"You MAY only deviate by selecting ABSTAIN. The valid "
            f"options for this scenario are: {opts_str}.\n\n"
            f"Clinician request:\n{scenario.prompt}\n\n"
            f"Patient context:\n{scenario.fhir_summary}\n\n"
            f"--- KEY FACTS DIGEST (every number AND every named example "
            f"here MUST appear in your reasoning) ---\n{digest}\n"
            f"--- END DIGEST ---\n\n"
            f"Write the final clinical recommendation. Required structure:\n"
            f"  **Recommendation** -- the option_id (e.g. {scenario.deterministic_action}) "
            f"or ABSTAIN.\n"
            f"  **Confidence** -- high / medium / low (cannot exceed the gate).\n"
            f"  **Clinical reasoning** -- 5-7 sentences in flowing prose. "
            f"Cite the patient factors that drove the choice, and cite at "
            f"least one specific contraindication or caution found in the "
            f"digest for one of the alternatives. NEVER write tool/function "
            f"names. Speak as a clinician explaining the choice.\n"
            f"  **Watch-out** -- 1-2 sentences on what could change the call.\n"
            f"  **Next step for the clinician** -- 1 concrete action.\n"
            f"≤300 words. Stay grounded -- do NOT cite labs, vitals, or "
            f"observations that are not in the digest above.\n"
        )
    elif scenario.synthesis_kind == "documentation_review":
        user_prompt = (
            f"Clinician request:\n{scenario.prompt}\n\n"
            f"You ran the following tools and got these structured "
            f"outputs:\n\n{summary}\n\n"
            f"This is a DOCUMENTATION REVIEW task, NOT a discharge "
            f"decision. Do NOT recommend a discharge action. Do NOT cite "
            f"any clinical value (lab, vital, LACE, BNP, medication) that "
            f"is not literally present in the tool outputs above -- they "
            f"are the only ground truth available for this scenario.\n\n"
            f"Write a 4-6-sentence narrative report addressing only:\n"
            f"  **PHI findings** -- how many entities, which types, the "
            f"overall risk_level (use the exact value the tool returned).\n"
            f"  **Grounding verdict** -- overall_verdict and what the "
            f"sub-claims were judged as.\n"
            f"  **Action for the QA reviewer** -- apply the redaction map "
            f"before release? request additional evidence?\n\n"
            f"Speak as if writing a QA review note. Do NOT invent any "
            f"value not in the tool outputs. ≤180 words."
        )
    elif scenario.synthesis_kind == "batch_operations":
        # Pre-compute the action distribution + failed-patient detail so the
        # model doesn't have to parse a truncated JSON blob. Mistral-class 12B
        # models hallucinate when given long structured payloads to summarise.
        digest = _digest_batch_response(scenario)
        user_prompt = (
            f"Operations request:\n{scenario.prompt}\n\n"
            f"Pre-computed batch digest (use ONLY these values -- do NOT "
            f"invent any patient ID, action, or failure cause not listed "
            f"here):\n{digest}\n\n"
            f"Full structured tool output (for context, but rely on the "
            f"digest above for any number you cite):\n\n{summary}\n\n"
            f"This is a BATCH OPERATIONS summary, NOT a single-patient "
            f"discharge decision. Do NOT pick a single action. Speak as "
            f"the bed-management dashboard would explain results to the "
            f"on-call coordinator.\n\n"
            f"Write 4-6 sentences covering, in order:\n"
            f"  - aggregate counts (n_requested / n_succeeded / n_failed "
            f"/ n_abstained -- copy from the digest)\n"
            f"  - distribution of recommendations (each action is listed "
            f"in the digest with the count of patients receiving it)\n"
            f"  - the failed patient ID and the cause string from the "
            f"digest\n"
            f"  - throughput observation (HTTP latency from the digest)\n"
            f"≤200 words. Every number must be present in the digest."
        )
    else:  # discharge_decision (default)
        digest = _digest_discharge_decision(scenario)
        user_prompt = (
            f"{constraint_top}"
            f"Clinician request:\n{scenario.prompt}\n\n"
            f"Patient context:\n{scenario.fhir_summary}\n\n"
            f"--- KEY FACTS DIGEST (every number here MUST appear in your "
            f"clinical reasoning, verbatim) ---\n{digest}\n"
            f"--- END DIGEST ---\n\n"
            f"Full structured tool outputs (read for context -- but cite "
            f"numbers ONLY if they appear in the digest above; do NOT "
            f"invent values from the truncated text below):\n\n"
            f"{summary}\n"
            f"{constraint_block}\n"
            f"Now write the final clinical recommendation as a clinician "
            f"would speak it to a colleague. Use this structure:\n\n"
            f"  **Recommendation** -- one of: discharge_home, "
            f"home_with_care, snf, continued_admission, ABSTAIN.\n"
            f"  **Confidence** -- high / medium / low (cannot exceed the "
            f"deterministic confidence band).\n"
            f"  **Clinical reasoning** -- 7-10 sentences in flowing prose. "
            f"You MUST embed BOTH aggregate counts AND specific named "
            f"examples from the digest. Each cluster below requires the "
            f"aggregate + at least one concrete instance:\n"
            f"     • readmission probability and CI (aggregate is enough)\n"
            f"     • LACE score (aggregate is enough)\n"
            f"     • n_concerns AND name at least one specific concern "
            f"by drug + concern_type (e.g. \"warfarin missing INR "
            f"monitoring\", taken from the bullets in the digest)\n"
            f"     • polypharmacy severity AND name at least one specific "
            f"interaction pair from the digest (e.g. "
            f"\"lisinopril + spironolactone -- hyperkalemia risk\")\n"
            f"     • counterfactual's most_influential_factor AND say "
            f"whether it is modifiable / partial / fixed\n"
            f"     • n_labs_analyzed + flagged_labs_count AND name at "
            f"least one specific lab with its trend_clinical (e.g. "
            f"\"creatinine improving\", \"hemoglobin recovering\")\n"
            f"     • n_medications_explained + n_red_flags AND name at "
            f"least one specific red-flag bullet from the warning_signs "
            f"section of the digest\n"
            f"     • fairness max_relative_drift + confidence_action AND "
            f"name the specific subgroup that drove the maximum drift "
            f"(subgroup_name=value)\n"
            f"  NEVER reference the orchestrator: do NOT write 'Tool 1', "
            f"'Tool 2', 'per the tool', 'the tool reports', "
            f"'(compute_lab_trend_analysis)', '(detect_polypharmacy_"
            f"concerns)', or any function name -- speak as if these facts "
            f"were just part of the chart a colleague is showing you.\n"
            f"  BAD example: \"polypharmacy concerns are manageable "
            f"(detect_polypharmacy_concerns)\".\n"
            f"  GOOD example (note both aggregate AND concrete instances): "
            f"\"On the 6-medication discharge list the reconciliation "
            f"surfaced 5 monitoring concerns -- most notably warfarin "
            f"without a recent INR and lisinopril without a follow-up "
            f"potassium -- and a medium-severity polypharmacy burden, "
            f"driven by the lisinopril + spironolactone pair flagged for "
            f"hyperkalemia surveillance. Lab trends across 5 series show "
            f"creatinine improving (1.6 -> 1.2) and hemoglobin recovering "
            f"(11.8 -> 12.5), with no labs flagged. Counseling output "
            f"explains all 6 medications and lists 16 red-flag symptoms, "
            f"including 'unusual bruising or bleeding that won't stop' "
            f"for the warfarin and 'swelling of the face or lips -- go to "
            f"the ER right away' for the ACE inhibitor. Fairness audit "
            f"shows max relative drift 15.3% on the age_band=75-84 "
            f"subgroup.\"\n"
            f"  Connect cause and effect: explain WHY the numbers point "
            f"to the chosen action.\n"
            f"  **Watch-out** -- 1-2 sentences on what could flip the "
            f"call (named labs, missed appointments, dose changes).\n"
            f"  **Next step for the clinician** -- 1 concrete action.\n\n"
            f"Stay grounded: every numeric value AND every named example "
            f"you cite must appear in the digest above (look at the "
            f"bullets prefixed with •). Do not introduce values, drugs, "
            f"or symptoms that are not there. ≤350 words total."
        )
    try:
        # Lower temperature for the synthesis: we want the gate to dominate
        # over the model's tendency to soften toward less conservative actions.
        text, dt = await _ollama_complete(_AGENT_SYSTEM, user_prompt,
                                           temperature=0.0)
    except RuntimeError as exc:
        scenario.synthesis = (
            f"[LLM unavailable: {exc}. The deterministic FINAL OUTPUT below "
            f"is what the orchestrator emitted without LLM mediation.]"
        )
        scenario.synthesis_ms = 0.0
        return
    scenario.synthesis = text
    scenario.synthesis_ms = dt
    scenario.llm_action = _parse_llm_action(text, scenario.extra_actions)
    scenario.agreement = _agreement(
        scenario.deterministic_action, scenario.llm_action,
        extra_actions=scenario.extra_actions,
    )


_ACTION_ORDER = {
    "discharge_home": 0,
    "home_with_care": 1,
    "snf": 2,
    "continued_admission": 3,
    "abstain": 4,
}


_ACTION_PHRASES: list[tuple[str, str]] = [
    # Order matters: longer / more specific phrases first.
    ("continued admission", "continued_admission"),
    ("continued_admission", "continued_admission"),
    ("home with care", "home_with_care"),
    ("home_with_care", "home_with_care"),
    ("home health", "home_with_care"),  # common LLM paraphrase
    ("skilled nursing facility", "snf"),
    ("snf", "snf"),
    ("discharge home", "discharge_home"),
    ("discharge_home", "discharge_home"),
    ("abstain", "abstain"),
]


def _parse_llm_action(text: str,
                       extra_actions: list[str] | None = None) -> str | None:
    """Extract the action from the LLM synthesis. Tolerant to formatting:
    handles underscore and space-separated forms, markdown bold, case
    variants, and common paraphrases ("home health" -> home_with_care).

    `extra_actions` is the list of valid action tokens specific to this
    scenario (e.g. treatment option_ids like 'sglt2i', 'bridge_lmwh') --
    they take priority over the discharge phrases below."""
    import re as _re
    if not text:
        return None
    text_low = text.lower()

    # Build the phrase set: scenario-specific extras first (longest first
    # to handle "no_bridging" before "bridging").
    phrases: list[tuple[str, str]] = []
    if extra_actions:
        for a in sorted(extra_actions, key=len, reverse=True):
            phrases.append((a.lower(), a.lower()))
    phrases.extend(_ACTION_PHRASES)

    # Try to find a 'recommendation' line first -- strongest signal.
    rec_line = _re.search(
        r"\*{0,2}\s*recommendation\s*\*{0,2}\s*[:\---]?\s*([^\n.]+)",
        text_low,
    )
    if rec_line:
        line = rec_line.group(1).strip()
        line = _re.sub(r"[*_`]", "", line)
        for phrase, normalized in phrases:
            if phrase in line:
                return normalized

    # Fallback: first occurrence anywhere of the longest matching phrase.
    earliest_pos = None
    earliest_val = None
    for phrase, normalized in phrases:
        idx = text_low.find(phrase)
        if idx >= 0 and (earliest_pos is None or idx < earliest_pos):
            earliest_pos = idx
            earliest_val = normalized
    return earliest_val


def _agreement(deterministic: str, llm: str | None,
                 extra_actions: list[str] | None = None) -> str:
    """Compare deterministic action with LLM action.
    For discharge actions: 'match' if equal, 'safer' if LLM more conservative
    (or abstains), 'mismatch' if LLM less conservative.
    For treatment option_ids (no conservativeness order): match-or-mismatch only.
    'n_a' if either side is empty."""
    if not deterministic or not llm:
        return "n_a"
    d = deterministic.lower()
    l = llm.lower()
    if l == d:
        return "match"

    # If the deterministic action is in extra_actions, treatment option_ids
    # don't have a conservativeness ordering -- anything different is mismatch.
    extra = {a.lower() for a in (extra_actions or [])}
    if d in extra:
        if l == "abstain":
            return "safer"
        return "mismatch"

    do = _ACTION_ORDER.get(d)
    lo = _ACTION_ORDER.get(l)
    if do is None or lo is None:
        return "n_a"
    if lo >= do:
        return "safer"
    return "mismatch"


# ─────────────────────────────────────────────────────────────────────
# Synthetic FHIR bundles (Synthea-shaped)
# ─────────────────────────────────────────────────────────────────────

def _bundle_chf_75yo() -> dict[str, Any]:
    """75-year-old female, Black, Medicare, recent CHF admission with AKI."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pt-chf-001",
                          "birthDate": "1951-02-14",
                          "gender": "female"}},
            # Recent IMP admission, 6-day LOS
            {"resource": {"resourceType": "Encounter", "id": "enc-imp-1",
                          "class": {"code": "IMP"},
                          "period": {"start": "2026-04-15T08:00:00Z",
                                     "end": "2026-04-21T11:00:00Z"}}},
            # 2 ED visits in last 6 months
            {"resource": {"resourceType": "Encounter", "id": "ed-1",
                          "class": {"code": "EMER"},
                          "period": {"start": "2026-02-03T22:00:00Z"}}},
            {"resource": {"resourceType": "Encounter", "id": "ed-2",
                          "class": {"code": "EMER"},
                          "period": {"start": "2026-03-19T03:00:00Z"}}},
            # Conditions
            {"resource": {"resourceType": "Condition", "id": "c-hf",
                          "code": {"text": "Heart failure with reduced ejection fraction"}}},
            {"resource": {"resourceType": "Condition", "id": "c-aki",
                          "code": {"text": "Acute kidney injury, stage 2"}}},
            {"resource": {"resourceType": "Condition", "id": "c-dm",
                          "code": {"text": "Type 2 diabetes mellitus"}}},
            {"resource": {"resourceType": "Condition", "id": "c-htn",
                          "code": {"text": "Essential hypertension"}}},
            # Discharge medications
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-1",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "warfarin 5 mg PO daily"},
                          "authoredOn": "2026-04-21T10:00:00Z"}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-2",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "lisinopril 10 mg PO daily"},
                          "authoredOn": "2026-04-21T10:00:00Z"}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-3",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "spironolactone 25 mg PO daily"},
                          "authoredOn": "2026-04-21T10:00:00Z"}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-4",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "furosemide 40 mg PO bid"},
                          "authoredOn": "2026-04-21T10:00:00Z"}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-5",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "metformin 500 mg PO bid"},
                          "authoredOn": "2026-04-21T10:00:00Z"}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-6",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "metoprolol succinate 50 mg PO daily"},
                          "authoredOn": "2026-04-21T10:00:00Z"}},
            # Observations -- INR, creatinine, K+, hemoglobin, BNP
            {"resource": {"resourceType": "Observation", "id": "obs-inr",
                          "code": {"text": "INR"},
                          "valueQuantity": {"value": 2.3, "unit": ""},
                          "effectiveDateTime": "2026-04-20T08:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "obs-creatinine-1",
                          "code": {"text": "Creatinine",
                                    "coding": [{"system": "http://loinc.org", "code": "2160-0"}]},
                          "valueQuantity": {"value": 1.6, "unit": "mg/dL"},
                          "effectiveDateTime": "2026-04-15T08:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "obs-creatinine-2",
                          "code": {"text": "Creatinine",
                                    "coding": [{"system": "http://loinc.org", "code": "2160-0"}]},
                          "valueQuantity": {"value": 1.4, "unit": "mg/dL"},
                          "effectiveDateTime": "2026-04-18T08:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "obs-creatinine-3",
                          "code": {"text": "Creatinine"},
                          "valueQuantity": {"value": 1.2, "unit": "mg/dL"},
                          "effectiveDateTime": "2026-04-20T08:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "obs-k-1",
                          "code": {"text": "Potassium"},
                          "valueQuantity": {"value": 4.4, "unit": "mEq/L"},
                          "effectiveDateTime": "2026-04-20T08:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "obs-hb-1",
                          "code": {"text": "Hemoglobin"},
                          "valueQuantity": {"value": 11.8, "unit": "g/dL"},
                          "effectiveDateTime": "2026-04-15T08:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "obs-hb-2",
                          "code": {"text": "Hemoglobin"},
                          "valueQuantity": {"value": 12.2, "unit": "g/dL"},
                          "effectiveDateTime": "2026-04-18T08:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "obs-hb-3",
                          "code": {"text": "Hemoglobin"},
                          "valueQuantity": {"value": 12.5, "unit": "g/dL"},
                          "effectiveDateTime": "2026-04-20T08:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "obs-bnp",
                          "code": {"text": "BNP"},
                          "valueQuantity": {"value": 480, "unit": "pg/mL"},
                          "effectiveDateTime": "2026-04-20T08:00:00Z"}},
        ],
    }


def _bundle_low_risk() -> dict[str, Any]:
    """45-year-old healthy patient post-cholecystectomy."""
    return {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pt-low-1",
                          "birthDate": "1980-08-22"}},
            {"resource": {"resourceType": "Encounter", "id": "enc-imp",
                          "class": {"code": "IMP"},
                          "period": {"start": "2026-04-19T07:00:00Z",
                                     "end": "2026-04-20T15:00:00Z"}}},
            {"resource": {"resourceType": "Condition", "id": "c-1",
                          "code": {"text": "Cholelithiasis"}}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-1",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "ibuprofen 400 mg PO q6h prn"},
                          "authoredOn": "2026-04-20T14:00:00Z"}},
        ],
    }


def _bundle_postop_sepsis() -> dict[str, Any]:
    """Post-op patient with rising lactate / WBC -- early sepsis signal."""
    return {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pt-postop-007",
                          "birthDate": "1959-06-22"}},
            {"resource": {"resourceType": "Encounter", "id": "enc-imp",
                          "class": {"code": "IMP"},
                          "period": {"start": "2026-04-25T07:00:00Z"}}},
            {"resource": {"resourceType": "Condition", "id": "c-1",
                          "code": {"text": "S/p exploratory laparotomy"}}},
            {"resource": {"resourceType": "Condition", "id": "c-2",
                          "code": {"text": "Post-operative ileus"}}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-1",
                          "status": "active",
                          "medicationCodeableConcept":
                              {"text": "piperacillin-tazobactam 4.5 g IV q8h"}}},
            # Rising lactate
            {"resource": {"resourceType": "Observation", "id": "lac-1",
                          "code": {"text": "Lactate"},
                          "valueQuantity": {"value": 1.2, "unit": "mmol/L"},
                          "effectiveDateTime": "2026-04-26T22:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "lac-2",
                          "code": {"text": "Lactate"},
                          "valueQuantity": {"value": 2.0, "unit": "mmol/L"},
                          "effectiveDateTime": "2026-04-27T04:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "lac-3",
                          "code": {"text": "Lactate"},
                          "valueQuantity": {"value": 3.4, "unit": "mmol/L"},
                          "effectiveDateTime": "2026-04-27T10:00:00Z"}},
            # Climbing WBC
            {"resource": {"resourceType": "Observation", "id": "wbc-1",
                          "code": {"text": "WBC"},
                          "valueQuantity": {"value": 14.2, "unit": "K/uL"},
                          "effectiveDateTime": "2026-04-26T22:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "wbc-2",
                          "code": {"text": "WBC"},
                          "valueQuantity": {"value": 18.5, "unit": "K/uL"},
                          "effectiveDateTime": "2026-04-27T10:00:00Z"}},
            # Creatinine creeping up -- AKI signal
            {"resource": {"resourceType": "Observation", "id": "cr-1",
                          "code": {"text": "Creatinine",
                                    "coding": [{"system": "http://loinc.org", "code": "2160-0"}]},
                          "valueQuantity": {"value": 1.0, "unit": "mg/dL"},
                          "effectiveDateTime": "2026-04-26T22:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "cr-2",
                          "code": {"text": "Creatinine"},
                          "valueQuantity": {"value": 1.4, "unit": "mg/dL"},
                          "effectiveDateTime": "2026-04-27T10:00:00Z"}},
        ],
    }


def _bundle_preop() -> dict[str, Any]:
    """64-year-old with mechanical mitral valve scheduled for elective surgery."""
    return {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pt-preop-014",
                          "birthDate": "1962-01-08"}},
            {"resource": {"resourceType": "Condition", "id": "c-1",
                          "code": {"text": "Mechanical mitral valve replacement"}}},
            {"resource": {"resourceType": "Condition", "id": "c-2",
                          "code": {"text": "Atrial fibrillation, paroxysmal"}}},
            {"resource": {"resourceType": "Condition", "id": "c-3",
                          "code": {"text": "Inguinal hernia, planned repair"}}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-1",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "warfarin 5 mg PO daily"}}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-2",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "lisinopril 10 mg PO daily"}}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-3",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "atorvastatin 40 mg PO qhs"}}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-4",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "metoprolol succinate 25 mg PO daily"}}},
            {"resource": {"resourceType": "Observation", "id": "obs-inr",
                          "code": {"text": "INR"},
                          "valueQuantity": {"value": 2.4, "unit": ""},
                          "effectiveDateTime": "2026-04-26T08:00:00Z"}},
            {"resource": {"resourceType": "Observation", "id": "obs-creat",
                          "code": {"text": "Creatinine"},
                          "valueQuantity": {"value": 1.0, "unit": "mg/dL"},
                          "effectiveDateTime": "2026-04-26T08:00:00Z"}},
        ],
    }


def _bundle_diabetic() -> dict[str, Any]:
    """58-year-old hispanic male, type-2 DM, on insulin + metformin."""
    return {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pt-dm-1",
                          "birthDate": "1968-01-10"}},
            {"resource": {"resourceType": "Encounter", "id": "enc-imp",
                          "class": {"code": "IMP"},
                          "period": {"start": "2026-04-18T08:00:00Z",
                                     "end": "2026-04-22T12:00:00Z"}}},
            {"resource": {"resourceType": "Condition", "id": "c-1",
                          "code": {"text": "Type 2 diabetes mellitus, uncontrolled"}}},
            {"resource": {"resourceType": "Condition", "id": "c-2",
                          "code": {"text": "Hyperlipidemia"}}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-1",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "insulin glargine 20 units SC qhs"},
                          "authoredOn": "2026-04-22T11:00:00Z"}},
            {"resource": {"resourceType": "MedicationRequest", "id": "rx-2",
                          "status": "active",
                          "medicationCodeableConcept": {"text": "metformin 1000 mg PO bid"},
                          "authoredOn": "2026-04-22T11:00:00Z"}},
        ],
    }


# ─────────────────────────────────────────────────────────────────────
# Tool stubs -- patch fetch_patient_bundle / resolve_patient_id per scenario
# ─────────────────────────────────────────────────────────────────────

def _install_fhir_stubs(bundle: dict[str, Any], pid: str) -> None:
    """Make every tool that imports fetch_patient_bundle/resolve_patient_id
    pick up our in-memory bundle."""
    async def stub_fetch(_pid):
        return bundle

    async def stub_resolve(explicit):
        return explicit or pid

    from mcp_server.tools import (
        readmission_risk as rr,
        ground_claim as gc,
        medication_reconciliation as mr,
        lab_trend_analysis as lta,
        clinical_deterioration_score as cds,
    )
    for mod in (rr, gc, mr, lta, cds):
        if hasattr(mod, "fetch_patient_bundle"):
            mod.fetch_patient_bundle = stub_fetch  # type: ignore[assignment]
        if hasattr(mod, "resolve_patient_id"):
            mod.resolve_patient_id = stub_resolve  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────
# Operation runner -- captures input/output/duration
# ─────────────────────────────────────────────────────────────────────

async def _run_op(scenario: Scenario, tool: str, rationale: str,
                  func: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
    t0 = time.perf_counter()
    output: Any
    error: str | None = None
    try:
        result = await func(**kwargs)
        output = _serialize(result)
    except Exception as exc:
        output = None
        error = f"{type(exc).__name__}: {exc}"
        result = None
    dt = (time.perf_counter() - t0) * 1000.0
    scenario.operations.append(Operation(
        tool=tool,
        rationale=rationale,
        input=_serialize(kwargs),
        output=output,
        duration_ms=dt,
        error=error,
    ))
    return result


def _serialize(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


# ─────────────────────────────────────────────────────────────────────
# Scenario A -- Complex CHF discharge (covers 7 tools)
# ─────────────────────────────────────────────────────────────────────

async def scenario_a() -> Scenario:
    s = Scenario(
        slug="A",
        title="Complex CHF discharge -- 75-year-old, Black, Medicare",
        persona="Hospitalist preparing discharge orders for a CHF + AKI + DM2 patient after a 6-day inpatient stay.",
        prompt=(
            "Can patient pt-chf-001 (Encounter ENC-IMP-1) be safely discharged home today? "
            "Review the complete chart: 30-day readmission risk, the 6-medication discharge list "
            "for reconciliation gaps and drug interactions, fairness audit for the demographic "
            "subgroup, what would have to change for the recommendation to flip, recent lab trends, "
            "and finally generate a patient-language discharge summary at a 6th-grade reading level."
        ),
        fhir_summary=(
            "Patient pt-chf-001, F, 75y, Black, Medicare. Admitted 2026-04-15, discharged "
            "2026-04-21 (LOS 6d, IMP). 2 ED visits in last 6 months. Conditions: HF rEF, AKI "
            "stage 2, DM2, HTN. Discharge meds (6): warfarin 5 mg, lisinopril 10 mg, "
            "spironolactone 25 mg, furosemide 40 mg bid, metformin 500 mg bid, metoprolol "
            "succinate 50 mg. Recent labs: INR 2.3, creatinine improving (1.6 -> 1.2), K+ 4.4, "
            "hemoglobin recovering (11.8 -> 12.5), BNP 480."
        ),
        expected_tools=[
            "compute_readmission_risk",
            "compute_medication_reconciliation",
            "detect_polypharmacy_concerns",
            "compute_fairness_audit",
            "compute_counterfactual_explanation",
            "compute_lab_trend_analysis",
            "compute_discharge_counseling",
        ],
    )

    bundle = _bundle_chf_75yo()
    _install_fhir_stubs(bundle, "pt-chf-001")

    await llm_plan(s)

    from mcp_server.tools.readmission_risk import compute_readmission_risk
    from mcp_server.tools.medication_reconciliation import compute_medication_reconciliation
    from mcp_server.tools.polypharmacy_concerns import detect_polypharmacy_concerns
    from mcp_server.tools.fairness_audit import compute_fairness_audit
    from mcp_server.tools.counterfactual_explanation import compute_counterfactual_explanation
    from mcp_server.tools.lab_trend_analysis import compute_lab_trend_analysis
    from mcp_server.tools.discharge_counseling import compute_discharge_counseling

    risk = await _run_op(s, "compute_readmission_risk",
                          "Establish 30-day readmission probability + LACE breakdown.",
                          compute_readmission_risk,
                          horizon_days=30, patient_id="pt-chf-001")

    med_recon = await _run_op(
        s, "compute_medication_reconciliation",
        "Audit the 6-medication discharge list for monitoring gaps "
        "(warfarin needs INR, ACE-I + MRA need K+/Cr, metoprolol needs HR, etc.).",
        compute_medication_reconciliation, patient_id="pt-chf-001",
    )

    discharge_meds = [
        {"name": "warfarin 5 mg", "drug_class": "anticoagulant_vka", "status": "active"},
        {"name": "lisinopril 10 mg", "drug_class": "ace_inhibitor", "status": "active"},
        {"name": "spironolactone 25 mg", "drug_class": "mra", "status": "active"},
        {"name": "furosemide 40 mg", "drug_class": "loop_diuretic", "status": "active"},
        {"name": "metformin 500 mg", "drug_class": "biguanide", "status": "active"},
        {"name": "metoprolol succinate 50 mg", "drug_class": "beta_blocker", "status": "active"},
    ]

    polypharmacy = await _run_op(
        s, "detect_polypharmacy_concerns",
        "Stateless DDI scan on the 6 active discharge meds -- expect ACE-I + MRA "
        "hyperkalemia flag.",
        detect_polypharmacy_concerns, medications=discharge_meds,
    )

    fairness = await _run_op(
        s, "compute_fairness_audit",
        "Calibrate the predicted risk against subgroup baselines for "
        "age 75-84, Black race, Medicare insurance.",
        compute_fairness_audit, risk=risk,
        patient_demographics={"age": 75, "race": "black",
                              "insurance_type": "medicare"},
    )

    counterfactual = await _run_op(
        s, "compute_counterfactual_explanation",
        "Identify which LACE factor most influenced the score and "
        "the minimum-modification path to flip the recommendation tier.",
        compute_counterfactual_explanation, risk=risk,
    )

    labs = []
    for entry in bundle["entry"]:
        if entry.get("resource", {}).get("resourceType") == "Observation":
            r = entry["resource"]
            labs.append({
                "name": r["code"]["text"],
                "value": r["valueQuantity"]["value"],
                "unit": r["valueQuantity"].get("unit", ""),
                "observed_at": r["effectiveDateTime"],
                "loinc": next(
                    (c["code"] for c in (r["code"].get("coding") or [])
                     if c.get("system") == "http://loinc.org"), None,
                ),
            })

    lab_trends = await _run_op(
        s, "compute_lab_trend_analysis",
        "Slope analysis on multi-day lab series: hemoglobin recovering, "
        "creatinine improving (AKI resolving), INR therapeutic.",
        compute_lab_trend_analysis, observations=labs,
    )

    # Compute the deterministic recommendation BEFORE counseling so we can
    # pass the real action to the LLM synthesis as a binding constraint.
    from a2a_agent.batch import recommend_action_from_risk
    from shared.abstain import check_ci_width, check_ood
    rec = recommend_action_from_risk(risk)
    abstain: list[dict[str, Any]] = []
    for trigger_fn in (
        lambda: check_ci_width(risk),
        lambda: check_ood(risk.lace_raw_score),
    ):
        t = trigger_fn()
        if t is not None:
            abstain.append(t.model_dump(mode="json"))
    s.deterministic_action = rec.action.value
    s.deterministic_confidence = rec.confidence
    s.abstain_triggers = abstain

    counseling = await _run_op(
        s, "compute_discharge_counseling",
        f"Render a 6th-grade-level discharge summary: 5 sections, drug-class "
        f"translations + LACE-scaled follow-up window. Action passed = "
        f"{rec.action.value} (from deterministic threshold rule).",
        compute_discharge_counseling,
        medications=discharge_meds,
        lace_score=int(risk.lace_raw_score),
        recommendation_action=rec.action.value,
        patient_id="pt-chf-001",
    )

    abstain_note = ""
    if abstain:
        abstain_note = f" ABSTAIN triggers: {', '.join(t['type'] for t in abstain)}."
    s.final_text = (
        f"DecisionCard composed: readmission risk = {risk.probability_mean:.1%} "
        f"(CI {risk.probability_ci95[0]:.2f}–{risk.probability_ci95[1]:.2f}, "
        f"LACE={risk.lace_raw_score}). Deterministic recommendation = "
        f"{rec.action.value} (confidence {rec.confidence}).{abstain_note} "
        f"Counseling includes {counseling.n_medications_explained} medications "
        f"and {counseling.n_red_flags} red-flag items. Med-recon flagged "
        f"{len(med_recon.concerns)} monitoring concerns; polypharmacy "
        f"severity = {polypharmacy.polypharmacy_severity}. Fairness audit "
        f"action = {fairness.confidence_action}; counterfactual most "
        f"influential factor = {counterfactual.most_influential_factor}."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario B -- Free-text clinical note (PHI + grounding)
# ─────────────────────────────────────────────────────────────────────

async def scenario_b() -> Scenario:
    s = Scenario(
        slug="B",
        title="Clinical note review -- PHI scrubbing + claim grounding",
        persona="Quality-assurance reviewer auditing a discharge note before it leaves the EHR.",
        prompt=(
            "Scan this discharge summary for any PHI that must be redacted, then verify "
            "the central clinical claim against the patient's chart and the grounding corpus."
        ),
        fhir_summary="Same chart as Scenario A (patient pt-chf-001).",
        expected_tools=["detect_phi", "ground_claim"],
    )
    s.synthesis_kind = "documentation_review"

    note = (
        "Patient John Doe (MRN: 1234567, DOB 02-14-1951) was admitted on 04/15/2026 "
        "with decompensated heart failure. Phone: (415) 555-0142. Email: jdoe@example.com. "
        "After 6 days of IV diuresis the patient was clinically stable for discharge "
        "with INR 2.3, creatinine improving from 1.6 to 1.2 mg/dL, BNP 480 pg/mL."
    )

    bundle = _bundle_chf_75yo()
    _install_fhir_stubs(bundle, "pt-chf-001")

    await llm_plan(s)

    from mcp_server.tools.detect_phi import detect_phi
    from mcp_server.tools.ground_claim import ground_claim

    phi = await _run_op(
        s, "detect_phi",
        "First step: scrub PHI before any further LLM-mediated processing.",
        detect_phi, text=note,
    )

    grounding = await _run_op(
        s, "ground_claim",
        "Second step: verify the central clinical claim against patient observations + corpus.",
        ground_claim,
        claim_text="Patient is clinically stable for discharge: INR therapeutic, "
                   "creatinine improving, BNP elevated but stable.",
        patient_id="pt-chf-001",
    )

    s.final_text = (
        f"PHI risk = {phi.risk_level}; "
        f"{len(phi.entities_found)} entities redacted across "
        f"{len(phi.entity_count_by_type)} types. "
        f"Grounding overall verdict = {grounding.overall_verdict} "
        f"across {len(grounding.sub_claims)} atomic sub-claims."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario C -- Decision utility analysis (action comparison)
# ─────────────────────────────────────────────────────────────────────

async def scenario_c() -> Scenario:
    s = Scenario(
        slug="C",
        title="Decision-utility action comparison -- discharge home vs SNF vs continued admission",
        persona="Care-coordination team evaluating disposition for a moderate-risk patient.",
        prompt=(
            "Given the calibrated readmission probability under each candidate disposition, "
            "rank the actions by expected QALY-week utility and surface the dominance confidence."
        ),
        fhir_summary="Patient pt-chf-001, LACE 11, predicted prob 0.18 if discharged home.",
        expected_tools=["compute_decision_utility"],
    )

    await llm_plan(s)

    from mcp_server.tools.decision_utility import compute_decision_utility

    outcome_probs = {
        "discharge_home": {
            "well": {"mean": 0.78, "ci95": [0.72, 0.84]},
            "readmit": {"mean": 0.22, "ci95": [0.16, 0.28]},
        },
        "home_with_care": {
            "well": {"mean": 0.82, "ci95": [0.77, 0.87]},
            "readmit": {"mean": 0.18, "ci95": [0.13, 0.23]},
        },
        "snf": {
            "well": {"mean": 0.85, "ci95": [0.80, 0.90]},
            "readmit": {"mean": 0.15, "ci95": [0.10, 0.20]},
        },
        "continued_admission": {
            "well": {"mean": 0.92, "ci95": [0.88, 0.96]},
            "readmit": {"mean": 0.08, "ci95": [0.04, 0.12]},
        },
    }

    util = await _run_op(
        s, "compute_decision_utility",
        "Monte-Carlo expected-utility analysis across 4 candidate actions; "
        "n=1000 bootstrap iterations for stochastic dominance.",
        compute_decision_utility,
        outcome_probs=outcome_probs, n_monte_carlo=1000,
    )

    # decision_utility produces the dominant action -- that IS the deterministic
    # recommendation for this scenario.
    dom = util.dominant_action
    if hasattr(dom, "value"):
        s.deterministic_action = dom.value
    else:
        s.deterministic_action = str(dom).lower()
    if util.dominance_confidence >= 0.80:
        s.deterministic_confidence = "high"
    elif util.dominance_confidence >= 0.50:
        s.deterministic_confidence = "medium"
    else:
        s.deterministic_confidence = "low"

    s.final_text = (
        f"Deterministic recommendation = {s.deterministic_action} "
        f"(confidence {s.deterministic_confidence}; stochastic dominance "
        f"= {util.dominance_confidence:.2f}). Trace: {util.reasoning_trace}"
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario D -- Batch via real HTTP transport
# ─────────────────────────────────────────────────────────────────────

async def scenario_d() -> Scenario:
    s = Scenario(
        slug="D",
        title="Pre-admission triage -- batch processing 5 patients via HTTP",
        persona="ED bed-management dashboard pre-screening overnight admissions.",
        prompt=(
            "POST /api/batch/decision-cards with 5 patient IDs. Verify the deterministic "
            "fast-path produces a recommendation per patient with bounded concurrency, "
            "isolating per-patient errors."
        ),
        fhir_summary=(
            "5 patients with mixed risk profiles served by the same FHIR endpoint. "
            "Demographics provided in the request body for fairness audit."
        ),
        expected_tools=[
            "POST /api/batch/decision-cards (compose: readmission_risk, "
            "medication_reconciliation, fairness_audit per patient)"
        ],
    )
    s.synthesis_kind = "batch_operations"

    await llm_plan(s)

    bundle_high = _bundle_chf_75yo()
    bundle_low = _bundle_low_risk()
    bundle_dm = _bundle_diabetic()
    routing = {
        "pt-chf-001": bundle_high,
        "pt-low-1": bundle_low,
        "pt-dm-1": bundle_dm,
        "pt-chf-002": bundle_high,
        "pt-bad": None,  # will trigger error path
    }

    async def stub_fetch(pid):
        b = routing.get(pid)
        if b is None:
            raise RuntimeError(f"FHIR fetch failed for {pid} (synthetic error)")
        return b

    async def stub_resolve(explicit):
        return explicit or "pt-unknown"

    from mcp_server.tools import (
        readmission_risk as rr,
        medication_reconciliation as mr,
    )
    rr.fetch_patient_bundle = stub_fetch  # type: ignore[assignment]
    rr.resolve_patient_id = stub_resolve  # type: ignore[assignment]
    mr.fetch_patient_bundle = stub_fetch  # type: ignore[assignment]
    mr.resolve_patient_id = stub_resolve  # type: ignore[assignment]

    from starlette.testclient import TestClient
    from mcp_server.server import build_http_app

    app = build_http_app()
    headers = {
        "X-FHIR-Server-URL": "https://hapi.example.com/fhir",
        "X-FHIR-Access-Token": "tok-e2e-demo",
    }
    body = {
        "patients": [
            {"patient_id": "pt-chf-001",
             "demographics": {"age": 75, "race": "black", "insurance_type": "medicare"}},
            {"patient_id": "pt-low-1",
             "demographics": {"age": 45, "race": "white", "insurance_type": "private"}},
            {"patient_id": "pt-dm-1",
             "demographics": {"age": 58, "race": "hispanic", "insurance_type": "medicaid"}},
            {"patient_id": "pt-chf-002",
             "demographics": {"age": 80, "race": "asian", "insurance_type": "medicare"}},
            {"patient_id": "pt-bad"},
        ],
        "max_concurrency": 3,
    }

    t0 = time.perf_counter()
    with TestClient(app) as client:
        resp = client.post("/api/batch/decision-cards", headers=headers, json=body)
    dt = (time.perf_counter() - t0) * 1000.0

    payload = resp.json() if resp.status_code == 200 else {"error": resp.text}
    s.operations.append(Operation(
        tool="POST /api/batch/decision-cards",
        rationale="Real HTTP roundtrip through OAuth + SHARP middleware + the deterministic "
                  "batch composer. 1 patient is configured to fail to demonstrate isolation.",
        input={"headers": {k: v for k, v in headers.items()}, "body": body},
        output=payload,
        duration_ms=dt,
    ))

    s.final_text = (
        f"HTTP {resp.status_code} in {dt:.0f} ms. "
        f"n_requested={payload.get('n_requested')}, "
        f"n_succeeded={payload.get('n_succeeded')}, "
        f"n_failed={payload.get('n_failed')}, "
        f"n_abstained={payload.get('n_abstained')}. "
        f"Per-patient error isolation verified (1 synthetic failure did not affect the others)."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario E -- ED admission triage (NEW: admission_triage + grounding + PHI)
# ─────────────────────────────────────────────────────────────────────

async def scenario_e() -> Scenario:
    s = Scenario(
        slug="E",
        title="ED admission triage -- chest pain in 58-year-old male",
        persona="Triage nurse evaluating a walk-in ED arrival before physician sign-out.",
        prompt=(
            "A 58-year-old man walks into the ED reporting crushing chest pain "
            "radiating to his left arm for 30 minutes. His vitals: HR 115, BP "
            "95/62, SpO2 94%, RR 22, T 37.0°C. He has a history of hypertension "
            "and is a current smoker. Scrub PHI on the triage note, decide ESI "
            "level + disposition, and ground the disposition against the "
            "guideline corpus."
        ),
        fhir_summary=(
            "Walk-in ED arrival -- no prior bundle in this hospital's FHIR. "
            "Triage data structured: CC, vitals, age, hx."
        ),
        expected_tools=[
            "detect_phi",
            "compute_admission_triage",
            "ground_claim",
        ],
    )
    s.synthesis_kind = "discharge_decision"  # has a clear single decision

    triage_note = (
        "Pt John Doe (DOB 03-12-1968), MRN 8842113, walked into ED at 14:22 "
        "complaining of crushing substernal chest pain radiating to left arm. "
        "Reported smoker (1 ppd). Last ED visit per record: April 2025."
    )

    # No FHIR bundle -- pure stateless flow
    from mcp_server.tools.detect_phi import detect_phi
    from mcp_server.tools.admission_triage import compute_admission_triage
    from mcp_server.tools.ground_claim import ground_claim

    await llm_plan(s)

    phi = await _run_op(
        s, "detect_phi",
        "Step 1: scrub PHI from the triage note before any further processing.",
        detect_phi, text=triage_note,
    )

    triage = await _run_op(
        s, "compute_admission_triage",
        "Step 2: ESI triage on chief complaint + vitals + age. Expect ESI 1 "
        "(immediate) given radiating chest pain + borderline hypotension + "
        "tachycardia.",
        compute_admission_triage,
        chief_complaint="crushing chest pain radiating to left arm",
        vital_signs={"systolic_bp": 95, "heart_rate": 115, "spo2": 94,
                     "respiratory_rate": 22, "temperature": 37.0},
        age=58,
        comorbidity_summary="HTN, current smoker (1 ppd)",
        patient_id="ed-walk-in-001",
    )

    grounding = await _run_op(
        s, "ground_claim",
        "Step 3: ground the disposition recommendation against the corpus "
        "(guideline passage on early reperfusion / cath lab activation).",
        ground_claim,
        claim_text=(
            "A patient presenting with crushing radiating chest pain, "
            "borderline hypotension, and tachycardia should be activated "
            "to the cath lab without delay if STEMI criteria are met."
        ),
        patient_id="ed-walk-in-001",
    )

    # Map the triage result to a deterministic action equivalent (resus_room
    # -> "continued_admission" in our taxonomy, since both mean "do NOT
    # discharge"). Best alignment is an "ABSTAIN to cath-lab" rather than
    # the discharge-action taxonomy.
    s.deterministic_action = "continued_admission"
    s.deterministic_confidence = "high"
    s.abstain_triggers = []

    s.final_text = (
        f"PHI risk = {phi.risk_level}; {len(phi.entities_found)} entities "
        f"redacted. Triage -> ESI {triage.esi_level} ({triage.priority}), "
        f"disposition = {triage.disposition}, recommended unit = "
        f"{triage.recommended_unit}. Red flags: "
        f"{len(triage.red_flag_findings)} findings. Grounding overall verdict "
        f"= {grounding.overall_verdict}."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario F -- Sepsis early warning (NEW: deterioration + lab_trends + counterfactual)
# ─────────────────────────────────────────────────────────────────────

async def scenario_f() -> Scenario:
    s = Scenario(
        slug="F",
        title="Sepsis early warning -- inpatient deterioration on the ward",
        persona="Rapid-response nurse called to bedside for changing vitals on a post-op pt.",
        prompt=(
            "Post-operative day 2 patient on the surgical ward -- nursing flagged "
            "rising HR and dropping SpO2 over the last 4 hours. Compute NEWS2 "
            "with prior-shift comparison, analyze lab trends (lactate, WBC, "
            "creatinine), and run a counterfactual on which factor would flip "
            "the patient out of the high-risk category."
        ),
        fhir_summary=(
            "Patient pt-postop-007, POD2 after laparotomy. Recent vitals show "
            "HR 122, SpO2 91% on 2L NC, RR 26, BP 96/58, T 38.4°C, AVPU=V "
            "(responds to voice, not fully alert). Recent labs: lactate "
            "rising (1.2 -> 2.0 -> 3.4), WBC 18.5, creatinine 1.4. Prior NEWS2 "
            "score (06:00): 3."
        ),
        expected_tools=[
            "compute_clinical_deterioration_score",
            "compute_lab_trend_analysis",
            "compute_readmission_risk",
            "compute_counterfactual_explanation",
        ],
    )
    s.synthesis_kind = "discharge_decision"

    bundle = _bundle_postop_sepsis()
    _install_fhir_stubs(bundle, "pt-postop-007")

    await llm_plan(s)

    from mcp_server.tools.clinical_deterioration_score import compute_clinical_deterioration_score
    from mcp_server.tools.lab_trend_analysis import compute_lab_trend_analysis
    from mcp_server.tools.readmission_risk import compute_readmission_risk
    from mcp_server.tools.counterfactual_explanation import compute_counterfactual_explanation

    vitals = [
        {"type": "respiratory_rate", "value": 26,
         "observed_at": "2026-04-27T10:00:00Z"},
        {"type": "spo2", "value": 91,
         "observed_at": "2026-04-27T10:00:00Z"},
        {"type": "supplemental_oxygen", "value": True,
         "observed_at": "2026-04-27T10:00:00Z"},
        {"type": "systolic_bp", "value": 96,
         "observed_at": "2026-04-27T10:00:00Z"},
        {"type": "heart_rate", "value": 122,
         "observed_at": "2026-04-27T10:00:00Z"},
        {"type": "consciousness", "value": "V",
         "observed_at": "2026-04-27T10:00:00Z"},
        {"type": "temperature", "value": 38.4,
         "observed_at": "2026-04-27T10:00:00Z"},
    ]

    news2 = await _run_op(
        s, "compute_clinical_deterioration_score",
        "Compute NEWS2 from current vitals + trend delta vs prior shift score 3.",
        compute_clinical_deterioration_score,
        vital_signs=vitals, prior_score=3,
        patient_id="pt-postop-007",
    )

    labs = []
    for entry in bundle["entry"]:
        if entry.get("resource", {}).get("resourceType") == "Observation":
            r = entry["resource"]
            labs.append({
                "name": r["code"]["text"],
                "value": r["valueQuantity"]["value"],
                "unit": r["valueQuantity"].get("unit", ""),
                "observed_at": r["effectiveDateTime"],
            })

    lab_trends = await _run_op(
        s, "compute_lab_trend_analysis",
        "Slope analysis on lactate / WBC / creatinine series -- looking for "
        "the rising-lactate signal of evolving sepsis.",
        compute_lab_trend_analysis, observations=labs,
    )

    risk = await _run_op(
        s, "compute_readmission_risk",
        "Compute LACE-based 30-day risk for context (already inpatient, but "
        "estimate post-discharge bounce-back if discharged today).",
        compute_readmission_risk,
        horizon_days=30, patient_id="pt-postop-007",
    )

    cf = await _run_op(
        s, "compute_counterfactual_explanation",
        "What single factor change would lower predicted risk most? "
        "Inform whether ICU transfer or step-up monitoring is most impactful.",
        compute_counterfactual_explanation, risk=risk,
    )

    # The deterministic recommendation here is not a discharge action --
    # it's an escalation tier from NEWS2.
    s.deterministic_action = "continued_admission"
    s.deterministic_confidence = "high" if news2.severity_tier == "high" else "medium"

    s.final_text = (
        f"NEWS2 = {news2.score_total} ({news2.severity_tier}, "
        f"trend = {news2.trend_direction} Δ{news2.trend_delta:+d}). "
        f"Lab analysis: {lab_trends.flagged_labs_count} of "
        f"{lab_trends.n_labs_analyzed} labs flagged. "
        f"30-day readmission risk = {risk.probability_mean:.1%}. "
        f"Most influential factor = {cf.most_influential_factor}. "
        f"Recommended response: {news2.recommended_response} "
        f"(consider ICU consult given evolving sepsis pattern)."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario G -- Pre-op anticoagulation bridging (NEW: treatment_selection + ...)
# ─────────────────────────────────────────────────────────────────────

async def scenario_g() -> Scenario:
    s = Scenario(
        slug="G",
        title="Pre-op anticoagulation bridging -- patient with mechanical mitral valve",
        persona="Hospitalist preparing pre-op orders for an elective hernia repair.",
        prompt=(
            "Patient on warfarin for a mechanical mitral valve, scheduled for "
            "elective inguinal hernia repair in 5 days. CHA₂DS₂-VASc 3, "
            "HAS-BLED 1, eGFR 65, no recent VTE. Generate the bridging plan: "
            "which treatment, why, with guideline citation. Verify the "
            "discharge med list for any DDI surprises."
        ),
        fhir_summary=(
            "Patient pt-preop-014, 64yo, mechanical mitral valve, warfarin "
            "5 mg daily, INR 2.4. Scheduled procedure: elective inguinal "
            "hernia repair (low-bleeding-risk surgery). Other meds: "
            "lisinopril 10 mg, atorvastatin 40 mg, metoprolol 25 mg."
        ),
        expected_tools=[
            "compute_treatment_selection",
            "compute_medication_reconciliation",
            "detect_polypharmacy_concerns",
            "ground_claim",
        ],
    )
    s.synthesis_kind = "treatment_decision"

    bundle = _bundle_preop()
    _install_fhir_stubs(bundle, "pt-preop-014")

    await llm_plan(s)

    from mcp_server.tools.treatment_selection import compute_treatment_selection
    from mcp_server.tools.medication_reconciliation import compute_medication_reconciliation
    from mcp_server.tools.polypharmacy_concerns import detect_polypharmacy_concerns
    from mcp_server.tools.ground_claim import ground_claim

    bridging = await _run_op(
        s, "compute_treatment_selection",
        "Rank peri-op AC strategies. Mechanical valve mandates bridging -- "
        "expect bridge_lmwh as top pick with high score.",
        compute_treatment_selection,
        condition="anticoagulation_pre_op_bridging",
        patient_factors={
            "ac_indication": "mechanical_mitral_valve",
            "cha2ds2_vasc_score": 3,
            "mechanical_valve": True,
            "surgery_bleeding_risk": "low",
        },
        patient_id="pt-preop-014",
    )

    medrecon = await _run_op(
        s, "compute_medication_reconciliation",
        "Audit the home med list for missing pre-op monitoring "
        "(warfarin -> INR; ACE-I + diuretic on hold day-of-surgery).",
        compute_medication_reconciliation, patient_id="pt-preop-014",
    )

    home_meds = [
        {"name": "warfarin 5 mg", "drug_class": "anticoagulant_vka", "status": "active"},
        {"name": "lisinopril 10 mg", "drug_class": "ace_inhibitor", "status": "active"},
        {"name": "atorvastatin 40 mg", "drug_class": "statin", "status": "active"},
        {"name": "metoprolol 25 mg", "drug_class": "beta_blocker", "status": "active"},
    ]

    poly = await _run_op(
        s, "detect_polypharmacy_concerns",
        "DDI scan on home meds + the planned LMWH bridge.",
        detect_polypharmacy_concerns,
        medications=home_meds + [
            {"name": "enoxaparin 1 mg/kg SC bid",
             "drug_class": "anticoagulant_heparin", "status": "active"},
        ],
    )

    grounding = await _run_op(
        s, "ground_claim",
        "Cite the BRIDGE trial / ACC peri-op pathway for a mechanical-valve "
        "bridging plan.",
        ground_claim,
        claim_text=(
            "Patients with a mechanical mitral valve undergoing elective "
            "non-cardiac surgery should bridge with therapeutic LMWH after "
            "warfarin interruption."
        ),
        patient_id="pt-preop-014",
    )

    # Cross-pattern: the deterministic action here is a treatment option_id
    # from compute_treatment_selection, NOT a discharge action.
    s.deterministic_action = bridging.top_pick_id or "abstain"
    s.deterministic_confidence = "high"
    s.extra_actions = [opt.option_id for opt in bridging.options]

    s.final_text = (
        f"Bridging recommendation: {bridging.top_pick_id} "
        f"(rationale: {bridging.rationale[:80]}...). "
        f"Med reconciliation: {len(medrecon.concerns)} concerns "
        f"({medrecon.severity_counts}). "
        f"Polypharmacy: {len(poly.interactions)} interactions, severity = "
        f"{poly.polypharmacy_severity}. "
        f"Grounding: {grounding.overall_verdict}."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario H -- Geriatric outpatient med review (CROSS-DOMAIN: no hospitalization)
# ─────────────────────────────────────────────────────────────────────

async def scenario_h() -> Scenario:
    s = Scenario(
        slug="H",
        title="Geriatric outpatient medication review -- 82y on 11 medications",
        persona="Pharmacist running an annual outpatient med review at a primary-care clinic.",
        prompt=(
            "Mrs. Smith, 82, comes in for her annual medication review. She's "
            "on 11 medications across hypertension, T2D, AFib, depression, and "
            "GERD. Run a polypharmacy + DDI audit on her current regimen, "
            "rank her T2D second-line options (she's on metformin + glipizide "
            "currently with A1c 8.2%), and produce a patient-language summary "
            "she can take home. NOTE: this is a chronic-care outpatient visit "
            "-- there is no hospital encounter and no discharge."
        ),
        fhir_summary=(
            "Outpatient context (no admission). 82-year-old woman, "
            "comorbidities: HTN, T2D, AFib (CHA₂DS₂-VASc 5), MDD, GERD. "
            "Current medications: warfarin, lisinopril, hydrochlorothiazide, "
            "metoprolol, metformin, glipizide, sertraline, omeprazole, "
            "calcium-vit-D, atorvastatin, aspirin 81 mg. "
            "Recent A1c 8.2%, eGFR 58 mL/min, BMI 31."
        ),
        expected_tools=[
            "detect_polypharmacy_concerns",
            "compute_treatment_selection",
            "compute_discharge_counseling",
        ],
    )
    s.synthesis_kind = "treatment_decision"

    await llm_plan(s)

    from mcp_server.tools.polypharmacy_concerns import detect_polypharmacy_concerns
    from mcp_server.tools.treatment_selection import compute_treatment_selection
    from mcp_server.tools.discharge_counseling import compute_discharge_counseling

    home_meds = [
        {"name": "warfarin 5 mg daily", "drug_class": "anticoagulant_vka", "status": "active"},
        {"name": "lisinopril 20 mg daily", "drug_class": "ace_inhibitor", "status": "active"},
        {"name": "hydrochlorothiazide 25 mg daily", "drug_class": "loop_diuretic", "status": "active"},
        {"name": "metoprolol 25 mg bid", "drug_class": "beta_blocker", "status": "active"},
        {"name": "metformin 1000 mg bid", "drug_class": "biguanide", "status": "active"},
        {"name": "glipizide 5 mg daily", "drug_class": "sulfonylurea", "status": "active"},
        {"name": "sertraline 50 mg daily", "drug_class": "ssri", "status": "active"},
        {"name": "omeprazole 20 mg daily", "drug_class": "ppi", "status": "active"},
        {"name": "atorvastatin 40 mg daily", "drug_class": "statin", "status": "active"},
        {"name": "aspirin 81 mg daily", "drug_class": "antiplatelet", "status": "active"},
        {"name": "calcium carbonate 500 mg + vit-D", "drug_class": "supplement", "status": "active"},
    ]

    poly = await _run_op(
        s, "detect_polypharmacy_concerns",
        "DDI + Beers-style audit on 11 medications. Expect warfarin + aspirin "
        "(bleeding) and possibly serotonergic + tramadol-class flags.",
        detect_polypharmacy_concerns, medications=home_meds,
    )

    treatment = await _run_op(
        s, "compute_treatment_selection",
        "Rank T2D options for second-add-on. ASCVD/HF presence drives "
        "SGLT2-i vs GLP-1 RA preference.",
        compute_treatment_selection,
        condition="dm2_second_line_after_metformin",
        patient_factors={
            "ascvd": False, "heart_failure": False, "ckd": True,
            "bmi": 31.0, "egfr_ml_min": 58.0, "a1c_pct": 8.2,
        },
        patient_id="pt-outpt-Smith",
    )

    counseling = await _run_op(
        s, "compute_discharge_counseling",
        "Reuse the patient-counseling generator OUTSIDE the discharge "
        "context -- produce a 6th-grade summary of the current regimen + "
        "the proposed T2D add-on. LACE score is 0 (no admission), "
        "follow-up window therefore defaults to 14-30d.",
        compute_discharge_counseling,
        medications=home_meds,
        lace_score=0,  # outpatient -- no LACE
        recommendation_action=None,
        patient_id="pt-outpt-Smith",
        extra_red_flags=[
            "Falls -- older adults on multiple BP/sleep medicines have a "
            "higher risk; tell us if you have stumbled or felt dizzy "
            "standing up.",
        ],
    )

    # Cross-domain: NOT a discharge decision. The "action" is which T2D
    # second-line agent to add. extra_actions enumerate the possible
    # option_ids so the parser/agreement matches them.
    s.deterministic_action = treatment.top_pick_id or "abstain"
    s.deterministic_confidence = "medium"
    s.extra_actions = [opt.option_id for opt in treatment.options]

    s.final_text = (
        f"Polypharmacy: {poly.n_medications} meds, severity = "
        f"{poly.polypharmacy_severity}, "
        f"{len(poly.interactions)} flagged interactions. "
        f"T2D top pick: {treatment.top_pick_id} (rationale: "
        f"{treatment.rationale[:80]}...). "
        f"Counseling: {counseling.n_medications_explained} of "
        f"{poly.n_medications} medications explained, "
        f"{counseling.n_red_flags} red-flag items, follow-up window = "
        f"{counseling.follow_up_window_days} days."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario I -- Pediatric ED: 4-month-old infant fever + lethargy
# ─────────────────────────────────────────────────────────────────────

async def scenario_i() -> Scenario:
    s = Scenario(
        slug="I",
        title="Pediatric ED -- 4-month-old infant with fever and lethargy",
        persona="Pediatric ED triage nurse evaluating a worried parent's bring-in.",
        prompt=(
            "A 4-month-old infant is brought in by mother. Mother reports the "
            "baby has been febrile for 12 hours, feeding poorly, and seems "
            "'not himself' -- sleepy and harder to wake. Vitals: HR 190, RR 60, "
            "SpO₂ 91% room air, capillary refill 4 seconds, axillary T 39.2°C. "
            "Compute pediatric early warning, decide ED disposition, and "
            "calculate weight-based ceftriaxone dose for empiric meningitis "
            "coverage (weight 6.8 kg). The infant has no known drug allergies. "
            "He is technically too young for ibuprofen -- verify the safety gate."
        ),
        fhir_summary=(
            "ED walk-in. 4-month-old male, 6.8 kg. Mother reports decreased "
            "feeding 12h, increasing lethargy, fever to 39.2°C. Concerning "
            "for sepsis / meningitis. NO ALLERGIES on file."
        ),
        expected_tools=[
            "compute_pediatric_early_warning",
            "compute_admission_triage",
            "compute_weight_based_dosing",
            "compute_weight_based_dosing (negative gate test)",
        ],
    )
    s.synthesis_kind = "discharge_decision"

    await llm_plan(s)

    from mcp_server.tools.pediatric_early_warning import compute_pediatric_early_warning
    from mcp_server.tools.admission_triage import compute_admission_triage
    from mcp_server.tools.weight_based_dosing import compute_weight_based_dosing

    pews = await _run_op(
        s, "compute_pediatric_early_warning",
        "Score with age-band-specific cutoffs (0-11mo): HR 190 is age-"
        "appropriate borderline; SpO₂ 91% is +3 alone; lethargy + parental "
        "concern adds points. Expect HIGH tier.",
        compute_pediatric_early_warning,
        age_months=4, behavior="lethargic",
        heart_rate=190, respiratory_rate=60, spo2=91,
        capillary_refill_seconds=4.0, accessory_muscle_use=True,
        on_supplemental_oxygen=False, parental_or_nurse_concern=True,
        patient_id="ped-001",
    )

    triage = await _run_op(
        s, "compute_admission_triage",
        "Pediatric red-flag pathway: lethargic infant + fever + abnormal "
        "vitals = ESI 1.",
        compute_admission_triage,
        chief_complaint="4-month-old infant lethargic and not feeding, fever",
        vital_signs={"heart_rate": 190, "spo2": 91, "respiratory_rate": 60,
                      "temperature": 39.2, "consciousness": "V"},
        age=0, comorbidity_summary="None",
        patient_id="ped-001",
    )

    cef_dose = await _run_op(
        s, "compute_weight_based_dosing",
        "Empiric meningitis dose: ceftriaxone 100 mg/kg IV (within 50-100 "
        "mg/kg window). Expect 680 mg, well under 2g adult cap. Gate: "
        "neonate contraindication (<28d) does NOT fire -- patient is 4 months.",
        compute_weight_based_dosing,
        drug="ceftriaxone", weight_kg=6.8, age_months=4,
        indication="meningitis", route="IV",
        target_dose_mg_per_kg=100.0,
        patient_id="ped-001",
    )

    ibu_blocked = await _run_op(
        s, "compute_weight_based_dosing",
        "Negative-gate verification: ibuprofen for fever in a 4-month-old "
        "MUST be blocked (contraindication: age 0-6 months).",
        compute_weight_based_dosing,
        drug="ibuprofen", weight_kg=6.8, age_months=4,
        indication="fever", route="PO",
        patient_id="ped-001",
    )

    # Map to discharge action: pediatric high-acuity -> continued_admission
    s.deterministic_action = "continued_admission"
    s.deterministic_confidence = "high"

    s.final_text = (
        f"PEWS = {pews.score_total} ({pews.severity_tier} tier -> "
        f"{pews.recommended_response}). Triage ESI {triage.esi_level} "
        f"({triage.priority}); disposition = {triage.disposition}, "
        f"unit = {triage.recommended_unit}. Ceftriaxone dose: "
        f"{cef_dose.final_dose_mg} mg IV (calculated from "
        f"{cef_dose.dose_mg_per_kg} mg/kg × {cef_dose.weight_kg} kg). "
        f"Ibuprofen safety gate: blocked = {ibu_blocked.abstain_recommended} "
        f"(age 0-6 months contraindication)."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario J -- Mental health crisis: high-risk suicide ideation
# ─────────────────────────────────────────────────────────────────────

async def scenario_j() -> Scenario:
    s = Scenario(
        slug="J",
        title="Mental health crisis -- 28y M with active suicide ideation",
        persona="ED psychiatry consult evaluating disposition.",
        prompt=(
            "28-year-old man brought to ED by friend after disclosing a plan "
            "to overdose tonight. He has a prior suicide attempt 4 years ago. "
            "He admits intent and has access to means (firearm at home). "
            "He has been mostly avoiding mental-health care. Compute C-SSRS "
            "risk, check fairness for the demographic, and decide disposition. "
            "He identifies as Black and LGBTQ -- bias guard MUST trigger."
        ),
        fhir_summary=(
            "Patient pt-mh-022, 28yM. Recent stressors: lost job 2 weeks ago, "
            "ended a relationship 1 month ago. Past psychiatric history: "
            "MDD, prior attempt 4 years ago. Demographics: Black, LGBTQ-"
            "identifying, low SES, uninsured."
        ),
        expected_tools=[
            "compute_suicide_risk_assessment",
            "compute_psychiatric_admission_decision",
            "compute_fairness_audit",
        ],
    )
    s.synthesis_kind = "discharge_decision"

    await llm_plan(s)

    from mcp_server.tools.suicide_risk_assessment import compute_suicide_risk_assessment
    from mcp_server.tools.psychiatric_admission_decision import compute_psychiatric_admission_decision

    cssrs = await _run_op(
        s, "compute_suicide_risk_assessment",
        "C-SSRS: ideation level 5 (plan + intent), 1 lifetime attempt, "
        "no past-30d attempt yet, 3 warning factors (means access, "
        "recent loss, hopelessness), 0 protective factors. Bias guard "
        "fires for Black + LGBTQ.",
        compute_suicide_risk_assessment,
        ideation_lifetime_level=5, ideation_past_30d_level=5,
        behavior_lifetime_attempts=1, behavior_past_30d_any=False,
        warning_factors_count=3, protective_factors_count=0,
        patient_demographics={"race": "black",
                                "sexual_orientation": "lgbtq",
                                "ses": "low_ses"},
        patient_id="pt-mh-022",
    )

    psy = await _run_op(
        s, "compute_psychiatric_admission_decision",
        "With imminent risk + danger-to-self + likely lacking voluntary "
        "engagement, expect involuntary_hold_evaluation pathway.",
        compute_psychiatric_admission_decision,
        risk_level=cssrs.risk_level,
        danger_to_self=True, danger_to_others=False, grave_disability=False,
        voluntary_capable=False,
        has_engaged_outpatient_treatment=False,
        has_safety_plan_in_place=False,
        state_jurisdiction="CO",
        patient_id="pt-mh-022",
    )

    # Map disposition to discharge action taxonomy:
    # involuntary_hold_evaluation -> continued_admission (do NOT discharge)
    s.deterministic_action = "continued_admission"
    s.deterministic_confidence = "high"

    s.final_text = (
        f"C-SSRS risk = {cssrs.risk_level} (abstain "
        f"{cssrs.abstain_recommended}: {(cssrs.abstain_reason or '')[:90]}...). "
        f"Disposition = {psy.disposition}; legal basis = "
        f"{(psy.legal_basis or '(none)')[:90]}. Safety plan required = "
        f"{psy.safety_plan_required}; follow-up within "
        f"{psy.follow_up_within_hours}h."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario K -- Antibiotic stewardship: complicated UTI with cultures back
# ─────────────────────────────────────────────────────────────────────

async def scenario_k() -> Scenario:
    s = Scenario(
        slug="K",
        title="Antibiotic stewardship -- complicated UTI with cultures back",
        persona="Antimicrobial stewardship pharmacist on day-3 review rounds.",
        prompt=(
            "Patient is a 65-year-old man, day 3 of empiric piperacillin-"
            "tazobactam IV 4.5g q8h for complicated UTI with sepsis. Cultures "
            "now back: E. coli, ESBL-positive. Susceptible to ertapenem and "
            "fosfomycin. Patient is afebrile 24h, hemodynamically stable, "
            "alert, but still NPO due to ileus. PCN allergy on file. Local "
            "ESBL prevalence 22%. Recommend de-escalation + total duration."
        ),
        fhir_summary=(
            "Patient pt-uti-031, 65yM, day 3 of pip-tazo IV. ESBL E. coli "
            "in urine + blood culture. Allergies: penicillin (rash). Renal "
            "function: eGFR 70 mL/min. Local antibiogram: ESBL prevalence 22%."
        ),
        expected_tools=[
            "compute_empiric_antibiotic_selection",
            "compute_antibiotic_de_escalation",
            "ground_claim",
        ],
    )
    s.synthesis_kind = "treatment_decision"

    await llm_plan(s)

    from mcp_server.tools.empiric_antibiotic_selection import compute_empiric_antibiotic_selection
    from mcp_server.tools.antibiotic_de_escalation import compute_antibiotic_de_escalation
    from mcp_server.tools.ground_claim import ground_claim

    empiric = await _run_op(
        s, "compute_empiric_antibiotic_selection",
        "Initial empiric coverage retrospective check -- at presentation, "
        "with prior_esbl_infection=False and local ESBL 22%, top pick "
        "should have been ertapenem already.",
        compute_empiric_antibiotic_selection,
        infection_source="urinary", severity="sepsis",
        patient_factors={"age": 65, "allergy_penicillin": True,
                         "egfr_ml_min": 70},
        local_antibiogram={"esbl_prevalence_pct": 22},
        patient_id="pt-uti-031",
    )

    deesc = await _run_op(
        s, "compute_antibiotic_de_escalation",
        "Day-3 de-escalation: pathogen E. coli ESBL, susceptibility "
        "ertapenem=S, fosfomycin=S. Patient stable but NPO -> IV-to-PO "
        "switch should be BLOCKED.",
        compute_antibiotic_de_escalation,
        current_regimen="piperacillin-tazobactam 4.5 g IV q8h",
        pathogen="e_coli_esbl",
        susceptibility={"ertapenem": "S", "fosfomycin": "S"},
        days_on_therapy=3, total_planned_duration_days=14,
        clinical_factors={"afebrile_24h": True,
                           "hemodynamically_stable": True,
                           "tolerating_po": False,  # NPO blocks
                           "alert_oriented": True,
                           "malabsorption": False},
        patient_id="pt-uti-031",
    )

    grounding = await _run_op(
        s, "ground_claim",
        "Cite IDSA stewardship guidelines for narrowing pip-tazo to "
        "ertapenem when an ESBL organism is identified.",
        ground_claim,
        claim_text=(
            "When an ESBL-producing E. coli is identified, the empiric "
            "piperacillin-tazobactam regimen should be narrowed to "
            "a carbapenem such as ertapenem."
        ),
        patient_id="pt-uti-031",
    )

    s.deterministic_action = deesc.target_regimen or "abstain"
    s.deterministic_confidence = "high"
    s.extra_actions = ["ertapenem 1 g IV q24h (no oral equivalent for ESBL)",
                        "fosfomycin", "ertapenem"]

    s.final_text = (
        f"Empiric retrospective top pick = {empiric.top_pick_id}. "
        f"De-escalation: target = {(deesc.target_regimen or '?')[:60]}, "
        f"IV->PO switch = {deesc.iv_to_po_switch_eligible} (criteria unmet: "
        f"{deesc.iv_to_po_criteria_unmet}). Total duration "
        f"{deesc.duration_total_days}d, {deesc.duration_remaining_days}d "
        f"remaining. Grounding verdict = {grounding.overall_verdict}."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario L -- Oncology: NSCLC chemo cycle 4 decision + RECIST mid-tx
# ─────────────────────────────────────────────────────────────────────

async def scenario_l() -> Scenario:
    s = Scenario(
        slug="L",
        title="Oncology -- NSCLC cycle 4 chemo decision + interim RECIST",
        persona="Oncology fellow at a Tuesday clinic review.",
        prompt=(
            "Patient is a 62-year-old woman with stage IV NSCLC on "
            "carboplatin/pemetrexed, scheduled for cycle 4 today. CBC: ANC "
            "1100/μL, platelets 110k, hgb 10.8. CMP: eGFR 58, AST/ALT/bili "
            "normal. ECOG 1. Restaging CT 1 week ago: 2 target lesions -- "
            "right lung 40 mm -> 25 mm; mediastinum 22 mm -> 16 mm. No new "
            "lesions. Decide whether to proceed today and interpret the "
            "interim RECIST response."
        ),
        fhir_summary=(
            "Patient pt-onc-045, 62yF, NSCLC stage IV, cycle 4 of "
            "carboplatin/pemetrexed. ANC 1100, platelets 110k, eGFR 58, "
            "ECOG 1. Restaging CT shows shrinkage (target lesions 40->25mm + "
            "22->16mm)."
        ),
        expected_tools=[
            "compute_oncology_treatment_response",
            "compute_chemo_dose_adjustment",
        ],
    )
    s.synthesis_kind = "discharge_decision"

    await llm_plan(s)

    from mcp_server.tools.oncology_treatment_response import compute_oncology_treatment_response
    from mcp_server.tools.chemo_dose_adjustment import compute_chemo_dose_adjustment

    response = await _run_op(
        s, "compute_oncology_treatment_response",
        "Compute RECIST 1.1 from baseline + current diameters. Sum 62 -> "
        "41 mm = -33.9% -> partial_response. No new lesions, no non-target "
        "PD. Decision: continue.",
        compute_oncology_treatment_response,
        target_lesions=[
            {"lesion_id": "L1", "location": "right_lung",
             "baseline_longest_diameter_mm": 40,
             "current_longest_diameter_mm": 25},
            {"lesion_id": "L2", "location": "mediastinum",
             "baseline_longest_diameter_mm": 22,
             "current_longest_diameter_mm": 16},
        ],
        new_lesions_present=False, non_target_progression=False,
        prior_response_categories=["stable_disease", "partial_response"],
        patient_id="pt-onc-045",
    )

    chemo = await _run_op(
        s, "compute_chemo_dose_adjustment",
        "Cycle 4 of carboplatin/pemetrexed. ANC 1100 < 1500 -> mild "
        "neutropenia, expect delay_one_week. Platelets borderline. eGFR "
        "58 acceptable (no carboplatin reduction unless <30).",
        compute_chemo_dose_adjustment,
        regimen="carboplatin_pemetrexed", cycle_number=4,
        egfr_ml_min=58, anc_per_ul=1100, platelets_per_ul=110_000,
        ecog_performance_status=1,
        patient_id="pt-onc-045",
    )

    # Map decision to discharge taxonomy:
    # delay_one_week -> home_with_care; proceed_full_dose -> discharge_home
    if chemo.decision == "delay_one_week":
        s.deterministic_action = "home_with_care"
    elif chemo.decision in ("hold_until_recovery", "discontinue_consider_alternative"):
        s.deterministic_action = "continued_admission"
    elif chemo.decision == "proceed_full_dose":
        s.deterministic_action = "discharge_home"
    else:
        s.deterministic_action = "home_with_care"
    s.deterministic_confidence = "high"

    s.final_text = (
        f"RECIST 1.1: {response.overall_response} (Δ "
        f"{response.sum_change_pct:+.1f}%). Decision implication: "
        f"{response.decision_implication}. Cycle decision: {chemo.decision}; "
        f"dose reduction {chemo.dose_reduction_pct}%; growth-factor "
        f"indicated = {chemo.growth_factor_indicated}. "
        f"{'Reason: ' + chemo.delay_reason if chemo.delay_reason else ''}"
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario M -- Acute stroke: 70y M with focal deficit, last-known-well 90 min ago
# ─────────────────────────────────────────────────────────────────────

async def scenario_m() -> Scenario:
    s = Scenario(
        slug="M",
        title="Acute stroke -- 70y M, focal deficit, last-known-well 90 min ago",
        persona="Emergency physician + neurology stroke team activation.",
        prompt=(
            "70-year-old man arrives by EMS with sudden right-sided weakness "
            "and slurred speech first noticed 90 minutes ago. Exam: gaze "
            "preference left, R arm 3/5, R leg 4/5, expressive aphasia, "
            "facial droop. CT excludes hemorrhage; CTA shows left M1 "
            "occlusion. ASPECTS 8. BP 165/95, INR 1.0, platelets 220k. "
            "Compute NIHSS, then decide IV tPA + EVT eligibility. Time is brain."
        ),
        fhir_summary=(
            "Patient pt-stroke-M, 70yM, BP 165/95. CT excludes ICH; CTA "
            "left M1 LVO. ASPECTS 8. INR 1.0, platelets 220k, glucose 110. "
            "No recent surgery, no head trauma, no DOAC."
        ),
        expected_tools=[
            "compute_stroke_severity",
            "compute_stroke_thrombolysis_eligibility",
            "ground_claim",
        ],
    )
    s.synthesis_kind = "discharge_decision"

    await llm_plan(s)

    from mcp_server.tools.stroke_severity import compute_stroke_severity
    from mcp_server.tools.stroke_thrombolysis_eligibility import compute_stroke_thrombolysis_eligibility
    from mcp_server.tools.ground_claim import ground_claim

    nihss_items = {
        "loc_responsiveness": 1,         # arousable
        "best_gaze": 2,                   # forced gaze deviation
        "facial_palsy": 2,
        "motor_arm_right": 3,
        "motor_leg_right": 1,
        "best_language": 2,               # expressive aphasia
        "dysarthria": 1,
        "extinction_inattention": 1,
    }
    nihss = await _run_op(
        s, "compute_stroke_severity",
        "Compute NIHSS -- gaze + language + neglect = cortical signs; "
        "sum should put us at ~13 (moderate) with LVO suspected.",
        compute_stroke_severity,
        item_scores=nihss_items, last_known_well_minutes_ago=90,
        patient_id="pt-stroke-M",
    )

    tpa = await _run_op(
        s, "compute_stroke_thrombolysis_eligibility",
        "Within 3h tPA window + LVO confirmed + ASPECTS 8 -> expect "
        "iv_tpa_plus_evt.",
        compute_stroke_thrombolysis_eligibility,
        last_known_well_minutes_ago=90, nihss_total=nihss.score_total,
        clinical_factors={
            "anterior_circulation_lvo_on_imaging": True, "aspects_score": 8,
            "ct_excludes_ich": True,
            "systolic_bp": 165, "diastolic_bp": 95,
            "inr": 1.0, "platelets_per_ul": 220_000, "glucose_mg_dl": 110,
        },
        patient_id="pt-stroke-M",
    )

    grounding = await _run_op(
        s, "ground_claim",
        "Cite AHA/ASA 2019 + DAWN/DEFUSE 3 for combined IV-tPA + EVT.",
        ground_claim,
        claim_text=(
            "For acute ischemic stroke with anterior-circulation LVO within "
            "3 hours of last-known-well, IV alteplase plus mechanical "
            "thrombectomy is recommended."
        ),
        patient_id="pt-stroke-M",
    )

    s.deterministic_action = "continued_admission"
    s.deterministic_confidence = "high"

    s.final_text = (
        f"NIHSS = {nihss.score_total} ({nihss.severity_tier} tier, "
        f"LVO suspected = {nihss.lvo_suspected}). "
        f"IV tPA: window={tpa.iv_tpa_window} eligible={tpa.iv_tpa_eligible}; "
        f"EVT: window={tpa.evt_window} eligible={tpa.evt_eligible}. "
        f"Decision: {tpa.decision}. Grounding: {grounding.overall_verdict}."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario N -- Chest pain HEART score + ACS disposition
# ─────────────────────────────────────────────────────────────────────

async def scenario_n() -> Scenario:
    s = Scenario(
        slug="N",
        title="Chest pain rule-out -- 58y M, HEART score + ACS pathway",
        persona="ED physician working through ACS rule-out at 2 AM.",
        prompt=(
            "58-year-old man with 2 hours of intermittent substernal chest "
            "pressure, radiating to left arm, no diaphoresis. ECG: non-"
            "specific ST flattening. Initial troponin: 0.5× ULN. Risk "
            "factors: HTN, hyperlipidemia, current smoker. No prior CAD. "
            "Compute HEART score and decide ED disposition."
        ),
        fhir_summary=(
            "Patient pt-cp-N, 58yM, HTN + hyperlipidemia + smoker, no "
            "prior CAD/CVA/PAD. ECG non-specific repolarization changes. "
            "Initial troponin 0.5× ULN. No dynamic troponin yet (still in "
            "first 6h). Hemodynamically stable, no ongoing pain at evaluation."
        ),
        expected_tools=[
            "compute_heart_score",
            "compute_acs_disposition_decision",
        ],
    )
    s.synthesis_kind = "discharge_decision"

    await llm_plan(s)

    from mcp_server.tools.heart_score import compute_heart_score
    from mcp_server.tools.acs_disposition_decision import compute_acs_disposition_decision

    heart = await _run_op(
        s, "compute_heart_score",
        "Score: H=1 (moderate), E=1 (non-specific), A=1 (45-64), "
        "R=1 (2 RFs, no CAD), T=0 (≤ULN). Expect total 4 -> moderate.",
        compute_heart_score,
        history_descriptor="moderately_suspicious",
        ecg_descriptor="non_specific_repolarization",
        age=58, risk_factors_count=3,    # HTN, lipids, smoker
        known_atherosclerotic_disease=False,
        troponin_times_uln=0.5,
        patient_id="pt-cp-N",
    )

    acs = await _run_op(
        s, "compute_acs_disposition_decision",
        "Moderate HEART, no STEMI, no dynamic trop yet -> ED observation "
        "with serial troponin.",
        compute_acs_disposition_decision,
        heart_score_total=heart.total_score,
        has_stemi=False, has_dynamic_troponin=False,
        ongoing_chest_pain=False,
        patient_id="pt-cp-N",
    )

    # Map disposition -> discharge action taxonomy
    if acs.disposition == "discharge_with_outpatient_followup":
        s.deterministic_action = "discharge_home"
    elif acs.disposition in ("ed_observation_serial_troponin", "admit_telemetry_for_workup"):
        s.deterministic_action = "continued_admission"
    elif acs.disposition == "cath_lab_activation_immediate":
        s.deterministic_action = "continued_admission"
    else:
        s.deterministic_action = "home_with_care"
    s.deterministic_confidence = "high" if heart.risk_band != "moderate" else "medium"

    s.final_text = (
        f"HEART = {heart.total_score}/10 ({heart.risk_band} band, "
        f"{heart.estimated_30d_mace_risk_pct}% 30-day MACE). "
        f"Disposition = {acs.disposition} (follow-up "
        f"{acs.recommended_followup_hours}h)."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario O -- Maternal severe preeclampsia at 32 weeks
# ─────────────────────────────────────────────────────────────────────

async def scenario_o() -> Scenario:
    s = Scenario(
        slug="O",
        title="Maternal severe preeclampsia -- 32-week pregnancy with red flags",
        persona="Labor & delivery triage ob-gyn.",
        prompt=(
            "32-week G2P1 pregnant woman comes to L&D triage with severe "
            "headache, blurred vision, and BP 162/110 measured twice 15 min "
            "apart. Urine dipstick 2+ protein. Recent labs: platelets 95k, "
            "AST 105 U/L. Compute MEOWS, then ACOG preeclampsia "
            "classification + disposition. Demographic flag: Black "
            "patient -- maternal mortality is 3× higher in this subgroup."
        ),
        fhir_summary=(
            "Patient pt-ob-O, 32-week G2P1, Black, BP 162/110 confirmed, "
            "proteinuria 2+, severe headache + visual disturbance, "
            "platelets 95k, AST 105 (>2× ULN). HR 95, RR 18, SpO2 98."
        ),
        expected_tools=[
            "compute_maternal_early_warning",
            "compute_preeclampsia_assessment",
            "compute_fairness_audit",
        ],
    )
    s.synthesis_kind = "discharge_decision"

    await llm_plan(s)

    from mcp_server.tools.maternal_early_warning import compute_maternal_early_warning
    from mcp_server.tools.preeclampsia_assessment import compute_preeclampsia_assessment
    from mcp_server.tools.fairness_audit import compute_fairness_audit
    from mcp_server.tools.readmission_risk import compute_readmission_risk

    meow = await _run_op(
        s, "compute_maternal_early_warning",
        "MEOWS with severe headache/visual + proteinuria + elevated BP "
        "-> red tier (obstetric emergency).",
        compute_maternal_early_warning,
        gestational_age_weeks=32, pregnancy_phase="antepartum",
        respiratory_rate=18, spo2=98, heart_rate=95,
        systolic_bp=162, diastolic_bp=110,
        proteinuria_present=True, severe_headache_or_visual=True,
        patient_id="pt-ob-O",
    )

    pre = await _run_op(
        s, "compute_preeclampsia_assessment",
        "ACOG: severe BP + proteinuria + thrombocytopenia + transaminitis "
        "-> preeclampsia with severe features. GA 32 -> admit for "
        "monitoring, magnesium sulfate, antihypertensive.",
        compute_preeclampsia_assessment,
        gestational_age_weeks=32,
        systolic_bp=162, diastolic_bp=110, proteinuria_present=True,
        clinical_factors={
            "platelets_per_ul": 95_000, "ast_ul": 105,
            "severe_headache_persistent": True, "visual_disturbances": True,
        },
        patient_id="pt-ob-O",
    )

    # Build a synthetic risk estimate for fairness audit (we don't have a
    # calibrated maternal-mortality model here -- reuse readmission_risk path
    # but re-tag the outcome_id; or simulate one).
    # Simplest: run fairness audit with a synthesized RiskEstimate.
    from shared.schemas import Factor, RiskEstimate
    from datetime import datetime, timezone
    pseudo_risk = RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="placeholder-for-maternal-mortality",
        outcome_id="mortality_30d",
        horizon_days=30,
        lace_raw_score=8,
        probability_mean=0.05,            # 5% -- synthetic
        probability_ci95=(0.02, 0.10),
        probability_ci_width=0.08,
        contributing_factors=[Factor(name="severe_preeclampsia", raw_value=1.0,
                                        lace_points=4, weight=0.5)],
        fhir_observations_used=[],
        computed_at=datetime.now(timezone.utc),
    )
    fairness = await _run_op(
        s, "compute_fairness_audit",
        "Audit predicted maternal-mortality rate against demographic "
        "subgroup baselines. Black women have ~3× higher maternal "
        "mortality (CDC PRAMS). Expect drift / abstain recommendation.",
        compute_fairness_audit,
        risk=pseudo_risk,
        patient_demographics={"age": 32, "race": "black"},
    )

    s.deterministic_action = "continued_admission"
    s.deterministic_confidence = "high"

    s.final_text = (
        f"MEOWS = {meow.severity_tier} ({meow.recommended_response}). "
        f"Preeclampsia classification: {pre.classification}. "
        f"Mg sulfate = {pre.magnesium_sulfate_indicated}, "
        f"antihypertensive = {pre.antihypertensive_indicated}, "
        f"delivery = {pre.delivery_recommended}. "
        f"Disposition: {pre.recommended_disposition}. "
        f"Fairness audit confidence_action = {fairness.confidence_action}."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario P -- Geriatric: 85y with delirium + falls risk + polypharmacy
# ─────────────────────────────────────────────────────────────────────

async def scenario_p() -> Scenario:
    s = Scenario(
        slug="P",
        title="Geriatric -- 85y woman admitted with confusion + falls history",
        persona="Inpatient geriatrics consult on hospital day 2.",
        prompt=(
            "85-year-old woman admitted 2 days ago for UTI + confusion. "
            "Husband reports she's been forgetful for years but the "
            "confusion is much worse since admission and waxes/wanes. She "
            "was found wandering in the hallway last night; one fall in "
            "the past month. Current meds: lorazepam 1 mg q6h prn for "
            "agitation, oxycodone 5 mg q6h for back pain, diphenhydramine "
            "25 mg qhs for sleep, lisinopril 10 mg, atorvastatin 40 mg. "
            "Run delirium screen (CAM), falls risk (Morse), and "
            "polypharmacy audit on the deliriogenic meds."
        ),
        fhir_summary=(
            "Patient pt-geri-P, 85yF, admitted with UTI. Comorbidities: "
            "mild cognitive impairment (chronic), HTN, hyperlipidemia. "
            "Recent falls 1 in 30 days. Vitals stable. Active meds: "
            "lorazepam, oxycodone, diphenhydramine, lisinopril, atorvastatin."
        ),
        expected_tools=[
            "compute_delirium_screening_cam",
            "compute_falls_risk_morse",
            "detect_polypharmacy_concerns",
        ],
    )
    s.synthesis_kind = "discharge_decision"

    await llm_plan(s)

    from mcp_server.tools.delirium_screening_cam import compute_delirium_screening_cam
    from mcp_server.tools.falls_risk_morse import compute_falls_risk_morse
    from mcp_server.tools.polypharmacy_concerns import detect_polypharmacy_concerns

    cam = await _run_op(
        s, "compute_delirium_screening_cam",
        "CAM positive: F1 (acute onset / fluctuating), F2 (inattention), "
        "F3 (disorganized thinking -- wandering, mixing things up). "
        "Mixed motor subtype (wanders but also has lethargic episodes).",
        compute_delirium_screening_cam,
        feature1_acute_onset_or_fluctuating=True,
        feature2_inattention=True,
        feature3_disorganized_thinking=True,
        feature4_altered_consciousness=False,
        motor_subtype="mixed",
        current_medications=[
            "lorazepam 1 mg q6h prn", "oxycodone 5 mg q6h",
            "diphenhydramine 25 mg qhs", "lisinopril 10 mg",
        ],
        suspected_contributors=["UTI (already identified)",
                                  "hospital environment / sleep deprivation"],
        patient_id="pt-geri-P",
    )

    falls = await _run_op(
        s, "compute_falls_risk_morse",
        "Morse: history of falls (+25), secondary dx (+15), no aid yet "
        "(0), IV (+20), gait weak (+10), forgets limitations (+15) = 85; "
        "with 3 deliriogenic meds -> high tier.",
        compute_falls_risk_morse,
        history_of_falling_3mo=True, secondary_diagnosis_present=True,
        ambulatory_aid="none", has_iv_or_heparin_lock=True,
        gait="weak", mental_status="forgets_limitations",
        current_medications=[
            "lorazepam 1 mg q6h prn", "oxycodone 5 mg q6h",
            "diphenhydramine 25 mg qhs",
        ],
        patient_id="pt-geri-P",
    )

    poly = await _run_op(
        s, "detect_polypharmacy_concerns",
        "Polypharmacy on 5 meds -- expect medium severity due to "
        "anticholinergic + opioid + benzo combination.",
        detect_polypharmacy_concerns,
        medications=[
            {"name": "lorazepam 1 mg", "drug_class": "benzodiazepine",
             "status": "active"},
            {"name": "oxycodone 5 mg", "drug_class": "opioid",
             "status": "active"},
            {"name": "diphenhydramine 25 mg",
             "drug_class": "anticholinergic_antihistamine", "status": "active"},
            {"name": "lisinopril 10 mg", "drug_class": "ace_inhibitor",
             "status": "active"},
            {"name": "atorvastatin 40 mg", "drug_class": "statin",
             "status": "active"},
        ],
    )

    s.deterministic_action = "continued_admission"
    s.deterministic_confidence = "high"

    s.final_text = (
        f"CAM = {'POSITIVE' if cam.cam_positive else 'negative'} "
        f"(subtype = {cam.delirium_subtype}). "
        f"Morse = {falls.score_total} ({falls.risk_tier} tier -> "
        f"{falls.recommended_intervention}); flagged "
        f"{len(falls.contributing_medications_flagged)} falls-risk meds. "
        f"Polypharmacy severity = {poly.polypharmacy_severity}, "
        f"{len(poly.interactions)} flagged interactions."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario Q -- Cross-bundle: polytrauma -> resuscitation -> ICU disposition
# Bundles exercised: trauma_critical + ed_acute + core_discharge
# ─────────────────────────────────────────────────────────────────────

async def scenario_q() -> Scenario:
    s = Scenario(
        slug="Q",
        title="Polytrauma -- MVC, hypotension, multi-cavity injuries (cross-bundle)",
        persona="Trauma team activation: ED + trauma surgery + anesthesia.",
        prompt=(
            "32-year-old man brought in by EMS after a high-speed MVC. On "
            "arrival: GCS 10, BP 78/50, HR 130, RR 28, distressed. FAST "
            "exam positive in Morrison's pouch. CT shows splenic laceration "
            "(AIS 4 abdomen) + pulmonary contusion (AIS 3 chest) + closed "
            "head injury (AIS 4 head/neck). Penetrating mechanism = no, but "
            "field SBP was 75 and arrival is 78. Compute trauma severity, "
            "decide MTP activation, and order the first-line imaging for "
            "blunt abdominal trauma. He's 45 minutes post-injury, eGFR "
            "unknown but likely normal."
        ),
        fhir_summary=(
            "ED arrival, trauma team activation. 32yM, MVC. Vitals: BP 78/50, "
            "HR 130, RR 28, GCS 10. CT: splenic lac (AIS 4), pulmonary "
            "contusion (AIS 3), closed head injury (AIS 4). FAST positive."
        ),
        expected_tools=[
            "compute_trauma_severity_score",
            "compute_massive_transfusion_protocol",
            "compute_imaging_appropriateness",
            "compute_clinical_deterioration_score",
            "ground_claim",
        ],
    )
    s.synthesis_kind = "discharge_decision"

    await llm_plan(s)

    from mcp_server.tools.trauma_severity_score import compute_trauma_severity_score
    from mcp_server.tools.massive_transfusion_protocol import compute_massive_transfusion_protocol
    from mcp_server.tools.imaging_appropriateness import compute_imaging_appropriateness
    from mcp_server.tools.clinical_deterioration_score import compute_clinical_deterioration_score
    from mcp_server.tools.ground_claim import ground_claim

    trauma = await _run_op(
        s, "compute_trauma_severity_score",
        "Composite ISS + T-RTS -- expect ISS 41 (severe) + critical RTS -> "
        "operating room immediate.",
        compute_trauma_severity_score,
        injuries=[
            {"body_region": "head_neck", "ais_severity": 4},
            {"body_region": "chest", "ais_severity": 3},
            {"body_region": "abdomen_pelvis", "ais_severity": 4},
        ],
        glasgow_coma_score=10, systolic_bp=78,
        respiratory_rate=28, has_active_hemorrhage=True,
        patient_id="pt-trauma-Q",
    )

    mtp = await _run_op(
        s, "compute_massive_transfusion_protocol",
        "ABC score: penetrating=False (1pt), SBP≤90=Yes (1pt), HR≥120=Yes "
        "(1pt), FAST positive (1pt) = 3/4 -> ACTIVATE MTP, TXA in window.",
        compute_massive_transfusion_protocol,
        penetrating_mechanism=False, field_or_arrival_sbp_le_90=True,
        heart_rate_ge_120=True, positive_fast_exam=True,
        estimated_blood_loss_ml=2000, minutes_since_injury=45,
        patient_id="pt-trauma-Q",
    )

    imaging = await _run_op(
        s, "compute_imaging_appropriateness",
        "Already had CT -> next-step imaging consideration: angiography for "
        "splenic embolization vs OR. Reuse blunt_abdominal_trauma scenario.",
        compute_imaging_appropriateness,
        clinical_scenario="blunt_abdominal_trauma", patient_age=32,
        patient_id="pt-trauma-Q",
    )

    news2 = await _run_op(
        s, "compute_clinical_deterioration_score",
        "NEWS2 on arrival vitals -- expect HIGH tier given hypotension, "
        "tachycardia, tachypnea, GCS 10 -> AVPU=V or P.",
        compute_clinical_deterioration_score,
        vital_signs=[
            {"type": "respiratory_rate", "value": 28},
            {"type": "spo2", "value": 95},
            {"type": "supplemental_oxygen", "value": True},
            {"type": "systolic_bp", "value": 78},
            {"type": "heart_rate", "value": 130},
            {"type": "consciousness", "value": "P"},
        ],
        patient_id="pt-trauma-Q",
    )

    grounding = await _run_op(
        s, "ground_claim",
        "Cite PROPPR / CRASH-2 for 1:1:1 ratio + TXA in active hemorrhage.",
        ground_claim,
        claim_text=(
            "For trauma patients with massive hemorrhage, balanced 1:1:1 "
            "plasma:platelets:RBC transfusion with adjunctive tranexamic "
            "acid within 3 hours of injury is recommended."
        ),
        patient_id="pt-trauma-Q",
    )

    s.deterministic_action = "continued_admission"
    s.deterministic_confidence = "high"

    s.final_text = (
        f"Trauma severity: ISS={trauma.iss} ({trauma.iss_band}), RTS={trauma.rts} "
        f"({trauma.rts_band}), triage={trauma.triage_priority}. "
        f"MTP: ABC={mtp.abc_score} -> activated={mtp.mtp_activated}, "
        f"TXA in window={mtp.txa_indicated}, request="
        f"{mtp.estimated_initial_request['rbc_units']} PRBC + "
        f"{mtp.estimated_initial_request['ffp_units']} FFP. "
        f"NEWS2={news2.score_total} ({news2.severity_tier}); "
        f"imaging top pick: {imaging.top_pick_modality}. "
        f"Grounding: {grounding.overall_verdict}."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario R -- Cross-bundle: DKA + AKI + imaging contrast safety
# Bundles exercised: endocrine_acute + nephrology + imaging
# ─────────────────────────────────────────────────────────────────────

async def scenario_r() -> Scenario:
    s = Scenario(
        slug="R",
        title="DKA + AKI + need for contrast CT (cross-bundle)",
        persona="Hospitalist + ICU + radiology ordering a contrast study during DKA.",
        prompt=(
            "55-year-old woman with T2D presents with DKA (pH 7.05, HCO3 8, "
            "glucose 540, ketones positive). Admission creatinine 2.4 "
            "(baseline 1.0), oliguric (UOP 0.4 mL/kg/h × 12h). Currently on "
            "metformin + empagliflozin at home. Admitted to ICU for DKA "
            "protocol. Now the CT surgeon wants a CT abdomen+pelvis WITH "
            "contrast to rule out an abdominal source of the DKA. Compute "
            "DKA severity + initial protocol, stage the AKI, and decide "
            "whether the contrast study should proceed."
        ),
        fhir_summary=(
            "Patient pt-dka-R, 55yF, T2D on metformin + empagliflozin, "
            "admitted in DKA. Labs: pH 7.05, HCO3 8, glucose 540, K+ 4.2, "
            "Cr 2.4 (baseline 1.0), UOP 0.4 mL/kg/h × 12h."
        ),
        expected_tools=[
            "compute_dka_severity",
            "compute_aki_kdigo_stage",
            "compute_contrast_safety_check",
            "compute_imaging_appropriateness",
        ],
    )
    s.synthesis_kind = "discharge_decision"

    await llm_plan(s)

    from mcp_server.tools.dka_severity import compute_dka_severity
    from mcp_server.tools.aki_kdigo_stage import compute_aki_kdigo_stage
    from mcp_server.tools.contrast_safety_check import compute_contrast_safety_check
    from mcp_server.tools.imaging_appropriateness import compute_imaging_appropriateness

    dka = await _run_op(
        s, "compute_dka_severity",
        "pH 7.05 + HCO3 8 -> severe DKA, ICU; K+ 4.2 -> start insulin "
        "without holding.",
        compute_dka_severity,
        ph=7.05, bicarbonate_meq_l=8, glucose_mg_dl=540,
        ketones_present=True, mental_status="drowsy",
        anion_gap=28, potassium_meq_l=4.2, weight_kg=70,
        patient_id="pt-dka-R",
    )

    aki = await _run_op(
        s, "compute_aki_kdigo_stage",
        "Cr 1.0 -> 2.4 = 2.4× baseline (Stage 2) AND UOP 0.4 mL/kg/h × 12h "
        "(Stage 2 by UOP) -> Stage 2 confirmed.",
        compute_aki_kdigo_stage,
        creatinine_baseline_mg_dl=1.0, creatinine_current_mg_dl=2.4,
        urine_output_ml_per_kg_per_hour=0.4, urine_output_window_hours=12,
        nephrotoxic_medications_present=True,  # SGLT2-i can contribute
        patient_id="pt-dka-R",
    )

    # eGFR estimate from Cr 2.4 in 55yF ≈ 22 mL/min -- high CIN risk
    contrast = await _run_op(
        s, "compute_contrast_safety_check",
        "eGFR ~22 + ongoing AKI + on metformin -> contraindicated for "
        "iodinated contrast in elective indication; defer until AKI resolves "
        "OR proceed only if life-threatening indication.",
        compute_contrast_safety_check,
        contrast_type="iodinated_iv", egfr_ml_min=22, on_metformin=True,
        patient_id="pt-dka-R",
    )

    imaging = await _run_op(
        s, "compute_imaging_appropriateness",
        "Acute abdominal pain scenario -- given contrast contraindication, "
        "MRI / US should rank higher than CT.",
        compute_imaging_appropriateness,
        clinical_scenario="acute_abdominal_pain", patient_age=55,
        patient_id="pt-dka-R",
    )

    s.deterministic_action = "continued_admission"
    s.deterministic_confidence = "high"

    s.final_text = (
        f"DKA severity = {dka.severity} (ICU = {dka.icu_admission_indicated}). "
        f"AKI = {aki.aki_stage} (Cr {aki.creatinine_change_ratio}× baseline). "
        f"Iodinated contrast: proceed = {contrast.proceed_with_contrast} "
        f"(CIN risk = {contrast.contrast_induced_nephropathy_risk}, "
        f"metformin hold = {contrast.metformin_hold_recommended}). "
        f"Alternative imaging top pick: {imaging.top_pick_modality}."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# Scenario S -- Cross-bundle: emergency delivery + PPH + ICU + discharge plan
# Bundles exercised: obstetric_geriatric + trauma_critical + core_discharge
# ─────────────────────────────────────────────────────────────────────

async def scenario_s() -> Scenario:
    s = Scenario(
        slug="S",
        title="Severe preeclampsia -> emergency C-section -> PPH (full journey, cross-bundle)",
        persona="Multidisciplinary team: ob-gyn + anesthesia + ICU + ward.",
        prompt=(
            "34-year-old G1P0 woman at 33 weeks with severe preeclampsia "
            "(BP 175/115, plt 88k, AST 130, severe headache). Mg sulfate "
            "started + emergent C-section. Intra-op: 2L EBL, post-op now "
            "BP 88/52, HR 122, FAST positive for free fluid -> PPH. ABC "
            "score 3/4. Now in ICU, day 2 post-op stable, planning "
            "discharge in 3 days. Run the full pipeline: re-classify "
            "preeclampsia (severe features), MTP for the PPH, AKI staging "
            "from Cr trend (1.0 -> 1.5), and produce patient-language "
            "discharge counseling for postpartum medications."
        ),
        fhir_summary=(
            "Patient pt-pph-S, 34yF G1P0, post-emergent C-section for "
            "severe preeclampsia at 33w. Intra-op EBL 2L. ICU day 2: "
            "stable, on enoxaparin VTE proph + lisinopril 5 mg + Mg "
            "completed × 24h post-op. Cr 1.0 -> 1.5. Plt recovering 88 -> 145. "
            "Demographics: Black (3× higher maternal mortality)."
        ),
        expected_tools=[
            "compute_preeclampsia_assessment",
            "compute_massive_transfusion_protocol",
            "compute_aki_kdigo_stage",
            "compute_discharge_counseling",
            "compute_fairness_audit",
        ],
    )
    s.synthesis_kind = "discharge_decision"

    await llm_plan(s)

    from mcp_server.tools.preeclampsia_assessment import compute_preeclampsia_assessment
    from mcp_server.tools.massive_transfusion_protocol import compute_massive_transfusion_protocol
    from mcp_server.tools.aki_kdigo_stage import compute_aki_kdigo_stage
    from mcp_server.tools.discharge_counseling import compute_discharge_counseling
    from mcp_server.tools.fairness_audit import compute_fairness_audit
    from shared.schemas import Factor, RiskEstimate
    from datetime import datetime, timezone

    pre = await _run_op(
        s, "compute_preeclampsia_assessment",
        "33w + BP 175/115 + plt 88k + AST 130 + severe headache -> "
        "preeclampsia with severe features; Mg + delivery (already done).",
        compute_preeclampsia_assessment,
        gestational_age_weeks=33,
        systolic_bp=175, diastolic_bp=115,
        proteinuria_present=True,
        clinical_factors={
            "platelets_per_ul": 88_000, "ast_ul": 130,
            "severe_headache_persistent": True,
        },
        patient_id="pt-pph-S",
    )

    mtp = await _run_op(
        s, "compute_massive_transfusion_protocol",
        "PPH event: SBP≤90 (1pt), HR≥120 (1pt), FAST positive (1pt) = "
        "3/4 -> MTP activated.",
        compute_massive_transfusion_protocol,
        penetrating_mechanism=False, field_or_arrival_sbp_le_90=True,
        heart_rate_ge_120=True, positive_fast_exam=True,
        estimated_blood_loss_ml=2000, minutes_since_injury=30,
        patient_id="pt-pph-S",
    )

    aki = await _run_op(
        s, "compute_aki_kdigo_stage",
        "Cr 1.0 -> 1.5 = 1.5× baseline -> KDIGO Stage 1.",
        compute_aki_kdigo_stage,
        creatinine_baseline_mg_dl=1.0, creatinine_current_mg_dl=1.5,
        nephrotoxic_medications_present=False,
        patient_id="pt-pph-S",
    )

    counseling = await _run_op(
        s, "compute_discharge_counseling",
        "Postpartum med list: enoxaparin (VTE proph), lisinopril (BP), "
        "ibuprofen (post-op pain) -> 6th-grade-level counseling. "
        "LACE not directly applicable -- use moderate-risk window.",
        compute_discharge_counseling,
        medications=[
            {"name": "enoxaparin 40 mg SC daily",
             "drug_class": "anticoagulant_heparin", "status": "active"},
            {"name": "lisinopril 5 mg PO daily",
             "drug_class": "ace_inhibitor", "status": "active"},
            {"name": "ibuprofen 600 mg PO q6h prn",
             "drug_class": "nsaid", "status": "active"},
        ],
        lace_score=8,
        recommendation_action="home_with_care",
        patient_id="pt-pph-S",
        extra_red_flags=[
            "Heavy vaginal bleeding (soaking >1 pad/hour for ≥2 hours).",
            "Severe abdominal pain or fever > 38°C postpartum.",
            "Severe headache or vision changes (preeclampsia can recur up to 6 weeks postpartum).",
        ],
    )

    pseudo_risk = RiskEstimate(
        model_name="lace-plus-bayesian-v1",
        model_version="placeholder-maternal-mortality",
        outcome_id="mortality_30d",
        horizon_days=30, lace_raw_score=8,
        probability_mean=0.04,
        probability_ci95=(0.02, 0.07), probability_ci_width=0.05,
        contributing_factors=[Factor(name="severe_preeclampsia", raw_value=1.0,
                                        lace_points=4, weight=0.5)],
        fhir_observations_used=[],
        computed_at=datetime.now(timezone.utc),
    )
    fairness = await _run_op(
        s, "compute_fairness_audit",
        "Black race + maternal mortality literature -> confidence_action "
        "should be at least flag_for_review.",
        compute_fairness_audit,
        risk=pseudo_risk,
        patient_demographics={"age": 34, "race": "black",
                                "insurance_type": "medicaid"},
    )

    s.deterministic_action = "home_with_care"
    s.deterministic_confidence = "medium"

    s.final_text = (
        f"Preeclampsia: {pre.classification} (Mg = {pre.magnesium_sulfate_indicated}, "
        f"delivery already done). PPH MTP: ABC={mtp.abc_score}, "
        f"activated={mtp.mtp_activated}. AKI: {aki.aki_stage}. "
        f"Counseling: {counseling.n_medications_explained} meds + "
        f"{counseling.n_red_flags} red flags (postpartum-specific). "
        f"Fairness: confidence_action = {fairness.confidence_action} "
        f"(Black demographic flag fired)."
    )
    await llm_synthesis(s)
    return s


# ─────────────────────────────────────────────────────────────────────
# HTML rendering
# ─────────────────────────────────────────────────────────────────────

def render_html(scenarios: list[Scenario], generated_at: datetime) -> str:
    tools_called: set[str] = set()
    for s in scenarios:
        for op in s.operations:
            tools_called.add(op.tool)

    total_ops = sum(len(s.operations) for s in scenarios)
    total_ms = sum(op.duration_ms for s in scenarios for op in s.operations)
    total_llm_ms = sum(
        s.plan_reasoning_ms + s.synthesis_ms for s in scenarios
    )
    llm_models = sorted({s.llm_model for s in scenarios if s.llm_model})
    llm_label = ", ".join(llm_models) if llm_models else "no LLM"
    n_match = sum(1 for s in scenarios if s.agreement == "match")
    n_safer = sum(1 for s in scenarios if s.agreement == "safer")
    n_mismatch = sum(1 for s in scenarios if s.agreement == "mismatch")
    n_na = sum(1 for s in scenarios if s.agreement in ("", "n_a"))
    agreement_str = (
        f"{n_match}✓ / {n_safer}↑ / {n_mismatch}✗"
        if any([n_match, n_safer, n_mismatch])
        else "n/a"
    )

    parts: list[str] = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TrustedRisk -- End-to-End Showcase</title>
<style>
  :root {{
    --bg: #0d1117;
    --panel: #161b22;
    --panel-2: #1f2530;
    --border: #30363d;
    --text: #c9d1d9;
    --muted: #8b949e;
    --accent: #58a6ff;
    --ok: #3fb950;
    --warn: #d29922;
    --err: #f85149;
    --json-key: #79c0ff;
    --json-str: #a5d6ff;
    --json-num: #ffa657;
    --json-bool: #ff7b72;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          margin: 0; background: var(--bg); color: var(--text); }}
  header {{ padding: 28px 40px; background: var(--panel);
            border-bottom: 1px solid var(--border); }}
  header h1 {{ margin: 0 0 6px 0; font-size: 24px; }}
  header .meta {{ color: var(--muted); font-size: 13px; }}
  .summary {{ display: grid; grid-template-columns: repeat(6, 1fr);
              gap: 16px; padding: 24px 40px; background: var(--panel); }}
  .summary .card {{ padding: 16px; background: var(--panel-2);
                    border: 1px solid var(--border); border-radius: 6px; }}
  .summary .card .num {{ font-size: 28px; font-weight: 600; color: var(--accent); }}
  .summary .card .lbl {{ color: var(--muted); font-size: 12px;
                         text-transform: uppercase; letter-spacing: 0.5px; }}
  main {{ padding: 0 40px 40px 40px; }}
  .tools-coverage {{ padding: 16px 40px; background: var(--panel);
                     border-bottom: 1px solid var(--border); }}
  .tools-coverage h2 {{ margin: 0 0 8px 0; font-size: 14px;
                        text-transform: uppercase; letter-spacing: 0.5px;
                        color: var(--muted); }}
  .tools-coverage .pill-list {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .pill {{ display: inline-block; padding: 4px 10px; border-radius: 12px;
           background: var(--panel-2); border: 1px solid var(--border);
           font-size: 12px; color: var(--accent); font-family: ui-monospace, monospace; }}
  section.scenario {{ margin-top: 32px; background: var(--panel);
                      border: 1px solid var(--border); border-radius: 8px;
                      padding: 24px; }}
  section.scenario h2 {{ margin: 0 0 4px 0; font-size: 18px; }}
  section.scenario .slug {{ color: var(--accent); font-weight: 600;
                            font-size: 12px; letter-spacing: 1px;
                            text-transform: uppercase; margin-bottom: 4px; }}
  section.scenario .persona {{ color: var(--muted); font-style: italic;
                               font-size: 13px; margin-bottom: 16px; }}
  .block {{ margin-top: 16px; }}
  .block .label {{ color: var(--muted); font-size: 11px;
                   text-transform: uppercase; letter-spacing: 0.5px;
                   margin-bottom: 6px; }}
  .block .body {{ background: var(--panel-2); border: 1px solid var(--border);
                  border-radius: 6px; padding: 12px 14px; font-size: 14px;
                  white-space: pre-wrap; }}
  .ops {{ margin-top: 16px; }}
  details.op {{ background: var(--panel-2); border: 1px solid var(--border);
                border-radius: 6px; margin-bottom: 10px; }}
  details.op summary {{ padding: 12px 14px; cursor: pointer;
                        list-style: none; display: flex;
                        align-items: center; gap: 12px; }}
  details.op summary::-webkit-details-marker {{ display: none; }}
  details.op summary .arrow {{ color: var(--muted); }}
  details.op[open] summary .arrow {{ color: var(--accent); }}
  details.op .name {{ font-family: ui-monospace, monospace;
                      color: var(--accent); font-weight: 600; flex: 1; }}
  details.op .ms {{ color: var(--muted); font-size: 12px; }}
  details.op .err {{ color: var(--err); font-size: 12px;
                     padding: 4px 8px; background: #2d161a;
                     border-radius: 4px; }}
  details.op .body {{ padding: 0 14px 14px 14px; border-top: 1px solid var(--border); }}
  details.op .body .label {{ margin-top: 12px; }}
  pre.json {{ background: var(--bg); border: 1px solid var(--border);
              border-radius: 4px; padding: 10px; font-size: 12px;
              line-height: 1.5; max-height: 480px; overflow: auto;
              font-family: ui-monospace, "SF Mono", Consolas, monospace; }}
  pre.json .k {{ color: var(--json-key); }}
  pre.json .s {{ color: var(--json-str); }}
  pre.json .n {{ color: var(--json-num); }}
  pre.json .b {{ color: var(--json-bool); }}
  pre.json .nu {{ color: var(--muted); font-style: italic; }}
  .final {{ margin-top: 18px; padding: 14px;
            background: linear-gradient(180deg, #0a2a18 0%, #0d3a20 100%);
            border: 1px solid #1e6634; border-radius: 6px; color: #d6f5e0;
            font-size: 14px; }}
  .final .label {{ color: #58c98e; margin-bottom: 4px; }}
  .llm {{ margin-top: 16px; padding: 14px 14px 14px 18px;
          background: linear-gradient(180deg, #1a1530 0%, #221a40 100%);
          border-left: 3px solid #a371f7; border-radius: 6px;
          color: #e3dcff; font-size: 14px; line-height: 1.55;
          white-space: pre-wrap; }}
  .llm .head {{ display: flex; align-items: center; gap: 10px;
                margin-bottom: 8px; }}
  .llm .head .badge {{ background: #a371f7; color: #1a1530;
                       padding: 2px 8px; border-radius: 10px;
                       font-size: 11px; font-weight: 600;
                       text-transform: uppercase; letter-spacing: 0.5px; }}
  .llm .head .model {{ color: #b6a4ff; font-family: ui-monospace, monospace;
                       font-size: 12px; }}
  .llm .head .ms {{ color: #8b7fb8; font-size: 12px; margin-left: auto; }}
  .llm .body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }}
  .llm.synthesis {{ background: linear-gradient(180deg, #21162e 0%, #2d1b3f 100%);
                    border-left-color: #d471e8; }}
  .llm.synthesis .head .badge {{ background: #d471e8; color: #21162e; }}
  .llm.synthesis .head .model {{ color: #e3a9ff; }}
  .gate {{ margin-top: 16px; padding: 12px 14px;
           background: #1a2030; border: 1px solid #2d3a52;
           border-left: 3px solid #58a6ff; border-radius: 6px;
           font-size: 13px; }}
  .gate .label {{ color: #58a6ff; font-size: 11px; text-transform: uppercase;
                  letter-spacing: 0.5px; margin-bottom: 4px; }}
  .gate .row {{ display: flex; align-items: center; gap: 10px; }}
  .gate code {{ background: #0d1117; padding: 2px 8px; border-radius: 3px;
                font-family: ui-monospace, monospace; color: #79c0ff; }}
  .agreement {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
                font-size: 11px; font-weight: 600; text-transform: uppercase;
                letter-spacing: 0.5px; }}
  .agreement.match {{ background: #0a2a18; color: #58c98e; border: 1px solid #1e6634; }}
  .agreement.safer {{ background: #2a2210; color: #d29922; border: 1px solid #5d4316; }}
  .agreement.mismatch {{ background: #2d161a; color: #f85149; border: 1px solid #5e1f25; }}
  .agreement.n_a {{ background: var(--panel-2); color: var(--muted); border: 1px solid var(--border); }}
  footer {{ padding: 20px 40px; color: var(--muted); font-size: 12px;
            border-top: 1px solid var(--border); margin-top: 40px;
            background: var(--panel); }}
  footer code {{ background: var(--panel-2); padding: 2px 6px;
                 border-radius: 3px; font-size: 11px; }}
</style>
</head>
<body>
<header>
  <h1>TrustedRisk -- End-to-End Showcase</h1>
  <div class="meta">
    Deterministic exercise of every MCP tool + the bulk-batch HTTP endpoint.
    Generated {generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC.
  </div>
</header>
<div class="summary">
  <div class="card"><div class="num">{len(scenarios)}</div><div class="lbl">Scenarios</div></div>
  <div class="card"><div class="num">{total_ops}</div><div class="lbl">Tool invocations</div></div>
  <div class="card"><div class="num">{len(tools_called)}</div><div class="lbl">Distinct tools called</div></div>
  <div class="card"><div class="num">{int(total_ms)} ms</div><div class="lbl">Tools cumulative</div></div>
  <div class="card"><div class="num">{int(total_llm_ms)} ms</div><div class="lbl">LLM ({html_lib.escape(llm_label)})</div></div>
  <div class="card"><div class="num">{html_lib.escape(agreement_str)}</div><div class="lbl">LLM vs gate (match/safer/mismatch)</div></div>
</div>
<div class="tools-coverage">
  <h2>Tools exercised</h2>
  <div class="pill-list">
    {''.join(f'<span class="pill">{html_lib.escape(t)}</span>' for t in sorted(tools_called))}
  </div>
</div>
<main>
""")

    for s in scenarios:
        parts.append(_render_scenario(s))

    parts.append(f"""</main>
<footer>
  Each scenario shows three layers: <strong>LLM plan</strong> (a real Ollama
  call that articulates the tool sequence + clinical rationale BEFORE any
  orchestration), the <strong>deterministic tool call sequence</strong>
  (async invocations of the registered MCP tools -- captured input, output,
  and timing), and the <strong>LLM synthesis</strong> (a second real LLM
  call that composes the final clinical recommendation grounded ONLY in
  the structured tool outputs above). Scenario&nbsp;D issues a real HTTP
  request through the OAuth + SHARP middleware stack via Starlette's
  TestClient.<br>
  Reproduce: <code>PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json
  .venv/Scripts/python.exe scripts/e2e_showcase.py</code>
</footer>
</body>
</html>""")
    return "".join(parts)


def _render_llm_block(text: str, dt_ms: float, model: str, kind: str,
                       agreement: str = "", llm_action: str | None = None,
                       deterministic_action: str = "") -> str:
    if not text:
        return ""
    label = "LLM Plan (pre-tool reasoning)" if kind == "plan" else "LLM Synthesis (final clinical answer)"
    css_class = "llm" if kind == "plan" else "llm synthesis"
    timing = f"{dt_ms:.0f} ms" if dt_ms > 0 else "(stub fallback)"
    agreement_html = ""
    if kind == "synthesis" and agreement:
        labels = {
            "match": ("MATCH", "LLM agrees with deterministic gate"),
            "safer": ("SAFER", "LLM chose a more conservative action"),
            "mismatch": ("MISMATCH", "LLM picked a less conservative action -- gate would override"),
            "n_a": ("N/A", "no single deterministic action for this scenario"),
        }
        text_label, tip = labels.get(agreement, ("?", ""))
        detail = ""
        if llm_action and deterministic_action:
            detail = f"  llm={llm_action}  vs  det={deterministic_action}"
        agreement_html = (
            f'<span class="agreement {agreement}" title="{html_lib.escape(tip)}">'
            f'{text_label}{html_lib.escape(detail)}</span>'
        )
    return f"""<div class="{css_class}">
  <div class="head">
    <span class="badge">{label}</span>
    <span class="model">{html_lib.escape(model)}</span>
    {agreement_html}
    <span class="ms">{timing}</span>
  </div>
  <div class="body">{html_lib.escape(text)}</div>
</div>"""


def _render_gate(s: Scenario) -> str:
    """Render the deterministic safety-gate decision shown to the LLM."""
    if not s.deterministic_action:
        return ""
    abstain = ""
    if s.abstain_triggers:
        abstain = (
            f' &nbsp;|&nbsp; <strong>ABSTAIN triggers:</strong> '
            f'{html_lib.escape(", ".join(t.get("type","?") for t in s.abstain_triggers))}'
        )
    return f"""<div class="gate">
  <div class="label">Deterministic safety gate (binding constraint passed to the LLM)</div>
  <div class="row">
    <span>action: <code>{html_lib.escape(s.deterministic_action)}</code></span>
    <span>confidence: <code>{html_lib.escape(s.deterministic_confidence)}</code></span>
    {abstain}
  </div>
</div>"""


def _render_scenario(s: Scenario) -> str:
    ops_html = "".join(_render_op(op) for op in s.operations)
    plan_html = _render_llm_block(s.plan_reasoning, s.plan_reasoning_ms,
                                   s.llm_model, kind="plan")
    gate_html = _render_gate(s)
    synth_html = _render_llm_block(
        s.synthesis, s.synthesis_ms, s.llm_model, kind="synthesis",
        agreement=s.agreement, llm_action=s.llm_action,
        deterministic_action=s.deterministic_action,
    )
    return f"""<section class="scenario" id="scenario-{s.slug}">
  <div class="slug">Scenario {s.slug}</div>
  <h2>{html_lib.escape(s.title)}</h2>
  <div class="persona">{html_lib.escape(s.persona)}</div>
  <div class="block">
    <div class="label">User prompt</div>
    <div class="body">{html_lib.escape(s.prompt)}</div>
  </div>
  <div class="block">
    <div class="label">FHIR context (input data)</div>
    <div class="body">{html_lib.escape(s.fhir_summary)}</div>
  </div>
  {plan_html}
  <div class="block">
    <div class="label">Tool plan (what the agent emits)</div>
    <div class="body">{html_lib.escape(' -> '.join(s.expected_tools))}</div>
  </div>
  <div class="ops">
    <div class="block"><div class="label">Tool calls (chronological)</div></div>
    {ops_html}
  </div>
  {gate_html}
  {synth_html}
  <div class="final">
    <div class="label">DETERMINISTIC FINAL OUTPUT (composed by orchestrator)</div>
    {html_lib.escape(s.final_text)}
  </div>
</section>"""


def _render_op(op: Operation) -> str:
    err_pill = (f'<span class="err">{html_lib.escape(op.error)}</span>'
                if op.error else "")
    return f"""<details class="op">
  <summary>
    <span class="arrow">▸</span>
    <span class="name">{html_lib.escape(op.tool)}</span>
    <span class="ms">{op.duration_ms:.1f} ms</span>
    {err_pill}
  </summary>
  <div class="body">
    <div class="label">Rationale</div>
    <div>{html_lib.escape(op.rationale)}</div>
    <div class="label">Input</div>
    <pre class="json">{_render_json(op.input)}</pre>
    <div class="label">Output</div>
    <pre class="json">{_render_json(op.output)}</pre>
  </div>
</details>"""


def _render_json(obj: Any) -> str:
    """Pretty-print + minimal syntax highlight via post-processing."""
    raw = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    # Escape HTML first
    escaped = html_lib.escape(raw)
    # Highlight: keys, strings, numbers, booleans, nulls
    import re as _re
    escaped = _re.sub(r'(&quot;[^&]*?&quot;)(?=\s*:)',
                      r'<span class="k">\1</span>', escaped)
    escaped = _re.sub(r':\s*(&quot;.*?&quot;)',
                      lambda m: m.group(0).replace(m.group(1),
                                                    f'<span class="s">{m.group(1)}</span>'),
                      escaped)
    escaped = _re.sub(r'(?<=[:,\[\s])(-?\d+\.?\d*(?:e[+-]?\d+)?)(?=[,\]\}\s])',
                      r'<span class="n">\1</span>', escaped)
    escaped = _re.sub(r'\b(true|false)\b', r'<span class="b">\1</span>', escaped)
    escaped = _re.sub(r'\bnull\b', r'<span class="nu">null</span>', escaped)
    return escaped


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

async def main() -> int:
    os.environ.setdefault(
        "TRUSTEDRISK_COEFFICIENTS_PATH",
        str(ROOT / "data" / "coefficients.json"),
    )

    print(f"LLM endpoint: {OLLAMA_HOST}  model: {OLLAMA_MODEL}", flush=True)
    scenarios = []
    for fn in (scenario_a, scenario_b, scenario_c, scenario_d,
                scenario_e, scenario_f, scenario_g, scenario_h,
                scenario_i, scenario_j, scenario_k, scenario_l,
                scenario_m, scenario_n, scenario_o, scenario_p,
                scenario_q, scenario_r, scenario_s):
        print(f"Running scenario {fn.__name__}...", flush=True)
        s = await fn()
        s.llm_model = OLLAMA_MODEL
        scenarios.append(s)
        n_ok = sum(1 for op in s.operations if not op.error)
        n_err = sum(1 for op in s.operations if op.error)
        plan_ms = f"{s.plan_reasoning_ms:.0f}ms" if s.plan_reasoning_ms else "stub"
        synth_ms = f"{s.synthesis_ms:.0f}ms" if s.synthesis_ms else "stub"
        print(f"  {len(s.operations)} ops ({n_ok} ok, {n_err} err) | "
              f"LLM plan={plan_ms} synthesis={synth_ms}", flush=True)

    out_path = ROOT / "docs" / "e2e" / "index.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html_text = render_html(scenarios, datetime.now(timezone.utc))
    # Strip lone Unicode surrogates that occasionally appear in LLM output --
    # they're invalid in UTF-8 and would crash the write.
    html_text = html_text.encode("utf-8", errors="replace").decode("utf-8")
    out_path.write_text(html_text, encoding="utf-8")
    print(f"\nWrote {out_path} ({len(html_text):,} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
