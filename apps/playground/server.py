"""TrustedRisk interactive playground (AMB-2).

A FastAPI server that lets a judge / clinician interact with the agent:
  - Pick one of the 19 demo scenarios → see live tool execution + LLM
    reasoning + deterministic safety gate verdict + critic ensemble
    aggregation.
  - Browse a patient's longitudinal timeline (AMB-5.3 cross-session memory).
  - Watch a federated A2A flow: a stub external agent (`darena-data-agent`)
    calling TrustedRisk via the same MCP tool surface a real client would.

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \
        .venv/Scripts/python.exe -m uvicorn apps.playground.server:app \
        --host 0.0.0.0 --port 8765 --reload

Then open: http://localhost:8765/

Self-contained: no new third-party dependencies (FastAPI is already in
the deployment stack via Starlette). UI is HTML+vanilla JS inline — no
build step.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse


# ─────────────────────────────────────────────────────────────────────
# Scenario registry — pulled from scripts/e2e_showcase.py
# ─────────────────────────────────────────────────────────────────────

_SCENARIO_SLUGS = list("ABCDEFGHIJKLMNOPQRS")  # 19 scenarios


def _import_showcase():
    """Lazy-import e2e_showcase. Loads heavy deps (sentence-transformers etc.)
    only on first scenario run."""
    spec = importlib.util.spec_from_file_location(
        "e2e_showcase",
        str(ROOT / "scripts" / "e2e_showcase.py"),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load e2e_showcase module")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e2e_showcase"] = mod
    spec.loader.exec_module(mod)
    return mod


_showcase = None


def get_showcase():
    global _showcase
    if _showcase is None:
        _showcase = _import_showcase()
    return _showcase


def _serialize(obj: Any) -> Any:
    """Generic Pydantic / dataclass / dict serializer."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if is_dataclass(obj) and not isinstance(obj, type):
        return _serialize(asdict(obj))
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
# FastAPI app
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="TrustedRisk Playground", version="1.0.0")


@app.get("/.well-known/agent-card.json")
async def well_known_agent_card() -> JSONResponse:
    """MARKET-1: public agent-card discovery endpoint."""
    p = ROOT / "src" / "a2a_agent" / "agent-card.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="agent_card_missing")
    return JSONResponse(content=json.loads(p.read_text(encoding="utf-8")))


@app.get("/.well-known/marketplace.json")
async def well_known_marketplace() -> JSONResponse:
    """MARKET-2: public marketplace manifest."""
    p = ROOT / "src" / "a2a_agent" / "marketplace.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="marketplace_missing")
    return JSONResponse(content=json.loads(p.read_text(encoding="utf-8")))


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse(content={"status": "ok", "version": "0.7.0"})


@app.get("/api/scenarios")
async def list_scenarios() -> dict[str, Any]:
    """Return the catalog of available scenarios + their metadata."""
    catalog = []
    showcase = get_showcase()
    for slug in _SCENARIO_SLUGS:
        fn_name = f"scenario_{slug.lower()}"
        if not hasattr(showcase, fn_name):
            continue
        # Don't run the scenario — just describe it from the docstring
        # and the slug. Full metadata loads on /api/run/{slug}.
        catalog.append({"slug": slug,
                          "title": _scenario_title_for_slug(slug),
                          "kind": _scenario_kind_for_slug(slug)})
    return {"scenarios": catalog,
              "n": len(catalog),
              "generated_at": datetime.now(timezone.utc).isoformat()}


def _scenario_title_for_slug(slug: str) -> str:
    titles = {
        "A": "CHF discharge — 75y, Black, Medicare",
        "B": "PHI scrubbing + claim grounding",
        "C": "Decision-utility action ranking",
        "D": "Batch HTTP — 5 patients",
        "E": "ED triage — chest pain 58y M",
        "F": "Sepsis early warning — postop ward",
        "G": "Pre-op AC bridging — mech mitral valve",
        "H": "Outpatient med review — 82y on 11 meds (cross-domain)",
        "I": "Pediatric ED — 4-month-old fever + lethargy",
        "J": "Mental health crisis — 28y M ideation + bias guard",
        "K": "Antibiotic stewardship — ESBL UTI",
        "L": "Oncology — NSCLC cycle 4 + RECIST",
        "M": "Acute stroke — 70y M LVO 90 min from LKW",
        "N": "Chest pain HEART score — 58y M",
        "O": "Severe preeclampsia — 32-week + Black demographic",
        "P": "Geriatric — 85y delirium + falls + polypharmacy",
        "Q": "Polytrauma cross-bundle (trauma + ED + discharge)",
        "R": "DKA + AKI + contrast safety cross-bundle",
        "S": "Severe preeclampsia + emergency CS + PPH + discharge",
    }
    return titles.get(slug, f"Scenario {slug}")


def _scenario_kind_for_slug(slug: str) -> str:
    cross_bundle = {"Q", "R", "S"}
    if slug in cross_bundle:
        return "cross_bundle"
    if slug in {"B"}:
        return "documentation_review"
    if slug in {"D"}:
        return "batch_operations"
    return "discharge_decision"


@app.post("/api/run/{slug}")
async def run_scenario(slug: str) -> JSONResponse:
    """Execute a scenario by slug and return the full trace.
    Same payload structure the static showcase HTML uses."""
    slug = slug.upper()
    if slug not in _SCENARIO_SLUGS:
        raise HTTPException(status_code=404,
                              detail=f"unknown scenario {slug!r}")
    showcase = get_showcase()
    fn = getattr(showcase, f"scenario_{slug.lower()}", None)
    if fn is None:
        raise HTTPException(status_code=404,
                              detail=f"scenario_{slug.lower()} not implemented")
    scenario = await fn()
    out = {
        "slug": scenario.slug,
        "title": scenario.title,
        "persona": scenario.persona,
        "prompt": scenario.prompt,
        "fhir_summary": scenario.fhir_summary,
        "expected_tools": scenario.expected_tools,
        "operations": [
            {
                "tool": op.tool,
                "rationale": op.rationale,
                "input": op.input,
                "output": op.output,
                "duration_ms": op.duration_ms,
                "error": op.error,
            }
            for op in scenario.operations
        ],
        "final_text": scenario.final_text,
        "plan_reasoning": scenario.plan_reasoning,
        "plan_reasoning_ms": scenario.plan_reasoning_ms,
        "synthesis": scenario.synthesis,
        "synthesis_ms": scenario.synthesis_ms,
        "llm_model": scenario.llm_model,
        "deterministic_action": scenario.deterministic_action,
        "deterministic_confidence": scenario.deterministic_confidence,
        "abstain_triggers": scenario.abstain_triggers,
        "llm_action": scenario.llm_action,
        "agreement": scenario.agreement,
        "synthesis_kind": scenario.synthesis_kind,
        "extra_actions": scenario.extra_actions,
    }
    return JSONResponse(content=out)


@app.get("/api/timeline/{patient_id}")
async def get_timeline(patient_id: str) -> JSONResponse:
    """Return the longitudinal timeline for one patient (AMB-5.3)."""
    from a2a_agent.memory import MemoryStore
    store = MemoryStore()
    timeline = store.build_patient_timeline(patient_id)
    return JSONResponse(content=_serialize(timeline))


@app.post("/api/timeline/{patient_id}/seed_demo")
async def seed_demo_timeline(patient_id: str) -> JSONResponse:
    """Seed a demo timeline for a patient (3-encounter trajectory with
    progressive deterioration → useful for the playground demo)."""
    from a2a_agent.memory import MemoryStore
    from shared.schemas import (
        Action, AuditBlock, ClaimGrounding, DecisionCard,
        DecisionReasoning, DecisionValidation, Factor, PHIReport,
        Recommendation, RiskEstimate, UtilityAnalysis,
    )

    store = MemoryStore()

    def _make_seed(idx: int, action: str, prob: float,
                     conf: str, lace: int) -> DecisionCard:
        risk = RiskEstimate(
            model_name="lace-plus-bayesian-v1", model_version="seed-v1",
            outcome_id="readmission_30d",
            horizon_days=30, lace_raw_score=lace,
            probability_mean=prob,
            probability_ci95=(max(0.0, prob - 0.05), min(1.0, prob + 0.05)),
            probability_ci_width=0.10,
            contributing_factors=[Factor(name="LACE_length_of_stay",
                                            raw_value=2.0,
                                            lace_points=lace // 2,
                                            weight=0.4)],
            fhir_observations_used=[],
            computed_at=datetime(2025, 1 + idx, 1, tzinfo=timezone.utc),
        )
        util = UtilityAnalysis(
            action_scores_qaly_weeks={Action(action): 0.8},  # type: ignore[arg-type]
            action_scores_ci95={Action(action): (0.6, 1.0)},  # type: ignore[arg-type]
            action_costs_usd={Action(action): 0.0},  # type: ignore[arg-type]
            dominant_action=Action(action),  # type: ignore[arg-type]
            dominance_confidence=0.95,
            reasoning_trace="seeded",
        )
        grounding = ClaimGrounding(
            claim_text="seeded patient stable", sub_claims=[],
            overall_verdict="supported", context_fingerprint=patient_id,
            grounded_at=datetime(2025, 1 + idx, 1, tzinfo=timezone.utc),
        )
        phi = PHIReport(entities_found=[], entity_count_by_type={},
                          redaction_map={}, risk_level="none")
        audit = AuditBlock(
            request_id=f"seed-{patient_id}-{idx}",
            context_fingerprint=patient_id,
            tool_trace=[], server_version="seed", agent_version="seed",
            model_coefficients_version="seed-v1",
            timestamp=datetime(2025, 1 + idx, 1, tzinfo=timezone.utc),
        )
        return DecisionCard(
            recommendation=Recommendation(
                action=Action(action),  # type: ignore[arg-type]
                confidence=conf),  # type: ignore[arg-type]
            reasoning=DecisionReasoning(risk_estimate=risk,
                                            utility_analysis=util),
            validation=DecisionValidation(grounding=grounding,
                                              phi_check=phi),
            abstain=None, audit=audit,
        )

    seed_seq = [
        ("discharge_home", 0.08, "high", 4),
        ("home_with_care", 0.18, "medium", 9),
        ("continued_admission", 0.32, "low", 14),
    ]
    for i, (action, prob, conf, lace) in enumerate(seed_seq):
        store.store_card(_make_seed(i, action, prob, conf, lace))
    return JSONResponse(content={"seeded": len(seed_seq), "patient_id": patient_id})


@app.get("/api/audit/log")
async def audit_log_endpoint(limit: int = 100,
                                request_id: str | None = None,
                                tenant_id: str | None = None,
                                abstain_only: bool = False) -> JSONResponse:
    """Decision trail browser (AUDIT-2)."""
    from a2a_agent.audit import read_audit_log
    events = read_audit_log(limit=limit, request_id=request_id,
                              tenant_id=tenant_id, abstain_only=abstain_only)
    return JSONResponse(content={"events": events, "n": len(events)})


@app.get("/api/audit/archived-decisions")
async def list_archived_endpoint(limit: int = 50) -> JSONResponse:
    from a2a_agent.audit import list_archived_decisions
    return JSONResponse(content={"decisions": list_archived_decisions(limit)})


@app.get("/api/audit/reproducibility/{request_id}")
async def check_reproducibility_endpoint(request_id: str) -> JSONResponse:
    """Pre-flight reproducibility check (AUDIT-1) — does the archive
    exist? Coefficient version drift?"""
    from a2a_agent.audit import check_reproducibility
    result = check_reproducibility(request_id, replay_fn=None)
    return JSONResponse(content=_serialize(result))


@app.post("/api/bundles/suggest")
async def suggest_bundles_endpoint(req: Request) -> JSONResponse:
    """Score and rank bundles for a patient (SAFE-2). Body:
    {chief_complaint, free_text_summary, structured_features}."""
    from a2a_agent.bundle_orchestrator import suggest_bundles
    body = await req.json()
    result = suggest_bundles(
        chief_complaint=body.get("chief_complaint", ""),
        free_text_summary=body.get("free_text_summary", ""),
        structured_features=body.get("structured_features") or {},
    )
    return JSONResponse(content=_serialize(result))


@app.get("/api/calibration/status")
async def calibration_status(window_hours: int = 168) -> JSONResponse:
    """Return a calibration drift report over the past `window_hours`
    DecisionCards in MemoryStore (SCI-3)."""
    from a2a_agent.drift_monitor import compute_drift_report
    report = compute_drift_report(window_hours=window_hours)
    return JSONResponse(content=_serialize(report))


@app.get("/api/bundles")
async def list_bundles() -> JSONResponse:
    """Expose the BUNDLES table (used to populate the bundle filter)."""
    from mcp_server.tools import BUNDLES
    return JSONResponse(content={"bundles": BUNDLES,
                                    "n_bundles": len(BUNDLES)})


@app.get("/api/impact/cumulative")
async def impact_cumulative_get(
    intervention_rrr: float = 0.25,
    avoided_event_cost_usd: float = 14000.0,
    archive_limit: int = 500,
) -> JSONResponse:
    """IMPACT-2: roll up archived DecisionCards into impact KPIs.

    Reads from the audit archive only. Returns zeros when no decisions
    have been archived yet (the audit DB is opt-in).
    """
    from a2a_agent.impact_aggregator import aggregate_impact_kpis
    kpis = aggregate_impact_kpis(
        decisions=None,
        pull_from_archive=True,
        intervention_relative_risk_reduction=intervention_rrr,
        avoided_event_cost_usd=avoided_event_cost_usd,
        archive_limit=archive_limit,
    )
    return JSONResponse(content=_serialize(kpis))


@app.post("/api/impact/cumulative")
async def impact_cumulative_post(req: Request) -> JSONResponse:
    """IMPACT-2: roll up an explicit list of DecisionCard dicts into KPIs.

    Body: {"decisions": [DecisionCard, ...],
           "intervention_rrr": 0.25,
           "avoided_event_cost_usd": 14000.0,
           "include_archive": false}
    """
    body = await req.json()
    decisions = body.get("decisions") or []
    if not isinstance(decisions, list):
        raise HTTPException(status_code=400,
                              detail="`decisions` must be a list")

    from a2a_agent.impact_aggregator import aggregate_impact_kpis
    kpis = aggregate_impact_kpis(
        decisions=decisions,
        pull_from_archive=bool(body.get("include_archive", False)),
        intervention_relative_risk_reduction=float(
            body.get("intervention_rrr", 0.25)),
        avoided_event_cost_usd=float(
            body.get("avoided_event_cost_usd", 14000.0)),
        archive_limit=int(body.get("archive_limit", 500)),
    )
    return JSONResponse(content=_serialize(kpis))


@app.post("/api/coin/dialog")
async def coin_dialog(req: Request) -> JSONResponse:
    """COIN-1 + COIN-3: NL-mediated A2A dialog with handshake auto-discovery."""
    body = await req.json()
    target = body.get("target_agent_id")
    prompt = body.get("prompt")
    if not isinstance(target, str) or not target:
        raise HTTPException(status_code=400,
                              detail="target_agent_id is required")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    from a2a_agent.coin import dialog_with_partner
    result = await dialog_with_partner(
        target_agent_id=target,
        prompt=prompt,
        structured_inputs=body.get("structured_inputs"),
        source_agent_id=body.get("source_agent_id", "trustedrisk-agent"),
    )
    return JSONResponse(content=_serialize(result))


@app.post("/api/notifications/format")
async def notifications_format(req: Request) -> JSONResponse:
    """PATIENT-4: render discharge counseling for SMS / email / print."""
    body = await req.json()
    counseling = body.get("counseling")
    if counseling is None:
        raise HTTPException(status_code=400,
                              detail="counseling is required")
    channels = body.get("channels")
    sms_limit = int(body.get("sms_chunk_limit", 160))
    from a2a_agent.notification_formatter import (
        format_discharge_notifications,
    )
    try:
        bundle = format_discharge_notifications(
            counseling, channels=channels, sms_chunk_limit=sms_limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=_serialize(bundle))


@app.post("/api/causal/ate")
async def causal_ate(req: Request) -> JSONResponse:
    """SCALE-2: causal-inference Average Treatment Effect via DoWhy."""
    body = await req.json()
    cohort = body.get("cohort")
    treatment = body.get("treatment")
    outcome = body.get("outcome")
    confounders = body.get("confounders") or []
    method = body.get("method", "linear_regression")
    run_refutations = bool(body.get("run_refutations", True))
    if not isinstance(cohort, list) or not cohort:
        raise HTTPException(status_code=400,
                              detail="cohort must be a non-empty list")
    if not isinstance(treatment, str) or not treatment:
        raise HTTPException(status_code=400, detail="treatment is required")
    if not isinstance(outcome, str) or not outcome:
        raise HTTPException(status_code=400, detail="outcome is required")

    from a2a_agent.causal_inference import compute_average_treatment_effect
    try:
        report = compute_average_treatment_effect(
            cohort=cohort, treatment=treatment, outcome=outcome,
            confounders=confounders, method=method,
            run_refutations=run_refutations,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=_serialize(report))


@app.post("/api/mdp/decision")
async def mdp_decision(req: Request) -> JSONResponse:
    """SCALE-3: 90-day finite-horizon MDP for discharge action selection."""
    body = await req.json()
    from a2a_agent.mdp_decision import compute_sequential_mdp_value
    try:
        report = compute_sequential_mdp_value(
            baseline_readmission_30d_prob=body.get(
                "baseline_readmission_30d_prob"),
            risk_estimate=body.get("risk_estimate"),
            horizon_days=int(body.get("horizon_days", 90)),
            patient_id=body.get("patient_id"),
            initial_state=body.get("initial_state", "home_well"),
            discount_factor=float(body.get("discount_factor", 0.97)),
            actions=body.get("actions"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=_serialize(report))


@app.post("/api/orchestrate/conversational")
async def orchestrate_conversational(req: Request) -> JSONResponse:
    """GENAI-5: NL → multi-agent workflow plan → execution.

    Body: {"query": "...", "patient_id": ..., "use_llm": false,
           "execute": false, "initial_context": {...}}
    """
    body = await req.json()
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=400,
                              detail="query is required")
    use_llm = bool(body.get("use_llm", False))

    from a2a_agent.conversational_orchestrator import (
        execute_workflow, plan_workflow,
    )
    plan = plan_workflow(query, use_llm=use_llm)

    if not body.get("execute", False):
        return JSONResponse(content=_serialize(plan))

    initial_context = dict(body.get("initial_context", {}))
    if "patient_id" in body and body["patient_id"]:
        initial_context.setdefault("patient_id", body["patient_id"])

    result = await execute_workflow(plan, initial_context=initial_context)
    return JSONResponse(content=_serialize(result))


@app.get("/api/tools/search")
async def search_tools(q: str, top_k: int = 5) -> JSONResponse:
    """GENAI-4: semantic search across the 40 MCP tools."""
    from a2a_agent.tool_discovery import semantic_search_tools
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="q is required")
    try:
        result = semantic_search_tools(q, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=_serialize(result))


@app.get("/api/lineage")
async def get_data_lineage() -> JSONResponse:
    """CMP-3: data lineage DAG (artifact hashes + Mermaid)."""
    from a2a_agent.data_lineage import compute_data_lineage
    return JSONResponse(content=_serialize(compute_data_lineage()))


@app.post("/api/economics/ladder")
async def economics_ladder(req: Request) -> JSONResponse:
    """SIM-3: build a $/QALY paragonability ladder vs published RCTs.

    Body: {"target_intervention_name": "...",
           "target_arr_30d_percentage_points": 3.5,
           "target_cost_per_patient_usd": 75.0,
           "avoided_event_cost_usd": 14000.0,
           "qaly_gained_per_avoided_event": 0.05}
    """
    body = await req.json()
    from a2a_agent.cost_effectiveness_ladder import \
        build_cost_effectiveness_ladder
    try:
        ladder = build_cost_effectiveness_ladder(
            target_intervention_name=body.get(
                "target_intervention_name",
                "TrustedRisk-recommended intervention"),
            target_arr_30d_percentage_points=float(body.get(
                "target_arr_30d_percentage_points", 3.0)),
            target_cost_per_patient_usd=float(body.get(
                "target_cost_per_patient_usd", 75.0)),
            avoided_event_cost_usd=float(body.get(
                "avoided_event_cost_usd", 14_000.0)),
            qaly_gained_per_avoided_event=float(body.get(
                "qaly_gained_per_avoided_event", 0.05)),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=_serialize(ladder))


@app.post("/api/equity/cohort")
async def equity_cohort(req: Request) -> JSONResponse:
    """SIM-2: cohort-wide equity rollup across (DecisionCard, demographics).

    Body: {"cohort": [{"decision_card": <card>, "demographics": {...}}, ...],
           "disparity_threshold_pct": 0.15}
    """
    body = await req.json()
    cohort = body.get("cohort")
    if not isinstance(cohort, list):
        raise HTTPException(status_code=400,
                              detail="cohort must be a list")
    from a2a_agent.equity_dashboard import compute_equity_dashboard
    try:
        dash = compute_equity_dashboard(
            cohort,
            disparity_threshold_pct=float(
                body.get("disparity_threshold_pct", 0.15)),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=_serialize(dash))


@app.post("/api/simulate/year")
async def simulate_year(req: Request) -> JSONResponse:
    """SIM-1: Monte Carlo simulation of one year of intervention rollout.

    Body: {
      "case_mix": [{"name": "...", "n_patients_per_year": 200,
                       "baseline_event_probability": 0.30}, ...],
      "intervention_name": "...",
      "intervention_relative_risk_reduction": 0.25,
      "intervention_cost_per_patient_usd": 75.0,
      "avoided_event_cost_usd": 14000.0,
      "qaly_gained_per_avoided_event": 0.05,
      "n_iterations": 2000,
      "seed": 42
    }
    """
    body = await req.json()
    case_mix = body.get("case_mix")
    if not isinstance(case_mix, list) or not case_mix:
        raise HTTPException(status_code=400,
                              detail="case_mix must be a non-empty list")

    from a2a_agent.outcomes_simulator import simulate_hospital_year
    try:
        report = simulate_hospital_year(
            case_mix=case_mix,
            intervention_name=body.get(
                "intervention_name",
                "Pharmacist-led discharge counseling"),
            intervention_relative_risk_reduction=float(body.get(
                "intervention_relative_risk_reduction", 0.25)),
            intervention_cost_per_patient_usd=float(body.get(
                "intervention_cost_per_patient_usd", 75.0)),
            avoided_event_cost_usd=float(body.get(
                "avoided_event_cost_usd", 14_000.0)),
            qaly_gained_per_avoided_event=body.get(
                "qaly_gained_per_avoided_event", 0.05),
            n_iterations=int(body.get("n_iterations", 2_000)),
            seed=int(body.get("seed", 42)),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(content=_serialize(report))


@app.get("/api/registry")
async def get_agent_registry() -> JSONResponse:
    """COMPOSE-3: list the 4 agents in the multi-agent stack.

    Returns id / role / configured URL / skills / embedded agent-card for
    each. Consumers (orchestrators, ops dashboards) discover partner agents
    here instead of hard-coding URLs.
    """
    from a2a_agent.registry import serialize_registry
    return JSONResponse(content=serialize_registry())


# ─────────────────────── Phase 8.1 — composer integration ───────────────────────


@app.get("/api/composer/templates")
async def composer_templates() -> JSONResponse:
    """List the 5 composer workflow templates."""
    from apps.composer.workflows import list_workflows
    return JSONResponse(content={"workflows": list_workflows()})


@app.post("/api/composer/run/{workflow_id}")
async def composer_run(workflow_id: str, req: Request) -> JSONResponse:
    """Run a composer workflow via the v7 playground."""
    from dataclasses import asdict
    from apps.composer.orchestrator import execute_workflow
    from apps.composer.workflows import REGISTRY
    if workflow_id not in REGISTRY:
        raise HTTPException(
            status_code=404, detail=f"unknown workflow {workflow_id!r}")
    body = await req.json()
    inputs = body.get("inputs") or body
    try:
        execution = await execute_workflow(REGISTRY[workflow_id], inputs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    out = asdict(execution)
    for step in out["steps"]:
        step.pop("output", None)
    return JSONResponse(content=out)


@app.get("/api/llm/polish/status")
async def llm_polish_status() -> JSONResponse:
    """Phase 6.2 — surface the resolved LLM polish client so the v7
    UI can show the polish toggle + post-check contract."""
    from a2a_agent.llm_polish import resolve_polish_client
    client = resolve_polish_client()
    return JSONResponse(content={
        "model_id": client.model_id,
        "is_active": client.model_id is not None,
        "post_check": (
            "cite-back / drug / ICD / dose preservation enforced; "
            "polish rejected if any must-preserve token drops"
        ),
    })


@app.get("/api/specialists")
async def list_specialists() -> JSONResponse:
    """Phase 8.1 — list the 11 federation specialists with their
    advertised skills + bundle counts. Drives the v7 'specialist
    cards' panel."""
    import json as _json
    from pathlib import Path as _Path
    apps_dir = _Path(__file__).resolve().parent.parent
    cards: list[dict] = []
    for spec_dir in sorted(apps_dir.glob("specialist_*")):
        card_path = spec_dir / "agent_card.json"
        if not card_path.exists():
            continue
        raw = _json.loads(card_path.read_text(encoding="utf-8"))
        bundles = [
            b for b in raw.get("_bundles", {}) if not b.startswith("_")
        ]
        cards.append({
            "id": spec_dir.name,
            "name": raw.get("name"),
            "skill_count": len(raw.get("skills", [])),
            "bundle_count": len(bundles),
            "bundles": bundles,
            "url": (raw.get("supportedInterfaces") or [{}])[0].get("url")
                or raw.get("url"),
        })
    return JSONResponse(content={"specialists": cards})


@app.post("/api/orchestrate/discharge_with_partners")
async def orchestrate_discharge_with_partners(req: Request) -> JSONResponse:
    """COMPOSE-3 multi-agent orchestration:

      1. Run a TrustedRisk discharge scenario → DecisionCard (decision agent).
      2. Hand the card to scheduler-agent  → follow-up visit plan.
      3. Trigger alert-agent with a (caller-supplied or synthetic) drift report
         to demonstrate calibration-drift escalation.

    Body: {"scenario": "A", "drift_report": {...}? }

    Returns a composite trace with an entry per participating agent.
    """
    body: dict[str, Any] = {}
    if req.headers.get("content-length") and \
            int(req.headers["content-length"]) > 0:
        body = await req.json()

    slug = (body.get("scenario") or "A").upper()
    if slug not in _SCENARIO_SLUGS:
        raise HTTPException(status_code=404,
                              detail=f"unknown scenario {slug!r}")

    showcase = get_showcase()
    fn = getattr(showcase, f"scenario_{slug.lower()}", None)
    if fn is None:
        raise HTTPException(status_code=404,
                              detail=f"scenario_{slug.lower()} not implemented")
    scenario = await fn()

    # Synthesize a DecisionCard-shaped dict from the scenario trace so the
    # partner agents can consume it.
    decision_card = {
        "recommendation": {
            "action": scenario.deterministic_action,
            "confidence": scenario.deterministic_confidence,
        } if scenario.deterministic_action else None,
        "reasoning": {
            "risk_estimate": _extract_first_risk(scenario.operations),
        },
        "audit": {
            "request_id": f"orchestrate-{slug}-{int(datetime.now(timezone.utc).timestamp())}",
            "tools": [{"tool": op.tool} for op in scenario.operations],
        },
        "validation": {},
        "abstain": [],
    }

    # 2. Scheduler agent (in-process)
    from apps.scheduler_agent import server as sched_srv
    discharge_date = body.get("discharge_date") or datetime.now(
        timezone.utc).date().isoformat()

    class _Req:
        def __init__(self, payload: dict[str, Any]):
            self._payload = payload
            self.headers = {"content-length": "1"}

        async def json(self) -> dict[str, Any]:
            return self._payload

    sched_response = await sched_srv.propose_followup_visits(
        _Req({"decision_card": decision_card,
              "discharge_date": discharge_date})
    )
    followup_plan = json.loads(sched_response.body)

    # 3. Alert agent (in-process). Use caller-supplied drift_report or a
    # default warn-tier synthetic to make the orchestration end-to-end visible.
    from apps.alert_agent import server as alert_srv
    drift_report = body.get("drift_report") or {
        "overall_tier": "warn",
        "rationale": ("Synthetic warn-tier drift report (orchestration demo). "
                        "In production, /api/calibration/status drives this."),
        "signals": [
            {"name": "ece_post_hoc", "value": 0.13, "baseline": 0.08,
             "tier": "warn", "detail": "ECE drift +5pp"},
        ],
        "window_label": "demo", "window_n_cards": 0,
    }

    alert_response = await alert_srv.check_drift_now(
        _Req({"drift_report": drift_report})
    )
    alert_payload = json.loads(alert_response.body)

    return JSONResponse(content={
        "orchestration_id": uuid.uuid4().hex,
        "agents_invoked": [
            "trustedrisk-agent",
            "trustedrisk-scheduler-agent",
            "trustedrisk-alert-agent",
        ],
        "scenario": {
            "slug": scenario.slug,
            "title": scenario.title,
            "tools_called": [op.tool for op in scenario.operations],
            "deterministic_action": scenario.deterministic_action,
            "deterministic_confidence": scenario.deterministic_confidence,
        },
        "decision_card": decision_card,
        "followup_plan": followup_plan,
        "drift_alert": alert_payload,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })


def _extract_first_risk(operations: list[Any]) -> dict[str, Any]:
    """Pull the first compute_readmission_risk output (if any) into a
    risk_estimate dict shape for downstream agents."""
    for op in operations:
        if op.tool == "compute_readmission_risk":
            out = op.output if isinstance(op.output, dict) else {}
            mean = out.get("probability_mean")
            if isinstance(mean, (int, float)):
                return {"probability_mean": float(mean)}
    return {}


@app.post("/api/federation/darena_query")
async def federation_demo(req: Request) -> JSONResponse:
    """Mock A2A flow: a 'darena-data-agent' calls TrustedRisk to evaluate a
    patient. The body has form {patient_summary: {...}, requested_skill:
    'safe_discharge_review'}. We simulate the federated handshake by
    routing to scenario A and re-tagging the response.
    """
    body = await req.json()
    requested_skill = body.get("requested_skill", "safe_discharge_review")
    showcase = get_showcase()
    scenario = await showcase.scenario_a()
    return JSONResponse(content={
        "federation_handshake": {
            "from_agent": "darena-data-agent (mock)",
            "to_agent": "trustedrisk-agent",
            "requested_skill": requested_skill,
            "negotiated_via": "A2A AgentCard (mock — playground stub)",
        },
        "decision_card_summary": {
            "recommendation_action": scenario.deterministic_action,
            "recommendation_confidence": scenario.deterministic_confidence,
            "agreement_with_llm": scenario.agreement,
            "n_tool_calls": len(scenario.operations),
            "scenario_used": scenario.slug,
        },
        "audit": {
            "tool_trace": [op.tool for op in scenario.operations],
            "synthesis_kind": scenario.synthesis_kind,
        },
    })


# ─────────────────────────────────────────────────────────────────────
# UI — single-page HTML with vanilla JS
# ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _INDEX_HTML


_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TrustedRisk Playground</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --panel-2: #1f2530;
    --border: #30363d; --text: #c9d1d9; --muted: #8b949e;
    --accent: #58a6ff; --ok: #3fb950; --warn: #d29922; --err: #f85149;
    --critic: #a371f7; --synthesis: #d471e8;
  }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          margin: 0; background: var(--bg); color: var(--text); }
  header { padding: 18px 32px; background: var(--panel);
           border-bottom: 1px solid var(--border); display: flex;
           align-items: center; gap: 24px; }
  header h1 { margin: 0; font-size: 18px; }
  header .nav { display: flex; gap: 16px; margin-left: auto; }
  header .nav a { color: var(--accent); text-decoration: none; font-size: 14px; }
  header .nav a:hover { text-decoration: underline; }
  main { padding: 24px 32px; max-width: 1300px; margin: 0 auto; }
  section { margin-bottom: 28px; }
  section h2 { font-size: 14px; text-transform: uppercase;
                letter-spacing: 0.5px; color: var(--muted);
                margin: 0 0 8px 0; }
  .scenario-grid { display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                    gap: 12px; }
  .scenario-card { padding: 12px 14px; background: var(--panel);
                    border: 1px solid var(--border); border-radius: 6px;
                    cursor: pointer; transition: all 0.15s; }
  .scenario-card:hover { background: var(--panel-2); border-color: var(--accent); }
  .scenario-card .slug { font-family: ui-monospace, monospace;
                          color: var(--accent); font-weight: 600;
                          font-size: 11px; letter-spacing: 1px; }
  .scenario-card .title { font-size: 13px; margin-top: 2px; }
  .scenario-card .kind { font-size: 11px; color: var(--muted);
                          margin-top: 4px; }
  .scenario-card.cross_bundle .slug::after { content: " ⚡"; }
  #result { display: none; }
  #result.active { display: block; }
  #result-header { padding: 16px; background: var(--panel);
                    border: 1px solid var(--border); border-radius: 6px;
                    margin-bottom: 16px; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.5px; }
  .badge.match { background: #0a2a18; color: var(--ok); border: 1px solid #1e6634; }
  .badge.safer { background: #2a2210; color: var(--warn); border: 1px solid #5d4316; }
  .badge.mismatch { background: #2d161a; color: var(--err); border: 1px solid #5e1f25; }
  .badge.n_a { background: var(--panel-2); color: var(--muted); border: 1px solid var(--border); }
  .reasoning-block { margin: 12px 0; padding: 14px;
                      background: linear-gradient(180deg, #1a1530 0%, #221a40 100%);
                      border-left: 3px solid var(--critic); border-radius: 6px;
                      color: #e3dcff; font-size: 13px; line-height: 1.6;
                      white-space: pre-wrap; }
  .reasoning-block.synthesis { background: linear-gradient(180deg, #21162e 0%, #2d1b3f 100%);
                                 border-left-color: var(--synthesis); }
  .gate-block { padding: 10px 14px; background: #1a2030;
                 border: 1px solid #2d3a52; border-left: 3px solid var(--accent);
                 border-radius: 6px; font-size: 13px; margin: 12px 0; }
  .gate-block code { background: var(--bg); padding: 2px 6px;
                      border-radius: 3px; color: #79c0ff;
                      font-family: ui-monospace, monospace; }
  details { background: var(--panel); border: 1px solid var(--border);
             border-radius: 6px; margin: 6px 0; }
  details summary { padding: 10px 14px; cursor: pointer;
                     font-family: ui-monospace, monospace; }
  details[open] summary { border-bottom: 1px solid var(--border); }
  details .body { padding: 12px 14px; }
  pre { background: var(--bg); padding: 10px; border-radius: 4px;
         font-size: 11px; max-height: 320px; overflow: auto;
         font-family: ui-monospace, monospace;
         white-space: pre-wrap; word-break: break-word; }
  button { padding: 8px 14px; background: var(--accent); color: white;
            border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  input { padding: 8px; background: var(--bg); color: var(--text);
           border: 1px solid var(--border); border-radius: 4px; font-size: 13px; }
  .spinner { display: inline-block; width: 14px; height: 14px;
              border: 2px solid var(--border); border-top-color: var(--accent);
              border-radius: 50%; animation: spin 0.7s linear infinite;
              margin-right: 6px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .timeline { display: grid; gap: 8px; }
  .timeline-entry { padding: 10px 14px; background: var(--panel);
                     border: 1px solid var(--border); border-radius: 6px;
                     display: grid; grid-template-columns: 140px 140px 1fr;
                     gap: 12px; align-items: center; font-size: 13px; }
  .timeline-entry .ts { color: var(--muted); font-size: 11px; }
  .drift-flag { padding: 6px 10px; background: #2a2210; border: 1px solid #5d4316;
                 color: var(--warn); border-radius: 4px; font-size: 12px;
                 margin: 4px 0; }
  .tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border);
           margin-bottom: 16px; }
  .tab { padding: 8px 14px; cursor: pointer; color: var(--muted);
          border-bottom: 2px solid transparent; }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .panel { display: none; }
  .panel.active { display: block; }
</style>
</head>
<body>
<header>
  <h1>TrustedRisk Playground</h1>
  <div style="font-size: 12px; color: var(--muted);">37 tools · 12 bundles · 19 scenarios · 802 tests</div>
  <div class="nav">
    <a href="#scenarios" onclick="showTab('scenarios')">Scenarios</a>
    <a href="#timeline" onclick="showTab('timeline')">Patient timeline</a>
    <a href="#federation" onclick="showTab('federation')">A2A federation</a>
  </div>
</header>
<main>
  <div id="panel-scenarios" class="panel active">
    <section>
      <h2>Pick a clinical scenario</h2>
      <div id="scenario-grid" class="scenario-grid">Loading…</div>
    </section>
    <section id="result"></section>
  </div>

  <div id="panel-timeline" class="panel">
    <section>
      <h2>Longitudinal patient timeline (cross-session memory)</h2>
      <p style="font-size: 13px; color: var(--muted);">
        Patient ID:
        <input type="text" id="patient-id-input" value="demo-pt-1" />
        <button onclick="seedDemoTimeline()">Seed 3-encounter demo trajectory</button>
        <button onclick="loadTimeline()">Load timeline</button>
      </p>
      <div id="timeline-result"></div>
    </section>
  </div>

  <div id="panel-federation" class="panel">
    <section>
      <h2>A2A federation demo — Darena-Data-Agent → TrustedRisk</h2>
      <p style="font-size: 13px; color: var(--muted);">
        Simulates a federated A2A request from a hypothetical
        <code>darena-data-agent</code> to TrustedRisk. The remote agent
        sends a patient summary and requests the
        <code>safe_discharge_review</code> skill; TrustedRisk routes to
        the appropriate bundle and returns the structured DecisionCard.
      </p>
      <button onclick="runFederation()">Run federated request</button>
      <div id="federation-result" style="margin-top: 16px;"></div>
    </section>
  </div>
</main>
<script>
function showTab(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
}

async function loadScenarios() {
  const r = await fetch('/api/scenarios');
  const data = await r.json();
  const grid = document.getElementById('scenario-grid');
  grid.innerHTML = '';
  data.scenarios.forEach(s => {
    const div = document.createElement('div');
    div.className = 'scenario-card ' + s.kind;
    div.onclick = () => runScenario(s.slug);
    div.innerHTML =
      '<div class="slug">Scenario ' + s.slug + '</div>' +
      '<div class="title">' + escapeHtml(s.title) + '</div>' +
      '<div class="kind">' + s.kind + '</div>';
    grid.appendChild(div);
  });
}

async function runScenario(slug) {
  const result = document.getElementById('result');
  result.innerHTML = '<div style="padding: 16px;"><span class="spinner"></span>Running scenario ' + slug + '… (15-30s with LLM)</div>';
  result.classList.add('active');
  result.scrollIntoView({behavior: 'smooth'});
  const r = await fetch('/api/run/' + slug, {method: 'POST'});
  if (!r.ok) {
    result.innerHTML = '<div style="color: var(--err);">Error: ' + r.status + '</div>';
    return;
  }
  const s = await r.json();
  result.innerHTML = renderScenario(s);
}

function renderScenario(s) {
  let html = '<div id="result-header">' +
    '<div class="slug" style="color: var(--accent); font-weight: 600; font-size: 12px; letter-spacing: 1px;">SCENARIO ' + s.slug + '</div>' +
    '<h2 style="margin: 4px 0 12px 0; font-size: 18px;">' + escapeHtml(s.title) + '</h2>' +
    '<div style="font-size: 13px; color: var(--muted); margin-bottom: 8px;">' + escapeHtml(s.persona) + '</div>' +
    '<div style="font-size: 13px; line-height: 1.5;">' + escapeHtml(s.prompt) + '</div>' +
    '</div>';

  if (s.plan_reasoning) {
    html += '<div class="reasoning-block">' +
      '<div style="font-size: 11px; color: #b6a4ff; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">LLM PLAN · ' + s.llm_model + ' · ' + Math.round(s.plan_reasoning_ms) + 'ms</div>' +
      escapeHtml(s.plan_reasoning) + '</div>';
  }

  html += '<div style="font-size: 12px; color: var(--muted); margin: 12px 0 6px 0;">TOOL CALLS (' + s.operations.length + ')</div>';
  s.operations.forEach(op => {
    html += '<details><summary><code>' + escapeHtml(op.tool) + '</code> · ' + Math.round(op.duration_ms) + 'ms' +
      (op.error ? ' · <span style="color: var(--err);">ERROR</span>' : '') +
      '</summary><div class="body">' +
      '<div style="font-size: 12px; color: var(--muted); margin-bottom: 6px;">' + escapeHtml(op.rationale) + '</div>' +
      '<details><summary style="font-size: 11px;">Input</summary><pre>' + escapeHtml(JSON.stringify(op.input, null, 2)) + '</pre></details>' +
      '<details><summary style="font-size: 11px;">Output</summary><pre>' + escapeHtml(JSON.stringify(op.output, null, 2)) + '</pre></details>' +
      '</div></details>';
  });

  if (s.deterministic_action) {
    html += '<div class="gate-block">' +
      '<div style="font-size: 11px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">Deterministic safety gate</div>' +
      'action: <code>' + escapeHtml(s.deterministic_action) + '</code> · ' +
      'confidence: <code>' + escapeHtml(s.deterministic_confidence) + '</code>' +
      '</div>';
  }

  if (s.synthesis) {
    let agreementHtml = '';
    if (s.agreement) {
      agreementHtml = ' · <span class="badge ' + s.agreement + '">' + (s.agreement === 'n_a' ? 'N/A' : s.agreement.toUpperCase()) + '</span>';
    }
    html += '<div class="reasoning-block synthesis">' +
      '<div style="font-size: 11px; color: #e3a9ff; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">LLM SYNTHESIS · ' + s.llm_model + ' · ' + Math.round(s.synthesis_ms) + 'ms' + agreementHtml + '</div>' +
      escapeHtml(s.synthesis) + '</div>';
  }

  html += '<div style="margin-top: 16px; padding: 14px; background: linear-gradient(180deg, #0a2a18 0%, #0d3a20 100%); border: 1px solid #1e6634; border-radius: 6px; color: #d6f5e0; font-size: 13px;">' +
    '<div style="font-size: 11px; color: #58c98e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Deterministic final output</div>' +
    escapeHtml(s.final_text) + '</div>';
  return html;
}

async function loadTimeline() {
  const pid = document.getElementById('patient-id-input').value;
  const r = await fetch('/api/timeline/' + encodeURIComponent(pid));
  const data = await r.json();
  const out = document.getElementById('timeline-result');
  if (data.n_encounters === 0) {
    out.innerHTML = '<div style="color: var(--muted); padding: 12px;">No encounters yet for ' + escapeHtml(pid) + '. Click "Seed demo trajectory" to populate.</div>';
    return;
  }
  let html = '<div style="margin: 12px 0;"><strong>' + data.n_encounters + ' encounters</strong> from ' + data.first_encounter_at + ' to ' + data.last_encounter_at + '</div>';
  if (data.drift_flags && data.drift_flags.length > 0) {
    html += '<div style="margin: 12px 0;"><strong>Drift signals (' + data.drift_flags.length + '):</strong></div>';
    data.drift_flags.forEach(f => { html += '<div class="drift-flag">' + escapeHtml(f) + '</div>'; });
  }
  html += '<div class="timeline">';
  data.entries.forEach(e => {
    html += '<div class="timeline-entry">' +
      '<div class="ts">' + e.encounter_at + '</div>' +
      '<div><code style="color: var(--accent);">' + e.encounter_type + '</code></div>' +
      '<div>' + (e.recommendation_action || '<em>abstain</em>') + ' (' + (e.recommendation_confidence || '—') + ')' +
      ' · prob=' + (e.risk_probability_mean !== null ? e.risk_probability_mean.toFixed(3) : '—') +
      ' · ' + e.abstain_triggers_count + ' abstain trigger(s)</div>' +
      '</div>';
  });
  html += '</div>';
  out.innerHTML = html;
}

async function seedDemoTimeline() {
  const pid = document.getElementById('patient-id-input').value;
  const r = await fetch('/api/timeline/' + encodeURIComponent(pid) + '/seed_demo', {method: 'POST'});
  const data = await r.json();
  alert('Seeded ' + data.seeded + ' encounters for ' + pid);
  loadTimeline();
}

async function runFederation() {
  const out = document.getElementById('federation-result');
  out.innerHTML = '<div><span class="spinner"></span>Running federated A2A handshake + scenario A…</div>';
  const r = await fetch('/api/federation/darena_query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      patient_summary: {patient_id: 'pt-fed-001', age: 72, conditions: ['CHF', 'AKI', 'DM2']},
      requested_skill: 'safe_discharge_review',
    }),
  });
  const data = await r.json();
  out.innerHTML = '<pre>' + escapeHtml(JSON.stringify(data, null, 2)) + '</pre>';
}

function escapeHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;',
  })[c]);
}

loadScenarios();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
