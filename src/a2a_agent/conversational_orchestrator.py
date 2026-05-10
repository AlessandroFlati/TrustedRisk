"""GENAI-5 -- Conversational orchestrator across the 4-agent registry.

Maps a natural-language workflow description (e.g. "for patient X, I want
risk + counseling + follow-up + drift check") to an executable plan that
chains calls across:

  - trustedrisk-agent (decision)
  - trustedrisk-scheduler-agent (follow-up planning)
  - trustedrisk-alert-agent (drift / calibration)
  - darena-data-agent (federation)

Two-layer extraction:
  1. Deterministic keyword extractor -- covers the 9 intent vocabulary items
     and is the floor.
  2. Optional LLM (Ollama) -- refines the intent set + drops false positives.

Execution is in-process: the orchestrator imports the partner agents'
route handlers and invokes them directly with a minimal Request stub,
avoiding HTTP overhead while still exercising the same code paths.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from shared.schemas import (
    WorkflowExecutionResult,
    WorkflowExecutionStep,
    WorkflowIntent,
    WorkflowPlan,
)


# ─────────────────────── Intent vocabulary ───────────────────────

_INTENT_PATTERNS: list[tuple[str, str]] = [
    # Stems use \b only at the start so suffixed forms ("counseling",
    # "readmission", "diagnoses", "differential") still match.
    (r"\b(?:risk|readmission|lace|probability)", "risk_assessment"),
    (r"\b(?:counsel|patient[\s-]?education|instructions|teach[\s-]?back)",
     "counseling"),
    (r"\b(?:follow[\s-]?up|appointment|pcp visit|scheduling)", "followup"),
    (r"\b(?:drift|calibrat|alert|monitor)", "drift_check"),
    (r"\b(?:equity|fairness|subgroup|disparity|bias)", "equity_audit"),
    (r"\b(?:diff|ddx|diagnos)", "differential_diagnosis"),
    (r"\b(?:cost[\s-]?effective|qaly|icer|nnt|economic)",
     "cost_effectiveness"),
    (r"\b(?:impact|kpi|cohort impact|cumulative)", "cumulative_impact"),
    (r"\b(?:resolve|find patient|which patient|chart open|admitted)",
     "patient_resolution"),
]


_INTENT_AGENT: dict[str, str] = {
    "risk_assessment":         "trustedrisk-agent",
    "counseling":              "trustedrisk-agent",
    "differential_diagnosis":  "trustedrisk-agent",
    "cost_effectiveness":      "trustedrisk-agent",
    "cumulative_impact":       "trustedrisk-agent",
    "equity_audit":            "trustedrisk-agent",
    "patient_resolution":      "trustedrisk-agent",
    "followup":                "trustedrisk-scheduler-agent",
    "drift_check":             "trustedrisk-alert-agent",
}


_INTENT_DESCRIPTION: dict[str, str] = {
    "risk_assessment":
        "Compute calibrated 30-day readmission risk + LACE breakdown.",
    "counseling":
        "Generate a 6th-grade-level discharge counseling section per medication "
        "+ red-flag list.",
    "differential_diagnosis":
        "Rank a differential diagnosis for the chief complaint with grounding.",
    "cost_effectiveness":
        "Estimate ICER + NNT + $/QALY for a proposed intervention.",
    "cumulative_impact":
        "Roll up archived DecisionCards into impact KPIs.",
    "equity_audit":
        "Run the population equity dashboard (subgroup × action × calibration).",
    "patient_resolution":
        "Resolve a natural-language patient reference to a FHIR Patient ID.",
    "followup":
        "Build a follow-up visit plan from the DecisionCard "
        "(PCP + specialist + modality + timing).",
    "drift_check":
        "Poll calibration drift + dispatch webhooks if tier escalates.",
}


# Dependencies -- counseling and follow-up depend on having a risk first.
# differential_diagnosis stands alone. equity_audit needs cohort data,
# but for in-process demo we let it run independently with a default cohort.
_INTENT_DEPENDENCIES: dict[str, list[str]] = {
    "counseling": ["risk_assessment"],
    "followup": ["risk_assessment"],
    "cost_effectiveness": ["risk_assessment"],
    "drift_check": [],
    "differential_diagnosis": [],
    "patient_resolution": [],
    "risk_assessment": [],
    "equity_audit": [],
    "cumulative_impact": [],
}


# ─────────────────────── Keyword extraction ───────────────────────

def _extract_intents_keyword(query: str) -> list[str]:
    q = query.lower()
    found: list[str] = []
    for pattern, intent in _INTENT_PATTERNS:
        if re.search(pattern, q) and intent not in found:
            found.append(intent)
    return found


# ─────────────────────── Optional LLM extraction ───────────────────────

_LLM_PROMPT = """Extract the workflow intents from this natural-language query.
Return ONLY a JSON list of intent IDs (no prose). Valid intent IDs:

  - risk_assessment
  - counseling
  - followup
  - drift_check
  - equity_audit
  - differential_diagnosis
  - cost_effectiveness
  - cumulative_impact
  - patient_resolution

Query: {query!r}

JSON list:"""


def _extract_intents_llm(query: str) -> list[str] | None:
    if os.environ.get("TRUSTEDRISK_DISABLE_LLM", "0") == "1":
        return None
    try:
        import ollama  # type: ignore
    except ImportError:
        return None
    try:
        resp = ollama.generate(
            model=os.environ.get("TRUSTEDRISK_ORCHESTRATOR_LLM_MODEL",
                                    "llama3.1:8b"),
            prompt=_LLM_PROMPT.format(query=query),
            options={"temperature": 0.0},
        )
        raw = str(resp.get("response", "")).strip()
    except Exception:
        return None
    import json as _json
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return None
    try:
        parsed = _json.loads(m.group(0))
    except _json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    valid = {iid for iid in _INTENT_AGENT}
    return [str(x) for x in parsed if isinstance(x, str) and x in valid]


# ─────────────────────── Topological sort ───────────────────────

def _topological_order(intents: list[str]) -> list[str]:
    """Order intents so dependencies execute first.

    Cycle-safe by construction (the dependency graph is a DAG by design).
    """
    available = set(intents)
    ordered: list[str] = []
    in_progress: set[str] = set()

    def visit(intent: str) -> None:
        if intent in ordered:
            return
        if intent in in_progress:
            return  # break cycle defensively
        in_progress.add(intent)
        for dep in _INTENT_DEPENDENCIES.get(intent, []):
            if dep in available:
                visit(dep)
        in_progress.discard(intent)
        if intent not in ordered:
            ordered.append(intent)

    for intent in intents:
        visit(intent)
    return ordered


# ─────────────────────── Plan + Execute ───────────────────────

def plan_workflow(
    natural_query: str,
    use_llm: bool = False,
) -> WorkflowPlan:
    """Build an execution plan from a natural-language workflow description."""
    if not isinstance(natural_query, str) or not natural_query.strip():
        return WorkflowPlan(
            natural_query=natural_query or "",
            extraction_method="keyword",
            intents=[],
            execution_order=[],
            rationale="Empty query -- no intents extractable.",
        )

    extraction_method = "keyword"
    intents = _extract_intents_keyword(natural_query)

    if use_llm:
        llm_intents = _extract_intents_llm(natural_query)
        if llm_intents is not None:
            extraction_method = "llm"
            # Take the union -- LLM can refine but never drop deterministic hits
            for i in llm_intents:
                if i not in intents:
                    intents.append(i)

    ordered = _topological_order(intents)
    nodes = [
        WorkflowIntent(
            intent_id=iid,                # type: ignore[arg-type]
            agent=_INTENT_AGENT[iid],     # type: ignore[arg-type]
            description=_INTENT_DESCRIPTION[iid],
            depends_on=[d for d in _INTENT_DEPENDENCIES.get(iid, [])
                          if d in intents],
        )
        for iid in ordered
    ]
    rationale = (
        f"Extracted {len(intents)} intent(s) via {extraction_method}: "
        f"{', '.join(intents) if intents else '(none)'}. "
        f"Execution order: {' -> '.join(ordered) if ordered else '(no-op)'}."
    )
    return WorkflowPlan(
        natural_query=natural_query,
        extraction_method=extraction_method,    # type: ignore[arg-type]
        intents=nodes,
        execution_order=ordered,
        rationale=rationale,
    )


# ─────────────────────── In-process execution adapters ───────────────────────

class _StubRequest:
    """Minimal Request stub for in-process partner-agent invocation."""
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.headers = {"content-length": "1"}

    async def json(self) -> dict[str, Any]:
        return self._payload


def _summarize_payload(payload: Any, max_chars: int = 240) -> str:
    """One-line summary suitable for the WorkflowExecutionStep.output_summary."""
    if isinstance(payload, dict):
        keys = sorted(list(payload.keys()))[:8]
        return f"keys={keys}"
    return str(payload)[:max_chars]


async def _execute_intent(
    intent: WorkflowIntent,
    context: dict[str, Any],
) -> WorkflowExecutionStep:
    """Run a single intent against the in-process partner agent and capture
    a structured execution step."""
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    err: str | None = None
    success = True
    summary = ""

    try:
        if intent.intent_id == "risk_assessment":
            from mcp_server.tools.readmission_risk import (
                compute_readmission_risk,
            )
            patient_id = context.get("patient_id")
            risk = await compute_readmission_risk(patient_id=patient_id)
            payload = risk.model_dump(mode="json")
            context["risk_estimate"] = payload
            summary = (
                f"probability_mean={payload['probability_mean']:.3f}, "
                f"lace_raw_score={payload['lace_raw_score']}"
            )

        elif intent.intent_id == "counseling":
            from mcp_server.tools.discharge_counseling import (
                compute_discharge_counseling,
            )
            counseling = await compute_discharge_counseling(
                patient_id=context.get("patient_id"))
            payload = counseling.model_dump(mode="json")
            context["counseling"] = payload
            summary = (
                f"sections={len(payload.get('sections', []))}, "
                f"reading_level={payload.get('reading_level_grade')}"
            )

        elif intent.intent_id == "differential_diagnosis":
            from mcp_server.tools.differential_diagnosis_ranker import (
                compute_differential_diagnosis_ranker,
            )
            cc = context.get("chief_complaint", "chest pain")
            features = context.get("features", {})
            free_text = context.get("free_text_summary")
            ddx = await compute_differential_diagnosis_ranker(
                chief_complaint=cc,
                structured_features=features,
                free_text_summary=free_text,
            )
            payload = ddx.model_dump(mode="json")
            context["ddx"] = payload
            summary = (
                f"items={len(payload.get('items', []))}, "
                f"cant_miss={len(payload.get('cant_miss_diagnoses', []))}"
            )

        elif intent.intent_id == "cost_effectiveness":
            from mcp_server.tools.expected_value_of_intervention import (
                compute_expected_value_of_intervention,
            )
            risk = context.get("risk_estimate")
            evi = await compute_expected_value_of_intervention(
                intervention=context.get("intervention", {
                    "name": "Pharmacist-led counseling",
                    "relative_risk_reduction": 0.30,
                    "cost_per_patient_usd": 75.0,
                    "evidence_grade": "A",
                }),
                risk_estimate=risk,
                cohort_size=context.get("cohort_size", 100),
            )
            payload = evi.model_dump(mode="json")
            context["cost_effectiveness"] = payload
            summary = f"decision={payload['decision']}"

        elif intent.intent_id == "patient_resolution":
            from mcp_server.tools.conversational_resolver import (
                compute_resolve_patient_from_query,
            )
            q = context.get("nl_query", context.get("query") or "")
            res = await compute_resolve_patient_from_query(query=q)
            payload = res.model_dump(mode="json")
            if payload.get("resolved_patient_id"):
                context["patient_id"] = payload["resolved_patient_id"]
            summary = f"resolved={payload.get('resolved_patient_id')}"

        elif intent.intent_id == "followup":
            from apps.scheduler_agent import server as sched_srv
            decision_card = context.get("decision_card") or _synthesize_card(
                context)
            req = _StubRequest({
                "decision_card": decision_card,
                "discharge_date": context.get("discharge_date"),
            })
            resp = await sched_srv.propose_followup_visits(req)
            import json
            payload = json.loads(resp.body)
            context["followup_plan"] = payload
            summary = f"visits={len(payload.get('visits', []))}"

        elif intent.intent_id == "drift_check":
            from apps.alert_agent import server as alert_srv
            drift_report = context.get("drift_report") or {
                "overall_tier": "warn",
                "rationale": "Synthetic warn-tier drift",
                "signals": [{"name": "ece_post_hoc", "value": 0.13,
                              "baseline": 0.08, "tier": "warn",
                              "detail": "drift +5pp"}],
                "window_label": "demo", "window_n_cards": 0,
            }
            req = _StubRequest({"drift_report": drift_report})
            resp = await alert_srv.check_drift_now(req)
            import json
            payload = json.loads(resp.body)
            context["drift_alert"] = payload
            summary = f"tier={payload.get('tier') or payload.get('severity')}"

        elif intent.intent_id == "equity_audit":
            from a2a_agent.equity_dashboard import compute_equity_dashboard
            cohort = context.get("equity_cohort", [])
            dash = compute_equity_dashboard(cohort)
            payload = dash.model_dump(mode="json")
            context["equity_dashboard"] = payload
            summary = (
                f"segments={payload['n_total_decisions']}, "
                f"flagged={len(payload.get('flagged_segments', []))}"
            )

        elif intent.intent_id == "cumulative_impact":
            from a2a_agent.impact_aggregator import aggregate_impact_kpis
            decisions = context.get("decisions", [])
            kpi = aggregate_impact_kpis(decisions=decisions)
            payload = kpi.model_dump(mode="json")
            context["impact_kpis"] = payload
            summary = (
                f"n_decisions={payload['n_decisions']}, "
                f"events_avoided={payload['estimated_events_avoided']:.1f}"
            )

        else:
            success = False
            err = f"unknown_intent:{intent.intent_id}"
    except Exception as exc:  # noqa: BLE001
        success = False
        err = f"{type(exc).__name__}: {exc}"
        summary = ""

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return WorkflowExecutionStep(
        intent_id=intent.intent_id,
        agent=intent.agent,
        started_at_iso=started.isoformat(),
        duration_ms=dt_ms,
        success=success,
        output_summary=summary,
        error=err,
    )


def _synthesize_card(context: dict[str, Any]) -> dict[str, Any]:
    """Compose a minimal DecisionCard from the accumulated context.

    Used when downstream intents (e.g. followup) need a card but no
    upstream produced one explicitly. When no risk_estimate is present
    in the context, returns an abstain card -- never a synthetic
    low-risk default. A downstream consumer must NOT treat the absence
    of an upstream estimate as a low-risk finding.
    """
    risk = context.get("risk_estimate")
    if not isinstance(risk, dict) or "probability_mean" not in risk:
        return {
            "recommendation": {"action": "abstain",
                                  "confidence": "none"},
            "reasoning": {"risk_estimate": None},
            "audit": {
                "request_id": context.get("request_id", "orchestrate-1"),
                "tools": [],
            },
            "validation": {},
            "abstain": ["risk_assessment_skipped"],
            "abstain_recommended": True,
            "abstain_reason": (
                "synthesized_card_without_risk_estimate: a downstream "
                "intent requested a DecisionCard but no upstream "
                "intent produced a calibrated risk estimate. The "
                "orchestrator refuses to substitute a low-risk default "
                "because that would let the consumer treat absence of "
                "data as a clinical finding."
            ),
        }
    return {
        "recommendation": {"action": "home_with_care",
                              "confidence": "medium"},
        "reasoning": {"risk_estimate": {
            "probability_mean": float(risk["probability_mean"]),
        }},
        "audit": {"request_id": context.get("request_id", "orchestrate-1"),
                    "tools": []},
        "validation": {},
        "abstain": [],
    }


async def execute_workflow(
    plan: WorkflowPlan,
    *,
    initial_context: dict[str, Any] | None = None,
) -> WorkflowExecutionResult:
    """Execute the plan in topological order, passing a shared context."""
    context = dict(initial_context or {})
    steps: list[WorkflowExecutionStep] = []
    intent_by_id = {n.intent_id: n for n in plan.intents}

    for intent_id in plan.execution_order:
        intent = intent_by_id[intent_id]
        step = await _execute_intent(intent, context)
        steps.append(step)

    overall = all(s.success for s in steps) and bool(steps)
    summary_parts = [
        f"{s.intent_id}: {s.output_summary}" if s.success
        else f"{s.intent_id}: ERROR ({s.error})"
        for s in steps
    ]
    composite = "; ".join(summary_parts) or "no-op"

    return WorkflowExecutionResult(
        plan=plan,
        steps=steps,
        overall_success=overall,
        composite_summary=composite,
    )
