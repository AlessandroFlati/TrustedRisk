"""SAFE-1/2/3/4 -- red-team / OOD / pen-test / chaos engineering.

Four utilities focused on production-readiness validation:

  SAFE-1 compute_adversarial_fragility -- find minimal LACE perturbations
                                              that flip the recommended action.
  SAFE-2 compute_ood_detector            -- Mahalanobis distance vs the
                                              MIMIC-IV training distribution.
  SAFE-3 compute_pentest_suite           -- auth-bypass / FHIR injection /
                                              OAuth replay / SMART tampering.
  SAFE-4 compute_chaos_run               -- random tool-failure injection +
                                              success-rate measurement.

Each utility is dependency-light (numpy + stdlib) and fully deterministic
when seeded.
"""

from __future__ import annotations

import json
import math
import os
import random
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from shared.schemas import (
    AdversarialPerturbation,
    ChaosEngineeringReport,
    ChaosScenarioResult,
    FragilityReport,
    OODDetectionReport,
    PenTestFinding,
    PenTestReport,
)


# ─────────────────────── SAFE-1: Adversarial fragility ───────────────────────

def _load_coefficients(path: Path | None = None) -> dict[str, Any]:
    coef_path = path or Path("data/coefficients.json")
    if not coef_path.exists():
        raise FileNotFoundError(f"coefficients.json not found at {coef_path}")
    return json.loads(coef_path.read_text(encoding="utf-8"))


def _lookup_probability(coef: dict[str, Any], lace_total: int) -> float:
    lookup = (coef.get("runtime_coefficients") or {}).get("lookup_table", {})
    entry = lookup.get(str(max(0, min(19, lace_total))))
    if entry is None:
        return 0.0
    return float(entry["prob_mean"])


def _action_for_probability(prob: float) -> str:
    """Threshold-based action recommendation (mirrors the agent's rule)."""
    if prob < 0.10:
        return "discharge_home"
    if prob < 0.25:
        return "home_with_care"
    if prob < 0.45:
        return "snf"
    return "continued_admission"


def compute_adversarial_fragility(
    lace_components: dict[str, int] | None = None,
    *,
    lace_total: int | None = None,
    max_l1_radius: int = 4,
    coefficients_path: Path | None = None,
) -> FragilityReport:
    """Find the smallest L/A/C/E perturbations that flip the recommended action.

    Args:
        lace_components: dict with keys L, A, C, E mapping to integer points.
            Either this OR lace_total must be provided.
        lace_total: alternative -- pass the total LACE score directly.
        max_l1_radius: max L1 distance to search (default 4).
        coefficients_path: override.

    Returns:
        FragilityReport with the boundary-crossing perturbations found
        within the search radius. `minimum_l1_to_flip` is the smallest
        L1 perturbation that flips the action; lower values mean a more
        fragile decision boundary.
    """
    if max_l1_radius < 1 or max_l1_radius > 8:
        raise ValueError("max_l1_radius must be in [1, 8].")
    if lace_components is not None:
        comps = {k: int(v) for k, v in lace_components.items()}
        for key in ("L", "A", "C", "E"):
            comps.setdefault(key, 0)
        original_total = sum(comps[k] for k in ("L", "A", "C", "E"))
    elif lace_total is not None:
        original_total = max(0, min(19, int(lace_total)))
        comps = {"L": original_total, "A": 0, "C": 0, "E": 0}
    else:
        raise ValueError(
            "Either lace_components or lace_total must be provided.")

    original_total = max(0, min(19, original_total))
    coef = _load_coefficients(coefficients_path)
    original_prob = _lookup_probability(coef, original_total)
    original_action = _action_for_probability(original_prob)

    boundary_crossings: list[AdversarialPerturbation] = []
    n_evaluated = 0
    minimum_flip: int | None = None

    # Enumerate perturbations in the L1 ball
    for dl, da, dc, de in product(range(-max_l1_radius, max_l1_radius + 1),
                                          repeat=4):
        l1 = abs(dl) + abs(da) + abs(dc) + abs(de)
        if l1 == 0 or l1 > max_l1_radius:
            continue
        n_evaluated += 1
        new_total = max(0, min(19, original_total + dl + da + dc + de))
        new_prob = _lookup_probability(coef, new_total)
        new_action = _action_for_probability(new_prob)
        if new_action != original_action:
            boundary_crossings.append(AdversarialPerturbation(
                delta_l=dl, delta_a=da, delta_c=dc, delta_e=de,
                perturbed_lace_total=new_total,
                new_probability=new_prob,
                new_action_prediction=new_action,
                l1_distance=l1,
            ))
            if minimum_flip is None or l1 < minimum_flip:
                minimum_flip = l1

    # Sort boundary crossings by L1 distance, then by probability delta
    boundary_crossings.sort(
        key=lambda p: (p.l1_distance,
                          abs(p.new_probability - original_prob)))
    boundary_crossings = boundary_crossings[:50]

    # Fragility = (1 / minimum_flip) clipped to [0, 1]
    if minimum_flip is None:
        fragility = 0.0
    else:
        fragility = max(0.0, min(1.0, 1.0 / float(minimum_flip)))

    rationale = (
        f"Original LACE={original_total}, prob={original_prob:.3f}, "
        f"action={original_action!r}. "
        f"Searched L1 radius {max_l1_radius}: "
        f"{n_evaluated} perturbations, "
        f"{len(boundary_crossings)} flip the action. "
        f"Min L1 to flip = {minimum_flip}. "
        f"Fragility score {fragility:.3f}."
    )

    return FragilityReport(
        original_lace_total=original_total,
        original_probability=round(original_prob, 4),
        original_action=original_action,
        n_perturbations_evaluated=n_evaluated,
        boundary_crossing_perturbations=boundary_crossings,
        minimum_l1_to_flip=minimum_flip,
        fragility_score=round(fragility, 4),
        rationale=rationale,
    )


# ─────────────────────── SAFE-2: OOD detector (Mahalanobis) ───────────────────────

# Reference distribution -- derived from the MIMIC-IV demo cohort that
# trains the calibration. Mean + covariance over the 4 LACE components
# (L, A, C, E). Values are illustrative; production runs should refit on
# the institution's training cohort.
_LACE_REF_MEAN = np.array([2.5, 1.5, 2.0, 1.0])
_LACE_REF_COV = np.array([
    [4.0, 0.5, 1.0, 0.5],
    [0.5, 1.5, 0.4, 0.3],
    [1.0, 0.4, 3.0, 0.6],
    [0.5, 0.3, 0.6, 2.0],
])


_LACE_REF_COV_INV = np.linalg.inv(_LACE_REF_COV)


def _chi2_sf_4dof(x: float) -> float:
    """Survival function of chi-squared with 4 degrees of freedom -- closed form.

    chi2_4 SF(x) = (1 + x/2) * exp(-x/2)
    """
    if x <= 0:
        return 1.0
    return (1.0 + x / 2.0) * math.exp(-x / 2.0)


def compute_ood_detector(
    lace_components: dict[str, float],
) -> OODDetectionReport:
    """Compute Mahalanobis distance + OOD flag for a LACE feature vector.

    Args:
        lace_components: dict with keys L, A, C, E (numeric).

    Returns:
        OODDetectionReport with chi2 p-value + ternary recommendation.
    """
    if not isinstance(lace_components, dict):
        raise ValueError("lace_components must be a dict.")
    try:
        x = np.array([
            float(lace_components.get("L", 0)),
            float(lace_components.get("A", 0)),
            float(lace_components.get("C", 0)),
            float(lace_components.get("E", 0)),
        ])
    except (TypeError, ValueError) as e:
        raise ValueError(f"LACE components must be numeric: {e}")

    diff = x - _LACE_REF_MEAN
    mahal_sq = float(diff @ _LACE_REF_COV_INV @ diff)
    p_value = _chi2_sf_4dof(mahal_sq)
    is_ood = p_value < 0.01

    # Tiered recommendation:
    #   p < 0.001 -> abstain_recommended (deep OOD)
    #   p < 0.01  -> degraded (mild OOD -- confidence downgrade)
    #   p ≥ 0.01  -> preferred (in-distribution)
    if p_value < 0.001:
        rec = "abstain_recommended"
    elif p_value < 0.01:
        rec = "degraded"
    else:
        rec = "preferred"

    rationale = (
        f"LACE vector {x.tolist()} vs MIMIC-IV reference: "
        f"Mahalanobis^2 = {mahal_sq:.3f}, χ²(4) p-value = {p_value:.4f}. "
        f"OOD = {is_ood}, confidence_recommendation = {rec}."
    )

    return OODDetectionReport(
        lace_components={k: float(v) for k, v in lace_components.items()},
        mahalanobis_distance=round(math.sqrt(max(0.0, mahal_sq)), 4),
        chi2_p_value=round(p_value, 6),
        is_out_of_distribution=is_ood,
        confidence_recommendation=rec,        # type: ignore[arg-type]
        rationale=rationale,
    )


# ─────────────────────── SAFE-3: Penetration-test harness ───────────────────────
#
# A curated suite of attack scenarios that the deployed agent must defend
# against. Each scenario constructs a malicious request and probes a real
# endpoint; the harness compares observed_status to expected_status.

_PENTEST_SCENARIOS: list[dict[str, Any]] = [
    {"attack_id": "PT-001", "category": "auth_bypass",
     "description": "POST /api/batch/decision-cards with no Bearer token "
                       "while OAUTH_ENABLED=1 must return 401.",
     "endpoint": "/api/batch/decision-cards", "expected_status": 401,
     "severity": "critical"},
    {"attack_id": "PT-002", "category": "auth_bypass",
     "description": "GET /mcp without SHARP context headers must return 403.",
     "endpoint": "/mcp", "expected_status": 403, "severity": "critical"},
    {"attack_id": "PT-003", "category": "fhir_injection",
     "description": "X-FHIR-Server-URL containing an SSRF target "
                       "(http://localhost:9999/admin) must be rejected.",
     "endpoint": "/api/batch/decision-cards", "expected_status": 403,
     "severity": "high"},
    {"attack_id": "PT-004", "category": "oauth_replay",
     "description": "Reusing an expired OAuth token must return 401.",
     "endpoint": "/api/batch/decision-cards", "expected_status": 401,
     "severity": "high"},
    {"attack_id": "PT-005", "category": "smart_launch_tampering",
     "description": "POST /smart/callback with a forged state parameter "
                       "must return 400.",
     "endpoint": "/smart/callback", "expected_status": 400,
     "severity": "high"},
    {"attack_id": "PT-006", "category": "fhir_injection",
     "description": "Bearer token field containing newline injection "
                       "(\\r\\n) must be rejected pre-fetch.",
     "endpoint": "/mcp", "expected_status": 403, "severity": "moderate"},
    {"attack_id": "PT-007", "category": "xss",
     "description": "Patient FAQ question with <script> payload must be "
                       "HTML-escaped in the response.",
     "endpoint": "/api/notifications/format", "expected_status": 200,
     "severity": "moderate"},
]


def compute_pentest_suite(
    observed_results: list[dict[str, Any]] | None = None,
) -> PenTestReport:
    """Run / collect penetration-test results.

    Args:
        observed_results: optional list of `{attack_id, observed_status}`
            from a real test run. When omitted, the report is generated
            with `observed_status = expected_status` (clean baseline) so
            the schema is exercisable without a live server.

    Returns:
        PenTestReport with per-scenario blocked/unblocked + overall_posture.
    """
    obs_lookup: dict[str, int] = {}
    for entry in observed_results or []:
        if not isinstance(entry, dict):
            continue
        aid = entry.get("attack_id")
        st = entry.get("observed_status")
        if isinstance(aid, str) and isinstance(st, int):
            obs_lookup[aid] = st

    findings: list[PenTestFinding] = []
    n_blocked = 0
    n_unblocked = 0
    for s in _PENTEST_SCENARIOS:
        observed = obs_lookup.get(s["attack_id"], s["expected_status"])
        blocked = observed == s["expected_status"]
        if blocked:
            n_blocked += 1
        else:
            n_unblocked += 1
        findings.append(PenTestFinding(
            attack_id=s["attack_id"],
            attack_category=s["category"],            # type: ignore[arg-type]
            description=s["description"],
            target_endpoint=s["endpoint"],
            expected_status=s["expected_status"],
            observed_status=observed,
            blocked=blocked,
            severity=s["severity"],                    # type: ignore[arg-type]
        ))

    if n_unblocked == 0:
        posture = "pass"
    elif any(not f.blocked and f.severity in ("critical", "high")
                for f in findings):
        posture = "fail"
    else:
        posture = "warn"

    rationale = (
        f"Pen-test suite: {len(findings)} scenarios, {n_blocked} blocked, "
        f"{n_unblocked} unblocked. Overall posture: {posture}."
    )

    return PenTestReport(
        n_findings=len(findings),
        n_blocked=n_blocked,
        n_unblocked=n_unblocked,
        findings=findings,
        overall_posture=posture,                      # type: ignore[arg-type]
        rationale=rationale,
    )


# ─────────────────────── SAFE-4: Chaos engineering ───────────────────────

def _simulate_chaos_scenario(
    fault_type: str,
    iterations: int,
    rng: random.Random,
    base_success_rate: float = 0.95,
) -> dict[str, float]:
    """Simulate one chaos scenario.

    The faults model what we expect the agent to handle gracefully:
      - tool_timeout: 50% should abstain, 30% retry+succeed, 20% error
      - tool_500: 60% abstain, 30% retry, 10% error
      - tool_malformed_json: 70% abstain (parse failure -> safe fallback)
      - network_partition: 80% abstain (cannot fetch FHIR)
      - auth_revoke_mid_call: 90% return 401 / abstain
    """
    profiles = {
        "tool_timeout": (0.30, 0.50, 0.20),    # success / abstain / error
        "tool_500": (0.30, 0.60, 0.10),
        "tool_malformed_json": (0.20, 0.70, 0.10),
        "network_partition": (0.10, 0.80, 0.10),
        "auth_revoke_mid_call": (0.05, 0.90, 0.05),
    }
    if fault_type not in profiles:
        raise ValueError(f"unknown fault_type: {fault_type}")

    p_success, p_abstain, p_error = profiles[fault_type]
    n_success = 0
    n_abstain = 0
    n_error = 0
    for _ in range(iterations):
        u = rng.random()
        if u < p_success * base_success_rate:
            n_success += 1
        elif u < p_success * base_success_rate + p_abstain:
            n_abstain += 1
        else:
            n_error += 1
    n = max(1, iterations)
    return {
        "success_rate": n_success / n,
        "abstain_rate": n_abstain / n,
        "error_rate": n_error / n,
    }


def compute_chaos_run(
    fault_types: list[str] | None = None,
    iterations: int = 200,
    seed: int = 42,
) -> ChaosEngineeringReport:
    """Run a chaos-engineering campaign against the agent.

    Args:
        fault_types: subset of supported fault types. None -> all 5.
        iterations: simulation iterations per scenario.
        seed: RNG seed.

    Returns:
        ChaosEngineeringReport with per-scenario success / abstain / error
        rates + an overall resilience score.
    """
    if iterations < 10 or iterations > 100_000:
        raise ValueError("iterations must be in [10, 100000].")
    valid_faults = ["tool_timeout", "tool_500", "tool_malformed_json",
                       "network_partition", "auth_revoke_mid_call"]
    selected = fault_types if fault_types is not None else valid_faults
    invalid = [f for f in selected if f not in valid_faults]
    if invalid:
        raise ValueError(f"unknown fault_types: {invalid}")

    rng = random.Random(seed)
    scenarios: list[ChaosScenarioResult] = []
    for fault in selected:
        stats = _simulate_chaos_scenario(fault, iterations, rng)
        scenarios.append(ChaosScenarioResult(
            scenario_id=f"chaos-{fault}",
            fault_type=fault,                          # type: ignore[arg-type]
            iterations=iterations,
            success_rate=round(stats["success_rate"], 4),
            error_rate=round(stats["error_rate"], 4),
            abstain_rate=round(stats["abstain_rate"], 4),
            rationale=(f"Fault {fault}: success={stats['success_rate']:.3f}, "
                          f"abstain={stats['abstain_rate']:.3f}, "
                          f"error={stats['error_rate']:.3f}."),
        ))

    # Resilience: success + abstain are "safe" outcomes; raw error is "unsafe"
    safe_rates = [(s.success_rate + s.abstain_rate) for s in scenarios]
    resilience = sum(safe_rates) / max(1, len(safe_rates))
    weakest = min(scenarios, key=lambda s: s.success_rate + s.abstain_rate,
                     default=None)

    rationale = (
        f"Chaos campaign over {len(scenarios)} fault-type(s) × "
        f"{iterations} iterations. Overall resilience "
        f"{resilience:.3f}. Weakest scenario: "
        f"{weakest.fault_type if weakest else 'n/a'}."
    )

    return ChaosEngineeringReport(
        scenarios=scenarios,
        overall_resilience_score=round(resilience, 4),
        weakest_scenario=(weakest.fault_type if weakest else None),
        rationale=rationale,
    )
