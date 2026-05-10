"""TIME-2 -- Incremental risk recompute.

Updates a prior `RiskEstimate` in response to one new FHIR resource event
without re-running the full LACE feature extraction pipeline. The
mapping from event-type to LACE component is deterministic:

  - Encounter (inpatient) admission -> L (length of stay starts), A (acuity)
  - Encounter (ED) -> E (ED visits 6-month count)
  - Condition -> C (Charlson comorbidity)
  - Observation, MedicationRequest, Procedure -> no LACE impact in v1

When the new event affects a component, we delta-update the LACE total +
look up the new posterior in the same lookup_table the full pipeline uses.
This is a fast-path; for full audit correctness, periodic full recomputes
are still recommended (returned via `full_recompute_recommended=True`
when the delta exceeds a tunable threshold).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.schemas import IncrementalRiskUpdate


_FULL_RECOMPUTE_DELTA_THRESHOLD = 0.10


def _load_coefficients(path: Path | None = None) -> dict[str, Any]:
    coef_path = path or Path("data/coefficients.json")
    if not coef_path.exists():
        raise FileNotFoundError(f"coefficients.json not found at {coef_path}")
    return json.loads(coef_path.read_text(encoding="utf-8"))


def _lookup_probability(coef: dict[str, Any], lace_total: int) -> float:
    lookup = (coef.get("runtime_coefficients") or {}).get("lookup_table", {})
    entry = lookup.get(str(max(0, min(19, lace_total))))
    if entry is None:
        raise RuntimeError(
            f"coefficients.json lookup_table missing entry for LACE={lace_total}")
    return float(entry["prob_mean"])


# ─────────────────────── Resource-type -> component mapping ───────────────────────

def _classify_observation(obs: dict[str, Any]) -> tuple[list[str], int]:
    """Return (affected_components, delta_to_lace_total)."""
    rtype = (obs.get("resourceType") or "").lower()

    if rtype == "encounter":
        cls = (obs.get("class") or {})
        code = (cls.get("code") if isinstance(cls, dict) else "") or ""
        code = code.upper()
        if code in ("EMER", "ED", "EMERGENCY"):
            # Each ED visit raises E by 1, capped at 4
            return ["E"], 1
        if code in ("IMP", "ACUTE", "INPATIENT"):
            # New inpatient admission resets the L/A counters; in the
            # incremental pathway we treat it as a partial reset signaling
            # full recomputation needed.
            return ["L", "A"], 0
        return [], 0

    if rtype == "condition":
        return ["C"], 1

    return [], 0


def recompute_with_observation(
    prior_risk: dict[str, Any],
    new_resource: dict[str, Any],
    coefficients_path: Path | None = None,
) -> IncrementalRiskUpdate:
    """Update a prior risk estimate given one new FHIR resource event.

    Args:
        prior_risk: dict with at least `lace_raw_score` and `probability_mean`.
        new_resource: a FHIR resource dict (Encounter / Condition / etc.).

    Returns:
        IncrementalRiskUpdate with prior, posterior, delta, and the affected
        LACE components. When the trigger is an admission (which resets
        L/A counters), `full_recompute_recommended=True`.
    """
    if not isinstance(prior_risk, dict):
        raise ValueError("prior_risk must be a dict.")
    if not isinstance(new_resource, dict):
        raise ValueError("new_resource must be a dict.")

    prior_prob = float(prior_risk.get("probability_mean") or 0.0)
    prior_lace = int(prior_risk.get("lace_raw_score") or 0)

    components, delta_lace = _classify_observation(new_resource)
    rtype = (new_resource.get("resourceType") or "unknown")

    if not components:
        return IncrementalRiskUpdate(
            prior_probability=prior_prob,
            posterior_probability=prior_prob,
            delta=0.0,
            affected_lace_components=[],
            trigger_observation_type=rtype,
            rationale=(
                f"{rtype} does not affect any LACE component "
                "(no incremental update needed)."
            ),
        )

    # Inpatient admissions reset L/A; defer to full pipeline
    full_recompute = False
    if rtype.lower() == "encounter" and \
            "L" in components and "A" in components:
        full_recompute = True
        return IncrementalRiskUpdate(
            prior_probability=prior_prob,
            posterior_probability=prior_prob,
            delta=0.0,
            affected_lace_components=components,
            trigger_observation_type="Encounter",
            rationale=(
                "Inpatient Encounter admission detected -- L and A "
                "counters reset. Schedule a full recompute via "
                "compute_readmission_risk."
            ),
            full_recompute_recommended=True,
        )

    new_lace = max(0, min(19, prior_lace + delta_lace))
    coef = _load_coefficients(coefficients_path)
    new_prob = _lookup_probability(coef, new_lace)
    delta = new_prob - prior_prob

    if abs(delta) >= _FULL_RECOMPUTE_DELTA_THRESHOLD:
        full_recompute = True

    rationale = (
        f"{rtype} affected components {components}: "
        f"LACE {prior_lace} -> {new_lace}, "
        f"probability {prior_prob:.3f} -> {new_prob:.3f} (Δ={delta:+.3f})."
    )

    return IncrementalRiskUpdate(
        prior_probability=prior_prob,
        posterior_probability=new_prob,
        delta=round(delta, 4),
        affected_lace_components=components,
        trigger_observation_type=rtype,
        rationale=rationale,
        full_recompute_recommended=full_recompute,
    )
