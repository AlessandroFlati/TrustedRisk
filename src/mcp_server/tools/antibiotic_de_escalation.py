"""healthcare.compute_antibiotic_de_escalation -- culture-driven narrowing + IV->PO.

Inputs:
  - current empiric regimen
  - identified pathogen (or none)
  - susceptibility pattern (S/I/R per drug class)
  - clinical course (afebrile 24-48h, tolerating PO, etc.)

Outputs:
  - whether to de-escalate
  - target narrower regimen
  - whether IV-to-PO switch criteria are met
  - duration recommendation

References:
  IDSA / SHEA Antimicrobial Stewardship Implementation Guidelines (2016).
  Cunha BA. Antibiotic stewardship: switching from intravenous to oral
    therapy. Infect Dis Clin North Am 2017;31(3):485-503.
"""

from __future__ import annotations

from typing import Any, Literal

from shared.schemas import DeEscalationPlan


# ─────────────────────────────────────────────────────────────────────
# Pathogen -> narrowest acceptable agent map (very short list)
# ─────────────────────────────────────────────────────────────────────

# Each entry: pathogen -> list of (susceptible_class_required, oral_option_label, iv_option_label)
_NARROW_TARGETS: dict[str, dict[str, str]] = {
    "e_coli": {
        "S_to_ampicillin": "amoxicillin 500 mg PO TID",
        "S_to_tmp_smx": "TMP-SMX DS PO BID",
        "S_to_ciprofloxacin": "ciprofloxacin 500 mg PO BID",
        "S_to_cephalexin": "cephalexin 500 mg PO QID",
    },
    "e_coli_esbl": {
        "S_to_ertapenem": "ertapenem 1 g IV q24h (no oral equivalent for ESBL)",
        "S_to_fosfomycin": "fosfomycin 3 g PO single-dose (cystitis only)",
    },
    "klebsiella_pneumoniae": {
        "S_to_ceftriaxone": "ceftriaxone 1 g IV q24h",
        "S_to_ciprofloxacin": "ciprofloxacin 500 mg PO BID",
    },
    "staphylococcus_aureus_mssa": {
        "S_to_oxacillin": "cefazolin 2 g IV q8h (or oxacillin)",
        "po_alt": "cephalexin 500 mg PO QID",
    },
    "staphylococcus_aureus_mrsa": {
        "S_to_vancomycin": "vancomycin 15-20 mg/kg IV q12h (continue)",
        "po_alt": "linezolid 600 mg PO BID OR doxycycline 100 mg PO BID",
    },
    "streptococcus_pneumoniae": {
        "S_to_penicillin": "amoxicillin 1 g PO TID",
    },
    "pseudomonas_aeruginosa": {
        "S_to_cefepime": "cefepime 2 g IV q8h",
        "S_to_ciprofloxacin": "ciprofloxacin 750 mg PO BID",
    },
}


# ─────────────────────────────────────────────────────────────────────
# IV -> PO switch criteria
# ─────────────────────────────────────────────────────────────────────

def _iv_to_po_criteria(factors: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """Check the IDSA "5 Cs" + tolerating PO criteria for IV-to-PO switch.

      1. Clinically improving (afebrile 24-48h, hemodynamically stable)
      2. Capable of oral intake (no dysphagia, no severe N/V, no NPO)
      3. Compliant (alert, oriented, can swallow pills)
      4. Comparable bioavailability available (oral option exists for pathogen)
      5. Cost / convenience (always met)
    """
    met: list[str] = []
    unmet: list[str] = []

    if factors.get("afebrile_24h", False):
        met.append("afebrile ≥24h")
    else:
        unmet.append("not yet afebrile 24h")

    if factors.get("hemodynamically_stable", False):
        met.append("hemodynamically stable")
    else:
        unmet.append("hemodynamically unstable or vasopressor-dependent")

    if factors.get("tolerating_po", False):
        met.append("tolerating PO")
    else:
        unmet.append("not tolerating PO (NPO / dysphagia / vomiting)")

    if factors.get("alert_oriented", True):
        met.append("alert and oriented (can swallow safely)")
    else:
        unmet.append("altered mental status -- aspiration risk")

    if not factors.get("malabsorption", False):
        met.append("no malabsorption")
    else:
        unmet.append("malabsorption (post-op gut, severe diarrhea)")

    return (len(unmet) == 0), met, unmet


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_antibiotic_de_escalation(
    current_regimen: str,
    pathogen: str | None = None,
    susceptibility: dict[str, str] | None = None,
    days_on_therapy: int = 0,
    total_planned_duration_days: int = 7,
    clinical_factors: dict[str, Any] | None = None,
    patient_id: str | None = None,
) -> DeEscalationPlan:
    """Decide whether to narrow the antibiotic regimen and / or switch IV->PO.

    Args:
        current_regimen: the current empiric regimen (free text).
        pathogen: identified organism slug (e.g. 'e_coli', 'staphylococcus_aureus_mssa').
            None -> cultures pending -> no de-escalation.
        susceptibility: dict mapping drug class -> 'S' / 'I' / 'R'. Used to
            select the narrowest susceptible option.
        days_on_therapy: days since therapy initiation. De-escalation review
            is typically at day 2-3.
        total_planned_duration_days: total planned course (informational).
        clinical_factors: dict for IV->PO criteria + clinical course.

    Returns:
        DeEscalationPlan.
    """
    factors = clinical_factors or {}
    susc = {k.lower(): v.upper() for k, v in (susceptibility or {}).items()
            if v in ("S", "I", "R", "s", "i", "r")}

    duration_remaining = max(0, total_planned_duration_days - days_on_therapy)

    if pathogen is None:
        # No pathogen yet -- defer
        return DeEscalationPlan(
            patient_id=patient_id,
            current_regimen=current_regimen,
            pathogen_identified=None,
            susceptibility_pattern={},
            de_escalation_recommended=False,
            target_regimen=None,
            iv_to_po_switch_eligible=False,
            duration_total_days=total_planned_duration_days,
            duration_remaining_days=duration_remaining,
            rationale=(
                f"Cultures pending after {days_on_therapy} days of therapy; "
                f"continue empiric regimen and reassess at day 3."
            ),
            abstain_recommended=True,
            abstain_reason="cultures_pending",
        )

    # Find a narrower regimen. Accept both the canonical slug used
    # by the narrowing database (`e_coli`) and the Latin display form
    # the workflow YAML or a CodeableConcept might emit
    # (`Escherichia coli`, abbreviated `E. coli`).
    _PATHOGEN_ALIASES = {
        "escherichia_coli": "e_coli",
        "e._coli": "e_coli",
        "staphylococcus_aureus": "staphylococcus_aureus_mssa",
        "s._aureus": "staphylococcus_aureus_mssa",
    }
    pathogen_norm = pathogen.strip().lower().replace(" ", "_")
    pathogen_norm = _PATHOGEN_ALIASES.get(pathogen_norm, pathogen_norm)
    targets = _NARROW_TARGETS.get(pathogen_norm) or {}

    susceptible_to = [k.replace("s_to_", "") for k, v in susc.items()
                       if v == "S"]

    target_regimen: str | None = None
    target_route: Literal["IV", "PO"] | None = None
    for key, label in targets.items():
        if key.startswith("S_to_"):
            class_name = key.replace("S_to_", "")
            if class_name in susceptible_to or any(class_name in d for d in susceptible_to):
                target_regimen = label
                target_route = "PO" if "PO" in label else "IV"
                break

    if target_regimen is None and pathogen_norm in _NARROW_TARGETS:
        # Pathogen known but no susceptible class flagged -- fall back
        first_label = next(iter(_NARROW_TARGETS[pathogen_norm].values()))
        target_regimen = first_label
        target_route = "PO" if "PO" in first_label else "IV"

    de_escalate = (target_regimen is not None
                    and target_regimen.lower() not in current_regimen.lower())

    iv_to_po_eligible, met, unmet = _iv_to_po_criteria(factors)
    if iv_to_po_eligible and target_route == "PO":
        # Target is already PO -- switch can happen
        pass
    elif iv_to_po_eligible and target_route == "IV":
        # All criteria met but no PO option for the pathogen
        unmet.append(f"no oral option with adequate bioavailability for {pathogen_norm}")
        iv_to_po_eligible = False

    abstain = False
    abstain_reason: str | None = None
    if not de_escalate and target_regimen is None:
        abstain = True
        abstain_reason = f"pathogen_not_in_narrowing_database:{pathogen_norm!r}"

    rationale = _build_rationale(current_regimen, pathogen_norm, susc,
                                   target_regimen, de_escalate,
                                   iv_to_po_eligible, met, unmet,
                                   days_on_therapy, duration_remaining)

    # Concrete recommendations derived from the calculation, not boilerplate.
    # Each line traces to a structured field on this same DeEscalationPlan.
    recs: list[str] = []
    if target_regimen and de_escalate:
        recs.append(
            f"De-escalate to {target_regimen} ({target_route or 'route TBD'}) "
            f"based on culture susceptibility."
        )
    elif target_regimen and not de_escalate:
        recs.append(
            f"Continue {current_regimen}; the target {target_regimen} "
            "matches current empiric coverage."
        )
    if iv_to_po_eligible:
        recs.append(
            f"IV-to-PO switch eligible -- transition to oral "
            f"{target_regimen or 'narrow agent'} when next dose due."
        )
    elif unmet:
        recs.append(
            "Defer IV-to-PO switch until the following criteria are met: "
            + "; ".join(unmet) + "."
        )
    if duration_remaining > 0:
        recs.append(
            f"Plan {duration_remaining} day(s) remaining of antimicrobial "
            f"therapy out of a {total_planned_duration_days}-day total course."
        )

    return DeEscalationPlan(
        patient_id=patient_id,
        current_regimen=current_regimen,
        pathogen_identified=pathogen_norm.replace("_", " "),
        susceptibility_pattern={k: v for k, v in susc.items()},  # type: ignore[arg-type]
        de_escalation_recommended=de_escalate,
        target_regimen=target_regimen,
        target_regimen_route=target_route,
        iv_to_po_switch_eligible=iv_to_po_eligible,
        iv_to_po_criteria_met=met,
        iv_to_po_criteria_unmet=unmet,
        duration_total_days=total_planned_duration_days,
        duration_remaining_days=duration_remaining,
        rationale=rationale,
        recommendations=recs,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
    )


def _build_rationale(current: str, pathogen: str, susc: dict[str, str],
                      target: str | None, de_escalate: bool,
                      iv_po_ok: bool, met: list[str], unmet: list[str],
                      days_on: int, days_left: int) -> str:
    parts = [f"Pathogen: {pathogen.replace('_', ' ')}.",
             f"Current regimen: {current}."]
    if susc:
        ssum = ", ".join(f"{k}={v}" for k, v in susc.items())
        parts.append(f"Susceptibility: {ssum}.")
    if de_escalate and target:
        parts.append(f"De-escalation recommended -> {target}.")
    elif target:
        parts.append(f"Current regimen already at narrowest acceptable: {target}.")
    if iv_po_ok:
        parts.append("IV-to-PO switch criteria MET (" + ", ".join(met) + ").")
    elif met or unmet:
        parts.append(f"IV-to-PO switch deferred: unmet criteria = "
                      f"{'; '.join(unmet) if unmet else 'none'}.")
    parts.append(f"Day {days_on} of therapy; {days_left} day(s) remaining.")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_antibiotic_de_escalation)
