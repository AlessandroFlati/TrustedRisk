"""healthcare.compute_dialysis_initiation_decision -- AEIOU + STARRT-AKI logic.

Emergent dialysis indications (AEIOU mnemonic):
  A -- refractory metabolic Acidosis (pH <7.1 not responding to bicarb)
  E -- refractory hyperkalemia (K+ >6.5 not responding to medical Rx)
  I -- Ingestion of dialyzable toxin (methanol, ethylene glycol, salicylates,
      lithium, theophylline, valproate, metformin)
  O -- volume Overload refractory to diuretics (pulmonary edema)
  U -- Uremia (encephalopathy, pericarditis, bleeding diathesis)

STARRT-AKI (NEJM 2020): in patients without an emergent indication,
  ACCELERATED initiation (within 12h of meeting eligibility) does NOT
  improve 90-day mortality vs STANDARD strategy (wait for emergent
  indication or persistent stage-3 AKI). Default to STANDARD unless
  emergent.

Modality:
  intermittent_hd  -- hemodynamically stable, single dialyzable indication
  crrt             -- hemodynamically unstable, severe shock, ICU
  peritoneal_dialysis -- chronic kidney disease bridging, special circumstances
"""

from __future__ import annotations

from typing import Any, Literal

from shared.schemas import DialysisInitiationReport

from ._chart_inputs import harden_clinical_inputs


# ─────────────────────────────────────────────────────────────────────
# AEIOU evaluator
# ─────────────────────────────────────────────────────────────────────

def _evaluate_aeiou(
    ph: float | None,
    bicarbonate: float | None,
    potassium: float | None,
    refractory_hyperkalemia: bool,
    dialyzable_toxin: str | None,
    volume_overload_refractory: bool,
    uremic_encephalopathy: bool,
    uremic_pericarditis: bool,
    uremic_bleeding: bool,
) -> list[str]:
    indications: list[str] = []

    if ph is not None and ph < 7.1:
        indications.append("A: refractory metabolic acidosis (pH < 7.1)")
    if bicarbonate is not None and bicarbonate < 12 and ph is not None and ph < 7.2:
        indications.append("A: severe acidosis with bicarb < 12")

    if potassium is not None and potassium > 6.5 and refractory_hyperkalemia:
        indications.append(f"E: refractory hyperkalemia (K+ {potassium} > 6.5)")

    if dialyzable_toxin:
        toxin_norm = dialyzable_toxin.strip().lower()
        if toxin_norm in (
            "methanol", "ethylene_glycol", "ethylene-glycol", "salicylate",
            "lithium", "theophylline", "valproate", "metformin",
        ):
            indications.append(f"I: dialyzable toxin ingestion ({toxin_norm})")

    if volume_overload_refractory:
        indications.append("O: volume overload refractory to diuretics")

    if uremic_encephalopathy:
        indications.append("U: uremic encephalopathy")
    if uremic_pericarditis:
        indications.append("U: uremic pericarditis")
    if uremic_bleeding:
        indications.append("U: uremic bleeding diathesis")

    return indications


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_dialysis_initiation_decision(
    aki_stage: str = "no_aki",
    ph: float | None = None,
    bicarbonate_meq_l: float | None = None,
    potassium_meq_l: float | None = None,
    refractory_hyperkalemia: bool = False,
    dialyzable_toxin: str | None = None,
    volume_overload_refractory: bool = False,
    uremic_encephalopathy: bool = False,
    uremic_pericarditis: bool = False,
    uremic_bleeding: bool = False,
    hemodynamically_unstable: bool = False,
    patient_id: str | None = None,
) -> DialysisInitiationReport:
    """Apply AEIOU + STARRT-AKI logic to decide dialysis initiation + modality.

    Args:
        aki_stage: KDIGO stage from compute_aki_kdigo_stage. Stage 3 with no
            emergent indication still falls to STANDARD strategy per
            STARRT-AKI; we surface that as non_urgent.
        ph / bicarbonate / potassium: lab values driving the AEIOU evaluation.
        refractory_*: each indicates the patient has failed standard medical
            management.
        dialyzable_toxin: free-text toxin name; matched against a curated list.
        hemodynamically_unstable: drives modality (CRRT vs intermittent HD).
    """
    _r, _sb, _ = await harden_clinical_inputs(
        {"ph": ph, "bicarbonate_meq_l": bicarbonate_meq_l,
         "potassium_meq_l": potassium_meq_l},
        chart_derivable={"ph", "bicarbonate_meq_l", "potassium_meq_l"},
    )
    if _sb:
        ph = _r.get("ph") if _r.get("ph") is not None else ph
        bicarbonate_meq_l = _r.get("bicarbonate_meq_l") if _r.get("bicarbonate_meq_l") is not None else bicarbonate_meq_l
        potassium_meq_l = _r.get("potassium_meq_l") if _r.get("potassium_meq_l") is not None else potassium_meq_l
    indications = _evaluate_aeiou(
        ph, bicarbonate_meq_l, potassium_meq_l, refractory_hyperkalemia,
        dialyzable_toxin, volume_overload_refractory, uremic_encephalopathy,
        uremic_pericarditis, uremic_bleeding,
    )

    indicated = bool(indications)
    if indicated:
        urgency: Literal["non_urgent", "urgent_within_24h", "emergent_immediately"]
        if any(i.startswith("A:") or i.startswith("E:") for i in indications):
            urgency = "emergent_immediately"
        elif any(i.startswith("U:") for i in indications):
            urgency = "emergent_immediately"
        elif any(i.startswith("I:") for i in indications):
            urgency = "emergent_immediately"
        else:
            urgency = "urgent_within_24h"
    elif aki_stage == "stage_3":
        # STARRT-AKI: stage 3 alone does not require emergent dialysis.
        urgency = "non_urgent"
    else:
        urgency = "non_urgent"

    modality: Literal["intermittent_hd", "crrt", "peritoneal_dialysis", "n_a"] | None
    if not indicated and aki_stage != "stage_3":
        modality = None
    elif hemodynamically_unstable:
        modality = "crrt"
    else:
        modality = "intermittent_hd"

    abstain = False
    abstain_reason: str | None = None
    if not indicated and aki_stage == "stage_3":
        abstain = True
        abstain_reason = (
            "Stage 3 AKI without an emergent (AEIOU) indication: per STARRT-AKI "
            "(NEJM 2020), accelerated initiation does NOT improve 90-day "
            "mortality vs standard strategy. Defer to nephrology."
        )

    rationale = _build_rationale(aki_stage, indications, indicated, urgency,
                                   modality, hemodynamically_unstable)

    return DialysisInitiationReport(
        patient_id=patient_id,
        aeiou_indications_met=indications,
        dialysis_indicated=indicated,
        urgency=urgency,
        modality_suggested=modality,
        rationale=rationale,
        abstain_recommended=abstain,
        abstain_reason=abstain_reason,
    )


def _build_rationale(stage: str, indications: list[str], indicated: bool,
                       urgency: str, modality: Any, unstable: bool) -> str:
    parts = [f"AKI stage: {stage}.",
             f"AEIOU indications met ({len(indications)}): "
             + ("; ".join(indications) if indications else "none")]
    parts.append(f"Dialysis indicated: {indicated}; urgency: {urgency}.")
    if modality:
        parts.append(f"Modality: {modality}"
                      + (" (hemodynamic instability -> CRRT)" if unstable else ""))
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_dialysis_initiation_decision)
