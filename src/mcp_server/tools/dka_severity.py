"""healthcare.compute_dka_severity -- ADA classification + initial protocol.

ADA 2009 / ISPAD 2018 DKA criteria:
  Glucose >250 mg/dL + ketones present + acidosis (pH <7.30 or HCO₃ <18)

Severity:
  mild      pH 7.25-7.30   HCO₃ 15-18    alert
  moderate  pH 7.00-7.24   HCO₃ 10-15    alert / drowsy
  severe    pH <7.00       HCO₃ <10      stupor / coma  -> ICU

Initial protocol (adult, simplified):
  Fluid: 0.9% NaCl 1 L bolus, then 250-500 mL/h titrated by hemodynamics.
    Switch to 0.45% NaCl when corrected serum Na in normal range.
    Add D5 to fluids when glucose < 200 mg/dL.
  Insulin: 0.1 U/kg IV bolus then 0.1 U/kg/h infusion (or 0.14 U/kg/h
    without bolus). DO NOT start insulin if K+ < 3.3 -- replace K+ first.
  Potassium: K+ < 3.3 -> replace BEFORE insulin. K+ 3.3-5.3 -> add 20-30 mEq/L
    to fluids. K+ > 5.3 -> no replacement; monitor.
  Bicarbonate: only if pH < 6.9 (controversial); otherwise NOT recommended.
"""

from __future__ import annotations

from typing import Literal

from shared.schemas import DKASeverityReport


# ─────────────────────────────────────────────────────────────────────
# Severity classification
# ─────────────────────────────────────────────────────────────────────

def _classify_severity(ph: float, bicarb: float, mental_status: str
                         ) -> str:
    if ph >= 7.30 and bicarb >= 18:
        return "not_dka"
    if ph < 7.00 or bicarb < 10 or mental_status in ("stupor", "stupor_coma", "coma"):
        return "severe"
    if ph < 7.25 or bicarb < 15:
        return "moderate"
    return "mild"


def _icu_indication(severity: str, glucose: float, mental_status: str,
                      anion_gap: float | None) -> bool:
    if severity == "severe":
        return True
    if mental_status in ("stupor_coma", "coma", "stupor"):
        return True
    if glucose > 600:
        return True
    if anion_gap is not None and anion_gap > 30:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_dka_severity(
    ph: float,
    bicarbonate_meq_l: float,
    glucose_mg_dl: float,
    ketones_present: bool,
    mental_status: str | None = None,
    anion_gap: float | None = None,
    potassium_meq_l: float | None = None,
    weight_kg: float = 70.0,
    patient_id: str | None = None,
) -> DKASeverityReport:
    """Classify DKA severity + emit initial fluid / insulin / K+ / bicarbonate plan.

    Args:
        ph / bicarbonate / glucose / ketones_present: ADA criteria.
        mental_status: alert / drowsy / stupor / stupor_coma / coma.
        anion_gap: optional; if >30 -> ICU bias.
        potassium_meq_l: drives the K+ replacement decision (insulin held
            if K+ <3.3).
        weight_kg: for insulin dose (0.1 U/kg/h).

    Returns:
        DKASeverityReport with severity, ICU recommendation, and a textual
        protocol for fluid / insulin / K+ / bicarbonate.
    """
    ms_norm: Literal["alert", "drowsy", "stupor_coma"]
    abstain_reasons: list[str] = []
    if mental_status is None or not mental_status.strip():
        # A missing mental status cannot be silently mapped to "alert":
        # an obtunded patient at the same pH/HCO3 has a different
        # severity assignment and ICU disposition.
        ms_norm = "alert"  # placeholder for the schema
        abstain_reasons.append(
            "missing_mental_status: severity classification and ICU "
            "indication depend on the alert/drowsy/stupor_coma axis. "
            "A missing value cannot be assumed 'alert'."
        )
    else:
        ms = mental_status.strip().lower().replace(" ", "_")
        if "stupor" in ms or "coma" in ms:
            ms_norm = "stupor_coma"
        elif "drowsy" in ms:
            ms_norm = "drowsy"
        else:
            ms_norm = "alert"

    severity = _classify_severity(ph, bicarbonate_meq_l, ms_norm)

    if not ketones_present and severity != "not_dka":
        abstain_reasons.append(
            "Acidosis present without ketones -- consider non-DKA acidosis "
            "(lactic acidosis, sepsis, ingestion); confirm beta-hydroxybutyrate."
        )

    icu = _icu_indication(severity, glucose_mg_dl, ms_norm, anion_gap)

    # Insulin decision is gated on K+ ≥ 3.3
    insulin_protocol: str
    k_replace_before_insulin = False
    if potassium_meq_l is not None and potassium_meq_l < 3.3:
        k_replace_before_insulin = True
        insulin_protocol = (
            "HOLD insulin until K+ ≥ 3.3 (otherwise risk of life-threatening "
            "hypokalemia). Replace KCl 20-40 mEq/h IV; recheck K+ in 1h."
        )
    else:
        insulin_units_per_hr = round(0.1 * weight_kg, 1)
        insulin_protocol = (
            f"Insulin regular IV: 0.1 U/kg bolus (~{round(0.1 * weight_kg, 1)} U), "
            f"then continuous infusion 0.1 U/kg/h (~{insulin_units_per_hr} U/h). "
            f"Glucose checks q1h. Add D5 to maintenance when glucose < 200 mg/dL."
        )

    fluid_protocol = (
        f"0.9% NaCl 1 L IV bolus over 30-60 min, then 250-500 mL/h. "
        f"Reassess corrected Na: switch to 0.45% NaCl if hypernatremia or "
        f"corrected Na ≥ 135. Add D5W when glucose < 200 mg/dL."
    )

    bicarbonate_indicated = ph < 6.9
    rationale = _build_rationale(
        ph, bicarbonate_meq_l, glucose_mg_dl, ketones_present, severity,
        icu, ms_norm, k_replace_before_insulin, bicarbonate_indicated,
    )

    return DKASeverityReport(
        patient_id=patient_id,
        ph=ph,
        bicarbonate_meq_l=bicarbonate_meq_l,
        glucose_mg_dl=glucose_mg_dl,
        anion_gap=anion_gap,
        ketones_present=ketones_present,
        mental_status=ms_norm,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        icu_admission_indicated=icu,
        fluid_protocol=fluid_protocol,
        insulin_protocol=insulin_protocol,
        potassium_replacement_at_initiation=k_replace_before_insulin,
        bicarbonate_indicated=bicarbonate_indicated,
        rationale=rationale,
        abstain_recommended=bool(abstain_reasons),
        abstain_reason="; ".join(abstain_reasons) if abstain_reasons else None,
    )


def _build_rationale(ph: float, bicarb: float, gluc: float, ketones: bool,
                       sev: str, icu: bool, ms: str, k_first: bool,
                       bicarb_indic: bool) -> str:
    parts = [
        f"pH {ph}, HCO₃ {bicarb} mEq/L, glucose {gluc} mg/dL, "
        f"ketones={ketones}, mental_status={ms}.",
        f"Severity: {sev}; ICU admission indicated = {icu}.",
    ]
    if k_first:
        parts.append("K+ < 3.3: HOLD insulin until KCl repletion has K+ ≥ 3.3.")
    if bicarb_indic:
        parts.append("Bicarbonate (pH <6.9): consider 100 mEq NaHCO3 IV (controversial).")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_dka_severity)
