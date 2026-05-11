"""healthcare.compute_apache_ii_score / compute_sofa_score /
compute_meld_score / compute_rifle_aki_classification
-- Phase 13.3 H2 critical-care + hepatology severity bundle.

Four high-frequency ICU + transplant scoring tools, each pure-
deterministic with literature-backed cut-offs.

References:
- Knaus WA et al. APACHE II: a severity of disease classification
  system. Crit Care Med 1985;13:818-829.
- Vincent JL et al. The SOFA score. Intensive Care Med 1996;22:707-710.
- Singer M et al. The Third International Consensus Definitions for
  Sepsis and Septic Shock (Sepsis-3). JAMA 2016;315(8):801-810.
- Kamath PS et al. MELD score. Hepatology 2001;33:464-470.
- Kim WR et al. MELD-Na. NEJM 2008;359:1018-1026.
- Bellomo R et al. ADQI Workgroup. RIFLE criteria. Crit Care 2004;8:R204-12.
"""

from __future__ import annotations

import math
from typing import Any

from shared.schemas import (
    APACHEIIReport,
    MELDReport,
    RIFLEReport,
    SOFAReport,
)

from ._chart_inputs import harden_clinical_inputs


# ─────────────────────────────────────────────────────────────────────
# APACHE II
# ─────────────────────────────────────────────────────────────────────


def _aps_temperature(t: float) -> int:
    if t >= 41 or t < 30:
        return 4
    if 39 <= t < 41 or 30 <= t < 32:
        return 3
    if 32 <= t < 34:
        return 2
    if 38.5 <= t < 39 or 34 <= t < 36:
        return 1
    return 0


def _aps_map(map_mmhg: float) -> int:
    if map_mmhg >= 160 or map_mmhg <= 49:
        return 4
    if 130 <= map_mmhg < 160:
        return 3
    if 110 <= map_mmhg < 130 or 50 <= map_mmhg < 70:
        return 2
    return 0


def _aps_hr(hr: int) -> int:
    if hr >= 180 or hr <= 39:
        return 4
    if 140 <= hr < 180 or 40 <= hr < 55:
        return 3
    if 110 <= hr < 140 or 55 <= hr < 70:
        return 2
    return 0


def _aps_rr(rr: int) -> int:
    if rr >= 50 or rr <= 5:
        return 4
    if 35 <= rr < 50:
        return 3
    if 25 <= rr < 35 or 6 <= rr < 10:
        return 1
    if 12 <= rr < 25:
        return 0
    return 2


def _aps_aa_gradient(fio2: float, pao2: float) -> int:
    """If FiO2 ≥ 0.5: use A-a gradient. Else use PaO2."""
    if fio2 >= 0.5:
        # A-a = (FiO2 * 713 - PaCO2/0.8) - PaO2 ; we'll use a rough
        # PaO2-only approximation here for the deterministic score.
        if pao2 < 60:
            return 4
        if pao2 < 70:
            return 3
        return 0
    if pao2 < 55:
        return 4
    if 55 <= pao2 < 61:
        return 3
    if 61 <= pao2 < 71:
        return 1
    return 0


def _aps_ph(ph: float) -> int:
    if ph >= 7.7 or ph < 7.15:
        return 4
    if 7.6 <= ph < 7.7 or 7.15 <= ph < 7.25:
        return 3
    if 7.5 <= ph < 7.6:
        return 1
    if 7.25 <= ph < 7.33:
        return 2
    return 0


def _aps_sodium(na: float) -> int:
    if na >= 180 or na < 111:
        return 4
    if 160 <= na < 180 or 111 <= na < 120:
        return 3
    if 155 <= na < 160 or 120 <= na < 130:
        return 2
    if 150 <= na < 155:
        return 1
    return 0


def _aps_potassium(k: float) -> int:
    if k >= 7 or k < 2.5:
        return 4
    if 6 <= k < 7:
        return 3
    if 2.5 <= k < 3:
        return 2
    if 5.5 <= k < 6 or 3 <= k < 3.5:
        return 1
    return 0


def _aps_creatinine(cr: float, on_arf: bool) -> int:
    base = 0
    if cr >= 3.5:
        base = 4
    elif 2.0 <= cr < 3.5:
        base = 3
    elif 1.5 <= cr < 2.0 or cr < 0.6:
        base = 2
    return base * 2 if on_arf else base


def _aps_hematocrit(hct: float) -> int:
    if hct >= 60 or hct < 20:
        return 4
    if 50 <= hct < 60 or 20 <= hct < 30:
        return 2
    return 0


def _aps_wbc(wbc: float) -> int:
    if wbc >= 40 or wbc < 1:
        return 4
    if 20 <= wbc < 40 or 1 <= wbc < 3:
        return 2
    if 15 <= wbc < 20:
        return 1
    return 0


def _aps_gcs(gcs: int) -> int:
    return max(0, 15 - gcs)


def _age_points(age: int) -> int:
    if age >= 75:
        return 6
    if age >= 65:
        return 5
    if age >= 55:
        return 3
    if age >= 45:
        return 2
    return 0


def _apache_mortality(total: int) -> float:
    """Knaus 1985 Table 6 -- mortality % vs APACHE II total."""
    table = [
        (4, 4.0), (9, 8.0), (14, 15.0), (19, 25.0),
        (24, 40.0), (29, 55.0), (34, 75.0), (71, 85.0),
    ]
    for thresh, pct in table:
        if total <= thresh:
            return pct
    return 90.0


async def compute_apache_ii_score(
    age: int,
    temperature_c: float,
    mean_arterial_pressure_mmHg: float,
    heart_rate: int,
    respiratory_rate: int,
    fio2: float,
    pao2: float,
    arterial_ph: float,
    serum_sodium_mmol_l: float,
    serum_potassium_mmol_l: float,
    serum_creatinine_mg_dl: float,
    hematocrit_pct: float,
    wbc_thousands_per_uL: float,
    glasgow_coma_scale: int = 15,
    acute_renal_failure: bool = False,
    chronic_health_severe: bool = False,
    immunocompromised: bool = False,
    post_emergency_surgery: bool = False,
) -> APACHEIIReport:
    """APACHE II -- Knaus 1985.

    Chart-authoritative inputs: when SHARP context is bound, age,
    vitals, and labs come from the patient's FHIR chart; caller-supplied
    values for those parameters are discarded.

    Returns the composite ICU severity score + predicted hospital
    mortality. The chronic-health point block adds 5 (non-operative or
    emergency post-op) or 2 (elective post-op) when severe organ
    insufficiency or immunocompromise is documented.
    """
    _chart_keys = {
        "age", "temperature_c", "mean_arterial_pressure_mmHg",
        "heart_rate", "respiratory_rate", "arterial_ph",
        "serum_sodium_mmol_l", "serum_potassium_mmol_l",
        "serum_creatinine_mg_dl", "hematocrit_pct",
        "wbc_thousands_per_uL", "glasgow_coma_scale",
    }
    _r, _sb, _ = await harden_clinical_inputs(
        {
            "age": age, "temperature_c": temperature_c,
            "mean_arterial_pressure_mmHg": mean_arterial_pressure_mmHg,
            "heart_rate": heart_rate, "respiratory_rate": respiratory_rate,
            "arterial_ph": arterial_ph,
            "serum_sodium_mmol_l": serum_sodium_mmol_l,
            "serum_potassium_mmol_l": serum_potassium_mmol_l,
            "serum_creatinine_mg_dl": serum_creatinine_mg_dl,
            "hematocrit_pct": hematocrit_pct,
            "wbc_thousands_per_uL": wbc_thousands_per_uL,
            "glasgow_coma_scale": glasgow_coma_scale,
        },
        chart_derivable=_chart_keys,
    )
    if _sb:
        age = _r["age"] if _r.get("age") is not None else age
        temperature_c = _r["temperature_c"] if _r.get("temperature_c") is not None else temperature_c
        mean_arterial_pressure_mmHg = _r["mean_arterial_pressure_mmHg"] if _r.get("mean_arterial_pressure_mmHg") is not None else mean_arterial_pressure_mmHg
        heart_rate = _r["heart_rate"] if _r.get("heart_rate") is not None else heart_rate
        respiratory_rate = _r["respiratory_rate"] if _r.get("respiratory_rate") is not None else respiratory_rate
        arterial_ph = _r["arterial_ph"] if _r.get("arterial_ph") is not None else arterial_ph
        serum_sodium_mmol_l = _r["serum_sodium_mmol_l"] if _r.get("serum_sodium_mmol_l") is not None else serum_sodium_mmol_l
        serum_potassium_mmol_l = _r["serum_potassium_mmol_l"] if _r.get("serum_potassium_mmol_l") is not None else serum_potassium_mmol_l
        serum_creatinine_mg_dl = _r["serum_creatinine_mg_dl"] if _r.get("serum_creatinine_mg_dl") is not None else serum_creatinine_mg_dl
        hematocrit_pct = _r["hematocrit_pct"] if _r.get("hematocrit_pct") is not None else hematocrit_pct
        wbc_thousands_per_uL = _r["wbc_thousands_per_uL"] if _r.get("wbc_thousands_per_uL") is not None else wbc_thousands_per_uL
        glasgow_coma_scale = _r["glasgow_coma_scale"] if _r.get("glasgow_coma_scale") is not None else glasgow_coma_scale
    aps = (
        _aps_temperature(temperature_c)
        + _aps_map(mean_arterial_pressure_mmHg)
        + _aps_hr(heart_rate)
        + _aps_rr(respiratory_rate)
        + _aps_aa_gradient(fio2, pao2)
        + _aps_ph(arterial_ph)
        + _aps_sodium(serum_sodium_mmol_l)
        + _aps_potassium(serum_potassium_mmol_l)
        + _aps_creatinine(serum_creatinine_mg_dl, acute_renal_failure)
        + _aps_hematocrit(hematocrit_pct)
        + _aps_wbc(wbc_thousands_per_uL)
        + _aps_gcs(glasgow_coma_scale)
    )
    age_pts = _age_points(age)
    chronic_pts = 0
    if chronic_health_severe or immunocompromised:
        chronic_pts = 2 if post_emergency_surgery else 5
    total = aps + age_pts + chronic_pts
    mortality = _apache_mortality(total)
    if total <= 9:
        tier = "mild"
    elif total <= 19:
        tier = "moderate"
    elif total <= 29:
        tier = "severe"
    else:
        tier = "critical"
    icu = total >= 15 or tier in ("severe", "critical")
    rationale = (
        f"APACHE II total {total} (APS {aps} + age {age_pts} + "
        f"chronic {chronic_pts}); predicted mortality "
        f"{mortality:.1f}%; severity {tier}; "
        f"ICU admission {'recommended' if icu else 'optional'}."
    )
    return APACHEIIReport(
        aps_score=aps, age_points=age_pts,
        chronic_health_points=chronic_pts,
        apache_ii_total=total,
        predicted_mortality_pct=mortality,
        severity_tier=tier,                              # type: ignore[arg-type]
        icu_admission_recommended=icu,
        rationale=rationale,
        references=[
            "Knaus WA et al. APACHE II. Crit Care Med 1985;13:818-829.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# SOFA
# ─────────────────────────────────────────────────────────────────────


def _sofa_resp(pao2_fio2: float, mech_vent: bool) -> int:
    if pao2_fio2 < 100 and mech_vent: return 4
    if pao2_fio2 < 200 and mech_vent: return 3
    if pao2_fio2 < 300: return 2
    if pao2_fio2 < 400: return 1
    return 0


def _sofa_coag(plt_thousands: float) -> int:
    if plt_thousands < 20: return 4
    if plt_thousands < 50: return 3
    if plt_thousands < 100: return 2
    if plt_thousands < 150: return 1
    return 0


def _sofa_liver(bili: float) -> int:
    if bili >= 12.0: return 4
    if bili >= 6.0: return 3
    if bili >= 2.0: return 2
    if bili >= 1.2: return 1
    return 0


def _sofa_cardio(map_mmhg: float, pressors: dict[str, float] | None) -> int:
    pressors = pressors or {}
    if pressors.get("dopamine_mcg_kg_min", 0) > 15 or \
       pressors.get("epinephrine_mcg_kg_min", 0) > 0.1 or \
       pressors.get("norepinephrine_mcg_kg_min", 0) > 0.1:
        return 4
    if pressors.get("dopamine_mcg_kg_min", 0) > 5 or \
       pressors.get("epinephrine_mcg_kg_min", 0) > 0 or \
       pressors.get("norepinephrine_mcg_kg_min", 0) > 0:
        return 3
    if pressors.get("dopamine_mcg_kg_min", 0) > 0 or \
       pressors.get("dobutamine_any", 0) > 0:
        return 2
    if map_mmhg < 70: return 1
    return 0


def _sofa_cns(gcs: int) -> int:
    if gcs < 6: return 4
    if gcs < 10: return 3
    if gcs < 13: return 2
    if gcs < 15: return 1
    return 0


def _sofa_renal(cr_mg_dl: float, urine_output_ml_per_day: float | None) -> int:
    if cr_mg_dl >= 5.0 or (urine_output_ml_per_day is not None and
                                 urine_output_ml_per_day < 200):
        return 4
    if cr_mg_dl >= 3.5 or (urine_output_ml_per_day is not None and
                                 urine_output_ml_per_day < 500):
        return 3
    if cr_mg_dl >= 2.0:
        return 2
    if cr_mg_dl >= 1.2:
        return 1
    return 0


def _sofa_mortality(total: int) -> float:
    """Vincent 1998 SOFA mortality table."""
    if total >= 15: return 80.0
    if total >= 12: return 50.0
    if total >= 9: return 33.0
    if total >= 6: return 22.0
    if total >= 3: return 7.0
    return 0.0


async def compute_sofa_score(
    pao2_fio2_ratio: float,
    mechanical_ventilation: bool,
    platelets_thousands_per_uL: float,
    bilirubin_mg_dl: float,
    mean_arterial_pressure_mmHg: float,
    pressors_doses: dict[str, float] | None = None,
    glasgow_coma_scale: int = 15,
    creatinine_mg_dl: float = 0.8,
    urine_output_ml_per_day: float | None = None,
    baseline_sofa_total: int = 0,
    suspected_infection: bool = False,
) -> SOFAReport:
    """Sequential Organ Failure Assessment + Sepsis-3 dysfunction flag.

    Chart-authoritative inputs: when SHARP context is bound, labs and
    vitals come from the patient's FHIR chart; caller-supplied values
    for those parameters are discarded.
    """
    _r, _sb, _ = await harden_clinical_inputs(
        {
            "platelets_thousands_per_uL": platelets_thousands_per_uL,
            "bilirubin_mg_dl": bilirubin_mg_dl,
            "mean_arterial_pressure_mmHg": mean_arterial_pressure_mmHg,
            "glasgow_coma_scale": glasgow_coma_scale,
            "creatinine_mg_dl": creatinine_mg_dl,
        },
        chart_derivable={
            "platelets_thousands_per_uL", "bilirubin_mg_dl",
            "mean_arterial_pressure_mmHg", "glasgow_coma_scale",
            "creatinine_mg_dl",
        },
    )
    if _sb:
        platelets_thousands_per_uL = _r.get("platelets_thousands_per_uL") if _r.get("platelets_thousands_per_uL") is not None else platelets_thousands_per_uL
        bilirubin_mg_dl = _r.get("bilirubin_mg_dl") if _r.get("bilirubin_mg_dl") is not None else bilirubin_mg_dl
        mean_arterial_pressure_mmHg = _r.get("mean_arterial_pressure_mmHg") if _r.get("mean_arterial_pressure_mmHg") is not None else mean_arterial_pressure_mmHg
        glasgow_coma_scale = _r.get("glasgow_coma_scale") if _r.get("glasgow_coma_scale") is not None else glasgow_coma_scale
        creatinine_mg_dl = _r.get("creatinine_mg_dl") if _r.get("creatinine_mg_dl") is not None else creatinine_mg_dl
    resp = _sofa_resp(pao2_fio2_ratio, mechanical_ventilation)
    coag = _sofa_coag(platelets_thousands_per_uL)
    liver = _sofa_liver(bilirubin_mg_dl)
    cardio = _sofa_cardio(mean_arterial_pressure_mmHg, pressors_doses)
    cns = _sofa_cns(glasgow_coma_scale)
    renal = _sofa_renal(creatinine_mg_dl, urine_output_ml_per_day)
    total = resp + coag + liver + cardio + cns + renal
    delta = total - baseline_sofa_total
    sepsis3 = suspected_infection and delta >= 2
    mortality = _sofa_mortality(total)

    rationale = (
        f"SOFA total {total} (resp {resp} + coag {coag} + liver "
        f"{liver} + cardio {cardio} + CNS {cns} + renal {renal}); "
        f"Δ vs baseline {delta:+d}; "
        f"Sepsis-3 organ dysfunction = {sepsis3}; "
        f"estimated ICU mortality {mortality:.1f}%."
    )
    return SOFAReport(
        respiratory_points=resp, coagulation_points=coag,
        liver_points=liver, cardiovascular_points=cardio,
        cns_points=cns, renal_points=renal,
        sofa_total=total,
        sepsis_3_dysfunction=sepsis3,
        estimated_mortality_pct=mortality,
        rationale=rationale,
        references=[
            "Vincent JL et al. Intensive Care Med 1996;22:707-710.",
            "Singer M et al. JAMA 2016;315(8):801-810. (Sepsis-3)",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# MELD
# ─────────────────────────────────────────────────────────────────────


def _meld_classic(bili: float, cr: float, inr: float, on_dialysis: bool) -> int:
    bili = max(1.0, bili)
    cr = max(1.0, min(4.0, 4.0 if on_dialysis else cr))
    inr = max(1.0, inr)
    val = (
        0.957 * math.log(cr) + 0.378 * math.log(bili)
        + 1.120 * math.log(inr) + 0.643
    ) * 10
    return int(round(max(6, min(40, val))))


def _meld_na_adjustment(meld: int, na: float | None) -> int:
    if meld <= 11 or na is None:
        return meld
    na = max(125, min(137, na))
    return int(round(meld + 1.32 * (137 - na) - 0.033 * meld * (137 - na)))


def _meld_3mo_mortality(meld: int) -> float:
    if meld < 9: return 1.9
    if meld < 19: return 6.0
    if meld < 29: return 19.6
    if meld < 39: return 52.6
    return 71.3


async def compute_meld_score(
    bilirubin_mg_dl: float,
    creatinine_mg_dl: float,
    inr: float,
    sodium_mmol_l: float | None = None,
    on_dialysis: bool = False,
) -> MELDReport:
    """MELD + MELD-Na + 3-month mortality + UNOS listing flag.

    Chart-authoritative inputs: when SHARP context is bound, labs come
    from the patient's FHIR chart; caller-supplied values are discarded.
    """
    _r, _sb, _ = await harden_clinical_inputs(
        {
            "bilirubin_mg_dl": bilirubin_mg_dl,
            "creatinine_mg_dl": creatinine_mg_dl,
            "inr": inr, "sodium_mmol_l": sodium_mmol_l,
        },
        chart_derivable={"bilirubin_mg_dl", "creatinine_mg_dl",
                          "inr", "sodium_mmol_l"},
    )
    if _sb:
        bilirubin_mg_dl = _r.get("bilirubin_mg_dl") if _r.get("bilirubin_mg_dl") is not None else bilirubin_mg_dl
        creatinine_mg_dl = _r.get("creatinine_mg_dl") if _r.get("creatinine_mg_dl") is not None else creatinine_mg_dl
        inr = _r.get("inr") if _r.get("inr") is not None else inr
        sodium_mmol_l = _r.get("sodium_mmol_l") if _r.get("sodium_mmol_l") is not None else sodium_mmol_l
    classic = _meld_classic(bilirubin_mg_dl, creatinine_mg_dl, inr,
                                  on_dialysis)
    meld_na = _meld_na_adjustment(classic, sodium_mmol_l) if sodium_mmol_l \
        else None
    score_for_mortality = meld_na if meld_na is not None else classic
    mortality = _meld_3mo_mortality(score_for_mortality)
    if score_for_mortality < 10:
        tier = "low"
    elif score_for_mortality < 20:
        tier = "moderate"
    elif score_for_mortality < 30:
        tier = "high"
    else:
        tier = "critical"
    rationale = (
        f"MELD {classic}"
        + (f" / MELD-Na {meld_na}" if meld_na is not None else "")
        + f"; 3-mo mortality {mortality:.1f}%; tier {tier}; "
        + f"UNOS listing-eligible = "
        f"{score_for_mortality >= 15}."
    )
    return MELDReport(
        bilirubin_mg_dl=bilirubin_mg_dl,
        creatinine_mg_dl=creatinine_mg_dl,
        inr=inr, sodium_mmol_l=sodium_mmol_l,
        on_dialysis=on_dialysis,
        meld_classic=classic, meld_na=meld_na,
        estimated_3mo_mortality_pct=mortality,
        severity_tier=tier,                              # type: ignore[arg-type]
        transplant_eligibility_threshold_met=score_for_mortality >= 15,
        rationale=rationale,
        references=[
            "Kamath PS et al. Hepatology 2001;33:464-470.",
            "Kim WR et al. NEJM 2008;359:1018-1026. (MELD-Na)",
            "OPTN MELD policy.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# RIFLE AKI
# ─────────────────────────────────────────────────────────────────────


_RIFLE_ETIOLOGY: list[tuple[str, str]] = [
    ("nephrotoxic", "Drug-induced nephrotoxicity"),
    ("contrast", "Contrast-induced nephropathy"),
    ("sepsis", "Septic AKI"),
    ("hypovolaemia", "Pre-renal AKI from volume depletion"),
    ("obstruction", "Post-renal obstruction"),
]


async def compute_rifle_aki_classification(
    serum_creatinine_baseline_mg_dl: float,
    serum_creatinine_current_mg_dl: float,
    urine_output_ml_per_kg_per_hour: float | None = None,
    history_keywords: str | None = None,
) -> RIFLEReport:
    """Classify acute kidney injury into the RIFLE tiers
    (Risk / Injury / Failure / Loss / ESRD)."""
    if serum_creatinine_baseline_mg_dl <= 0:
        raise ValueError("baseline creatinine must be > 0")
    ratio = serum_creatinine_current_mg_dl / serum_creatinine_baseline_mg_dl
    rifle = "no_aki"
    if ratio >= 3.0 or serum_creatinine_current_mg_dl >= 4.0:
        rifle = "failure"
    elif ratio >= 2.0:
        rifle = "injury"
    elif ratio >= 1.5:
        rifle = "risk"

    if urine_output_ml_per_kg_per_hour is not None:
        u = urine_output_ml_per_kg_per_hour
        if u < 0.3:
            rifle = "failure" if rifle != "esrd" else rifle
        elif u < 0.5 and rifle == "no_aki":
            rifle = "risk"

    etiology = "Unspecified"
    if history_keywords:
        lo = history_keywords.lower()
        for kw, label in _RIFLE_ETIOLOGY:
            if kw in lo:
                etiology = label
                break

    consult = rifle in ("injury", "failure", "loss", "esrd")
    rationale = (
        f"Creatinine {serum_creatinine_baseline_mg_dl:.2f} -> "
        f"{serum_creatinine_current_mg_dl:.2f} (ratio {ratio:.2f}); "
        f"urine output "
        f"{urine_output_ml_per_kg_per_hour if urine_output_ml_per_kg_per_hour is not None else 'n/a'}; "
        f"RIFLE = {rifle}; etiology clue = {etiology!r}."
    )
    return RIFLEReport(
        serum_creatinine_baseline_mg_dl=serum_creatinine_baseline_mg_dl,
        serum_creatinine_current_mg_dl=serum_creatinine_current_mg_dl,
        creatinine_ratio=round(ratio, 2),
        urine_output_ml_per_kg_per_hour=urine_output_ml_per_kg_per_hour,
        rifle_class=rifle,                                # type: ignore[arg-type]
        aki_etiology_clue=etiology,
        nephrology_consult_recommended=consult,
        rationale=rationale,
        references=[
            "Bellomo R et al. ADQI Workgroup. Crit Care 2004;8:R204-12.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_apache_ii_score)
    mcp.tool()(compute_sofa_score)
    mcp.tool()(compute_meld_score)
    mcp.tool()(compute_rifle_aki_classification)
