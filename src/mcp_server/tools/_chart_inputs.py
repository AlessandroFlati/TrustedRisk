"""Shared chart-authoritative resolver for clinical-input parameters.

When SHARP-on-MCP context is bound, clinical inputs (demographics,
vital signs, labs, medications, gestational age) come from the
patient's FHIR chart, NOT from caller-supplied values. This prevents
the chat-side LLM from hallucinating clinical data to satisfy a tool
schema (the PGx bug class).

Each Tier-1 tool wraps its parameter list:

    from ._chart_inputs import harden_clinical_inputs

    resolved, sharp_bound, missing = await harden_clinical_inputs(
        caller={'age': age, 'creatinine_mg_dl': creatinine_mg_dl, ...},
        chart_derivable={'age', 'creatinine_mg_dl', ...},
    )
    if sharp_bound and missing:
        return ToolReport(
            abstain_recommended=True,
            abstain_reason=(
                f"unverified_in_chart: chart-derivable inputs not present "
                f"in the SHARP-bound patient's FHIR bundle: {sorted(missing)}. "
                f"Caller-supplied values are discarded for these parameters."
            ),
            ...
        )
    age = resolved.get('age')
    creatinine_mg_dl = resolved.get('creatinine_mg_dl')

Behaviour:
  - SHARP bound + chart has the key: chart value REPLACES caller's
  - SHARP bound + chart missing the key: caller's value DISCARDED;
    key included in `missing`; tool short-circuits with abstain
  - SHARP not bound (offline / tests): caller's value passes through
    unchanged (legacy behavior)

The chart-derivable set must list ONLY parameters whose authoritative
source is a FHIR resource. Clinical-judgment booleans (Killip class,
suspected_infection, mental_status, ECG descriptors) are NOT chart-
derivable and remain caller-supplied even under SHARP.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any

from ..fhir.client import fetch_patient_bundle, resolve_patient_id


# ─────────────────────────────────────────────────────────────────────
# LOINC code -> tool parameter name(s)
# ─────────────────────────────────────────────────────────────────────

# Lab map: each LOINC -> list of (param_name, role).
# Role: 'current' = most recent reading, 'baseline' = oldest,
# 'current_thousands' = per-uL count divided by 1000 if > 1000,
# 'current_bun_mmol' = BUN mg/dL converted to urea mmol/L.
_LOINC_LAB_MAP: dict[str, list[tuple[str, str]]] = {
    "2160-0": [("creatinine_current_mg_dl", "current"),
                ("creatinine_baseline_mg_dl", "baseline"),
                ("creatinine_mg_dl", "current"),
                ("serum_creatinine_mg_dl", "current"),
                ("preop_creatinine_mg_dl", "current")],
    "62238-1": [("egfr_ml_min", "current")],
    "11558-4": [("ph", "current"), ("arterial_ph", "current")],
    "2744-1":  [("ph", "current"), ("arterial_ph", "current")],
    "1963-8":  [("bicarbonate_meq_l", "current")],
    "1959-6":  [("bicarbonate_meq_l", "current")],
    "1863-0":  [("anion_gap", "current")],
    "2345-7":  [("glucose_mg_dl", "current")],
    "2339-0":  [("glucose_mg_dl", "current")],
    "2823-3":  [("potassium_meq_l", "current"),
                 ("serum_potassium_mmol_l", "current")],
    "2951-2":  [("serum_sodium_mmol_l", "current"),
                 ("sodium_mmol_l", "current")],
    "777-3":   [("platelets_per_ul", "current"),
                 ("platelets_thousands_per_uL", "current_thousands"),
                 ("platelet_count_per_ul", "current")],
    "1975-2":  [("bilirubin_mg_dl", "current"),
                 ("total_bilirubin_mg_dl", "current"),
                 ("serum_bilirubin_mg_dl", "current")],
    "1920-8":  [("ast_u_l", "current"), ("ast_ul", "current"),
                 ("ast_iu_l", "current")],
    "1742-6":  [("alt_u_l", "current"), ("alt_ul", "current"),
                 ("alt_iu_l", "current")],
    "718-7":   [("hemoglobin_g_dl", "current"),
                 ("preop_hemoglobin_g_dl", "current")],
    "4544-3":  [("hematocrit_pct", "current")],
    "6690-2":  [("wbc_thousands_per_uL", "current_thousands"),
                 ("wbc_per_ul_thousand", "current_thousands")],
    "751-8":   [("anc_per_ul", "current"),
                 ("anc_thousands_per_uL", "current_thousands"),
                 ("absolute_neutrophil_count_per_ul", "current"),
                 ("neutrophil_count_per_ul", "current")],
    "3094-0":  [("bun_mg_dl", "current"),
                 ("blood_urea_mmol_l", "current_bun_mmol")],
    "32693-4": [("lactate_mmol_l", "current"),
                 ("lactate_mg_dl", "current")],
    "1751-7":  [("albumin_g_dl", "current"),
                 ("serum_albumin_g_dl", "current")],
    "6301-6":  [("inr", "current")],
    "5902-2":  [("patient_pt_seconds", "current"),
                 ("prothrombin_time_sec", "current")],
}

# Vital-sign map: LOINC -> tool parameter name.
_LOINC_VITAL_MAP: dict[str, str] = {
    "8867-4": "heart_rate",
    "9279-1": "respiratory_rate",
    "59408-5": "spo2",
    "2708-6": "spo2",
    "8480-6": "systolic_bp",
    "8462-4": "diastolic_bp",
    "8310-5": "temperature_c",
    "8716-3": "temperature_c",
    "85354-9": "blood_pressure_panel",
    "9269-2": "glasgow_coma_scale",
    "9270-0": "glasgow_coma_scale",
}

# Demographic LOINCs (body weight, body height)
_LOINC_WEIGHT = "29463-7"
_LOINC_HEIGHT = "8302-2"


# ─────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────


def _entries(bundle: dict | None) -> list[dict]:
    if not isinstance(bundle, dict):
        return []
    return bundle.get("entry", []) or []


def _patient(bundle: dict | None) -> dict | None:
    for e in _entries(bundle):
        r = e.get("resource") if isinstance(e, dict) else None
        if isinstance(r, dict) and r.get("resourceType") == "Patient":
            return r
    return None


def _observation_loinc(r: dict) -> str | None:
    code = r.get("code") or {}
    for c in code.get("coding") or []:
        if isinstance(c, dict) and c.get("system") == "http://loinc.org":
            return c.get("code")
    return None


def _latest_observation_value(bundle: dict | None, loinc: str) -> float | None:
    """Most-recent valueQuantity.value of an Observation with the given LOINC."""
    best: tuple[float, str] | None = None
    for e in _entries(bundle):
        r = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "Observation":
            continue
        if _observation_loinc(r) != loinc:
            continue
        vq = r.get("valueQuantity") or {}
        val = vq.get("value")
        if val is None:
            continue
        try:
            value = float(val)
        except (TypeError, ValueError):
            continue
        observed = (
            r.get("effectiveDateTime") or r.get("issued") or "1970-01-01"
        )
        if best is None or observed > best[1]:
            best = (value, observed)
    return best[0] if best else None


# ─────────────────────────────────────────────────────────────────────
# Demographic extractors
# ─────────────────────────────────────────────────────────────────────


def extract_chart_demographics(bundle: dict | None) -> dict[str, Any]:
    """Pull age (years + months), sex (string + boolean), weight, height."""
    out: dict[str, Any] = {}
    pt = _patient(bundle)
    if pt:
        bd = pt.get("birthDate")
        if isinstance(bd, str):
            try:
                born = _dt.date.fromisoformat(bd[:10])
                today = _dt.date.today()
                years = today.year - born.year - (
                    (today.month, today.day) < (born.month, born.day)
                )
                months = (today.year - born.year) * 12 + (today.month - born.month)
                if today.day < born.day:
                    months -= 1
                years = max(0, years)
                months = max(0, months)
                out["age"] = years
                out["age_years"] = years
                out["age_months"] = months
                out["patient_age"] = years
            except ValueError:
                pass
        gender = pt.get("gender")
        if gender:
            out["sex"] = "female" if gender == "female" else "male"
            out["sex_female"] = (gender == "female")
    weight = _latest_observation_value(bundle, loinc=_LOINC_WEIGHT)
    if weight is not None:
        out["weight_kg"] = weight
    height = _latest_observation_value(bundle, loinc=_LOINC_HEIGHT)
    if height is not None:
        out["height_cm"] = height
    return out


# ─────────────────────────────────────────────────────────────────────
# Vital-sign extractor
# ─────────────────────────────────────────────────────────────────────


def extract_chart_vitals(bundle: dict | None) -> dict[str, Any]:
    """Return most-recent vital signs keyed by tool parameter name.

    Adds canonical synonyms: `mean_arterial_pressure` /
    `mean_arterial_pressure_mmHg` (computed from SBP+DBP), `gcs` /
    `glasgow_coma_score` (alias of glasgow_coma_scale), `preop_spo2_pct`,
    `temperature` (alias of temperature_c).
    """
    if not bundle:
        return {}
    by_kind: dict[str, tuple[float, str]] = {}
    for e in _entries(bundle):
        r = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "Observation":
            continue
        loinc = _observation_loinc(r)
        if not loinc:
            continue
        kind = _LOINC_VITAL_MAP.get(loinc)
        if not kind:
            continue
        observed = (
            r.get("effectiveDateTime") or r.get("issued") or "1970-01-01"
        )
        if kind == "blood_pressure_panel":
            for comp in r.get("component") or []:
                cc = comp.get("code") or {}
                cloinc = next(
                    (c.get("code") for c in cc.get("coding") or []
                     if isinstance(c, dict)
                     and c.get("system") == "http://loinc.org"),
                    None,
                )
                ckind = _LOINC_VITAL_MAP.get(cloinc or "")
                if not ckind or ckind == "blood_pressure_panel":
                    continue
                vq = comp.get("valueQuantity") or {}
                val = vq.get("value")
                if val is None:
                    continue
                try:
                    value = float(val)
                except (TypeError, ValueError):
                    continue
                if ckind not in by_kind or observed > by_kind[ckind][1]:
                    by_kind[ckind] = (value, observed)
            continue
        vq = r.get("valueQuantity") or {}
        val = vq.get("value")
        if val is None:
            continue
        try:
            value = float(val)
        except (TypeError, ValueError):
            continue
        if kind not in by_kind or observed > by_kind[kind][1]:
            by_kind[kind] = (value, observed)
    out: dict[str, Any] = {k: v[0] for k, v in by_kind.items()}
    if "systolic_bp" in out and "diastolic_bp" in out:
        sbp = out["systolic_bp"]
        dbp = out["diastolic_bp"]
        map_val = round(dbp + (sbp - dbp) / 3.0, 1)
        out["mean_arterial_pressure"] = map_val
        out["mean_arterial_pressure_mmHg"] = map_val
        out["map"] = map_val
    if "glasgow_coma_scale" in out:
        out["gcs"] = out["glasgow_coma_scale"]
        out["glasgow_coma_score"] = out["glasgow_coma_scale"]
    if "spo2" in out:
        out["preop_spo2_pct"] = out["spo2"]
    if "temperature_c" in out:
        out["temperature"] = out["temperature_c"]
    # Suffix aliases used by some tools (e.g. compute_glasgow_blatchford_ugib
    # uses `systolic_bp_mmHg` rather than `systolic_bp`).
    if "systolic_bp" in out:
        out["systolic_bp_mmHg"] = out["systolic_bp"]
    if "diastolic_bp" in out:
        out["diastolic_bp_mmHg"] = out["diastolic_bp"]
    return out


# ─────────────────────────────────────────────────────────────────────
# Lab extractor
# ─────────────────────────────────────────────────────────────────────


def extract_chart_labs(bundle: dict | None) -> dict[str, Any]:
    """Return labs keyed by all known tool kwarg names. Most-recent wins
    for 'current' role; oldest wins for 'baseline' role; thousands and
    BUN-mmol conversions applied per `_LOINC_LAB_MAP` role tags."""
    if not bundle:
        return {}
    by_code: dict[str, list[tuple[float, str]]] = {}
    for e in _entries(bundle):
        r = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "Observation":
            continue
        loinc = _observation_loinc(r)
        if not loinc or loinc not in _LOINC_LAB_MAP:
            continue
        vq = r.get("valueQuantity") or {}
        val = vq.get("value")
        if val is None:
            continue
        try:
            value = float(val)
        except (TypeError, ValueError):
            continue
        observed = (
            r.get("effectiveDateTime") or r.get("issued") or "1970-01-01"
        )
        by_code.setdefault(loinc, []).append((value, observed))

    out: dict[str, Any] = {}
    for loinc, observations in by_code.items():
        observations.sort(key=lambda x: x[1])
        oldest_value = observations[0][0]
        latest_value = observations[-1][0]
        for kwarg, role in _LOINC_LAB_MAP[loinc]:
            if role == "baseline":
                out[kwarg] = oldest_value
            elif role == "current":
                out[kwarg] = latest_value
            elif role == "current_thousands":
                out[kwarg] = (latest_value / 1000.0
                               if latest_value > 1000 else latest_value)
            elif role == "current_bun_mmol":
                # BUN mg/dL to urea mmol/L: divide by 2.801
                out[kwarg] = round(latest_value / 2.801, 2)
    return out


# ─────────────────────────────────────────────────────────────────────
# Medications extractor
# ─────────────────────────────────────────────────────────────────────


def extract_chart_medications(bundle: dict | None) -> list[str]:
    """Return active medications from MedicationRequest +
    MedicationAdministration as a list of distinct names.

    Falls back to parsing `# Medications` sections in clinical-note
    DocumentReferences (Synthea-style) when the bundle has no
    MedicationRequest / MedicationAdministration. Mirrors the
    `compute_resolve_active_meds` source-order contract but with a
    flatter return shape (just the names).
    """
    import base64
    if not bundle:
        return []
    names: list[str] = []
    seen: set[str] = set()

    def _add(nm: str | None) -> None:
        if not isinstance(nm, str):
            return
        nm = nm.strip()
        if not nm:
            return
        key = nm.lower()
        if key in seen:
            return
        seen.add(key)
        names.append(nm)

    for e in _entries(bundle):
        r = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(r, dict):
            continue
        rt = r.get("resourceType")
        if rt not in ("MedicationRequest", "MedicationAdministration"):
            continue
        status = (r.get("status") or "").lower()
        if status in ("completed", "stopped", "cancelled", "entered-in-error"):
            continue
        mc = r.get("medicationCodeableConcept") or {}
        nm = (mc.get("text") or "").strip()
        if not nm:
            for c in mc.get("coding") or []:
                if isinstance(c, dict):
                    nm = (c.get("display") or "").strip()
                    if nm:
                        break
        _add(nm)

    if names:
        return names

    # Fallback: parse a Synthea-style `# Medications` section from a
    # clinical-note DocumentReference body. Same heuristic as
    # active_meds_resolver.compute_resolve_active_meds.
    section_re = re.compile(
        r"^\s*#+\s*medications?\s*$", re.IGNORECASE | re.MULTILINE,
    )
    next_section_re = re.compile(r"^\s*#+\s+\S", re.MULTILINE)
    bullet_re = re.compile(r"^\s*[-*]\s+(.+?)\s*$", re.MULTILINE)
    for e in _entries(bundle):
        r = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "DocumentReference":
            continue
        for content in r.get("content") or []:
            att = content.get("attachment") or {}
            data = att.get("data")
            if not isinstance(data, str):
                continue
            try:
                body = base64.b64decode(data).decode("utf-8", "ignore")
            except Exception:
                continue
            m = section_re.search(body)
            if not m:
                continue
            tail = body[m.end():]
            nxt = next_section_re.search(tail)
            if nxt:
                tail = tail[: nxt.start()]
            for bullet in bullet_re.findall(tail):
                _add(bullet)
    return names


# ─────────────────────────────────────────────────────────────────────
# Gestational age extractor
# ─────────────────────────────────────────────────────────────────────


def extract_chart_gestational_age(bundle: dict | None) -> int | None:
    """Return gestational age in weeks from Conditions / Observations /
    Encounters that mention 'gestational age', 'weeks pregnancy',
    'gestation'. Returns None when nothing matches."""
    if not bundle:
        return None
    pattern = re.compile(r"(\d{1,2})\s*(?:weeks?|wks?|w)\b", re.IGNORECASE)
    for e in _entries(bundle):
        r = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(r, dict):
            continue
        rt = r.get("resourceType")
        if rt not in ("Condition", "Observation", "Encounter"):
            continue
        haystack: list[str] = []
        code = r.get("code") or {}
        haystack.append((code.get("text") or "").lower())
        for c in code.get("coding") or []:
            if isinstance(c, dict):
                haystack.append((c.get("display") or "").lower())
        for n in r.get("note") or []:
            if isinstance(n, dict):
                haystack.append((n.get("text") or "").lower())
        if rt == "Observation":
            vq = r.get("valueQuantity") or {}
            unit = (vq.get("unit") or "").lower()
            val = vq.get("value")
            if val is not None and any(u in unit for u in ("wk", "week")):
                try:
                    return int(float(val))
                except (TypeError, ValueError):
                    pass
        blob = " ".join(haystack)
        if not any(k in blob for k in ("gestational age", "gestation",
                                       "weeks pregnancy", "wks pregnancy")):
            continue
        m = pattern.search(blob)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


# ─────────────────────────────────────────────────────────────────────
# Public resolver
# ─────────────────────────────────────────────────────────────────────


async def harden_clinical_inputs(
    caller: dict[str, Any],
    chart_derivable: set[str] | None = None,
) -> tuple[dict[str, Any], bool, set[str]]:
    """Resolve clinical inputs from the SHARP-bound chart.

    Args:
        caller: dict of param name -> caller-supplied value. Keys not
            in `chart_derivable` (or all keys when None) are passed
            through unchanged.
        chart_derivable: set of param names that MUST be sourced from
            the FHIR chart when SHARP context is bound. When None, all
            keys in `caller` are considered chart-derivable.

    Returns:
        (resolved, sharp_bound, missing) where:
          - resolved: dict same shape as caller; chart values override
            caller for keys in `chart_derivable` when SHARP bound.
          - sharp_bound: True when the FHIR bundle was successfully
            fetched (production / live SHARP context).
          - missing: set of chart_derivable keys for which the chart
            had no reading. When SHARP bound and `missing` is non-empty,
            the calling tool should abstain (clinical data cannot come
            from the chat-side caller).

    Offline behavior:
        When no SHARP context is bound, `resolved` is a copy of `caller`,
        `sharp_bound` is False, and `missing` is empty. Tests + direct
        CLI invocations keep working without modification.
    """
    bundle: dict | None = None
    try:
        pid = await resolve_patient_id(None)
        bundle = await fetch_patient_bundle(pid)
        sharp_bound = True
    except Exception:
        sharp_bound = False

    if not sharp_bound:
        return dict(caller), False, set()

    chart: dict[str, Any] = {}
    chart.update(extract_chart_demographics(bundle))
    chart.update(extract_chart_vitals(bundle))
    chart.update(extract_chart_labs(bundle))
    meds = extract_chart_medications(bundle)
    if meds:
        for k in ("medications", "current_medications",
                  "active_medications", "home_medications"):
            chart[k] = meds
    ga = extract_chart_gestational_age(bundle)
    if ga is not None:
        chart["gestational_age_weeks"] = ga

    keys_to_override = (
        chart_derivable if chart_derivable is not None else set(caller.keys())
    )
    resolved = dict(caller)
    missing: set[str] = set()
    for key in keys_to_override:
        if key not in caller:
            continue
        if key in chart and chart[key] is not None:
            resolved[key] = chart[key]
        else:
            # Chart has nothing -- discard caller's value (which might
            # be hallucinated) and mark as missing so the tool abstains.
            resolved[key] = None
            missing.add(key)
    return resolved, sharp_bound, missing


def chart_abstain_reason(missing: set[str]) -> str:
    """Canonical abstain_reason string for tools that short-circuit on
    `missing` returned by `harden_clinical_inputs`."""
    keys = ", ".join(sorted(missing))
    return (
        f"unverified_in_chart: chart-derivable inputs not present in the "
        f"SHARP-bound patient's FHIR bundle: [{keys}]. Caller-supplied "
        f"values are discarded for these parameters to prevent the "
        f"chat-side LLM from injecting clinical data."
    )
