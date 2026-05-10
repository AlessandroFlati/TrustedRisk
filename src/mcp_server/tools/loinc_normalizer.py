"""healthcare.compute_normalize_observations -- DATA-3.

Normalize FHIR-shaped Observation entries to LOINC. Input: list of dicts
with at least one of {code+system, name}. Output: a list of
LOINCNormalizedObservation records keyed back to the source.

Strategy (in order):
  1. If `system == http://loinc.org`, accept as canonical and look up the
     canonical name from the embedded table.
  2. If a local-code synonym exists (`(system, code)` in `_LOCAL_SYNONYMS`),
     map to the canonical LOINC.
  3. Fallback: case-insensitive match on `name` against `_NAME_SYNONYMS`.
  4. If nothing matches, return the entry with `normalized_loinc_code=None`.

The embedded table covers the ~50 most common labs the LACE / discharge /
deterioration tools consume. For institutional deployments the full LOINC
table can be loaded from `LOINC_TABLE_PATH` (CSV with code,name,...).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from shared.schemas import (
    LOINCNormalizationReport,
    LOINCNormalizedObservation,
)


# ─────────────────────── Embedded canonical table ───────────────────────
#
# Each row: loinc_code -> (canonical_name, short_name, unit_canonical,
# component_class). Curated subset; expand via LOINC_TABLE_PATH override.

_LOINC_CANONICAL: dict[str, tuple[str, str, str, str]] = {
    # Electrolytes
    "2951-2": ("Sodium [Moles/volume] in Serum or Plasma",
                "Na",  "mmol/L", "electrolyte"),
    "2823-3": ("Potassium [Moles/volume] in Serum or Plasma",
                "K",   "mmol/L", "electrolyte"),
    "2069-3": ("Chloride [Moles/volume] in Serum or Plasma",
                "Cl",  "mmol/L", "electrolyte"),
    "2028-9": ("Carbon dioxide, total [Moles/volume] in Serum or Plasma",
                "CO2", "mmol/L", "electrolyte"),
    # Renal
    "2160-0": ("Creatinine [Mass/volume] in Serum or Plasma",
                "Cr",  "mg/dL",  "renal"),
    "3094-0": ("Urea nitrogen [Mass/volume] in Serum or Plasma",
                "BUN", "mg/dL",  "renal"),
    "33914-3": ("Glomerular filtration rate/1.73 sq M.predicted "
                  "[Volume Rate/Area] in Serum, Plasma or Blood",
                "eGFR", "mL/min/1.73m2", "renal"),
    # Hematology
    "718-7": ("Hemoglobin [Mass/volume] in Blood",
                "Hgb", "g/dL", "hematology"),
    "4544-3": ("Hematocrit [Volume Fraction] of Blood by Automated count",
                "Hct", "%", "hematology"),
    "26515-7": ("Platelets [#/volume] in Blood",
                  "Plt", "10*3/uL", "hematology"),
    "6690-2": ("Leukocytes [#/volume] in Blood by Automated count",
                "WBC", "10*3/uL", "hematology"),
    "789-8": ("Erythrocytes [#/volume] in Blood by Automated count",
                "RBC", "10*6/uL", "hematology"),
    # Glycemic
    "2345-7": ("Glucose [Mass/volume] in Serum or Plasma",
                "Glu",   "mg/dL", "glycemic"),
    "2339-0": ("Glucose [Mass/volume] in Blood",
                "Glu (whole blood)", "mg/dL", "glycemic"),
    "4548-4": ("Hemoglobin A1c/Hemoglobin.total in Blood",
                "HbA1c", "%", "glycemic"),
    # Cardiac
    "10839-9": ("Troponin I.cardiac [Mass/volume] in Serum or Plasma",
                  "TnI", "ng/mL", "cardiac"),
    "67151-1": ("Troponin T.cardiac [Mass/volume] in Serum or Plasma "
                  "by High sensitivity method",
                  "hs-TnT", "ng/L", "cardiac"),
    "33762-6": ("Natriuretic peptide.B prohormone N-Terminal "
                  "[Mass/volume] in Serum or Plasma",
                  "NT-proBNP", "pg/mL", "cardiac"),
    "30934-4": ("Natriuretic peptide.B [Mass/volume] in Serum or Plasma",
                  "BNP", "pg/mL", "cardiac"),
    # Coagulation
    "5902-2": ("Prothrombin time (PT)",
                "PT", "s", "coagulation"),
    "6301-6": ("INR in Platelet poor plasma by Coagulation assay",
                "INR", "{ratio}", "coagulation"),
    "3173-2": ("aPTT in Platelet poor plasma by Coagulation assay",
                "aPTT", "s", "coagulation"),
    # Inflammation
    "1988-5": ("C reactive protein [Mass/volume] in Serum or Plasma",
                "CRP", "mg/L", "inflammation"),
    "30341-2": ("Erythrocyte sedimentation rate",
                  "ESR", "mm/h", "inflammation"),
    "2951-3": ("Procalcitonin [Mass/volume] in Serum or Plasma",
                "PCT", "ng/mL", "inflammation"),
    # Liver
    "1742-6": ("Alanine aminotransferase [Enzymatic activity/volume]",
                "ALT", "U/L", "liver"),
    "1920-8": ("Aspartate aminotransferase [Enzymatic activity/volume]",
                "AST", "U/L", "liver"),
    "1975-2": ("Bilirubin.total [Mass/volume] in Serum or Plasma",
                "TBili", "mg/dL", "liver"),
    "1751-7": ("Albumin [Mass/volume] in Serum or Plasma",
                "Alb", "g/dL", "liver"),
    # Lipids
    "2093-3": ("Cholesterol [Mass/volume] in Serum or Plasma",
                "Chol", "mg/dL", "lipid"),
    "2571-8": ("Triglycerides [Mass/volume] in Serum or Plasma",
                "TG",   "mg/dL", "lipid"),
    "2085-9": ("Cholesterol in HDL [Mass/volume] in Serum or Plasma",
                "HDL", "mg/dL", "lipid"),
    "2089-1": ("Cholesterol in LDL [Mass/volume] in Serum or Plasma",
                "LDL", "mg/dL", "lipid"),
    # Vital signs (LOINC encodes these)
    "8867-4": ("Heart rate", "HR", "/min", "vital_sign"),
    "9279-1": ("Respiratory rate", "RR", "/min", "vital_sign"),
    "8310-5": ("Body temperature", "Temp", "Cel", "vital_sign"),
    "8480-6": ("Systolic blood pressure", "SBP", "mm[Hg]", "vital_sign"),
    "8462-4": ("Diastolic blood pressure", "DBP", "mm[Hg]", "vital_sign"),
    "59408-5": ("Oxygen saturation in Arterial blood by Pulse oximetry",
                  "SpO2", "%", "vital_sign"),
    "29463-7": ("Body weight", "Weight", "kg", "vital_sign"),
    "8302-2": ("Body height", "Height", "cm", "vital_sign"),
    "39156-5": ("Body mass index (BMI)", "BMI", "kg/m2", "vital_sign"),
    # Acid-base / arterial gas
    "2744-1": ("pH of Arterial blood", "pH", "{pH}", "abg"),
    "1925-7": ("Bicarbonate [Moles/volume] in Arterial blood",
                "HCO3", "mmol/L", "abg"),
    "2019-8": ("Carbon dioxide partial pressure in Arterial blood",
                "pCO2", "mm[Hg]", "abg"),
    "2703-7": ("Oxygen partial pressure in Arterial blood",
                "pO2", "mm[Hg]", "abg"),
    "32693-4": ("Lactate [Moles/volume] in Blood",
                  "Lactate", "mmol/L", "abg"),
}


# Common local code -> LOINC mappings for major US/EU EHR vendors that
# emit Observations in their own dictionaries. The (system, raw_code)
# tuple is the lookup key; many EHRs publish under the empty / null
# system, in which case we fall back to name matching.

_LOCAL_SYNONYMS: dict[tuple[str, str], str] = {
    # Epic-style local codes for vital signs
    ("epic", "HR"): "8867-4",
    ("epic", "RR"): "9279-1",
    ("epic", "TEMP"): "8310-5",
    ("epic", "BPS"): "8480-6",
    ("epic", "BPD"): "8462-4",
    ("epic", "SPO2"): "59408-5",
    # Cerner local codes
    ("cerner", "K"): "2823-3",
    ("cerner", "NA"): "2951-2",
    ("cerner", "GLU"): "2345-7",
    ("cerner", "CR"): "2160-0",
    ("cerner", "BUN"): "3094-0",
    ("cerner", "HGB"): "718-7",
    # Generic
    ("local", "K"): "2823-3",
    ("local", "Na"): "2951-2",
    ("local", "Cl"): "2069-3",
    ("local", "Cr"): "2160-0",
    ("local", "BUN"): "3094-0",
    ("local", "Hgb"): "718-7",
    ("local", "Plt"): "26515-7",
    ("local", "WBC"): "6690-2",
    ("local", "INR"): "6301-6",
    ("local", "TnI"): "10839-9",
    ("local", "BNP"): "30934-4",
    ("local", "Glu"): "2345-7",
    ("local", "HbA1c"): "4548-4",
    ("local", "Lactate"): "32693-4",
    ("local", "ALT"): "1742-6",
    ("local", "AST"): "1920-8",
    ("local", "TBili"): "1975-2",
    ("local", "Alb"): "1751-7",
    ("local", "CRP"): "1988-5",
    ("local", "eGFR"): "33914-3",
    ("local", "pH"): "2744-1",
    ("local", "HCO3"): "1925-7",
}


# Build name synonym table from the canonical labels (case + abbreviation
# variants).
_NAME_SYNONYMS: dict[str, str] = {}
for code, (canonical, short, _u, _c) in _LOINC_CANONICAL.items():
    _NAME_SYNONYMS[canonical.lower()] = code
    _NAME_SYNONYMS[short.lower()] = code

# Hand-curated additional aliases
_NAME_SYNONYMS.update({
    "sodium": "2951-2",
    "potassium": "2823-3",
    "creatinine": "2160-0",
    "hemoglobin": "718-7",
    "haemoglobin": "718-7",
    "platelets": "26515-7",
    "white blood cell count": "6690-2",
    "leukocytes": "6690-2",
    "glucose": "2345-7",
    "blood glucose": "2345-7",
    "hba1c": "4548-4",
    "hemoglobin a1c": "4548-4",
    "troponin": "10839-9",
    "troponin i": "10839-9",
    "high-sensitivity troponin": "67151-1",
    "bnp": "30934-4",
    "nt-probnp": "33762-6",
    "inr": "6301-6",
    "ptt": "3173-2",
    "aptt": "3173-2",
    "alt": "1742-6",
    "ast": "1920-8",
    "alanine aminotransferase": "1742-6",
    "lactate": "32693-4",
    "egfr": "33914-3",
    "heart rate": "8867-4",
    "pulse": "8867-4",
    "respiratory rate": "9279-1",
    "body temperature": "8310-5",
    "temperature": "8310-5",
    "oxygen saturation": "59408-5",
    "spo2": "59408-5",
    "blood pressure systolic": "8480-6",
    "blood pressure diastolic": "8462-4",
})


# ─────────────────────── External LOINC table override ───────────────────────

def _load_external_table_if_available() -> None:
    """If LOINC_TABLE_PATH is set, augment _LOINC_CANONICAL from the CSV.

    Expected CSV columns: LOINC_NUM, COMPONENT, SHORTNAME, EXAMPLE_UCUM_UNITS,
    CLASS -- matches the official LOINC table layout. Idempotent -- call once
    at module load.
    """
    path = os.environ.get("LOINC_TABLE_PATH")
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = (row.get("LOINC_NUM") or "").strip()
                if not code or code in _LOINC_CANONICAL:
                    continue
                _LOINC_CANONICAL[code] = (
                    (row.get("COMPONENT") or "")[:200],
                    (row.get("SHORTNAME") or "")[:50],
                    (row.get("EXAMPLE_UCUM_UNITS") or "")[:30],
                    (row.get("CLASS") or "other")[:30],
                )
    except Exception:
        pass


_load_external_table_if_available()


# ─────────────────────── Lookup helpers ───────────────────────

def _system_normalized(system: str) -> str:
    s = (system or "").strip().lower()
    if "loinc" in s:
        return "loinc"
    if "snomed" in s:
        return "snomed"
    if "rxnorm" in s:
        return "rxnorm"
    if s.startswith("urn:") or s.startswith("http"):
        # Unknown URI -- treat as opaque local
        return "local"
    return s or "local"


def _resolve_loinc(obs: dict[str, Any]) -> tuple[str | None, str, str]:
    """Return (loinc_code | None, raw_code, raw_system)."""
    raw_code = str(obs.get("code", "")).strip()
    raw_system = _system_normalized(str(obs.get("system", "")))
    name = str(obs.get("name", "")).strip()

    # 1. Already LOINC?
    if raw_system == "loinc" and raw_code in _LOINC_CANONICAL:
        return raw_code, raw_code, "loinc"

    # 2. Local-code synonym
    direct_key = (raw_system, raw_code)
    if direct_key in _LOCAL_SYNONYMS:
        return _LOCAL_SYNONYMS[direct_key], raw_code, raw_system
    # Try case-insensitive
    upper_key = (raw_system, raw_code.upper())
    if upper_key in _LOCAL_SYNONYMS:
        return _LOCAL_SYNONYMS[upper_key], raw_code, raw_system

    # 3. Name synonym
    name_l = name.lower()
    if name_l in _NAME_SYNONYMS:
        return _NAME_SYNONYMS[name_l], raw_code, raw_system

    # 4. No match
    return None, raw_code, raw_system


# ─────────────────────── Public API ───────────────────────

async def compute_normalize_observations(
    observations: list[dict[str, Any]],
) -> LOINCNormalizationReport:
    """Normalize a list of FHIR-shaped Observation dicts to LOINC.

    Args:
        observations: list of dicts with any subset of {code, system, name}.

    Returns:
        LOINCNormalizationReport -- one entry per input, with LOINC code +
        canonical name when resolvable, else null fields.
    """
    if not isinstance(observations, list):
        raise ValueError("observations must be a list of dicts.")

    out: list[LOINCNormalizedObservation] = []
    n_normalized = 0
    n_unmapped = 0

    for obs in observations:
        if not isinstance(obs, dict):
            continue
        code, raw_code, raw_system = _resolve_loinc(obs)
        if code is None or code not in _LOINC_CANONICAL:
            n_unmapped += 1
            out.append(LOINCNormalizedObservation(
                raw_code=raw_code, raw_system=raw_system,
                normalized_loinc_code=None,
            ))
            continue
        canonical, short, unit, klass = _LOINC_CANONICAL[code]
        n_normalized += 1
        out.append(LOINCNormalizedObservation(
            raw_code=raw_code, raw_system=raw_system,
            normalized_loinc_code=code,
            canonical_name=canonical,
            short_name=short,
            unit_canonical=unit,
            component_class=klass,
        ))

    method = "loinc_table" if n_normalized > 0 else "deterministic_fallback"
    return LOINCNormalizationReport(
        n_input=len([o for o in observations if isinstance(o, dict)]),
        n_normalized=n_normalized,
        n_unmapped=n_unmapped,
        observations=out,
        method=method,                  # type: ignore[arg-type]
    )


def register(mcp) -> None:
    mcp.tool()(compute_normalize_observations)
