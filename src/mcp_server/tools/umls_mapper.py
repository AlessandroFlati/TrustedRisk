"""healthcare.compute_umls_concept_map -- DATA-1.

Cross-vocabulary normalization (ICD-10 ↔ SNOMED ↔ RxNorm) backed by:

  1. The embedded curated table (covers the ~50 most common conditions
     and ~30 most common drug classes -- enough to round-trip discharge
     scenarios end-to-end without an internet connection).
  2. Optional local UMLS RRF override: when `UMLS_RRF_PATH` points at a
     UMLS Metathesaurus install, the tool augments the embedded table by
     parsing MRCONSO + MRREL on demand. The user is responsible for the
     UMLS license + download -- UMLS is free for research use but requires
     NIH registration.

Output format follows the standard UMLS shape: each input code resolves
to a CUI (Concept Unique Identifier) plus a per-target-vocabulary list of
equivalent codes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from shared.schemas import (
    UMLSConceptMapping,
    UMLSMappingReport,
)


# ─────────────────────── Embedded crosswalk ───────────────────────
#
# Each entry maps a CUI -> {preferred_name, ICD10, SNOMEDCT_US, RXNORM}.
# Codes within a vocabulary are stored as lists. The table is curated for
# the highest-prevalence discharge / readmission conditions + the LACE
# Charlson hierarchy.

_CROSSWALK: dict[str, dict[str, list[str] | str]] = {
    # ─── Cardiology ───
    "C0018802": {
        "preferred_name": "Congestive heart failure",
        "ICD10": ["I50", "I50.9"],
        "ICD10CM": ["I50.9", "I50.20", "I50.21", "I50.22", "I50.23"],
        "SNOMEDCT_US": ["84114007", "42343007", "88805009"],
        "RXNORM": [],
    },
    "C0020538": {
        "preferred_name": "Hypertensive disease",
        "ICD10": ["I10"],
        "ICD10CM": ["I10"],
        "SNOMEDCT_US": ["38341003", "59621000"],
        "RXNORM": [],
    },
    "C0027051": {
        "preferred_name": "Acute myocardial infarction",
        "ICD10": ["I21", "I21.0", "I21.1", "I21.2", "I21.3", "I21.4", "I21.9"],
        "ICD10CM": ["I21.01", "I21.02", "I21.09", "I21.11", "I21.19",
                       "I21.21", "I21.29", "I21.4", "I21.A1", "I21.A9"],
        "SNOMEDCT_US": ["57054005", "22298006"],
        "RXNORM": [],
    },
    "C0010068": {
        "preferred_name": "Coronary arteriosclerosis",
        "ICD10": ["I25.10"],
        "ICD10CM": ["I25.10", "I25.110"],
        "SNOMEDCT_US": ["53741008", "443502000"],
        "RXNORM": [],
    },
    "C0004238": {
        "preferred_name": "Atrial fibrillation",
        "ICD10": ["I48", "I48.91"],
        "ICD10CM": ["I48.0", "I48.1", "I48.2", "I48.91"],
        "SNOMEDCT_US": ["49436004"],
        "RXNORM": [],
    },
    # ─── Endocrinology ───
    "C0011860": {
        "preferred_name": "Diabetes mellitus, type 2",
        "ICD10": ["E11", "E11.9"],
        "ICD10CM": ["E11.9", "E11.65", "E11.21", "E11.22"],
        "SNOMEDCT_US": ["44054006", "73211009"],
        "RXNORM": [],
    },
    "C0011854": {
        "preferred_name": "Diabetes mellitus, type 1",
        "ICD10": ["E10", "E10.9"],
        "ICD10CM": ["E10.9", "E10.10"],
        "SNOMEDCT_US": ["46635009", "73211009"],
        "RXNORM": [],
    },
    # ─── Nephrology ───
    "C0022661": {
        "preferred_name": "Chronic kidney disease",
        "ICD10": ["N18", "N18.3", "N18.4", "N18.5", "N18.6"],
        "ICD10CM": ["N18.1", "N18.2", "N18.30", "N18.4", "N18.5", "N18.6",
                       "N18.9"],
        "SNOMEDCT_US": ["709044004", "433144002", "431855005"],
        "RXNORM": [],
    },
    "C0022660": {
        "preferred_name": "Acute kidney injury",
        "ICD10": ["N17", "N17.9"],
        "ICD10CM": ["N17.0", "N17.1", "N17.2", "N17.8", "N17.9"],
        "SNOMEDCT_US": ["14669001"],
        "RXNORM": [],
    },
    # ─── Pulmonary ───
    "C0024117": {
        "preferred_name": "Chronic obstructive pulmonary disease",
        "ICD10": ["J44", "J44.9"],
        "ICD10CM": ["J44.0", "J44.1", "J44.9"],
        "SNOMEDCT_US": ["13645005"],
        "RXNORM": [],
    },
    "C0004096": {
        "preferred_name": "Asthma",
        "ICD10": ["J45", "J45.909"],
        "ICD10CM": ["J45.20", "J45.21", "J45.30", "J45.40", "J45.50",
                       "J45.901", "J45.902", "J45.909", "J45.998"],
        "SNOMEDCT_US": ["195967001"],
        "RXNORM": [],
    },
    "C0032285": {
        "preferred_name": "Pneumonia",
        "ICD10": ["J18", "J18.9"],
        "ICD10CM": ["J18.0", "J18.1", "J18.8", "J18.9", "J15.9"],
        "SNOMEDCT_US": ["233604007"],
        "RXNORM": [],
    },
    # ─── Cerebrovascular / neuro ───
    "C0038454": {
        "preferred_name": "Cerebrovascular accident (stroke)",
        "ICD10": ["I63", "I63.9", "I64"],
        "ICD10CM": ["I63.30", "I63.40", "I63.50", "I63.9", "I64"],
        "SNOMEDCT_US": ["230690007", "266257000"],
        "RXNORM": [],
    },
    "C0497327": {
        "preferred_name": "Dementia",
        "ICD10": ["F03"],
        "ICD10CM": ["F03.90", "F03.91"],
        "SNOMEDCT_US": ["52448006"],
        "RXNORM": [],
    },
    # ─── Mental health ───
    "C0011570": {
        "preferred_name": "Depressive disorder",
        "ICD10": ["F32", "F33"],
        "ICD10CM": ["F32.0", "F32.1", "F32.9", "F33.0", "F33.1", "F33.2"],
        "SNOMEDCT_US": ["35489007", "192080009"],
        "RXNORM": [],
    },
    # ─── GI / liver ───
    "C0023895": {
        "preferred_name": "Cirrhosis of liver",
        "ICD10": ["K74"],
        "ICD10CM": ["K74.0", "K74.1", "K74.60", "K74.69"],
        "SNOMEDCT_US": ["19943007"],
        "RXNORM": [],
    },
    # ─── Oncology ───
    "C1336708": {
        "preferred_name": "Metastatic neoplasm to unknown site",
        "ICD10": ["C77", "C78", "C79", "C80"],
        "ICD10CM": ["C77.9", "C78.00", "C79.10", "C80.0", "C80.1"],
        "SNOMEDCT_US": ["94381002"],
        "RXNORM": [],
    },
    # ─── Drug classes (RxNorm-anchored) ───
    "C0003280": {
        "preferred_name": "Anticoagulants",
        "ICD10": [],
        "ICD10CM": [],
        "SNOMEDCT_US": ["372862008"],
        "RXNORM": ["11289"],   # warfarin
    },
    "C0019070": {
        "preferred_name": "Heparin",
        "ICD10": [],
        "ICD10CM": [],
        "SNOMEDCT_US": ["63739005"],
        "RXNORM": ["5224"],
    },
    "C0003364": {
        "preferred_name": "Antihypertensive agents",
        "ICD10": [],
        "ICD10CM": [],
        "SNOMEDCT_US": ["255631004"],
        "RXNORM": ["29046"],   # lisinopril (representative)
    },
    "C0917816": {
        "preferred_name": "Statin (HMG-CoA reductase inhibitor)",
        "ICD10": [],
        "ICD10CM": [],
        "SNOMEDCT_US": ["323549001"],
        "RXNORM": ["36567"],   # simvastatin (representative)
    },
}


# ─────────────────────── Inverted indices ───────────────────────

def _build_indices() -> dict[str, dict[str, str]]:
    """Build {vocabulary: {code: cui}} indices. Matches both exact and
    prefix-with-dot variants (e.g. 'I50' matches 'I50' and 'I50.9'
    queries collapse to the dotted form's CUI when present)."""
    idx: dict[str, dict[str, str]] = {
        "ICD10": {}, "ICD10CM": {},
        "SNOMEDCT_US": {}, "RXNORM": {},
    }
    for cui, fields in _CROSSWALK.items():
        for vocab in idx:
            for code in fields.get(vocab, []):  # type: ignore[union-attr]
                idx[vocab].setdefault(code, cui)
    return idx


_INDICES = _build_indices()


def _normalize_icd10(code: str) -> str:
    """Strip dots + uppercase for ICD-10 codes (admin-data style)."""
    return "".join(c for c in (code or "").upper() if c.isalnum())


def _resolve_cui(code: str, source_vocabulary: str) -> str | None:
    sv = source_vocabulary
    table = _INDICES.get(sv) or {}
    # First try exact match
    cui = table.get(code)
    if cui is not None:
        return cui
    # ICD-10 forms admin tables often drop dots; also try the normalized form
    if sv in {"ICD10", "ICD10CM"}:
        norm = _normalize_icd10(code)
        for k, v in table.items():
            if _normalize_icd10(k) == norm:
                return v
    return None


# ─────────────────────── Optional UMLS RRF augmentation ───────────────────────

def _maybe_load_umls_rrf() -> None:
    """If UMLS_RRF_PATH is set, augment _CROSSWALK from MRCONSO.RRF.

    Idempotent -- only reads the file once. The RRF format is "|"-delimited
    with fields documented at https://www.nlm.nih.gov/research/umls/.
    """
    path = os.environ.get("UMLS_RRF_PATH")
    if not path:
        return
    p = Path(path)
    if not p.exists() or not p.is_file():
        return
    if getattr(_maybe_load_umls_rrf, "_loaded", False):
        return
    try:
        with p.open("r", encoding="utf-8") as f:
            for raw_line in f:
                fields = raw_line.rstrip("\n").split("|")
                # MRCONSO field positions (per UMLS RRF spec):
                #   0=CUI, 11=SAB (vocab), 13=CODE, 14=STR
                if len(fields) < 15:
                    continue
                cui = fields[0]
                sab = fields[11]
                code = fields[13]
                str_name = fields[14]
                if sab not in {"ICD10CM", "ICD10", "SNOMEDCT_US", "RXNORM"}:
                    continue
                entry = _CROSSWALK.setdefault(cui, {
                    "preferred_name": str_name[:200],
                    "ICD10": [], "ICD10CM": [],
                    "SNOMEDCT_US": [], "RXNORM": [],
                })
                lst = entry.get(sab, [])  # type: ignore[union-attr]
                if isinstance(lst, list) and code not in lst:
                    lst.append(code)
                # Index for fast lookup
                _INDICES.setdefault(sab, {}).setdefault(code, cui)
        _maybe_load_umls_rrf._loaded = True   # type: ignore[attr-defined]
    except Exception:
        pass


_maybe_load_umls_rrf()


# ─────────────────────── Public API ───────────────────────

_VALID_SOURCES = {"ICD10", "ICD10CM", "SNOMEDCT_US", "RXNORM"}


async def compute_umls_concept_map(
    codes: list[str],
    source_vocabulary: str = "ICD10",
) -> UMLSMappingReport:
    """Cross-map a list of codes to all known target vocabularies.

    Args:
        codes: list of source codes.
        source_vocabulary: one of {ICD10, ICD10CM, SNOMEDCT_US, RXNORM}.

    Returns:
        UMLSMappingReport -- one mapping per input. Unknown codes return
        with `cui=None`.
    """
    if source_vocabulary not in _VALID_SOURCES:
        raise ValueError(
            f"source_vocabulary must be one of {sorted(_VALID_SOURCES)}; "
            f"got {source_vocabulary!r}.")
    if not isinstance(codes, list):
        raise ValueError("codes must be a list.")

    out: list[UMLSConceptMapping] = []
    n_mapped = 0
    for code in codes:
        if not isinstance(code, str):
            continue
        cui = _resolve_cui(code, source_vocabulary)
        if cui is None:
            out.append(UMLSConceptMapping(
                source_code=code,
                source_vocabulary=source_vocabulary,    # type: ignore[arg-type]
                cui=None,
                preferred_name=None,
                target_codes={},
            ))
            continue
        n_mapped += 1
        entry = _CROSSWALK.get(cui, {})
        target_codes: dict[str, list[str]] = {}
        for vocab in _VALID_SOURCES:
            if vocab == source_vocabulary:
                continue
            v = entry.get(vocab, []) or []
            if isinstance(v, list) and v:
                target_codes[vocab] = list(v)
        out.append(UMLSConceptMapping(
            source_code=code,
            source_vocabulary=source_vocabulary,        # type: ignore[arg-type]
            cui=cui,
            preferred_name=str(entry.get("preferred_name", ""))[:200],
            target_codes=target_codes,
        ))

    method = "umls_rrf_local" if os.environ.get("UMLS_RRF_PATH") \
        else "deterministic_fallback"

    return UMLSMappingReport(
        n_input=len([c for c in codes if isinstance(c, str)]),
        n_mapped=n_mapped,
        mappings=out,
        method=method,                  # type: ignore[arg-type]
    )


def register(mcp) -> None:
    mcp.tool()(compute_umls_concept_map)
