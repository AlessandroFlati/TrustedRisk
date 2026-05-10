"""healthcare.compute_charlson_elixhauser_index -- DATA-4.

Maps a list of ICD-10 (or ICD-10-CM) codes to two canonical comorbidity
indices:

  - Charlson Comorbidity Index (CCI) per Quan H et al. *Coding algorithms
    for defining comorbidities in ICD-9-CM and ICD-10 administrative data.*
    Med Care 2005;43:1130. (17 conditions, weighted 1-6).

  - Elixhauser Comorbidity Score per the AHRQ HCUP refined ICD-10-CM
    software, summarized via the van Walraven 2009 single-point score
    (Med Care 2009;47:626) -- 30 conditions, mapped to a single value.

Pure deterministic mapping (no external lookup); the ICD-10 prefix tables
are encoded in this module under the cited primary references. The output
flags every recognized condition + the contributing codes for full
auditability.
"""

from __future__ import annotations

import re
from typing import Iterable

from shared.schemas import ComorbidityIndices


# ─────────────────────── Charlson -- Quan 2005 ICD-10 ───────────────────────
#
# Each entry: (condition_label, weight, list of ICD-10 prefixes).
# Prefixes match by startswith() after normalization (uppercase, strip).
#
# Quan 2005 Table 1; weights from the original Charlson 1987 update.

_CHARLSON: list[tuple[str, int, list[str]]] = [
    ("myocardial_infarction", 1,
     ["I21", "I22", "I252"]),
    ("congestive_heart_failure", 1,
     ["I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426",
      "I427", "I428", "I429", "I43", "I50", "P290"]),
    ("peripheral_vascular_disease", 1,
     ["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792",
      "K551", "K558", "K559", "Z958", "Z959"]),
    ("cerebrovascular_disease", 1,
     ["G45", "G46", "H340", "I60", "I61", "I62", "I63", "I64", "I65",
      "I66", "I67", "I68", "I69"]),
    ("dementia", 1,
     ["F00", "F01", "F02", "F03", "F051", "G30", "G311"]),
    ("chronic_pulmonary_disease", 1,
     ["I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46",
      "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67",
      "J684", "J701", "J703"]),
    ("connective_tissue_disease", 1,
     ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"]),
    ("peptic_ulcer_disease", 1,
     ["K25", "K26", "K27", "K28"]),
    ("mild_liver_disease", 1,
     ["B18", "K700", "K701", "K702", "K703", "K709", "K717", "K73",
      "K74", "K760", "K762", "K763", "K764", "K768", "K769", "Z944"]),
    ("diabetes_uncomplicated", 1,
     ["E100", "E101", "E106", "E108", "E109",
      "E110", "E111", "E116", "E118", "E119",
      "E120", "E121", "E126", "E128", "E129",
      "E130", "E131", "E136", "E138", "E139",
      "E140", "E141", "E146", "E148", "E149"]),
    ("diabetes_with_complications", 2,
     ["E102", "E103", "E104", "E105", "E107",
      "E112", "E113", "E114", "E115", "E117",
      "E122", "E123", "E124", "E125", "E127",
      "E132", "E133", "E134", "E135", "E137",
      "E142", "E143", "E144", "E145", "E147"]),
    ("paraplegia_hemiplegia", 2,
     ["G041", "G114", "G801", "G802", "G81", "G82",
      "G830", "G831", "G832", "G833", "G834", "G839"]),
    ("renal_disease", 2,
     ["I120", "I131", "N032", "N033", "N034", "N035", "N036", "N037",
      "N052", "N053", "N054", "N055", "N056", "N057", "N18", "N19",
      "N250", "Z490", "Z491", "Z492", "Z940", "Z992"]),
    ("solid_tumor_localized", 2,
     ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08",
      "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17",
      "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
      "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40",
      "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C50", "C51",
      "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60", "C61",
      "C62", "C63", "C64", "C65", "C66", "C67", "C68", "C69", "C70",
      "C71", "C72", "C73", "C74", "C75", "C76", "C81", "C82", "C83",
      "C84", "C85", "C88", "C90", "C91", "C92", "C93", "C94", "C95",
      "C96", "C97"]),
    ("moderate_severe_liver_disease", 3,
     ["I850", "I859", "I864", "I982", "K704", "K711", "K721", "K729",
      "K765", "K766", "K767"]),
    ("metastatic_solid_tumor", 6,
     ["C77", "C78", "C79", "C80"]),
    ("hiv_aids", 6,
     ["B20", "B21", "B22", "B24"]),
]


# ─────────────────────── Elixhauser -- AHRQ HCUP ICD-10 ───────────────────────
#
# 30 conditions per the AHRQ Comorbidity Software for ICD-10-CM (refined
# version). Each is mapped to a van Walraven point weight (Med Care 2009;
# 47:626) -- sum of weights yields the single-value Elixhauser score.

_ELIXHAUSER: list[tuple[str, int, list[str]]] = [
    ("congestive_heart_failure", 7,
     ["I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426",
      "I427", "I428", "I429", "I43", "I50", "P290"]),
    ("cardiac_arrhythmia", 5,
     ["I441", "I442", "I443", "I456", "I459", "I47", "I48", "I49",
      "R000", "R001", "R008", "T821", "Z450", "Z950"]),
    ("valvular_disease", -1,
     ["A520", "I05", "I06", "I07", "I08", "I091", "I098", "I34", "I35",
      "I36", "I37", "I38", "I39", "Q230", "Q231", "Q232", "Q233", "Z952",
      "Z953", "Z954"]),
    ("pulmonary_circulation_disorders", 4,
     ["I26", "I27", "I280", "I288", "I289"]),
    ("peripheral_vascular_disease", 2,
     ["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792",
      "K551", "K558", "K559", "Z958", "Z959"]),
    ("hypertension_uncomplicated", 0,
     ["I10"]),
    ("hypertension_complicated", 0,
     ["I11", "I12", "I13", "I15"]),
    ("paralysis", 7,
     ["G041", "G114", "G801", "G802", "G81", "G82",
      "G830", "G831", "G832", "G833", "G834", "G839"]),
    ("other_neurological", 6,
     ["G10", "G11", "G12", "G13", "G20", "G21", "G22", "G254", "G255",
      "G312", "G318", "G319", "G32", "G35", "G36", "G37", "G40", "G41",
      "G931", "G934", "R470", "R56"]),
    ("chronic_pulmonary_disease", 3,
     ["I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46",
      "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67",
      "J684", "J701", "J703"]),
    ("diabetes_uncomplicated", 0,
     ["E100", "E101", "E109", "E110", "E111", "E119", "E120", "E121",
      "E129", "E130", "E131", "E139", "E140", "E141", "E149"]),
    ("diabetes_complicated", 0,
     ["E102", "E103", "E104", "E105", "E106", "E107", "E108",
      "E112", "E113", "E114", "E115", "E116", "E117", "E118",
      "E122", "E123", "E124", "E125", "E126", "E127", "E128",
      "E132", "E133", "E134", "E135", "E136", "E137", "E138",
      "E142", "E143", "E144", "E145", "E146", "E147", "E148"]),
    ("hypothyroidism", 0,
     ["E00", "E01", "E02", "E03", "E890"]),
    ("renal_failure", 5,
     ["I120", "I131", "N18", "N19", "N250", "Z490", "Z491", "Z492",
      "Z940", "Z992"]),
    ("liver_disease", 11,
     ["B18", "I85", "I864", "I982", "K70", "K711", "K713", "K714", "K715",
      "K717", "K72", "K73", "K74", "K760", "K762", "K763", "K764", "K765",
      "K766", "K767", "K768", "K769", "Z944"]),
    ("peptic_ulcer_excl_bleeding", 0,
     ["K257", "K259", "K267", "K269", "K277", "K279", "K287", "K289"]),
    ("aids_hiv", 0,
     ["B20", "B21", "B22", "B24"]),
    ("lymphoma", 9,
     ["C81", "C82", "C83", "C84", "C85", "C88", "C96", "C900", "C902"]),
    ("metastatic_cancer", 12,
     ["C77", "C78", "C79", "C80"]),
    ("solid_tumor_no_metastasis", 4,
     ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08",
      "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17",
      "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
      "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40",
      "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C50", "C51",
      "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60", "C61",
      "C62", "C63", "C64", "C65", "C66", "C67", "C68", "C69", "C70",
      "C71", "C72", "C73", "C74", "C75", "C76", "C97"]),
    ("rheumatoid_arthritis", 0,
     ["L940", "L941", "L943", "M05", "M06", "M08", "M120", "M123",
      "M30", "M31", "M32", "M33", "M34", "M35", "M45", "M461", "M468",
      "M469"]),
    ("coagulopathy", 3,
     ["D65", "D66", "D67", "D68", "D691", "D693", "D694", "D695",
      "D696"]),
    ("obesity", -4,
     ["E66"]),
    ("weight_loss", 6,
     ["E40", "E41", "E42", "E43", "E44", "E45", "E46", "R634", "R64"]),
    ("fluid_electrolyte_disorders", 5,
     ["E222", "E86", "E87"]),
    ("blood_loss_anemia", -2,
     ["D500"]),
    ("deficiency_anemia", -2,
     ["D508", "D509", "D51", "D52", "D53"]),
    ("alcohol_abuse", 0,
     ["F10", "E52", "G621", "I426", "K292", "K700", "K703", "K709",
      "T51", "Z502", "Z714", "Z721"]),
    ("drug_abuse", -7,
     ["F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19", "Z715",
      "Z722"]),
    ("psychoses", 0,
     ["F20", "F22", "F23", "F24", "F25", "F28", "F29", "F302", "F312",
      "F315"]),
    ("depression", -3,
     ["F204", "F313", "F314", "F315", "F32", "F33", "F341", "F412",
      "F432"]),
]


# ─────────────────────── Code normalization + matching ───────────────────────

def _normalize_code(code: str) -> str:
    """Strip whitespace + dots, uppercase. ICD-10 admin tables drop dots."""
    return re.sub(r"[^A-Z0-9]", "", (code or "").upper())


def _condition_match(code: str, prefixes: list[str]) -> bool:
    if not code:
        return False
    for p in prefixes:
        if code.startswith(p):
            return True
    return False


# ─────────────────────── Public API ───────────────────────

async def compute_charlson_elixhauser_index(
    icd10_codes: list[str],
) -> ComorbidityIndices:
    """Compute Charlson + Elixhauser comorbidity scores from ICD-10 codes.

    Args:
        icd10_codes: list of ICD-10 (or ICD-10-CM) codes. Codes may
            contain dots (e.g. "I50.9") or not -- both are normalized.

    Returns:
        ComorbidityIndices with both scores + the conditions present.
    """
    if not isinstance(icd10_codes, list):
        raise ValueError("icd10_codes must be a list of strings.")

    normalized = [_normalize_code(c) for c in icd10_codes
                     if isinstance(c, str)]
    normalized = [c for c in normalized if c]

    charlson_present: dict[str, int] = {}
    elixhauser_present: dict[str, int] = {}
    recognized: set[str] = set()

    for code in normalized:
        for label, weight, prefixes in _CHARLSON:
            if _condition_match(code, prefixes):
                charlson_present.setdefault(label, weight)
                recognized.add(code)
        for label, weight, prefixes in _ELIXHAUSER:
            if _condition_match(code, prefixes):
                elixhauser_present.setdefault(label, weight)
                recognized.add(code)

    # Charlson hierarchy: when "diabetes_with_complications" is present,
    # drop "diabetes_uncomplicated" to avoid double-counting (Quan 2005).
    if "diabetes_with_complications" in charlson_present:
        charlson_present.pop("diabetes_uncomplicated", None)
    if "moderate_severe_liver_disease" in charlson_present:
        charlson_present.pop("mild_liver_disease", None)
    if "metastatic_solid_tumor" in charlson_present:
        charlson_present.pop("solid_tumor_localized", None)

    # Elixhauser hierarchy
    if "metastatic_cancer" in elixhauser_present:
        elixhauser_present.pop("solid_tumor_no_metastasis", None)
    if "diabetes_complicated" in elixhauser_present:
        elixhauser_present.pop("diabetes_uncomplicated", None)
    if "hypertension_complicated" in elixhauser_present:
        elixhauser_present.pop("hypertension_uncomplicated", None)

    charlson_score = max(0, min(40, sum(charlson_present.values())))
    elixhauser_score = max(0, min(40, sum(elixhauser_present.values())))

    return ComorbidityIndices(
        icd10_codes_input=[c for c in icd10_codes if isinstance(c, str)],
        icd10_codes_recognized=sorted(recognized),
        charlson_conditions_present=sorted(charlson_present.keys()),
        charlson_score=charlson_score,
        elixhauser_conditions_present=sorted(elixhauser_present.keys()),
        elixhauser_score=elixhauser_score,
    )


def register(mcp) -> None:
    mcp.tool()(compute_charlson_elixhauser_index)
