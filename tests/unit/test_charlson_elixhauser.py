"""DATA-4 unit tests for ICD-10 -> Charlson + Elixhauser comorbidity scoring."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.charlson_elixhauser import (
    _normalize_code,
    compute_charlson_elixhauser_index,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Code normalization ───────────────────────

def test_normalize_strips_dots():
    assert _normalize_code("I50.9") == "I509"


def test_normalize_uppercases():
    assert _normalize_code("i21") == "I21"


def test_normalize_strips_whitespace():
    assert _normalize_code("  I21  ") == "I21"


def test_normalize_handles_empty():
    assert _normalize_code("") == ""


# ─────────────────────── Validation ───────────────────────

def test_non_list_input_raises():
    with pytest.raises(ValueError, match="must be a list"):
        _run(compute_charlson_elixhauser_index("I21"))  # type: ignore[arg-type]


def test_empty_list_yields_zero_scores():
    r = _run(compute_charlson_elixhauser_index([]))
    assert r.charlson_score == 0
    assert r.elixhauser_score == 0
    assert r.charlson_conditions_present == []


# ─────────────────────── Charlson -- single condition checks ───────────────────────

def test_mi_charlson_weight_1():
    r = _run(compute_charlson_elixhauser_index(["I21.0"]))
    assert "myocardial_infarction" in r.charlson_conditions_present
    assert r.charlson_score == 1


def test_chf_charlson_weight_1():
    r = _run(compute_charlson_elixhauser_index(["I50.9"]))
    assert "congestive_heart_failure" in r.charlson_conditions_present
    assert r.charlson_score == 1


def test_metastatic_cancer_charlson_weight_6():
    r = _run(compute_charlson_elixhauser_index(["C77.0"]))
    assert "metastatic_solid_tumor" in r.charlson_conditions_present
    assert r.charlson_score == 6


def test_hiv_charlson_weight_6():
    r = _run(compute_charlson_elixhauser_index(["B20"]))
    assert "hiv_aids" in r.charlson_conditions_present
    assert r.charlson_score == 6


def test_dementia_charlson_weight_1():
    r = _run(compute_charlson_elixhauser_index(["F03"]))
    assert "dementia" in r.charlson_conditions_present
    assert r.charlson_score == 1


def test_renal_charlson_weight_2():
    r = _run(compute_charlson_elixhauser_index(["N18.5"]))
    assert "renal_disease" in r.charlson_conditions_present
    assert r.charlson_score == 2


def test_diabetes_complicated_charlson_weight_2():
    r = _run(compute_charlson_elixhauser_index(["E11.2"]))
    assert "diabetes_with_complications" in r.charlson_conditions_present
    assert r.charlson_score == 2


# ─────────────────────── Charlson hierarchy ───────────────────────

def test_dm_complicated_drops_uncomplicated():
    """Quan 2005 rule: when complicated diabetes is present, the uncomplicated
    flag is dropped to avoid double-counting."""
    r = _run(compute_charlson_elixhauser_index(["E11.0", "E11.2"]))
    assert "diabetes_with_complications" in r.charlson_conditions_present
    assert "diabetes_uncomplicated" not in r.charlson_conditions_present
    assert r.charlson_score == 2


def test_metastatic_drops_localized():
    r = _run(compute_charlson_elixhauser_index(["C50.9", "C77.0"]))
    assert "metastatic_solid_tumor" in r.charlson_conditions_present
    assert "solid_tumor_localized" not in r.charlson_conditions_present
    assert r.charlson_score == 6


def test_severe_liver_drops_mild_liver():
    r = _run(compute_charlson_elixhauser_index(["K70.9", "K72.1"]))
    assert "moderate_severe_liver_disease" in r.charlson_conditions_present
    assert "mild_liver_disease" not in r.charlson_conditions_present
    assert r.charlson_score == 3


# ─────────────────────── Charlson -- multi-condition sum ───────────────────────

def test_multi_condition_sum_correct():
    """MI (1) + CHF (1) + DM uncomplicated (1) + COPD (1) = 4"""
    r = _run(compute_charlson_elixhauser_index([
        "I21.0", "I50.9", "E11.9", "J44.0",
    ]))
    assert r.charlson_score == 4
    expected = {"myocardial_infarction", "congestive_heart_failure",
                  "diabetes_uncomplicated", "chronic_pulmonary_disease"}
    assert expected <= set(r.charlson_conditions_present)


def test_charlson_score_capped_at_40():
    """Even pathological inputs cap at 40."""
    codes = ["B20", "C77.0", "C78.0", "C79.0", "K72.1",
              "I21.0", "I50.9", "E11.2", "G81", "N18.5"]
    r = _run(compute_charlson_elixhauser_index(codes))
    assert r.charlson_score <= 40


# ─────────────────────── Elixhauser -- single condition ───────────────────────

def test_chf_elixhauser_weight_7():
    r = _run(compute_charlson_elixhauser_index(["I50.9"]))
    assert "congestive_heart_failure" in r.elixhauser_conditions_present
    assert r.elixhauser_score == 7


def test_liver_elixhauser_weight_11():
    r = _run(compute_charlson_elixhauser_index(["K72.1"]))
    assert "liver_disease" in r.elixhauser_conditions_present
    assert r.elixhauser_score == 11


def test_metastatic_elixhauser_weight_12():
    r = _run(compute_charlson_elixhauser_index(["C78.0"]))
    assert "metastatic_cancer" in r.elixhauser_conditions_present
    assert r.elixhauser_score == 12


def test_obesity_negative_weight():
    """Obesity has a NEGATIVE van Walraven weight (-4) -- score is clamped
    to ≥ 0 since the schema enforces ge=0."""
    r = _run(compute_charlson_elixhauser_index(["E66.9"]))
    assert "obesity" in r.elixhauser_conditions_present
    # Negative weight clamped -> 0
    assert r.elixhauser_score == 0


# ─────────────────────── Elixhauser hierarchy ───────────────────────

def test_metastatic_drops_localized_solid_tumor_elixhauser():
    r = _run(compute_charlson_elixhauser_index(["C50.9", "C78.0"]))
    assert "metastatic_cancer" in r.elixhauser_conditions_present
    assert "solid_tumor_no_metastasis" not in \
        r.elixhauser_conditions_present


def test_complicated_diabetes_drops_uncomplicated_elixhauser():
    r = _run(compute_charlson_elixhauser_index(["E11.0", "E11.2"]))
    assert "diabetes_complicated" in r.elixhauser_conditions_present
    assert "diabetes_uncomplicated" not in \
        r.elixhauser_conditions_present


def test_complicated_htn_drops_uncomplicated():
    r = _run(compute_charlson_elixhauser_index(["I10", "I11.0"]))
    assert "hypertension_complicated" in r.elixhauser_conditions_present
    assert "hypertension_uncomplicated" not in \
        r.elixhauser_conditions_present


# ─────────────────────── Output shape ───────────────────────

def test_recognized_codes_are_listed():
    codes = ["I21.0", "I50.9", "Z99.99"]   # Z99.99 not in any table
    r = _run(compute_charlson_elixhauser_index(codes))
    # I21.0 + I50.9 normalized to I210, I509 should appear
    assert "I210" in r.icd10_codes_recognized
    assert "I509" in r.icd10_codes_recognized
    # Unrecognized code does NOT appear
    assert "Z9999" not in r.icd10_codes_recognized


def test_input_codes_preserved_verbatim():
    codes = ["I21.0", "i50.9"]
    r = _run(compute_charlson_elixhauser_index(codes))
    assert r.icd10_codes_input == codes


def test_unknown_codes_yield_zero():
    r = _run(compute_charlson_elixhauser_index(
        ["Z00.00", "R10.84", "Q99.999"]))
    assert r.charlson_score == 0
    assert r.elixhauser_score == 0


def test_invalid_string_inputs_skipped():
    """Non-string entries in the list are ignored without raising."""
    r = _run(compute_charlson_elixhauser_index(
        ["I21.0", None, 12345, "I50.9"]))  # type: ignore[list-item]
    assert "myocardial_infarction" in r.charlson_conditions_present
    assert "congestive_heart_failure" in r.charlson_conditions_present


# ─────────────────────── References + provenance ───────────────────────

def test_method_is_quan_2005():
    r = _run(compute_charlson_elixhauser_index(["I21.0"]))
    assert r.method == "quan_2005_charlson_aahrq_elixhauser"


def test_references_include_quan_and_ahrq():
    r = _run(compute_charlson_elixhauser_index(["I21.0"]))
    refs = " ".join(r.references)
    assert "Quan" in refs
    assert "AHRQ" in refs or "HCUP" in refs


# ─────────────────────── Bundle registration ───────────────────────

def test_charlson_in_data_normalization_bundle():
    from mcp_server.tools import BUNDLES
    assert "data_normalization" in BUNDLES
    assert "compute_charlson_elixhauser_index" in BUNDLES["data_normalization"]
