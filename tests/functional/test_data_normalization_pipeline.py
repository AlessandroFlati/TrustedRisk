"""Functional tests for the data normalization pipeline.

Chains: UMLS ↔ ICD10 ↔ SNOMED ↔ RxNorm + LOINC normalization +
Charlson/Elixhauser scoring + RxNorm DDI. Validates that the agent can
normalize EHR data from any vocabulary into a canonical form before
calling the clinical scoring tools.
"""
from __future__ import annotations

import asyncio


# ─────────────────────── UMLS concept mapping ───────────────────────

def test_umls_chf_icd10cm_to_snomed():
    from mcp_server.tools.umls_mapper import compute_umls_concept_map
    r = asyncio.run(compute_umls_concept_map(
        ["I50.9"], source_vocabulary="ICD10CM"))
    assert r.n_mapped == 1
    snomed = r.mappings[0].target_codes.get("SNOMEDCT_US", [])
    assert "84114007" in snomed


def test_umls_round_trip_icd10_snomed_icd10():
    """ICD10 -> SNOMED -> ICD10 should return overlapping codes."""
    from mcp_server.tools.umls_mapper import compute_umls_concept_map
    forward = asyncio.run(compute_umls_concept_map(
        ["I50.9"], source_vocabulary="ICD10CM"))
    snomed = forward.mappings[0].target_codes.get("SNOMEDCT_US", [])
    assert snomed
    backward = asyncio.run(compute_umls_concept_map(
        [snomed[0]], source_vocabulary="SNOMEDCT_US"))
    icd_back = backward.mappings[0].target_codes.get("ICD10CM", [])
    # The original I50.9 stem should appear among the back-mapped codes
    assert any(c.startswith("I50") for c in icd_back)


def test_umls_unknown_code_falls_back_gracefully():
    from mcp_server.tools.umls_mapper import compute_umls_concept_map
    r = asyncio.run(compute_umls_concept_map(
        ["Z99.99"], source_vocabulary="ICD10CM"))
    assert r.mappings[0].cui is None


# ─────────────────────── LOINC normalization ───────────────────────

def test_loinc_normalize_local_code_table():
    from mcp_server.tools.loinc_normalizer import compute_normalize_observations
    r = asyncio.run(compute_normalize_observations([
        {"code": "K", "system": "local"},
        {"code": "Hgb", "system": "local"},
        {"code": "INR", "system": "local"},
    ]))
    codes = [o.normalized_loinc_code for o in r.observations]
    assert "2823-3" in codes   # K
    assert "718-7" in codes    # Hgb
    assert "6301-6" in codes   # INR


def test_loinc_normalize_epic_vital_signs():
    from mcp_server.tools.loinc_normalizer import compute_normalize_observations
    r = asyncio.run(compute_normalize_observations([
        {"code": "HR", "system": "epic"},
        {"code": "BPS", "system": "epic"},
    ]))
    codes = {o.normalized_loinc_code for o in r.observations}
    assert "8867-4" in codes
    assert "8480-6" in codes


def test_loinc_normalize_name_fallback():
    from mcp_server.tools.loinc_normalizer import compute_normalize_observations
    r = asyncio.run(compute_normalize_observations([
        {"name": "Hemoglobin A1c"},
        {"name": "Sodium"},
        {"name": "Creatinine"},
    ]))
    codes = {o.normalized_loinc_code for o in r.observations}
    assert {"4548-4", "2951-2", "2160-0"} <= codes


# ─────────────────────── Charlson/Elixhauser ───────────────────────

def test_charlson_chf_dm_htn_aki_cohort():
    from mcp_server.tools.charlson_elixhauser import (
        compute_charlson_elixhauser_index,
    )
    r = asyncio.run(compute_charlson_elixhauser_index([
        "I50.9", "E11.9", "I10", "N17.9",
    ]))
    # CHF (1) + DM uncomplicated (1) + AKI N17 mapped via Elixhauser only
    # so on Charlson the score is at least 2
    assert r.charlson_score >= 2
    assert "congestive_heart_failure" in r.charlson_conditions_present
    assert "diabetes_uncomplicated" in r.charlson_conditions_present


def test_charlson_diabetes_complicated_drops_uncomplicated():
    from mcp_server.tools.charlson_elixhauser import (
        compute_charlson_elixhauser_index,
    )
    r = asyncio.run(compute_charlson_elixhauser_index([
        "E11.0", "E11.2", "E11.9",
    ]))
    assert "diabetes_with_complications" in r.charlson_conditions_present
    assert "diabetes_uncomplicated" not in r.charlson_conditions_present


def test_charlson_metastatic_cancer_dominates_localized():
    from mcp_server.tools.charlson_elixhauser import (
        compute_charlson_elixhauser_index,
    )
    r = asyncio.run(compute_charlson_elixhauser_index([
        "C50.9", "C77.0", "C78.0",
    ]))
    assert "metastatic_solid_tumor" in r.charlson_conditions_present
    assert "solid_tumor_localized" not in r.charlson_conditions_present


def test_elixhauser_negative_weight_clamped_to_zero():
    """Obesity has negative van Walraven weight; score clamps at 0."""
    from mcp_server.tools.charlson_elixhauser import (
        compute_charlson_elixhauser_index,
    )
    r = asyncio.run(compute_charlson_elixhauser_index(["E66.9"]))
    assert r.elixhauser_score == 0


# ─────────────────────── RxNorm DDI lookup ───────────────────────

def test_rxnorm_warfarin_aspirin_high_severity():
    from mcp_server.tools.rxnorm_ddi import compute_rxnorm_ddi_lookup
    r = asyncio.run(compute_rxnorm_ddi_lookup(["warfarin", "aspirin"]))
    assert r.n_interactions == 1
    assert r.interactions[0].severity == "high"


def test_rxnorm_full_chf_cohort_ddi_pairs():
    """Realistic 6-drug CHF discharge has multiple expected DDIs."""
    from mcp_server.tools.rxnorm_ddi import compute_rxnorm_ddi_lookup
    r = asyncio.run(compute_rxnorm_ddi_lookup([
        "warfarin", "aspirin", "lisinopril", "spironolactone",
        "amiodarone", "simvastatin",
    ]))
    pairs = {(i.drug_a, i.drug_b) for i in r.interactions}
    assert ("warfarin", "aspirin") in pairs
    assert ("warfarin", "amiodarone") in pairs
    assert ("lisinopril", "spironolactone") in pairs


def test_rxnorm_clean_combination_no_ddis():
    from mcp_server.tools.rxnorm_ddi import compute_rxnorm_ddi_lookup
    r = asyncio.run(compute_rxnorm_ddi_lookup([
        "acetaminophen", "metformin", "pantoprazole"]))
    assert r.n_interactions == 0


# ─────────────────────── End-to-end normalization chain ───────────────────────

def test_full_normalization_chain_for_chf_cohort():
    """Run the 4 normalization tools over a realistic patient and confirm
    they're internally consistent + produce well-formed reports."""
    from mcp_server.tools.charlson_elixhauser import (
        compute_charlson_elixhauser_index,
    )
    from mcp_server.tools.loinc_normalizer import compute_normalize_observations
    from mcp_server.tools.rxnorm_ddi import compute_rxnorm_ddi_lookup
    from mcp_server.tools.umls_mapper import compute_umls_concept_map

    icds = ["I50.9", "E11.9", "I10", "N17.9"]
    meds = ["warfarin", "lisinopril", "spironolactone"]
    obs = [{"code": "K", "system": "local"},
              {"code": "Hgb", "system": "local"},
              {"name": "Creatinine"}]

    cci = asyncio.run(compute_charlson_elixhauser_index(icds))
    loinc = asyncio.run(compute_normalize_observations(obs))
    ddi = asyncio.run(compute_rxnorm_ddi_lookup(meds))
    umls = asyncio.run(compute_umls_concept_map(icds,
                                                       source_vocabulary="ICD10CM"))

    assert cci.charlson_score >= 2
    assert loinc.n_normalized == 3
    assert ddi.n_interactions >= 1   # ACE-I + MRA = hyperkalemia DDI
    assert umls.n_mapped >= 3   # CHF, DM, HTN known
