"""DATA-1 unit tests for UMLS concept mapping."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.umls_mapper import (
    _normalize_icd10,
    compute_umls_concept_map,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Validation ───────────────────────

def test_invalid_source_vocabulary_raises():
    with pytest.raises(ValueError, match="source_vocabulary"):
        _run(compute_umls_concept_map(["I50.9"], source_vocabulary="MADEUP"))


def test_non_list_codes_raises():
    with pytest.raises(ValueError, match="codes must be a list"):
        _run(compute_umls_concept_map("I50.9"))  # type: ignore[arg-type]


def test_empty_list_returns_zero_mappings():
    r = _run(compute_umls_concept_map([]))
    assert r.n_input == 0
    assert r.n_mapped == 0


def test_non_string_codes_skipped():
    r = _run(compute_umls_concept_map([None, 42, "I50.9"]))
    assert r.n_input == 1


# ─────────────────────── Code normalization ───────────────────────

def test_normalize_strips_dot():
    assert _normalize_icd10("I50.9") == "I509"


def test_normalize_uppercases():
    assert _normalize_icd10("i50.9") == "I509"


# ─────────────────────── ICD-10 -> SNOMED ───────────────────────

def test_chf_icd10_maps_to_snomed():
    r = _run(compute_umls_concept_map(["I50.9"], source_vocabulary="ICD10CM"))
    assert r.n_mapped == 1
    m = r.mappings[0]
    assert m.cui is not None
    assert m.preferred_name and "heart failure" in m.preferred_name.lower()
    assert "84114007" in m.target_codes.get("SNOMEDCT_US", [])


def test_diabetes_t2_maps_to_snomed():
    r = _run(compute_umls_concept_map(["E11.9"], source_vocabulary="ICD10CM"))
    assert r.n_mapped == 1
    snomed_targets = r.mappings[0].target_codes.get("SNOMEDCT_US", [])
    assert any(c in snomed_targets for c in ["44054006", "73211009"])


def test_htn_icd10_maps():
    r = _run(compute_umls_concept_map(["I10"], source_vocabulary="ICD10"))
    assert r.n_mapped == 1
    m = r.mappings[0]
    assert "Hypertensive" in (m.preferred_name or "")


def test_aki_n17_maps_to_snomed():
    r = _run(compute_umls_concept_map(["N17.9"], source_vocabulary="ICD10CM"))
    assert r.n_mapped == 1
    assert "14669001" in r.mappings[0].target_codes.get("SNOMEDCT_US", [])


def test_dotted_and_undotted_icd10_resolve_same_cui():
    """Both 'I50.9' and 'I509' should map to the same CUI."""
    r1 = _run(compute_umls_concept_map(["I50.9"], source_vocabulary="ICD10"))
    r2 = _run(compute_umls_concept_map(["I509"], source_vocabulary="ICD10"))
    assert r1.mappings[0].cui == r2.mappings[0].cui


# ─────────────────────── SNOMED -> ICD-10 ───────────────────────

def test_snomed_to_icd10_chf():
    r = _run(compute_umls_concept_map(["84114007"],
                                              source_vocabulary="SNOMEDCT_US"))
    assert r.n_mapped == 1
    icd_targets = r.mappings[0].target_codes.get("ICD10CM", [])
    assert any(c.startswith("I50") for c in icd_targets)


def test_snomed_diabetes_to_icd():
    r = _run(compute_umls_concept_map(["44054006"],
                                              source_vocabulary="SNOMEDCT_US"))
    assert r.n_mapped == 1
    icd_targets = r.mappings[0].target_codes.get("ICD10CM", [])
    assert any(c.startswith("E11") for c in icd_targets)


# ─────────────────────── RxNorm -> other ───────────────────────

def test_warfarin_rxnorm_maps_to_anticoag_class():
    r = _run(compute_umls_concept_map(["11289"],
                                              source_vocabulary="RXNORM"))
    assert r.n_mapped == 1
    m = r.mappings[0]
    assert m.cui == "C0003280"
    assert "anticoag" in (m.preferred_name or "").lower()


# ─────────────────────── Unmapped paths ───────────────────────

def test_unknown_code_unmapped():
    r = _run(compute_umls_concept_map(["Z99.99"], source_vocabulary="ICD10CM"))
    assert r.n_mapped == 0
    assert r.mappings[0].cui is None


def test_partial_match_for_truncated_icd10():
    """An ICD-10 stem like 'I50' should still resolve (CHF chapter)."""
    r = _run(compute_umls_concept_map(["I50"], source_vocabulary="ICD10"))
    assert r.n_mapped == 1


# ─────────────────────── Source-vocabulary excluded from target_codes ───────────────────────

def test_target_codes_excludes_source_vocabulary():
    """The mapping should not list the source's own codes in target_codes."""
    r = _run(compute_umls_concept_map(["I50.9"], source_vocabulary="ICD10CM"))
    m = r.mappings[0]
    assert "ICD10CM" not in m.target_codes


def test_target_codes_only_lists_present_vocabularies():
    r = _run(compute_umls_concept_map(["11289"],
                                              source_vocabulary="RXNORM"))
    m = r.mappings[0]
    # Drug class C0003280 has SNOMED but no ICD10CM codes -- that vocabulary
    # should be absent (not present-but-empty)
    assert "RXNORM" not in m.target_codes
    assert "SNOMEDCT_US" in m.target_codes


# ─────────────────────── Multi-code cohort ───────────────────────

def test_multi_code_cohort_mapped_independently():
    r = _run(compute_umls_concept_map(
        ["I50.9", "E11.9", "Z99.99"],
        source_vocabulary="ICD10CM",
    ))
    assert r.n_input == 3
    assert r.n_mapped == 2  # last one is unknown


def test_full_cohort_chf_dm_aki_htn():
    r = _run(compute_umls_concept_map(
        ["I50.9", "E11.9", "N17.9", "I10"],
        source_vocabulary="ICD10CM",
    ))
    assert r.n_mapped == 4
    # Each mapping has at least one SNOMED target
    for m in r.mappings:
        assert m.target_codes.get("SNOMEDCT_US")


# ─────────────────────── Method classification ───────────────────────

def test_method_is_fallback_without_umls_rrf():
    r = _run(compute_umls_concept_map(["I50.9"], source_vocabulary="ICD10"))
    assert r.method == "deterministic_fallback"


# ─────────────────────── References ───────────────────────

def test_references_include_umls():
    r = _run(compute_umls_concept_map(["I50.9"], source_vocabulary="ICD10"))
    refs = " ".join(r.references)
    assert "UMLS" in refs


# ─────────────────────── Bundle registration ───────────────────────

def test_umls_in_data_normalization_bundle():
    from mcp_server.tools import BUNDLES
    assert "compute_umls_concept_map" in BUNDLES["data_normalization"]
