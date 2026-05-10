"""DATA-3 unit tests for LOINC observation normalization."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.loinc_normalizer import (
    _system_normalized,
    compute_normalize_observations,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Validation ───────────────────────

def test_non_list_raises():
    with pytest.raises(ValueError, match="must be a list"):
        _run(compute_normalize_observations("nope"))  # type: ignore[arg-type]


def test_empty_list_yields_zero_counts():
    r = _run(compute_normalize_observations([]))
    assert r.n_input == 0
    assert r.n_normalized == 0
    assert r.observations == []


def test_non_dict_entries_skipped():
    r = _run(compute_normalize_observations(["nope", 42, None,
                                                    {"code": "K"}]))
    assert r.n_input == 1


# ─────────────────────── System normalization ───────────────────────

def test_system_loinc_url_normalized():
    assert _system_normalized("http://loinc.org") == "loinc"
    assert _system_normalized("https://LOINC.ORG") == "loinc"


def test_system_snomed_normalized():
    assert _system_normalized("http://snomed.info/sct") == "snomed"


def test_system_unknown_url_treated_as_local():
    assert _system_normalized("urn:oid:1.2.3.4") == "local"


def test_system_empty_treated_as_local():
    assert _system_normalized("") == "local"


# ─────────────────────── Path 1 -- already canonical LOINC ───────────────────────

def test_already_loinc_passes_through():
    r = _run(compute_normalize_observations([
        {"code": "2823-3", "system": "http://loinc.org"},
    ]))
    assert r.n_normalized == 1
    obs = r.observations[0]
    assert obs.normalized_loinc_code == "2823-3"
    assert obs.short_name == "K"
    assert obs.canonical_name.startswith("Potassium")


def test_loinc_unknown_code_unmapped():
    """A LOINC code not in our table -> unmapped (we don't fabricate names)."""
    r = _run(compute_normalize_observations([
        {"code": "99999-9", "system": "http://loinc.org"},
    ]))
    assert r.observations[0].normalized_loinc_code is None
    assert r.n_unmapped == 1


# ─────────────────────── Path 2 -- local-code synonym ───────────────────────

def test_local_potassium_mapped():
    r = _run(compute_normalize_observations([
        {"code": "K", "system": "local"},
    ]))
    assert r.observations[0].normalized_loinc_code == "2823-3"
    assert r.observations[0].short_name == "K"


def test_local_creatinine_mapped():
    r = _run(compute_normalize_observations([
        {"code": "Cr", "system": "local"},
    ]))
    assert r.observations[0].normalized_loinc_code == "2160-0"


def test_epic_local_vital_signs_mapped():
    r = _run(compute_normalize_observations([
        {"code": "HR", "system": "epic"},
        {"code": "BPS", "system": "epic"},
        {"code": "SPO2", "system": "epic"},
    ]))
    codes = [o.normalized_loinc_code for o in r.observations]
    assert codes == ["8867-4", "8480-6", "59408-5"]


def test_cerner_local_codes_mapped():
    r = _run(compute_normalize_observations([
        {"code": "GLU", "system": "cerner"},
        {"code": "BUN", "system": "cerner"},
    ]))
    codes = {o.normalized_loinc_code for o in r.observations}
    assert codes == {"2345-7", "3094-0"}


def test_case_insensitive_local_match():
    """Local codes are matched case-insensitively (typical EHR-emitted codes
    can be uppercase)."""
    r = _run(compute_normalize_observations([
        {"code": "k", "system": "local"},
        {"code": "POTASSIUM", "system": "local"},  # name fallback
    ]))
    # K -> upper match in synonyms
    assert r.observations[0].normalized_loinc_code == "2823-3"


# ─────────────────────── Path 3 -- name synonym fallback ───────────────────────

def test_name_synonym_fallback():
    r = _run(compute_normalize_observations([
        {"name": "potassium"},
        {"name": "Hemoglobin A1c"},
        {"name": "Sodium"},
    ]))
    codes = {o.normalized_loinc_code for o in r.observations}
    assert codes == {"2823-3", "4548-4", "2951-2"}


def test_name_synonym_handles_aliases():
    r = _run(compute_normalize_observations([
        {"name": "BNP"},
        {"name": "INR"},
        {"name": "Troponin"},
    ]))
    codes = {o.normalized_loinc_code for o in r.observations}
    assert "30934-4" in codes  # BNP
    assert "6301-6" in codes   # INR
    assert "10839-9" in codes  # Troponin I


def test_name_canonical_full_string_match():
    """Full canonical names (case-insensitive) should also resolve."""
    r = _run(compute_normalize_observations([
        {"name": "Hemoglobin [Mass/volume] in Blood"},
    ]))
    assert r.observations[0].normalized_loinc_code == "718-7"


# ─────────────────────── Path 4 -- unmapped ───────────────────────

def test_truly_unknown_code_unmapped():
    r = _run(compute_normalize_observations([
        {"code": "Z99", "system": "local", "name": "alien_lab"},
    ]))
    assert r.observations[0].normalized_loinc_code is None
    assert r.observations[0].canonical_name is None
    assert r.n_unmapped == 1


def test_obs_without_any_identifier_unmapped():
    r = _run(compute_normalize_observations([
        {"value": 1.5, "unit": "mmol/L"},
    ]))
    assert r.observations[0].normalized_loinc_code is None


# ─────────────────────── Output shape ───────────────────────

def test_unit_canonical_propagated():
    r = _run(compute_normalize_observations([
        {"code": "2823-3", "system": "http://loinc.org"},
    ]))
    assert r.observations[0].unit_canonical == "mmol/L"


def test_component_class_propagated():
    r = _run(compute_normalize_observations([
        {"code": "K", "system": "local"},
    ]))
    assert r.observations[0].component_class == "electrolyte"


def test_n_input_n_normalized_n_unmapped_consistent():
    r = _run(compute_normalize_observations([
        {"code": "K", "system": "local"},      # mapped
        {"code": "Cr", "system": "local"},     # mapped
        {"code": "Z99", "system": "local"},    # unmapped
    ]))
    assert r.n_input == 3
    assert r.n_normalized == 2
    assert r.n_unmapped == 1


def test_method_is_loinc_table_when_at_least_one_mapped():
    r = _run(compute_normalize_observations([
        {"code": "K", "system": "local"}]))
    assert r.method == "loinc_table"


def test_method_falls_back_when_nothing_mapped():
    r = _run(compute_normalize_observations([
        {"code": "Z99", "system": "local"}]))
    assert r.method == "deterministic_fallback"


# ─────────────────────── Mixed input cohort ───────────────────────

def test_mixed_input_handled_correctly():
    """A realistic multi-source observation set: LOINC codes + Epic vital
    signs + Cerner labs + free-text names -- all should resolve where the
    table covers them."""
    cohort = [
        {"code": "2823-3", "system": "http://loinc.org"},  # LOINC
        {"code": "HR", "system": "epic"},                    # Epic local
        {"code": "GLU", "system": "cerner"},                 # Cerner
        {"name": "Lactate"},                                  # name match
        {"code": "Z99", "system": "local"},                  # unmapped
    ]
    r = _run(compute_normalize_observations(cohort))
    assert r.n_normalized == 4
    assert r.n_unmapped == 1


# ─────────────────────── References ───────────────────────

def test_references_include_loinc():
    r = _run(compute_normalize_observations([{"code": "K", "system": "local"}]))
    assert any("LOINC" in ref for ref in r.references)


# ─────────────────────── Bundle registration ───────────────────────

def test_loinc_in_data_normalization_bundle():
    from mcp_server.tools import BUNDLES
    assert "compute_normalize_observations" in BUNDLES["data_normalization"]
