"""Phase 7.2 -- pharmacogenomic DSS tools (PGX-1/2/3)."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.pgx import (
    compute_pgx_dose_adjustment,
    compute_pgx_drug_alternatives,
    compute_pgx_eligibility_check,
)
from shared.schemas import PGxGenotype


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Dose adjustment ───────────────────────


def test_warfarin_dose_decreased_for_cyp2c9_pm():
    out = _run(compute_pgx_dose_adjustment(
        medications=["warfarin 5 mg"],
        genotypes=[
            PGxGenotype(gene="CYP2C9", phenotype="poor_metabolizer",
                          diplotype="*3/*3"),
        ],
    ))
    assert out.n_adjustments >= 1
    warf = next(a for a in out.adjustments if "warfarin" in a.drug.lower())
    assert warf.recommendation == "decrease_dose"
    assert warf.dose_modifier_pct == -50
    assert warf.cpic_evidence_level == "A"


def test_codeine_avoided_for_cyp2d6_um():
    out = _run(compute_pgx_dose_adjustment(
        medications=["codeine 30 mg PO PRN"],
        genotypes=[PGxGenotype(gene="CYP2D6",
                                  phenotype="ultra_rapid_metabolizer")],
    ))
    cod = next(a for a in out.adjustments if "codeine" in a.drug.lower())
    assert cod.recommendation == "avoid_drug"
    assert any("respiratory depression" in r.lower()
                  or "non-codeine" in r.lower()
                  for r in [cod.rationale])


def test_clopidogrel_alternative_for_cyp2c19_pm():
    out = _run(compute_pgx_dose_adjustment(
        medications=["clopidogrel 75 mg"],
        genotypes=[PGxGenotype(gene="CYP2C19",
                                  phenotype="poor_metabolizer")],
    ))
    clop = next(a for a in out.adjustments if "clopidogrel" in a.drug.lower())
    assert clop.recommendation == "alternative_drug_strongly_recommended"
    assert clop.cpic_evidence_level == "A"


def test_simvastatin_alternative_for_slco1b1_pm():
    out = _run(compute_pgx_dose_adjustment(
        medications=["simvastatin 40 mg"],
        genotypes=[PGxGenotype(gene="SLCO1B1",
                                  phenotype="poor_metabolizer")],
    ))
    sim = next(a for a in out.adjustments if "simvastatin" in a.drug.lower())
    assert sim.recommendation == "avoid_drug"


def test_dose_adjustment_empty_meds_abstains():
    out = _run(compute_pgx_dose_adjustment(
        medications=[],
        genotypes=[PGxGenotype(gene="CYP2D6", phenotype="poor_metabolizer")],
    ))
    assert out.abstain_recommended is True


def test_dose_adjustment_no_genotypes_abstains():
    out = _run(compute_pgx_dose_adjustment(
        medications=["warfarin"], genotypes=[],
    ))
    assert out.abstain_recommended is True


def test_dose_adjustment_no_match_returns_empty_adjustments():
    """Genotypes don't match any drug -> empty list, no abstain."""
    out = _run(compute_pgx_dose_adjustment(
        medications=["lisinopril 10 mg"],
        genotypes=[PGxGenotype(gene="CYP2D6", phenotype="poor_metabolizer")],
    ))
    assert out.adjustments == []
    assert out.abstain_recommended is False   # not actionable but not an error


def test_genotype_input_accepts_dict_form():
    out = _run(compute_pgx_dose_adjustment(
        medications=["warfarin"],
        genotypes=[{
            "gene": "CYP2C9", "phenotype": "intermediate_metabolizer",
            "diplotype": "*1/*2",
        }],
    ))
    assert out.n_adjustments >= 1


# ─────────────────────── Drug alternatives ───────────────────────


def test_alternatives_for_blocked_clopidogrel():
    out = _run(compute_pgx_drug_alternatives(
        requested_drug="clopidogrel",
        genotypes=[PGxGenotype(gene="CYP2C19",
                                  phenotype="poor_metabolizer")],
    ))
    assert out.n_alternatives >= 2
    drugs = {a.drug for a in out.alternatives}
    assert "prasugrel" in drugs
    assert "ticagrelor" in drugs


def test_alternatives_for_abacavir_with_hla():
    out = _run(compute_pgx_drug_alternatives(
        requested_drug="abacavir",
        genotypes=[PGxGenotype(gene="HLA-B*5701", phenotype="positive")],
    ))
    assert out.n_alternatives >= 1
    assert out.blocking_genotypes


def test_alternatives_no_blocking_genotype_returns_empty():
    out = _run(compute_pgx_drug_alternatives(
        requested_drug="clopidogrel",
        genotypes=[PGxGenotype(gene="CYP2C19",
                                  phenotype="normal_metabolizer")],
    ))
    assert out.blocking_genotypes == []
    assert out.alternatives == []


# ─────────────────────── Eligibility check ───────────────────────


def test_eligibility_high_when_drug_in_cpic_a():
    out = _run(compute_pgx_eligibility_check(
        requested_test="CYP2C19",
        medications_in_consideration=["clopidogrel 75 mg"],
    ))
    assert out.is_eligible is True
    assert out.expected_clinical_actionability == "high"


def test_eligibility_low_when_drug_not_in_consideration():
    out = _run(compute_pgx_eligibility_check(
        requested_test="CYP2C19",
        medications_in_consideration=["lisinopril"],
    ))
    assert out.is_eligible is False
    assert out.expected_clinical_actionability == "low"


def test_eligibility_experimental_for_unknown_test():
    out = _run(compute_pgx_eligibility_check(
        requested_test="MTHFR",
        medications_in_consideration=["warfarin"],
    ))
    assert out.is_eligible is False
    assert out.expected_clinical_actionability == "experimental"


def test_eligibility_lists_cpic_supported_drugs():
    out = _run(compute_pgx_eligibility_check(
        requested_test="DPYD",
    ))
    assert "fluorouracil" in out.cpic_supported_drugs
    assert "capecitabine" in out.cpic_supported_drugs


# ─────────────────────── Bundles + scopes wiring ───────────────────────


def test_pgx_bundle_present():
    from mcp_server.tools import BUNDLES
    assert "pharmacogenomics" in BUNDLES
    bundle = BUNDLES["pharmacogenomics"]
    assert "compute_pgx_dose_adjustment" in bundle


def test_pgx_scopes_declared():
    from mcp_server.scopes import BUNDLE_SCOPES
    assert "pharmacogenomics" in BUNDLE_SCOPES
    scopes = BUNDLE_SCOPES["pharmacogenomics"]
    assert "patient/MolecularSequence.rs" in scopes
