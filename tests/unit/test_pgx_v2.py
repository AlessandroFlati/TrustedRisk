"""Phase 12.4 D1 -- CPIC PGx v2 expansion (25->79 entries, 52 drugs)."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.pgx import _CPIC_TABLE, compute_pgx_dose_adjustment


def _run(coro):
    return asyncio.run(coro)


def _ask(drug: str, gene: str, phenotype: str, diplotype: str = ""):
    return _run(compute_pgx_dose_adjustment(
        medications=[drug],
        genotypes=[{
            "gene": gene, "phenotype": phenotype,
            "diplotype": diplotype or phenotype,
        }],
    ))


# ─────────────────────────────────────────────────────────────────────
# Table shape
# ─────────────────────────────────────────────────────────────────────

def test_table_size_at_least_70_entries():
    assert len(_CPIC_TABLE) >= 70


def test_table_covers_at_least_50_distinct_drugs():
    drugs = {e["drug"] for e in _CPIC_TABLE}
    assert len(drugs) >= 50


def test_table_covers_at_least_15_distinct_genes():
    genes = {e["gene"] for e in _CPIC_TABLE}
    assert len(genes) >= 15


def test_table_includes_new_high_value_drugs():
    drugs = {e["drug"] for e in _CPIC_TABLE}
    for d in (
        "tacrolimus", "tamoxifen", "phenytoin", "celecoxib",
        "escitalopram", "citalopram", "amitriptyline", "atorvastatin",
        "allopurinol", "atazanavir", "gentamicin", "ivacaftor",
        "valproate", "dapsone", "atomoxetine",
    ):
        assert d in drugs, f"Missing drug {d!r}"


def test_table_includes_new_genes():
    genes = {e["gene"] for e in _CPIC_TABLE}
    for g in (
        "CYP3A5", "MT-RNR1", "CFTR", "POLG", "HLA-B*5801",
        "HLA-A*3101", "OPRM1",
    ):
        assert g in genes, f"Missing gene {g!r}"


# ─────────────────────────────────────────────────────────────────────
# Functional smokes -- go through the report's `adjustments` list
# ─────────────────────────────────────────────────────────────────────

def test_cyp3a5_intermediate_increases_tacrolimus_dose():
    out = _ask("tacrolimus", "CYP3A5", "intermediate_metabolizer", "*1/*3")
    assert out.n_adjustments >= 1
    adj = next(a for a in out.adjustments if a.drug == "tacrolimus")
    assert adj.recommendation == "increase_dose"


def test_hla_b_5801_blocks_allopurinol():
    out = _ask("allopurinol", "HLA-B*5801", "positive")
    assert out.n_adjustments >= 1
    adj = next(a for a in out.adjustments if a.drug == "allopurinol")
    assert adj.recommendation == "avoid_drug"


def test_polg_blocks_valproate():
    out = _ask("valproate", "POLG", "deficient")
    adj = next(a for a in out.adjustments if a.drug == "valproate")
    assert adj.recommendation == "avoid_drug"


def test_mt_rnr1_blocks_aminoglycoside():
    """All three aminoglycosides (gentamicin, amikacin, tobramycin)
    should be flagged as alternative-required for MT-RNR1 m.1555A>G."""
    for drug in ("gentamicin", "amikacin", "tobramycin"):
        out = _ask(drug, "MT-RNR1", "m.1555A>G")
        assert out.n_adjustments >= 1, f"{drug}: no adjustments returned"
        adj = next(a for a in out.adjustments if a.drug == drug)
        assert adj.recommendation == "alternative_drug_strongly_recommended", (
            f"{drug}: got {adj.recommendation}"
        )


def test_cyp2c9_pm_decreases_phenytoin():
    out = _ask("phenytoin", "CYP2C9", "poor_metabolizer", "*3/*3")
    adj = next(a for a in out.adjustments if a.drug == "phenytoin")
    assert adj.recommendation == "decrease_dose"


def test_cyp2c19_pm_decreases_escitalopram():
    out = _ask("escitalopram", "CYP2C19", "poor_metabolizer", "*2/*2")
    adj = next(a for a in out.adjustments if a.drug == "escitalopram")
    assert adj.recommendation == "decrease_dose"


# ─────────────────────────────────────────────────────────────────────
# Schema invariants
# ─────────────────────────────────────────────────────────────────────

def test_every_entry_has_required_fields():
    required = {"gene", "phenotype", "drug", "rec",
                    "dose_pct", "level", "rationale", "monitoring"}
    for e in _CPIC_TABLE:
        missing = required - set(e.keys())
        assert not missing, f"Entry {e}: missing fields {missing}"


def test_every_entry_has_valid_level():
    for e in _CPIC_TABLE:
        assert e["level"] in ("A", "B", "C"), (
            f"Entry {e!r}: bad level {e['level']!r}"
        )


def test_avoid_drug_entries_have_zero_dose_pct():
    """Sanity: an `avoid_drug` recommendation should never modulate dose."""
    for e in _CPIC_TABLE:
        if e["rec"] == "avoid_drug":
            assert e["dose_pct"] == 0, (
                f"Entry {e!r}: avoid_drug with dose_pct {e['dose_pct']}"
            )
