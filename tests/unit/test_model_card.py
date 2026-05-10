"""Phase 16.F1 - Model Card + Datasheet tests."""

from __future__ import annotations

from a2a_agent.model_card import (
    Datasheet, ModelCard, build_datasheet, build_model_card,
    render_datasheet_md, render_model_card_md,
)


# Model card

def test_model_card_has_nine_sections():
    card = build_model_card()
    assert card.n_sections == 9


def test_model_card_section_titles_match_mitchell_2019():
    card = build_model_card()
    titles = [s.title for s in card.sections]
    expected_keywords = [
        "Model details", "Intended use", "Factors", "Metrics",
        "Evaluation data", "Training data", "Quantitative analyses",
        "Ethical considerations", "Caveats and recommendations",
    ]
    for kw in expected_keywords:
        assert any(kw in t for t in titles), f"missing section: {kw}"


def test_model_card_renders_with_title():
    card = build_model_card()
    md = render_model_card_md(card)
    assert md.startswith("# Model Card:")
    for s in card.sections:
        assert s.title in md


def test_model_card_round_trip_serialises():
    card = build_model_card()
    payload = card.model_dump(mode="json")
    rebuilt = ModelCard.model_validate(payload)
    assert rebuilt.n_sections == card.n_sections


def test_model_card_metrics_section_includes_w1_baseline():
    card = build_model_card()
    metrics = next(s for s in card.sections if "Metrics" in s.title)
    assert "0.0078" in metrics.body_md
    assert "0.590" in metrics.body_md


# Datasheet

def test_datasheet_has_seven_sections():
    ds = build_datasheet()
    assert ds.n_sections == 7


def test_datasheet_section_titles_match_gebru_2021():
    ds = build_datasheet()
    titles = [s.title for s in ds.sections]
    expected_keywords = [
        "Motivation", "Composition", "Collection",
        "Preprocessing", "Uses", "Distribution", "Maintenance",
    ]
    for kw in expected_keywords:
        assert any(kw in t for t in titles), f"missing section: {kw}"


def test_datasheet_renders_with_title():
    ds = build_datasheet()
    md = render_datasheet_md(ds)
    assert md.startswith("# Datasheet:")


def test_datasheet_round_trip_serialises():
    ds = build_datasheet()
    payload = ds.model_dump(mode="json")
    rebuilt = Datasheet.model_validate(payload)
    assert rebuilt.n_sections == ds.n_sections
