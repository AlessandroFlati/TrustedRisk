# Bundle: `chart_intelligence` -- Clinical NER + NegEx + Discharge-Summary Structuring

Free-text chart intelligence: clinical NER over discharge summaries,
negation + temporal context (Chapman 2001 NegEx), and a structurer that
converts a free-text discharge summary into a `DecisionCard`-shaped
artifact.

## When to use

- Ingesting a free-text discharge summary or admission note from a
  legacy EHR (no FHIR endpoint).
- Extracting structured concept annotations from a chart for downstream
  rule-based scoring.
- Resolving negation ("no chest pain") and temporal scope ("history of
  CHF" vs "active CHF") so downstream tools don't interpret a denied
  condition as present.

## Typical tool sequence

1. compute_clinical_ner(text) -> list of clinical entities with offsets
   + concept-unique-identifier (CUI)
1. compute_negation_temporal(text, entities) -> per-entity negation +
   temporal-scope flags
1. compute_structure_discharge_summary(text) -> DecisionCard-shaped
   artifact with conditions / medications / followup / vitals

## Tools in this bundle

- `compute_clinical_ner` (CHART-1)
- `compute_negation_temporal` (CHART-2, NegEx)
- `compute_structure_discharge_summary` (CHART-3)

## Determinism

- NER and NegEx are fully deterministic (rule-based + dictionary-based).
- The structurer composes the deterministic NER + NegEx output into a
  schema-conformant artifact; no LLM required.

## Minimum inputs

- Free-text clinical note (admission H&P, discharge summary, consult
  note, progress note)
- Optional: pre-extracted entities (for NegEx-only pass)

## Composed output

- `ClinicalNERReport` (entities with offsets + CUIs)
- `NegationTemporalReport` (per-entity negation + temporal scope flags)
- `StructuredDischargeSummary` (DecisionCard-shaped artifact derived
  from free text)

## Demo scenarios

- v6 showcase: PA-imaging-mri-lumbar exercises chart_intelligence for
  evidence-pack chart excerpts.
- The `scribe` and `pa` specialists both consume this bundle when
  the input is free-text rather than structured FHIR.

## Caveats

- The NER vocabulary is curated, not exhaustive. Drugs and conditions
  outside the curated list may be missed.
- NegEx triggers are English-language; multilingual support is out of
  scope for this release.
- The structurer's DecisionCard-shaped output is intended as a feed for
  downstream tools, not as the final clinical recommendation. The
  agentic loop must still run on the structured artifact.
