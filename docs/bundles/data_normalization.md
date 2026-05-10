# Bundle: `data_normalization` -- Terminology Crosswalks + Comorbidity Index

Terminology normalisation surfaces required to stitch together FHIR data
from heterogeneous EHRs into the canonical TrustedRisk feature space.
LOINC for observations, RxNorm for medications and drug-drug interactions,
UMLS for cross-vocabulary concept mapping, and the Charlson + Elixhauser
comorbidity indices for risk-adjusted analytics.

## When to use

- Ingesting a FHIR Bundle from an EHR that uses local codes (not LOINC)
  -- normalize first, then feed downstream tools.
- Drug-drug interaction lookup against the NIH RxNav RxNorm-DDI graph.
- UMLS Metathesaurus crosswalk between SNOMED CT, ICD-10, RxNorm,
  LOINC, and CPT.
- Comorbidity-adjusted risk: Charlson (Quan 2005) + Elixhauser
  (van Walraven 2009) derived from the Conditions resource.

## Typical tool sequence

1. compute_normalize_observations(observations) -> LOINC-coded
   observations with units normalised
1. compute_charlson_elixhauser_index(conditions) -> Charlson + Elixhauser
   integers + per-comorbidity flags
1. compute_rxnorm_ddi_lookup(medication_list) -> DDI severity + mechanism
1. compute_umls_concept_map(source_code, source_vocab, target_vocab) ->
   target-vocab code + concept-unique-identifier (CUI)

## Tools in this bundle

- `compute_charlson_elixhauser_index` (DATA-4)
- `compute_normalize_observations` (DATA-3, LOINC)
- `compute_rxnorm_ddi_lookup` (DATA-2, NIH RxNav)
- `compute_umls_concept_map` (DATA-1, UMLS Metathesaurus)

## Determinism

All four tools are fully deterministic. They consume external knowledge
sources (LOINC, RxNorm, UMLS) but the lookup itself is a pure function
of the inputs and the snapshot of the knowledge artifact bundled with
the runtime.

## Minimum inputs

- For Charlson/Elixhauser: `Conditions[]` (FHIR Condition list)
- For LOINC normalisation: `Observations[]` with local codes
- For RxNorm DDI: `Medications[]` (RxNorm or text)
- For UMLS: source code + source vocabulary + target vocabulary

## Composed output

- `CharlsonElixhauserIndex` (integers + per-comorbidity flags)
- `NormalizedObservations[]` (LOINC-coded)
- `DDIReport` (severity tiers + mechanisms)
- `UMLSConceptMap` (target-vocab code + CUI)

## Demo scenarios

- Used as upstream pre-processing on every cross-bundle scenario.
- v5 showcase: scenarios A, F, K consume normalised observations.

## Caveats

- UMLS Metathesaurus license is required for production deployments;
  the development snapshot bundled with TrustedRisk is for evaluation
  only.
- RxNorm DDI graph is a snapshot -- must be re-fetched periodically to
  pick up new interaction reports.
