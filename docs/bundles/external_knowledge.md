# Bundle: `external_knowledge` -- PubMed + ClinicalTrials.gov + Pricing + RePORTER

External-knowledge retrieval surfaces. PubMed (Entrez API) for primary
literature, ClinicalTrials.gov v2 API for matching active trials to a
patient phenotype, drug pricing for the cost-of-care arm, and NIH
RePORTER v2 API for ongoing federally-funded research relevant to the
patient's condition.

## When to use

- Patient + family question: "is there a trial my mother could enroll
  in?"
- Clinician evidence pull: "what's the latest on apixaban dose
  adjustment for AKI patients?"
- Cost-of-care surface: "what's the typical out-of-pocket cost for
  this regimen?"
- Research-context surface: "what NIH-funded studies are running on
  this condition right now?"

## Typical tool sequence

1. compute_pubmed_search(query, filters) -> ranked PubMed citations
1. compute_clinical_trials_matcher(patient_id, condition) -> matched
   open trials with eligibility-criteria delta
1. compute_drug_pricing(rxnorm_codes, payer) -> unit prices + 30-day
   estimate
1. compute_nih_reporter_search(condition, agency_filter) -> active
   federally-funded studies

## Tools in this bundle

- `compute_pubmed_search` (KNOWLEDGE-1, Entrez API)
- `compute_clinical_trials_matcher` (KNOWLEDGE-2, CT.gov v2)
- `compute_drug_pricing` (KNOWLEDGE-3)
- `compute_nih_reporter_search` (KNOWLEDGE-4, RePORTER v2)

## Determinism floor

- PubMed retrieval is fully deterministic given a fixed query.
- The trials matcher does NOT use an LLM to evaluate eligibility -- it
  applies a deterministic eligibility-criteria delta over the patient's
  structured FHIR snapshot.
- Drug pricing is a lookup against a snapshot of the source pricing
  table.
- RePORTER search is a fully deterministic API query.

## Minimum inputs

- For PubMed: `query` text, optional MeSH filters
- For trials matcher: `patient_id` + condition + structured features
- For drug pricing: RxNorm codes + payer
- For RePORTER: condition / agency filter

## Composed output

- `PubMedSearchResult` (ranked citations + abstract excerpts)
- `ClinicalTrialsMatch[]` (matched trials + eligibility delta)
- `DrugPricingReport` (unit prices + 30-day estimate)
- `NIHRePORTERSearchResult[]` (active funded studies)

## Demo scenarios

- v6 showcase: PA-imaging-mri-lumbar uses external knowledge for
  literature support of the medical-necessity argument.
- The `evidence` specialist (port 8772) is the natural home for
  this bundle.

## Caveats

- Network calls are required for the live pubmed / CT.gov / RePORTER
  paths. Operators can pin a local mirror for air-gapped deployments.
- Drug pricing data is volatile and US-centric. Re-fit before clinical
  use.
- The eligibility-criteria delta is conservative -- it surfaces a trial
  as "potentially eligible" rather than "definitely eligible". Final
  determination requires the trial team's review.
