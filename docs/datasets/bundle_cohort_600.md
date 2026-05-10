# Dataset Card: Bundle Cohort 600

**Format**: Datasheet for Datasets (Gebru et al. FAccT 2018)
**Generator**: `scripts/generate_synthetic_cohorts.py`
**Last regenerated**: see commit history of `tests/golden/cohorts/*.json`
**License**: MIT

## Motivation

Why was this dataset created? -- To exercise every clinical-vertical
bundle's tool surface end-to-end with realistic + deliberately
adversarial cases. Without this, drift in tool clinical correctness
would only surface at deployment time.

Who funded the creation? -- The TrustedRisk project (independent
research prototype).

## Composition

- **Total instances**: 600 (50 per bundle × 12 clinical bundles).
- **Per-bundle distribution**: skewed toward worrisome / abstain-
  triggering cases (sepsis-biased PEWS, low/moderate/high HEART
  splits, drug-allergy contraindication examples).
- **Format**: JSON -- `tests/golden/cohorts/{bundle_id}.json`. Each file:
  ```json
  {
    "bundle_id": "...",
    "tool": "compute_...",
    "cases": [
      {"name": "...", "input": {...},
       "expected_subset": {...} | "schema_only": true}
    ]
  }
  ```
- **Fields**: tool input -> expected subset of the output (or
  `schema_only` for shape-only validation).
- **Sensitive attributes**: synthetic -- no PHI. Demographics use
  realistic distributions for fairness-audit testing without
  identifying real patients.

## Collection process

Synthetic generator with `random.Random(seed=4242)`. Per-bundle hand-
authored input distributions; outputs cross-validated against the
golden subset assertions in `tests/golden/test_bundle_cohorts.py`.

## Preprocessing / labels

No labels -- the cases assert subset-match on the deterministic tool
output. Schema validity is enforced by Pydantic on every load.

## Uses

- **Recommended**: regression suite to catch tool clinical drift
  before deployment. CI fails on any subset-match miss.
- **Out of scope**: training data for any ML model -- these are
  fixtures, not population samples. Calibration / discrimination
  studies must use Synthea-10k or MIMIC-IV (see other dataset cards).

## Distribution

- Shipped with the source repository under `tests/golden/cohorts/`.
- License: MIT.
- Citation: cite the TrustedRisk repository commit SHA.

## Maintenance

- Maintainer: TrustedRisk maintainers.
- Refresh: regenerate when (a) a bundle's tool surface changes, or
  (b) the golden assertions need expansion. Use the same seed
  (4242) for reproducibility.
- Bug reports: GitHub Issues.
