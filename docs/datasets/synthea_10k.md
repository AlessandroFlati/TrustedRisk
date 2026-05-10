# Dataset Card: Synthea-10k Validation Cohort

**Format**: Datasheet for Datasets (Gebru et al. FAccT 2018)
**Generator**: `scripts/synthea_10k_validation.py`
**Last regenerated**: see `docs/validation/SYNTHEA_10K.md` (header line)
**License**: MIT (synthetic -- no real-patient data)

## Motivation

Why? -- Calibration + subgroup-ECE estimation requires 10k+ patients
to produce stable estimates. Real-EHR cohorts at this scale require
a DUA + IRB and aren't suitable for an open public release.
The Synthea-10k cohort is the calibrated-runtime stress test.

## Composition

- **Size**: 10 000 patients (configurable via `--n`).
- **Schema** (per patient row):
  - `id`: stable synthetic ID.
  - `lace`: total LACE (1-19).
  - `lace_components`: dict of {L, A, C, E}.
  - `p_calibrated`: posterior probability from the W1 calibration.
  - `outcome_30d_readmission`: 0 or 1 sampled from `Bernoulli(p_calibrated)`.
  - `race`, `insurance`, `age_band`, `sex`: realistic demographic
    distributions.
- **Distribution**: LACE marginals match the W1 calibration cohort
  (mean LACE 8.4, SD 4.1, ~ 17 % readmission rate).
- **Sensitive attributes**: synthetic -- no real-patient data.

## Collection process

Pure-deterministic Python generator with `random.Random(seed=42)`.
Demographics drawn from US-population proportions; LACE components
sampled from the W1 cohort marginals.

## Preprocessing / labels

Outcome `outcome_30d_readmission` is sampled from `Bernoulli(p_calibrated)`
where `p_calibrated` is the W1 bucket probability. So the cohort is
trivially calibrated by construction; the harness verifies that the
*runtime* path doesn't drift from the calibrated table.

## Uses

- **Recommended**: regression on overall ECE, AUROC, Brier, subgroup
  parity, calibration table. Re-run on each release tag.
- **Out of scope**: clinical conclusions about real populations.
  Subgroup parity differences in this cohort reflect random sampling
  noise, not real-world disparities.
- **External validation** with real-EHR data is the prospective study
  scoped in `docs/PROSPECTIVE_STUDY_PROTOCOL.md`.

## Distribution

- Generator script + raw JSON dump (~ 2.1 MB) shipped with the repo
  under `docs/validation/synthea_10k_raw.json`.
- License: MIT.
- Citation: TrustedRisk commit SHA + `data/coefficients.json` version.

## Maintenance

- Refresh on calibration bumps + on release tags.
- Versioning: `synthea_<n_thousand>k_raw.json` filename pattern.
- Bug reports: GitHub Issues.
