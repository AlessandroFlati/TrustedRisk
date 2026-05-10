# Datasheet: TrustedRisk synthetic cohorts (W1 + Synthea-100k + subgroup-audit + prospective-eval)

**Generated**: 2026-05-10 10:35 UTC - **Sections**: 7

Format: Gebru et al. 2021 (CACM) Datasheets for Datasets.

## 1. Motivation

- **For what purpose was the dataset created?** To fit and validate a 30-day readmission risk model using the LACE feature space, plus to drive the fairness audit and prospective evaluation. The pipeline is designed as a research prototype, not for clinical deployment.
- **Funding**: independent research (no grant funding).
- **Conflicts of interest**: none disclosed.

## 2. Composition

- **Instances**: synthetic patient encounters with LACE components (L, A, C, E), simulated demographics (age, sex, race, ethnicity, insurance, language), 30-day readmission binary outcome.
- **Number of instances per cohort**: W1 n=7,880; Synthea-100k n=100,000; Phase-15 subgroup-audit cohort n=100,000; prospective-eval cohort n=10,000.
- **Sample weighting**: the subgroup-audit + prospective cohorts use ground-truth distributions skewed toward US Medicare-age populations (60% age >=65).
- **Sensitive attributes**: race, ethnicity, insurance, language are present and used in evaluation; no real PHI is contained.

## 3. Collection process

- **Acquisition method**: programmatic generation via Synthea (BSD-3) for W1 + Synthea-100k; deterministic seeded sampling for the Phase-15 cohorts (seed=20260430).
- **Sampling strategy**: stratified by LACE bin marginals matching W1; outcomes drawn from fixed calibrated bucket rates {0.072, 0.103, 0.158, 0.234, 0.327}.
- **Time period**: encounters generated 2026-04-24 to 2026-04-30. No real time-stamped patient data.

## 4. Preprocessing / cleaning / labeling

- **Cleaning**: none required - generators produce well-formed LACE features by construction.
- **Filtering**: encounters with LACE > 19 are clipped to 19 (the upper bound of the calibrated bucket map).
- **Software used**: Python stdlib only; no Pandas/NumPy dependency in the cohort generation path.

## 5. Uses

- **Has the dataset been used?**: yes - W1 calibration, subgroup-audit (Phase 15.B1), prospective-eval (Phase 15.C), regulatory-pack fairness section (Phase 14.17 P1).
- **Other tasks the dataset could be used for**: any binary classification benchmark on the LACE feature space; fairness research on US-shaped cohorts; calibration research (the dataset is well-calibrated by construction so it provides a clean baseline).
- **Tasks the dataset should NOT be used for**: real clinical decision-making (it's synthetic); training a classifier intended for a non-US setting (the demographic and outcome multipliers are US-anchored).

## 6. Distribution

- **Distribution channel**: included in the TrustedRisk GitHub repository under `data/` and `docs/fairness/` + `docs/prospective/`.
- **License**: Apache 2.0.
- **Format**: JSON (artefacts) + Python source for the generators.

## 7. Maintenance

- **Who is supporting?**: TrustedRisk team.
- **Update cadence**: re-run on every release of the TrustedRisk codebase; the generators are deterministic so regenerating is free.
- **Erratum / errata**: none currently. Issues are tracked via GitHub.
- **Will the dataset be archived?**: yes - the JSON artefacts are committed to the repository, and the source generators are versioned alongside the code.
