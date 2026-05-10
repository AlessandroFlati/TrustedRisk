# Dataset Card: MIMIC-IV Demo 2.2 (cross-fold)

**Format**: Datasheet for Datasets (Gebru et al. FAccT 2018)
**Source**: PhysioNet -- https://physionet.org/content/mimic-iv-demo/2.2/
**License**: PhysioNet Open Data License v1.0 (no DUA required)
**Local mirror**: `docs/validation/mimic_iv/` (16 MB ZIP)

## Motivation

Why? -- External-cohort validation against a real EHR data source
(Pillar-2 of the FDA SaMD Clinical Evaluation framework). The demo
release is the only fully open-access subset of MIMIC-IV that doesn't
require the standard credentialed-DUA process.

## Composition

- **Size**: 100 patients / 275 admissions / ~ 50 K observations,
  per the PhysioNet demo 2.2 release.
- **Origin**: Beth Israel Deaconess Medical Center, Boston (US).
- **Period**: 2008-2019.
- **Schema**: standard MIMIC-IV `hosp/*.csv.gz`.
- **Sensitive attributes**: anonymised at the source per the
  PhysioNet de-identification pipeline. Date shifting applied; no
  re-identifiable PHI.

## Collection process

Routine clinical care at Beth Israel Deaconess. PhysioNet maintainers
filtered + de-identified the demo subset.

## Preprocessing / labels

`scripts/mimic_iv_to_lace.py` parses the demo into LACE features
(L = LOS days, A = acute admission flag, C = Charlson points,
E = ED visits in past 6 mo). 30-day readmission is derived from the
`hadm_id` adjacency table.

## Uses

- **Recommended**: external calibration (`scripts/external_validation.py`).
  Outputs ECE 0.082, AUROC 0.555, Brier 0.16 at the W1 cutoff --
  documented in `docs/validation/VALIDATION.md`.
- **Out of scope**: training (the demo is too small + single-site to
  generalise); pediatric, obstetric, or non-US generalisation
  claims.
- **Limitation**: small n (100 patients) inflates the credible-
  interval width on every per-subgroup estimate; intersectional
  fairness claims should NOT be made from this card alone.

## Distribution

- PhysioNet ZIP (16 MB): `https://physionet.org/content/mimic-iv-demo/get-zip/2.2/`.
- Local mirror in `docs/validation/mimic_iv/` for reproducibility.
- License: PhysioNet Open Data License v1.0.
- Citation: see PhysioNet citation guidance for MIMIC-IV demo 2.2.

## Maintenance

- Refresh when PhysioNet releases a new demo version.
- The PhysioNet account `aleksandros` (per repo records) maintains
  the local mirror.
- Bug reports: PhysioNet community + TrustedRisk GitHub Issues.
