# Model Card: TrustedRisk readmission risk + decision pipeline

**Generated**: 2026-05-10 10:35 UTC - **Version**: 1.0.0 - **Sections**: 9

Format: Mitchell et al. 2019 (FAT*) Model Cards for Model Reporting.

## 1. Model details

- **Person/organization developing the model**: TrustedRisk team.
- **Model date**: 2026-04-30 (W1 coefficients promoted 2026-04-24).
- **Model type**: Binary classifier on the LACE feature space (L=length-of-stay, A=acuity, C=Charlson, E=ED visits) + 5-bin Beta-Binomial posterior. Combined with a 4-critic ensemble + 3-agent debate that downgrades or abstains.
- **Information about training algorithms**: per-bin Beta-Binomial conjugate update with weakly-informative prior alpha=2, beta=18.
- **License**: Apache 2.0 for the codebase; coefficients produced by an internal calibration workflow (not redistributed).
- **Where to send questions or comments**: GitHub Issues.

## 2. Intended use

- **Primary intended uses**: assistive decision-support at inpatient discharge planning. The model emits a calibrated 30-day readmission probability + recommended action; a clinician reviews before any care decision is enacted.
- **Primary intended users**: licensed clinicians.
- **Out-of-scope use cases**: unsupervised patient-facing automation; replacement of clinician judgement; settings where the LACE feature space is unavailable; pediatric patients (the model was not fit on a pediatric cohort).

## 3. Factors

- **Relevant factors**: age band (5 strata), sex, race (White/Black/Hispanic/Asian/Indigenous/Other), ethnicity (Hispanic/Non-Hispanic), insurance type (Commercial/Medicare/Medicaid/Uninsured/TRICARE), language.
- **Evaluation factors**: same as relevant factors plus the 5-bin LACE stratification.
- **Group attributes that drive different performance**: Black + Indigenous + Medicaid + Uninsured patients show the largest TPR / DP gap (per the Phase 15 subgroup audit).
- **Instrumentation**: predictions are deterministic given the LACE bin; ties are broken on tool name.

## 4. Metrics

**Performance measures**: ECE (Expected Calibration Error, preferred gate <=0.05), Brier score, AUROC. Decision-level TPR + Demographic-Parity gap surfaced per subgroup.

| Cohort | n | ECE | Brier | AUROC |
| --- | ---: | ---: | ---: | ---: |
| W1 spec_002 | 7,880 | 0.0078 | 0.124 | 0.590 |
| Synthea-100k | 100,000 | 0.0001 | 0.1527 | 0.6093 |
| MIMIC-IV demo | 275 | 0.0187 | 0.1489 | 0.6399 |

**Decision threshold**: 0.20. **Confidence interval**: 5-bin Beta-Binomial posterior, width scaled by bin sample size.

## 5. Evaluation data

- **Datasets**: (a) W1 internal calibration cohort (n=7,880); (b) Synthea-10k validation; (c) Synthea-100k recalibration; (d) MIMIC-IV demo 2.2 (n=275 in-cohort); (e) Phase-15 subgroup-audit synthetic cohort (n=100,000) with rich demographics.
- **Motivation**: the W1 cohort drives the initial fit; Synthea-100k stress-tests calibration at scale; MIMIC-IV demo provides external validation; the subgroup-audit cohort drives fairness analysis.
- **Preprocessing**: LACE features extracted from FHIR Bundle (length-of-stay from Encounter.period, acuity from Encounter.priority + chief complaint, Charlson from Conditions, ED visits from past Encounters).

## 6. Training data

- **Training cohort**: W1 internal calibration workflow output, n=7,880, Synthea-generated synthetic 30-day readmission cohort with the LACE feature space.
- **Synthea version**: 3.3.0 (BSD-3 licence). Generation controlled by the internal `fhir-readmission-calibration` workflow (W1).
- **Outcome label**: binary 30-day all-cause readmission drawn from a fixed bucket map {LACE 0-2: 7.2%, 3-5: 10.3%, 6-9: 15.8%, 10-12: 23.4%, 13+: 32.7%}; calibrated against AHRQ HCUP HRRP rates.

## 7. Quantitative analyses

**Disaggregated evaluation results**: per Phase-15 subgroup-audit + prospective-eval artefacts.
- Worst EOO gap: 7.14% on race=indigenous.
- Worst DP gap: 1.85% on race=indigenous.
- Prospective eval (n=2,000): abstain=0.00%, downgrade=64.10%, calibration gap=3.04%.

## 8. Ethical considerations

- **Sensitive data**: predictions consume protected demographics (race, ethnicity, insurance). The 4-critic ensemble's `fairness_critic` blocks any recommendation that would adversely change for a flagged subgroup at matched LACE.
- **Mitigation strategies**: deterministic abstain trigger for demographic-bias subgroups without a fresh fairness audit; differential-privacy publication of subgroup rates (Laplace, epsilon=1.0).
- **Risks of harm**: under-prediction in Black + Indigenous + Medicaid populations is documented in the literature (AHRQ HCUP 2022, Joynt & Jha 2014). The system surfaces this risk directly via the Fairness Watch block.

## 9. Caveats and recommendations

- **Caveats**: the LACE feature space is reductive. Real deployments should re-fit on local data and re-run the fairness audit on the local cohort before release.
- **Recommendations**: do not deploy without a clinician in the loop; do not deploy on cohorts where sub-population coverage falls below the suppression threshold (n=20 raw); re-calibrate quarterly, monitor drift continuously.
