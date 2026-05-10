# Prospective Clinical Validation Study Protocol -- TrustedRisk v0.7.0

**Document type**: IRB-ready study protocol (template)
**Status**: pre-IRB submission, internal feasibility document -- TrustedRisk has NOT been clinically validated.
**Companion**: `docs/FDA_SAMD_ANALYSIS.md` -- closes Pillar-3 (clinical validation) gap.
**Sponsor**: Alessandro Flati (independent research prototype)
**Generated**: 2026-04-29 UTC
**Version**: 0.1 (template)

---

## 1. Study Title

> **Prospective Clinical Validation of TrustedRisk Calibrated Discharge Risk and Decision-Support Recommendations**

---

## 2. Purpose

The FDA SaMD Pillar-3 (clinical validation) is the binding open item on
the TrustedRisk submission roadmap (`docs/FDA_SAMD_ANALYSIS.md` §7).
Pillars 1 (valid clinical association) and 2 (analytical validation) are
met. This protocol scopes a prospective cohort study at a single
institution (expandable to multi-site) that will produce the
clinical-validation evidence FDA requires.

Concurrent secondary aim: capture clinician-utility data sufficient to
support a 510(k) substantial-equivalence argument vs the cleared
predicate (Bayesian Health, K201992).

---

## 3. Study Design

| Field | Value |
|---|---|
| **Design** | Prospective single-arm cohort + paired clinician-vs-algorithm comparison |
| **Setting** | Adult inpatient hospitalist service at ≥ 1 academic medical center |
| **Enrollment target** | 1 000 patients (powered as below; +20% buffer for losses to follow-up) |
| **Duration** | 12 months enrollment + 30-day follow-up per patient |
| **Intervention** | None (observational -- TrustedRisk runs alongside usual care; clinicians make all decisions) |

The investigators will run TrustedRisk on every eligible discharge in
parallel with the institution's standard discharge process. Both the
TrustedRisk recommendation and the clinician's recorded decision will
be captured at the moment of discharge. The 30-day readmission outcome
is the primary endpoint.

---

## 4. Eligibility

### 4.1 Inclusion criteria
- Age ≥ 18
- Inpatient admission with planned discharge from a participating hospitalist service
- FHIR R4 chart accessible via SHARP-compliant export

### 4.2 Exclusion criteria
- AMA / against-medical-advice discharge (out of decision-support scope)
- Hospice / comfort-care disposition (mortality-driven; out of scope for readmission risk)
- Pediatric patients (model not validated < 18; pediatric specialist exists but uses a different model)
- Pregnant patients on the L&D service (covered by a separate obstetric specialist)

### 4.3 Vulnerable populations
The protocol explicitly enrolls patients in groups that prior literature
has flagged for risk of mismeasurement (Black, Indigenous, Hispanic /
Latino, low-SES). The fairness-audit endpoint (§5.3) is the safeguard --
the runtime's `compute_fairness_audit` tool runs on every prediction
and can trigger an abstain if subgroup drift exceeds the literature
baseline.

---

## 5. Endpoints

### 5.1 Primary endpoint
**30-day all-cause readmission rate**, ascertained via:
1. Hospital EHR encounter records, AND
2. Statewide HIE / HEDIS hospital-readmission feed
(both queried at day-30 + day-60 to capture out-of-network readmissions)

### 5.2 Secondary endpoints
- AUROC, ECE, Brier score on the calibrated risk model -- overall and per-subgroup (race, age band, insurance, sex)
- Subgroup parity: maximum subgroup-level intervention-rate disparity vs the cohort mean
- Clinician-utility composite: time saved per decision (clinician-reported), perceived usefulness (Likert 1-5), abstention agreement
- Adverse-event flag rate: did TrustedRisk's deterministic safety gate trigger a force_abstain that the clinician overrode? Of those overridden, what was the 30-day outcome?

### 5.3 Safety endpoints
- Number of `force_abstain` events
- Subgroup-drift triggers (per `compute_fairness_audit`)
- OOD detector triggers (per `compute_ood_detector` -- Mahalanobis χ²)

---

## 6. Sample-size justification

### 6.1 Primary (calibration)
H₀: ECE ≥ 0.10 (clinically unacceptable miscalibration).
H₁: ECE < 0.10 (the prevalidation target).

For binomial proportion with expected ECE 0.05 and one-sided α = 0.025
+ power 0.80, n = 873 patients are needed. We round up to 1 000 with
20 % buffer for losses -> enrollment target 1 200.

### 6.2 Secondary (subgroup parity)
For ≥ 80 % power to detect a 0.05 absolute difference in subgroup-level
intervention rate (Bonferroni-corrected for 6 subgroups), n ≈ 950 is
sufficient. The 1 200-patient enrollment cleanly satisfies both.

### 6.3 Clinician-utility
At ≥ 200 unique clinician-decision pairs, paired-comparison Wilcoxon
signed-rank has 80 % power to detect a Cohen's d ≥ 0.20 effect on the
Likert composite. This is achieved within the first 3 months of
enrollment at typical hospitalist volumes.

---

## 7. Procedures

### 7.1 At admission
- Patient enrolled with informed consent (waiver-of-consent IRB
  petition for the observational arm; the algorithm is not making
  binding clinical decisions).
- FHIR R4 chart auto-exported at every shift change to the TrustedRisk
  staging environment.

### 7.2 At discharge decision
- Clinician records discharge decision in the EHR.
- TrustedRisk runs the full `core_discharge` bundle in the background:
  `compute_readmission_risk -> compute_decision_utility -> compute_fairness_audit -> compute_counterfactual_explanation -> compute_lab_trend_analysis -> compute_discharge_counseling`. The 4-critic ensemble votes; a `DecisionCard` is archived.
- The DecisionCard is NOT shown to the clinician during the study
  (avoids contaminating clinician judgement); a separate clinician-
  utility arm with 200 patients shows it post-hoc and captures the
  Likert score.

### 7.3 At day 30
- 30-day readmission ascertainment (§5.1).
- Statewide HIE feed queried via FHIR `Encounter?patient=...&type=...&period=...`.
- All chart-confirmed readmissions adjudicated by an independent
  clinician (blinded to the TrustedRisk recommendation).

### 7.4 At day 60
- Out-of-network readmission catch via HEDIS feed.

### 7.5 Statistical analysis plan (SAP)
- Primary analysis: ECE on the held-out 1 000-patient cohort,
  bootstrap 95 % CI by stratified resample (10 000 replicates).
- Secondary analyses: Brier, AUROC, MCE, calibration intercept + slope
  (Cox regression).
- Subgroup analyses: per-subgroup ECE + intervention rate, Bonferroni
  α = 0.05/6.
- Sensitivity analyses: varying the abstain threshold; varying the
  fairness-baseline source (literature vs cohort-calibrated W5 artifact).
- Pre-registered: Open Science Framework registration prior to
  enrollment lock.

---

## 8. Data Management

### 8.1 Sources
- EHR (Epic/Cerner/MEDITECH -- SHARP-on-MCP context propagation)
- HEDIS hospital-readmission file
- HIE encounter feed

### 8.2 De-identification
- All TrustedRisk audit logs (`a2a_agent.audit`) use SHA-256 PHI-redacted hashing.
- Reproducibility archive (`a2a_agent.audit.fetch_archived_decision`) retains structured inputs without identifying fields.
- Merkle audit chain (`a2a_agent.merkle_audit`) provides RFC 6962 tamper-evidence.

### 8.3 Storage
- Encrypted at rest (AES-256, key managed by institutional KMS).
- Encrypted in transit (TLS 1.3+).
- Access logs reviewed quarterly.

### 8.4 Retention
- Identified data: 6 years per institutional IRB policy.
- De-identified analytical dataset: indefinitely; published with the
  primary manuscript via OSF.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Patient harm from TrustedRisk recommendation | OBSERVATIONAL -- clinician makes all decisions; algorithm output not shown during the primary arm. Clinician-utility arm uses post-hoc disclosure under separate IRB review. |
| Demographic-bias amplification | Fairness audit is built into the runtime; subgroup drift > 30 % triggers abstain. The C-SSRS demographic bias guard explicitly abstains for groups documented as risk-mismeasured. |
| Reproducibility loss | Reproducibility archive enables byte-identical replay of any historical decision. Merkle audit chain provides tamper evidence. |
| FHIR access-token compromise | OAuth 2.0 client_credentials with per-tenant allowlist. SHARP middleware enforces FHIR context per request. |

---

## 10. Regulatory + IRB

### 10.1 IRB approval pathway
- IRB submission expected 2-3 months prior to enrollment lock.
- Anticipated waiver-of-consent for the observational arm (no patient-
  facing intervention; algorithm output not shown during primary arm).
- Non-waiver consent for the clinician-utility arm.

### 10.2 FDA pre-submission interaction
- Q-Submission package (Pre-Sub) anticipated 90 days before enrollment.
- Asks: IRB scope, statistical analysis plan, primary endpoint
  definition (HEDIS feed alignment), subgroup parity threshold.

### 10.3 Independent Data Safety Monitoring Board (DSMB)
- DSMB activated at month 6 + every 3 months thereafter.
- Stopping rules: aggregate harm signal (force_abstain rate > 8 %
  combined with subgroup-drift triggers > 15 %).

---

## 11. Investigator Roster

> **Pre-submission stub** -- finalised at IRB submission.

- Principal Investigator: TBD (institutional affiliation required).
- Co-Investigators: TBD (≥ 1 hospitalist, ≥ 1 biostatistician, ≥ 1 health informaticist, ≥ 1 patient/family advisor).
- Sponsor: Alessandro Flati (model developer, study sponsor).
- Independent adjudicator: TBD (blinded clinician for outcome adjudication).

---

## 12. Anticipated timeline

```
Pre-submission Q-Sub ->  3 months
IRB submission     ->   2 months after Pre-Sub
IRB approval       ->   2 months after submission
Enrollment lock    ->  12 months
30-day follow-up   ->   1 month after last enrollment
Statistical analysis ->  3 months
Primary manuscript draft ->  2 months
Submission to FDA  ->   1 month after manuscript
```

End-to-end timeline ≈ 24-30 months.

---

## 13. References

- FDA. *Software as a Medical Device (SaMD): Clinical Evaluation* (Aug 2017).
- FDA. *Cybersecurity in Medical Devices: Quality System Considerations* (Sept 2023).
- IMDRF. *"Software as a Medical Device": Possible Framework for Risk Categorization* (IMDRF/SaMD WG/N12FINAL:2014).
- 21st Century Cures Act, §3060 (P.L. 114-255).
- Bayesian Health Inc. K201992 -- 510(k) clearance (candidate predicate).
- Walraven C, et al. *Derivation and validation of an index to predict early death or unplanned readmission after discharge.* CMAJ 2010;182:551.
- Mitchell M, et al. *Model Cards for Model Reporting.* JAMA 2019. (companion model card at `docs/MODEL_CARD.md`)
- Sanders GD, et al. *Recommendations for Conduct, Methodological Practices, and Reporting of Cost-effectiveness Analyses.* JAMA 2016;316:1093.

---

## 14. Disclaimer

This document is a **template prepared by the model developer** --
not an IRB-approved protocol. Actual study execution requires:
- Institutional IRB approval
- Investigator credentialing
- DSMB convening
- Pre-Sub interaction with FDA
- Data Use Agreement(s) with the institution(s)

This template is suitable as the starting draft for any of the above
processes; it is NOT itself a regulatory submission.
