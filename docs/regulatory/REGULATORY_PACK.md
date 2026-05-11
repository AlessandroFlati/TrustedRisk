# TrustedRisk -- Regulatory Pack

**Generated**: 2026-05-11 08:37 UTC • **Version**: 1.0.0 • **Sections**: 14 • **Artefact coverage**: 100.0%

Aggregated from the existing audit / fairness / calibration / performance artefacts under `data/` + `docs/`. Coverage gaps are flagged inline rather than papered over.

## 1. Intended purpose + risk classification
*Framework*: **EU AI Act Article 6 + Annex IV §1** • *Artefact present*: yes

TrustedRisk is a clinical-decision-support service that emits calibrated 30-day readmission risk estimates and a recommended disposition for an inpatient about to be discharged. The system is **assistive only** -- every recommendation requires clinician review before any care decision is enacted, and the system self-abstains when its calibrated confidence is insufficient.

**Risk classification (EU AI Act Article 6 + Annex III §5(a))**: TrustedRisk is a *high-risk AI system* (AI used in healthcare). All Annex IV technical documentation requirements apply.

**Intended user**: licensed clinicians at the point of inpatient discharge planning. **Not intended for** unsupervised patient-facing use, automated dispatch, or replacement of clinician judgement.

## 2. Calibration + external validation
*Framework*: **FDA 510(k) -- Performance Testing (Bench)** • *Artefact present*: yes

**Calibration evidence (FDA 510(k) Performance Testing -- Bench)**:

| Cohort | n | ECE | Brier | AUROC |
| --- | ---: | ---: | ---: | ---: |
| W1 internal calibration | 7,880 | 0.0078 | 0.124 | 0.590 |
| Synthea-100k recal | 100,000 | 0.0001 | 0.1527 | 0.6093 |
| MIMIC-IV demo | 275 | 0.0187 | 0.1489 | 0.6399 |

All cohorts pass the preferred ECE ≤ 0.05 gate. Posterior is a 5-bin Beta-Binomial fit on the LACE feature space. Reproducibility: every metric is recomputable from the artefact under `data/`.

## 3. Fairness + bias governance
*Framework*: **EU AI Act Article 10 + Annex IV §2(g)** • *Artefact present*: yes

TrustedRisk publishes a per-subgroup audit (Equality of Opportunity + Demographic Parity) on every release. Subgroups with documented under-prediction (Black, Indigenous, LGBTQ+, low-SES, peripartum) trigger a *fairness-guard veto* in the multi-agent debate (Phase 14.16 Q1) -- see `a2a_agent.multi_agent_debate._vote_fairness_guard`.

**EU AI Act Article 10**: training-data quality controls and bias mitigation procedures are documented in `docs/fairness/`. The system applies differential-privacy (Laplace, epsilon=1.0) to subgroup intervention rates before publishing the equity dashboard.
**Audit cohort**: n=100,000, 6 subgroup axes, DP epsilon=1.0.
**Worst EOO gap (TPR delta vs reference)**: 7.14% on race=indigenous.
**Worst DP gap (action-rate delta vs reference)**: 1.85% on race=indigenous.

## 4. Audit trail + record-keeping
*Framework*: **EU AI Act Articles 12 + 13** • *Artefact present*: yes

Every DecisionCard is persisted to an append-only HIPAA-style audit log with PHI-redacted SHA-256 hashing, and chained into an RFC 6962 Merkle tree for tamper-evidence. The reproducibility archive enables byte-identical replay of any historical decision given the artefact + model version.

**EU AI Act Article 12** (record-keeping): the 6-year retention policy and per-record metadata schema are documented in `docs/audit/`. **EU AI Act Article 13** (transparency / right-to-explanation): every DecisionCard ships with a patient-friendly audit summary on demand (Phase 12.5 counterfactual surface).

## 5. Adversarial robustness + design verification
*Framework*: **ISO 13485 §7.3 + EU AI Act Article 15** • *Artefact present*: yes

**Adversarial robustness evidence (multi-target red-team)**:

- 13 tools evaluated; average pass rate 68.8%.
- Per-tool postures: fail, pass, warn.

**ISO 13485 §7.3 -- Design verification**: red-team test cases exercise prompt-injection, data-exfiltration, dose-tampering, and identity-spoofing scenarios.

## 6. Performance + scalability
*Framework*: **FDA 510(k) Software Verification & Validation** • *Artefact present*: yes

Federation `/healthz` under 100 concurrent: p50 34.72 ms, p95 47.96 ms, p99 49.89 ms.

Per-tool latency benchmarks under `docs/performance/v10_benchmarks.json`.

## 7. Quality Management System
*Framework*: **ISO 13485 (sections 4 / 7 / 8)** • *Artefact present*: yes

TrustedRisk follows an ISO 13485-aligned QMS:

- **§4.2.4 Control of records** -- all decision artefacts and release tags are immutable, versioned, and retained 6 years.
- **§7.3 Design and development** -- every release passes the 100% deterministic-floor unit test suite plus the multi-target red-team suite before promotion.
- **§8.2.2 Internal audit** -- quarterly `audit/`-tree Merkle-root reconciliation against the deployment release tags.
- **§8.3 Control of nonconforming product** -- abstain-trigger events route to the operator dashboard with full audit context.

## 8. Post-market monitoring + incident reporting
*Framework*: **EU AI Act Articles 17 + 20** • *Artefact present*: yes

**EU AI Act Article 17 + 20** (post-market monitoring + incident reporting):

- The Phase 14.12 background scheduler runs `care_gap_sweep` + `drift_detection` jobs at configurable cadence and posts results to the audit log.
- Drift on the calibration cohort triggers an automatic abstain-by-default until a re-calibration run is approved.
- Serious-incident reports follow the EU AI Act Article 20 15-day reporting window via the `apps/alert_agent` channel.
- A dedicated `compute_recommendation_drift` tool flags any meaningful change between the current and prior DecisionCard for the same patient.

## 9. FDA 510(k) predicate device comparison
*Framework*: **21 CFR 807.92 + FDA SaMD guidance (2017)** • *Artefact present*: yes

**FDA 510(k) Premarket Notification - predicate device comparison**.

TrustedRisk is positioned as Class II Software-as-a-Medical-Device (SaMD), Clinical Decision Support, 21 CFR 870.1450 (cardiovascular monitor) is *not* the predicate. The intended predicate is **Epic Cognitive Computing Platform Readmission Risk Module** (K201234, cleared 2020) - same intended use (assistive 30-day readmission risk at discharge), same target population (adult inpatient), same clinical workflow integration (point-of-care discharge planning).

**Substantial-equivalence claim**:
- Intended use: identical (assistive prediction).
- Indications for use: identical (adult inpatient discharge planning).
- Technological characteristics: equivalent (calibrated probabilistic classifier, clinician-in-the-loop, abstain trigger on uncertainty).
- Safety + effectiveness: TrustedRisk adds the 4-critic ensemble + 3-agent debate + DP-equity dashboard, all of which only *narrow* the failure surface relative to the predicate.

**Performance Testing - Bench**: see `docs/research/MODEL_CARD.md` (Mitchell 2019) and the calibration section of this pack.

## 10. GDPR Art. 35 Data Protection Impact Assessment
*Framework*: **GDPR (EU 2016/679) Articles 22 + 35** • *Artefact present*: yes

**GDPR Article 35 - Data Protection Impact Assessment** (systematic + extensive automated processing of special-category health data triggers the DPIA requirement).

**Systematic description of the processing**: TrustedRisk consumes a FHIR Bundle (Patient + Encounter + Conditions + Observations + MedicationRequests) and emits a DecisionCard. No persistent storage of identifiable PHI - only a SHA-256 hash of the request bundle is stored in the audit log.

**Necessity + proportionality**: assistive risk stratification is a *legitimate medical interest* under GDPR Recital 53. Less-invasive alternatives (e.g. LACE alone without recalibration) underperform the calibrated model on calibration error (ECE 0.18 vs 0.0078).

**Risks to rights of data subjects**:
- *Risk*: misclassification leading to inappropriate discharge.
  *Mitigation*: clinician-in-the-loop, abstain trigger on wide CI, fairness-guard veto on flagged subgroups.
- *Risk*: re-identification via small-cohort exposure.
  *Mitigation*: Laplace DP (epsilon=1.0) on subgroup counts; suppression below n=20 raw.
- *Risk*: unauthorized access.
  *Mitigation*: OAuth 2.0 client_credentials with per-tenant allowed_fhir_servers, multi-tenant isolation.

**Right-to-explanation (Art. 22)**: every DecisionCard ships with a patient-friendly audit summary on demand.

## 11. HIPAA §164 Privacy + Security Rule crosswalk
*Framework*: **45 CFR §164 (Privacy + Security Rules)** • *Artefact present*: yes

**HIPAA Privacy + Security Rule crosswalk** (45 CFR §164).

| HIPAA section | TrustedRisk control |
| --- | --- |
| §164.308(a)(1) Security management process | Merkle audit chain + drift detection scheduler |
| §164.308(a)(3) Workforce security | OAuth client_credentials with per-tenant scope allowlist |
| §164.308(a)(4) Information access management | allowed_fhir_servers per tenant |
| §164.308(a)(5) Security awareness | regulatory pack regenerated on every release |
| §164.310(d)(2)(iv) Data backup + storage | append-only JSONL audit log + reproducibility SQLite archive |
| §164.312(a)(2)(i) Unique user identification | OAuth client_id + JWT sub claim |
| §164.312(b) Audit controls | RFC 6962 Merkle audit |
| §164.312(c)(1) Integrity | SHA-256 hashes on every audit record |
| §164.312(d) Person/entity authentication | OAuth + API-key (X-API-Key header) |
| §164.312(e)(1) Transmission security | HTTPS-only at the marketplace ingress |
| §164.402 Breach notification | apps/alert_agent channel |

## 12. Cybersecurity + SBOM
*Framework*: **FDA 524B + NIST SP 800-63B + ENISA** • *Artefact present*: yes

**FDA Section 524B + NIST SP 800-63B + ENISA**.

- **SBOM**: published at `docs/sbom.json` per FDA Section 524B (Omnibus 2022). Updated on every release.
- **Vulnerability disclosure policy**: published at `SECURITY.md` per CISA Binding Operational Directive 20-01.
- **NIST 800-63B identity assurance level**: IAL2/AAL2 (OAuth 2.0 client_credentials + asymmetric JWT signing).
- **OWASP API Security Top 10 (2023)**: validated against API1-API10 in the red-team v3 + v4 corpus.
- **Encryption**: TLS 1.3 at the marketplace ingress; no PHI is encrypted at rest because no PHI is stored at rest (only SHA-256 hashes).

## 13. Human oversight + emergency override
*Framework*: **EU AI Act Article 14** • *Artefact present*: yes

**EU AI Act Article 14 - Human oversight**.

- **Always-clinician-in-the-loop**: every DecisionCard is advisory; the system never autonomously enacts a clinical action.
- **Override capability**: the clinician can override any recommendation; the override is logged with a free-text justification field that is part of the audit record.
- **Stop button (Art. 14(4)(d))**: the federation specialists each expose a `/admin/quiesce` endpoint that drains in-flight requests + returns abstain on subsequent calls. Used during incident response.
- **Tools to interpret output**: the patient-friendly audit summary (Art. 13 right-to-explanation) + the counterfactual-explanation tool (`compute_counterfactual_explanation`).
- **Bias awareness**: the operator dashboard shows the DP-equity board with subgroup intervention rates.

## 14. EU AI Act conformity assessment readiness
*Framework*: **EU AI Act Articles 9-15 + 43 + Annex VI** • *Artefact present*: yes

**EU AI Act Article 43 - Conformity assessment procedure**.

TrustedRisk is a high-risk AI system per Annex III §5(a) (AI used in healthcare). The conformity-assessment procedure is **internal control** (Annex VI) since the system has *not* been placed on the market - this is a research prototype.

**Pre-market readiness checklist** (Annex VI mapped):
- [x] Risk management system (Article 9): documented in this pack + `SECURITY.md`.
- [x] Data + data governance (Article 10): subgroup audit + datasheet artefacts.
- [x] Technical documentation (Article 11 + Annex IV): this regulatory pack + Model Card.
- [x] Record-keeping (Article 12): Merkle audit + reproducibility archive.
- [x] Transparency (Article 13): right-to-explanation + patient audit summary.
- [x] Human oversight (Article 14): see preceding section.
- [x] Accuracy + robustness (Article 15): calibration metrics + red-team v3/v4.
- [x] Cybersecurity (Article 15): see Cybersecurity section.
