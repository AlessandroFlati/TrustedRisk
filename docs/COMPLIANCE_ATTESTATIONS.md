# Compliance Attestations -- TrustedRisk

**Document type**: structured attestations templates for the three
regulatory frameworks the runtime must satisfy.
**Status**: pre-deployment -- these are templates; institutional
counsel + privacy officer must sign off before clinical use.
**Generated**: 2026-04-30 UTC
**Companion**: `docs/FDA_SAMD_ANALYSIS.md`, `docs/PROSPECTIVE_STUDY_PROTOCOL.md`,
`docs/ISO_13485_CHECKLIST.md`, `SECURITY.md`.

---

## §1. HIPAA Privacy Rule attestation (45 CFR §164)

### 1.1 Scope of PHI handling

TrustedRisk handles Protected Health Information (PHI) only as a
**Business Associate** of the deploying healthcare organisation
(Covered Entity). The runtime:

- Receives PHI through the SHARP-on-MCP context headers
  (`X-FHIR-Server-URL`, `X-FHIR-Access-Token`, `X-Patient-ID`).
- Reads patient FHIR resources (Conditions, Observations,
  MedicationRequests, etc.) only with valid OAuth-issued tokens.
- Persists nothing PHI-identifying beyond a single request scope
  unless the optional memory layer is enabled by the operator.
- When the memory layer IS enabled, every persisted record is
  PHI-redacted (SHA-256 hashed) before storage.

### 1.2 Required Business Associate Agreement

Before clinical deployment, the operator MUST execute a HIPAA
Business Associate Agreement (BAA) with the publisher / deployer of
TrustedRisk. The BAA must cover:

- Permitted uses: clinical decision support, quality improvement,
  regulatory submission preparation. Prohibited uses: marketing,
  reselling de-identified data without a separate Data Use Agreement.
- Safeguards: encryption at rest (AES-256), encryption in transit
  (TLS 1.2+), access controls, audit logging.
- Sub-contractors: any cloud provider (e.g., Google Cloud Run) must
  itself sign a BAA with the deployer.
- Breach notification: 60-day window per §164.410.
- Termination: data destruction or return on contract end.

### 1.3 Implementation evidence

| Safeguard | Module / Test | Evidence |
|---|---|---|
| PHI redaction in audit log | `a2a_agent.audit._hash` | SHA-256 over redacted input fields |
| `compute_detect_phi` | `tests/property/test_phi_redaction_stress.py` | Hypothesis-based residual-PHI check |
| Multi-tenant FHIR allowlist | `tests/unit/test_oauth.py`, `tests/unit/test_sharp_multitenant.py` | Per-tenant `allowed_fhir_servers` |
| Encryption in transit | `Dockerfile` + Cloud Run | TLS termination at load balancer |
| Tamper-evident audit | `a2a_agent.merkle_audit` | RFC 6962 Merkle chain |

### 1.4 Attestation statement (template)

> The deploying institution attests that:
> 1. A BAA has been executed with the TrustedRisk publisher;
> 2. PHI flows are limited to those covered by the BAA + the SHARP
>    headers contract;
> 3. The audit log + reproducibility archive are reviewed quarterly
>    by the Privacy Officer.

---

## §2. GDPR Art. 22 -- Right to not be subject to a decision based solely on automated processing

### 2.1 Position

TrustedRisk is **clinician-supervised decision support**, not an
autonomous decision-maker. Per Art. 22(1), the right to "not be
subject to a decision based solely on automated processing" is
satisfied because:

1. Every recommendation is paired with `confidence` + `abstain` triggers
   that prompt clinician adjudication.
2. The deterministic safety gate the LLM cannot relax means the
   clinician retains final authority -- at most the system can ABSTAIN
   or escalate to a more conservative action; never relax the
   clinician's decision.
3. Every output ships with `valid_for_minutes` ≤ 60, forcing
   re-evaluation before the action is taken.

### 2.2 Worked example: right to explanation

Patient X requests an explanation of TrustedRisk's discharge-risk
recommendation under GDPR Art. 22(3). The runtime produces:

1. **Calibrated probability** with 95 % credible interval --
   `RiskEstimate.probability_mean ± probability_ci95`.
2. **LACE feature breakdown** -- the four input features (L, A, C, E)
   with their contributions (Phase 7.4 SHAP attributions).
3. **Counterfactual** -- `compute_counterfactual_explanation` produces
   "what if your LACE score were 8 instead of 12?".
4. **Patient-language audit summary** --
   `compute_patient_audit_summary` emits a non-technical summary of
   the decision flow + abstain triggers + critic verdict.
5. **Right-to-explanation document** --
   `compute_right_to_explanation_document` (a2a_agent module) emits
   the formal artefact required by Art. 22 + Recital 71:
   - Algorithm logic (LACE Beta-Binomial, calibrated)
   - Model version + coefficient SHA
   - Inputs consumed (per-FHIR-resource ID)
   - Outputs + confidence
   - Cohort baseline against which the patient's risk is compared
   - Path to challenge the decision (clinician override + audit-log
     re-review)

### 2.3 Implementation evidence

| Right | Module | Test |
|---|---|---|
| Right to be informed | `a2a_agent.right_to_explanation` | `tests/unit/test_right_to_explanation.py` |
| Right to explanation | `a2a_agent.patient_audit_summary` | `tests/unit/test_patient_audit_summary.py` |
| Right to challenge | Clinician override path + audit log | `a2a_agent.audit` + Merkle chain |
| Right to data portability | DecisionCard JSON export | Pydantic schema = canonical contract |

---

## §3. EU AI Act Art. 13 -- Transparency and provision of information

### 3.1 Position

TrustedRisk is a "high-risk AI system" under Annex III of the EU AI
Act because it provides clinical decision support to healthcare
professionals. Art. 13(1) requires "sufficiently transparent and
explained" outputs.

### 3.2 Implementation evidence

| Art. 13 requirement | TrustedRisk surface |
|---|---|
| Operating instructions | `docs/MODEL_CARD.md` + `docs/TUTORIALS.md` + `docs/ARCHITECTURE.md` |
| Performance characteristics | `docs/validation/VALIDATION.md`, `docs/validation/SYNTHEA_10K.md`, `docs/performance/v7_benchmarks.md` |
| Risks (incl. residual) | `docs/MODEL_CARD.md` §8 + `docs/FDA_SAMD_ANALYSIS.md` §4 |
| Capabilities + limitations | Agent card capability declarations + per-tool `references` field |
| Human-oversight measures | Multi-critic ensemble + abstain triggers + clinician override |
| Continuous-learning controls | `a2a_agent.drift_monitor` + PCCP sketch in `docs/FDA_SAMD_ANALYSIS.md` §6 |
| Logging | `a2a_agent.audit` JSONL + `a2a_agent.merkle_audit` |
| Cybersecurity | `SECURITY.md` + `docs/CVE_SCAN.md` |
| Bias / discrimination | `compute_fairness_audit` + `a2a_agent.dp_equity` + demographic bias guards |

### 3.3 CE mark prerequisites (EU MDR + AI Act)

- ISO 13485 QMS (per `docs/ISO_13485_CHECKLIST.md` §1-§11)
- Notified-body conformity assessment
- Technical documentation matching Annex IV
- Post-market surveillance plan (Art. 72)
- Risk-management file matching ISO 14971

These are out of scope for this prototype release; the runtime
exposes the substrate, the institutional QMS process completes the
formal certification.

### 3.4 Disclaimer

TrustedRisk is currently a **research prototype**, not a CE-marked
medical device. Production deployment in the EU requires:
- CE marking under MDR + AI Act
- Notified-body conformity assessment (Class IIa or IIb depending on
  intended use)
- Updated `docs/MODEL_CARD.md` + EU-AI-Act-aligned technical file

The clinician-supervised decision-support framing (§2.1, §3.1) keeps
TrustedRisk inside the *advisory* perimeter while the formal
certification is being pursued.

---

## §4. References

- HIPAA Privacy Rule (45 CFR §164)
- HIPAA Security Rule (45 CFR §164 Subpart C)
- GDPR (Regulation (EU) 2016/679), Art. 22 + Recital 71
- EU AI Act (Regulation (EU) 2024/1689), Art. 13 + Annex III
- EU MDR (Regulation (EU) 2017/745)
- ISO 14971 -- Application of risk management to medical devices
- ISO 13485 -- QMS for medical-device organisations
- IEC 62304 -- Medical-device software lifecycle
- FDA *Cybersecurity in Medical Devices* (Sept 2023)
- FDA *Clinical Decision Support Software* (Sept 2022)
