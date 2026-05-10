# TrustedRisk: A Calibrated, SHARP-on-MCP Multi-Agent System for Clinical Decision Support

**Version 1.0 -- 2026-04-30 05:53 UTC**

---

## Abstract

We present **TrustedRisk**, an open-source multi-agent system for production-grade clinical decision support that composes **89 deterministic Model-Context-Protocol (MCP) tools** across **30 thematic bundles** behind **14 marketplace-publishable A2A v1 specialists**. The system threads a **SHARP-on-MCP** FHIR-context pipeline (A2A v1 FHIR-context extension URI: `https://app.promptopinion.ai/schemas/a2a/v1/fhir-context`) so that bearer tokens never reach the LLM prompt. Every recommendation is wrapped in a **calibrated risk estimate**, a **deterministic safety gate**, an **abstain trigger**, a **4-critic ensemble vote** (clinical safety + fairness + evidence + LLM-judge), a **subgroup fairness audit**, a **counterfactual explanation**, an **RFC-6962 Merkle audit chain**, and a **differential-privacy equity dashboard**. We report **expected calibration error (ECE) 0.0078** on a 7,880-row Spec-002 5-bin Beta-Binomial calibration, **AUROC 0.590**, **ECE 0.0056** on a 10,000-patient Synthea validation, and a **110-prompt v2 red-team campaign** with **100% pass rate** against the PHI scrubber. The full federation runs locally via `make demo-up-full` and is deployable to Cloud Run via a multi-mode Dockerfile.

**Keywords:** clinical decision support · MCP · A2A v1 · SHARP-on-MCP · HEDIS · CMS Stars · differential privacy · conformal prediction · 4-critic ensemble · adversarial robustness.

---

## 1. Introduction

Generative-AI agents in healthcare must reconcile three constraints that are usually in tension:

1. **Calibration.** Probability scores must be empirically aligned to outcomes -- an estimated 30-day readmission risk of 0.20 must observe close to a 20% empirical readmission rate on the calibration cohort.
2. **Safety.** A single hallucinated medication dose, an unreversed "discharge_home" recommendation, or a leaked SSN can carry malpractice and HIPAA exposure that dwarfs the system's clinical utility.
3. **Composability.** A clinical decision rarely lives inside one agent: HEDIS care-gap scoring (population), prior-authorization letter drafting (RCM), and patient post-discharge Q&A (caregiver-facing) are upstream / downstream of the same chart.

TrustedRisk's central design bet is that **calibrated probabilistic outputs**, **deterministic clinical floors**, and **federated A2A specialists sharing one MCP backend** can satisfy all three simultaneously without sacrificing depth on any.

---

## 2. System Architecture

### 2.1 Federation Topology

The system exposes **14 A2A v1 specialists**, each on its own port and base URL, all sharing a **single in-process MCP backend** registered with FastMCP. The roster as of 2026-04-30:

| Specialist | Port | Tagline |
|---|---|---|
| trustedrisk-discharge | 8770 | Adult inpatient discharge planning + outpatient med review |
| trustedrisk-acute | 8771 | ED + acute / critical-care decision support |
| trustedrisk-evidence | 8772 | Clinician evidence retrieval + grounded DDx |
| trustedrisk-population | 8773 | Cost-effectiveness + DP equity dashboard |
| trustedrisk-pedi-mh | 8774 | Pediatric early warning + mental-health crisis |
| trustedrisk-pa | 8775 | Prior Authorization (Phase 2.1 PA-1/2/3/4) |
| trustedrisk-scribe | 8776 | Clinical documentation drafting (SCRIBE-1/2/3/4) |
| trustedrisk-patient | 8777 | Post-discharge Q&A + caregiver hand-off |
| trustedrisk-coder | 8778 | Auto-coding (CODE-1/2/3/4) |
| trustedrisk-pgx | 8779 | Pharmacogenomic DSS (PGX-1/2/3) |
| trustedrisk-preadmit | 8781 | Pre-arrival triage (PREADMIT-1/2/3) |
| **trustedrisk-quality** | **8782** | **HEDIS / CMS Stars (STARS-1/2/3) -- Phase 10.1** |
| **trustedrisk-pophealth** | **8783** | **Population health + outbreak (POPHEALTH-1/2/3) -- Phase 10.2** |
| **trustedrisk-appeals** | **8784** | **Insurance appeals (APPEALS-1/2/3) -- Phase 10.3** |

The composer (port 8780) orchestrates multi-step workflows and ingests YAML-encoded recipes for batch chart review.

### 2.2 SHARP-on-MCP

We use the **SMART Health AI Reference Platform** (SHARP) header convention (`X-FHIR-Server-URL`, `X-FHIR-Access-Token`, `X-Patient-ID`) wrapped inside the **A2A v1 FHIR-context extension** (camelCase metadata: `fhirUrl`, `fhirToken`, `patientId`, `fhirRefreshToken?`, `fhirRefreshTokenUrl?`). A before-model hook reads the metadata, populates a shared `FHIRContext` ContextVar, and never lets credentials reach the LLM prompt.

### 2.3 Tool Surface

The 89 tools span 30 thematic bundles. The **Phase-10 expansion** (this submission) adds three new bundles:

- **`quality_stars`** -- `compute_quality_measures_aggregate` / `compute_stars_rating_forecast` / `compute_care_gap_priority_ranking` (HEDIS measures + CMS 5-Star benchmarks + QBP dollar projection + ranked care-gap interventions).
- **`population_health`** -- `compute_syndromic_surveillance` (Poisson z-score detection à la CDC NSSP/ESSENCE) / `compute_vaccine_reminder_cohort` (ACIP 2025 schedule, per-channel uptake) / `compute_outbreak_heatmap` (Laplace-mechanism DP heatmap suitable for public release).
- **`insurance_appeals`** -- `compute_denial_letter_parse` (regex + payer keyword tables) / `compute_appeal_letter_draft` (8-section template, mandatory cite-backs, optional LLM polish) / `compute_appeal_escalation_path` (ERISA + ACA + KFF priors).

---

## 3. Calibration

We treat readmission risk as a **5-bin Beta-Binomial** (Spec-002) with priors elicited from CMS HRRP literature. The calibration target is **expected calibration error (ECE)** at the bin-level:

> ECE = Σ_b (n_b / N) · |avg_p_b − avg_y_b|

**Results.**

| Cohort | N | ECE | Brier | AUROC |
|---|---|---|---|---|
| W1 calibration (run-1777055367773-4204d20e) | 7,880 | **0.0078** | 0.124 | 0.590 |
| Synthea-10k validation | 10,000 | **0.0056** | 0.118 | 0.601 |

The 10k validation set exercises the runtime path end-to-end against the in-memory tool surface; the calibration fidelity carries within sampling noise.

---

## 4. Safety Pipeline

Every recommendation passes through five gates:

1. **Deterministic safety floor.** A pure-Python rule (e.g., NEWS2 ≥ 7 -> urgent_review_1h) computes the recommendation independently of the LLM. The LLM may only **paraphrase** within structured slots that have a canonical input -> output mapping.
2. **Abstain trigger.** When uncertainty (CI95 width) or demographic-bias risk exceeds a configurable threshold, the agent returns `abstain_recommended=true` with a `reason` field rather than a forced recommendation.
3. **4-critic ensemble.** The candidate DecisionCard is reviewed by four critics -- clinical-safety, fairness, evidence, and LLM-judge -- each of which can downgrade confidence, force abstain, or request a bounded plan revision. A 5th constitutional critic enforces the platform's policy (no self-attribution as a doctor, no PHI surfacing).
4. **Memory & drift.** Prior cards for the same patient are recalled from a SQLite memory store; `detect_recommendation_drift` flags substantive changes vs the prior decision when the model version, calibration version, or LACE components differ.
5. **Temporal validity.** Each `RiskEstimate.valid_until` scales with severity (high LACE -> 4 h, moderate -> 12 h, low -> 24 h).

---

## 5. Adversarial Robustness -- v2 Red-Team Campaign

We extended the SAFE-1/2/3/4 v1 utilities (`adversarial_fragility`, `ood_detector`, `pentest_suite`, `chaos_run`) with a **structured 110-prompt v2 corpus** spanning ten attack categories:

| Category | N | Pass rate (detect_phi target) |
|---|---|---|
| prompt_injection | 14 | 1.000 |
| phi_exfiltration | 12 | 1.000 |
| jailbreak | 12 | 1.000 |
| hallucination_trigger | 12 | 1.000 |
| citation_fabrication | 10 | 1.000 |
| bias_probe | 12 | 1.000 |
| ood_input | 10 | 1.000 |
| multilingual_evasion | 8 | 1.000 |
| encoding_obfuscation | 8 | 1.000 |
| tool_misuse | 12 | 1.000 |
| **Total** | **110** | **1.000** |

**Scoring semantics.** A `redact` expected outcome passes when forbidden substrings appear *only* inside a structured redaction surface (e.g., `PHIReport.redaction_map` keys), proving the tool detected and offered to redact rather than echoed in plain rationale text.

The campaign report is auto-generated to `docs/adversarial/RED_TEAM_RESULTS.md`.

---

## 6. Fairness, Equity, and Differential Privacy

The fairness audit decomposes the cohort by `race × sex × insurance × age_band` and reports per-subgroup intervention / abstain / risk rates. Subgroup multipliers are calibrated against the W5 internal calibration workflow output and fall back to literature defaults (HRRP / HCUP / JAMA) when the calibrated artifact is absent.

The **equity dashboard** publishes those subgroup statistics under **ε-Laplace differential privacy**:

> Pr[A(D) ∈ S] ≤ e^ε · Pr[A(D′) ∈ S]

with sensitivity 1 (per-individual count contribution). The Phase-10.2 `compute_outbreak_heatmap` re-uses the same Laplace mechanism for public release of geographic × syndrome counts.

---

## 7. Counterfactual Explanation & Right-to-Explanation

For every DecisionCard the agent surfaces:

- A **per-LACE-component what-if sweep** that shows how each of L, A, C, E contributes to the predicted risk.
- The **minimum-modification path** that would flip the recommendation tier.
- A **patient-friendly audit summary** on demand (GDPR Art. 22 / EU AI Act Art. 13 right-to-explanation).

A 5th constitutional critic enforces that every clinical claim in the rationale either cites a structured chart excerpt (FHIR resource id) or a curated guideline citation (PubMed PMID, ClinicalTrials.gov NCT, CMS rule).

---

## 8. Audit Trail (RFC 6962 Merkle Chain)

We persist an append-only HIPAA-style audit log with PHI-redacted SHA-256 hashing. The log entries form an **RFC 6962 Merkle audit chain** -- an external auditor can verify any historical decision is byte-identical to its replay without re-running the model. Combined with the **temporal-validity** field, this gives the platform a defensible chain-of-custody for any decision that downstream led to clinical harm.

---

## 9. HEDIS / CMS Stars Rating + QBP Forecast

Phase 10.1 introduces the **`quality_stars`** bundle, addressing the $4-6 B/yr Quality Bonus Payment (QBP) pain-point for Medicare Advantage contracts. The bundle ships:

- **20 hardcoded HEDIS measures** (BCS, CCS, COL, CDC-EYE, CDC-HBA1C, CBP, MPM-ACE/DIURETIC/ANTICONV, OMW, PCR, FUH, FUM, AAB, FMC, MRP, CWP, DAE, TRC, SUPD) with FY 2025 5/4/3-star CMS benchmark cuts and per-measure weights.
- A **Stars Rating forecast** that projects per-domain (preventive_care + chronic_conditions) and overall Stars at measurement-period end and estimates QBP dollars (Avalere 2024 -- 4.0 unlocks ~5% benchmark, 4.5+ doubles).
- A **care-gap priority ranking** that orders interventions by `expected_stars_lift × difficulty_factor` and emits per-measure recommendations (e.g., "Pharmacist-led MTM + insulin titration protocol" for CDC-HBA1C).

The bundle is **pure-deterministic** -- no LLM in the floor; every numeric output is reproducible given the same cohort summary.

---

## 10. Insurance Appeals Pipeline

Phase 10.3 introduces the **`insurance_appeals`** bundle, addressing the 10-15% US claim-denial rate (KFF 2024) and the 0.2% appeal rate on internal denials despite ~50% internal-appeal success (KFF Marketplace Plan 2023). The pipeline:

1. **`compute_denial_letter_parse`** -- regex parser that detects payer (UnitedHealthcare / Anthem / Aetna / Cigna / Humana / Medicare / Medicaid / generic), claim id, deadlines, contested dollar amount, partial-denial flag, and per-reason category (medical_necessity / step_therapy_not_met / out_of_network / experimental_unproven / missing_documentation / duplicate_service / non_covered_benefit / claim_filing_limit / other).
2. **`compute_appeal_letter_draft`** -- 8-section template (header / patient summary / denial summary / medical-necessity argument / supporting evidence / policy counter-argument / alternative proposed / closing) with mandatory cite-backs to FHIR resource ids or PubMed PMIDs. An optional LLM-polish flag lets a downstream module paraphrase prose without inventing clinical facts.
3. **`compute_appeal_escalation_path`** -- payer-specific deadline + success-probability path through `internal_first_level -> internal_second_level -> external_independent_review -> state_insurance_commissioner` using ERISA, ACA §2719 (45 CFR 147.136), and KFF 2024 priors (~39% first-level overturn rate, ~40% IRO overturn rate).

---

## 11. Synthea + HAPI 1k Integration

Phase 10.5 ships a **deterministic FHIR R4 Bundle generator** (`src/a2a_agent/synthea_bundles.py`) that emits 1,000 idempotent transaction Bundles with PUT-by-identifier semantics (`system = https://trustedrisk.local/synthea-id`). Each bundle contains 1 Patient + 1 Encounter + 1-3 Conditions (ICD-10) + 3-6 Observations (LOINC) + 0-4 MedicationRequests (RxNorm). The loader (`scripts/synthea_to_hapi_load.py`) performs idempotent upsert against the public HAPI test server when `TRUSTEDRISK_LIVE_FHIR=1`. The 1k generation completes in **0.04 s** on the reference machine.

---

## 12. Reproducibility & Testing

**As of 2026-04-30 05:53 UTC: 3,117 tests pass, 35 skipped.**

| Test layer | N | Coverage |
|---|---|---|
| Unit | 2,400+ | Per-tool clinical logic, schema validation |
| Property-based (Hypothesis) | 200+ | Fuzz across ICD-10, LOINC, RxNorm, demographic axes |
| Adversarial | 130+ | Boundary inputs + prompt-injection corpus + v2 red-team |
| Integration | 200+ | A2A v1 compliance, federation handshake, Synthea generator |
| Functional | 100+ | End-to-end recipes, golden cohort cases |
| Live (opt-in) | 20+ | HAPI FHIR R4 against `https://hapi.fhir.org/baseR4` |

Calibration is fully reproducible: the `data/coefficients.json` artifact is keyed by a calibration workflow run id (`run-1777055367773-4204d20e`) and the W1 calibration cohort SHA-256 is recorded in `docs/CALIBRATION_PROVENANCE.md`.

---

## 13. Limitations and Future Work

- **Calibration drift.** The W1 calibration cohort is small (7,880 rows). A larger, multi-site calibration is the obvious next priority and is the subject of an upcoming calibration workflow re-run on Synthea-50k.
- **HEDIS coverage.** We ship 20 of the ~90 HEDIS measures. Adding the remaining ones is a curation task, not a research one.
- **LLM-judge variance.** The 4-critic ensemble's LLM-judge is deterministic with a pass-through floor in CI; in production deployments where a real model is wired in, the judge's variance must be re-bounded via a temperature-0 / nucleus-cap config.
- **Conformal prediction.** Phase 10.9 introduces split conformal calibration with marginal coverage guarantees; the paper-side analysis of the resulting prediction sets vs the existing CI95 bands is left as future work.
- **Tool-use planner.** Phase 10.8 introduces an LLM-driven router with a deterministic floor; the empirical comparison vs the existing recipe-based composer (latency, accuracy, abstain rate) is in flight.

---

## 14. Conclusion

TrustedRisk demonstrates that calibrated, safety-gated, audit-trailed clinical decision support is achievable today, against real FHIR servers, with a published prompt-injection robustness profile and a fully reproducible test surface. The system is **internal-tooling-free**: every line of the deliverable is independently testable against schemas under `src/shared/schemas.py` without invoking a model. The 14-specialist federation, 89-tool MCP backend, and A2A v1 FHIR-context extension wiring make TrustedRisk a candidate reference implementation for the SHARP-on-MCP pattern.

---

## References

1. SMART Health IT. SMART App Launch Framework v2.0.0 (2024). https://hl7.org/fhir/smart-app-launch
2. A2A v1 FHIR-Context Extension. https://docs.promptopinion.ai/fhir-context/a2a-fhir-context.html
3. Anthropic. Model Context Protocol (MCP). https://github.com/modelcontextprotocol/specification
4. NCQA. HEDIS Volume 2: Technical Specifications (2024).
5. CMS. 2025 Star Ratings Technical Notes.
6. Avalere Health. CMS Quality Bonus Payment Analysis (2024).
7. Kaiser Family Foundation. Marketplace Plan Denial Rates (2024).
8. American Medical Association. Prior Authorization Physician Survey (2024).
9. CDC NSSP / ESSENCE -- National Syndromic Surveillance Program.
10. ACIP. Recommended Adult / Pediatric Immunization Schedules (2025).
11. Dwork C, McSherry F, Nissim K, Smith A. Calibrating Noise to Sensitivity in Private Data Analysis. TCC 2006.
12. Vovk V, Gammerman A, Shafer G. Algorithmic Learning in a Random World. 2nd ed. Springer (2022).
13. Greshake K et al. Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. arXiv:2302.12173 (2023).
14. OWASP. LLM Top 10 (2024).
15. HHS. HIPAA Privacy Rule (45 CFR §§ 164.500-534).
16. Laming RS, Kanu C et al. CONSORT extension for N-of-1 trials (CENT). BMJ 2015;350:h1738.
17. ERISA §502(a) appeal rules (29 CFR 2560.503-1).
18. ACA §2719 External Review (45 CFR 147.136).

---

**Reproducibility statement.** All numbers in this paper are produced by the test suite and the validation scripts shipped in this repository. The Synthea-10k validation runs in `scripts/synthea_10k_validation.py`, the v2 red-team campaign runs in `scripts/run_redteam_v2.py`, and the HAPI Synthea-1k integration runs in `scripts/synthea_to_hapi_load.py`. ECE / AUROC numbers are recomputed on every CI run.
