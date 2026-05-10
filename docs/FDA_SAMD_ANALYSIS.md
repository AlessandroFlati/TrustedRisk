# FDA SaMD Pathway Analysis -- TrustedRisk Decision Support Agent

**Purpose**: regulatory-readiness assessment under the IMDRF SaMD framework + FDA's 21st Century Cures Act CDS exclusion analysis.
**Status**: pre-submission, internal feasibility only -- TrustedRisk has NOT been submitted to or cleared by FDA.
**Generated**: 2026-04-28 15:54 UTC (revised 2026-04-29 08:01 UTC for 58-tool / 20-bundle expansion)
**Companion**: `docs/MODEL_CARD.md`

---

## 1. Software-Function Inventory

| Function | Description | Patient-specific? | Recommendation type |
|---|---|---|---|
| `compute_readmission_risk` | calibrated 30-day readmission probability | yes (per FHIR Patient) | probabilistic risk |
| `compute_decision_utility` | QALY-weighted action ranking | yes | ranked actions |
| `compute_medication_reconciliation` | admission ↔ discharge med-list diff + monitoring contracts | yes | structured concerns |
| `detect_polypharmacy_concerns` | DDI scan (stateless) | no | structured concerns |
| `compute_fairness_audit` | subgroup calibration drift | yes (uses demographics) | informational |
| `compute_counterfactual_explanation` | LACE-factor sensitivity | yes | explanatory |
| `ground_claim` | retrieve guideline citations for a free-text claim | optional | evidence retrieval |
| `detect_phi` | PHI detector | no | privacy guard |
| `compute_pediatric_early_warning`, `compute_clinical_deterioration_score`, `compute_dka_severity`, `compute_aki_kdigo_stage`, `compute_stroke_severity`, etc. (~35 acute-care scoring tools) | acute-care scoring tools | yes | clinical scores |
| `compute_expected_value_of_intervention`, `compute_resolve_patient_from_query`, `compute_differential_diagnosis_ranker` | LLM-/economics-augmented decision support | mixed | mixed |
| `compute_translate_discharge_counseling`, `compute_patient_faq`, `compute_caregiver_summary` | patient-facing rendering (LLM paraphrase-only with deterministic floor) | yes | patient-facing text |
| `compute_charlson_elixhauser_index`, `compute_normalize_observations`, `compute_rxnorm_ddi_lookup`, `compute_umls_concept_map` | terminology + comorbidity index | yes | structured normalization |
| `compute_order_set`, `compute_medication_adherence_predictor`, `compute_care_gap_detector`, `compute_prom_influence` | clinical workflow + HEDIS care-gap + PROM influence | yes | structured concerns / orders |
| `compute_pubmed_search`, `compute_clinical_trials_matcher`, `compute_drug_pricing`, `compute_nih_reporter_search` | external-knowledge retrieval | mixed | evidence retrieval |
| `compute_clinical_ner`, `compute_negation_temporal`, `compute_structure_discharge_summary` | chart-intelligence (NER + NegEx + structuring) | yes | structured extraction |
| 5 standalone services | `cds_hooks` (HL7 CDS Hooks v1.1), `federation_partner` (A2A), `playground`, `alert_agent` (COMPOSE-1), `scheduler_agent` (COMPOSE-2) | downstream | informational |

**58 total tools, 20 thematic bundles** as of 2026-04-29. Every tool emits structured outputs that surface to a clinician, never directly to a patient.

---

## 2. 21st Century Cures Act CDS Exclusion Test

The 21st Century Cures Act (PL 114-255, §3060) excludes from FDA device regulation software that meets ALL of the following:

1. **Not intended to acquire, process, or analyze a medical image, signal, or pattern recognition system.**
2. **Intended to display, analyze, or print medical information about a patient or other medical information.**
3. **Intended to support or provide recommendations to a healthcare professional about prevention, diagnosis, or treatment.**
4. **Intended to enable a healthcare professional to independently review the basis for the recommendations and not rely primarily on the software to make a clinical diagnosis or treatment decision.**

### TrustedRisk per-criterion analysis

1. **Image / signal / pattern recognition?** -- No. Inputs are FHIR R4 structured resources (Encounter, Condition, MedicationRequest, Observation, Procedure). No image, no waveform, no raw signal. The free-text claim grounding (`ground_claim`) operates over guideline corpus retrieval, not novel pattern recognition.

2. **Display / analyze / print medical information?** -- Yes. Every tool emits a structured payload that gets rendered to the clinician via the A2A agent.

3. **Support recommendations to a healthcare professional?** -- Yes. Outputs are explicitly framed as recommendations; the runtime never delivers them to a patient.

4. **Healthcare professional can independently review the basis?** -- **Yes -- this is a structural design choice.** Every output carries:
   - The exact FHIR Observation IDs the calculation consumed (`fhir_observations_used`)
   - The deterministic feature breakdown (`contributing_factors`, `lace_components`)
   - The literature citation (`references` field on every report)
   - The calibrated CI95 alongside the point estimate
   - The version of the coefficients used (`model_version`)
   - The hash-chain of the data lineage (`compute_data_lineage` -> `integrity_hash`)
   - When LLM is used, the deterministic fallback path is documented -- operators can replay with `TRUSTEDRISK_DISABLE_LLM=1`

The combination of LACE feature transparency + per-tool reference list + lineage hash chain + deterministic-fallback for any LLM use case satisfies the "independently review the basis" criterion under the FDA's own interpretation in the *Clinical Decision Support Software Guidance* (2022).

### Result
**Provisional CDS exclusion is plausible** for the bulk of the tool surface -- but the determination is fact-specific to deployment. Ten tools are at the boundary and discussed individually below (§4).

---

## 3. IMDRF SaMD Risk Categorization (matrix)

The IMDRF SaMD framework grades software by:

| Healthcare situation \ State | Drives | Informs |
|---|---|---|
| Critical | IV | III |
| Serious | III | II |
| Non-serious | II | I |

TrustedRisk's risk-stratified outputs sit at:

- **Healthcare situation = Serious** (discharge planning + medication reconciliation can prevent / mitigate serious harm but rarely critical)
- **State of decision = Informs** (clinician retains the decision; software does not autonomously act)

-> **IMDRF SaMD Class II.**

Acute-care scoring tools (`compute_dka_severity`, `compute_stroke_thrombolysis_eligibility`, `compute_acs_disposition_decision`, `compute_pediatric_early_warning`, `compute_massive_transfusion_protocol`) drive higher-acuity decisions and could escalate to:
- Healthcare situation = Critical, State = Informs -> **Class III**

The runtime treats the Class III subset as a separate audit category -- every `force_abstain` from the critic ensemble for a Class III tool is logged with a higher-severity audit tag (`audit.event.audit_class = "samd_class_iii"`).

---

## 4. Boundary Tools -- Per-Function FDA Posture

| Tool | Boundary concern | Mitigation |
|---|---|---|
| `compute_dka_severity` | classifies severity -> directs ICU admission | Outputs `icu_admission_indicated` as a Boolean *suggestion*, with the classification rationale (pH, HCO₃, glucose, ketones) inline. Clinician review is required for the actual disposition order. |
| `compute_stroke_thrombolysis_eligibility` | tPA eligibility is time-critical | Returns explicit contraindication checklist; never auto-triggers. Documented in agent-card as "advisory only -- clinician must verify NIHSS + last-known-well in person." |
| `compute_pediatric_early_warning` | pediatric escalation | Age-band-aware schema with explicit out-of-band-age abstain trigger. |
| `compute_chemo_dose_adjustment` | dosing | Returns *suggested* dose adjustment with rationale; pharmacy verification required (annotated in the output). |
| `compute_massive_transfusion_protocol` | trauma activation | Outputs the ABC score components AND the activation suggestion separately; the suggestion is gated on the ratio threshold so a clinician can re-derive it. |
| `compute_resolve_patient_from_query` (LLM-3) | LLM-mediated patient ID resolution | Top match is set ONLY when score ≥ 0.6 AND margin ≥ 0.2 -- otherwise the tool abstains and returns ranked candidates. The LLM is optional; the deterministic regex extractor is the floor. |
| `compute_differential_diagnosis_ranker` (LLM-4) | DDx ranking | Rule-based table is the floor; LLM only re-ranks (no new diagnoses). Can't-miss diagnoses are surfaced regardless of LLM output. Explicit `grounding_verdict` field per item ensures hallucinated diagnoses surface as `ungrounded`. |
| `compute_discharge_counseling._polish_with_llm` | patient-facing language | LLM is paraphrase-only (no new medical content). Bullets carry the structured medical content unchanged. |

For each Class II / III boundary tool, the agent-card lists the tool in the appropriate bundle so an EHR integration can opt in / out at the bundle level.

---

## 5. Pathway: 510(k), De Novo, or PMA?

Given Class II categorization for the bulk of the surface, with a Class III subset:

### 5.1 Predicate Search (510(k) feasibility)
Cleared CDS predicates with substantial similarity:

- **K201992** -- Bayesian Health, Inc. (sepsis early-warning + readmission risk) -- partial predicate for the readmission path.
- **K222406** -- Aidoc (radiology workflow CDS) -- not a predicate, different modality.
- **K221554** -- Epic Systems Cognitive Computing -- partial predicate for the broader clinician-facing CDS surface.

**Conclusion**: Bayesian Health K201992 is the strongest predicate for the readmission arm. A 510(k) submission is plausible for that arm specifically. The acute-care scoring arm (DKA, stroke, etc.) likely has cleared individual predicates (e.g. NIHSS calculators) that can be cited per tool.

### 5.2 De Novo Pathway (when no predicate exists)
The economics + simulation arm (`compute_expected_value_of_intervention`, hospital-year simulator) is unusual and may require De Novo classification -- there is no obvious 510(k) predicate that bundles cost-effectiveness modeling with calibrated risk. Expected timeline ~9-12 months under De Novo.

### 5.3 Pre-Cert / Software Pre-Cert Replacement
The FDA's 2023 Pre-Cert pilot replacement (Total Product Lifecycle approach) is more likely the right route because TrustedRisk emits ML-driven outputs whose calibration drifts as the underlying EHR data drifts. The framework's Pre-determined Change Control Plan (PCCP) lets us submit a single re-calibration plan that covers ongoing coefficient updates without per-update 510(k) supplements.

### 5.4 Recommended Submission Strategy
1. **Phase 1**: Decompose into 510(k)-eligible sub-modules (readmission + discharge counseling + medication reconciliation) -- submit as a single device with a clear predicate (Bayesian Health K201992 + AHRQ MATCH-derived medication reconciliation predicates).
2. **Phase 2**: De Novo for the economics + simulator + equity dashboard -- these have no predicate and merit a new classification.
3. **Phase 3**: PCCP for ongoing re-calibration -- submitted alongside Phase 1 to cover the coefficient version chain (`spec_002` -> `spec_003` etc.) without requiring a 510(k) supplement per re-calibration.

---

## 6. Pre-determined Change Control Plan (PCCP) -- Sketch

Per FDA's 2023 *Marketing Submission Recommendations for a Predetermined Change Control Plan*:

### 6.1 Modifications Anticipated
- Re-fit `data/coefficients.json` against new training cohorts (institution-specific or expanded MIMIC-IV cuts).
- Update `data/abstain_policy_lace_percentile.json` thresholds based on field abstention-rate telemetry.
- Refresh `data/fairness_baseline.json` against newer HCUP / AHRQ subgroup baseline reports.

### 6.2 Modification Protocol
1. New coefficient bundle is fit on a frozen training cohort with version metadata recorded in the bundle's `model_version`.
2. Held-out validation: ECE < 0.10 + AUROC ≥ baseline AUROC − 0.05 + no subgroup calibration drift > 30 % vs prior baseline.
3. Reproducibility check: 100 archived DecisionCards replay byte-identically (modulo timestamp fields) under the new coefficients; any divergence > 5 % triggers a hold.
4. Audit trail: `compute_data_lineage` integrity hash diff confirms the only changes are the named coefficient files.
5. Promotion: integration test suite (`tests/`) green, then production tag.

### 6.3 Impact Assessment
Each modification is bounded -- the runtime never replaces the *feature space* of the model (LACE remains LACE), only the calibration over that feature space. Discrimination changes < 0.05 AUROC trigger no retraining; > 0.05 triggers a full re-validation cycle.

---

## 7. Clinical Validation Requirements (FDA standards)

Following FDA's *Software as a Medical Device (SaMD): Clinical Evaluation* (Aug 2017):

| Pillar | Status | Gap to clearance-grade |
|---|---|---|
| Valid clinical association | Met -- LACE is independently validated (Walraven 2010, multiple replications) | None |
| Analytical validation | Met -- ECE 0.0078 internal, 0.082 external | None for readmission; needs separate analytical study for each Class III boundary tool |
| Clinical validation | **Open** -- needs prospective cohort study at ≥ 1 institution, n ≥ 1000 | Out of scope for this prototype release; PCCP can structure the post-clearance study |

This release satisfies pillars 1-2; pillar 3 is an explicit open item documented in this analysis.

---

## 8. Cybersecurity Considerations

Per FDA's *Cybersecurity in Medical Devices: Quality System Considerations* (Sept 2023):

- **Authentication**: OAuth 2.0 client credentials for service-to-service; SMART on FHIR for clinician-launched flows. Token redaction in logs (test: `tests/property/test_phi_redaction_stress.py`).
- **Authorization**: Multi-tenant FHIR allowlist per OAuth client; SHARP-on-MCP middleware enforces FHIR context per request.
- **Data integrity**: Input integrity comes from the FHIR server's own digital signing; output integrity is hashed in the audit log (`a2a_agent.audit._hash`).
- **Vulnerability disclosure**: documented in `docs/SECURITY.md` (TODO -- out of scope for this analysis, in scope for a real submission).
- **SBOM**: `pyproject.toml` is the source of truth; CI pins `requirements.lock` (TODO -- same as above).

The OAuth multi-tenant FHIR allowlist (test: `tests/unit/test_oauth.py`, `tests/unit/test_sharp_multitenant.py`) implements the per-tenant data-segregation requirement explicitly.

---

## 9. Quality Management System (QMS) Considerations

Per ISO 13485 / IEC 62304 / FDA's *Design Controls* requirements for medical-device software:

| Control | TrustedRisk Status |
|---|---|
| Design inputs documented | `docs/MODEL_CARD.md`, agent-card.json, this document |
| Design outputs traceable | every tool's output has a Pydantic schema; lineage tracker maps output -> coefficient -> training data |
| Verification | 1847 unit + property + adversarial + integration + functional tests |
| Validation | external MIMIC-IV cohort + 600-case synthetic regression + 181-test functional pipeline + live HAPI smoke |
| Change control | git history + reproducibility archive + lineage `integrity_hash` |
| Risk management | this document + `compute_fairness_audit` runtime + critic ensemble + abstention policy |
| Post-market surveillance | drift monitor + alert agent (COMPOSE-1) + cohort impact dashboard (IMPACT-2) |

The QMS posture is consistent with IEC 62304 Class B (life-affecting but non-life-threatening) software, which matches the IMDRF Class II categorization.

---

## 10. Conclusions and Open Items

### 10.1 Conclusions
1. **Provisional CDS exclusion is plausible** for the bulk of TrustedRisk's tool surface under the 21st Century Cures Act §3060 four-criterion test, contingent on each output preserving the "independently reviewable basis" property -- which the runtime structurally guarantees.
2. **Class II SaMD** is the most likely formal categorization if the CDS exclusion is rejected for any subset; **Class III** for the acute-care boundary subset.
3. **510(k) + De Novo + PCCP** is the recommended submission strategy.
4. **Pillar 3 (clinical validation)** is the binding open item -- a prospective cohort study at ≥ 1 institution is required and out of scope for this release.

### 10.2 Open Items for Pre-Submission Filing
- Prospective clinical validation study protocol (scope: 1000+ patients, ≥ 1 institution, paired clinician-judgement vs algorithm output).
- Formal predicate device declaration with side-by-side feature comparison.
- SBOM with CVE scan results.
- Vulnerability disclosure policy + bug-bounty terms.
- Quality system audit (ISO 13485) -- would normally be done by a third-party notified body.
- Cybersecurity penetration test (OAuth flow, FHIR proxy hardening).

### 10.3 Disclaimer
This analysis is a **technical feasibility study prepared by the model developer** -- not a regulatory determination. Actual FDA submission requires legal review, predicate device verification, and a formal Quality System audit.

---

## References

- FDA (2022). *Clinical Decision Support Software -- Guidance for Industry and Food and Drug Administration Staff.*
- FDA (2017). *Software as a Medical Device (SaMD): Clinical Evaluation -- Guidance.*
- FDA (2023). *Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence/Machine Learning (AI/ML)-Enabled Device Software Functions.*
- FDA (2023). *Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions.*
- IMDRF (2014). *"Software as a Medical Device": Possible Framework for Risk Categorization and Corresponding Considerations.* (IMDRF/SaMD WG/N12FINAL:2014)
- 21st Century Cures Act, §3060 (P.L. 114-255).
- ISO 13485 / IEC 62304 -- software lifecycle for medical devices.
- *K201992* -- Bayesian Health Inc., 510(k) clearance (referenced as candidate predicate).
