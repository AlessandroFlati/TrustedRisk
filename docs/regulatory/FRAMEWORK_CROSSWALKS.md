# TrustedRisk - NIST AI RMF + OECD AI Principles crosswalks

**Generated**: 2026-05-10 10:35 UTC - **Total rows**: 19 (NIST: 14, OECD: 5)

## NIST AI RMF 1.0 (Jan 2023)

| Section | Requirement | TrustedRisk evidence | Artefact |
| --- | --- | --- | --- |
| GOVERN 1.1 | Legal + regulatory requirements involving AI are understood, managed, and documented. | Regulatory pack maps EU AI Act, FDA 510(k), ISO 13485, GDPR, HIPAA in a single 14-section document. | `docs/regulatory/REGULATORY_PACK.md` |
| GOVERN 2.1 | Roles, responsibilities, lines of communication are documented + clear to all personnel. | ISO 13485 QMS section of the regulatory pack; the audit log carries the operator + clinician role on every decision record. | `docs/regulatory/REGULATORY_PACK.md` |
| GOVERN 3.2 | Policies + procedures address oversight of AI system development + deployment. | Phase 14.16 multi-agent debate + Phase 14.15 planner-revision feedback loop; both ship with deterministic-floor critics. | `src/a2a_agent/{multi_agent_debate,planner_revision}.py` |
| GOVERN 4.1 | Organizational practices + accountability structures address AI risks. | Append-only audit + RFC 6962 Merkle tree + reproducibility SQLite archive. | `src/a2a_agent/merkle_audit.py` |
| MAP 1.1 | Intended purposes, potentially beneficial uses, context-specific laws, norms, expectations are documented. | Model Card section 1+2 (Mitchell 2019 format); datasheet motivation section. | `docs/research/MODEL_CARD.md` |
| MAP 1.2 | Inter-disciplinary AI actors are involved in assessing risks and impacts. | 4-critic ensemble (clinical_safety, fairness, evidence, llm_judge) + 3-agent debate (clinical_conservative, evidence_aggressive, fairness_guard) span clinical, statistical, ethical, ML perspectives. | `src/a2a_agent/{critique,multi_agent_debate}.py` |
| MAP 2.3 | Scientific integrity + Test, Evaluation, Verification + Validation (TEVV) considerations are documented. | Property-based tests, golden cohorts, adversarial v2/v3/v4 corpus, prospective synthetic eval at n=10k. | `tests/{property,golden,adversarial}/ + docs/prospective/PROSPECTIVE_EVAL.md` |
| MAP 5.1 | Likelihood + magnitude of impacts on individuals + society are characterised. | Subgroup audit with EOO + DP gaps; differential-privacy publication of subgroup rates (Laplace, epsilon=1.0). | `docs/fairness/subgroup_audit.json` |
| MEASURE 1.3 | Metrics + benchmarks reflect the intended uses + context. | ECE preferred gate <=0.05; Brier; AUROC; per-subgroup TPR + DP gaps; conformal target coverage 1-alpha. | `docs/research/MODEL_CARD.md` |
| MEASURE 2.7 | Trustworthy characteristics (validity, safety, fairness) are measured. | Phase 16.F2 split-conformal multi-class + Pleiss 2017 calibration tension witness + selective-classification risk-coverage curve. | `src/a2a_agent/trustworthy_ml.py` |
| MEASURE 3.2 | Risk + benefit measurements are repeated + compared over time. | Drift detection scheduler runs at configurable cadence; abstain-by-default on drift. | `src/a2a_agent/{scheduler,drift_monitor}.py` |
| MANAGE 1.2 | Treatment of AI risks is informed by ranked + documented risks. | Per-issue planner critique (Phase 14.15) emits ranked + suggested-action issue list before any tool runs. | `src/a2a_agent/planner_revision.py` |
| MANAGE 2.4 | Mechanisms are in place to deactivate, disengage, or override the AI system. | Stop-button per Article 14(4)(d) - federation exposes /admin/quiesce + abstain-trigger on uncertainty + clinician override surface. | `docs/regulatory/REGULATORY_PACK.md` |
| MANAGE 4.1 | Post-deployment AI system monitoring plans are implemented. | Background scheduler runs care_gap_sweep + drift_detection; recommendation-drift tool flags per-patient changes. | `src/a2a_agent/scheduler.py` |

## OECD AI Principles (May 2019)

| Section | Requirement | TrustedRisk evidence | Artefact |
| --- | --- | --- | --- |
| Principle 1.1 - Inclusive growth | AI should benefit people + planet by driving inclusive growth, sustainable development, well-being. | Subgroup audit + fairness-guard veto narrow the predicted-impact gap on under-served populations. | `docs/fairness/subgroup_audit.json` |
| Principle 1.2 - Human-centred values | AI systems should respect rule of law, human rights, democratic values, diversity. | GDPR Art. 22 right-to-explanation surface + patient audit summary on demand + multi-language patient-facing translation. | `src/a2a_agent/right_to_explanation.py` |
| Principle 1.3 - Transparency | Stakeholders should understand AI-based outcomes + challenge them. | Counterfactual explanations + SHAP attributions + multi-agent debate vote breakdown surfaced on every DecisionCard. | `src/mcp_server/tools/{counterfactual_explanation,model_research}.py` |
| Principle 1.4 - Robustness + safety | AI systems should be robust, secure, safe throughout their entire lifecycle. | Red-team v3 multi-target + v4 indirect prompt injection corpus; OOD detection (Mahalanobis chi^2); chaos-engineering harness. | `src/a2a_agent/{redteam_v3,safety_redteam}.py` |
| Principle 1.5 - Accountability | Organizations + individuals developing or deploying AI should be accountable. | Append-only HIPAA audit + RFC 6962 Merkle chain + data-lineage hash chain + reproducibility SQLite archive. | `src/a2a_agent/{audit,merkle_audit,data_lineage}.py` |
