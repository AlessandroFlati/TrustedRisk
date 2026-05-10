# TrustedRisk - a2a_agent Module Catalog

**Modules**: 98 - **Modules with Pydantic models**: 42 - **Total Pydantic models**: 105 - **Total public functions**: 237 - **Total LoC (non-blank, non-comment)**: 24,056

## `a2a_agent.a2a_messaging`
_A2A v1 JSON-RPC messaging shared by every TrustedRisk agent endpoint._

- **LoC**: 418
- **Public functions** (4): `build_composer_handler`, `build_root_handler`, `build_specialist_handler`, `handle_jsonrpc`

## `a2a_agent.a2a_streaming`
_Phase 17.AJ - A2A streaming SSE + cancellation + resume._

- **LoC**: 119
- **Pydantic models** (1): `StreamEvent`
- **Public functions** (3): `collect_stream`, `render_sse`, `stream_decision_card`

## `a2a_agent.active_learning`
_Phase 17.AX - Active learning acquisition functions._

- **LoC**: 136
- **Pydantic models** (2): `ActiveLearningPick`, `ActiveLearningReport`
- **Public functions** (2): `select_top_k_by_committee`, `select_top_k_to_label`

## `a2a_agent.agent`
_TrustedRisk A2A agent composer -- Option C hybrid._

- **LoC**: 139
- **Public functions** (2): `root_agent`, `to_a2a_app`

## `a2a_agent.agent_card_normalizer`
_Normalize on-disk agent cards to the A2A v1 proto-schema served shape._

- **LoC**: 45
- **Public functions** (1): `normalize_agent_card`

## `a2a_agent.audit`
_HIPAA-style structured audit logging + reproducibility (AUDIT-1, AUDIT-2)._

- **LoC**: 305
- **Public functions** (7): `append_audit_event`, `archive_decision`, `check_reproducibility`, `fetch_archived_decision`, `list_archived_decisions`, `log_tool_call`, `read_audit_log`

## `a2a_agent.batch`
_Deterministic batch composer for POST /api/batch/decision-cards._

- **LoC**: 166
- **Public functions** (5): `compose_decision_for_patient`, `parse_batch_request`, `process_batch`, `recommend_action_from_risk`, `serialize_batch_response`

## `a2a_agent.bayes_net`
_Phase 17.AL - Bayesian network inference for patient state._

- **LoC**: 210
- **Pydantic models** (3): `BNNode`, `BayesianNetwork`, `InferenceResult`
- **Public functions** (4): `build_network`, `build_readmission_network`, `cpt_lookup`, `infer`

## `a2a_agent.bulk_fhir`
_Phase 16.H1 - Bulk FHIR $export consumer._

- **LoC**: 141
- **Pydantic models** (1): `BulkFhirSummary`
- **Public functions** (2): `consume_bulk_fhir_lines`, `consume_bulk_fhir_path`

## `a2a_agent.bundle_orchestrator`
_Bundle composition orchestrator (SAFE-2)._

- **LoC**: 252
- **Public functions** (1): `suggest_bundles`

## `a2a_agent.causal_depth`
_Phase 17.M - Causal + counterfactual depth._

- **LoC**: 315
- **Pydantic models** (4): `FrontDoorReport`, `RosenbaumSensitivityReport`, `WachterCounterfactualReport`, `WaldIVReport`
- **Public functions** (4): `find_min_distance_counterfactual`, `front_door_adjustment`, `rosenbaum_gamma_bound`, `wald_iv_estimate`

## `a2a_agent.causal_forest`
_Phase 17.AS - Causal forest for conditional ATE (Athey & Wager 2019)._

- **LoC**: 235
- **Pydantic models** (2): `CausalForest`, `CausalForestReport`
- **Public functions** (4): `fit_causal_forest`, `overall_ate`, `predict_cate`, `report_causal_forest`

## `a2a_agent.causal_inference`
_SCALE-2 -- Causal-inference Average Treatment Effect via DoWhy._

- **LoC**: 236
- **Public functions** (1): `compute_average_treatment_effect`

## `a2a_agent.cds_hooks_card`
_Phase 16.H3 - CDS Hooks v1.1 Card builder._

- **LoC**: 169
- **Pydantic models** (6): `CDSHooksAction`, `CDSHooksCard`, `CDSHooksLink`, `CDSHooksOverrideReason`, `CDSHooksResponse`, `CDSHooksSuggestion`
- **Public functions** (2): `build_decision_card`, `build_response`

## `a2a_agent.clinical_scenarios`
_Phase 11.6 -- 5 end-to-end clinical scenario runners._

- **LoC**: 502
- **Public functions** (8): `list_scenarios`, `run_all_scenarios`, `run_scenario`, `scenario_acute_stroke`, `scenario_geriatric_polypharmacy`, `scenario_mental_health_crisis`, `scenario_polytrauma_mtp`, `scenario_sepsis_bundle`

## `a2a_agent.coin`
_COIN -- Conversational Interoperability between A2A agents._

- **LoC**: 308
- **Public functions** (2): `dialog_with_partner`, `match_skill_to_prompt`

## `a2a_agent.concept_drift`
_Phase 17.AT - Concept-drift online detection._

- **LoC**: 193
- **Pydantic models** (2): `ADWINReport`, `DDMReport`
- **Public functions** (2): `detect_drifts_adwin`, `detect_drifts_ddm`

## `a2a_agent.conformal`
_Phase 10.9 -- Split conformal prediction with marginal coverage._

- **LoC**: 166
- **Public functions** (3): `fit_split_conformal`, `predict_interval`, `predict_set`

## `a2a_agent.conversational_orchestrator`
_GENAI-5 -- Conversational orchestrator across the 4-agent registry._

- **LoC**: 430
- **Public functions** (2): `execute_workflow`, `plan_workflow`

## `a2a_agent.cost_effectiveness_ladder`
_SIM-3 -- Cost-effectiveness ladder vs published literature benchmarks._

- **LoC**: 149
- **Public functions** (1): `build_cost_effectiveness_ladder`

## `a2a_agent.cost_simulator`
_Phase 17.AA - Cost simulator vs LACE-only baseline._

- **LoC**: 183
- **Pydantic models** (2): `CostBreakdown`, `CostSimulationReport`
- **Public functions** (1): `simulate_cost_impact`

## `a2a_agent.counterfactual_fairness`
_Phase 17.AR - Counterfactual fairness audit (Kusner 2017)._

- **LoC**: 173
- **Pydantic models** (3): `CFFairnessFinding`, `CFFairnessReport`, `StructuralCausalModel`
- **Public functions** (2): `audit_counterfactual_fairness`, `flip_protected_attribute`

## `a2a_agent.cqr`
_Phase 17.AY - Conformalised quantile regression (Romano 2019)._

- **LoC**: 180
- **Pydantic models** (2): `CQRReport`, `LinearQuantileModel`
- **Public functions** (3): `fit_cqr`, `fit_linear_quantile`, `predict_quantile`

## `a2a_agent.critique`
_Self-critique stage applied to a candidate DecisionCard._

- **LoC**: 289
- **Public functions** (3): `aggregate_critic_verdicts`, `apply_critique`, `collapse_ensemble_to_single_critique`

## `a2a_agent.data_lineage`
_CMP-3 -- Data lineage tracker._

- **LoC**: 236
- **Public functions** (1): `compute_data_lineage`

## `a2a_agent.decision_card_pdf`
_Phase 14.10 N1 -- PDF export of DecisionCard._

- **LoC**: 152
- **Public functions** (1): `render_decision_card_pdf`

## `a2a_agent.deterioration_nowcast`
_TIME-3 -- Real-time deterioration nowcasting._

- **LoC**: 158
- **Public functions** (1): `nowcast_deterioration`

## `a2a_agent.dicom_sr`
_Phase 16.H3 - DICOM SR (Structured Report) parser._

- **LoC**: 263
- **Pydantic models** (2): `DICOMSRItem`, `DICOMSRReport`
- **Public functions** (2): `build_minimal_dicom_sr_bytes`, `parse_dicom_sr_bytes`

## `a2a_agent.distribution_shift`
_Phase 17.N - Distribution shift formalism._

- **LoC**: 241
- **Pydantic models** (3): `BBSELabelShiftReport`, `EnergyOODReport`, `KSTestReport`
- **Public functions** (3): `energy_based_ood_score`, `lipton_bbse_label_shift`, `two_sample_ks_test`

## `a2a_agent.doubly_robust`
_Phase 17.AU - Doubly-robust ATE estimators._

- **LoC**: 102
- **Pydantic models** (1): `DoublyRobustReport`
- **Public functions** (1): `estimate_ate_doubly_robust`

## `a2a_agent.dp_equity`
_AUDIT-2 -- Differential privacy on the population equity dashboard._

- **LoC**: 105
- **Public functions** (2): `compute_dp_equity_dashboard`, `math_log`

## `a2a_agent.drift_monitor`
_Calibration drift monitor (SCI-3)._

- **LoC**: 312
- **Public functions** (1): `compute_drift_report`

## `a2a_agent.ehr_integrations`
_EHR-1/2/3 -- Legacy-system integrations._

- **LoC**: 494
- **Public functions** (3): `compute_sdoh_score`, `parse_ccda_document`, `parse_hl7v2_adt`

## `a2a_agent.equity_dashboard`
_SIM-2 -- Population equity dashboard._

- **LoC**: 190
- **Public functions** (1): `compute_equity_dashboard`

## `a2a_agent.eval_quality`
_EVAL-1/2/3/4 -- Quality benchmarks and comparative-effectiveness._

- **LoC**: 597
- **Public functions** (4): `compute_comparative_effectiveness`, `compute_hedis_score`, `compute_hospital_compare_benchmark`, `compute_schwartz_quality_score`

## `a2a_agent.event_sourced_log`
_TIME-1 -- Event-sourced patient state log._

- **LoC**: 194
- **Public functions** (4): `append_event`, `list_events`, `replay_to_state`, `truncate_log`

## `a2a_agent.explainability`
_Phase 7.4 -- Explainability depth._

- **LoC**: 488
- **Public functions** (4): `compute_ale_plot`, `compute_concept_bottleneck_rationale`, `compute_constitutional_critic_check`, `compute_lace_shap_attribution`

## `a2a_agent.fairness_advanced`
_Phase 13.2 I1 -- Equalized Odds + Demographic Parity audit._

- **LoC**: 257
- **Public functions** (1): `compute_fairness_advanced`

## `a2a_agent.federated_learning`
_Phase 17.AN - Federated Learning simulator._

- **LoC**: 459
- **Pydantic models** (6): `BBPosterior`, `FederatedRound`, `FederatedTrainingReport`, `PrivacyUtilityPoint`, `PrivacyUtilityReport`, `SiteCohort`
- **Public functions** (6): `add_laplace_noise`, `fedavg_aggregate`, `generate_site_cohort`, `local_update`, `privacy_utility_curve`, `run_federated_training`

## `a2a_agent.federation_chain`
_Phase 17.U - Federation chained-call orchestrator._

- **LoC**: 242
- **Pydantic models** (2): `ChainHop`, `FederationChainTrace`
- **Public functions** (1): `run_federation_chain`

## `a2a_agent.federation_registry`
_Phase 17.AI - Federation marketplace registry + bundle coverage audit._

- **LoC**: 219
- **Pydantic models** (2): `FederationRegistry`, `SpecialistEntry`
- **Public functions** (3): `build_federation_registry`, `render_marketplace_manifest`, `render_registry_md`

## `a2a_agent.fedprox`
_Phase 17.AO - FedProx for heterogeneous federated clients._

- **LoC**: 163
- **Pydantic models** (2): `FedProxRound`, `FedProxTrainingReport`
- **Public functions** (1): `run_fedprox_training`

## `a2a_agent.fgsm_attack`
_Phase 17.AK - FGSM-style adversarial attack._

- **LoC**: 206
- **Pydantic models** (2): `FGSMReport`, `RobustnessReport`
- **Public functions** (2): `adversarial_robustness_sweep`, `fgsm_attack`

## `a2a_agent.fine_gray`
_Phase 17.AQ - Survival analysis with competing risks (Fine-Gray)._

- **LoC**: 241
- **Pydantic models** (1): `FineGrayReport`
- **Public functions** (1): `fit_fine_gray`

## `a2a_agent.framework_crosswalks`
_Phase 16.G2 - NIST AI RMF 1.0 + OECD AI Principles 2019 crosswalks._

- **LoC**: 355
- **Pydantic models** (2): `CrosswalkRow`, `FrameworkCrosswalkReport`
- **Public functions** (2): `build_framework_crosswalks`, `render_crosswalks_md`

## `a2a_agent.guideline_crosswalk`
_Phase 17.P - Clinical guideline crosswalk for the 145-tool surface._

- **LoC**: 531
- **Pydantic models** (2): `GuidelineCrosswalkReport`, `GuidelineRow`
- **Public functions** (2): `build_guideline_crosswalk`, `render_guideline_crosswalk_md`

## `a2a_agent.hospital_year_simulator`
_Phase 17.AH - Monte Carlo hospital-year outcomes simulator._

- **LoC**: 108
- **Pydantic models** (2): `HospitalYearReport`, `TrajectoryStat`
- **Public functions** (1): `simulate_hospital_year`

## `a2a_agent.hrrp_benchmark`
_Phase 17.S - HRRP literature benchmark._

- **LoC**: 191
- **Pydantic models** (3): `HRRPBenchmarkReport`, `HRRPBin`, `HRRPSubgroupRow`
- **Public functions** (2): `benchmark_against_hrrp`, `render_hrrp_md`

## `a2a_agent.impact_aggregator`
_IMPACT-2 -- aggregate impact KPIs across a cohort of DecisionCards._

- **LoC**: 180
- **Public functions** (2): `aggregate_impact_kpis`, `load_archived_decisions`

## `a2a_agent.incremental_risk`
_TIME-2 -- Incremental risk recompute._

- **LoC**: 118
- **Public functions** (1): `recompute_with_observation`

## `a2a_agent.kalman_vitals`
_Phase 17.AF - Kalman filter for vitals time-series._

- **LoC**: 239
- **Pydantic models** (4): `Kalman1DReport`, `KalmanStep`, `MultiKalmanReport`, `MultiKalmanStep`
- **Public functions** (2): `kalman_filter_1d`, `kalman_filter_multivariate`

## `a2a_agent.llm_critic`
_GENAI-3 -- LLM-driven critic for the multi-critic ensemble._

- **LoC**: 171
- **Public functions** (1): `llm_judge_critic`

## `a2a_agent.llm_polish`
_Phase 6.2 -- Real LLM polish client._

- **LoC**: 395
- **Public functions** (4): `find_preserved_tokens`, `post_check_preserved_tokens`, `reset_client_cache`, `resolve_polish_client`

## `a2a_agent.local_explainer`
_Phase 17.AG - Anchors + LIME local explanations._

- **LoC**: 240
- **Pydantic models** (3): `AnchorReport`, `LIMEAttribution`, `LIMEReport`
- **Public functions** (2): `anchors_explain`, `lime_explain`

## `a2a_agent.mdp_decision`
_SCALE-3 -- Sequential decision modeling via finite-horizon MDP._

- **LoC**: 221
- **Public functions** (1): `compute_sequential_mdp_value`

## `a2a_agent.medqa_eval`
_Phase 11.8 -- MedQA-USMLE-style eval harness._

- **LoC**: 617
- **Public functions** (3): `list_bench`, `main`, `run_medqa`

## `a2a_agent.memory`
_Patient-history memory layer._

- **LoC**: 308
- **Public functions** (1): `detect_recommendation_drift`

## `a2a_agent.merkle_audit`
_AUDIT-1 -- Merkle audit chain._

- **LoC**: 219
- **Public functions** (4): `build_inclusion_proof`, `compute_merkle_audit_root`, `verify_audit_chain`, `verify_inclusion`

## `a2a_agent.milp_staffing`
_Phase 17.AV - Mixed-integer program for nurse-staffing._

- **LoC**: 192
- **Pydantic models** (1): `MILPStaffingPlan`
- **Public functions** (1): `solve_milp_nurse_staffing`

## `a2a_agent.model_card`
_Phase 16.F1 - Model Card (Mitchell 2019) + Datasheet (Gebru 2021)._

- **LoC**: 386
- **Pydantic models** (4): `Datasheet`, `DatasheetSection`, `ModelCard`, `ModelCardSection`
- **Public functions** (4): `build_datasheet`, `build_model_card`, `render_datasheet_md`, `render_model_card_md`

## `a2a_agent.multi_agent_debate`
_Phase 14.16 Q1 -- Multi-agent debate / round-table._

- **LoC**: 220
- **Pydantic models** (3): `DebateInput`, `DebateOutcome`, `DebateVote`
- **Public functions** (1): `run_debate`

## `a2a_agent.neural_ode`
_Phase 17.AW - Neural-ODE-style trajectory model._

- **LoC**: 155
- **Pydantic models** (3): `NODEParams`, `NODETrajectoryReport`, `TrajectoryStep`
- **Public functions** (5): `euler_step`, `init_node_params`, `integrate_trajectory`, `rk4_step`, `vector_field`

## `a2a_agent.notification_formatter`
_PATIENT-4 -- Multi-channel notification formatter._

- **LoC**: 169
- **Public functions** (1): `format_discharge_notifications`

## `a2a_agent.observability`
_Phase 6.3 -- Observability primitives (Prometheus + OpenTelemetry)._

- **LoC**: 211
- **Public functions** (5): `default_rate_limiter`, `maybe_init_otel`, `rate_key_for_request`, `render_prometheus_metrics`, `time_tool`

## `a2a_agent.omop_cdm`
_Phase 16.H2 - OMOP CDM v5.4 exporter._

- **LoC**: 217
- **Pydantic models** (2): `OMOPExportReport`, `OMOPRow`
- **Public functions** (1): `export_to_omop`

## `a2a_agent.openapi_generator`
_Phase 17.V - OpenAPI 3.1 spec generator._

- **LoC**: 384
- **Pydantic models** (1): `OpenAPISpec`
- **Public functions** (1): `build_openapi`

## `a2a_agent.openapi_surface`
_Phase 12.9 A2 -- OpenAPI 3.1 surface for the federation._

- **LoC**: 224
- **Public functions** (3): `build_openapi_spec`, `make_docs_route`, `make_openapi_route`

## `a2a_agent.outcomes_simulator`
_SIM-1 -- Monte Carlo hospital outcomes simulator._

- **LoC**: 253
- **Public functions** (1): `simulate_hospital_year`

## `a2a_agent.patient_advocate`
_Phase 16.J1 - Patient-advocate A2A specialist._

- **LoC**: 223
- **Pydantic models** (3): `AxisFinding`, `PatientAdvocateInput`, `SecondOpinionCard`
- **Public functions** (1): `evaluate_patient_advocate`

## `a2a_agent.patient_audit_summary`
_AUDIT-4 -- Patient-facing audit summary._

- **LoC**: 154
- **Public functions** (1): `compute_patient_audit_summary`

## `a2a_agent.plan_revision`
_Plan revision orchestrator (AMB-5.2)._

- **LoC**: 119
- **Public functions** (1): `run_with_plan_revision`

## `a2a_agent.planner`
_Phase 10.8 -- Tool-use planner brain._

- **LoC**: 378
- **Public functions** (1): `plan_tool_use`

## `a2a_agent.planner_revision`
_Phase 14.15 Q2 -- Planner revision feedback loop._

- **LoC**: 282
- **Pydantic models** (2): `PlannerCritique`, `PlannerIssue`
- **Public functions** (3): `critique_plan`, `revise_plan`, `run_with_planner_revision`

## `a2a_agent.po_fhir_context`
_A2A v1 FHIR-context extension reader._

- **LoC**: 121
- **Public functions** (4): `bind_po_fhir_context`, `extract_po_fhir_context`, `release_po_fhir_context`, `to_fhir_context`

## `a2a_agent.prospective_eval`
_Phase 15.C -- Synthetic prospective evaluation harness._

- **LoC**: 301
- **Pydantic models** (1): `ProspectiveSummary`
- **Public functions** (1): `run_prospective_eval`

## `a2a_agent.push_notifications`
_Phase 3.4 -- A2A Push notification config registry + dispatcher._

- **LoC**: 173
- **Public functions** (3): `default_registry`, `dispatch_task_artifact_update`, `dispatch_task_status_update`

## `a2a_agent.redteam_v2`
_Phase 10.4 -- Adversarial evaluation suite v2._

- **LoC**: 247
- **Public functions** (2): `load_corpus`, `run_redteam_corpus`

## `a2a_agent.redteam_v3`
_Phase 12.7 B2 -- Adversarial v3 multi-target campaign._

- **LoC**: 223
- **Public functions** (1): `run_redteam_multi_target`

## `a2a_agent.redteam_v4`
_Phase 16.I1 - Red-team v4: indirect prompt injection + adversarial._

- **LoC**: 341
- **Pydantic models** (3): `RedTeamV4Case`, `RedTeamV4Report`, `RedTeamV4Result`
- **Public functions** (3): `build_redteam_v4_corpus`, `run_redteam_v4`, `run_redteam_v4_sync`

## `a2a_agent.refusal_classifier`
_Pure-Python refusal classifier mirroring the A2A root agent's safety policy._

- **LoC**: 188
- **Public functions** (1): `classify_query`

## `a2a_agent.registry`
_COMPOSE-3 -- A2A agent registry._

- **LoC**: 115
- **Public functions** (3): `find_agent`, `list_agents`, `serialize_registry`

## `a2a_agent.regulatory_pack`
_Phase 14.17 P1 -- Regulatory pack generator._

- **LoC**: 513
- **Pydantic models** (2): `RegulatoryPack`, `RegulatoryPackSection`
- **Public functions** (2): `build_regulatory_pack`, `render_regulatory_pack_md`

## `a2a_agent.right_to_explanation`
_AUDIT-3 -- Right-to-explanation generator (GDPR Art. 22 / EU AI Act)._

- **LoC**: 174
- **Public functions** (1): `compute_right_to_explanation`

## `a2a_agent.safety_redteam`
_SAFE-1/2/3/4 -- red-team / OOD / pen-test / chaos engineering._

- **LoC**: 381
- **Public functions** (4): `compute_adversarial_fragility`, `compute_chaos_run`, `compute_ood_detector`, `compute_pentest_suite`

## `a2a_agent.saga_composer`
_Phase 13.13 G2 -- Saga composer cross-agent._

- **LoC**: 209
- **Public functions** (2): `build_discharge_saga`, `run_saga`

## `a2a_agent.scenario_counterfactuals`
_Phase 12.5 -- per-scenario counterfactual analysis._

- **LoC**: 452
- **Public functions** (3): `list_counterfactuals`, `run_all_counterfactuals`, `run_counterfactual`

## `a2a_agent.scheduler`
_Phase 14.12 L3 -- Background scheduler._

- **LoC**: 140
- **Public functions** (6): `get_history`, `list_jobs`, `register_default_jobs`, `register_job`, `reset_registry`, `tick`

## `a2a_agent.smart_on_fhir`
_Phase 16.H3 - SMART-on-FHIR EHR launch flow._

- **LoC**: 182
- **Pydantic models** (3): `SmartLaunchContext`, `SmartLaunchTrace`, `SmartTokenResponse`
- **Public functions** (6): `begin_smart_launch`, `code_challenge_for`, `decode_id_token_unsafe`, `generate_code_verifier`, `load_smart_configuration`, `simulate_ehr_token_exchange`

## `a2a_agent.stigma_linter`
_Phase 17.Z - Stigma + reading-level linter._

- **LoC**: 274
- **Pydantic models** (4): `PatientFacingTextAudit`, `ReadabilityReport`, `StigmaFinding`, `StigmaLintReport`
- **Public functions** (3): `assess_readability`, `audit_patient_facing_text`, `lint_stigma`

## `a2a_agent.streaming`
_Phase 3.3 -- Server-Sent Events (SSE) streaming utilities._

- **LoC**: 208
- **Public functions** (6): `sse_artifact_update`, `sse_done`, `sse_event`, `sse_status_update`, `stream_llm_polish`, `stream_outcomes_simulation`

## `a2a_agent.synthea_bundles`
_Phase 10.5 -- Synthea-style FHIR R4 Bundle generator._

- **LoC**: 254
- **Public functions** (2): `generate_bundle`, `generate_cohort`

## `a2a_agent.synthea_evaluator`
_SCALE-1 -- Evaluate the calibrated readmission model against a Synthea cohort._

- **LoC**: 225
- **Public functions** (2): `derive_readmission_label`, `evaluate_synthea_cohort`

## `a2a_agent.thompson_sampling`
_Phase 17.AP - Multi-armed bandits + Thompson sampling._

- **LoC**: 175
- **Pydantic models** (2): `ArmPosterior`, `BanditTrace`
- **Public functions** (2): `regret_bound_o_sqrt_n_log_k`, `simulate_bandit_run`

## `a2a_agent.tool_discovery`
_GENAI-4 -- Semantic tool discovery across the 40 MCP tools._

- **LoC**: 187
- **Public functions** (1): `semantic_search_tools`

## `a2a_agent.tool_retrieval_tfidf`
_Phase 14.14 Q3 -- Pure-Python TF-IDF tool retrieval._

- **LoC**: 204
- **Pydantic models** (2): `ToolRetrievalHit`, `ToolRetrievalResult`
- **Public functions** (1): `retrieve_tools_by_query`

## `a2a_agent.trajectory_predictor`
_TIME-4 -- Bayesian state-space trajectory predictor._

- **LoC**: 97
- **Public functions** (1): `predict_trajectory`

## `a2a_agent.trustworthy_ml`
_Phase 16.F2 - Trustworthy-ML formalisms._

- **LoC**: 280
- **Pydantic models** (4): `ConformalMultiClassReport`, `SelectiveClassificationPoint`, `SelectiveClassificationReport`, `SubgroupCalibrationReport`
- **Public functions** (3): `per_subgroup_calibration_tension`, `selective_classification_curve`, `split_conformal_multiclass`

## `a2a_agent.ws_chat`
_Phase 13.12 G1 -- WebSocket A2A streaming + multi-turn chat._

- **LoC**: 154
- **Public functions** (4): `get_session_state`, `handle_user_message`, `reset_session`, `ws_chat_endpoint`
