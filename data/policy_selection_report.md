# Policy selection report

**Run ID:** run-1777280387352-c8a21500
**Workflow:** abstain-boundary-discovery
**Completed:** 2026-04-27T09:26:58Z
**Decision:** PROMOTE `lace_percentile` policy
**Rationale summary:** `threshold_selector` halted with `decision: "halt_no_viable"` because no sweep point had `abstain_rate_holdout`, `synthetic_detection_overall`, or `correlation_auroc` populated (the workflow lacks `holdout_preparer` and `synthetic_fixture_generator` nodes); per Step 1 of the gate-integrative rule, the topological branch has no candidate to evaluate, so the LACE-percentile fallback is promoted.

## Gate outcome matrix

| Gate | Target | Measured (topological) | Classification | Effect on decision |
|------|--------|------------------------|-----------------|-----|
| Statistical (abstain rate on holdout) | abstain ∈ [5%, 20%] | NOT EVALUABLE — `abstain_rate_holdout` null on all 9 sweep points; only `abstain_rate_clustered_proxy` (17.3%, on training) available | n/a — gate not evaluable | n/a — Step 1 short-circuit |
| Predictive (AUROC distance → error_high) | AUROC ≥ 0.65 | NOT EVALUABLE — `correlation_auroc` null; no holdout predictions or labels (degraded proxy AUROC 0.872 on noise-vs-clustered training only, with explicit caveat) | n/a — gate not evaluable | n/a — Step 1 short-circuit |
| Synthetic OOD detection | detection ≥ 80% | NOT EVALUABLE — `synthetic_detection_overall` null; `synthetic_fixture_generator` node not present | n/a — gate not evaluable | n/a — Step 1 short-circuit |

No topological-policy candidate exists. The integrative gate rule (Step 2 / Step 3) is not entered: Step 1 takes precedence whenever `threshold_selector.decision == "halt_no_viable"`.

## Decision path

- **Step 1 (threshold_selector halt):** `threshold_selector.decision = "halt_no_viable"`, `halt_reason = "no_sweep_point_in_abstain_range"`, `halt_subreason = "abstain_rate_holdout_field_null_on_all_sweep_points"`. `sweep_points_in_range = 0` (out of 9 considered, 8 distinct thresholds). `topological_policy_artifact_written = false`. `expected_critic_action = "promote_lace_percentile_fallback"`. **Match → promote fallback immediately.**
- **Step 2 (gate evaluation):** Skipped. With no topological threshold to evaluate at, gate values are not measurable, and the upstream `ood_analyzer` already classified all three quality gates as NOT EVALUABLE / PROXY ONLY (`gates_evaluable: false`).
- **Step 3 (promotion rule):** Skipped. Step 1 short-circuits the integrative rule.
- **Step 4 (process quality adjustment):** Recorded for audit. `clusterer.clusters_with_forced_naming = 52` (well above the 4 threshold; in fact 100% of the 52 clusters have forced structural naming because `encounter_metadata` was not materialised by `trajectory_embedder`). Independently, this would force fallback under the gate-integrative rule whenever any gate is not pass_comfortable. It reinforces, but is not the primary driver of, the decision.
- **Step 5 (decision):** **Promote LACE-percentile fallback.**

## Process quality observations

- **Pipeline degradation (root cause).** The deployed DAG for this run lacks two nodes that the v0.3 design (sec. 4.5) requires for the topological branch: `holdout_preparer` and `synthetic_fixture_generator`. Their absence cascades to `ood_analyzer` (`gates_evaluable: false`, `degraded_pipeline: true`) and `threshold_selector` (no sweep point in abstain range). This is an architectural gap in the run, not a data-quality problem; it must be fixed before any future topological promotion is possible.
- **Cluster narrative weakness.** `clusterer` selected `combo_1` in `least_bad_selected` mode: 52 clusters (target 8-20), silhouette 0.211, davies-bouldin 1.65, noise fraction 11.9%. All 52 clusters were forced to structural naming (size + peripherality + intra-dispersion in 768-dim space) because `encounter_metadata` was not materialised. Even if the topological policy had a numerical threshold, the interpretive legitimacy of the cluster narrative is weak.
- **Promising peripheral signal (informational, not gate-evidence).** `ood_analyzer` reports a degraded proxy AUROC of 0.872 for distance-to-nearest-centroid discriminating HDBSCAN noise (n=749) vs clustered (n=5553) within training. This suggests the embedding+centroid geometry would carry a useful OOD signal once a real holdout / synthetic fixture set is available, but it does not validate the predictive gate (which requires AUROC of distance vs holdout `error_high`, not noise-vs-core within training).
- **Threshold-selector confidence:** N/A. The selector did not promote a threshold; it halted by design. Per the preset, this is the correct conservative behaviour given degraded upstream evidence.

## Hypothesis evaluation (upstream findings)

- `ood_analyzer`'s explicit downstream recommendation — *"critic should promote the LACE-percentile fallback policy already produced by fallback_policy_builder, since the topological path lacks the predictive-gate evidence required by v0.3 sec.4.5"* — is corroborated and adopted.
- `threshold_selector`'s `expected_critic_action: "promote_lace_percentile_fallback"` is corroborated and adopted.
- `clusterer`'s finding *"the W4 critic should weight this when choosing between the topological policy and the LACE-percentile fallback"* (regarding forced semantic naming on every cluster) is corroborated; it is treated as a Step-4 reinforcing factor, not the primary driver.

## Alternative scenario

The non-selected policy is the **topological policy**, but in this run no topological policy file was produced. The factual counterfactual must therefore be expressed against the closest-to-policy artefact upstream produced — the degraded `ood_analyzer` sweep optimum:

1. **Configuration that would have been promoted, had a topological policy existed.** The degraded sweep optimum was `threshold = 0.13593` (a distance-to-nearest-centroid cutoff in mpnet-base 768-dim space), corresponding to `abstain_rate_clustered_proxy = 17.3%` on the *training-clustered* subset (not the holdout) and `flag_rate_noise_proxy = 71.2%` on training noise (not synthetic OOD fixtures). With 52 clusters of forced naming, the threshold has no clean clinical interpretation; it is a peripherality cutoff in embedding space.
2. **Runtime behaviour the topological policy would have produced.**
   - Abstain rate: unknown on holdout (no holdout was ever embedded). Best estimate from the proxy is in the 15-20% region, but this is not directly comparable; healthcare encounters in the unseen 1576-row holdout could distribute differently in embedding space than the clustered training subset.
   - OOD detection: only the noise-vs-clustered proxy AUROC of 0.872 is available, which is *not* the predictive-gate metric. Real synthetic fixtures (Frankenstein, pediatric, psychiatric, polypharmacy, out-of-scope diagnosis) were never generated; we cannot estimate detection rate on the five required categories.
   - Patients flagged: peripheral encounters in the trained-on cohort (high distance from any of the 52 centroids). Without metadata-grounded cluster names this captures "encounters that look unusual in mpnet space" but cannot guarantee they correspond to clinically out-of-scope patients.
3. **Narrative / pitch implications had we promoted topological anyway.**
   - The pitch *would* gain interpretive language ("we built a clustering-based topological OOD detector"), but the audit trail would show that all three quality gates were non-evaluable, that 100% of clusters had forced naming, and that the predictive evidence was a noise-vs-core proxy on training. This is the exact case the preset's anti-pattern *"Promoting topological despite fail_significant gate"* is designed to prevent — extended here to *"Promoting topological despite NOT EVALUABLE gates"*.
   - At runtime the topological gate would either be too permissive (threshold tuned on training proxy is too narrow once hit by genuinely unfamiliar holdout encounters) or too aggressive (hits too many in-distribution holdout encounters because the proxy abstain rate of 17.3% was computed only on the easier clustered subset). Either failure mode is a healthcare safety concern.

The LACE-percentile fallback is materially different and demonstrably more defensible for this run: it is computed on the full n=7878 cohort, has a clear clinical anchor (LACE ≥ 13 is the established healthcare 30-day-readmission high-risk threshold), is threshold-only (no runtime embedding dependency, simpler operations), and yields a measured 10.6% abstain rate on training that is squarely inside the [5%, 20%] statistical-gate range. A reviewer six months from now can directly verify the LACE cutoff against the cohort distribution.

## Runtime configuration updates (via env artifact)

`output/trustedrisk_env_updates.env`:

```
TRUSTEDRISK_OOD_POLICY_TYPE=lace_percentile
TRUSTEDRISK_OOD_POLICY_PATH=data/abstain_policy_lace_percentile.json
```

(No `TRUSTEDRISK_COHORT_EMBEDDINGS_PATH` is needed — the LACE-percentile fallback is threshold-only and has `runtime_consumers.no_embeddings_required = true`.)

The porting script `scripts/port_calibration_artifacts.sh` merges this into the actual `trustedrisk/.env` at integration time. Also add to `SUPPORTED_ABSTAIN_POLICY_VERSIONS` in `src/shared/schemas.py` (manual edit, non-automated):

```python
SUPPORTED_ABSTAIN_POLICY_VERSIONS = [
    "abstain-fallback-run-1777280387352-c8a21500",  # lace_percentile, promoted
    # No topological version — threshold_selector halted; topological policy file not produced.
]
```

## Known limitations carried into production

- **LACE-only OOD signal.** The promoted policy abstains only on `LACE_total_score >= 13`. It does NOT capture out-of-scope patients whose LACE is low but who are clinically outside the training distribution (e.g., pediatric encounters with LACE 3-5; out-of-scope diagnosis families; Frankenstein polypharmacy with low LACE). This is an explicit known gap of the LACE-percentile method, carried over from `fallback_policy_builder.known_limitations`.
- **No cohort-embedding-based OOD at runtime.** TrustedRisk runtime will not require the mpnet embedder for the abstain decision. Conversely, no embedding-space OOD detection is available; reviewers with concerns about embedding-space outliers must rely on downstream monitoring rather than the abstain gate.
- **Architectural gap in this run.** The DAG lacked `holdout_preparer` and `synthetic_fixture_generator`. Until both are added (and `trajectory_embedder` is wired to consume holdout encounters and emit `encounter_metadata`), the topological branch cannot be validated against design v0.3 sec. 4.5 gates and cannot be promoted on any future run either. Tracking this as a workflow-level remediation is required before re-attempting topological promotion.
- **Cluster interpretive narrative is not production-ready.** Should the topological branch be revived in a future run, the upstream `encounter_metadata` port must be materialised so clusters receive metadata-grounded names rather than forced structural naming.
