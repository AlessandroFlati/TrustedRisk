# Embedder comparison report

**Run ID:** `run-1777278481011-f6af8431`
**Workflow:** `grounding-corpus-build` (W3)
**Completed:** 2026-04-27T08:34:28Z
**Winner:** `sentence-transformers/multi-qa-mpnet-base-dot-v1` (promoted to runtime)
**Confidence:** low

## Summary table

| Embedder | Dim | MRR@5 (manual) | Recall@10 (manual) | MRR@5 (fixture) | Recall@10 (fixture) | Combined Score | Rank |
|----------|-----|----------------|--------------------|-----------------|---------------------|----------------|------|
| multi-qa-mpnet-base-dot-v1 | 768 | 0.9500 | 1.0000 | — | — | 0.9750 | 1 |
| all-mpnet-base-v2 | 768 | 0.9250 | 1.0000 | — | — | 0.9625 | 2 |
| all-MiniLM-L6-v2 | 384 | 0.9167 | 1.0000 | — | — | 0.9583 | 3 |

*Combined Score = 0.5 × MRR@5 + 0.5 × Recall@10 (single-subset renormalisation — the spec's manual+fixture split was not provided by the upstream benchmark; see Limitations).*

Per-embedder paired-bootstrap 95% CI on the combined score (2000 resamples over 20 queries):
- all-MiniLM-L6-v2: [0.9083, 1.0000]
- all-mpnet-base-v2: [0.9250, 1.0000]
- multi-qa-mpnet-base-dot-v1: [0.9375, 1.0000]

![Embedder comparison](embedder_comparison.png)

## Winner declaration

**Selected:** `sentence-transformers/multi-qa-mpnet-base-dot-v1`

**Confidence:** low

**Margin to runner-up:** 0.0125 combined score (multi-qa-mpnet-base-dot-v1 0.9750 vs all-mpnet-base-v2 0.9625). Paired-bootstrap 95% CI on the margin: [+0.0000, +0.0375].

**Rationale:** multi-qa-mpnet-base-dot-v1 ranks first on the combined score across the 20-query benchmark. The 0.05 / 0.03 thresholds for `high` / `medium` confidence are not met; with 20 benchmark queries and only 4 candidate documents in the corpus, observed margins are within sampling variability (margin paired-bootstrap 95% CI: [+0.0000, +0.0375]).

**What would resolve to higher confidence:** expand the benchmark beyond 20 queries; introduce a held-out subset of clinically-derived queries with SME-curated ground-truth chunks (not just documents); re-evaluate with chunk-level GT so MRR@5 measures within-document ranking, not just first-relevant-document ranking.

**Runtime artifact:** `grounding_index_multi_qa_mpnet_base_dot_v1.pkl` (under workflow output_dir).
**TrustedRisk config update required:** `EMBEDDING_MODEL=sentence-transformers/multi-qa-mpnet-base-dot-v1`

## Hard cases analysis

[0 queries qualified as hard cases.]

No queries triggered hard-case analysis (every embedder retrieved a chunk from a relevant document within the top-5 for every benchmark query).

Hard cases are flagged when at least one embedder's first relevant chunk appears at rank > 5 or not at all in the top-10. The commentary below is qualitative — without clinical SME re-review of the retrieved chunks, the label simply marks queries where the three embedders disagree on the depth of the answer in the corpus.

## Per-query aggregates

### Manual queries (N=20)
- All-three-embedders mean MRR@5: 0.9306
- All-three-embedders mean Recall@10: 1.0000
- Queries where all three failed (no relevant chunk in top-5): 0
- Queries where exactly one failed: 0
- Queries where all three were perfect (MRR@5 = 1.0 AND Recall@10 = 1.0): 15

### Fixture-derived queries (N=0)
The upstream benchmark loaded from `data/benchmark_queries.json` ships only manually-curated queries (q01–q20); there is no fixture-derived subset. Combined Score formula renormalised accordingly (see Limitations).

## Limitations

- Benchmark is hand-curated with 20 queries over a corpus of only 4 documents (28 chunks total). Margins between embedders fall within sampling variability — see the bootstrap CIs above.
- Ground-truth granularity is **document-level** (`relevant_doc_ids`), not chunk-level. MRR@5 measures the rank of the first relevant *document* in the deduped retrieval ranking; it cannot distinguish a tightly-targeted chunk hit from a broad one in the same document.
- The spec's `manual` vs `fixture_derived` split is not present in the actual benchmark; the combined-score formula was renormalised to `0.5 * MRR + 0.5 * Recall` to keep its range in [0, 1] and preserve the meaningfulness of the spec's 0.05 / 0.03 / 0.02 confidence thresholds.
- The spec describes PubMedBERT as the third candidate; the workflow.yaml as configured at run time uses `multi-qa-mpnet-base-dot-v1` instead. The validator evaluated whichever three embedders the indexer actually built, per the indexer signal.
- All three indices use `IndexFlatIP` over L2-normalised embeddings (verified: `np.linalg.norm` of stored embeddings is exactly 1.0 for the mpnet pickle); query encodings used `normalize_embeddings=True` in the same fashion.
- No clinical SME review of either the queries or the retrieved chunks. The benchmark is in-house only.

## Methodology

- Per embedder: load the model from HuggingFace by name (the same name the indexer recorded under `embedder.name` in each pickle).
- Encode the 20 queries with `normalize_embeddings=True`, cast to `float32`, then `index.search(q_emb, k=10)` against the embedder's `IndexFlatIP`.
- Map each returned chunk to its `source_document`. Recall@10 is the fraction of `relevant_source_docs` (translated from the benchmark's `relevant_doc_ids`) that appear in the top-10 chunks. MRR@5 is `1 / rank` of the first relevant document in the deduped document ranking from the top-10 chunks (0 if no relevant doc in top-5 deduped docs).
- Aggregates use `numpy.mean` per embedder over the 20 queries; `combined = 0.5 * MRR@5 + 0.5 * Recall@10`.
- Confidence per the spec, single-subset adapted: `high` if margin >= 0.05, `medium` if margin >= 0.03, `low` otherwise (or if the three are within 0.02). The ranks-on-both-subsets criterion is N/A in this run.
- Bootstrap CIs are paired across embedders (each resample picks the same 20 query indices for all three), so the margin CI is conservative and does not double-count noise.
- No `healthcare.evidence.retrieval_benchmark` tool was invoked (per spec, computed inline). The only registered tool used was `stats.bootstrap_ci` (substituted with an inline numpy bootstrap because the registry tool is for marginal CIs of a single sample, while we need paired CIs across embedders).
