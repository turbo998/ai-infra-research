# Enterprise Embedding & Reranker Two-Stage Retrieval Pipeline Evaluator (Round 400)

Deterministic analytical evaluator comparing four enterprise Embedding + Reranker retrieval pipelines at `500 RPS` peak (`30,000,000` queries/month):

1. **Tier 1 (`tier1_single_stage_3072d_embedding_only`)**: Single-Stage `3072-dim` Embedding Only (`76.8%` NDCG@10, `62.0 ms` P95 latency, `$19,800/mo`).
2. **Tier 2 (`tier2_heavy_7b_embed_plus_9b_reranker_top100`)**: Heavy `7B` Embedding + `9B` LLM Reranker over Top-100 (`91.5%` NDCG@10, `480.0 ms` P95 latency, `$34,500/mo`).
3. **Tier 3 (`tier3_standard_568m_reranker_fixed_top100`)**: Standard `0.6B` Embedding + `568M` Cross-Encoder over Fixed Top-100 (`90.8%` NDCG@10, `95.0 ms` P95 latency, `$7,200/mo`).
4. **Tier 4 (`tier4_mrl_1024d_margin_early_exit_trt_fp8_top30`)**: Matryoshka (`MRL`) `1024-dim` Embedding + Score-Margin Early-Exit (`40%` queries skip reranker) + TensorRT FP8 Top-30 Cross-Encoder + Relevance Cutoff (`92.4%` NDCG@10 [`+15.6 pts` vs. Tier 1], `24.0 ms` P95 latency [`20.0x` faster than Tier 2, `3.96x` faster than Tier 3], `$2,900/mo` [`-91.6%` vs. Tier 2]).

## Usage

```bash
python experiments/enterprise_embedding_reranker_two_stage_pipeline_selection/run.py
python experiments/enterprise_embedding_reranker_two_stage_pipeline_selection/run.py --check
```
