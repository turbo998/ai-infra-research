# Enterprise Batch & Offline LLM Inference (50B Tokens/Mo), Cloud Batch API, Off-Peak PTU Backfill & K8s Spot GPU Evaluator

Deterministic evaluator comparing four enterprise offline/batch LLM processing architectures across 50 Billion monthly tokens (`1.2M` documents/night) with an 8-hour overnight completion SLA:
- **Tier 1**: Synchronous Real-Time PAYGO API Loop (`$2.70 / 1M tokens`, `26.5%` HTTP 429 / SLA failure rate, `18.0%` prefix cache hit, `$135,000/mo`)
- **Tier 2**: Dedicated Batch-Only Provisioned Throughput (`$2.16 / 1M tokens`, `0.0%` SLA miss, `35.0%` prefix cache hit, `$108,000/mo`)
- **Tier 3**: Pure Managed Cloud Async Batch API (`50% Off`, `$1.35 / 1M tokens`, `7.8%` 8h-SLA miss under 24h queue, `50.0%` cache hit, `$67,500/mo`)
- **Tier 4**: Daytime-PTU Nightly Backfill + K8s Spot GPU (`Ray Data + vLLM Offline Engine` with Lexicographical Prefix Sorting & Parquet Micro-Shard Checkpointing) (`$0.42 / 1M tokens`, `0.0%` SLA miss, `91.5%` prefix cache hit, `$21,000/mo`)

## Usage
```bash
python experiments/enterprise_batch_offline_llm_inference_spot_gpu_cost_selection/run.py
python experiments/enterprise_batch_offline_llm_inference_spot_gpu_cost_selection/run.py --check
```
