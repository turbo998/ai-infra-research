# Megatron-LM Hybrid DSA-GQA & GDN CUDA Graph, Mooncake Store Reshard & Quota, and NIXL Telemetry Experiment

Deterministic Python 3.12 standard-library model reproducing:
1. **`NVIDIA/Megatron-LM` PR `#7156` (`d14b699755`) & PR `#8353` (`29e1a7b669`)**: Hybrid `DSGroupedSelfAttention` + `DSAIndexerLossLoggingHelper` `raw_values` loss tracking and Gated DeltaNet (`gdn.py`) prefill CUDA-graph replay via single-update GDP (`num_householder=1`) with fixed-capacity chunk descriptors.
2. **`kvcache-ai/Mooncake` PR `#4762` (`e2d6e8e0f2`) & PR `#4809` (`44c4d9cf4a`)**: `plan_kv_cache_store_upload` (`replica_ordinal == 0` writer deduplication) + `plan_kv_cache_store_load` (`HeadIntervalIndex` cover across all target replicas) and `TenantQuotaPolicy` (`Held` / `Carried` upsert accounting + watermark eviction).
3. **`ai-dynamo/nixl` PR `#2479` (`03d9b74648`)**: 64-bit `nixl.xfer_request_id` (zero-avoiding redraw) and `nixl.xfer_execution` per-post execution counter.

## Run

```powershell
python experiments\megatron_dsa_gqa_gdn_mooncake_store_reshard_quota\run.py --check
```
