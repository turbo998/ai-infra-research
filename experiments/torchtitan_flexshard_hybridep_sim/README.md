# PyTorch TorchTitan FlexShard, DistMuon & GB200 HybridEP Simulator (`torchtitan_flexshard_hybridep_sim`)

Deterministic Python 3.12 standard-library model of PyTorch TorchTitan's `FlexShard` (`torchtitan/distributed/flex_shard/optimizer_reshard.py`, `_optimizer_reshard_schedule.py`, `dist_muon.py`) and GB200 NVLink72 `HybridEP` (`torchtitan/distributed/deepep/hybridep.py`, commit `3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0`).

## What It Models

1. **`BlockShard` Heterogeneous Periodic Sharding (`optimizer_reshard.py`)**:
   - Models `BlockShard(0, (512, 64))` for Kimi's shared `wkv_a` projection (alternating 512-row KV latent and 64-row RoPE key matrices), preserving whole matrix boundaries and per-block aspect-ratio LR scaling $\sqrt{\max(1, R/C)}$ during Newton-Schulz orthogonalization.
2. **Bucketed FP32-to-BF16 Redistribution & Communication Overlap (`_optimizer_reshard_schedule.py`)**:
   - Models `ComputeLayout.compute_dtype = torch.bfloat16` cutting redistribution volume by 50% and overlapping packed collectives with Newton-Schulz compute across `BucketConfig` buckets.
3. **GB200 NVLink72 `HybridEP` Non-Blocking Capacity Factor (`hybridep.py`)**:
   - Models `_num_permuted_tokens_for_non_blocking` (`num_tokens * ep_size * min(num_local_experts, top_k) * capacity_factor` aligned to `pad_multiple=128`), quantifying activation buffer size vs sticky `_over_budget` token drop thresholds.

## Commands

```bash
python experiments/torchtitan_flexshard_hybridep_sim/run.py
python experiments/torchtitan_flexshard_hybridep_sim/run.py --check
```
