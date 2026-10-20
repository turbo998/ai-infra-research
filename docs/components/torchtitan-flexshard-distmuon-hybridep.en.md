# PyTorch TorchTitan FlexShard (`ComputeLayout` / `BlockShard` / `Owned`) Heterogeneous Optimizer Resharding, DistMuon Zero-Copy Views, and GB200 NVLink72 HybridEP Non-Blocking Capacity Factor

> **中文版**: [torchtitan-flexshard-distmuon-hybridep.md](torchtitan-flexshard-distmuon-hybridep.md) | **Experiment**: [`experiments/torchtitan_flexshard_hybridep_sim/README.md`](../../experiments/torchtitan_flexshard_hybridep_sim/README.md)

### Executive Summary

PyTorch TorchTitan's **FlexShard (`torchtitan/distributed/flex_shard`)** decouples persistent DTensor parameter storage layouts from temporary optimizer-compute sharding (`ComputeLayout`). By introducing periodic heterogeneous block sharding via `BlockShard(dim=0, block_sizes=(512, 64))` (such as Kimi's shared `wkv_a` projection alternating 512-row KV latent and 64-row RoPE key matrices), dynamic single-owner placement (`Owned`), and `compute_dtype=torch.bfloat16` bucketed communication-compute overlap (`BucketConfig`), `DistMuon` preserves exact per-matrix Newton-Schulz orthogonalization boundaries and aspect-ratio LR scaling $\sqrt{\max(1, R/C)}$ while cutting redistribution traffic by 50% and achieving a **1.716x optimizer step speedup**. On the MoE token-dispatch path for GB200 NVLink72, **`HybridEP` (`torchtitan/distributed/deepep/hybridep.py`)** eliminates blocking `cudaStreamSynchronize` D2H syncs via `_num_permuted_tokens_for_non_blocking`, reducing per-GPU pre-allocated dispatch buffers from `16,132.5 MiB` (`capacity_factor=1.0`) to `4,033.12 MiB` (`capacity_factor=0.25`, **75% memory reduction**) with zero dropped tokens under P99 routing skew.

### Upstream Source Responsibility & Locations

| Module & File | Upstream Repo & Commit / Blob SHA | Core Engineering Responsibility |
| :--- | :--- | :--- |
| `torchtitan/distributed/flex_shard/README.md` | `pytorch/torchtitan` commit `3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0` / blob `fe82e542c0acb07894312b1f64899685d1bcd4f4` | Specifies `ComputeLayout`, `BlockShard`, `Owned`, `BucketConfig`, `shard_order_by_tensor_dim`, and Kimi `wkv_a` `(512, 64)` recipes. |
| `torchtitan/distributed/flex_shard/optimizer_reshard.py` | `pytorch/torchtitan` commit `3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0` / blob `cc4d63592b8bbc54e441191fff77fc86d6ad729c` | Implements `BlockShard` (`num_blocks` and `block_start` periodic boundary math), `Owned`, `ComputeLayout`, and `_FrozenConfigMapping`. |
| `torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py` | `pytorch/torchtitan` commit `3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0` / blob `477bdaa818ad2aa435618158b718ba2b94d0734b` | Implements `_bind_bucket_configs`, `_resolve_buckets`, `_TensorRegion` routing, and packed collective schedules. |
| `torchtitan/distributed/deepep/hybridep.py` | `pytorch/torchtitan` commit `3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0` / blob `b82c43862a59203545c08e0c68aa0b7a4ade59cf` | Implements GB200 NVLink72 TMA-optimized `hybridep::dispatch`/`combine` custom ops, `DispatchHandle` opaque FX type, `_num_permuted_tokens_for_non_blocking`, and sticky `_over_budget` tracking. |

### Core Implementation Path, Memory Layout & Synchronization Contracts

1. **`BlockShard` Heterogeneous Periodic Partitioning (`optimizer_reshard.py:37-85`)**:
   - `BlockShard(dim=0, block_sizes=(512, 64))` partitions a fused 2D parameter into repeating whole blocks of 512 and 64 rows (`period = 576`) without splitting any individual matrix across ranks, allowing `DistMuon` to run Newton-Schulz and aspect-ratio scaling independently on each matrix via zero-copy strided views.
2. **FP32-to-BF16 Bucketed Redistribution Overlap (`_optimizer_reshard_schedule.py:32-95`, `dist_muon.py`)**:
   - Setting `ComputeLayout.compute_dtype = torch.bfloat16` converts FP32 parameter shards to BF16 before packed redistribution and sends orthogonalized update directions back in BF16, halving network traffic and overlapping bucket communication with Newton-Schulz compute.
3. **GB200 NVLink72 `HybridEP` Non-Blocking Buffer Sizing (`hybridep.py:73-120`)**:
   - `_num_permuted_tokens_for_non_blocking` sizes the fused permute output buffer upfront as `int(num_tokens * ep_size * min(num_local_experts, top_k) * capacity_factor)` aligned up to `pad_multiple` (128 for MXFP8), avoiding `cudaStreamSynchronize` and enabling end-to-end `torch.compile` graph capture via `DispatchHandle`.

### Key Equations, Units & Engineering Assumptions

$$P = \sum_{i=0}^{k-1} b_i, \quad N_{\text{blocks}}(D) = \frac{D}{P} \cdot k, \quad \text{block\_start}(m) = \left\lfloor \frac{m}{k} \right\rfloor \cdot P + \sum_{i=0}^{(m \bmod k)-1} b_i$$

$$N_{\text{aligned}} = \left\lceil \frac{\left\lfloor T \cdot \text{EP}_{\text{size}} \cdot \min(E_{\text{local}}, K_{\text{top}}) \cdot \alpha_{\text{cap}} \right\rfloor}{M_{\text{pad}}} \right\rceil \cdot M_{\text{pad}}$$

### Deterministic Offline Simulation & Quantitative Comparison

Validated via [`experiments/torchtitan_flexshard_hybridep_sim/run.py`](../../experiments/torchtitan_flexshard_hybridep_sim/run.py) and [`experiments/torchtitan_flexshard_hybridep_sim/results.json`](../../experiments/torchtitan_flexshard_hybridep_sim/results.json):

| Dimension | Metric / Scenario | Baseline | TorchTitan FlexShard / HybridEP | Quantitative Finding |
| :--- | :--- | :--- | :--- | :--- |
| **Kimi `wkv_a` Sharding (`D=4608`)** | 8 repetitions of `(512, 64)` (16 blocks) | Naive 6-rank uniform split breaks **4 matrix boundaries** | `BlockShard(0, (512, 64))` **0 boundary violations** | Preserves exact per-matrix Newton-Schulz orthogonalization |
| **DistMuon Step Latency** | 4 buckets × 8 params `[4096, 4096]`, 8 ranks | Sequential FP32 reshard + compute: `8.7570 ms` | Overlapped BF16 `compute_dtype`: `5.1033 ms` | **50% less comm volume** and **1.716x step speedup** |
| **GB200 NVLink72 HybridEP Buffer** | `T=4096, EP=72, E_local=4, top_k=8` | `capacity_factor=1.0`: `16,132.50 MiB` (`1,179,648` tokens) | `capacity_factor=0.25`: `4,033.12 MiB` (`294,912` tokens) | **75.0% memory reduction**, 0 D2H syncs, **0 drops** at P99 skew |

### Architectural Impact on Quality, SLO, Topology, Cost & Operations

- **Mathematical Integrity**: Prevents FSDP `Shard(0)` from slicing mid-matrix across fused heterogeneous projections (`wkv_a`), ensuring exact Newton-Schulz convergence.
- **Checkpoint Restore Contract**: `ComputeLayout` is construction-time configuration and is not persisted in `state_dict`; checkpoint recovery must reconstruct `DistMuon` with identical `compute_sharding_by_fqn`.
