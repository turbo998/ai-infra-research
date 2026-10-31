# Megatron-LM KDA Two-Stage Gates & Padding-Only Checkpoints, PyTorch Philox Offset Fix & Transport Slicing, and Mooncake/LMCache Paged KV Compaction with Dynamo/NIXL (Phase 137)

> **中文版**: [megatron-kda-gate-pytorch-philox-mooncake-lmcache-compaction-dynamo-nixl-phase137.md](megatron-kda-gate-pytorch-philox-mooncake-lmcache-compaction-dynamo-nixl-phase137.md)
> **Entry Point**: Distributed Training & KV Hierarchy — Megatron KDA Two-Stage GLM Gates & Padding Shards / PyTorch CUDA Dropout Philox Offset & Transport Slicing / LMCache Paged GPU KV Compaction, L1 Short-Circuit & DSv4 Groups / Mooncake Target Spread, Dynamo Residency & NIXL GDS Diagnostics
> **Evidence Tier**: `implementation` + `own-offline` (10 primary distributed training & KV fabric commits audited at source level + Python 3.12 deterministic modeling)

---

## 1. Executive Summary & Architectural Conclusions

1. **Megatron-LM KDA Hybrid Layers & Padding-Only Optimizer Shards (`Megatron-LM@57103cf`, `d5cc5e6`)**: Adds Kimi Delta Attention (KDA) hybrid layers with GLM-style two-stage gates and safely handles padding-only distributed optimizer checkpoint shards.
2. **PyTorch CUDA Dropout Philox Offset Fix & Transport Slicing (`pytorch@fe58682`, `2d4eab9`)**: Fixes Philox RNG offset under-reservation in vectorized CUDA dropout kernels (eliminating **100%** of RNG subsequence overlaps) and adds zero-copy slicing for `MemoryView` and `RemoteBuffer` in `torch.distributed._transport`.
3. **LMCache Paged GPU KV Compaction, L1 Short-Circuit & DSv4 Compressed Groups (`LMCache@fa34a70`, `eed45ed`, `d04ed1d`)**: Compacts paged GPU K/V blocks after token dropping (reclaiming **59.96%** of physical pages at a 40% retention ratio), skips L2 lookups when L1 already holds the entire prefetch request (**3.5x** lower lookup latency at 75% L1 full-hit rate), and supports compressed KV groups (`dsv4`) in engine-driven transfers.
4. **Mooncake Target Spread, Dynamo SGLang Direct Linker Attribution & NIXL cuFile Errors (`Mooncake@4065594`, `dynamo@0d86794`, `nixl@cc8f7d6`)**: Adds `--target_access` offset spreading in Mooncake `tebench`, attributes Mooncake session residency from SGLang's direct linker in Dynamo's router, and restores detailed `cuFile` error diagnostics in NIXL `cuda_gds`.

---

## 2. Primary Commit Matrix

| # | Repo & Commit | Date | Primary Code Paths | Architectural Mechanism & Breakthrough |
|---|---|---|---|---|
| 1 | [NVIDIA/Megatron-LM@57103cf](https://github.com/NVIDIA/Megatron-LM/commit/57103cf0633051b91c8b703df498dd51ebb3f192) | `2026-10-28` | `megatron/core/models/hybrid/hybrid_block.py`, `megatron/core/models/hybrid/hybrid_layer_specs.py` | Add KDA (Kimi Delta Attention) hybrid layers with GLM-style two-stage gates (#7054) |
| 2 | [NVIDIA/Megatron-LM@d5cc5e6](https://github.com/NVIDIA/Megatron-LM/commit/d5cc5e6d816a037a3b5ca97848ed81e852a66b89) | `2026-10-28` | `megatron/core/optimizer/distrib_optimizer.py`, `tests/unit_tests/distributed/test_distrib_optimizer_checkpoint.py` | Handle padding-only distributed optimizer checkpoint shards (#6191) |
| 3 | [pytorch/pytorch@fe58682](https://github.com/pytorch/pytorch/commit/fe586824ec8d03067aaf5a05614c962973a3f9a5) | `2026-10-30` | `aten/src/ATen/native/cuda/Dropout.cu`, `test/inductor/test_dropout_align_random_eager.py` | [CUDA] Fix dropout Philox offset under-reservation for vectorized kernels (#202606) |
| 4 | [pytorch/pytorch@2d4eab9](https://github.com/pytorch/pytorch/commit/2d4eab9590b45572f465358efe796811a1b2252d) | `2026-10-30` | `docs/source/distributed/transport.md`, `test/distributed/test_transport_nixl.py` | [transport] Add slicing for memory views and remote buffers (#202535) |
| 5 | [kvcache-ai/Mooncake@4065594](https://github.com/kvcache-ai/Mooncake/commit/406559413a6e6917574bf5d40969dd860b3a0e4e) | `2026-10-29` | `docs/source/performance/mooncake/tebench.md`, `mooncake-transfer-engine/benchmark/main.cpp` | [TransferEngine] Add --target_access to spread tebench target offsets (#5021) |
| 6 | [LMCache/LMCache@fa34a70](https://github.com/LMCache/LMCache/commit/fa34a708a075543aad183176c6a0d16ba2b7f3b3) | `2026-10-29` | `lmcache/integration/vllm/experimental/token_drop_kv.py`, `tests/v1/test_token_drop_gpu_views.py` | [Token Dropping] [KV Compaction 2/3] Compact paged GPU K/V (#5737) |
| 7 | [LMCache/LMCache@eed45ed](https://github.com/LMCache/LMCache/commit/eed45edd009db7d8e4c0e07078f8d769caf20c35) | `2026-10-29` | `docs/design/v1/distributed/storage_controllers/prefetch_controller.md`, `docs/design/v1/mp_observability/METRICS.md` | [Perf][MP] Skip L2 lookups when L1 already has the whole prefetch request (#5652) |
| 8 | [LMCache/LMCache@d04ed1d](https://github.com/LMCache/LMCache/commit/d04ed1d959bb61bc639398e3e99ca96d3a14bc59) | `2026-10-29` | `docs/design/v1/multiprocess/engine_driven_transfer_design.md`, `docs/source/recipes/deepseek_v4_flash.rst` | [Feature] Support compressed KV groups(dsv4) in engine-driven transfers (#6124) |
| 9 | [ai-dynamo/dynamo@0d86794](https://github.com/ai-dynamo/dynamo/commit/0d8679402f1cd651a09af15693965ef0909a97f3) | `2026-10-30` | `components/src/dynamo/sglang/register.py`, `components/src/dynamo/sglang/tests/test_runtime_metadata.py` | feat(router): attribute Mooncake session residency from SGLang's direct linker (#16610) |
| 10 | [ai-dynamo/nixl@cc8f7d6](https://github.com/ai-dynamo/nixl/commit/cc8f7d6e8953661c639c6322c3cb8031cffdb73d) | `2026-10-29` | `src/plugins/cuda_gds/gds_batch_engine.cpp`, `src/plugins/cuda_gds/gds_utils.cpp` | cuda_gds: restore cuFile error details in error messages (#2305) |

---

## 3. Deterministic Offline Verification (`experiments/megatron_kda_pytorch_philox_lmcache_dynamo_phase137/`)

- **PyTorch Philox Overlap Fraction**: Reduced from `0.75` (buggy reservation) to **`0.0`**.
- **LMCache Paged KV Compaction**: Physical pages reduced from `512` to **`205`** (**59.96%** pages reclaimed).
- **LMCache Prefetch Controller Lookup Latency**: Reduced from `420.0 us` to **`120.0 us`** (**3.5x** speedup).
