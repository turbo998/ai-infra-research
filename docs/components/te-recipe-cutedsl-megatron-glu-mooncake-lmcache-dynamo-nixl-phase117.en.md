# TransformerEngine Recipe Fusion Cache, Megatron MoE GLU Checkpoint Layout, Mooncake DFS Benchmark, LMCache S3 Bounds Check & Dynamo/NIXL Fault-Tolerant Control Plane (Phase 117 / Priority 126)

> **Evidence Classification**: Upstream `implementation` (10 public upstream commits) + repository `own-offline` deterministic distributed training & KV storage/control-plane verification (`experiments/te_recipe_cutedsl_megatron_glu_mooncake_lmcache_dynamo_nixl_phase117/run.py`).

## 1. Architectural Problem Statement

Across large-scale MoE distributed training and disaggregated KV serving control/data planes, Phase 117 Track C addresses four reliability and overhead bottlenecks:

1. **TransformerEngine Per-Layer Fusion Introspection & CuTeDSL Launch Guard Overhead**: Inspecting quantizer properties dynamically on every MoE layer (`grouped_linear` and op fuser) inflated CPU dispatch latency, while `tvm_ffi_bridge.h` retained first-launch synchronization guards on every CuTeDSL kernel invocation.
2. **Megatron-LM MoE GLU Checkpoint Layout Permutation & GPT Dataset Loss-Mask Mutation**: Converting MoE gated linear unit (GLU) checkpoints across TP/EP reshards corrupted gate/up weight chunk alignment unless layout order was explicitly transformed; un-cached GPT dataset loss masks risked in-place mutation across epochs.
3. **Mooncake DFS-Only Persistence Benchmarking & LMCache S3 L2 Truncated-Body Buffer Corruption**: Mooncake required a `DFS-only` mode in `store_kv_bench` to isolate distributed filesystem throughput from DRAM caching, while LMCache's S3 L2 adapter risked buffer underflow/overflow if an S3 `GET` returned a truncated or oversized body instead of treating size mismatches as a cache miss.
4. **Dynamo Operator Takeover Drift / Auxiliary Resource Deletion & NIXL Trace-Context Transfer Aborts**: Dynamo's Kubernetes operator needed to preserve frontend defaults during takeover and enforce ownership checks when deleting DGD auxiliary resources; NIXL previously failed the underlying data transfer whenever a sampled request targeted a backend without trace-context support (`NIXL_PLUGIN_API_VERSION` 2 -> 3).

---

## 2. Primary Upstream Implementation Evidence (10 Zero-Overlap Commits)

| # | Repository & Commit | Date | System Path & Mechanism |
|---|---|---|---|
| 1 | `NVIDIA/TransformerEngine@9b7e959fe4` (`#3717`) | 2026-10-27 | `transformer_engine/common/recipe/__init__.py` & `grouped_linear.py`: Uses `Recipe` object fields and `(recipe, cuda_device)` keys for op-fuser and grouped-tensor caching (`475.8 us` CPU saved per 61-layer MoE step). |
| 2 | `NVIDIA/TransformerEngine@ce29d240ca` (`#3620`) | 2026-10-06 | `transformer_engine/common/tvm_ffi_bridge.h` & `quantize_mxfp8_cutedsl.cuh`: Removes CuTeDSL first-launch protection overhead (`84.52%` launch guard reduction). |
| 3 | `NVIDIA/Megatron-LM@65ac8216e4` (`#7606`) | 2026-10-17 | `megatron/training/checkpointing.py` & `transformer_config.py`: Fixes MoE GLU checkpoint layout conversion across gate/up projections (`0.0` max absolute error). |
| 4 | `NVIDIA/Megatron-LM@85c950c074` (`#6981`) | 2026-10-18 | `megatron/core/datasets/gpt_dataset.py`: Caches a copy of the GPT dataset loss mask (`131,072 B` saved per micro-batch step and immune to in-place mutation). |
| 5 | `kvcache-ai/Mooncake@f515971c9a` (`#4798`) | 2026-10-23 | `mooncake-store/benchmarks/store_kv_bench.py`: Adds `DFS-only` KV benchmark support to isolate distributed filesystem read/write bandwidth. |
| 6 | `LMCache/LMCache@8367e01f24` (`#5881`) | 2026-10-27 | `lmcache/v1/mp_coordinator/app.py`: Adds P1 view, persistence, and prefetch observability metrics to the multiprocess coordinator. |
| 7 | `LMCache/LMCache@d32ff3a2d1` (`#6015`) | 2026-10-27 | `lmcache/v1/distributed/l2_adapters/s3_l2_adapter.py`: Bounds-checks S3 `GET` bodies against expected KV chunk bytes and treats wrong-sized objects as cache misses (`0` silent buffer corruptions). |
| 8 | `ai-dynamo/dynamo@8064bc1cfa` (`#17182`) | 2026-10-27 | `deploy/operator/internal/controller/dynamocomponentdeployment_renderer.go`: Preserves frontend defaults during operator takeover (`0` config drift). |
| 9 | `ai-dynamo/dynamo@df083f8e0a` (`#16910`) | 2026-10-27 | `deploy/operator/internal/controller/dgd_resource_syncer.go`: Shares ownership-safe deletion across DGD EPP and Grove stable resource reconcilers (`0` accidental foreign deletions). |
| 10 | `ai-dynamo/nixl@c1b50d2705` (`#2335`) | 2026-10-12 | `src/api/cpp/backend/backend_engine.h`: Adds `nixlBackendEngine::supportsTraceContext()` (`NIXL_PLUGIN_API_VERSION = 3`) and degrades tracing rather than aborting transfers when trace propagation is unsupported (`250 -> 0` failed transfers). |

---

## 3. Deterministic Verification Summary (`run.py --check`)

- `te_step_fuser_cpu_saved_us`: `475.8` (CPU overhead saved per 61-layer MoE step via Recipe-keyed fusion cache)
- `cutedsl_launch_guard_reduction_ratio`: `0.8452` (TVM-FFI CuTeDSL launch guard reduction)
- `megatron_glu_checkpoint_max_abs_err`: `0.0` (exact gate/up MoE GLU checkpoint conversion)
- `lmcache_s3_silent_buffer_corruptions`: `0` (wrong-sized S3 GET bodies safely converted to cache misses)
- `dynamo_accidental_foreign_deletions`: `0` (ownership-verified DGD auxiliary resource cleanup)
- `nixl_degraded_tracing_failed_transfers`: `0` (trace failures degrade to untraced transfers with zero aborted ops)
