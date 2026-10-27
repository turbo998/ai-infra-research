# TransformerEngine Per-Instance FP8 Quantizer, Megatron Contiguous MLA RoPE & MFSDP v2 MXFP8, TorchTitan MoE/RL, Mooncake MemFabric/DFS, LMCache & NIXL HF3FS (Phase 114)

> **Evidence Classification**: `implementation` + `own-offline` (upstream commit analysis and reproducible CPU/offline model verification; not real GPU/RDMA hardware profiling)

## 1. Upstream Code Paths & Distributed System Invariants

| Evidence ID | Upstream Commit | Key Implementation Path & Distributed Invariant |
|---|---|---|
| `te-build-fp8-recipe-quantizer-params-per-instance-2026` | `NVIDIA/TransformerEngine@14be43fb92` (`2026-10-08`) | Instantiates fresh `QParams(power_2_scale=use_power_2_scales, amax_epsilon=0.0)` inside `Float8CurrentScaling` per instance instead of sharing class-level defaults. |
| `te-common-enable-all-activations-mxfp8-cpp-test-2026` | `NVIDIA/TransformerEngine@f57b8b8e61` (`2026-10-09`) | Enables `GeLU`, `SiLU`, `ReLU`, `QGeLU`, and `SReLU` across single-tensor and grouped MXFP8 cast C++ tests. |
| `megatron-mla-query-rope-contiguous-forward-load-2026` | `NVIDIA/Megatron-LM@a4b2beebc0` (`2026-10-16`) | Loads MLA query RoPE dimensions contiguously in `fused_mla_yarn_rope_apply.py` forward kernel instead of strided `0::2`/`1::2` loads (`50%` fewer memory transactions per warp). |
| `megatron-mfsdp-v2-mcore-adapter-enable-mxfp8-2026` | `NVIDIA/Megatron-LM@7f8dcee979` (`2026-09-24`) | Enables MXFP8 compute and `fp8_param_gather` in `mcore_fsdp_adapter.py` (`2.0x` parameter gather compression vs BF16, validated on DGX GB200). |
| `torchtitan-moe-skip-per-expert-token-count-unused-2026` | `pytorch/torchtitan@72a3a126fe` (`2026-10-27`) | Passes boolean `routing_map_TE` directly and skips `torch.bincount` in MoE backends that do not read `num_tokens_per_expert_E` (`870 us/step` saved across 60 layers). |
| `torchtitan-rl-preserve-per-group-kv-cache-policy-update-2026` | `pytorch/torchtitan@ab51d1311d` (`2026-10-01`) | Preserves per-group KV cache across RL policy updates when `hot_swap=True` or `cache_salt` isolates policy versions, while rejecting unsafe `hot_swap=False && !reset_prefix_cache_on_weight_sync`. |
| `mooncake-tent-add-memfabric-transport-2026` | `kvcache-ai/Mooncake@129c9614de` (`2026-10-16`) | Adds Huawei `MemFabric` (`smem_trans`) shared-memory transport under `tent/src/transport/memfabric`. |
| `mooncake-store-allow-dfs-only-replica-placement-2026` | `kvcache-ai/Mooncake@c6f1d08e1c` (`2026-10-19`) | Allows DFS-only replica placement (`replica_num = 0, nof_replica_num = 0, dfs_replica_num = 1`) in `master_service.cpp`. |
| `lmcache-coordinator-serve-cache-delete-eviction-controller-2026` | `LMCache/LMCache@d05401790f` (`2026-10-22`) | Binds `/cache/delete` to `EvictionController` alongside `/quota` and `/cache/pins` so pinned keys are held back during deletion. |
| `nixl-hf3fs-plugin-fix-atomic-races-2026` | `ai-dynamo/nixl@4ef0a4d9be` (`2026-10-12`) | Converts `stop_thread`, `error_status`, and `submitted_ios` in `hf3fs_backend.{h,cpp}` to `std::atomic` to eliminate thread-safety races. |

## 2. Verification Command

```bash
python experiments/te_fp8_quantizer_megatron_mla_rope_mfsdp_mxfp8_mooncake_phase114/run.py --check
```
