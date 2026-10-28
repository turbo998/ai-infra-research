# Megatron-LM mHC Precision & Reusable Pinned Offload Buffer, TE FA3/FA4 THD Mask & Permutation Memory Reduction, Mooncake Logical-Box Resharding, LMCache MP Store Bound, Dynamo BLAKE3 Verification, and NIXL V3 Telemetry (Phase 119 / Priority 128)

> **Evidence Classification**: Upstream `implementation` + `author-benchmark` (10 public upstream commits dated 2026-10-01 to 2026-10-28) + repository `own-offline` deterministic arithmetic/topology verification (`experiments/megatron_mhc_recompute_pinned_offload_te_fa4_perm_mooncake_lmcache_dynamo_nixl_phase119/run.py`). This note makes no unverified live GPU hardware re-benchmark claims.

## 1. Core Architectural Problems & Context

Across large-scale hybrid training (mHC / MoE), FP8/FA4 context parallelism, cross-cluster weight resharding (Mooncake), and multiprocess KV caching / NIXL transport, five critical memory and correctness bottlenecks were resolved:

1. **Autocast Drift in Megatron-LM mHC Recompute and Unbounded Pinned Host Memory Growth in Variable-Shape MoE Offloading**:
   - `CheckpointWithoutOutputFunction` replayed backward recomputation under ambient autocast state rather than the forward pass's autocast policy, and retaining `self.probs` in `combine_postprocess` pinned the recomputed MoE graph until the next iteration;
   - Variable-shape MoE offload groups (`expert_fc1`, `moe_act`, `fused_group_mlp`) allocated directly from the PyTorch pinned host allocator, which rounds every request up to a **power of two** and caches freed blocks, ratcheting up pinned host memory as expert routing fluctuates.
2. **Unwritten FA3/FA4 THD Padding Rows Under CP and 30 GB Memory Peak in TE Permutation Reference**: FA3/FA4 kernels can leave physically padded rows and buffer tails unwritten, corrupting results if not zeroed before inverse CP All-to-All; meanwhile, the JAX MoE permutation reference materialized `[num_tokens * num_experts, hidden_size]`, consuming **30 GB** of memory.
3. **Rank-Coupled Weight Resharding and Container-Broken Ascend DSMI Affinity in Mooncake**: Published weight snapshots (`StoredWeightManifest`) needed logical-box byte planning decoupled from source training rank semantics; on Ascend NPUs, `ASCEND_RT_VISIBLE_DEVICES` renumbers ACL user device IDs away from driver logical IDs, breaking DSMI URMA affinity lookups that mistakenly queried `aclrtGetPhyDevIdByLogicDevId`.
4. **Signed Pointer Wraparound and Partial Hybrid/DCP Under-Storing in LMCache MP**: `lmcache_objects_ptrs` used signed `int64_t` instead of `uintptr_t`; furthermore, when partial hybrid/DCP hits floored `vllm_hit` (`112` computed tokens floored to `64`), bounding stores by `num_scheduled_tokens + max(vllm_hit, lmcache_hit)` capped the store range at `252 -> 192` instead of `300 -> 256`.
5. **Unverified Checkpoint Staging in Dynamo GMS and Destructor / Drop-Visibility Gaps in NIXL Telemetry**: NIXL GDS/POSIX staging lacked thread-bounded streaming BLAKE3 verification, and `nixlAgentData` declared `telemetry_` after `backendEngines_`, risking use-after-free during engine teardown while silently dropping trace phases when ring buffers filled.

---

## 2. Primary Upstream Implementation Analysis (10 Zero-Overlap Commits)

| # | Repository & Commit | Date | Core Implementation Path & Microarchitectural Mechanism |
|---|---|---|---|
| 1 | `NVIDIA/Megatron-LM@8e134118ba` (`#7521`) | 2026-10-27 | `megatron/core/transformer/hyper_connection.py` & `recompute.py`: Adds mHC precision controls (FP32 `_sinkhorn_iterations(input_logits.detach())`, autocast-disabled `native_h_post_bda` `torch.bmm`), optional mean output contract (`mhc_learned_output_contract=False`), saved CUDA autocast replay in `CheckpointWithoutOutputFunction`, `self.probs = None` in `combine_postprocess`, and `param_to_inner_param` refresh in `HybridOptimizer.update_fp32_param_by_new_param`. |
| 2 | `NVIDIA/Megatron-LM@57018e45e4` (`#7961`) | 2026-10-27 | `megatron/core/pipeline_parallel/pinned_activation_buffer.py`: Introduces `--fine-grained-offloading-buffer-size-gib` (`PinnedActivationBuffer`), allocating one 256-byte-aligned `cudaHostAlloc` buffer per rank (avoiding power-of-two rounding) that packs all offloaded activations and reuses space once all backups reload, reserving `[0, graph_end)` for CUDA Graphs and `[graph_end, capacity)` for eager epochs. |
| 3 | `NVIDIA/TransformerEngine@f38a1fce7f` (`#3205`) | 2026-10-02 | `transformer_engine/pytorch/attention/dot_product_attention/`: Extends `get_thd_padding_mask` to cover inter-sequence gaps and buffer tails `[cu_seqlens_padded[-1], num_tokens)`, adds `_MaskTHDPaddingGrad` for non-CP backward, and zeroes FA3/FA4 padded rows via `_zero_thd_padding` before inverse A2A in CP forward/backward. |
| 4 | `NVIDIA/TransformerEngine@0838ecbe4f` (`#3609`) | 2026-10-01 | `tests/jax/test_permutation.py`: Scatters 1D token indices into `src_tokens` (`[num_out_tokens]`) before gathering rows via `inp.at[src_tokens].get(mode="fill", fill_value=0)`, avoiding `[num_tokens * num_experts, hidden_size]` materialization and cutting memory from **30 GB to 1.75 GB** (**17.14x** reduction). |
| 5 | `kvcache-ai/Mooncake@c21a541827` (`#4827`) | 2026-10-28 | `mooncake-reshard/`: Validates `StoredWeightManifest` and `plan_stored_transfer_to_target_placement` logical-box byte-range intersections across nonuniform `4->8` and `8->4` layouts, both dimensions, shuffled fragments, and padded object offsets without training rank semantics. |
| 6 | `kvcache-ai/Mooncake@a8acb4da20` (`#4971`) | 2026-10-28 | `mooncake-transfer-engine/src/transport/ascend_transport/ascend_direct_transport/utils.cpp`: Replaces `CpusForLogicDevice` with `CpusForUserDevice`, resolving `aclrtGetLogicDevIdByUserDevId(user_id, &logic_id)` before querying DSMI `kDsmiUbSubUrmaName` under `ASCEND_RT_VISIBLE_DEVICES`. |
| 7 | `LMCache/LMCache@b540ae9ea0` (`#5672`) | 2026-10-28 | `csrc/cuda/mp_mem_kernels.cu` & `csrc/kv_transfer_plan_types.h`: Types `lmcache_objects_ptrs` as `std::vector<uintptr_t>` instead of `std::vector<int64_t>` across `KernelGroupSpec`, CUDA transfer kernels, and pybind11 bindings. |
| 8 | `LMCache/LMCache@4f7d283a4d` (`#5911`) | 2026-10-20 | `lmcache/integration/vllm/lmcache_mp_metadata.py`: Tracks `num_engine_computed_tokens = num_computed_tokens + num_new_tokens` and bounds `GetStoreMetadata` by `max(num_scheduled_tokens + max(vllm_hit, lmcache_hit), num_engine_computed_tokens)`, storing `(64, 256)` instead of truncating at `(64, 192)` on partial hybrid/DCP hits. |
| 9 | `ai-dynamo/dynamo@17bffaf9c7` (`#16146`) | 2026-10-28 | `components/src/dynamo/common/Storage/`: Adds streaming and pre-restore BLAKE3 integrity verification (`FileIntegrity`) to GMS NIXL checkpoint staging with per-rank thread bound `blake3_threads_per_shard(w) = max(1, min(4, 16 // w))`. |
| 10 | `ai-dynamo/nixl@349514f94e` (`#2323`) | 2026-10-28 | `src/core/telemetry/` & `src/api/cpp/nixl.h`: Adds atomic `AGENT_TRACE_PHASES_DROPPED = 20` (`AGENT_TELEMETRY_EVENTS_DROPPED = 21`), bumps `TELEMETRY_VERSION` to `5`, MP store schema to `2`, and plugin API to `V3`, and declares `telemetry_` before `tracer_` and `backendEngines_` so it outlives engine progress threads. |

---

## 3. Key Formulas, Data Movement & Offline Verification

### 3.1 256B-Aligned Pinned Buffer vs Power-of-Two Cache Growth & Index-Scatter Permutation
1. **Pinned Host Memory Savings**: Across variable MoE offload requests $\{130, 190, 260, 310, 400\}\text{ MiB}$, power-of-two caching retains $\sum 2^{\lceil \log_2 s_i \rceil} = 2048\text{ MiB}$, whereas sequential 256B-aligned packing in `PinnedActivationBuffer` requires $1290\text{ MiB}$ (**37.01%** savings).
2. **Index-Scatter Permutation Memory Ratio**: Scattering 4-byte indices into `[num_out_tokens]` before gathering rows reduces reference memory from `30.0 GB` to `1.75 GB` (**17.14x** reduction).

### 3.2 Offline Verification Summary (`run.py --check`)

- `pinned_alignment_bytes` / `pinned_host_memory_savings_ratio`: `256` / `0.3701`
- `thd_padding_mask_exact` / `thd_nonzero_padded_rows_after_cp_cleanup`: `True` / `0`
- `te_perm_memory_reduction_factor`: `17.14` (`30.0 GB -> 1.75 GB`)
- `mooncake_4to8_intersections` / `mooncake_exact_elements_reconstructed`: `8` / `True` (`437` elements verified)
- `dsmi_resolved_logic_ids`: `[2, 3]`
- `buggy_stored_end` / `fixed_stored_end`: `192` / `256`
- `blake3_threads_w8` / `blake3_total_threads_w8`: `2` / `16`
- `nixl_plugin_api_version` / `nixl_telemetry_version`: `3` / `5`

---

## 4. Architect Exercises & Falsifiable Questions

1. **Calculation Exercise**: When a request has `112` already-computed tokens in vLLM, a floored `64`-token chunk hit in LMCache, and `188` newly scheduled tokens (`chunk_size = 64`), why did the old formula drop `[192, 256)` and how does `num_engine_computed_tokens` restore it? (Answer: $188 + 64 = 252 < 256$ floored to $192$, whereas $\max(252, 112 + 188) = 300 \ge 256$ floors to $256$, preserving all 3 uncached full chunks `[64, 256)`).
2. **Falsifiable Hypothesis**: Why must `PinnedActivationBuffer` reserve captured CUDA Graph offload ranges `[0, graph_end)` for the lifetime of the buffer while restricting eager offloads to `[graph_end, capacity)`? Because CUDA Graph replay executes raw device-to-host DMA into fixed host virtual addresses without running Python allocator bookkeeping, so allowing eager epochs to reuse `[0, graph_end)` would overwrite graph-owned activation backups across replays.
