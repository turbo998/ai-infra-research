# SGLang CUTLASS/Marlin MoE Routing Budget Guards, TRT-LLM Connector KV Reuse Accounting, and vLLM Mamba/UVA Scheduling Optimizations

> **Document Status**: 2026-10-24 · Phase 62 (`priority: 71`, Track B)  
> **Evidence Level**: `implementation` + `own-offline` (`experiments/sglang-cutlass-marlin-moe-trtllm-kv-connector-vllm-mamba-uva/run.py` deterministic Python stdlib model)  
> **Question ID**: `q-sglang-cutlass-marlin-moe-trtllm-kv-connector-vllm-mamba-uva`

---

## 1. Architectural Context & Problem Statement

Production LLM serving engines (SGLang, TensorRT-LLM, and vLLM) face subtle correctness and tail-latency pitfalls when handling inactive MoE routing slots under Expert Parallelism (EP), accounting for external KV connector blocks, and scheduling hybrid Mamba-Attention or UVA host-offloaded MoE weights. Eight recent upstream commits resolve these critical bottlenecks:

1. **SGLang CUTLASS MoE Complete Routing Metadata & 128-Aligned Blockscales (`ac3db731b3`, PR `#45917`)**: JIT-compiles `cutlass_prepare_moe_input` and `shuffle_weighted_sum` so every unused map entry is deterministically initialized and slot-order FMA runs without FTZ.
2. **SGLang Marlin MoE Inactive Block Zeroing (`8b9b992e41`, PR `#45462`)**: Adds `kExpertIdsAreSorted`, `kZeroInactiveOutput`, and `kHasInactivePrefixCount` to zero out inactive routing rows (`expert_ids[block] == -1`).
3. **SGLang DeepSeek-V4 Decoder-Tail Token-Count Ownership (`1340044e97`, PR `#45915`) & AMD AITER FP8 DSA Prefill (`0931dfedc9`, PR `#39083`)**: Packs `(global_rows << 32) | local_rows` and restricts CUDA Graph capture of `global_num_token_non_padded` to `attn_tp_sequence_sharded` batches; enables AITER FP8 KV cache prefill on AMD GPUs.
4. **TensorRT-LLM KV Cache Manager V2 Connector Block Reuse (`2706f4dfb6` `#20180`) & Multi-Turn Generation KV Reuse (`fac6d3f794` `#19710`)**: Books only fully filled connector blocks `[divUp(tokenBegin, B), tokenEnd / B)` as reused while keeping partially written boundary blocks as misses, and finalizes V2 generation KV resources before disaggregated transfer termination.
5. **vLLM Mamba Committed Tail Boundary Guard (`d561af5dfc` `#62841`) & UVA Host-Resident MoE Sorted Path (`3b364baef5` `#62054`)**: Prevents sibling requests sharing a committed Mamba boundary from deferring each other across steps (**8 steps -> 1 step**), and forces host-resident UVA expert weights onto the sorted fused-MoE path (**3.2x** PCIe traffic reduction).

---

## 2. Source-Level Mechanics & Mathematical Modeling

### 2.1 SGLang CUTLASS MoE Input Preparation & Marlin Inactive Row Zeroing (`sglang@ac3db731b3`, `8b9b992e41`)

- **`cutlass_prepare_moe_input.cuh` (`#45917`)**:
  1. `count_expert_routes<kTransposeProblemSizes>` counts routes $c_e$ for each local expert $e \in [0, E)$ and populates `problem_sizes1[e]` ($[c_e, 2N, K]$ or transposed $[2N, c_e, K]$) and `problem_sizes2[e]` ($[c_e, K, N]$).
  2. `compute_expert_offsets` computes both dense `expert_offsets[e+1] = expert_offsets[e] + c_e` and 128-rounded `blockscale_offsets` for block-scaled FP8/FP4 GEMMs:
     $$\text{blockscale\_offsets}[e+1] = \text{blockscale\_offsets}[e] + \left\lfloor \frac{c_e + 127}{128} \right\rfloor \times 128$$
  3. `complete_route_maps` writes `output_permutation[route] = -1` for out-of-range routes ($e \notin [0, E)$) and points all `input_permutation` entries beyond `expert_offsets[E]` to token `0` so full vectorized gathers never read out-of-bounds memory.
- **`marlin_template.h` (`#45462`)**: `zero_inactive_output_rows` iterates over blocks where `expert_ids[block] == -1` and writes `{0, 0, 0, 0}` (`int4`) to every valid `route < num_routes`, preventing uninitialized workspace memory from corrupting inactive EP routes.

### 2.2 SGLang DeepSeek-V4 Decoder-Tail Counter Ownership (`sglang@1340044e97`)

In `row_bucket_graph_runner.py` and `deepseek_v4_replay_tail.py`, `_pack_token_counts(local_rows, global_rows)` encodes `(global_rows << 32) | local_rows`. `DecoderReplayTail._can_capture_moe_bound` permits CUDA Graph capture of `moe_num_token_non_padded()` only when the bound is `None`, `forward_batch.num_token_non_padded`, or **`forward_batch.global_num_token_non_padded` under `forward_batch.attn_tp_sequence_sharded`**, matching the exact set of count domains backed by graph-owned replay buffers.

### 2.3 TRT-LLM KV Cache Manager V2 Connector Reuse Accounting (`trtllm@2706f4dfb6`, `fac6d3f794`)

In `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.cpp`, `KvCache::recordConnectorReuse(int tokenBegin, int tokenEnd)` records prompt tokens `[tokenBegin, tokenEnd)` loaded by an external KV connector:

$$\text{blockBegin} = \text{divUp}(\text{tokenBegin}, B), \quad \text{blockEnd} = \left\lfloor \frac{\text{tokenEnd}}{B} \right\rfloor$$

- **Boundary Invariant**: Only uncommitted blocks completely filled by `[tokenBegin, tokenEnd)`—namely $[\text{blockBegin}, \text{blockEnd})$—count as reused blocks (`recordConnectorReuse(blockBegin, blockEnd)`). Any boundary block partially written by the forward pass remains a miss, while `tokenEnd - tokenBegin` is booked in `recordConnectorCachedTokens` before `commitPendingStats()`.
- **Multi-Turn Generation KV Reuse (`#19710`)**: `_handle_executed_batch` in `py_executor.py` invokes both `_update_v2_context_resources` and `_update_v2_generation_resources` prior to `_send_kv_async`, and passes `skip_kv_cache_manager=True` to `resource_manager.update_resources` so generated tokens are committed for cross-turn radix-tree reuse before request termination.

### 2.4 vLLM Mamba Tail Boundary Guard (`vllm@d561af5dfc`) & UVA MoE Sorting (`vllm@3b364baef5`)

1. **Mamba Shared-Prefix Sibling Concurrency (`#62841`)**: In `_cache_partial_tail_block` (`vllm/v1/core/single_type_kv_cache_manager.py`), re-caching a prompt-tail block that already had `source_block.block_hash_num_tokens == num_tokens` from a prior committed step marked its hash as newly cached in the current step, causing every sibling request hitting that boundary to be deferred to subsequent steps (running 1 sibling per step). Returning `None` immediately when `source_block.block_hash_num_tokens == num_tokens` lets all siblings execute concurrently in **1 step**.
2. **Sorted Fused-MoE Path for UVA Host-Resident Weights (`#62054`)**: In `_prepare_expert_assignment` (`vllm/model_executor/layers/fused_moe/fused_moe.py`), when `--offload-backend uva` leaves expert weights in host RAM (`_vllm_is_uva_offloaded = True`), the small-batch heuristic `num_tokens * top_k_num * 4 <= global_num_experts` previously selected `naive_block_assignment = True`, reading an expert's weights over PCIe once per `(token, expert)` pair. Adding `and not weights_host_resident` forces the sorted `moe_align_block_size` path so each active expert is read over PCIe only once per batch.

---

## 3. Deterministic Verification Summary (`own-offline`)

Validated via `python experiments/sglang-cutlass-marlin-moe-trtllm-kv-connector-vllm-mamba-uva/run.py --check`:

| Module & PR | Scenario Parameters | Pre-Fix Behavior | Post-Fix Deterministic Result |
|---|---|---|---|
| SGLang CUTLASS (`#45917`) | `8` routes (`2` inactive `-1`), `4` experts | Uninitialized tail & missing blockscale offsets | `expert_offsets=[0,3,4,6,6]`, `blockscale=[0,128,256,384,384]`, tail padded to token `0` |
| SGLang Marlin (`#45462`) | `3` blocks (middle `eid=-1`), `6` routes | Dirty workspace values in inactive block | Inactive routes `[1, 4]` explicitly zeroed; all `[0..5]` covered |
| SGLang Tail Count (`#45915`) | `local=16, global=128` | Captures unowned global count when unsharded | Captures `global` bound only when `attn_tp_sequence_sharded=True` |
| TRT-LLM Connector (`#20180`) | `B=64`, range `[16, 240)` (`224` tokens) | Miscounts partial boundary blocks as reused | Books `[1, 3)` (`2` full blocks) as reused; `2` partial boundary blocks stay misses |
| vLLM Mamba (`#62841`) | `8` siblings hitting committed Mamba boundary | `8` serial engine steps (1 per step) | **1 engine step** for all 8 siblings (**8.0x** step reduction) |
| vLLM UVA MoE (`#62054`) | `4` tokens $\times$ `top-8` (`32` pairs), `10` unique experts (`48 MiB`/expert) | `32` PCIe reads (`1536 MiB`) on naive path | `10` PCIe reads (`480 MiB`, **3.2x** PCIe traffic reduction) on sorted path |

---

## 4. Architect Guidance

1. **UVA Offloaded MoE Serving**: Always enforce `moe_align_block_size` sorting (`weights_host_resident=True`) when serving host-resident MoE weights over PCIe, preventing redundant PCIe reads when multiple tokens route to the same expert.
2. **External KV Connector Alignment**: Align external KV cache transfers to `mTokensPerBlock` boundaries whenever possible so all transferred tokens qualify for full-block reuse accounting in TRT-LLM KV Cache Manager V2.
