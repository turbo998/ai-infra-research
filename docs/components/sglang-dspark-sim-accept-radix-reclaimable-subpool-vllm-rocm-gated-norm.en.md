# Deep Dive: SGLang DSpark CUDA Simulated-Acceptance Fast Path, Radix-Cache Reclaimable Value Lengths, Sub-Pool End-First Allocation & vLLM ROCm Gated RMSNorm

> **Lang / 语言**: [中文](./sglang-dspark-sim-accept-radix-reclaimable-subpool-vllm-rocm-gated-norm.md) | **English (Current)**
>
> **Evidence Class**: `implementation` (verified against upstream commits) + `own-offline` (deterministic Python 3.12 standard-library model `experiments/sglang_dspark_sim_accept_radix_reclaimable_subpool_vllm_rocm/run.py --check`).

---

## 1. Architectural Context & Core Problem Statement

Across speculative decoding, hybrid prefix caching (Full Attention + Sliding-Window Attention + Mamba state), and compiler pattern fusion on ROCm/CUDA, four upstream commits landed on 2026-10-20 resolve critical runtime bottlenecks and accounting invariants:

1. **Redundant Rejection-Sampling Softmax During DSpark Simulated Acceptance on CUDA**: SGLang's DSpark speculative execution supports `simulated_correct_len` for deterministic acceptance-rate benchmarking. However, `dspark_verify.py` and `dspark_worker_v2.py` previously gated `accept_sampling_info = None`, `simulate_fold`, and `_simulate_acc_greedy` behind `_is_hip`. On NVIDIA CUDA, simulated-acceptance runs still passed `sampling_info` into `accept_draft_tokens`, computing full vocabulary draft/target softmax and rejection sampling only for `correct_len` to be overwritten immediately afterward.
2. **Ledger Drift Between Logical `value_len` and Allocator-Backed `reclaimable_value_len` in Unified Radix Cache**: In `UnifiedTreeCore` (both Python and Rust `sglang-radix-tree`), a node's stored device-index tensor length (`value_len`) can diverge from the number of allocator slots reclaimed upon eviction when a component hands back rows early, attaches side backing, or holds a root/tombstone entry. Mixing `len(value)` on insert/lock/unlock with `reclaimable_tokens` during `sanity_check` caused `component_evictable_size_` and `component_protected_size_` to drift.
3. **Interior Fragmentation in Unified-Memory `MultiEndedAllocator`**: Over a shared `UnifiedKVPool`, the FULL sub-pool grows upward (`grow_direction="up"`, from low physical addresses) while the SWA sub-pool grows downward (`grow_direction="down"`, from high physical addresses). Previously, `take_physical` sliced `self._free_phys_pages[:n_drain]` in unsorted FIFO order, handing out interior free pages that fragmented the central contiguous band needed by compaction and H2D transfers.
4. **Graph-Matching Gap for `RMSNormGated` on vLLM ROCm**: Hybrid linear-attention models such as Qwen3.8-27B rely on `RMSNormGated` from Flash Linear Attention (`rmsnorm_fn`). Calling `rmsnorm_fn` directly inside `RMSNormGated.forward_cuda` without registering a custom op prevented `torch.compile` and ROCm AITER fusion passes (`RMSNormGatedQuantPattern`) from matching the gated norm as a single opaque operator node.

---

## 2. First-Hand Source Implementation Analysis

### 2.1 Unlocking DSpark Simulated-Acceptance Fast Paths on CUDA (`7d9bc10493`)

In `sgl-project/sglang` commit [`7d9bc10493`](https://github.com/sgl-project/sglang/commit/7d9bc10493275a371989bd03a13147588005f136) (PR `#45421`, 2026-10-20), `dspark_verify.py` and `dspark_worker_v2.py` drop the `_is_hip` gate:

```python
# dspark_verify.py:
simulate = self._simulated_correct_len is not None
accept_sampling_info = None if simulate else sampling_info

# dspark_worker_v2.py:
simulate_fold = envs.SGLANG_DSPARK_SIMULATE_FOLD.get()
self._simulate_acc_greedy = (
    self._simulate_acc_len > 0 and envs.SGLANG_SIMULATE_ACC_GREEDY.get()
)
```

For batch size $B=16$, draft length $K=5$, and vocabulary size $V=152{,}064$, setting `accept_sampling_info = None` eliminates $2 \times B \times K \times V = 24{,}330{,}240$ discarded softmax elements per verification step on CUDA.

### 2.2 Separating Logical `value_len` and `reclaimable_value_len` (`0b536ca112`)

In `sgl-project/sglang` commit [`0b536ca112`](https://github.com/sgl-project/sglang/commit/0b536ca112652008d5fa8766df7258b0700109f3) (PR `#45238`, 2026-10-20):
- `value_len(node)` / `device_value_len(ct)` returns the logical length of the stored device-index tensor.
- `reclaimable_value_len(node)` / `device_reclaimable_value_len(ct)` returns the exact number of pool tokens the allocator gets back when the node's device value is evicted (`0` for root or absent device values). Every ledger mutation site across `FULL`, `SWA`, and `MAMBA` in Python and Rust now uses `reclaimable_value_len`.
- `adjust_component_size` validates `updated = sizes[component_type] + delta` before assignment and raises `ValueError` without mutating the ledger if `updated < 0`.

### 2.3 Lazy Device Sort and End-First Sub-Pool Page Allocation (`bab4a808d8`)

In `sgl-project/sglang` commit [`bab4a808d8`](https://github.com/sgl-project/sglang/commit/bab4a808d81fd47a039889a3f71514679d799136) (PR `#45222`, 2026-10-20), `MultiEndedAllocator` in `unified_sub_pool.py` tracks tensor identity via `self._free_phys_sorted` to sort `_free_phys_pages` on the device only when rebound after frees, and drains from `"low"` when `grow_direction == "up"` or `"high"` when `grow_direction == "down"`:

```python
def _take_free_pages_from_end(self, n: int, end: str) -> torch.Tensor:
    n_free = int(self._free_phys_pages.shape[0])
    if n == n_free:
        taken, rest = self._free_phys_pages, self._free_phys_pages[:0]
    else:
        free = self._sorted_free_phys_pages()
        if end == "low":
            taken, rest = free[:n], free[n:]
        else:
            taken, rest = free[n_free - n :], free[: n_free - n]
    self._free_phys_pages = rest
    self._free_phys_sorted = rest
    return taken
```

### 2.4 vLLM ROCm `fla_rms_norm_gated` Custom Op Registration & Default (`9da5f2d310`)

In `vllm-project/vllm` commit [`9da5f2d310`](https://github.com/vllm-project/vllm/commit/9da5f2d3103df460dfca53df4ab4884def5c935f) (PR `#60250`, 2026-10-20), `vllm/model_executor/layers/layernorm.py` registers `torch.ops.vllm.fla_rms_norm_gated` via `direct_register_custom_op`, `RMSNormGatedMatcher.forward_custom` matches the custom op, and `RocmPlatform.apply_config_platform_defaults` appends `"+rms_norm_gated"` by default whenever `model_config.quantization is None`.

---

## 3. Deterministic Verification (`own-offline`)

Running `python experiments/sglang_dspark_sim_accept_radix_reclaimable_subpool_vllm_rocm/run.py --check` verifies:
- `24,330,240` redundant softmax elements saved per DSpark verify step on CUDA;
- Zero ledger drift (`33` reclaimable vs `40` logical slots) and atomic `ValueError` rejection on negative `adjust_component_size`;
- End-first sub-pool allocation (`[0, 2]` for FULL `"low"` and `[12, 14]` for SWA `"high"`, preserving `[4, 6, 8, 10]` in the interior without re-sorting on the second drain);
- Default `+rms_norm_gated` activation on ROCm unquantized configs.

---

## 4. Architect Design Guidelines & Falsifiable Follow-Ups

1. **Key Cache Admission Ledgers to Reclaimable Allocator Units, Never Logical Index Counts**: Whenever a prefix-tree component supports partial row release or shared backing, admission control must sum `reclaimable_value_len` so prefill admission never over-commits physical KV memory.
2. **Preserve Interior Contiguity in Dual-Ended Pools via Directional Free-List Draining**: Allocating free pages nearest each sub-pool's anchor end keeps interior gaps contiguous for compaction.
3. **Falsifiable Follow-Up Question**: What is the end-to-end token-per-second speedup on MI300X/MI350X for Qwen3.8-27B when combining `+rms_norm_gated` with AITER `+quant_fp8` fusion once quantized serving is validated?
