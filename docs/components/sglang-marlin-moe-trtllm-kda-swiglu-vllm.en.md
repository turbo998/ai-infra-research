# SGLang Marlin WNA16 MoE Sentinel Routing & Disaggregated Heartbeat Reset, TensorRT-LLM SM100 SwiGLU Epilogue Fusion & KDA Integration, and vLLM Mamba/Spec-Decode Guards

> **Bilingual Navigation / 双语导航**: [中文版](sglang-marlin-moe-trtllm-kda-swiglu-vllm.md) | [English Version](sglang-marlin-moe-trtllm-kda-swiglu-vllm.en.md)
> **Evidence IDs**: `src-sglang-marlin-moe-trtllm-kda-swiglu-vllm-2026` · `ev-sglang-marlin-moe-trtllm-kda-swiglu-vllm-2026` (`priority: 73`, Phase 64 Track B)
> **Reproducible Experiment**: `python experiments/sglang-marlin-moe-trtllm-kda-swiglu-vllm/run.py --check`

---

## 1. Architect Problem Statement & Engineering Context

In large-scale MoE serving, disaggregated prefill-decode (PD) orchestration, Blackwell SM100 dense/hybrid execution, and Mamba/speculative decoding, serving engines face four critical bottlenecks and correctness hazards:

1. **Marlin WNA16 MoE Inactive Prefix Blocks and Out-of-Bounds Expert IDs**: When virtual experts or masked routes produce `-1` sentinel IDs (mapped to bucket `0` under the `+1` offset convention), scanning inactive blocks serially per thread or scheduling empty CTA blocks wastes execution bandwidth, while negative IDs outside `[-1, E)` (e.g., `-99`) index out-of-bounds in `cumsum_buffer`.
2. **Stale Heartbeat Miss Counts Dropping Recovered Prefill Nodes in SGLang**: When a prefill worker reaches `max_failures` missed heartbeats and is evicted by `_handle_node_failure`, retaining its `heartbeat_failures[addr]` entry causes a subsequent reconnection to be dropped immediately on its very first transient timeout.
3. **Redundant `[M, 2I]` BF16 HBM Traffic and Double Rounding in TensorRT-LLM GatedMLP**: Writing the `[M, 2I]` gate/up projection to HBM in BF16 and re-reading it in a standalone SwiGLU kernel generates hundreds of megabytes of avoidable HBM traffic per layer at large token counts (e.g., MiniMax-H3 $M=4096, I=14336$).
4. **vLLM Mamba Shared-Prefix Sibling Serialization and V2 Speculative Decoding RoPE OOB**: Re-caching an already committed Mamba prompt-tail block marks its hash as newly cached in the current step, deferring sibling requests to run one per step; meanwhile, the V2 model runner drafts at every accepted target length and will read out-of-bounds on the draft rotary `cos_sin_cache` if the draft's positional limit is smaller than the target's.

---

## 2. Core Source Changes & Mathematical Modeling

### 2.1 SGLang Marlin WNA16 MoE `int2` Prefix Publication & Unsigned Sentinel Mapping (`#45462`, `8b9b992`) and Disaggregated Heartbeat Reset (`#45922`, `16aa816`)

1. **Unsigned-Shift `sentinel_expert_bucket`**: In `python/sglang/kernels/jit/csrc/moe/moe_align_kernel.cu`:
   ```cpp
   template <typename scalar_t>
   __device__ __forceinline__ int32_t sentinel_expert_bucket(scalar_t expert_id, int32_t num_buckets) {
     using unsigned_t = std::make_unsigned_t<scalar_t>;
     const unsigned_t shifted = static_cast<unsigned_t>(expert_id) + unsigned_t{1};
     return shifted < static_cast<unsigned_t>(num_buckets) ? static_cast<int32_t>(shifted) : 0;
   }
   ```
   Using unsigned wraparound maps `-1 -> 0`, valid experts $i \in [0, E-1] \to i+1$, and clamps any other negative (`-99`) or out-of-range (`>= E`) ID to sentinel bucket `0` with a single comparison.
2. **Single-Transaction `int2` Count Publication & Inactive Prefix Skip**: When `kWriteInactivePrefixCount=true`, `align_single_token_kernel` and `moe_align_block_size` publish `make_int2(num_tokens_past_padded, inactive_prefix_blocks)` in one aligned store. The Marlin kernel (`marlin_template.h`) loads both values via a single `__ldg(reinterpret_cast<const int2*>(num_tokens_past_padded_ptr))`, skipping the `inactive_prefix_blocks` sentinel blocks directly (cutting **37.5%** of scheduled blocks when 6 of 16 blocks are inactive) and zeroing inactive output rows via `zero_inactive_output_rows`.
3. **Disaggregated Prefill Heartbeat Counter Reset (`#45922`, `16aa816`)**: `CommonKVManager._handle_node_failure` now calls `self.heartbeat_failures.pop(failed_bootstrap_addr, None)`. Under probe sequence `["miss", "miss", "miss", 200, "miss", 200]` with `max_failures=2`, a reconnected prefill node survives a single transient missed heartbeat instead of being prematurely evicted a second time.

### 2.2 TensorRT-LLM SM100 QuACK BF16 Gate/Up + SwiGLU Epilogue Fusion (`#19773`, `90d4dcb`) & FlashInfer KDA Prefill Integration (`#20653`, `222676c`)

1. **SM100/SM103 QuACK `gemm_act` Epilogue Fusion**: `tensorrt_llm/_torch/modules/gate_up_swiglu_quack.py` invokes QuACK's `gemm_act(..., activation="swiglu", concat_layout=("B",), tuned=False)` on TRT-LLM's native `[gate; up]` weight layout, applying $\text{SiLU}(\text{gate}) \odot \text{up}$ directly to FP32 accumulators in the GEMM epilogue and writing `[M, I]` BF16 once:
   $$\text{Bytes}_{\text{unfused}} = 10 \cdot M \cdot I\text{ B} \quad \longrightarrow \quad \text{Bytes}_{\text{fused}} = 2 \cdot M \cdot I\text{ B}$$
   On MiniMax-H3 ($M=4096, K=5376, I=14336$), per-layer activation HBM traffic drops from **560.0 MiB** to **112.0 MiB**, saving **448.0 MiB per layer (-80.0%)** and reducing BF16 roundings from 2 to 1.
2. **FlashInfer KDA `bt16` Fused-Convolution Prefill Integration (`#20653`, `222676c`)**: Replaces three standalone Blackwell KDA files (`akk_inverse.py`, `fused_k123.py`, `fused_k1234.py`, **-5,706 lines**) with vendored FlashInfer `kda_chunked_bt16_small_bh.py` plus a 207-line `flashinfer-kda-bt16.patch` adding indexed recurrent-state pool access (`mStateIndices`).

### 2.3 vLLM Mamba Prompt-Tail Re-Caching Guard (`#62841`, `d561af5`) & V2 Speculative Decoding RoPE Bounds (`#63251`, `37cf712`)

1. **Mamba Sibling Same-Step Admission (`#62841`)**: `SingleTypeKVCacheManager._cache_partial_tail_block` returns `None` immediately when `source_block.block_hash_num_tokens == num_tokens`, preventing a committed Mamba boundary from being re-marked as cached in the current step and admitting all 3 sibling requests in **1 engine step** instead of 3 serial steps (**`3.0x`** faster sibling admission).
2. **V2 Speculative Decoding RoPE Cache & `max_model_len` Alignment (`#63251`)**: `SpeculativeConfig` raises draft `max_position_embeddings` to match `target_model_config.max_model_len` (e.g., `2048 -> 131072` for EAGLE3 on Llama-3.1) for position-sharing methods whose `rope_type` is in `_CACHE_ONLY_ROPE_TYPES = {"default", "linear", "llama3", "ntk", "proportional"}`, and otherwise lowers the V2 engine `max_model_len` to `draft_max_model_len` via `_limit_max_model_len_to_draft()`.

---

## 3. Deterministic Verification Summary (`own-offline`)

Running `python experiments/sglang-marlin-moe-trtllm-kda-swiglu-vllm/run.py --check` verifies:

| Framework & Module | Mechanism | Quantitative Verification | Engineering Impact |
|---|---|---|---|
| **SGLang Marlin MoE** (`#45462`) | `sentinel_expert_bucket` + `int2` prefix counts | `-1/-99/256 -> bucket 0`, `0 -> 1`; skips `6/16` inactive blocks | Prevents OOB atomic writes and skips **37.5%** inactive CTA blocks |
| **SGLang PD Disagg** (`#45922`) | `heartbeat_failures.pop(failed_addr, None)` | Survives single miss after reconnect (drops `2 -> 1`) | Eliminates false eviction of recovered prefill workers |
| **TRT-LLM SM100 SwiGLU** (`#19773`) | QuACK `gemm_act` FP32 epilogue fusion | MiniMax-H3 activation HBM `560 MiB -> 112 MiB` | Saves **448.0 MiB/layer (-80.0%)**, BF16 roundings `2 -> 1` |
| **TRT-LLM KDA Prefill** (`#20653`) | FlashInfer `bt16` + `mStateIndices` patch | Removes **5,706 lines** of legacy kernels for 207-line patch | Unifies paged state-pool indirection and fused conv prefill |
| **vLLM Mamba + Spec** (`#62841`, `#63251`) | `block_hash_num_tokens` guard + RoPE classification | 3 siblings admitted in `1` step vs `3` (`3.0x`); EAGLE3 `131072` | Fixes Mamba shared-prefix deferral & V2 Spec RoPE OOB |
