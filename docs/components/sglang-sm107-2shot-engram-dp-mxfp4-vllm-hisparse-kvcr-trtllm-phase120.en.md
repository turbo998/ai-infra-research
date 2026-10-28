# SGLang SM107 2-Shot AllReduce Heuristics, DeepSeek-V4.1 Engram DP Auto-Collective Selection, Triton 3.9 Swap-AB Scale Broadcast, Split-Expert MXFP4 Marlin MoE, vLLM Spec-Decode LoRA Logits Alignment, HiSparse State-Slot Indexing, KVCR Prefix Recency Alignment & TRT-LLM B300 MoE Re-Enablement (Phase 120)

> **Module Positioning**: Designed for cloud Solution Architects optimizing LLM serving engines (SGLang, vLLM, TensorRT-LLM). Based on 10 upstream commits landed between 2026-10-27 and 2026-10-28, this document examines Rubin (`sm_107`) custom AllReduce thresholds, DeepSeek-V4.1 Engram DP `all_to_all` traffic compression, split-expert MXFP4 Marlin execution, speculative decoding `lm_head` LoRA alignment, HiSparse residency layout transposition, and KVCR prefix-ordered cache recency.
> **Evidence Class**: `implementation` (verified at fixed commit SHAs) + `author-benchmark` (upstream PR measurements) + `own-offline` (reproducible Python standard-library model in this repository).

---

## 1. Architectural Problem Statement & 10 Upstream Commits Overview

| # | Repository & Commit | Date | Core Mechanism & Architectural Change | Evidence Class |
| :--- | :--- | :--- | :--- | :--- |
| B1 | `sgl-project/sglang` [`08b9ec6`](https://github.com/sgl-project/sglang/commit/08b9ec6083b419b2e13c431f64da5f5841d75405) | 2026-10-28 | `custom_all_reduce_v2`: SM107 (Rubin VR) TP4/TP8/TP16 enable `two_shot_push = 16 MB` and lower 1-shot crossover thresholds | `implementation` |
| B2 | `sgl-project/sglang` [`e35b5bc`](https://github.com/sgl-project/sglang/commit/e35b5bccf0a51fb2ab5ff13dc72f295be9c49621) | 2026-10-28 | `engram.py`: DeepSeek-V4.1 Engram DP lookup `auto` collective selector (`all_to_all` cuts collective bytes to $1/\text{TP}$) | `implementation` + `author-benchmark` |
| B3 | `sgl-project/sglang` [`faa56e2`](https://github.com/sgl-project/sglang/commit/faa56e21a2d9493ef3d49428c74a3cb907f0ef00) | 2026-10-28 | `fused_moe_triton_kernels`: Fix scalar block scale broadcast with `swap_ab` on Triton 3.9 (`a_scale * b_scale[None, :]`) | `implementation` |
| B4 | `sgl-project/sglang` [`bb4bb01`](https://github.com/sgl-project/sglang/commit/bb4bb01df4ab0d1e9d17b194151295376a0b3838) | 2026-10-28 | `compressed_tensors_w4a16_mxfp4_moe`: Serve split-expert MXFP4 MoE via FP4 Marlin (`S_TYPE=kFE8M0fnu`, `group_blocks=2`) | `implementation` |
| B5 | `sgl-project/sglang` [`2eab19a`](https://github.com/sgl-project/sglang/commit/2eab19a385bf2982a9dc7a4408b068a13bf93bc4) | 2026-10-28 | `deepseek_v4_backend`: Refuse `glue_draft_metadata` fusion when `self.needs_cpu_seq_lens` is True | `implementation` |
| B6 | `vllm-project/vllm` [`b71c1d5`](https://github.com/vllm-project/vllm/commit/b71c1d59cfbe1aaeb8be3507bc20545fcbe40072) | 2026-10-28 | `kimi_k3/amd/model.py`: Forward `process_weights_after_loading()` on ROCm Kimi-K3 root model to fold `q_b_proj / W_UK` | `implementation` |
| B7 | `vllm-project/vllm` [`06e172b`](https://github.com/vllm-project/vllm/commit/06e172bd14d0c7e5490934968f5611df4a2a0fb4) | 2026-10-28 | `v1/worker/gpu/lora_utils.py`: Align MRV2 `lm_head` LoRA mapping with logits rows via `lora_ids.repeat(num_logits)` | `implementation` |
| B8 | `vllm-project/vllm` [`66f1d0f`](https://github.com/vllm-project/vllm/commit/66f1d0fd64d1da071fa939909a9c89df1be35891) | 2026-10-28 | `hisparse`: Transpose residency to `[resident_groups, max_num_reqs, max_num_pages]` and index prefill rows by `request_state_indices` | `implementation` |
| B9 | `vllm-project/vllm` [`e3883b7`](https://github.com/vllm-project/vllm/commit/e3883b7444b8ecbd5f0c41d1fc12fb967bc460e8) | 2026-10-27 | `kvcr/manager.py`: Align KVCR remote cache recency with request prefixes via `align_sequence` and `processed_tokens` | `implementation` |
| B10 | `NVIDIA/TensorRT-LLM` [`f4bc8ee`](https://github.com/NVIDIA/TensorRT-LLM/commit/f4bc8ee407803a7682b76d00df4aa7df88e2afc4) | 2026-10-28 | `test_moe_backend.py`: Re-enable B300 (SM103) TRTLLM-Gen block-scale MoE cases after route-map over-read fix | `implementation` |

---

## 2. Deep Technical Analysis & Formulations

### 2.1 SGLang SM107 Custom AllReduce & DeepSeek-V4.1 Engram DP Collectives (`08b9ec6`, `e35b5bc`)

1. **SM107 (Rubin VR) 2-Shot Push Heuristics (`08b9ec6`)**:
   - Enables `two_shot_push = 16 MB` on TP4, TP8, and TP16 in `_sm107_configs` while lowering the 1-shot upper bounds so medium-sized tensors transition earlier into `2shot_push`:
     - **TP4**: Graph `6.938 MB -> 1.000 MB`, Eager `9.812 MB -> 1.000 MB`;
     - **TP8**: Graph & Eager `2.438 MB -> 384 KB` (**6.5013x** reduction);
     - **TP16**: Graph & Eager `832 KB -> 256 KB` (**3.25x** reduction).
2. **DeepSeek-V4.1 Engram DP Lookup `auto` Collective (`e35b5bc`)**:
   - DeepSeek-V4.1 Engram tables have $C = 24$ hash columns sharded across the TP group. Under DP attention with $N$ total tokens and embedding dimension $D$:
     - `reduce_scatter` exchanges rows for all $C=24$ columns across all TP ranks, moving $B_{\text{RS}} = \frac{\text{TP}-1}{\text{TP}} \cdot N \cdot C \cdot D \cdot \text{sizeof(dtype)}$ bytes per rank;
     - When $C \bmod \text{TP} = 0$ ($\text{TP} \in \{2, 3, 4, 6, 8, 12\}$), each rank owns exactly $k = C / \text{TP}$ columns. `all_to_all` looks up only those $k$ columns and dispatches them via `all_to_all_single`, moving:
       $$B_{\text{A2A}} = \frac{1}{\text{TP}} B_{\text{RS}}$$
   - **Author-Benchmarked Gains (`author-benchmark`)**: Across 144 shapes on GB300, B300, and H200 (DP 2–8, host tables, `512–65,536` tokens/rank), `all_to_all` was fastest in **121 of 144 shapes** and **6%–60% faster** at $\ge 8,192$ tokens/rank.

### 2.2 SGLang Split-Expert MXFP4 Marlin MoE & Triton 3.9 Swap-AB Fix (`faa56e2`, `bb4bb01`, `2eab19a`)

- **Triton 3.9 `swap_ab` Broadcast Fix (`faa56e2`)**: Replaces invalid 0D slice `a_scale[:, None] * b_scale` with `a_scale * b_scale[None, :]` when `swap_ab=True` and `BLOCK_SIZE_N <= group_n`.
- **Split-Expert MXFP4 Marlin MoE (`bb4bb01`)**: Adds `CompressedTensorsW4A16Mxfp4MoE` for split-expert checkpoints (`experts.{e}.{gate,up,down}_proj.*`) and parameterizes the C++ Marlin template on scale type `s_type_id` (`kFE8M0fnu` with `group_blocks = 2` and `dequant_skip_flop = false` for MXFP4 vs `kFE4M3fn` with `group_blocks = 1` for NVFP4).
- **DSV4 Draft Metadata Glue Guard (`2eab19a`)**: Refuses `glue_draft_metadata` when `self.needs_cpu_seq_lens` is True so CPU sequence lengths never fall behind GPU-incremented `seq_lens`.

### 2.3 vLLM Spec-Decode LoRA Alignment, HiSparse State-Slot Indexing & KVCR Prefix Recency (`b71c1d5`, `06e172b`, `66f1d0f`, `e3883b7`, `f4bc8ee`)

1. **Speculative Decoding MRV2 `lm_head` LoRA Alignment (`06e172b`)**: Uses `prompt_lora_mapping = tuple(lora_ids.repeat(num_logits))` where `num_logits = np.diff(cu_num_logits_np)` (`1 + num_drafts` rows per request) so every draft verification row uses its own request's LoRA adapter.
2. **HiSparse Residency Transposition & State-Slot Indexing (`66f1d0f`)**: Transposes `residency` to `[resident_groups, max_num_reqs, max_num_pages]` to eliminate per-step `index_select` copies and maps batch request index `req` through `idx_mapping = request_state_indices` inside `sparse_mla_index_remap_kernel`.
3. **KVCR Prefix Recency Alignment (`e3883b7`)**: Invokes `align_sequence` in prefix order upon load/store completion and finalizes recency at request completion up to:
   $$\text{processed\_tokens} = \min\big(\max(0, \text{computed\_tokens} - \text{in\_flight}), \max(\text{prompt\_tokens}, \text{num\_tokens} - 1)\big)$$
   so earlier prefix blocks are never evicted ahead of later blocks from the same request.

---

## 3. Offline Reproducible Verification (`own-offline`)

Experiment directory: `experiments/sglang_sm107_2shot_engram_dp_mxfp4_vllm_hisparse_kvcr_trtllm_phase120/`

- **Command**: `python experiments/sglang_sm107_2shot_engram_dp_mxfp4_vllm_hisparse_kvcr_trtllm_phase120/run.py --check`
- **Verified Outputs**:
  - `sm107_2shot_allreduce.tp8_1shot_threshold_reduction_ratio = 6.5013`;
  - `dsv41_engram_dp_collective.divisible_tp_sizes = [2, 3, 4, 6, 8, 12]`;
  - `vllm_and_trtllm_fixes.spec_lora_mapping_matches_logits_rows = true`;
  - `vllm_and_trtllm_fixes.kvcr_processed_tokens = 112` (`["b0", "b1", "b2"]`).
