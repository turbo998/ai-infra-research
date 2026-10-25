# Deep Dive: SGLang MiniMax-M3 Hopper FP8 Index-K Sparse Attention, DFlash DP Attention & DraftParallelSpec, Disaggregated Diffusion Isolation, and vLLM ModelOpt Draft Mapping

> **Status**: Phase 80 (`priority: 89`) Primary-Source Code Audit  
> **Bilingual Pair**: [`sglang-minimax-m3-hopper-sparse-dflash-dp-placement-disagg-isolation-vllm-modelopt.md`](sglang-minimax-m3-hopper-sparse-dflash-dp-placement-disagg-isolation-vllm-modelopt.md)  
> **Reproducible Experiment**: [`experiments/sglang_minimax_m3_sparse_fp8_dflash_dp_attn_draft_layout_vllm_modelopt_mixed/run.py`](../../experiments/sglang_minimax_m3_sparse_fp8_dflash_dp_attn_draft_layout_vllm_modelopt_mixed/run.py)

---

## 1. Primary-Source Evidence Matrix

| Evidence ID | Upstream Commit | Key Module & Change | Category |
| --- | --- | --- | --- |
| `src-sglang-minimax-m3-hopper-sparse-attn-2026` | `sgl-project/sglang@db7b66c1b6a7` (`#46135`) | MiniMax-M3 Hopper (`sm_90`): `StoreTrait` BF16->FP8 E4M3 fused `index_k` store (`-50%` footprint/BW) + 64-bit `tile_ptrs` | `implementation` |
| `src-sglang-dflash-cuda-dp-attention-2026` | `sgl-project/sglang@38ff13ea7928` (`#46330`) | DFlash/DFlash2 CUDA `--attn-dp-size > 1`, `gather_draft_embedding`, and idle DP rank M-RoPE shape `(3, 0)` | `implementation` |
| `src-sglang-draft-parallel-layout-spec-2026` | `sgl-project/sglang@6623ea445a28` (`#46290`) | Declarative `DraftParallelSpec` (`SINGLE_GROUP` vs `TARGET_LAYOUT`) & pre-KV-pool `resolve_dflash_draft_cell_size` | `implementation` |
| `src-sglang-diffusion-disagg-fault-isolation-2026` | `sgl-project/sglang@a996b5f2244e` (`#46120`) | Disaggregated `DiffusionServer` per-request fault isolation across 10 fault classes and slot recovery | `implementation` |
| `src-vllm-modelopt-mixed-draft-quant-map-2026` | `vllm-project/vllm@5f63ea461ccd` (`#63001`) | `modelopt_mixed` normalization, MI300 `e4m3fnuz` conversion, and reverse-order `map_draft_quant_config` | `implementation` |
| `src-vllm-weight-transfer-torch-accelerator-2026` | `vllm-project/vllm@57ab2f35ac6e` (`#63457`) | Migrate RL weight transfer engines (IPC/NCCL/RDT/SparseNCCL) from `torch.cuda` to `torch.accelerator` | `implementation` |

---

## 2. MiniMax-M3 Hopper Sparse Attention: Fused BF16→FP8 `StoreTrait` & 64-Bit Tile Pointers

In `sgl-project/sglang@db7b66c1b6a7` (`#46135`), MiniMax-M3 sparse attention on Hopper (`sm_90`) is accelerated via two structural changes:

1. **Fused BF16→FP8 Index-K Store (`fused_store_kv_index.cuh`)**:
   - `StoreTrait<kHeadBytes, kCastFp8, kUsePDL>` sets `kInHeadBytes = kCastFp8 ? 2 * kHeadBytes : kHeadBytes` and `kWorkerSize = kInHeadBytes / 16` (`16` threads/head for `head_dim=128` when `kCastFp8=true`), converting BF16 inputs to `float8_e4m3fn` in registers before writing `index_k_cache`.
   - On SM90 with `kv_cache_dtype == float8_e4m3fn`, `get_minimax_sparse_index_dtype` now stores `index_k` in FP8 even when `fp8_attn_gemm` is disabled (widening to FP16 in Triton decode), cutting `index_k` memory and bandwidth by **50%** (`2.0x` reduction).
2. **64-Bit Masked Tile Pointers (`tile_ptrs_1d/2d/3d`)**:
   - Replaces removed Triton 3.9 `tl.make_block_ptr` calls with explicit `tl.arange(0, BLOCK).to(tl.int64)` pointer arithmetic, preventing 32-bit signed integer overflow when `seq_idx * stride > 2^31 - 1` (e.g., `20,000,000 * 128 = 2.56e9`).

---

## 3. DFlash CUDA DP Attention & Declarative `DraftParallelSpec`

- **CUDA DP Attention for DFlash/DFlash2 (`#46330`)**: Enables `--attn-dp-size > 1` with `--enable-dp-lm-head`, adds `gather_draft_embedding` so quantized `VocabParallelEmbedding` weights are looked up locally before all-gather, and initializes `ret.mrope_positions` with shape `(3, 0)` on idle DP ranks so M-RoPE does not crash during target verification.
- **Pre-Pool `DraftParallelLayout` Resolution (`#46290`)**: Resolves `DraftParallelSpec` (`SINGLE_GROUP` vs `TARGET_LAYOUT`) before KV pool allocation so `resolve_dflash_draft_cell_size` prices draft KV cache cells using the draft's actual `attn_tp_size` rather than the global TP size (preventing a **2.0× underestimation** at `TP=8, DP=2`).

---

## 4. Disaggregated Diffusion Fault Isolation (`#46120`) & vLLM Reverse-Order Layer Mapping (`#63001`)

- **Disaggregated Diffusion Isolation (`#46120`)**: Isolates 10 classes of malformed frames, duplicate `request_id` collisions, and worker role errors in `DiffusionServer` without crashing the routing loop or leaking worker receive slots.
- **vLLM `map_draft_quant_config` Reverse-Order Remapping (`#63001`)**: Iterates `reversed(range(num_hidden_layers))` when shifting `layers.{i}` to `layers.{start_layer_id + i}` for Eagle/MTP draft quantization configs, preventing the forward-order cascade bug where `start_layer_id = 1` collapsed `layers.0, layers.1, layers.2` into `layers.3`.
