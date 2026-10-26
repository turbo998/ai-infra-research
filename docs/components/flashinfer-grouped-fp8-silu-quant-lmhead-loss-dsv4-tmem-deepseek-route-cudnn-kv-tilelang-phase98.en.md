# FlashInfer Grouped FP8 Fused SiLU-Quant Pair+Tail & In-Place Rebinding, LM-Head Loss Single-Sync Compaction, DSv4 TMEM Lane-Quarter Isolation, DeepSeek Routing Bounds, cuDNN Loose KV Bound & TileLang Evolution (Phase 98)

> **中文版本**: [flashinfer-grouped-fp8-silu-quant-lmhead-loss-dsv4-tmem-deepseek-route-cudnn-kv-tilelang-phase98.md](flashinfer-grouped-fp8-silu-quant-lmhead-loss-dsv4-tmem-deepseek-route-cudnn-kv-tilelang-phase98.md)
> **Priority**: `107` · **Question ID**: `flashinfer-grouped-fp8-silu-quant-lmhead-loss-dsv4-tmem-deepseek-route-cudnn-kv-tilelang-phase98` · **Evidence Class**: `implementation` + `author-benchmark` + `own-offline`

---

## 1. Executive Summary & Architectural Findings

Across 10 zero-overlap upstream commits in FlashInfer and TileLang on Blackwell (`sm_100a`/`sm_103a`) and Ascend/CUDA targets, Phase 98 establishes the following architectural invariants:

1. **SM100/SM103 Grouped FP8 Fused SiLU-Quant Pair + Tail Kernels and Per-Call Operand Rebinding (`flashinfer#7454`, `1290367e8d`; `flashinfer#6049`, `492fb0128b`)**: Large-$M$ MoE `gate_up` + SwiGLU + FP8 quantization routes achieve `1.037x–1.154x` CUDA-graph speedups (`1.050x–1.146x` eager) while leaving unchanged routes neutral (`0.984x–1.008x`). Rebinding per-call token operands (`a`, `a_scale`, `m_indices`, `out_q`, `out_s`) directly inside `launch()` eliminates staging copies (`512 MiB/step` saved across 32 layers) and skips writes for `m_indices == -1` padding rows.
2. **Chunked LM-Head Loss Single-Sync Valid-Row Compaction and Split-Tail `dX` Wave (`flashinfer#7449`, `c93782c532`)**: Reusing the forward pass's compacted valid-row index during the backward pass cuts device synchronizations from `2` to `1` per call and combines with dedicated log-probability and split-tail `dX` GEMM kernels for `1.024x–1.068x` speedups on GB300 and `1.023x–1.051x` on B200.
3. **DSv4 BF16 H128 Sparse-MLA TMEM Lane-Quarter Isolation (`flashinfer#6895`, `c7f1e3dfe5`)**: Blackwell `tcgen05.ld`/`tcgen05.st` instructions require each warp to access its own 32-lane TMEM quarter. Indexing `meta_addr` with `score_row_base << 16` (`warp_idx * 32`) instead of the shared `head_row_base << 16` isolates all 4 warps into disjoint TMEM lane quarters.
4. **DeepSeek Fused Routing Bounded Group Candidate (`flashinfer#7111`, `6137d20476`)**: Initializing `group_index_candidate = (unsigned int)(n_group - 1)` instead of `0xFFFFFFFF` (`4294967295`) guarantees by construction that selected group indices remain strictly `< n_group` even under NaN or degenerate scores.
5. **Kimi K3 JIT Source Packaging and cuDNN Paged Prefill Loose KV Bound Intersection (`flashinfer#7107`, `f810e74b5e`; `flashinfer#7382`, `395c50712c`)**: Bundles `kimi_k3_latent_moe` and `kimi_k3_fused_router` `csrc/**` files in wheels and clamps `graph_s_kv = min(graph_s_kv, block_tables.shape[1] * k_cache.shape[2])` (e.g., `min(128, 3 * 16) = 48`) so loose caller sequence upper bounds never read past the physical page table.
6. **TileLang Ascend Base Offset Preservation, Periodic Buffer Lifetimes & CUDA 16-Bit Math (`tilelang#3627`, `f5c6b44cfe`; `#3606`, `ed06ed1f01`; `#3172`, `43f13c7ad1`)**: Preserves `phys_buf.elem_offset + sum(min_i * stride_i)` (`388` instead of `132`) in Ascend DMA copies and strided slices, keeps periodic buffer lifetimes intact across parent loop trips by restricting early exit to `child == nullptr`, and adds 11 CUDA 16-bit inverse/binary math intrinsics with integer-stepped `nextafter` that preserves BF16 subnormals.

---

## 2. Primary Source Evidence & Quantitative Metrics

| # | Evidence ID / Commit | Mechanism & Hardware Contract | Quantitative Gain / Correctness Invariant |
| :--- | :--- | :--- | :--- |
| 1 | `flashinfer-cake-grouped-fp8-fused-silu-quant-pair-tail-7454-1290367-2026` (`1290367e8d`) | `cake_grouped_fp8_fused_silu_quant`: SM100/SM103 Pair+Tail large-$M$ kernels | CUDA graph `1.037x–1.154x`; eager `1.050x–1.146x`; neutral `0.984x–1.008x` |
| 2 | `flashinfer-cake-grouped-fp8-gemm-per-call-operands-1-p-6049-492fb01-2026` (`492fb0128b`) | `grouped_fp8_gemm`: per-call `launch(a, a_scale, m_indices, out_q, out_s)` + `-1` pad | `0` staging copies (`512 MiB/step` saved), in-bounds `-1` padding skip |
| 3 | `flashinfer-cake-lm-head-loss-log-probability-kernels-s-7449-c93782c-2026` (`c93782c532`) | `cake_lm_head_loss`: forward valid-row index reuse + split-tail `dX` GEMM | Device syncs `2 -> 1`; GB300 `1.024x–1.068x`; B200 `1.023x–1.051x` |
| 4 | `flashinfer-cake-dsv4-keep-bf16-h128-softmax-stats-in-w-6895-c7f1e3d-2026` (`c7f1e3dfe5`) | `sparse_mla_fwd_bf16_h128.cu`: `score_row_base << 16` TMEM lane-quarter offset | All 4 warps isolated in disjoint 32-lane TMEM quarters (`tcgen05.ld/st`) |
| 5 | `flashinfer-cake-deepseek-fused-routing-bound-selected-7111-6137d20-2026` (`6137d20476`) | `routing_deepseek.cu`: `group_index_candidate = n_group - 1` fallback | Selected group index guaranteed `0 <= idx < n_group` (no `0xFFFFFFFF` OOB) |
| 6 | `flashinfer-build-package-kimi-k3-jit-sources-7107-f810e74-2026` (`f810e74b5e`) | `pyproject.toml`: package `kimi_k3_latent_moe` & `kimi_k3_fused_router` `csrc/**` | Wheel distributions include complete Kimi K3 JIT CUDA sources |
| 7 | `flashinfer-cudnn-accept-loose-kv-upper-bounds-in-paged-7382-395c507-2026` (`395c50712c`) | `cudnn_Helpers.py`: `min(graph_s_kv, block_tables.shape[1] * k_cache.shape[2])` | Clamps loose `graph_s_kv=128` to physical capacity `48` (`3 * 16`) |
| 8 | `tilelang-ascend-preserve-buffer-elem-offset-in-dma-cop-3627-f5c6b44-2026` (`f5c6b44cfe`) | `ascend.cc`: `phys_buf.elem_offset + sum(min_i * stride_i)` | Strided slice offset `132 -> 388` (`base=256` preserved) |
| 9 | `tilelang-ascend-preserve-complete-periodic-buffer-life-3606-ed06ed1-2026` (`ed06ed1f01`) | `memory_planning.cc`: root-only (`child == nullptr`) periodic loop cutoff | Zero false buffer aliasing across parent loop iterations |
| 10 | `tilelang-cuda-support-missing-16-bit-inverse-and-binar-3172-43f13c7-2026` (`43f13c7ad1`) | `half_math.h`: 11 16-bit math intrinsics + integer-stepped `nextafter` | Exact BF16 subnormal progression (`0x0001 -> 0x0002`) |

---

## 3. Architect Design Exercises & Falsifiable Questions

1. **Calculation Exercise (TMEM Lane-Quarter Addressing)**: In `sparse_mla_fwd_bf16_h128`, four warps have `score_row_base = 0, 32, 64, 96` and `head_row_base = 0`. Show why computing `meta_addr` with `head_row_base << 16` maps all 4 warps to TMEM lanes `[0, 31]`, violating `tcgen05.ld`/`tcgen05.st` lane-quarter ownership, whereas `score_row_base << 16` maps warps `0..3` to `[0..31], [32..63], [64..95], [96..127]`.
2. **Falsifiable Hypothesis**: Can `nextafter(__nv_bfloat16 x, __nv_bfloat16 y)` be implemented by promoting to `float`, calling `nextafterf`, and converting back via `__float2bfloat16`? Source and IEEE-754 bit-level verification prove **No**: `nextafterf` advances by 1 FP32 ULP (`2^-23`), which rounds back to the identical BF16 value (`2^-7` mantissa); bit-exact 16-bit integer stepping is required.

---

## 4. Reproducible Offline Experiment

- **Experiment Script**: `experiments/flashinfer_grouped_fp8_silu_quant_lmhead_loss_dsv4_tmem_deepseek_route_cudnn_kv_tilelang_phase98/run.py`
- **Verification Command**: `python experiments\flashinfer_grouped_fp8_silu_quant_lmhead_loss_dsv4_tmem_deepseek_route_cudnn_kv_tilelang_phase98\run.py --check`
- **Unit Test**: `tests/test_phase98_experiments.py`
