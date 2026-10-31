# GPU Software Stack Phase 168: FlashInfer Frost FP4 Wide-Load & Invalid-Route Masking, Rubin SM107 MegaMoE Fused Quant, Cake TinyGEMM2 `PACKED_M64_N8`, and FlashAttention SM90 Backward & Causal Sliding Window Fixes

## 1. Architectural Context & Core Problems

Across Blackwell (`SM100a`/`SM103a`), Rubin (`SM107`), and Hopper (`SM90`) kernel pipelines, three correctness and performance issues emerged:
1. **Frost Low-Latency FP4 Wide-Load Corruption & Invalid-Route `NaN` Propagation**: Eight-value NVFP4/MXFP4 wide loads on SM100 produced corrupted weights without explicit packed-halfword byte extraction, while multiplying zero partials by `NaN` scales on invalid routes propagated `NaN` into final MoE outputs.
2. **Rubin `SM107` MegaMoE Multi-Launch Overhead & Small-Batch `tcgen05` Tile Gaps**: Separate kernel launches for routing validation, MXFP8 quantization, and `TopkReduce` staging inflated decode latency on `SM107`, while `cake_tinygemm2_tcgen05` required `PACKED_M64_N8` kernels and unbiased routing windows on `sm_100a`/`sm_103a`.
3. **CuTe SM90 Backward Crash for `head_dim ∈ (128, 192]` & Causal Sliding-Window Overplanning**: With `dKV_swapAB=True`, placing `head_dim` on the SM90 BF16 WGMMA `M=64` axis crashed when `tile_hdim` was rounded to 16 (`160 % 64 != 0`). Meanwhile, `(window_size_right or max_seqlen_k)` treated `window_size_right=0` as falsy, causing causal sliding-window attention to schedule the entire `max_seqlen_k` sequence.

---

## 2. Primary Upstream Mechanism Deep Dive (11 Zero-Overlap Commits)

### 2.1 FlashInfer: Frost FP4 Wide Loads, Rubin SM107 MegaMoE Fusion & Cake TinyGEMM2 `tcgen05`
- **`bd460fcb23` (`2026-10-29`, `#7845`)**: Decodes packed halfwords via explicit byte extraction in `fp4_weight.py` for 8-value wide loads.
- **`f416759b9c` (`2026-10-29`, `#7838`)**: Masks invalid Frost routes before finalization and repairs MXFP4 FFMA decoding.
- **`8cc2453f02` (`2026-10-29`, `#7806`)**: Fuses routing validation into MXFP8 input quantization and writes `TopkReduce` directly to caller output in Rubin `SM107` MegaMoE (`3` launches → `1`).
- **`9318fc8621` (`2026-10-29`, `#7826`)**: Builds `36` `PACKED_M64_N8` kernels in `cake_tinygemm2_tcgen05` for `sm_100a`/`sm_103a` with unbiased routing windows.

### 2.2 FlashAttention & FlashMLA: SM90 `M=64` Backward Alignment & Sliding-Window Zero-Bound Fix
- **`9d61d35ba8` (`2026-09-09`, `#2482`)**: Aligns `tile_hdim` to a multiple of `64` (`192`) for SM90 backward when `dKV_swapAB=True` and `128 < head_dim <= 192`.
- **`d8dbfad57d` (`2026-09-09`, `#2624`)**: Preserves `window_size_right = 0` as a valid bound in `seqlen_k_loaded` rather than substituting `max_seqlen_k`.
- **`082094b793` (`2026-01-16`, `#150`) & `1408756a88` (`2025-09-30`)**: Refactors FlashMLA kernels and documentation.

### 2.3 CUTLASS & Triton: AR RS Update, Return-Tuple Validation & AMD Per-Kernel `clopt`
- **`a3a1b5cfe6` (`2026-07-16`, `#3381`)**: Updates CUTLASS AR RS kernels.
- **`7302258a3e` (`2026-10-20`, `#12381`) & `9b12eff0a6` (`2026-10-20`, `#12382`)**: Rejects mismatched return-tuple lengths in Triton's frontend and adds per-kernel `clopt` control for AMD backends.

---

## 3. Deterministic Offline Verification Summary

Validated via `experiments/flashinfer_frost_fp4_rubin_megamoe_fa_sm90_bwd_phase168/run.py`:

| Subsystem | Baseline Behavior | Phase 168 Verified Behavior | Architectural Impact |
|---|---|---|---|
| Frost FP4 & Rubin SM107 MegaMoE | Halfword corruption + `NaN`; `3` launches | **`0` corruptions / `0` `NaN`s**; **`1` launch** | **`3.0x` launch reduction**; `36` `PACKED_M64_N8` kernels |
| CuTe SM90 Backward (`head_dim=160, dKV_swapAB=True`) | `tile_hdim=160` (`160 % 64 = 32` crash) | `tile_hdim=192` (**`192 % 64 = 0`**) | Eliminates WGMMA partition crash on `(128, 192]` |
| CuTe Causal Sliding Window (`max_k=128K, left=4096, right=0`) | `135,168` planned KV tokens | **`4,096` planned KV tokens** | **`33.0x` reduction in KV scheduler overplanning** |
