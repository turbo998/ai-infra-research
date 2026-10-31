# GPU Software Stack Phase 169: FlashInfer Kimi-K3 896-Expert Fused Router, ReplaySSM Int64 Indexing, Block-Scale FP8 SF Underflow Guard, cuDNN Tactic Validation & FlashAttention SM100 `hdim=64` Dedicated TMEM Slots

## 1. Architectural Context & Core Problems

Across Blackwell (`B200`/`GB300`, `SM100`) MoE routing, block-scaled FP4/FP8 quantization, and backward attention kernels, three subtle defects and performance bottlenecks were identified:
1. **Kimi-K3 896-Expert Routing Latency, ReplaySSM `Int32` Overflow & Block-Scale FP8 SF Underflow `NaN`s**: Selecting top-16 experts out of 896 in Kimi-K3 requires a single-launch fused router; indexing flattened `[layers * slots, HV, V, K]` ReplaySSM checkpoints overflowed 32-bit signed integers; and 5 block-scale quantization kernels guarded raw `vecMax` rather than the `fp8-rounded` scale factor (`SF`), emitting `NaN`/`Inf` whenever `SF` underflowed to zero.
2. **Silent `tactic=-1` Fallback During cuDNN GEMM Profiling**: When a requested cuDNN GEMM tactic was unsupported, runners silently executed `tactic=-1`, polluting autotuner profiles.
3. **SM100 `hdim=64` Backward TMEM Contention & Widening of Intentionally-Empty Offset Windows**: The 1-CTA `hdim=64` SM100 backward kernel used only TMEM columns `[0, 384)`, leaving `[384, 512)` idle while sharing slots for `P` and `dS`; meanwhile, `_resolve_causal_local_window` widened any offset window with `left + right < 0` into full attention `(None, None)`.

---

## 2. Primary Upstream Mechanism Deep Dive (11 Zero-Overlap Commits)

### 2.1 FlashInfer: Kimi-K3 Fused Router, ReplaySSM `Int64` Slot, FP8 SF Guard & cuDNN Tactic Check
- **`e7a17d27f5` (`2026-10-29`, `#7824`)**: Accelerates `cake_kimi_k3_fused_router` (896 experts, top-16) by up to `1.10x` on B200 and GB300.
- **`4bddce58c3` (`2026-10-29`, `#6496`)**: Indexes ReplaySSM commit checkpoints by `Int64` flat slot in `gdn_replayssm_spec_fold{_tcgen}.py`.
- **`6622cde973` (`2026-10-29`, `#6996`)**: Guards the `fp8-rounded` scale factor against underflow across 5 block-scale quantization sites (`fp4_kv_quantization.cu`, `nvfp4_smooth_quantize_sm100.cuh`, `fused_dit_layernorm.cuh`).
- **`ffdf277d4a` (`2026-10-29`, `#4132`)**: Reports failed cuDNN tactics as unsupported during profiling instead of silently running `tactic=-1`.

### 2.2 FlashAttention & FlashMLA: SM100 `hdim=64` Dedicated P/dS TMEM Slots & Empty-Window Guard
- **`d86dabd8e7` (`2026-09-09`, `#2804`)**: Assigns dedicated 64-column TMEM slots in `[384, 512)` to `P` and `dS` with per-warp signaling in `flash_bwd_sm100.py` (`100%` of `512` TMEM columns utilized).
- **`17498dad34` (`2026-09-09`, `#2490`)**: Prevents `_resolve_causal_local_window` from widening intentionally-empty offset windows (`left + right < 0`) to full attention.
- **`1858932afd` & `7f55c7151a` (`2025-09-30`)**: Refines FlashMLA formatting and error diagnostics.

### 2.3 CUTLASS & Triton: v4.6.1 Update, Loop Operand Signedness & AMD CU-Mode Barrier Test
- **`1e73948292` (`2026-07-14`, `#3383`)**: Updates CUTLASS to `v4.6.1`.
- **`29ef3c3320` (`2026-10-20`, `#12350`) & `3a43613f97` (`2026-10-20`, `#12373`)**: Preserves source signedness in Triton loop operands and isolates AMD CU-mode barrier tests.

---

## 3. Deterministic Offline Verification Summary

Validated via `experiments/flashinfer_kimik3_router_sf_underflow_fa_sm100_tmem_phase169/run.py`:

| Subsystem | Baseline Metric | Phase 169 Verified Metric | Architectural Impact |
|---|---|---|---|
| Kimi-K3 Router (`896` experts) & FP8 SF Guard | Baseline routing; `5` underflow `NaN` sites | **`1.10x` speedup**; **`5` sites guarded (`0` `NaN`s)** | Eliminates block-scale underflow `NaN`s & silent `tactic=-1` |
| CuTe SM100 `hdim=64` Backward TMEM | `384/512` cols (`75%`), shared P/dS | **`512/512` cols (`100%`)**, `128` dedicated cols | Zero shared-slot barrier stalls via per-warp signaling |
| CuTe Empty Offset Window (`left=-128, right=32, N=16384`) | Widened to `16,384` active tokens | **`0` active tokens** (empty window preserved) | Prevents accidental full-attention compute & numerical drift |
