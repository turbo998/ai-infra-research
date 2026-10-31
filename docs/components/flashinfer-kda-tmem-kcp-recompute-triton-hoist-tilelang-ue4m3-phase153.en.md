# FlashInfer KDA TMEM Epilogue Release & Comm-Free KCP Recompute, Nonblocking Upload & Bisect Halo Lookup, Triton Conversion Hoisting & TileLang NVFP4 UE4M3 NaN Decode (Phase 153)

- **Evidence Class**: `implementation` (upstream commits) + `own-offline` (`experiments/flashinfer_kda_tmem_kcp_recompute_triton_tilelang_phase153/run.py`)
- **Covered Commits (10 zero-overlap primary commits)**:
  1. `flashinfer-ai/flashinfer@8d7d03c` (`#7828`, 2026-10-29): `fix(cake_kda): release the compute warps' TMEM map writes to the prefix epilogue`
  2. `flashinfer-ai/flashinfer@8d52829` (`#7846`, 2026-10-29): `perf(cake_causal_conv1d): with grad, communication-free KCP checkpoint recompute with a recompute handle`
  3. `flashinfer-ai/flashinfer@27932c3` (`#7872`, 2026-10-29): `perf(cake_kda_train): nonblocking KCP table uploads`
  4. `flashinfer-ai/flashinfer@fd2b1f1` (`#7855`, 2026-10-29): `perf(cake_kda_train): bisect the KCP halo-plan owner lookup`
  5. `flashinfer-ai/flashinfer@78b4734` (`#7841`, 2026-10-29): `perf(cake_kda_kcp): with grad, lower the forward-only workspace of the KCP route`
  6. `triton-lang/triton@d364bf1` (`#12521`, 2026-10-27): `Remove can_reorder`
  7. `triton-lang/triton@a367c52` (`#12298`, 2026-10-21): `Hoist conversions over multiple expansions`
  8. `tile-ai/tilelang@8d231ca` (`#3439`, 2026-10-07): `[BugFix][CUDA] Gate fp8 T.gemm on SM89 instead of trapping at launch`
  9. `tile-ai/tilelang@0e687c1` (`#3429`, 2026-10-05): `[BugFix][Transform] Preserve explicit casts during index promotion`
  10. `tile-ai/tilelang@ddbdef8` (`#3334`, 2026-10-03): `[BugFix] Decode the reserved NVFP4 UE4M3 NaN scale code`

## 1. Key Implementation Paths & Architectural Mechanisms

1. **Blackwell TMEM Map Write Release & Comm-Free KCP Recompute (`#7828`, `#7846`, `#7841`)**:
   - Releases compute warps' TMEM map writes to the prefix epilogue in `cake_kda` (`4 -> 0` hazards), enables communication-free KCP checkpoint recompute via `recompute_handle` (`94.0 us -> 61.5 us`, **1.5285x**), and lowers forward-only KCP workspace from `384.0 MB` to `224.0 MB` (**-41.67%**).
2. **Nonblocking KCP Table Uploads & Bisect Halo Owner Lookup (`#7872`, `#7855`)**:
   - Converts KCP table uploads to nonblocking H2D transfers (`42.0 us -> 11.5 us`, **3.6522x**) and replaces linear halo owner scan with `bisect` (`128 -> 7` steps, **18.2857x**).
3. **Triton Pre-Expansion Conversion Hoisting & TileLang NVFP4 UE4M3 NaN (`#12298`, `#3439`, `#3334`)**:
   - Hoists type conversions across multiple `expand_dims/broadcast` ops (**8.0x** fewer converted elements), gates FP8 GEMM on `SM89+` at compile time, preserves explicit casts during int64 index promotion, and decodes reserved `UE4M3=0x7F` as `NaN`.

## 2. Offline Verification (`experiments/flashinfer_kda_tmem_kcp_recompute_triton_tilelang_phase153/run.py`)

- Comm-free KCP recompute speedup: **1.5285x**, workspace reduction: **41.67%**, nonblocking KCP upload speedup: **3.6522x**, bisect lookup speedup: **18.2857x**.
