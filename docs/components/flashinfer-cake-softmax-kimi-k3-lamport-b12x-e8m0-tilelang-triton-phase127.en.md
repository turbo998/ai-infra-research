# FlashInfer Cake Softmax Balanced Chunks & GB300 Persistent Route, Cake Kimi-K3 Fused Router, NVLink One-Sided All-to-All Lamport Transport, SM12x `b12x` W4A16 E8M0 Re-Bias & `-1` Route Filter, and TileLang/Triton Kernel Guards (Phase 127 · Priority 136)

> **Language / 语言**: [中文 (`flashinfer-cake-softmax-kimi-k3-lamport-b12x-e8m0-tilelang-triton-phase127.md`)](flashinfer-cake-softmax-kimi-k3-lamport-b12x-e8m0-tilelang-triton-phase127.md) | **English (Current)**

## 1. Executive Architectural Findings

In Phase 127 Track A, we audited **11 upstream commits** across `flashinfer-ai/flashinfer`, `tile-ai/tilelang`, and `triton-lang/triton` covering Blackwell (`sm_100a` B200 / `sm_103a` GB300), Hopper (`sm_90a`), and RTX PRO / GeForce Blackwell (`sm_120`):

1. **Cake Blackwell Softmax Balanced Chunks & GB300 Persistent Route (`flashinfer@80a8b2a8`)**:
   - Routes GB300 (`sm_103a`) 6-tile 8-CTA chunks between `20/9` and `2.5` rows per persistent cluster (80–89 rows) to the persistent program (**`1.088x` geomean**, up to **`1.132x`** at `87x98304`: `13.41 -> 11.84 us`) and applies balanced per-CTA chunks `ceil(vocab / CTAs)` rounded up to 32 elements across 16-CTA programs (**`1.066x` geomean** across 95 GB300 shapes; **`1.022x` geomean** and **`1.038x` peak** across 39 B200 shapes).
2. **Cake Kimi-K3 Fused Router (`896` Experts Top-16) (`flashinfer@2c6a3ea6`)**:
   - Dispatches `LC1_NC` at 1 token `block_m=16`, `LCG_LW` at 29–32 tokens on `sm_100a`, `Q4S_E16` (16 tokens/lane emit), and `Q4SPW_LR` (16-byte candidate ranks with ballot-plane scans), cutting 1-token `block_m=16` latency from `3.232 -> 2.656 us` on GB300 (**`1.22x`**) and `3.552 -> 2.943 us` on B200 (**`1.21x`**).
3. **NVLink One-Sided All-to-All Lamport Transport (`flashinfer@1f306300`)**:
   - Pushes 16-byte vectors carrying their own arrival over sentinel-armed landing rows so receivers poll payload data directly without completion flags or system fences (`fence.acq_rel.sys`). Under the default `159`-token floor (`64 MiB` speed bound), DeepSeek-V3 (`7168, top-8, 256E`) achieves **`2.77x` (BF16: `43.6 -> 15.7 us`)** and **`2.69x` (NVFP4: `42.3 -> 15.7 us`)** at `EP=8, T=1`, **`2.54x` (`50.0 -> 19.7 us`)** at `EP=64, T=1`, and saves **`2.0–2.6 ms/step`** on end-to-end DEP16 decode.
4. **SM12x `b12x` W4A16 BF16 E8M0 Scale Exponent Re-Bias (`flashinfer@30cd27ea`) & Unrouted (`-1`) Top-K Sum Guard (`flashinfer@f083cef0`)**:
   - Shifts `packed_dequant_e8m0x4_to_bfloat2x2` scale bias from `7 -> 71` (global compensation `2^55`, clamp `183`), moving the normal BF16 window from `121..247` to **`57..183`** (`2^64` extra headroom above subnormal underflow for checkpoints like MiMo-V2.6-Pro `e in [111, 117]`).
   - Skips `raw_expert < 0` in the map-free branch of `W4A16TopKSumKernel`, eliminating uninitialized `fc2_out` accumulation on expert sidecars (**`cos 0.29 / NaN -> 0.99999`**).
5. **Frost MX FFMA Cross-Target Loader (`64177fe5`)**, **Cake Ulysses Direct Fence (`d304863b`, `1.021x`)**, **Cake MoE AllReduce Union First-Token Reduce-Scatter Overlap (`1e3f47f2`, `1.0332x`)**, **Hopper FMHAv2 In-Kernel Ticket Counter Reset (`32a8e5a7`)**, **TileLang Sparse MMA Tail Guard (`fca340e5`)**, and **Triton Local-Store CTA Replica Broadcast (`51a27535`)**.

## 2. Deterministic Verification Summary (`experiments/flashinfer_cake_softmax_kimi_k3_lamport_b12x_e8m0_tilelang_triton_phase127`)

| Subsystem / Kernel Path | Baseline Behavior | Phase 127 Optimization | Verified Quantitative Gain |
|---|---|---|---|
| Cake Softmax (`GB300`, `87x98304`) | `8-CTA shared-tile` (`13.41 us`) | `8-CTA persistent` (`11.84 us`) | **`1.132x` speedup** (`1.066x` geomean on 95 shapes) |
| Cake Kimi-K3 Router (`1 token, bm=16`) | GB300 `3.232 us` / B200 `3.552 us` | `LC1_NC` GB300 `2.656 us` / B200 `2.943 us` | **`1.22x` (GB300) / `1.21x` (B200)** |
| NVLink One-Sided A2A (`EP=8, T=1`) | Fence path BF16 `43.6 us` / NVFP4 `42.3 us` | Lamport 16B sentinel polling `15.7 us` / `15.7 us` | **`2.77x` (BF16) / `2.69x` (NVFP4)** |
| `b12x` W4A16 BF16 E8M0 Bias (`30cd27ea`) | Bias `7` (normal `121..247`, MiMo `111..117` subnormal) | Bias `71` (normal `57..183`, compensation `2^55`) | **`2^64` subnormal headroom, 0 flush-to-zero** |
| `b12x` W4A16 `-1` Top-K Sum (`f083cef0`) | Sums unwritten `fc2_out` rows (`NaN` or `cos=0.29`) | Skips `raw_expert < 0` with `int32` route IDs | **`cos = 0.99999` (8/8 E2E pass)** |
