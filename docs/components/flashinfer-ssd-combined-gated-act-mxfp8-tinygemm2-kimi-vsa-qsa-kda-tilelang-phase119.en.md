# FlashInfer SM107a SSD Exact-Scan, MXFP8 8-Range Dispatch, TinyGEMM2 L2 256B Promotion, Kimi-K3 Router, Hopper VSA/QSA/KDA, and TileLang Ascend Pipelines (Phase 119 / Priority 128)

> **Evidence Classification**: Upstream `implementation` + `author-benchmark` (10 public upstream commits dated 2026-10-28) + repository `own-offline` deterministic arithmetic/topology verification (`experiments/flashinfer_ssd_combined_gated_act_mxfp8_tinygemm2_kimi_vsa_qsa_kda_tilelang_phase119/run.py`). This note makes no unverified live GPU hardware re-benchmark claims.

## 1. Core Architectural Problems & Context

Recent kernel engineering across Blackwell (`SM100a`), Blackwell Ultra (`SM103a`), transitional Rubin (`SM107a` / `R200`), Hopper (`SM90a`), and Ascend NPUs exposed five critical microarchitectural and compiler bottlenecks:

1. **Dual-Launch Overhead in SM107a (`R200`) SSD Exact-Scan and KDA Backward Tail**: `cake_ssd_combined` incurred redundant split launches on FP32-state single-chunk calls above the item bound and retained no-ring calls, while `cake_kda` on `SM107a` still executed `glm_intra` and `gate_epilogue` as two separate launches with intermediate `dg_int` / `db_int` buffers.
2. **Register Spill and Occupancy Degradation Across $M \times K$ Regimes in Gated-Act MXFP8**: A coarse dispatch over `SM100a`/`SM103a` could not simultaneously optimize $< 2^{20}$ latency and $\ge 2^{26}$ bandwidth, and the `SM103a` backward rowwise program suffered an 8-byte local stack spill at 12 CTAs/SM.
3. **Suboptimal L2 Promotion and 17–23 Row Tile Utilization in TinyGEMM2**: Narrow-tile TMA descriptors on `SM107a` lacked explicit 256-byte L2 promotion, and unbiased 17–23 row GEMMs (exceeding a single 16-row tile but underfilling two tiles) were not routed to the packed-tile kernel.
4. **Cluster Barrier Stalls and Shared-Memory Contention in Kimi-K3 Fused Router (896 Experts, Top-16)**: The 16-CTA cluster kernel for 3–32 tokens waited for its cluster at entry before row selection, and batches above device capacity $C$ ($592$ on GB300, $568$ on B200) incurred cross-CTA ranking overhead.
5. **Ascend Manual Pipeline Configuration Register Overwrites and Bisheng Cache Collisions**: Hardware configuration registers (`HF32`, `MMAD` direction, `PadValue`, `AtomicMode`) share a single physical register across pipeline stages in Ascend NPUs, and `.aibin` cache keys collided across Bisheng compiler builds sharing identical Clang version strings.

---

## 2. Primary Upstream Implementation Analysis (10 Zero-Overlap Commits)

| # | Repository & Commit | Date | Core Implementation Path & Microarchitectural Mechanism |
|---|---|---|---|
| 1 | `flashinfer-ai/flashinfer@372e9435f4` (`#7558`) | 2026-10-28 | `flashinfer/experimental/cake_ssd_combined/`: Regenerates 17 `sm_107a` programs (`exactpre` + `exactfused`), rerouting FP32-state 1-chunk calls above the item bound to `exactpre` and retained no-ring calls to `exactfused`. All 44 changed/rerouted shapes accelerate on R200 (`1.012x–1.562x` on `nem_v8x64_float32`; `perf_varlen_128x32` drops `449.52 -> 348.32 us`, **1.291x**), with a **1.0487x** geomean across all 93 shapes. |
| 2 | `flashinfer-ai/flashinfer@1d74db1e9d` (`#7568`) | 2026-10-28 | `flashinfer/cake_gated_act_mxfp8/`: Expands dispatch to 8 per-arch $M \times K$ ranges (`5` on `sm_100a`, `3` on `sm_103a`; `45 -> 50` programs): `< 2^20` uses `32x32` tiles at 12 CTAs/SM; `sm_100a` uses band-path at 5 CTAs/SM from $2^{26}$; `sm_103a` backward rowwise $\ge 2^{26}$ replaces the 12-CTA/SM build (`8 B` stack spill) with packed `f32x2` sigmoid at 10 CTAs/SM (`0 B` spill), yielding **1.07x–1.74x** speedups on changed shapes. |
| 3 | `flashinfer-ai/flashinfer@efa2600909` (`#7588`) | 2026-10-28 | `csrc/cake_tinygemm2_tcgen05/`: Enables `CU_TENSOR_MAP_L2_PROMOTION_L2_256B` on `sm_107a` narrow-tile TMA tensor maps (**1.02x** on R200, e.g., `1x128x14336` `5.31 -> 5.22 us`) and routes unbiased `17..23`-row GEMMs (`1536..4096` outputs, $\text{out}\times K \ge 6\times 10^6$, activation $< 128\text{ KiB}$) to packed-tile (**1.10x–1.46x** across 78 shapes). |
| 4 | `flashinfer-ai/flashinfer@95ab86637c` (`#7567`) | 2026-10-28 | `flashinfer/experimental/kimi_k3_fused_router/`: Adds `LCG_LW` (16-CTA cluster waiting after row selection), `Q4SB` (single-row next-row prefetch from $C=592$ to 1024 tokens on `sm_103a`), and `Q4SPW_LR` (owner-CTA shared-memory pair ranking), achieving up to **1.05x** speedup on GB300 (`6.02 -> 5.73 us` at 256 tokens) and B200 (`6.24 -> 5.95 us`). |
| 5 | `flashinfer-ai/flashinfer@acce856763` (`#7580`) | 2026-10-28 | `flashinfer/cake_vsa_sm90_backend/`: Pipelines Hopper (`sm_90a`) grouped-query FP8 queue (splitting K/V wait barriers and prefetching next queue word 1 step ahead) and overlaps BF16 compact pair-mode tile output stores with the next tile start. Achieves **1.0236x** geomean across 37 changed rows (**1.0274x** across 29 FP8 queue rows; `3869.0 -> 3767.2 us` at `109632` tokens). |
| 6 | `flashinfer-ai/flashinfer@3055feed86` (`#7578`) | 2026-10-28 | `flashinfer/experimental/cake_qsa_nvfp4/`: Enables direct partial epilogue writes in the persistent walk on `SM103a` (**1.03x** at 128 rows on GB300, `64.68 -> 62.94 us`) and exact warp-redux softmax max, early block-table loads, and direct partial epilogue on `SM107a` (**1.06x** at 64 rows NHD on R200). |
| 7 | `flashinfer-ai/flashinfer@b17b12b2e5` (`#7576`) | 2026-10-28 | `flashinfer/experimental/cake_kda_train/`: Dispatches gate-fused intra-chunk backward kernel `glm_intra_v18` on `SM107a` (`R200`) with immediate named-barrier slots so 2 CTAs co-reside per SM, eliminating the separate `gate_epilogue` launch (**1.0306x–1.0481x** backward speedup; `33.5401 -> 32.0011 ms` on `P14 H=64, S=261393`). |
| 8 | `flashinfer-ai/flashinfer@8cf7089cf6` (`#7583`) | 2026-10-28 | `csrc/cake_trtllm_moe_allreduce_union/`: Clears 2 previous-epoch workspace packets per consume-loop step with sentinel `0x80008000` while waiting on RMS exchange and prefetches reduce-scatter 2 tokens ahead at `1536..2048` tokens (**1.017x–1.062x** on B200, **1.026x–1.060x** on B300). |
| 9 | `tile-ai/tilelang@dfe83e84ec` (`#3624`) | 2026-10-28 | `tilelang/ascend/transform/z3_scheduler.py`: Tracks `kConfigurationRegisters` (`kHf32Mode`, `kMmadDirection`, `kPadValue`, `kAtomicMode`) and adds Z3 modulo phase constraints $r[v] + \text{wrap} \cdot II \ge r[u] + \text{gap}$ ($\text{wrap} \in \{0, 1\}$) in manual offset schedules. |
| 10 | `tile-ai/tilelang@1c0f402e16` (`#3691`) | 2026-10-28 | `tilelang/contrib/bisheng.py` & `tilelang/cache/ascend_binary_cache.py`: Queries and caches `bisheng --version` first line (preserving build timestamp and revisions while ignoring `InstalledDir`) in the `.aibin` device binary cache key. |

---

## 3. Key Formulas, Data Movement & Offline Verification

### 3.1 Ascend Configuration Register Modulo Phase Constraint & MXFP8 Zero-Spill Fit
1. **Physical Configuration Register Wrap Constraint**: With initiation interval $II$, a single physical configuration register shared across logical stages requires every reader $u$ to consume its reaching setter's value before the next physical setter $v$ overwrites it:
   $$r[v] + \text{physical\_wrap} \cdot II \ge r[u] + \text{gap}, \quad \text{physical\_wrap} \in \{0, 1\}$$
2. **SM103a MXFP8 Backward Rowwise Register Fit**: At $M \times K \ge 2^{26}$ ($67,108,864$), dropping occupancy from 12 to 10 CTAs/SM and using packed `f32x2` sigmoid eliminates the `8 B -> 0 B` stack spill.

### 3.2 Offline Verification Summary (`run.py --check`)

- `ssd_sm107a_regenerated_programs`: `17` (`1.562x` max speedup across 44 changed shapes, `1.0487x` geomean across 93 shapes)
- `mxfp8_total_mk_ranges`: `8` (`5` ranges on `sm_100a` + `3` on `sm_103a`, `0 B` spill, up to `1.74x` speedup)
- `tinygemm2_l2_promotion_bytes`: `256` (`256 B` L2 promotion on `sm_107a` + `17..23` row packed-tile up to `1.46x`)
- `kimi_gb300_capacity_c`: `592` (152-SM GB300 4-CTA cluster capacity bound, `1.0506x` at 256 tokens)
- `vsa_changed_geomean_speedup`: `1.0236` (`1.0236x` geomean across 37 changed rows, `1.0274x` across 29 FP8 queue rows)
- `kda_sm107a_ctas_per_sm`: `2` (`glm_intra_v18` immediate named-barrier slots enabling 2 CTAs/SM, `1.0481x` bwd speedup on `P14`)
- `ascend_reg_phase_constraint_satisfied`: `True`
- `bisheng_builds_separated` / `bisheng_relocation_invariant`: `True` / `True`

---

## 4. Architect Exercises & Falsifiable Questions

1. **Calculation Exercise**: For `cake_tinygemm2_tcgen05` unbiased packed-tile routing at $\text{batch} = 22$, verify why $K = 2816$ is the minimum $K$ at $\text{out} = 4096$ and compute the activation working set size in KiB. (Answer: $22 \times 2816 \times 2\text{ B} = 123,904\text{ B} = 121.0\text{ KiB} < 128\text{ KiB}$, and $4096 \times 2816 = 11,534,336 \ge 6,000,000$).
2. **Falsifiable Hypothesis**: Why must `get_bisheng_version()` retain the build timestamp from line 1 of `bisheng --version` while stripping `InstalledDir`? Because separate CANN releases can ship Bisheng compilers with identical Clang version numbers and Git hashes that differ only by build timestamp, while stripping `InstalledDir` keeps the cache invariant under toolchain directory relocation.
