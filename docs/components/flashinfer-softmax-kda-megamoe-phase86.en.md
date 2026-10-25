# FlashInfer Blackwell Per-Tile Softmax Turnover Ring, FlashKDA 10-Slot Single-Owner Ring & MegaMoE CuTe DSL Next (Phase 86)

> **Language / 语言**: [中文](flashinfer-softmax-kda-megamoe-phase86.md) | **English (Current)**

## 1. Background and Architectural Evolution

Across Blackwell (`sm_100a` B200 / `sm_103a` GB300 / `sm_120a` RTX 5090 & RTX PRO 6000 Blackwell) and Hopper (`sm_90a`), `flashinfer-ai/flashinfer` landed 10 primary kernel commits on `2026-10-25` targeting persistent cluster softmax, FlashKDA N16 auxiliary MMA barrier rings, GB300 TMA output-store pacing, the MegaMoE CuTe DSL Next backend, and Frost low-latency FP8/block-FP8 decode:

1. **Blackwell Persistent Cluster Softmax Per-Tile Turnover Ring** ([PR #7359](https://github.com/flashinfer-ai/flashinfer/commit/9e016bdfc39624c1659a10d60ed8c9af4c3cea61)): Eliminates inter-row CTA-wide barriers in persistent 8-CTA and 16-CTA DSMEM softmax by writing row $i$ one tile at a time and immediately reading row $i+1$'s corresponding tile into the same registers.
2. **FlashKDA N16 10-Slot Single-Owner Auxiliary Barrier Ring** ([PR #7358](https://github.com/flashinfer-ai/flashinfer/commit/62c2ada6a5ef277acdd1d744016b335bb9f3f235)): Fixes intermittent CTA deadlocks and corrupted outputs in `kernel_flashkda_bf16_fused_m128` and `prepare_bf16_kda_prefill` when a `(sequence, head)` has $\ge 6$ chunks of 16 tokens caused by two auxiliary MMA warps alternating on an odd 5-stage ring.
3. **GB300 (`sm_103a`) 5120-Column Input-Gradient (`dgrad`) `_sw25` Paced Output-Store Programs** ([PR #7361](https://github.com/flashinfer-ai/flashinfer/commit/7dbb267ae89d9c2f6586ded623ad9d10f5f8a303)): Turns an 8-warp simultaneous TMA bulk-store burst in the `128x256` CTA-pair `pd=2` epilogue drain into a 7-step ramp of $n=25$ dependent `u32` multiply-adds per step.
4. **Blackwell MegaMoE CuTe DSL Next, Frost FP8 Decode, Router GEMM Per-Shape Schedules & MiniMax-H3 SM120 FP8 Varlen Attention** ([PR #7001](https://github.com/flashinfer-ai/flashinfer/commit/a70a503844d94a07fea1f43ab21dab74476d95d4), [PR #7212](https://github.com/flashinfer-ai/flashinfer/commit/d3d7c2488783d85e678bd693318bbee19e2fbb97), [PR #7311](https://github.com/flashinfer-ai/flashinfer/commit/2866b59272d7976d10409e34cf15ce9dd9400f70), [PR #7310](https://github.com/flashinfer-ai/flashinfer/commit/06da6cc640a059b95286d275ee464e0d26c9c16f), [PR #7289](https://github.com/flashinfer-ai/flashinfer/commit/ba12a832238575233f2cee795bd93479494fc69a), [PR #7285](https://github.com/flashinfer-ai/flashinfer/commit/dc39352926e8db106670d138c8cfb53f09e1e388), [PR #7287](https://github.com/flashinfer-ai/flashinfer/commit/7b220a67a48606c82105cb4cafa22189489583ee)).

---

## 2. First-Hand Source Code & Mathematical Analysis

### 2.1 Blackwell Persistent Softmax Per-Tile Turnover Ring (PR #7359)

In `csrc/cake_blackwell_softmax.cu`, `csrc/cake_blackwell_softmax_cached_cluster.cu`, and `csrc/cake_blackwell_softmax_cached_cluster_persistent.cu`:

- **Bottleneck in Prior Kernel**: For `num_tiles` between `kMaxRegTiles = 4` and `kMaxSharedTiles = 16`, persistent 8-CTA and 16-CTA cluster kernels required a CTA-wide barrier between rows while refilling shared memory for row $i+1$.
- **Per-Tile Turnover Ring**:
  - After row 0 is loaded into SMEM, each thread writes normalized tile $t$ of row $i$ to global memory and immediately issues the global load for tile $t$ of row $i+1$ into the same registers before storing it back to SMEM—eliminating inter-row CTA barriers (except for 16-CTA clusters walking $>24$ rows on `sm_100a`, which refill one row at a time).
  - Warp-level max reduction uses hardware `redux.sync.max` via sign-preserving IEEE-754 integer bit reinterpretation.
  - `select_cached_launch` thresholds are updated to `kSharedTilesAnyRowsMaxTiles = 5`, `kPersistentTripFillNum / Den = 4 / 5` ($0.80$), `kSharedTilesRowsPerClusterFullChunkC16 = 8`, and `C8 = 7`.
- **Measured Speedups**: Across 41 shapes, geomean speedup is **1.047x** on B200 (`sm_100a`) and **1.042x** on GB300 (`sm_103a`), peaking at **1.247x** on B200 (`22.54 -> 18.08 us`) and **1.197x** on GB300 (`21.33 -> 17.82 us`) for `[128, 128000]`, and **1.205x** on B200 (`24.90 -> 20.67 us`) for `[64, 256000]`.

### 2.2 FlashKDA N16 10-Slot Single-Owner Auxiliary Barrier Ring (PR #7358)

In `csrc/cake_flashkda.cu` (`kernel_flashkda_bf16_fused_m128` and `prepare_bf16_kda_prefill`):

- **Root Cause (Alternating Warp Ownership on an Odd 5-Stage Ring)**:
  - Two auxiliary MMA warps (`aux_idx` $\in \{0, 1\}$) process chunks `chunk = step * 2 + aux_idx`.
  - When `aux_pairwise_inputs_ready`, `aux_pairwise_consumed`, and `aux_inverse_ready` were indexed on a 5-stage ring ($\text{stage} = \text{chunk} \bmod 5$, $\text{parity} = \lfloor \text{chunk} / 5 \rfloor \bmod 2$), consecutive laps of the same stage belonged to different warps (`chunk = 0` on Warp 0 vs `chunk = 5` on Warp 1). When Warp 0 ran ahead of Warp 1, its completions flipped the stage parity and allowed Warp 1's parity wait to pass one lap early.
- **10-Slot Single-Owner Fix**:
  - Expands the auxiliary hand-off ring to `NUM_AUX_RING_STAGES = 10` ($\text{slot} = \text{chunk} \bmod 10$, $\text{parity} = \lfloor \text{chunk} / 10 \rfloor \bmod 2$). Even slots $\{0, 2, 4, 6, 8\}$ are exclusively owned by `aux_idx = 0`, and odd slots $\{1, 3, 5, 7, 9\}$ are exclusively owned by `aux_idx = 1` (`0` cross-warp conflicts vs `15` across 20 chunks).
  - Total mbarriers increase by 15 (`97 -> 112` for N16, `107 -> 122` for prepared path), fitting inside the existing `1024 B` SMEM header with `64` registers unchanged. Verified with **0 hangs** across **200,000** N16 launches and **1,000,000** prepared-path launches on B200 and GB300.

### 2.3 SM103a `_sw25` Paced Stores, MegaMoE CuTe DSL Next & Frost/Router/MiniMax Optimizations

| Commit / PR | Core Mechanism | Quantitative Result |
|---|---|---|
| [PR #7361](https://github.com/flashinfer-ai/flashinfer/commit/7dbb267ae89d9c2f6586ded623ad9d10f5f8a303) (`7dbb267`) | `_sw25` output-store pacing on `sm_103a` (GB300) for K-major `dgrad` `5120->512` and `5120->1024`: staggers 8 epilogue warps via a 7-step ramp of $n=25$ dependent `u32` MADs | GB300 $T=16384$ `5120->512` `54.3 -> 52.7 us` (**1.030x**); `5120->1024` `92.1 -> 91.1 us` (**1.011x**) |
| [PR #7001](https://github.com/flashinfer-ai/flashinfer/commit/a70a503844d94a07fea1f43ab21dab74476d95d4) (`a70a503`) | Adds `sm100_nvfp4_nvfp4_bf16_cutedsl_next` and `sm100_mxfp8_mxfp8_bf16_cutedsl_next` (`46` exported Python modules) supporting `swiglu`, `situ`, pre-FC2 routing weights, and `FinalizeMode::CAKE` | Unified Blackwell MegaMoE NVFP4/MXFP8 AOT kernel package |
| [PR #7212](https://github.com/flashinfer-ai/flashinfer/commit/d3d7c2488783d85e678bd693318bbee19e2fbb97) (`d3d7c24`) | Extends `FrostLowLatencyRunner` (`H=2048, I=512, top_k=10`) to grouped FP8 and $128\times 128$ block-FP8 on SM90 (`wgmma`) and SM120 (`warp_mma`), bounding native FP8 accumulation to $K \le 64$ | Eliminates FP8 mantissa drift and Triton dequantization fallback |
| [PR #7311](https://github.com/flashinfer-ai/flashinfer/commit/2866b59272d7976d10409e34cf15ce9dd9400f70) (`2866b59`) | Optimizes `cake_blackwell_bf16_fp4` `tcgen05` `cta_group::2` M128 pair programs (Warp 7 `B_descale` prefetch, doubled mbarriers, `512 -> 160` thread barrier) | Lower 2-CTA M128 synchronization latency on SM100/SM103 |
| [PR #7310](https://github.com/flashinfer-ai/flashinfer/commit/06da6cc640a059b95286d275ee464e0d26c9c16f) (`06da6cc`) | Routes $M \in [256, 960]$ token tails in `kimi_k3_tp12_tail` through `("k1_twoshot_ess:grouped", "k3_ess:grouped")` (`GROUPED_MAX_TOKENS = 960`) | GB300 $M=512$ `60.59 -> 56.10 us` (**1.080x**), $M=256$ **1.060x** |
| [PR #7289](https://github.com/flashinfer-ai/flashinfer/commit/ba12a832238575233f2cee795bd93479494fc69a) (`ba12a83`) | 3-stage `minimax_h3_sm120_varlen_attention_fp8` (`KV_STATS`, `QUANTIZE`, unit-scale `mma.sync kind::f8f6f4` with split K/V refill) | RTX 5090 `center_4s` `73,983 -> 52,679 us` (**1.404x**) |
| [PR #7285](https://github.com/flashinfer-ai/flashinfer/commit/dc39352926e8db106670d138c8cfb53f09e1e388) (`dc39352`) | `cake_fused_norm_combine` `OWNER_PIPELINED` ($\text{tokens} \ge 480$) at lag `3 * num_bids` with final store 4 steps behind | 8x B200 `30.656 -> 30.112 us` (**1.018x**), 8x B300 **1.016x** |
| [PR #7287](https://github.com/flashinfer-ai/flashinfer/commit/7b220a67a48606c82105cb4cafa22189489583ee) (`7b220a6`) | `cake_router_gemm` per-shape schedules (`32` programs for $M=1..16 \times K \in \{6144, 7168\}$ + `narrow` 1-CTA/SM variants) | SM100/SM103 $K=6144, E=256, M=16$ `6.400 -> 4.320 us` (**1.481x**) |

---

## 3. Reproducible Verification (`experiments/flashinfer_softmax_kda_megamoe_phase86`)

Running `python experiments/flashinfer_softmax_kda_megamoe_phase86/run.py --check` validates all metrics deterministically:
- **Softmax Turnover Ring**: `1.247x` speedup on B200 `[128, 128000]` and `1.197x` on GB300;
- **FlashKDA 10-Slot Ring**: `0` cross-warp parity conflicts across 20 chunks vs `15` on the 5-slot ring;
- **SM103a Paced Store & Router GEMM**: 8-store burst converted to a 7-step $\times 25$ `u32` MAD ramp; Router GEMM $M=16$ speedup `1.481x`.

## 4. Architect Exercises & Falsifiable Questions

1. **Multi-Warp Ring Ownership Invariant**: Prove that when $W$ warps round-robin over an $S$-stage mbarrier ring, $S \equiv 0 \pmod W$ is necessary and sufficient for every slot to have a single static warp owner across all laps.
2. **Falsifiable Hypothesis**: On `sm_100a` 16-CTA cluster softmax walking $>24$ rows per cluster, does enabling per-tile turnover instead of row-at-a-time refill saturate the MIO load queue and regress latency?
