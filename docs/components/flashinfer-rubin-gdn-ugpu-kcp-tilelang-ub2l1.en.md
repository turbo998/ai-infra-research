# Deep Dive into FlashInfer Rubin (`sm_107a`) BF16-State GDN Decode uGPU Localization, Context-Parallel KCP Chunked Training Plan, Rubin MLA/Kimi-K3 PDL Gating, and TileLang Ascend `Nd2NzGeometry`/PTO Vector-Local/CUDA Cast-Rounding Guards

> **Language / 语言**: [English](flashinfer-rubin-gdn-ugpu-kcp-tilelang-ub2l1.en.md) | [中文](flashinfer-rubin-gdn-ugpu-kcp-tilelang-ub2l1.md)  
> **Phase**: 69 (`priority: 78`, Track A — GPU Kernel & Quantization)  
> **Evidence Tiers**: `implementation` (fixed-commit source audit) + `author-benchmark` (author-measured VR200 `sm_107a` / GB300 / B300 / B200 reports) + `own-offline` (deterministic standard-library simulation `experiments/flashinfer-rubin-gdn-ugpu-kcp-tilelang-ub2l1/run.py`)

---

## 1. Architectural Context & Core Engineering Questions

Across NVIDIA Rubin (`sm_107a` VR200), Blackwell Ultra (`sm_103a` GB300/B300), and Huawei Ascend NPUs, recent kernel-stack advances address four critical hardware-software co-design challenges:

1. **Rubin (`sm_107a` VR200) Dual-Die (`uGPU`) Memory Locality & Instruction-Bandwidth Imbalance (`flashinfer-ai/flashinfer@6fc0f52beae0`, `#6604`)**: VR200 delivers `19.15 TB/s` of HBM peak bandwidth (`~2.5x` B200), whereas its `212` SMs provide `~1.8x` B200's instruction issue rate and are physically split across two `106`-SM locality domains (`uGPUs`). Why does the generic B200 Gated Delta Net (GDN) BF16-state decode kernel plateau at `39%–63%` bandwidth utilization on VR200, and how do instruction trimming plus `CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN` binding reach **`88%`** of `19.15 TB/s`?
2. **Rubin MLA NVFP4 Decode Runtime PDL Switch (`flashinfer-ai/flashinfer@513c4ae68a5a`, `#7004`) and Kimi-K3 `attn_res` Snapshot-Write (`flashinfer-ai/flashinfer@4bfe0963180d`, `#7014`)**: Why does statically enabling Programmatic Dependent Launch (PDL) in `cake_rubin_mla_nvfp4_decode` regress latency whenever the batch spills into a second wave of clusters?
3. **Context-Parallel (CP) Packed KDA Chunked Training Planner (`flashinfer-ai/flashinfer@4c7318173bac`, `#7002`)**: When variable-length packed sequences are partitioned across CP ranks, how do FP32 affine state summaries and the `"wave_fill1"` segment planner (`kcp_plan`) remove serial prefix stalls?
4. **TileLang Ascend Partial UB-to-L1 NZ Geometry, PTO `SimtVF` Vector-Local Scalarization, and CUDA Stochastic-Rounding Guards (`tile-ai/tilelang@59dc2b6763b8` `#3530`, `@a2679e5c6d75` `#3523`, `@e84ce3de8ab7` `#3350`)**: How are exact source/destination pitches and burst gaps derived for partial `copy_ub_to_l1` tiles, and why must `float32 -> float8_e8m0fnu` be rejected before vectorized CUDA `rs` lowering?

---

## 2. Core Implementation Paths & Source-Level Mechanics

### 2.1 FlashInfer Rubin (`sm_107a`) BF16-State GDN Decode & `uGPU` Locality-Domain Partitioning (`#6604`)

In `flashinfer/gdn_kernels/gdn_decode_bf16_state_rubin.py` and `flashinfer/gdn_decode.py`, FlashInfer co-designs the instruction pipeline and physical memory placement for VR200 (`sm_107a`, `212` SMs = `2 × 106` SMs/die, `19.15 TB/s`):

- **Instruction Trimming (`~38%` Warp Instruction Reduction)**:
  - **Identical Copy-Compute Partition Eliminates CTA Barriers**: Each $128 \times 128$ BF16 state matrix is split into two $64 \times 128$ halves (`64` rows = `16 KB` SMEM per CTA, 2 CTAs/head). Both `cp.async` bulk loads and compute assign warp group $g \in [0, 7]$ to rows $g, g+8, \dots, g+56$, so every thread reads only the shared-memory rows it wrote via `cp.async.wait_group 0`—requiring zero CTA `bar.sync` barriers.
  - **5-Shuffle Transpose-Reduce & `FFMA2` Packing**: 16-lane Q/K vectors pack into 8 `f32x2` register pairs per thread with a 3-shuffle butterfly L2 norm, followed by a 5-shuffle register pair exchange (replacing the 16-iteration cross-lane broadcast loop) and packed `FFMA2` state updates.
  - **L2 `evict_last` Lookahead Prefetch**: Before storing updated state rows, each CTA issues `prefetch.global.L2::evict_last` for the first cache line of the tile
    $$
    \text{pf\_dist} = \left\lfloor \frac{\text{num\_SMs} \times 25}{3} \right\rfloor = \left\lfloor \frac{212 \times 25}{3} \right\rfloor = 1766 \text{ tiles ahead}
    $$
    (`~1.2` residency generations at 7 CTAs/SM).
- **`uGPU` Dual-Die Physical Allocation (`ugpu_state_pool`)**:
  - Standard `cudaMalloc` interleaves pages across both dies, routing ~50% of traffic across the inter-die fabric (capping pure copy at `68%` vs `90%` die-local).
  - `ugpu_state_pool(pool_size, HV)` uses `cuMemCreate` with `CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN` (`id = 0` and `id = 1`) to allocate `[pool_size, HV, 2, V // 2, K]`, pinning `half = 0` (rows `0..63`) in Die 0 HBM and `half = 1` (rows `64..127`) in Die 1 HBM, and launching each half on its die's `106`-SM green-context stream.
  - **Dispatch Thresholds**: `backend="auto"` selects the `rubin` kernel on SM107 when $B \times H_V \ge 512$ (`_AUTO_MIN_STATES`). For `ugpu` pools, eager calls with $B \times H_V < 8192$ (`_UGPU_EAGER_MIN_STATES`) launch both halves back-to-back on the caller stream to avoid the `~5 us` cross-stream event fork/join overhead; larger batches and CUDA-graph captures fork/join the two die streams concurrently.
- **Author-Measured VR200 Performance (`author-benchmark`, `HV=64, HQ=HK=16, K=V=128`)**:
  - `B=32`: Generic `18.0 us` (`39%`) $\to$ `rubin` `13.2 us` (`53%`, `1.36x`) $\to$ `rubin_ugpu` **`12.3 us` (`57%`, `1.46x`)**
  - `B=512`: Generic `187.0 us` (`61%`) $\to$ `rubin` `163.0 us` (`69%`, `1.15x`) $\to$ `rubin_ugpu` **`132.6 us` (`85%`, `1.41x`)**
  - `B=2048`: Generic `726.7 us` (`62%`) $\to$ `rubin` `640.2 us` (`71%`, `1.14x`) $\to$ `rubin_ugpu` **`516.5 us` (`88%`, `1.41x`)**
  - `B=8192`: Generic `2887.0 us` (`63%`) $\to$ `rubin` `2549.8 us` (`71%`, `1.13x`) $\to$ `rubin_ugpu` **`2059.8 us` (`88%`, `1.40x`)**

### 2.2 Rubin MLA NVFP4 Decode Runtime PDL Switch (`#7004`) & Kimi-K3 Snapshot-Write (`#7014`)

1. **Rubin MLA NVFP4 Decode `pdl_for(plan)` (`#7004`)**:
   - In `flashinfer/experimental/rubin_mla_nvfp4_decode/api.py`, single-wave cluster capacity is defined by `UNITS_PAIRS_BY_SM_COUNT = {208: (50, 4), 212: (46, 14)}` (`54` units on 208 SMs, `60` units on 212 SMs).
   - When `plan.items > plan.units`, a second wave of decode clusters is required; enabling PDL lets merge CTAs steal SMs as soon as wave-1 clusters signal `launch_dependents`. Thus PDL is gated by:
     $$
     \operatorname{pdl\_for}(\text{plan}) = (\text{plan.max\_splits} > 1) \land (\text{plan.items} \le \text{plan.units})
     $$
2. **Kimi-K3 `attn_res` Snapshot-Write & SM103 Held-Release (`#7014`)**:
   - In `flashinfer/experimental/kimi_k3_attn_res/api.py`, depth-2 `K=4` snapshot-write (`persistent:k4_nc3_d2_f111000000000000_write`, pinned in `WRITE_PERSISTENT_KEY_CELLS`) and SM103 depth-3 `K=5` `nc3` held-release programs reduce latency on GB300 (`M=4096, K=5`: `85.248 -> 83.920 us`, `1.0160x`) and B300 (`M=4096, K=4 write`: `87.649 -> 85.473 us`, `1.0255x`).

### 2.3 Context-Parallel (CP) Packed KDA Chunked Training Planner (`#7002`)

In `flashinfer/experimental/kcp/cake_kcp.py` and `cake_kcp_plan.py`, FlashInfer implements Context-Parallel chunked KDA recurrence for variable-length packed sequences:
- **3-Stage Affine State Reduction**: Each local segment summarizes its chunks into an FP32 affine transition $(A_{\text{seg}}, B_{\text{seg}})$, exchanges summaries across the CP group via `all_gather_into_tensor`, composes the exact initial state per segment, and writes full-precision outputs `O` and final states `ht`, supporting `contiguous`, `zigzag`, and `zigzag_per_document` layouts plus width-4 causal `conv1d + SiLU` halo exchange.
- **`"wave_fill1"` Segment Planner (`kcp_plan`)**:
  - When a rank holds $n_{\text{frag}} = 1$ split fragment with $0 < NC_{\max} < 257$, segment count $P$ expands to fill one SM wave:
    $$
    P = \max\!\left(P_{\text{chain}},\; \min\!\left(\left\lfloor \frac{\text{SM}}{H} \right\rfloor,\; \min(64, NC_{\max})\right)\right), \quad P_{\text{chain}} = \left\lceil \frac{NC_{\max}}{16} \right\rceil
    $$
  - When $n_{\text{frag}} \le 3$ and $NC_{\max} \ge 257$, it minimizes $\lceil n_{\text{frag}} P H / \text{SM} \rceil \cdot \lceil NC_{\max} / P \rceil \cdot t_{\text{walk}} + (P - 1) t_{\text{compose}}$, achieving **`1.3058x–2.4679x` over FLA KCP** and **`1.6184x–2.3319x` over Cake head-CP** across 144 GB300 configs.

### 2.4 TileLang Ascend `Nd2NzGeometry`, PTO Vector-Local Scalarization & CUDA Cast-Rounding Guards (`#3530`, `#3523`, `#3350`)

1. **Ascend Partial UB-to-L1 NZ Copies (`Nd2NzGeometry`, `#3530`)**:
   - `src/tl_templates/ascend/common.h` and `src/ascend/transform/lower_Tile_op.cc` add a pitched `copy_ub_to_l1` overload. With $C_0 = 32 / \text{dst\_bytes}$ elements per 32-byte fractal block, $\text{Bursts} = \text{cols} / C_0$, $\text{src\_gap} = \text{src\_pitch} - \text{rows}$, $\text{dst\_gap} = \text{dst\_pitch} - \text{rows}$, and byte span:
     $$
     \operatorname{SpanBytes}(\text{pitch}) = \bigl((\text{Bursts} - 1) \cdot \text{pitch} + \text{rows}\bigr) \times 32
     $$
     enforcing $1 \le \text{rows} \le 65535$ and $1 \le \text{Bursts} \le 4095$.
2. **PTO `T.SimtVF` Vector-Local Scalarization (`#3523`)**:
   - `tilelang/Carver/arch/driver/pto_Pass.py` lowers `T.alloc_local((N,), "float32xL")` into a scalar-addressed `pto.alloc_buffer((N * L,), pto.f32)` with lane offsets `slot * L + lane`.
3. **CUDA `cast_round="rs"` Stochastic-Rounding Validation (`#3350`)**:
   - `src/transform/lower_vectorize.cc` (`CastRoundIsSupported`) rejects non-`float32` sources, missing `rbits`, and `float8_e8m0fnu` targets (since E8M0 rounds toward $+\infty$ unconditionally) with a clear compile-time error before vectorized PTX emission.

---

## 3. Deterministic Offline Verification (`own-offline`)

Running `python experiments/flashinfer-rubin-gdn-ugpu-kcp-tilelang-ub2l1/run.py --check` produces `summary.json` and verifies all invariants:

| Verification Dimension | Metric / Scenario | Baseline / Negative Control | Optimized / Guarded Result | Evidence Tier |
|---|---|---|---|---|
| **Rubin GDN Decode + uGPU (`#6604`)** | VR200 (`212` SMs, `19.15 TB/s`) `B=32..8192` | Generic B200 kernel: `B=2048` takes `726.7 us` (`62%` BW) | `pf_dist=1766` tiles; `rubin` `640.2 us` (`71%`); `rubin_ugpu` **`516.5 us` (`88%`, `1.41x`)** | `author-benchmark` + `own-offline` |
| **Rubin MLA PDL & Kimi-K3 (`#7004/#7014`)** | 208-SM `items=54` vs `55`; B300 `M=4096, K=4` write | `items=55` with PDL steals wave-2 SMs; old write `87.649 us` | `items=54 -> PDL=True`, `items=55 -> PDL=False`; B300 new write **`85.473 us` (`1.0255x`)** | `author-benchmark` + `own-offline` |
| **CP KDA `"wave_fill1"` (`#7002`)** | GB300 (`148` SMs) `n_frag=1, NC=64, H=16` | Fixed `P_chain = 4` (`64/148` SMs active) | `kcp_plan` expands to `P=9` to fill SM wave; **`1.31x–2.47x`** vs FLA KCP across 144 configs | `author-benchmark` + `own-offline` |
| **TileLang Ascend / PTO / Cast (`#3530/#3523/#3350`)** | `16x64` FP16 subtile; `float32x4` 2 slots; `f32->e8m0` `rs` | Full-tile copy overwrites cols; vector ptr error; silent `e8m0` | `dst_span=6656 B` (`rows>65535` rejected); scalar buffer `8` elems; `e8m0`/`f16` `rs` rejected | `implementation` + `own-offline` |

---

## 4. Exercises & Falsifiable Follow-Up Questions for Solution Architects

1. **Architecture Sizing Exercise (VR200 uGPU Traffic & Bandwidth Roofline)**: For `B=2048, HV=64, K=V=128`, total BF16 state size is $2048 \times 64 \times 128 \times 128 \times 2 \text{ B} = 4.295 \text{ GB}$, requiring $8.590 \text{ GB}$ of read+write HBM traffic per decode step. Verify that at `19.15 TB/s` peak HBM bandwidth, the `516.5 us` latency achieved by `rubin_ugpu` corresponds to `88%` effective bandwidth utilization.
2. **Falsifiable Hypothesis**: Forcing `enable_pdl=True` when `plan.items > plan.units` (e.g., `items = 64 > 54` on a 208-SM GPU) in `cake_rubin_mla_nvfp4_decode` will cause Nsight Compute traces to show merge CTAs occupying SMs immediately after wave-1 `launch_dependents`, delaying the remaining 10 wave-2 decode clusters.
