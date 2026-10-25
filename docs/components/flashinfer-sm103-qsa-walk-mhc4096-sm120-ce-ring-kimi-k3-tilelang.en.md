# FlashInfer SM103 QSA Persistent-Walk Decode & tcgen05 Merge, SM120 QSA mma.sync 188-SM Split Overrides, Hidden-4096 4-Route mHC Sinkhorn Fusion, MiniMax-H3 Half-Width Tail, SM120 PCIe Copy-Engine Ring Piped All-Reduce, Kimi-K3 L2 Evict-First & Resident 2-CTA TP1 Tail GEMM, and TileLang Cross-Scope Partial-Sync Named-Barrier Deduplication

> **Language / 语言**: [English](flashinfer-sm103-qsa-walk-mhc4096-sm120-ce-ring-kimi-k3-tilelang.en.md) | [中文](flashinfer-sm103-qsa-walk-mhc4096-sm120-ce-ring-kimi-k3-tilelang.md)
> **Phase**: Phase 73 (`priority: 82`, Track A)
> **Question ID**: `q-phase73-flashinfer-sm103-qsa-walk-mhc4096-sm120-ce-ring-kimi-k3-tilelang`
> **Evidence Tiers**: `implementation` + `author-benchmark` + `own-offline` (`mastery: not-assessed`)

---

## 1. Executive Summary & Architectural Insights

On `2026-10-25`, upstream commits in `flashinfer-ai/flashinfer` and `tile-ai/tilelang` landed seven critical kernel and compiler invariants across Blackwell datacenter (`sm_100a` B200/GB200, `sm_103a` B300/GB300), Blackwell workstation/server (`sm_120a` RTX PRO 6000 Blackwell Server Edition, 188 SMs), and Hopper (`sm_90a` H100) pipelines:

1. **SM103 `cake_qsa_nvfp4` Persistent-Walk Decode & `tcgen05` One-Tile Fused Merge** (`flashinfer@131f85942e`): On GB300 (`152` SMs), at `rows=128`, `num_kv_heads=2`, `plain_splits=1`, the plain split-KV grid launches `256` CTAs (`33` page tiles each) across `ceil(256 / 152) = 2` waves, leaving `48` SMs idle during the second wave. The new `decode_tcgen05_persistent` kernel activates on `sm_103a` when `1 < ceil(rows * num_kv_heads * plain_splits / sm_count) <= WALK_MAX_WAVES = 2`, launching `152` persistent CTAs that evenly walk `256 * 33 = 8,448` `(row, kv_head, tile)` items (`55–56` tiles/CTA) and emit at most `walk_splits = 2` partial slots per `(row, kv_head)`. Concurrently, `merge_tcgen05` replaces four `16x256` sub-tile UMMA steps with a single `64x256` `tcgen05` accumulator tile, eliminating three intermediate `tcgen05.Wait` stalls. On GB300, this delivers **`1.07x–1.15x`** at `rows=1..4` and **`1.06x` eager (`71.89 -> 67.74 us` NHD, `72.11 -> 67.79 us` HND) / `1.07x` CUDA Graph (`72.37 -> 67.74 us` HND)** at `rows=128`.
2. **SM120 `cake_qsa_nvfp4` `mma.sync` Decode & 188-SM Split Overrides** (`flashinfer@09a77503f0`): For `sm_120a` (`188` SMs, 2 co-resident CTAs/SM at `69,632 B` SMEM and `192` threads -> `376` co-resident capacity), `cake_qsa_nvfp4` adds `mma.sync.aligned.m16n8k16` plain/cooperative decode and merge kernels, while `_SPLIT_OVERRIDES[188]` lowers splits at `rows=4, 8, 16, 32` from `[33, 23, 11, 5]` (`264–368` CTAs) to `[16, 11, 5, 3]` (`128–192` CTAs, one CTA/SM), eliminating same-SM contention and halving merge list lengths. On RTX PRO 6000 Blackwell Server Edition, this achieves **`1.27x–1.89x` eager speedup** (`rows=128` HND `320.93 -> 169.62 us` **`1.89x`**, NHD `301.26 -> 161.36 us` **`1.87x`**) and **`1.21x–1.90x` CUDA Graph speedup** over the FlashInfer reference.
3. **Hidden-4096 4-Route mHC Sinkhorn Fusion & TMA `CheckedMul` Int64 Overflow Guard** (`flashinfer@847172573a`): Adds five `sm_100a`/`sm_103a` programs (`bf16`, `tailpipe`, `tailpipe_wsmem`, `fp8_group128`, `nvfp4`) for `hidden=4096, routes=4` (20 Sinkhorn iterations) in `cake_deepgemm_mega_mhc` with `VARIANT_WSMEM_REGIME_SPLITS = 8` and `VARIANT_PLAIN_MIN_WAVES = 7`, plus `CheckedMul(int64_t, int64_t)` in host TMA descriptor encoding to prevent signed 32-bit stride overflow. Delivers **`1.038x–1.426x` on B200** (`T=1024`: `39.095 -> 27.424 us`) and **`1.041x–1.387x` on GB300** (`T=1024`: `35.510 -> 25.608 us`) over vLLM's `MHCFusedPostPreOp.forward_cuda`.
4. **MiniMax-H3 `out_proj_nvfp4` N-Split Half-Width Tail (`256x128`)** (`flashinfer@b4125442a5`): Dispatches a `256x128` output-column kernel when $M \ge 32$ and the partial last wave of the `256x256` grid fits within half a wave (up to 10 waves on `sm_100a` and 48 waves on `sm_103a`), shrinking SMEM from `187 KiB` to `122 KiB` and achieving **`1.003x–1.041x` on GB300** (`P=8, M=4816`: `82.2 -> 79.0 us`) and **`1.017x–1.052x` on GB200** (`98.2 -> 93.4 us`).
5. **SM120 `IpcVariant.COPY_ENGINE_RING_PIPED` Slow-Hop-Aware Pipelined Ring All-Reduce** (`flashinfer@68e3e09ad3`): Introduces an 8th PCIe IPC All-Reduce variant on `sm_120a` for 4/8 ranks when `chunk >= 4 MiB`, splitting each chunk into 2–3 pieces to overlap DMA Copy Engine peer pushes (`ce_stream`) with SM reduction kernels (`stream`) while weighting pieces via `decide_ring_slow_hops` (`ce_slow_hops` bitmask) across inter-socket CPU links. Achieves **`1.050x` at 48 MiB/rank (`2.1832 -> 2.0792 ms`)** and **`1.111x` at 96 MiB/rank (`4.4753 -> 4.0287 ms`, 3 pieces)** on 8×SM120 PCIe Gen5.
6. **Kimi-K3 Latent MoE `L2::evict_first` Decode Streaming & Resident 2-CTA TP1 Tail GEMM** (`flashinfer@1c0d2be434`): Marks stream-once weights with `L2::evict_first` at $T \le 128$ (boosting Front by **`1.079x–1.201x`** and TP1 Tail by **`1.156x–1.191x`** across B200/B300/GB200/GB300), and replaces the prefill TP1 Tail ($T \ge 1024$) with a claimed-item resident 2-CTA cluster + fused RMSNorm and a `256x512` wide tile ($T \ge 4096$ on 152 SMs, $T \ge 8192$ on 148 SMs), achieving **`1.028x–1.121x`** (GB200 $T=4096$: `244.7 -> 218.2 us` **`1.121x`**).
7. **TileLang `ThreadSync` Cross-Scope Partial-Sync Named-Barrier Deduplication** (`tilelang@efa1ffc1e3`): Fixes `ThreadPartialSyncRewriter::Rewrite` so that running `ThreadSync` sequentially on `"shared"` and `"shared.dyn"` scans existing `tvm_storage_sync` barrier IDs into `used_barrier_ids_` via `PostOrderVisit`, preventing producer warpgroup `[256, 384)` and consumer warpgroup `[0, 128)` from colliding on `bar.sync 3, 128` in `deepseek_v32/sparse_mla_fwd_pipelined.py` on H100.

---

## 2. Mathematical Formulation & Hardware Synchronization Boundaries

### 2.1 SM103 QSA Persistent-Walk Wave & Slot Equations

Given batch rows $R$, KV heads $H_{kv}=2$, selected pages per head $P_{sel} = 1 + N_{\text{topk}} = 33$, and plain splits $S_{\text{plain}}$, the plain grid size and wave count are:

$$G_{\text{plain}} = R \cdot H_{kv} \cdot S_{\text{plain}}, \quad W_{\text{plain}} = \left\lceil \frac{G_{\text{plain}}}{\text{SM}_{\text{count}}} \right\rceil$$

On `sm_103a` ($\text{SM}_{\text{count}} = 152$), `decode_tcgen05_persistent` is selected iff $S_{\text{plain}} < P_{sel}$ and $1 < W_{\text{plain}} \le \text{WALK\_MAX\_WAVES} = 2$. At $R=128, S_{\text{plain}}=1$, $G_{\text{walk}} = 152$ persistent CTAs partition $N_{\text{work}} = 128 \times 2 \times 33 = 8,448$ tiles into contiguous spans of $55\text{–}56$ tiles per CTA, bounding partial slots per `(row, kv_head)` to:

$$S_{\text{walk}} = 1 + \left\lfloor \frac{\lfloor N_{\text{work}} / G_{\text{walk}} \rfloor}{P_{sel}} \right\rfloor = 2$$

---

## 3. Quantitative Evaluation Matrix & Reproducible Verification

Run the deterministic stdlib simulator to verify all 7 invariants:

```powershell
python experiments\flashinfer_sm103_qsa_walk_mhc4096_sm120_ce_ring_kimi_k3_tilelang_sim\run.py --check
```

| Subsystem / Kernel | Hardware Platform | Workload / Config | Baseline Latency / State | Optimized Latency / State | Speedup / Outcome | Evidence Tier |
|---|---|---|---|---|---|---|
| `cake_qsa_nvfp4` Persistent-Walk + `tcgen05` Merge | GB300 (`sm_103a`, 152 SMs) | `rows=1..4` / `rows=128` (HND) | `11.71–15.92 us` / `72.11 us` (`72.37 us` Graph) | `10.53–13.95 us` / `67.79 us` (`67.74 us` Graph) | **`1.07x–1.15x`** / **`1.06x–1.07x`** | `author-benchmark` + `own-offline` |
| `cake_qsa_nvfp4` `mma.sync` + 188-SM Split Override | RTX PRO 6000 (`sm_120a`, 188 SMs) | `rows=4` NHD / `rows=128` HND | `28.59 us` / `320.93 us` (`321.59 us` Graph) | `16.13 us` / `169.62 us` (`169.10 us` Graph) | **`1.77x`** / **`1.89x` (`1.90x` Graph)** | `author-benchmark` + `own-offline` |
| `cake_deepgemm_mega_mhc` Hidden-4096 (4-Route, 20-iter) | B200 (`sm_100a`) / GB300 (`sm_103a`) | `T=1024` / `T=4096` vs vLLM | B200 `39.095 / 118.465 us`; GB300 `35.510 / 109.945 us` | B200 `27.424 / 86.913 us`; GB300 `25.608 / 79.744 us` | B200 **`1.426x / 1.363x`**; GB300 **`1.387x / 1.379x`** | `author-benchmark` + `own-offline` |
| `cake_minimax_h3` Out-Proj Half-Width Tail (`256x128`) | GB300 (`sm_103a`) / GB200 (`sm_100a`) | `P=8, M=4816` (`608` CTAs -> partial wave `16`) | GB300 `82.2 us` / GB200 `98.2 us` | GB300 `79.0 us` / GB200 `93.4 us` | GB300 **`1.041x`** / GB200 **`1.052x`** | `author-benchmark` + `own-offline` |
| `IpcVariant.COPY_ENGINE_RING_PIPED` (Slow-Hop Aware) | 8× SM120 PCIe Gen5 | `48 MiB/rank` (2 pieces) / `96 MiB/rank` (3 pieces) | `2.1832 ms` / `4.4753 ms` | `2.0792 ms` / `4.0287 ms` | **`1.050x`** / **`1.111x`** | `author-benchmark` + `own-offline` |
| `cake_kimi_k3_latent_moe` `evict_first` + 2-CTA TP1 Tail | B200 / B300 / GB200 / GB300 | Decode $T \le 128$ / Prefill $T=4096$ (GB200) | Decode baseline / `244.7 us` | `evict_first` / `218.2 us` (`256x512` wide tile) | Decode **`1.079x–1.201x`** / Prefill **`1.121x`** | `author-benchmark` + `own-offline` |
| TileLang `ThreadPartialSyncRewriter` Deduplication | Hopper H100 (`sm_90a`) | `sparse_mla_fwd_pipelined` (`shared` + `shared.dyn`) | Producer `[256,384)` & Consumer `[0,128)` collide on `bar.sync 3, 128` | `used_barrier_ids_` assigns `bar.sync 4, 128` | **Eliminates cross-warpgroup phase corruption** | `implementation` + `own-offline` |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Design Exercise (SM103 vs. SM100 Persistent-Walk Boundary)**: Why is `decode_tcgen05_persistent` restricted to `sm_103a` (`WALK_MAX_WAVES = 2`) and disabled on `sm_100a`?
   - *Answer Guide*: On `sm_100a` (GB200, 148 SMs), the integer division and boundary-switching overhead of walking `(row, kv_head, tile)` tuples outweighs the partial-wave savings at `rows=64` (`-9% to -10%`), whereas on `sm_103a` (GB300, 152 SMs) at `rows=128` (`256` CTAs = 1.68 waves), collapsing the second wave saves `4.15–4.63 us` (`+6% to +7%`).
2. **Falsifiable Hypothesis (SM120 Piped CE Ring Chunk Floor)**: Lowering `PIPED_MIN_CHUNK_BYTES` from `4 MiB` to `512 KiB` on 8×SM120 PCIe Gen5 will degrade latency below unpiped `COPY_ENGINE_RING` because per-piece CUDA Event synchronization and `cudaMemcpyPeerAsync` launch overhead dominate sub-`256 KiB` pieces.
