# FlashInfer SM103 QSA NVFP4 Two-Wave Persistent Walk & One-Tile Fused Merge, SM120 mma.sync Decode, Hidden-4096 4-Route Mega-mHC, SM120 Topology-Weighted PCIe CE Piped Ring, Kimi-K3 L2::evict_first & TileLang Barrier Fix

> **中文版本**: [flashinfer-sm103-qsa-persistent-mhc4096-sm120-piped-kimi-k3-tilelang.md](flashinfer-sm103-qsa-persistent-mhc4096-sm120-piped-kimi-k3-tilelang.md)  
> **Question ID**: `flashinfer-sm103-qsa-persistent-mhc4096-sm120-piped-kimi-k3-tilelang` (Phase 74, Priority 83)  
> **Reproducible Experiment**: [`experiments/flashinfer_sm103_qsa_persistent_mhc4096_sm120_piped_kimi_k3_tilelang_sim/run.py`](../../experiments/flashinfer_sm103_qsa_persistent_mhc4096_sm120_piped_kimi_k3_tilelang_sim/run.py)  
> **Evidence Class**: `implementation` / `author-benchmark` (upstream commits & PR benchmarks) + `own-offline` (deterministic stdlib simulator)

---

## 1. Architectural Motivation & Bottlenecks

Across Blackwell datacenter and workstation GPUs (`sm_100a` B200, `sm_103a` GB300, `sm_107a` Rubin, and `sm_120a` RTX PRO 6000 Blackwell Server Edition), six kernel and compiler bottlenecks emerged between `2026-10-22` and `2026-10-25`:

1. **SM103 Two-Wave Tail Latency and `tcgen05` Fused Merge Overhead** ([`flashinfer@131f85942edb`](https://github.com/flashinfer-ai/flashinfer/commit/131f85942edb00c9258bc6e393ccbe57b88b55bc)): On GB300 (`152` SMs) at `rows=128`, 2 KV heads, and `2,051` selected tokens (`33` 64-token tiles), `plain_splits=1` launches `256` CTAs across two waves (`152` + `104` SMs), forcing the busy SMs to serialize `2 × 33 = 66` tiles while `48` SMs sit idle in wave 2. Meanwhile, the small-batch (`rows=1,2`) `tcgen05` fused merge reduced only 8 output columns per iteration across 4 passes after `grid.sync()`.
2. **SM120 QSA NVFP4 Decode Without `tcgen05`/TMEM and Over-Splitting** ([`flashinfer@09a77503f004b8f36a38d23074fc5da6877de811`](https://github.com/flashinfer-ai/flashinfer/commit/09a77503f004b8f36a38d23074fc5da6877de811)): `sm_120a` (`188` SMs, 2 co-resident `99 KiB`-SMEM decode CTAs per SM = `376` fused CTAs) lacks `tcgen05` and TMEM, and generic split counts (`33,23,11,5` at `rows=4,8,16,32`) spend excessive bandwidth in partial-output merge.
3. **GLM-5.3-Flash Hidden-4096 4-Route Mega-mHC and 32-Bit TMA Stride Overflow** ([`flashinfer@847172573ad1d5a929a8e7fb67aa7c072f8f7dcb`](https://github.com/flashinfer-ai/flashinfer/commit/847172573ad1d5a929a8e7fb67aa7c072f8f7dcb)): Mega-mHC previously covered `H ∈ {1280, 2560, 3072}` without `H=4096` (`K = 4 × 4096 = 16,384`) kernels, and unguarded 32-bit `rows * stride` arithmetic in TMA descriptor setup overflows at `T >= 131,072`.
4. **SM120 Dual-Socket PCIe Slow Hops in Ring All-Reduce** ([`flashinfer@68e3e09ad3a52cbbd6b39d159d2fad7d1653073d`](https://github.com/flashinfer-ai/flashinfer/commit/68e3e09ad3a52cbbd6b39d159d2fad7d1653073d)): On 8×SM120 PCIe Gen5 systems spanning two CPU sockets, ring hops `3→4` and `7→0` traverse inter-socket links (`NVML SYSTEM`), throttling uniform single-piece rings.
5. **Kimi-K3 Stable LatentMoE Decode L2 Thrashing & TP1 Tail Imbalance** ([`flashinfer@1c0d2be434d6d57ca16a428d439c433b8b1e6f81`](https://github.com/flashinfer-ai/flashinfer/commit/1c0d2be434d6d57ca16a428d439c433b8b1e6f81)): At `T <= 128`, streamed expert weight tiles evict reused token activations from L2; at `T >= 1024`, the TP1 tail persistent tile-walk kernel suffers cross-SM tail imbalance.
6. **TileLang Cross-Address-Space Partial Barrier Deadlock & Top-K Shared-Memory Overflow** ([`tilelang@efa1ffc1e3dc`](https://github.com/tile-ai/tilelang/commit/efa1ffc1e3dc547b11fee5acf671daff1f0f94cb), [`tilelang@4da574c01326`](https://github.com/tile-ai/tilelang/commit/4da574c01326eab5da5345af0aec1d19ce473c6a)): Clearing `used_barrier_ids_` between `"shared"` and `"shared.dyn"` passes assigned `bar.sync 3, 128` to disjoint warpgroups (`[256, 384)` vs `[0, 128)`), deadlocking DeepSeek-V3.2 sparse MLA; meanwhile `tl_topk_impl`'s fixed `[2, 4096]` shared-memory queue overflowed when $>4,096$ elements fell into the same radix bucket.

---

## 2. Core Implementation Mechanics

### 2.1 SM103 QSA NVFP4 Persistent Walk & One-Tile Fused Merge (`131f85942edb`)

- **Persistent Walk (`decode_tcgen05_persistent`)**: Enabled on `WALK_ARCHES = ("sm_103a",)` whenever the plain split grid spans two waves (`1 < ceil(rows * num_kv_heads * plain_splits / sm_count) <= 2`) and the longest walked CTA is shorter than two plain waves. At `rows=128` on GB300 (`152` SMs, `256` pairs × `33` tiles = `8,448` slots), `152` persistent CTAs walk `55..56` contiguous `(pair, tile)` slots (`< 66` two-wave serial tiles) and emit `walk_splits = 2` partial slots per pair.
- **One-Tile `tcgen05` Fused Merge**: After `grid.sync()` in the cooperative `rows=1,2` build, all 32 threads reduce 32 groups of 8 columns (`256` head_dim) in a single pass over the `33 × 32` shared-memory staging table.

### 2.2 SM120 `mma.sync` Decode & `SPLIT_OVERRIDES` (`09a77503f004b8f36a38d23074fc5da6877de811`)

- Uses `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` with double-buffered `cp.async.cg` 64-token NVFP4 tiles and in-SMEM `cvt.rn.bf16x2.e2m1x2` + UE8M0 scale unpacking.
- `SPLIT_OVERRIDES` replaces formula splits `33,23,11,5` at `rows=4,8,16,32` with `16,11,5,3` while keeping `33` at `rows=1,2`.

### 2.3 Hidden-4096 4-Route Mega-mHC & `CheckedMul` (`847172573ad1d5a929a8e7fb67aa7c072f8f7dcb`)

- Adds 5 persistent `H=4096` (`K=16,384`) kernels (`bf16`, `tailpipe`, `tailpipe_wsmem`, `fp8_group128`, `nvfp4`) for GLM-5.3-Flash on `sm_100a` and `sm_103a`, fusing post-mixing, `BF16x3` split-K mapping, 20 Sinkhorn iterations, collapse, and RMSNorm, plus `__builtin_mul_overflow` (`CheckedMul`) TMA stride validation.

### 2.4 SM120 Topology-Weighted Pipelined PCIe CE Ring (`68e3e09ad3a52cbbd6b39d159d2fad7d1653073d`)

- `decide_ring_slow_hops` builds bitmask `0x88` (`(1 << 3) | (1 << 7)`) for inter-socket `NVML SYSTEM` hops, assigns larger weights to chunks that avoid slow hops, and splits payloads `>= 4 MiB/rank` into `2..4` pipeline pieces under `IpcVariant.COPY_ENGINE_RING_PIPED`.

### 2.5 Kimi-K3 `L2::evict_first` & Claimed-Item Resident TP1 Tail GEMM (`1c0d2be434d6d57ca16a428d439c433b8b1e6f81`)

- Streams expert weights with `L2::evict_first` (`DECODE_A_EVICT_FIRST = True`) at `T <= 128` and introduces a claimed-item resident `256×512` 2-CTA TP1 tail GEMM with fused RMSNorm at `T >= 1024`.

### 2.6 TileLang Barrier Tracking & Global Top-K Workspace (`efa1ffc1e3dc`, `4da574c01326`)

- Preserves `used_barrier_ids_` across `"shared"` and `"shared.dyn"` rewrites and replaces the fixed `[2, 4096]` shared-memory radix queue in `tl_topk_impl` with a `(batch, 2, seq_len)` global tensor workspace.

---

## 3. Quantitative Comparison & Verification

Run `python experiments/flashinfer_sm103_qsa_persistent_mhc4096_sm120_piped_kimi_k3_tilelang_sim/run.py --check`:

| Subsystem / Kernel | Hardware | Baseline | Optimized | Speedup | Evidence Class |
|---|---|---|---|---|---|
| QSA NVFP4 `rows=128` Two-Wave Persistent Walk | GB300 (`sm_103a`, 152 SMs) | `71.89 us` (256 CTAs) | `67.74 us` (152 CTAs) | **1.061x** (CG **1.068x**) | `author-benchmark` + `own-offline` |
| QSA NVFP4 `rows=1` One-Tile Fused Merge (read flush) | GB300 (`sm_103a`) | `11.70 us` | `10.02 us` | **1.168x** | `author-benchmark` + `own-offline` |
| QSA NVFP4 `rows=4` `mma.sync` + `SPLIT_OVERRIDES` | RTX PRO 6000 (`sm_120a`, 188 SMs) | `28.59 us` (33 splits) | `16.13 us` (16 splits) | **1.772x** | `author-benchmark` + `own-offline` |
| QSA NVFP4 `rows=128` HND `mma.sync` (CUDA Graph) | RTX PRO 6000 (`sm_120a`) | `321.59 us` | `169.10 us` | **1.902x** | `author-benchmark` + `own-offline` |
| Mega-mHC `H=4096` 4-Route (`T=1024`) | B200 (`sm_100a`) / GB300 (`sm_103a`) | `39.095 us` / `35.510 us` | `27.424 us` / `25.608 us` | **1.426x** / **1.387x** | `author-benchmark` + `own-offline` |
| PCIe CE Ring Piped All-Reduce (`96 MiB/rank` BF16) | 8×SM120 Dual-Socket PCIe Gen5 | `4.4753 ms` | `4.0287 ms` (3 pieces) | **1.111x** | `author-benchmark` + `own-offline` |
| Kimi-K3 LatentMoE Front `L2::evict_first` (`T=64`) | GB300 (`sm_103a`) | `53.07 us` | `44.18 us` | **1.201x** | `author-benchmark` + `own-offline` |
| Kimi-K3 LatentMoE TP1 Resident Tail GEMM (`T=1024`) | GB200 (`sm_100a`) | `120.87 us` | `107.81 us` | **1.121x** | `author-benchmark` + `own-offline` |
