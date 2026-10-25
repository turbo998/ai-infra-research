# FlashInfer SM103 NVFP4 TMEM Attention, EP16 MXFP8 MegaMoE Relaxed Fence/Dispatch, Kimi-K3 TP12 ESS Routing, SM120 FP8 Varlen Attention, & TileLang Compiler Evolution

> **Question ID**: `flashinfer-sm103-nvfp4-tmem-ep16-megamoe-sm120-fp8-varlen-kimi-tp12-tilelang` (Phase 76, `priority: 85`)  
> **Depends On**: `flashinfer-cake-norm-combine-w4a16-moe-minimax-nsplit-frost-fp8-tilelang`  
> **Evidence ID**: `ev-flashinfer-sm103-nvfp4-ep16-megamoe-kimi-sm120-tilelang-2026`  
> **Reproducible Experiment**: `experiments/flashinfer_sm103_nvfp4_tmem_megamoe_ep16_kimi_ess_sm120_tilelang/run.py`  
> **Chinese Version**: [`flashinfer-sm103-nvfp4-tmem-megamoe-ep16-kimi-ess-sm120-tilelang.md`](flashinfer-sm103-nvfp4-tmem-megamoe-ep16-kimi-ess-sm120-tilelang.md)

---

## 1. Architectural Motivation & Bottlenecks

Across NVIDIA Blackwell Ultra (`sm_103a` / GB300), Blackwell (`sm_100a` / GB200), workstation/consumer Blackwell (`sm_120a` / RTX 5090 & RTX PRO 6000), and multi-backend Ascend/ROCm codegen in late October 2026, five low-level bottlenecks were addressed in upstream commits:

1. **SMEM Staging & Tile-Boundary Stalls in `sm_103a` NVFP4 Attention**: In `cake_nvfp4_attention` (BF16 `[B, H, S, 128]` quantized to NVFP4 E2M1 + E4M3 block scales, noncausal, `S % 512 == 0`), staging the attention probability matrix $P$ through shared memory and serializing tile boundaries limited `tcgen05` utilization on GB300.
2. **Cross-Rank Barrier Serialization & Dispatch Overhead in 16-Rank NVL72 `CakeMxfp8MegaMoeEp16`**: Issuing 16 sequential system-scope `release` atomics per cross-rank NVLink barrier (32 total across 2 barriers) serialized memory flushes across all 16 peers, while CTA-wide dispatch re-read `6 KiB` BF16 token rows per route.
3. **Premature Switch to Persistent Tail in Kimi-K3 TP12 LatentMoE (`256 <= M <= 960`)**: `cake_kimi_k3_tp12_tail` previously switched from Early-Shared-Scatter (`k1_twoshot_ess:grouped` + `k3_ess:grouped`) to `k3_persist_bulk` at `M = 256`, incurring persistent pipeline setup overhead at medium token counts (`M = 256..512`).
4. **SM120 MiniMax-H3 FP8 Varlen Attention & Out-Projection Host Sync Bubbles**: `minimax_h3_sm120_varlen_attention_fp8` suffered from separate statistics kernels and unified K/V refills, while `minimax_h3_nvfp4_out_proj` triggered two host stream synchronizations per call to read global scales and compute `alpha`, creating `186.9–223.3 us` GPU idle gaps between back-to-back launches.
5. **TileLang Ascend Partial L0 Pitch & ROCm HIP FP4 Constant Emission**: Ascend `copy_l1_to_l0` computed destination L0 stride from the partial slice width instead of the full L0 matrix width, while ROCm HIP codegen lacked `__tl_float_to_fp4(...)` constant emission and `enable_fp4_` header gating.

---

## 2. First-Hand Upstream Commit Audit

| Repository & Commit | Date | Evidence Tier | Key Source Files & Architectural Changes |
|---|---|---|---|
| `flashinfer-ai/flashinfer@d08a1f732193` (PR `#7328`) | `2026-10-25` | `implementation` + `author-benchmark` | `flashinfer/experimental/nvfp4_attention/csrc/cake_nvfp4_attention_aeef857fb82e9e29a633_kernel.cu`: expands `TMEM_NCOLS` from `432` to `480` with `TMEM_TMEM_P0_LO/HI_OFFSET` (`448`/`456`) and `P1_LO/HI_OFFSET` (`464`/`472`) to hold $P$ in Tensor Memory and overlap tile boundaries (**`1.02x–1.05x`** faster on GB300 `sm_103a`). |
| `flashinfer-ai/flashinfer@a4dc7d17fb6a` (PR `#7299`) | `2026-10-25` | `implementation` + `author-benchmark` | `csrc/cake_mxfp8_megamoe_ep16_f8a1e691688d06811749_kernel.cu`: replaces 16 sequential system-scope release atomics per barrier with **1 `fence.acq_rel.sys` + 16 relaxed system-scope atomics**, adds per-warp `cake_megamoe_dispatch_rows` copying each `6 KiB` BF16 row once, and balances FC1/FC2 static tiles (**`1.13x–1.18x`** faster on 16×GB300 NVL72). |
| `flashinfer-ai/flashinfer@06da6cc640a0` (PR `#7310`) & `506fac6203ae` (PR `#7317`) | `2026-10-25` | `implementation` + `author-benchmark` | `flashinfer/experimental/kimi_k3_tp12_tail/cake_backend.py`: raises `K3_PERSIST_MIN_TOKENS` from `256` to `961` so `9 <= M <= 960` uses `k3_ess:grouped` (**`1.060x–1.080x`** faster at `M=256, 512` on GB300 & GB200); caches `_cudnn_version_supports_shape_override()` (`-91.97%` overhead). |
| `flashinfer-ai/flashinfer@ba12a8322385` (PR `#7289`) & `4796d2effd9e` (PR `#7309`) | `2026-10-25` | `implementation` + `author-benchmark` | Fuses statistics, unit-scale MMA, and split K/V refill in `minimax_h3_sm120_varlen_attention_fp8` (**`1.244x–1.404x`** on RTX 5090; **`1.005x–1.104x`** on RTX PRO 6000); computes `alpha` in device FP64 in `minimax_h3_nvfp4_out_proj`, cutting back-to-back GPU idle gaps from **`186.9–223.3 us` to `3.1–3.6 us` (`~60x` reduction)**. |
| `tile-ai/tilelang@0bd33f953cf5` (`#3511`) & `9253431f2a17` (`#3548`) | `2026-10-25` | `implementation` | `ascend_lower_opaque.cc` + `ascend_sync_insert.cc`: passes explicit `l0_pitch` for partial L1->L0 copies and records exact L1 write extents; `codegen_hip.cc`: emits `__tl_float_to_fp4(...)` for `float4_e2m1fn` constants on ROCm. |

---

## 3. Formulation & Quantitative Analysis

### 3.1 SM103 NVFP4 TMEM Probability Staging
In `cake_nvfp4_attention` (`sm_103a`, CUDA 13.4), `TMEM_NCOLS` grows from `432` to `480` to allocate dedicated Tensor Memory columns `(448, 456, 464, 472)` for `P0_LO/HI` and `P1_LO/HI` alongside `NUM_K_PIPE_STAGES = 4` and `NUM_V_PIPE_STAGES = 5`. On GB300 (cold L2):
- `(4, 8, 4096)`: `0.1145 -> 0.1094 ms` (**`1.05x`**);
- `(1, 8, 32768)`: `1.4432 -> 1.4201 ms` (**`1.02x`**);
- `(8, 32, 8192)`: `2.8392 -> 2.7562 ms` (**`1.03x`**).

### 3.2 `CakeMxfp8MegaMoeEp16` Single-Fence Relaxed Atomics & Per-Warp Dispatch
Across 16 GB300 GPUs (`H=3072, I=5120, E=512, top_k=8`):
1. **NVLink Barrier**: Replaces $16$ sequential `atomicAdd.release.sys` per barrier with $1\times \texttt{fence.acq\_rel.sys} + 16\times \texttt{atomicAdd.relaxed.sys}$.
2. **Per-Warp Dispatch**: $9\text{ CTAs/rank} \times 12\text{ warps/CTA} = 108\text{ warps}$ dynamically claim tokens. Lanes `0..7` claim expert row slots for the 8 routes, broadcast via warp shuffle, and the 32-lane warp reads the `6 KiB` BF16 input row **once** ($32 \times 6 \times 16\text{ B} \times 2 = 6144\text{ B}$), fan-writing to all 8 remote NVLink buffers.
3. **16×GB300 NVL72 Speedups**: `16 tokens/rank` **`1.14x`** (`355.4 -> 312.6 us`), `32 tokens/rank` **`1.16x`** (`368.4 -> 318.5 us` hotset50), `64 tokens/rank` **`1.18x`** (`377.5 -> 320.5 us` hotset50).

### 3.3 Kimi-K3 TP12 ESS Routing & SM120 Device-Side `alpha`
- Raising `K3_PERSIST_MIN_TOKENS` from `256` to `961` keeps `9 <= M <= 960` on `k3_ess:grouped`, delivering **`1.060x–1.080x`** speedups at `M=256` and `M=512` on both GB300 and GB200 NVL72.
- Computing $\alpha = \text{float32}(1.0 / (\text{double}(s_a) \cdot \text{double}(s_w)))$ on device in `cake_minimax_h3_sm120_quant_out_proj` removes 2 host synchronizations per call, shrinking back-to-back GPU idle gaps by **`~60x`** (`186.9–223.3 us` to `3.1–3.6 us`).
