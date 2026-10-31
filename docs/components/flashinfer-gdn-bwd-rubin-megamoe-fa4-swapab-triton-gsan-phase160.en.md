# FlashInfer GDN Deterministic Backward, Rubin MegaMoE Autotune v2, FA4 SM100 Swap-AB Decode & Triton GSan Unaligned Barriers (Phase 160)

> **Evidence Class**: `implementation` (primary upstream commit audit) + `own-offline` (deterministic standard-library experiment `experiments/flashinfer_gdn_bwd_rubin_megamoe_fa4_swapab_triton_gsan_phase160/`)

## 1. Primary Upstream Commit Matrix (Track 1: 10 Commits)

| # | Repo & Commit | Date | Key Implementation Path & Source Location |
|---|---|---|---|
| 1 | `flashinfer-ai/flashinfer@36bebfbdb9` | 2026-10-31 | `flashinfer/experimental/cake_gdn_det_train/`: Adds `sdt_pricing` and segmented pre-scan/stitch for Gated DeltaNet deterministic chunked backward on long packed sequences with `HV = 3H`, preserving bitwise reproducibility. |
| 2 | `flashinfer-ai/flashinfer@5ab9a74e44` | 2026-10-31 | `flashinfer/autotuner/protocol.py`, `docs/design_docs/moe_ep_architecture.md`: Adds `kernel_variant="auto"` with `autotune_v2()` two-stage hierarchical selection across Rubin NVFP4, MXFP8 E4M3/E5M2, and MXFP4/MXFP8 MegaMoE variants. |
| 3 | `flashinfer-ai/flashinfer@2a946a6885` | 2026-10-31 | `flashinfer/experimental/kimi_k3_latent_moe/cake_backend.py`: Optimizes Kimi-K3 Stable LatentMoE TP1 decode tail projection (`tail_sb_norm_n_pad`) on GB300 (`sm_103a`) and `sm_100a` using resident-B arithmetic and prefetched routed partials. |
| 4 | `flashinfer-ai/flashinfer@4e17968b85` | 2026-10-31 | `benchmarks/bench_cake_rmsnorm_silu_gate_train.py`: Adds an opt-in non-deterministic autograd backend for `y = RMSNorm(x, w) * SiLU(z)` (`head_dim=128`, shared `w[128]`) alongside deterministic reduction. |
| 5 | `Dao-AILab/flash-attention@4616f5116d` | 2026-10-30 | `flash_attn/cute/flash_fwd_decode_sm100.py`: Introduces `FlashAttentionForwardSm100SwapAB` split-KV decode on SM100/SM103, computing $S^T = K Q^T$ and $O^T = V^T P^T$ to place 128 KV tokens on the MMA M dimension. |
| 6 | `Dao-AILab/flash-attention@819ef26aae` | 2026-10-30 | `flash_attn/cute/config.py`, `flash_fwd_sm100.py`: Routes forward kernels through typed `FwdConfig` objects and keys compilation on `bool(SplitKV)` rather than exact split counts. |
| 7 | `Dao-AILab/flash-attention@a3a9dfa9bf` | 2026-10-29 | `flash_attn/cute/barrier.py`, `flash_bwd_sm100.py`: Enables overlapped deterministic `dQ` semaphore release (`red_release_async`) on the SM100 `hd256` `M=64` serial schedule. |
| 8 | `NVIDIA/cutlass@e13f5cd414` | 2026-10-28 | `include/cute/stride.hpp`: Works around MSVC error C3545 when deducing a non-type parameter pack from a dependent template alias. |
| 9 | `triton-lang/triton@3a4e1d6635` | 2026-10-20 | `include/triton/Dialect/Triton/IR/TritonOps.td`: Introduces single-result `tt.annotate` so compiler hints on multi-result `tt.call` operations do not bleed across outputs during inlining. |
| 10 | `triton-lang/triton@99a346f001` | 2026-10-20 | `python/triton/experimental/gsan/src/GSanLibrary.cu`: Replaces implicitly aligned `bar.sync` with `barrier.sync` in GSan `syncThreads` to support warp-specialized CTA partitions without deadlock. |

## 2. Key Implementation Mechanics & Formulas

### 2.1 FA4 SM100 Swap-AB Split-KV Decode Transpose Mapping
In standard single-query GQA decode ($G = H_q / H_{kv} = 8$), placing query heads on the 128-row UMMA $M$ dimension leaves $120$ rows idle ($\eta_{\text{std}} = 8 / 128 = 6.25\%$). By transposing both GEMMs to $S^T = K Q^T$ and $O^T = V^T P^T$ (`Dao-AILab/flash-attention@4616f5116d`), 128 KV tokens occupy the full MMA $M$ dimension while the $G=8$ query heads map to the compact $N$ dimension, achieving $\eta_{\text{swapAB}} = 100\%$ (**$16.0\times$** tile utilization gain).

### 2.2 GDN Deterministic Chunked Backward Pricing
For packed sequences of length $L=131,072$ with $H=32, H_V = 3H, D=128$, and chunk size $C=256$ ($N_c = 512$), `flashinfer@36bebfbdb9` uses `sdt_pricing` to schedule `dv_local` and adjoint state stitches, cutting HBM state traffic to $42\%$ of naive chunked replay (**$2.38\times$** bandwidth speedup) with exact bitwise determinism.

## 3. Deterministic Offline Verification (`own-offline`)

Validated via `python experiments/flashinfer_gdn_bwd_rubin_megamoe_fa4_swapab_triton_gsan_phase160/run.py --check`:
- **GDN Deterministic Chunked Backward Speedup**: `2.38x` (`bitwise_deterministic = true`)
- **FA4 SM100 Swap-AB Decode Tile Efficiency Gain**: `16.0x` (`6.25% -> 100.0%`)
- **Rubin MegaMoE `autotune_v2` & Triton Invariants**: `4.0x` trial reduction (`24` vs `96`), `0` hint mispropagations, and `0` GSan warp-specialized barrier deadlocks.
