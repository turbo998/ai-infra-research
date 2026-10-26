# FlashInfer Kimi-K3 Fused Router, All-Gather Matmul Producer-Tail, KDA 2-CTA CP State Walk, MiniMax-H3 NVFP4, RMSNorm Staged dWeight, SSU P-Split, Blackwell BF16xFP4 & TileLang SM120 Unroll Deep Dive (Phase 95 / Priority 104)

> **Evidence Class**: Upstream source implementation (`implementation`) + deterministic Python 3.12 standard-library modeling (`own-offline`). No live GPU hardware re-benchmarking is claimed.

## 1. Core Architectural Questions & Primary Upstream Evidence

| Source ID | Repository & Commit | Key Implementation Path & Architectural Invariants |
|---|---|---|
| `flashinfer-kimi-k3-fused-router-sm100a-pr7441-2026` | `flashinfer-ai/flashinfer` @ `bdd721538c` (PR #7441) | Updates `kimi_k3_fused_router` (896 experts, top-16, FP32 logits+bias, `block_m` $\in \{8, 16\}$, 28 cells) on `sm_100a`/`sm_103a`: 1-token closed-form 1-CTA builder; `2..16` tokens use `num_tokens`-CTA DSMEM cluster; 32-token cell uses a 16-CTA cluster serving up to 32 rows; 256-token cell uses an explicit 1-CTA cluster; 4-CTA clusters follow `CAPACITY_ROUTES`. |
| `flashinfer-allgather-matmul-peer-cta-release-wait-pr6885-2026` | `flashinfer-ai/flashinfer` @ `0af9594727` (PR #6885) | Fixes a CTA exit race in 7 `cta_group::2` Cake all-gather matmul programs by adding a 6-stage `producer_tail` wait on `mma_done` before the peer CTA load warp exits, preventing undefined async multicast `tcgen05.commit` arrives on exited CTA shared memory (PTX ISA §2.2.2, §9.7.18.6). |
| `flashinfer-cake-kda-2cta-per-launch-cp-state-walk-pr7436-2026` | `flashinfer-ai/flashinfer` @ `4f14b25dc6` (PR #7436) | Adds per-launch selectable 2-CTA records (each CTA handling 64 of the 128 state columns per `(segment, head)`) to the four context-parallel fragment state walks in `cake_kda_train`. |
| `flashinfer-minimax-h3-nvfp4-pre-attention-sm100-pr7434-2026` | `flashinfer-ai/flashinfer` @ `9e656abf02` (PR #7434) | Delivers all 4 `MiniMaxH3Nvfp4PreAttention` variants (`nvfp4`/`e4m3` × `sm_100a`/`sm_103a`), reporting `1.0257x–1.1198x` (`nvfp4`) and `1.0274x–1.1140x` (`e4m3`) speedups on GB300 (author-benchmark). |
| `flashinfer-cake-rmsnorm-staged-dweight-chunk-taper-pr7417-2026` | `flashinfer-ai/flashinfer` @ `a8d23b20bf` (PR #7417) | Updates `cake_rmsnorm_train` backward pairs on `sm_100a`/`sm_103a`/`sm_107a` with staged `dweight` reduction and chunk tapers (`max_halving=5`, `tail_div=12`, `rows_per_issue=4`), reaching `1.13x–1.18x` on `sm_107a` at $H=2048$. |
| `flashinfer-ssu-psplit-tma-ring-persistent-decode-pr7407-2026` | `flashinfer-ai/flashinfer` @ `fdd121f46b` (PR #7407) | Adds P-split (splitting 128-row program heads across 2 or 4 CTAs of 64/32 rows), TMA-ring, and persistent decode programs to Cake `selective_state_update` on `sm_100a` and `sm_103a`. |
| `flashinfer-blackwell-bf16-fp4-tcgen05-small-m-pr7419-2026` | `flashinfer-ai/flashinfer` @ `328e862b34` (PR #7419) | Regenerates 10 of 97 `mm_bf16_fp4` programs: 3 two-CTA $M=128$ `tcgen05` programs free TMEM while final stores are in flight via `tmem_dealloc_bar_addr`, and 7 small-$M$ split-K programs preload/convert all block scales before the main loop. |
| `flashinfer-grouped-mxfp8-quant-row-tiled-k7168-pr7431-2026` | `flashinfer-ai/flashinfer` @ `bdaa2401ad` (PR #7431) | Adds $K$-selected row-tiled `cake_grouped_mxfp8_quantize` programs for inputs $\ge 4\text{ MiB}$ and $B\cdot M \ge 16384$ ($K=4096$ 256-thread 2-row CTA, $K=2048$ 128-thread CTA) and integer UE8M0 scale-byte math at $K=7168$. |
| `tilelang-out-idx-caller-launch-stream-alloc-pr3649-2026` | `tile-ai/tilelang` @ `89b5b5b9ad` (PR #3649) | Allocates `out_idx` tensors on the caller-supplied `stream=` across `cython`, `nvrtc`, and `cutedsl` adapters via `output_allocation_stream`, preventing caching-allocator cross-stream overwrite races. |
| `tilelang-sm120-block-scaled-rega-gemm-k-unroll-pr3286-2026` | `tile-ai/tilelang` @ `32f56122e2` (PR #3286) | Explicitly unrolls the SM120 `is_gemm_rs` K-atom loop (`T.unroll(explicit=True)`) to prevent local-memory spills when $A$ spans $>1$ 64-wide K atom, and cuts shared-memory scale loads from 24 to 3 words per lane per K atom (`8.0x` reduction). |

## 2. Deterministic Verification

- **Command**: `python experiments/flashinfer_kimi_router_ag_matmul_kda_minimax_rmsnorm_ssu_bf16fp4_mxfp8_tilelang_phase95/run.py --check`
