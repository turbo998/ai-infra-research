# FlashInfer MUSE Spark Norm SM107a, Grouped FP8 Whole-Row Store, MiniMax-H3 NVFP4, DSA Dynamic Tickets, Deterministic GDN Training, and FA4/TileLang Validation (Phase 118 / Priority 127)

> **Evidence Classification**: Upstream `implementation` + `author-benchmark` (10 public upstream commits, 2026-09-23 to 2026-10-28) + repository `own-offline` deterministic arithmetic/topology validation (`experiments/flashinfer_muse_spark_norm_fp8_gemm_minimax_dsa_gdn_fa4_tilelang_phase118/run.py`). No unverified on-GPU hardware re-benchmarking is claimed.

## 1. Architectural Problem Statement & Background

Across Blackwell (`SM100a`), Blackwell Ultra (`SM103a`), Rubin-transition (`SM107a` / `R200`), and GeForce Blackwell (`SM120a`) operator stacks, five low-level bottlenecks recently surfaced:

1. **Under-occupied 16-Head `rmsnorm_sigmoid_gate` CTAs and Missing Small-Token PDL Windows on `SM107a`**: Previous `cake_muse_spark_norm` routed 16-head calls to a 64-thread 16-row (single-token) program and lacked native `bf16_pdl` programs for 128 heads (9–24 tokens) and 32 heads (10–12 tokens), falling back to CuTe under programmatic dependent launch.
2. **Scalar Epilogue Degradation on Unaligned Grouped FP8 GEMM Outputs**: When the BF16 output tensor of `group_gemm_fp8_nt_groupwise_contiguous` is only 2-byte aligned rather than 16-byte TMA-aligned, the host routes to `deepk_c2_scalar_output`, where element-wise scalar stores severely degraded B200/GB300 latency.
3. **DSA Indexer Tail Imbalance, One-Shot MoE All-Reduce Broadcast Amplification, and GDN/MLA Backward Numerics**:
   - Static CTA unit assignment in `cake_dsa_indexer_topk` on `SM103a` allowed a single re-streamed wide-scan unit to dictate tail latency;
   - One-shot `cake_moe_finalize_allreduce_fusion` at world sizes 4 and 8 broadcast every rank's partial across all peers, multiplying NVLink traffic at medium/large token counts;
   - Gated DeltaNet (`chunk_gated_delta_rule`) training lacked a bitwise-deterministic stream-ordered backend on SM100/SM103, and FlashAttention-4 SM100 sparse MLA backward performed three premature BF16 roundings before forming $dS = P \circ (dP - dP_{\text{sum}}) \cdot \text{scale}$.

---

## 2. Primary Upstream Implementation Analysis (10 Zero-Overlap Commits)

| # | Repository & Commit | Date | Implementation Path & Microarchitectural Mechanism |
|---|---|---|---|
| 1 | `flashinfer-ai/flashinfer@b08e86326a` (`#7505`) | 2026-10-28 | `flashinfer/cake_muse_spark_norm_backend/`: Unifies `ROW_ALIGNMENT_BYTES = 32` (`GATE_STRIDE_ELEMENTS = 16`), adds an `sm_107a` 128-thread multi-token program (4 tokens of 16 heads per CTA, 2 rows/thread at 256-bit width) and tail-free `bf16_pdl` programs. Measured on R200 (`sm_107a`, 212 SMs): **1.5366x / 1.5616x** speedup at `(16, 16384)`. |
| 2 | `flashinfer-ai/flashinfer@ccb6fc8d4c` (`#7597`) | 2026-10-28 | `csrc/cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_3be71ae2bb86939e1cd8_kernel.cu`: Adds a whole-row output store program for `deepk_c2_scalar_output` on `sm_100a`/`sm_103a` (28 programs behind 26 host routes). On B200 with 2-byte offset outputs, `correctness_unaligned_output` drops from `21.95 us` to `5.36 us` (**4.10x**) and `ldpz_mixed_m301` reaches **2.08x**. |
| 3 | `flashinfer-ai/flashinfer@574d6d8d53` (`#7599`) | 2026-10-28 | `csrc/cake_minimax_h3_sm120_nvfp4_varlen_attention/`: Producer-first overlapped `SM120` NVFP4 varlen attention (`[tokens, 56, 128]`), enabling `allow_oob_box=(false, true, false)` for tail-tile zero fill, inserting `fence.proxy.async.shared::cta`, and caching the `49536 B` dynamic shared memory attribute via a `thread_local uint64_t prepared_devices` bitmask. |
| 4 | `flashinfer-ai/flashinfer@b532fa4443` (`#7571`) | 2026-10-28 | `flashinfer/experimental/cake_dsa_indexer/`: `sm_103a` wide-scan programs dynamically draw work units from a 32-bit ticket word in a 128-byte workspace tail (`TICKET_TAIL_BYTES = 128`), handing out near-equal units in ascending block order and unequal units longest-first. Reaches **1.13x** speedup on GB300 `Z1k4096` (`12.701 ms` -> `11.285 ms`). |
| 5 | `flashinfer-ai/flashinfer@3cf451835d` (`#7511`) | 2026-10-28 | `csrc/cake_moe_finalize_allreduce_fusion/`: Adds two-shot (reduce-scatter to owner rank + rank-order sum + all-gather in the same Lamport buffer) programs for WS4 (`>=96` tokens) and WS8 (`>64` tokens) at `H=7168`, plus an `own-partial` WS2 large-token program. Achieves **1.121x** geomean across 720 shapes and **2.902x / 2.874x** at WS8 2048 tokens on B300/B200 (up to **3.468x**). |
| 6 | `flashinfer-ai/flashinfer@6b6c9f996d` (`#7592`) | 2026-10-28 | `flashinfer/experimental/cake_gdn_det_train/`: Adds a bitwise-deterministic Gated DeltaNet chunked training backend for `SM100`/`SM103` (`d=128, chunk=64`) with a fused backward tail at `HV = 2H` and zero atomics. Outperforms FLA 0.5.2 by **1.184x–3.587x** on B200 (`1x16384` drops from `2.9256 ms` to `0.9421 ms`, **3.106x**). |
| 7 | `flashinfer-ai/flashinfer@ee76d93ae8` (`#7595`) | 2026-10-28 | `flashinfer/experimental/nvfp4_mla_decode/`: Adds `decode_xv` with vote-gated max exchange (`XV_WARMUP_TILES = 48`, `XV_MIN_STEADY_TILES = 307`) on `SM103` and `decode_short_epi_warp` (`SHORT_EPI_MIN_TAIL_ROWS = 32`) on `SM100`/`SM103`. |
| 8 | `flashinfer-ai/flashinfer@94090279f3` (`#4559`) | 2026-10-27 | `flashinfer/fused_moe/`: Reduces W4A16 MoE scale factors without materializing large intermediate temporary tensors (saving `28.0 MiB` on a `[64, 4096, 56]` FP16 shape). |
| 9 | `Dao-AILab/flash-attention@ce1d8313d5` | 2026-09-23 | `flash_attn/cute/flash_bwd_mla_sm100.py`: Keeps $P^T$ normalization (`scale_p`) and the $dS = P \circ (dP - dP_{\text{sum}}) \cdot \text{scale}$ product in FP32 inside SM100 sparse MLA backward, eliminating 2 of the 3 BF16 roundings. |
| 10 | `tile-ai/tilelang@af08dae6e4` (`#3465`) | 2026-10-28 | `src/backend/common/op/atomic_simt.h`: Shares SIMT atomic iteration and index validation across CUDA and ROCm (`atomic_add.cc` / `atomic_reduce.h`), rejecting mismatched region ranks or extents with a clear `ValueError`. |

---

## 3. Key Formulas, Data Movement & Offline Validation Results

### 3.1 MoE Finalize + All-Reduce: One-Shot vs. Two-Shot NVLink Traffic
For world size $W$, hidden dimension $H = 7168$, and element size $b = 2\text{ B}$:
- **One-Shot Broadcast**: Every rank writes its partial sum to all $W - 1$ peers:
  $$V_{\text{1-shot}} = W(W - 1)Hb$$
- **Two-Shot (Reduce-Scatter + All-Gather)**: Every rank sends a $1/W$ slice to the chunk owner ($(W-1)Hb$ total) and the owner broadcasts the rank-ordered sum back ($(W-1)Hb$ total):
  $$V_{\text{2-shot}} = 2(W - 1)Hb$$
At $W = 8$, $V_{\text{2-shot}} / V_{\text{1-shot}} = 2 / 8 = 0.25$, cutting cross-GPU NVLink traffic by **75%** (`moe_ar_ws8_nvlink_traffic_reduction_ratio = 0.75`) while maintaining bitwise identity via deterministic rank-ordered addition.

### 3.2 Offline Verification Summary (`run.py --check`)

- `norm_row_alignment_bytes`: `32` (`1.5367x` s1 speedup at `(16, 16384)` on R200 `sm_107a`)
- `grouped_fp8_gemm_unaligned_b200_speedup`: `4.1` (`4.10x` B200 speedup on 2-byte offset output; `2.08x` on `ldpz_mixed_m301`)
- `minimax_dyn_smem_bytes`: `49536` (`thread_local` per-device bitmask + axis-1 `allow_oob_box` tail-tile zero fill)
- `dsa_ticket_tail_bytes`: `128` (`1.1255x` / reported `1.13x` speedup on GB300 `Z1k4096`)
- `moe_ar_ws8_nvlink_traffic_reduction_ratio`: `0.75` (`2.902x` speedup at 2048 tokens on B300)
- `gdn_det_train_16k_b200_speedup`: `3.105` (`2.9256 ms` -> `0.9421 ms` on B200 `1x16384 H8/HV16`)
- `nvfp4_mla_xv_min_total_tiles`: `355` (`48` warm-up tiles + `307` steady-state tiles)
- `fa4_sparse_mla_bwd_roundings_saved`: `2` (single BF16 rounding at $dS$ store instead of 3)
- `tilelang_simt_invalid_atomic_shapes_rejected`: `2` (rejects both rank and extent mismatches)

---

## 4. Exercises & Falsifiable Questions for Cloud Solution Architects

1. **Sizing Exercise**: For $W = 8, H = 7168$, and BF16 ($2\text{ B}$) at $T = 2048$ tokens, compute the total NVLink data transferred per fused MoE finalize + all-reduce launch under One-Shot vs. Two-Shot. (*Answer*: One-Shot transfers $1568\text{ MiB}$ whereas Two-Shot transfers $392\text{ MiB}$, saving $1176\text{ MiB}$ per layer).
2. **Falsifiable Hypothesis**: Why does `cake_nvfp4_mla_decode` gate `decode_xv` behind `XV_MIN_STEADY_TILES = 307` past a `48`-tile warm-up? Because vote-gated max exchange incurs a setup pipeline overhead equivalent to ~48 tiles per piece; on short KV sequences (`< 355` tiles), `decode_short` / `decode_short_epi_warp` avoids this pipeline tax and frees tensor memory immediately after the last epilogue read.
