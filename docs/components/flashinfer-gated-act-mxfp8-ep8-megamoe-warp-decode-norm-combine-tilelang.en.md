# FlashInfer Cake Gated SwiGLU MXFP8 (SM100/SM103/SM107), BF16 Rank-Major EP8 MegaMoE Arrive-Only Mailboxes, Warp-Decode Schedule Retuning, Router GEMM, Fused Norm Combine Lag-3 Pipeline & TileLang Target/Autotuner Guards

> **Document Status**: Phase 77 (`priority: 86`) First-Hand Source Audit + CPU/Offline Reproducible Verification  
> **Target Audience**: Cloud Solution Architects deepening expertise in NVIDIA Blackwell (`sm_100a` B200 / `sm_103a` GB300) and Rubin (`sm_107a` R200) kernel generation, MoE routing/decode pipelines, and TileLang compiler toolchains  
> **中文伴侣文档**: [flashinfer-gated-act-mxfp8-ep8-megamoe-warp-decode-norm-combine-tilelang.md](flashinfer-gated-act-mxfp8-ep8-megamoe-warp-decode-norm-combine-tilelang.md)

---

## 1. Architectural Context & Problem Statement

In frontier MoE training and low-latency decode pipelines, activation quantization, single-node 8-GPU Expert Parallel (EP8) collectives, small-batch routed fused-MoE decode, and DSL cross-compilation face five tightly coupled bottlenecks:

1. **Bandwidth Amplification in Dual-Direction (Rowwise + Colwise) SwiGLU + MXFP8 Quantization**: Forward `silu_and_mul_mxfp8_quantize` (`[M, 2K] -> [M, K]`) and backward `silu_and_mul_mxfp8_quantize_backward` (`[M, K] -> [M, 2K]`) must emit E4M3 payloads plus per-32-element RCEIL E8M0 scales in `128x4` swizzled TMA layout across `sm_100a` (B200, 148 SMs), `sm_103a` (GB300, 152 SMs), and `sm_107a` (R200, 216 SMs).
2. **EP8 MegaMoE Multi-GPU Barrier & Small-Batch Routing Launch Overhead**: On 8×B200 (`H=7168, I=2048, E=256` with 32 experts/rank, `top_k=8`, `1..128` tokens/rank), `MoEEpLayer` previously launched 3 separate staging copy kernels, used full all-peer barriers for input/partial stages, and launched a standalone routing kernel even when `M <= 16` (12 kernels total).
3. **Schedule Mismatches in NVFP4 W4A4 Routed Fused-MoE Decode (`cake_warp_decode`) and Router GEMM (`cake_router_gemm`)**: Across `1..32` decode tokens, the crossover between device-workfeed and output clearing strategies shifts by expert geometry; meanwhile, 1-CTA-per-expert scheduling in `cake_router_gemm` (`out[M, N] = A[M, K] · B[N, K]^T`) redundantly reloads `A[M, K]` across experts for `M=1..16`.
4. **Cross-Peer Reduction Memory Latency in `cake_fused_norm_combine` at `tokens >= 480`**: Tight loop distances across 8 peers and 2 norm tracks (`H=2560`) expose NVLink/HBM load latency during large-batch combine.
5. **Silent `sm_50` Mis-Compilation on GPU-Less Hosts & Autotuner Cache Key Collisions in TileLang**: Returning a bare `"cuda"` string when no GPU is detected caused TVM to silently default to `arch=sm_50`, while `AutoTuner.get_tunner()` omitted `warmup`, `rep`, and `timeout` when calling `set_profile_args()`, colliding `hash(profile_args)` cache keys.

---

## 2. First-Hand Source Audit & Implementation Mechanics

### 2.1 `flashinfer@bfe3de571c09` (`#7305`): `cake_gated_act_mxfp8_backend` Across SM100, SM103, and SM107

`flashinfer.cake_gated_act_mxfp8_backend` implements all 6 operator forms—forward (`[M, 2K] -> [M, K]`) and backward (`[M, K] -> [M, 2K]`) across `rowwise`, `colwise`, and `both` quantization modes:

- **Multi-Architecture Coverage**: Ships native programs for `sm_100a` (B200, 148 SMs), `sm_103a` (GB300, 152 SMs), and `sm_107a` (R200, 216 SMs), selecting one program per call by `M * K` with bitwise-identical output to the FlashInfer baseline.
- **Author-Reported Speedups**:
  - **B200 (`sm_100a`)**: **`1.075x–1.501x`** overall (forward **`1.196x–1.501x`**, e.g., `128x128` fwd colwise `3.53 -> 2.35 us` **`1.501x`**, `131072x8192` fwd colwise `1117.21 -> 829.95 us` `1.346x`; backward **`1.075x–1.474x`**, e.g., `128x128` bwd colwise `3.71 -> 2.52 us` **`1.474x`**).
  - **GB300 (`sm_103a`)**: **`1.044x–1.411x`** overall (forward **`1.169x–1.411x`**, backward **`1.044x–1.383x`**).
  - **R200 (`sm_107a`, 216 SMs)**: **`1.094x–1.697x`** overall (forward **`1.331x–1.697x`**, e.g., `131072x8192` fwd colwise `864.60 -> 509.53 us` **`1.697x`**, `16384x7168` fwd colwise `99.37 -> 60.56 us` `1.641x`; backward **`1.094x–1.427x`**).

### 2.2 `flashinfer@f911858bfc32` (`#7300`): BF16 Rank-Major EP8 MegaMoE Arrive-Only Mailboxes & Inline Routing

On 8×B200 (`sm_100a`), PR `#7300` optimizes `MoEEpLayer` (`H=7168, I=2048, E=256`, 32 experts/rank, `top_k=8`, `1..128` tokens/rank) via three mechanisms:

1. **Fused 3-into-1 Input Staging Copy**: Replaces 3 separate staging copy kernels with a single fused kernel.
2. **1-Warp Arrive-Only Per-Peer Mailboxes**: Converts the input and partial barriers into 1-warp arrive-only kernels that release a per-peer mailbox so `dispatch` and `combine` wait only for peers they actually read.
3. **Inline Routing Inside Dispatch for `M <= 16`**: Computes top-k routing directly inside `dispatch` when `M <= 16` tokens/rank, skipping the standalone route kernel and launching only **9 kernels** total.
- **Author-Reported Speedups (8×B200)**: **`1.03x–1.06x` eager speedup** (`173.1 -> 162.8 us` **`1.06x`** at `M=1` all-ranks, `533.6 -> 513.0 us` `1.04x` at `M=128`) and **`1.03x–1.08x` CUDA Graph speedup** (`165.9 -> 153.8 us` **`1.08x`** at `M=1` all-ranks, `160.7 -> 149.3 us` **`1.08x`** at `M=1` rank0-only).

### 2.3 `flashinfer@ba24b4214024` (`#7292`), `7b220a67a486` (`#7287`), `dc39352926e8` (`#7285`) & `2866b59272d7` (`#7311`)

- **`cake_warp_decode` Schedule Retuning (`#7292`)**: Reroutes **75** shape-token rows on `sm_100a` (GB200) for up to **`1.289x` (`1.29x`) speedup** (`e256_i1536_k8` at `M=20`: `161.68 -> 125.43 us`, ratio `0.776`) and **26** rows on `sm_103a` (GB300) for up to **`1.237x` (`1.24x`) speedup** (`e256_i1536_k8` at `M=1`: `24.02 -> 19.42 us`, ratio `0.809`).
- **`cake_router_gemm` Per-Shape Schedules (`#7287`)**: Gives each of the 32 `(M in 1..16, K in {6144, 7168})` programs its own schedule that reuses each loaded `A` chunk across multiple experts and splits `K` across a thread-block cluster where faster, beating 1-CTA-per-expert on all 96 rows/GPU—up to **`1.694x`** on GB200 (`6.720 -> 3.968 us`) and **`1.733x`** on GB300 (`6.656 -> 3.840 us`).
- **`cake_fused_norm_combine` Lag-3 Pipeline (`#7285`) & `cake_blackwell_bf16_fp4` M128 Pair PDL (`#7311`)**: Pipelines the `tokens >= 480` owner program at lag 3 with final store 4 steps behind (**`1.018x`** on 8×B200 `30.656 -> 30.112 us` at `1024` tokens, **`1.016x`** on 8×B300 `31.040 -> 30.560 us`); `#7311` regenerates large-M `tcgen05` M128 pair PDL programs for up to **`1.029x`** on GB300 (`10.860 -> 10.552 us`).

### 2.4 `tilelang@f1a34efea68c` (`#2860`) & `90f8908b5f17` (`#3337`): Compiler Target & Autotuner Contracts

- **Rejecting Arch-Less Bare `"cuda"` Targets (`#2860`)**: `_cuda_target_from_arch(arch: str | None) -> Target` raises `ValueError` when `arch is None` instead of returning `"cuda"`, preventing TVM from silently filling `arch=sm_50` on GPU-less build hosts.
- **Forwarding Autotuner Profile Args (`#3337`)**: `AutoTuner.get_tunner()` passes `warmup=self.warmup, rep=self.rep, timeout=self.timeout` into `.set_profile_args(...)` so `hash(profile_args)` cache keys never collide across distinct profiling budgets.

---

## 3. Quantitative Summary & Offline Verification

Executed via `experiments/flashinfer_gated_act_mxfp8_ep8_megamoe_warp_decode_norm_combine_tilelang/run.py`:

| Kernel / Subsystem | Architecture & Shape | Baseline ($\mu\text{s}$) | Optimized ($\mu\text{s}$) | Speedup / Gain | Provenance |
|---|---|---:|---:|---|---|
| `cake_gated_act_mxfp8` (fwd colwise) | `sm_100a` B200, `128x128` | `3.53` | `2.35` | **`1.501x`** | Author-benchmark (`#7305`) |
| `cake_gated_act_mxfp8` (fwd colwise) | `sm_107a` R200, `131072x8192` | `864.60` | `509.53` | **`1.697x`** | Author-benchmark (`#7305`) |
| `MoEEpLayer` EP8 MegaMoE (CUDA Graph) | 8×B200, `M=1` all-ranks | `165.9` | `153.8` | **`1.08x`** (9 vs 12 kernels) | Author-benchmark (`#7300`) |
| `cake_warp_decode` (NVFP4 W4A4) | `sm_100a` GB200, `e256_i1536_k8`, `M=20` | `161.68` | `125.43` | **`1.289x`** (ratio `0.776`) | Author-benchmark (`#7292`) |
| `cake_router_gemm` (`K=7168, N=128`) | `sm_103a` GB300, BF16, `M=16`, PDL=on | `6.656` | `3.840` | **`1.733x`** | Author-benchmark (`#7287`) |
| `cake_fused_norm_combine` (lag-3) | 8×B200, `H=2560`, `1024` tokens | `30.656` | `30.112` | **`1.018x`** | Author-benchmark (`#7285`) |

---

## 4. Architect Exercises & Falsifiable Follow-Ups

1. **Sizing Exercise**: Compute the single-layer latency reduction on GB200 when combining `cake_router_gemm` (`M=16`), `MoEEpLayer` EP8 MegaMoE (`M=16`, CUDA Graph), and `cake_fused_norm_combine` (`1024` tokens), and quantify the step-change at `M=17` when exiting inline routing (`9 -> 10` kernels).
2. **Falsifiable Hypotheses**:
   - **H1**: Extending inline routing in `MoEEpLayer` from `M <= 16` to `M <= 32` increases register pressure in `dispatch` enough to offset the saved standalone route kernel launch.
   - **H2**: On `sm_107a` (R200, 216 SMs), `cake_gated_act_mxfp8` in `both` mode remains HBM write-bandwidth bound rather than SFU bound for `M * K >= 10^8`.
