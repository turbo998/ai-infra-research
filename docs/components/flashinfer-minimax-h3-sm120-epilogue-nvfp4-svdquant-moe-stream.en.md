# FlashInfer MiniMax-H3 SM120 Epilogue Proxy Fence & Ping-Pong Barrier Close, CAKE NVFP4 SVDQuant 15-Program GEMM Backend, and Rubin MoE `execute_on_streams` API

> **Language / 语言**: [中文 (`flashinfer-minimax-h3-sm120-epilogue-nvfp4-svdquant-moe-stream.md`)](flashinfer-minimax-h3-sm120-epilogue-nvfp4-svdquant-moe-stream.md) | **English (Current)**
> **Evidence Class**: `implementation` + `author-benchmark` (upstream commits) + `own-offline` (deterministic stdlib simulator)
> **Upstream Anchors**:
> - `flashinfer-ai/flashinfer@aacd0f134f2a35eaea9ea534aa5b5b31534db449` (`2026-10-20`, PR `#6847`)
> - `flashinfer-ai/flashinfer@ec3bcf0301dac6455897d4a450115fdd8fea1bd3` (`2026-10-20`, PR `#6878`)
> - `flashinfer-ai/flashinfer@6445497657b7bc06fcc768d6d499e38627221818` (`2026-10-20`, PR `#6488`)
> - `flashinfer-ai/flashinfer@5ea501832bf7e37a7acde903a32329d580c0e9b4` (`2026-10-20`, PR `#6869`)

---

## 1. Architectural Findings & Root-Cause Analysis

### 1.1 MiniMax-H3 SM120 Dense & Varlen Attention: Generic/Async Proxy Fence Ordering and Ping-Pong Named Barrier Closure (`aacd0f1`)

In `kernel_minimax_h3_dense_attention` (`csrc/cake_minimax_h3_dense_attention_sm120a.cu`) and its FP8/NVFP4 varlen counterparts on Blackwell (`sm_120a` / `sm_103a`), two hardware-level synchronization bugs were fixed:

1. **Cross-Warp Generic-to-Async Proxy Hazard (`fence.proxy.async.shared::cta`)**:
   - The epilogue stages output $O$ in the shared-memory buffer `q_smem` via the **generic proxy** (which Q scaling also writes and the epilogue reads back for global stores). In the next persistent work item, Warp 0's elected thread refills `q_smem` using a TMA load (`mbarrier_arrive_expect_tx(q_full_addr, 32768)`) via the **async proxy**.
   - **Pre-fix defect**: The kernel issued `fence.proxy.async.shared::cta` at the *start* of the next work item, *after* the end-of-item `__syncthreads()`. Because a thread's proxy fence only orders that thread's own prior generic-proxy accesses, warps `1..7` had unordered generic-proxy writes/reads relative to Warp 0's async TMA refill (`7` unordered warps per refill, `49` hazards across 8 items).
   - **Fix**: Every thread now executes `asm volatile("fence.proxy.async.shared::cta;" ::: "memory");` after its last `q_smem` access and *before* the end-of-item `__syncthreads()`, eliminating the start-of-item fence (`49 -> 0` hazards).

2. **Unclosed Ping-Pong Named Barrier Round at CTA Exit**:
   - In dense, FP8 varlen, and NVFP4 varlen attention kernels, Warp Group 0 (`warp < 4`, 128 threads) and Warp Group 1 (`warp >= 4`, 128 threads) alternate on named barriers `1` and `2` (`256` threads). Group 1 pre-arrives on barrier `1` once before the persistent loop, running one round ahead of Group 0's syncs. At loop termination, Group 1's final arrival leaves named barrier `1` open (`128/256` threads arrived).
   - **Fix**: Group 0 executes one closing `if (warp < 4) { asm volatile("barrier.sync 1, 256;" ::: "memory"); }` after the persistent loop, closing the round cleanly (`compute-sanitizer racecheck/synccheck` clean on GB300 `sm_103a`, within `±0.35%` timing).

---

### 1.2 CAKE NVFP4 SVDQuant 15-Program GEMM Backend Convergence (`ec3bcf0`)

`flashinfer.mm_nvfp4_svdquant(..., backend="cake")` computes the fused NVFP4 + low-rank BF16 LoRA GEMM:
$$\text{out} = \text{BF16}\!\left(\alpha \cdot \left(\text{deq}(A)\,\text{deq}(B)^\top + D\,L_1^\top\right) + \text{bias}\right)$$
across LoRA ranks $r \in \{32, 64, 96, 128\}$ with and without bias. PR `#6878` converges the backend into the vendorable package `flashinfer/cake_nvfp4_svdquant_gemm_backend` with **15 Cake programs** (replacing 7 legacy programs) for `sm_100a` (B200) and `sm_103a` (GB300):

| Target GPU | Shape $(M, N, K, r, \text{bias})$ | Previous Cake ($\mu\text{s}$) | Best CUTLASS ($\mu\text{s}$) | New 15-Prog Cake ($\mu\text{s}$) | Speedup vs Prev Cake | Speedup vs CUTLASS | Algorithmic $\text{TFLOPS/s}$ |
|---|---|---:|---:|---:|---:|---:|---:|
| **B200 (`sm_100a`)** | `(129, 3072, 3072, 32, 1)` | 10.66 | 8.19 | **6.46** | **1.65x** | **1.267x** | 381.20 |
| **B200 (`sm_100a`)** | `(129, 3072, 3072, 32, 0)` | 6.46 | 8.19 | **6.34** | **1.02x** | **1.293x** | 388.42 |
| **B200 (`sm_100a`)** | `(129, 3072, 12288, 128, 1)` | 26.37 | 16.29 | **13.95** | **1.89x** | **1.167x** | 705.29 |
| **B200 (`sm_100a`)** | `(6889, 12288, 3072, 32, 1)` | 319.39 | 103.23 | **97.38** | **3.28x** | **1.060x** | 5407.76 |
| **B200 (`sm_100a`)** | `(16384, 12288, 3072, 32, 1)` | 731.39 | 253.40 | **242.88** | **3.01x** | **1.043x** | 5152.94 |
| **GB300 (`sm_103a`)** | `(129, 3072, 3072, 32, 1)` | 10.46 | 7.20 | **5.98** | **1.75x** | **1.203x** | 411.80 |
| **GB300 (`sm_103a`)** | `(6889, 12288, 3072, 32, 1)` | 293.30 | 85.17 | **83.17** | **3.53x** | **1.024x** | 6331.70 |
| **GB300 (`sm_103a`)** | `(16384, 12288, 3072, 32, 1)` | 673.32 | 195.98 | **187.87** | **3.58x** | **1.043x** | 6661.77 |

---

### 1.3 Rubin Locality-Domain MoE `execute_on_streams` Adaptation (`6445497`)

- Following `pytorch/pytorch#199128`, `torch.cuda.execute_on_streams(streams, fn, *inputs)` requires explicit callback input sequences rather than passing an implicit stream index.
- FlashInfer updates both FC1 and FC2 locality-domain execution in `fused_moe/cute_dsl/fused_moe.py` to pass `range(len(localized_streams))` explicitly, and computes `cp.reduce.async.bulk` byte counts (`cta_tile_n * (c_dtype.width // 8)`, `512 B` for BF16 and `1024 B` for FP32 at `cta_tile_n=256`) directly at the const-specialized call sites.

---

## 2. Deterministic Verification (`own-offline`)

- **Experiment Script**: [`experiments/flashinfer_minimax_h3_sm120_epilogue_nvfp4_svdquant_moe/run.py`](../../experiments/flashinfer_minimax_h3_sm120_epilogue_nvfp4_svdquant_moe/run.py)
- **Command**: `python experiments/flashinfer_minimax_h3_sm120_epilogue_nvfp4_svdquant_moe/run.py --check`
- **Verified Invariants**:
  - `sm120_proxy_hazards_eliminated = true` (`49 -> 0` cross-warp proxy hazards)
  - `sm120_ping_pong_barrier1_closed = true` (`1 -> 0` open named-barrier rounds)
  - `cake_nvfp4_svdquant_15_programs_converged = true` (`3.28x` B200 / `3.58x` GB300 vs previous Cake; `1.293x` vs CUTLASS)
  - `rubin_moe_explicit_stream_inputs_verified = true` (`512 B` BF16 / `1024 B` FP32 bulk reduce copy sizes)
