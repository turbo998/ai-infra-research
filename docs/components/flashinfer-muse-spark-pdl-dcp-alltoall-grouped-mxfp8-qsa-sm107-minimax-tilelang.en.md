# FlashInfer Muse-Spark Norm PDL, DCP All-to-All LL128 FIFO, Grouped MXFP8 Shape Dispatch, Rubin SM107 NVFP4 QSA Decode, and TileLang Pipeline Barrier Fixes

**中文版**: [flashinfer-muse-spark-pdl-dcp-alltoall-grouped-mxfp8-qsa-sm107-minimax-tilelang.md](flashinfer-muse-spark-pdl-dcp-alltoall-grouped-mxfp8-qsa-sm107-minimax-tilelang.md)

## 1. Context and Core Architectural Problems

Across NVIDIA Blackwell (`sm_100a` B200/GB200, `sm_103a` B300/GB300, workstation `sm_120a` RTX 5090/PRO 6000) and next-generation Rubin (`sm_107a` R200 with 212 SMs), short-sequence decode and Mixture-of-Experts (MoE) quantization tail latency are increasingly governed by four microarchitectural bottlenecks:

1. **Per-Layer Norm/Gate Launch Overhead and $O(W)$ Admission Scanning**: In Muse-Spark models, `rmsnorm_sigmoid_gate` and `gated_residual_rmsnorm` execute in $0.86\text{--}2.27\;\mu\text{s}$ at $T=1\text{--}8$. Without Programmatic Dependent Launch (PDL) and a sub-linear window admission index, launch gaps and Python dispatch loops erode kernel gains.
2. **Decode Context Parallel (DCP) NVLink LL128 FIFO Slot Release Latency**: In `cake_dcp_alltoall` (CP2/CP4), releasing the final LL128 FIFO slot at the tail of the current kernel and staging received outputs through shared memory add avoidable synchronization latency.
3. **Wasted CTAs under Sparse MoE Masks in Grouped MXFP8 Quantization**: `cake_grouped_mxfp8_quantize` quantizes `x[B, M, K]` using device-resident `mask[B]`. Reading `mask[B]` on the host incurs a D2H sync, while launching one CTA row per $M$ (`row-2D`) wastes most CTAs when $B \cdot M \ge 16,384$ and expert masks are sparse.
4. **Rubin `SM107` (`R200`) `tcgen05` NVFP4 Sparse Paged GQA and TileLang Compiler Pipeline Invariants**: R200 (`sm_107a`, 212 SMs) requires `tcgen05` shared-memory descriptor specialization and capacity-aware split-KV sizing; meanwhile, TileLang's CUDA backend had a two-phase `mbarrier` hang in multi-warp pure-TMA producers and a 4D buffer double-expansion crash across sibling `T.Pipelined` loops.

---

## 2. First-Hand Source Code Audit

### 2.1 FlashInfer `cake_muse_spark_norm` PDL Route and $O(\log W)$ Admit Index (`#7341`, commit `31d0a811aebe`)

In `flashinfer/cake_muse_spark_norm_backend/__init__.py`, the `bf16_pdl` route selects per-token-window launch forms (`pdl`, `late_pdl`, `plain` borrowed without the PDL attribute where PDL regresses, or `gap` delegating to CuTe DSL + PDL). At module import, windows are folded per architecture and head count into sorted disjoint spans $\text{above} < \text{rows} \le \text{at\_most}$ queried via `bisect`:
- **B200 (`sm_100a`)**: `rmsnorm_sigmoid_gate` (`h32, T=1`) improves from $1.248\;\mu\text{s}$ to $0.866\;\mu\text{s}$ (**1.441x**); `gated_residual_rmsnorm` (`mode1, M=8`) improves from $2.272\;\mu\text{s}$ to $1.506\;\mu\text{s}$ (**1.509x**).
- **GB300 (`sm_103a`)**: `rmsnorm_sigmoid_gate` (`h32, T=1`) improves from $1.235\;\mu\text{s}$ to $0.862\;\mu\text{s}$ (**1.433x**); `gated_residual_rmsnorm` (`mode1, M=1`) improves from $2.007\;\mu\text{s}$ to $1.369\;\mu\text{s}$ (**1.466x**).

### 2.2 `cake_dcp_alltoall` Deferred Last LL128 FIFO Slot Release and Register Plain Stores (`#7340`, commit `53feaae34c0a`)

In `csrc/generated/dcp_alltoall/cake_dcp_alltoall_{ff3dfded867e5738be23,b8b8ebc8b91fd813f375}_kernel.cu`, the receiver releases its last NVLink LL128 FIFO slot at the start of the *next* launch and writes received entries directly from registers using plain stores:
- **Speedups (`1.15x–1.33x` across all shapes)**: On GB300 (`sm_103a`), CP2 $T=1$ warm drops from $5.872\;\mu\text{s}$ to $4.784\;\mu\text{s}$ (**1.23x**) and CP4 $T=1$ from $5.440\;\mu\text{s}$ to $4.480\;\mu\text{s}$ (**1.21x**). On GB200 (`sm_100a`), CP2 $T=1$ warm drops from $5.456\;\mu\text{s}$ to $4.288\;\mu\text{s}$ (**1.27x**) and cold from $5.584\;\mu\text{s}$ to $4.192\;\mu\text{s}$ (**1.33x**).

### 2.3 `cake_grouped_mxfp8_quantize` Host-Free Shape-Based Program Selection (`#7322`, commit `5396dc3b7a95`)

In `flashinfer/quantization/fp8_quantization.py`, the host selects among three programs per dtype without reading `mask[B]`:

$$\text{Program}(B, M, K) = \begin{cases} \texttt{bulk-copy}, & \text{if } B \cdot M \cdot K \cdot \text{sizeof(dtype)} \le 4\;\text{MiB} \\ \texttt{row-tiled}, & \text{else if } B \cdot M \ge 16,384 \\ \texttt{row-2D}, & \text{otherwise} \end{cases}$$

- **`bulk-copy`**: Overlaps shared-memory tile prefetching with the `mask[B]` word load, yielding **1.07x–1.11x** speedup on small full-mask inputs.
- **`row-tiled`**: Each thread quantizes 2 rows of its group (`grid_y = ceil(M / (2 * max(1, 128 // blocks_per_row)))`), cutting launched CTAs in half (or more) under sparse MoE masks and achieving **1.41x–1.94x** speedup (`[32, 1024, 7168]` with `512 in group 0`: $34.724 \to 17.955\;\mu\text{s}$, **1.9340x** on GB300; $36.392 \to 18.802\;\mu\text{s}$, **1.9357x** on GB200).

### 2.4 Rubin `SM107` (`R200`) `cake_qsa_nvfp4` and `SM120` MiniMax-H3 Tuning (`#7284`, `#7334`, `#7333`)

- **Rubin `SM107` (`cc8552a3a015`)**: Adds `sm_107a` `tcgen05` decode and split-KV merge for NVIDIA R200 (212 SMs, `CTAS_PER_SM=1`), keeping split form on all hosts. Across `rows = [1..128]` (`24x2` heads, `2051` selected tokens), `num_splits` evaluates to `[33, 33, 26, 13, 6, 3, 1, 1]` and delivers **1.33x–2.76x** whole-call speedup over the FlashInfer reference (`rows=128` NHD: $151.44 \to 55.52\;\mu\text{s}$, **2.73x**).
- **SM120 MiniMax-H3 (`b8ed186df31b`, `53ecbde224f4`)**: Early PV hand-off and next-unit prefetch improve FP8 varlen attention by **1.014x–1.055x** on RTX 5090 (**1.030x–1.041x** on RTX PRO 6000), while 1-M-tile-per-swath persistent FP8 out-proj adds **1.006x–1.011x**.

### 2.5 TileLang Warp-Specialized TMA Producer Barrier and Sibling Pipeline Buffer Reconciler (`#3503`, `#2342`)

1. **Multi-Warp Pure-TMA Producer Barrier Fix (`fe6d319c78c0`)**: In `src/cuda/transform/producer_consumer_ws.cc`, `fwd_arrive_count` is now set to `1` only when `producer_is_single_warp = !has_async_producer && !has_producer_prefix && !loop_body_condition.defined()`, and to `producer_extent` (e.g., `128`) otherwise—preventing producer warps 1–3 from falling two phases behind and hanging (`#3470`).
2. **Sibling `T.Pipelined` Shared Buffer Reconciler (`fc0fbf0e50de`)**: `PipelineBufferReconciler` in `src/transform/inject_pipeline.cc` detects already-expanded shared buffers across sibling pipelines with asymmetric `num_stages` (e.g., `2` and `4`), prevents 4D double expansion `(4, 2, M, K)`, and reconciles all AST references to a single 3D `(4, M, K)` allocation (`#2309`).

---

## 3. Reproducible Experiment and Quantitative Summary

Verified via `experiments/flashinfer_muse_spark_pdl_dcp_alltoall_grouped_mxfp8_qsa_sm107_minimax_tilelang/run.py`:

| Sub-Track | Configuration / Shape | Baseline ($\mu\text{s}$) | Optimized ($\mu\text{s}$) | Speedup / Verification | Evidence Class |
|---|---|---:|---:|---|---|
| `cake_muse_spark_norm` PDL | B200 `gated_residual_rmsnorm` `M=8` | 2.272 | 1.506 | **1.509x** (GB300 up to **1.466x**) | `author-benchmark` + `own-offline` |
| `cake_dcp_alltoall` LL128 | GB200 CP2 `T=1` cold / warm | 5.584 / 5.456 | 4.192 / 4.288 | **1.33x** / **1.27x** | `author-benchmark` + `own-offline` |
| `cake_grouped_mxfp8_quantize` | `[32, 1024, 7168]` sparse (`512 in group 0`) | 34.724 (GB300) | 17.955 (GB300) | **1.934x** (GB200 **1.936x**) | `author-benchmark` + `own-offline` |
| `cake_qsa_nvfp4` Rubin SM107 | R200 (212 SMs) NHD `rows=128, topk=2051` | 151.44 | 55.52 | **2.73x** (read flush **2.76x**) | `author-benchmark` + `own-offline` |
| TileLang WS & Sibling Pipeline | `num_stages=(2, 4)` shared `(64, 32)` | 4D `[4,2,64,32]` crash | 3D `[4,64,32]` pass | Eliminates 2-phase hang & `ICHECK` | `implementation` + `own-offline` |
