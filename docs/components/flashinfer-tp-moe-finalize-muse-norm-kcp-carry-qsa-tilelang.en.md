# FlashInfer Cake TP MoE Finalize + RMSNorm, Muse Spark Gated RMSNorm, KCP Carry Plane, and TileLang Ascend DMA Boundary Evolution

> **Document Status**: 2026-10-24 · Phase 62 (`priority: 71`, Track A)  
> **Evidence Level**: `implementation` + `author-benchmark` (upstream PR measurements) + `own-offline` (`experiments/flashinfer-tp-moe-finalize-muse-norm-kcp-carry-qsa-tilelang/run.py` deterministic Python stdlib model)  
> **Question ID**: `q-flashinfer-tp-moe-finalize-muse-norm-kcp-carry-qsa-tilelang`

---

## 1. Architectural Context & Problem Statement

On NVIDIA Blackwell (`sm_100a` B200 / `sm_103a` GB300) and Rubin (`sm_107a` R200) GPUs, tensor-parallel MoE finalize + RMSNorm collectives, gated residual RMSNorm kernels, context-parallel linear attention backward passes, and sparse NVFP4 decode kernels are converging on **self-contained AOT/JIT Cake kernel packages, symmetric-memory Lamport rings, and strict cross-plan storage guards**. Simultaneously, on Huawei Ascend NPUs, TileLang hardened its compiler passes around `INT32_MAX` region boundaries, strided view `elem_offset` preservation, and periodic buffer lifetimes.

Drawing on 8 upstream commits merged on 2026-10-23, this document analyzes:

1. **FlashInfer `cake_tp_moe_finalize_rmsnorm` (`0e832d0636`, PR `#7120`)**: How 5 symmetric-memory Lamport ring program kinds (`eo`, `onehop`, `rsag`, `token1`, `tokown`) route by token count to deliver up to **1.334x** speedup on GB300 TP4 and **1.244x** on B200 TP8.
2. **FlashInfer `cake_muse_spark_norm` (`787132c835`, PR `#7119`)**: How three fused gated RMSNorm kernels (`gated_residual_rmsnorm`, `gated_residual_next_rmsnorm`, `rmsnorm_sigmoid_gate`) achieve up to **1.818x** (B200) and **1.702x** (GB300) speedups over CuTe DSL baselines.
3. **FlashInfer `cake_kda` KCP Carry Plane Reuse (`ddc9524d9f`, PR `#7117`)**: Retaining the forward carry plane when `P > 1` to eliminate backward segment offset re-walks and carry merges, along with its exact byte-budget formula.
4. **FlashInfer `deepgemm_mega_mhc` (`85e886c875`, PR `#7113`) & `cake_qsa_nvfp4` (`34d28b3232`, PR `#7103`)**: SM107 3D TMA `L2::cache_hint` evict-first residual streaming, `_claim_launch_epochs` whole-storage and matching-grid guards, and H100 NVFP4 sparse paged GQA decode speedups up to **1.127x**.
5. **TileLang Ascend Compiler Fixes (`c412c2d4b1` `#3625`, `f5c6b44cfe` `#3627`, `ed06ed1f01` `#3606`)**: `Int64Promoter` before overlap analysis, preserving `phys_buf.elem_offset` in DMA copies/strided slices, and rejecting incomplete child-only buffer lifetime summaries.

---

## 2. Source-Level Mechanics & Mathematical Modeling

### 2.1 Cake TP MoE Finalize + RMSNorm: 5 Lamport Ring Program Kinds (`flashinfer@0e832d0636`)

In `flashinfer/cake_tp_moe_finalize_rmsnorm_backend/cake_backend.py`, `flashinfer.comm.tp_moe_finalize_rmsnorm(..., backend="cake")` implements a single-stream, CUDA-Graph-capturable collective + RMSNorm for `HIDDEN = 4096`, `TOPK = 8`, and `ALIGNMENT = 256` over `torch.distributed._symmetric_memory` and NVLink multicast:

| Program Kind (`kind`) | Launch Granularity | Token Window (`T`) | Communication & Reduction Pattern |
|---|---|---|---|
| `eo` | 1 Cluster per token | Micro-batch (`T = 1..8`) | Early-out low-latency ring exchange |
| `onehop` | 1 Cluster per token | Small batch (`T = 8..32`) | Single-hop NVLink symmetric memory write |
| `rsag` | 1 CTA per token | Medium-small (`T = 32..64`) | Fused Reduce-Scatter + All-Gather |
| `token1` | 1 Cluster per `world_size` token group | Medium (`T = 64..152`) | World-grouped token aggregation |
| `tokown` | Co-resident persistent grid (`TOKOWN_MAX_CTAS`) | Large (`T >= 145/153`) | Persistent ring capped by min co-resident CTA capacity across ranks |

- **Routing Guard**: The package ships and routes `sm_100a` (`TP2, TP8`) and `sm_103a` (`TP2, TP4`), explicitly refusing unshipped combinations (`sm_100a TP4` and `sm_103a TP8`).
- **Author-Benchmark Results (`author-benchmark`, CUDA 13.0 vs. CuTe DSL PR `#6544`)**:
  - **GB300 (`sm_103a`) TP4**: `T=1` improves from `9.36 µs` to `8.25 µs` (**1.135x**); `T=153` improves from `39.55 µs` to `30.36 µs` (**1.303x**); `T=256` improves from `50.55 µs` to `37.91 µs` (**1.334x**).
  - **B200 (`sm_100a`) TP8**: `T=8` improves from `16.78 µs` to `13.67 µs` (**1.228x**); `T=145` improves from `51.66 µs` to `42.04 µs` (**1.229x**); `T=256` improves from `65.96 µs` to `53.04 µs` (**1.244x**).

### 2.2 Muse Spark Gated RMSNorm Kernels (`flashinfer@787132c835`)

`flashinfer/cake_muse_spark_norm_backend` provides Cake kernels for three fused RMSNorm operations (`H = 8192`):

1. **`gated_residual_rmsnorm`**: Takes BF16 `branch` and FP32 `residual / alpha / beta / out` in `mode 0` (no pre-norm weight) and `mode 1` (weight before norm), shipping both non-PDL and PDL builds. Achieves **1.234x** (`1.35 -> 1.09 µs`) at `M=1, mode 0` on GB300 and **1.142x** (`232.32 -> 203.36 µs`) at `M=16384, mode 1` on B200.
2. **`gated_residual_next_rmsnorm`**: Produces FP32 `residual_out` and BF16 `ffn_input`:
   - **B200 (`sm_100a`) V1**: **1.602x** at `M=1024` (`31.58 -> 19.71 µs`), **1.747x** at `M=4096` (`113.67 -> 65.06 µs`), and **1.818x** at `M=16384` (`438.82 -> 241.37 µs`).
   - **GB300 (`sm_103a`) V1**: **1.617x** at `M=1024` (`29.05 -> 17.96 µs`) and **1.702x** at `M=16384` (`403.39 -> 237.02 µs`).
3. **`rmsnorm_sigmoid_gate`**: Normalizes per-`(token, head)` rows of 64 BF16 elements in place across `heads ∈ {16, 32, 128}`, reaching **1.375x** (`2.42 -> 1.76 µs`) at `H=16, T=17` on GB300.

### 2.3 KCP Context-Parallel Saved Carry Plane (`flashinfer@ddc9524d9f`)

In `flashinfer/experimental/cake_kda_train/cake_kcp_plan.py`, when a rank splits its fragments into `P > 1` segments, `SAVED_CARRY_DEFAULT = 1` retains the forward pass's entire carry plane in `KcpSaved.carry`. The saved-context backward pass binds this plane read-only instead of re-walking every segment (`K1 _offset`) and re-merging carries, producing bitwise-identical gradients.

The exact per-layer, per-rank byte addition in `kcp_saved_bytes` is:

$$\text{Bytes}_{\text{carry\_plane}} = \mathbb{I}[\text{saved\_carry} \land P > 1] \cdot \Big(n_{\text{frag}} \cdot (P - 1) + \max(n_{\text{in}}, n_{\text{out}})\Big) \cdot s, \quad s = H \times 128 \times 128 \times 4\text{ B}$$

For $H = 64$, each state slot $s = 4\text{ MiB}$. On `Q4_CP16_contiguous` ($n_{\text{frag}}=4, P=5, n_{\text{in}}=n_{\text{out}}=2$), retaining 18 slots (**72.0 MiB**) reduces GB300 `B3` (forward + checkpoint replay + backward) latency from `9.788 ms` to `9.423 ms` (**1.0387x** speedup).

### 2.4 MegaMHC `launch_epochs` Guard (`85e886c875`) & QSA NVFP4 (`34d28b3232`)

- **MegaMHC (`#7113`)**: Replaces 3D TMA loads on SM107 with `cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint` using a token-count-dependent evict-first `io_policy`, and enforces in `_claim_launch_epochs` that `launch_epochs` owns its entire storage (`storage_offset() == 0` and `storage.nbytes() == nbytes`) and that all live plans sharing it launch the exact same CTA grid size.
- **Cake QSA NVFP4 (`#7103`)**: Loads K first, restores `evict_first`, and uses single-instruction `1-prmt` dequantization for NVFP4 (`E2M1` with `E4M3` scales) sparse paged GQA decode on H100 (`sm_90a`), achieving **1.127x** speedup at 32 rows in both `NHD` (`36.42 -> 32.32 µs`) and `HND` (`38.83 -> 34.45 µs`).

### 2.5 TileLang Ascend Compiler Invariants (`tilelang@c412c2d4b1`, `f5c6b44cfe`, `ed06ed1f01`)

1. **Int64 Region Bound Promotion (`#3625`)**: Promotes region bounds via `Int64Promoter` before `RegionsMayConflict` overlap checks so half-open endpoints at `INT32_MAX` (`2^31 - 1`) do not overflow `int32` and drop `MTE3_MTE2` DMA synchronization barriers.
2. **Buffer Offset Preservation in DMA & Strided Slices (`#3627`)**: Initializes strided slice offsets to `phys_buf->elem_offset + sum(min_i * stride_i)` instead of `0`.
3. **Complete Periodic Buffer Lifetimes (`#3606`)**: Rejects child-only lifetime summaries in `BufferAliasAnalyzerImpl` whenever a non-control or single-trip sibling access exists in the parent scope, preventing scratch buffers from overwriting preloaded values across pipelined loop iterations.

---

## 3. Deterministic Verification Summary (`own-offline`)

Validated via `python experiments/flashinfer-tp-moe-finalize-muse-norm-kcp-carry-qsa-tilelang/run.py --check`:

| Verification Dimension | Metric / Invariant | Result |
|---|---|---|
| Cake TP MoE Finalize (`#7120`) | GB300 (`sm_103a`) TP4 `T=256` speedup | **1.334x** (`50.55 µs -> 37.91 µs`, `tokown`) |
| Cake TP MoE Finalize (`#7120`) | B200 (`sm_100a`) TP8 `T=256` speedup | **1.244x** (`65.96 µs -> 53.04 µs`, `tokown`) |
| Muse Spark Norm (`#7119`) | B200 / GB300 `gated_residual_next_rmsnorm` V1 `M=16384` | **1.818x** (`438.82 -> 241.37 µs`) / **1.702x** (`403.39 -> 237.02 µs`) |
| KCP Carry Plane (`#7117`) | `Q4_CP16_contiguous` ($P=5$) memory & B3 speedup | **72.0 MiB** (`18` slots $\times 4\text{ MiB}$), **1.0387x** (`9.788 -> 9.423 ms`) |
| MegaMHC Epoch Guard (`#7113`) | Sliced view (`offset=8`) / Mismatched grid (`64 vs 148`) | Both rejected (`accepted = False`); matching grid (`64 == 64`) accepted |
| QSA NVFP4 Decode (`#7103`) | H100 `rows=32` (`NHD` / `HND`) speedup | **1.127x** (`36.42 -> 32.32 µs` / `38.83 -> 34.45 µs`) |
| TileLang Ascend (`#3625/#3627/#3606`) | `INT32_MAX+1` promotion, `elem_offset=512`, sibling lifetime guard | `2147483648 > 0`, `offset = 776`, `rejected = True` |

---

## 4. Architect Guidance

1. **Blackwell TP MoE Collective Selection**: Enable `backend="cake"` on GB300 TP4 and B200 TP8 for `H=4096, topk=8` MoE models; keep `backend="auto"` on GB300 TP8 or B200 TP4 so unsupported routes fall back cleanly to CuTe DSL.
2. **MegaMHC `launch_epochs` Isolation**: Never slice a shared tensor for `launch_epochs` or share it across token buckets that compile to different CTA grid counts (`num_launch_sms`).
3. **KCP Memory Budgeting**: Reserve `12–80 MiB` per layer per rank when `P > 1` in context-parallel KDA training to leverage carry-plane reuse in backward passes.
