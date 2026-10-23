# FlashInfer SM107a Hidden-4096 Fused mHC Quantized Outputs, Frost Single TVM-FFI Entry & TileLang Pipeline Transitive Dependencies

> **Language**: [中文](flashinfer-mhc-h4096-frost-tvm-ffi-tilelang-pipeline.md) | [English](flashinfer-mhc-h4096-frost-tvm-ffi-tilelang-pipeline.en.md)
> **Phase**: 65 (`priority: 74`) · **Track A**: GPU Kernels, Low-Bit Quantization & Compiler Passes
> **Verified Upstream Sources**: `flashinfer-ai/flashinfer@21ac136220` (`#7114`), `flashinfer-ai/flashinfer@c0e51ae29b` (`#7127`), `flashinfer-ai/flashinfer@b90521314b` (`#7037`), `tile-ai/tilelang@4b2032a6ce` (`#2822`), `tile-ai/tilelang@1bafce1a37` (`#3581`), `tile-ai/tilelang@d9a31ddd1c` (`#3579`)

---

## 1. Core Architectural Problem & Background

As Blackwell (`sm_100a` / `sm_103a` / `sm_107a`) inference kernels and next-generation DSL compilers (CuTe-DSL / TileLang) mature, end-to-end operator efficiency is increasingly governed by **epilogue quantization fusion**, **host-side FFI submission overhead**, **mid-batch tile granularity coverage**, and **soundness of compiler pipeline dependency analysis**:

1. **Redundant HBM Re-Reads in SM107a Hidden-4096 Fused mHC (`flashinfer@21ac136220`)**: Previously, `flashinfer.mega_mhc` on `sm_107a` (hidden size 4096) only shipped `sf_layout="bf16"`. Feeding an FP8 Group-128 or NVFP4 downstream layer required writing `y_bf16` (`[T, 4096]`) to HBM and launching a standalone vLLM quantization kernel (`per_token_group_quant_fp8` or `scaled_fp4_quant`) to read `8 KiB/token` back from HBM.
2. **Python/CuTe Multi-Launch FFI Overhead in Frost Low-Latency MoE (`flashinfer@c0e51ae29b`)**: On NVIDIA B300 at decode batch sizes $T \in \{1, 2, 4\}$, the GPU kernels for Frost MoE execute in just `10.33–26.65 µs`. However, warm Python calls reconstructed DLPack wrappers and pointer descriptors on every invocation and crossed the Python/CuTe FFI boundary four separate times (`pack_scale`, `gate_up`, `down_project`, `combine`), incurring `83.25–260.85 µs` of CPU submission latency.
3. **Missing Tile-128 in Cake StepFun Fused MoE MXFP8 (`flashinfer@b90521314b`)**: Cake's StepFun SwiGLU MoE (`H=4096, I=1536, E=64, top_k=8`) previously offered MXFP8 tiles `{8, 16, 32, 64, 256}` without `128`. At $T=1024$, the planner had to pick either `tile=64` (doubling CTA count and weight reloads) or `tile=256` (wasting tile capacity on sparsely populated experts).
4. **Dropped Transitive Copy Dependencies & Thread-Var Cancellation in TileLang (`tilelang@4b2032a6ce` & `1bafce1a37`)**:
   - In `pipeline_planning.cc`, when Stage 1's global-to-shared copy used an address loaded from Stage 0's shared memory buffer (`A[idx_smem[i]]`), tracking only direct data copies missed the transitive producer, dropping required TMA/`cp.async` waits.
   - Comparing two `BufferRegion`s without first expanding thread variables over their full domains allowed a shared `threadIdx.x` symbol to cancel algebraically between producer and consumer stages, hiding cross-thread RAW/WAR conflicts.
   - In Ascend PTO codegen (`PTOMmadDirectionAnalyzer`), merging identical predicates across $N$ mutable-memory guards unconditionally built `S || S`, causing $\mathcal{O}(2^N)$ AST blowup and >10-minute compile stalls (`tilelang@1bafce1a37`).

---

## 2. Source-Level Mechanics & Mathematical Formulation

### 2.1 FlashInfer Cake Mega-mHC Hidden-4096 Quantized Output Variants (`21ac136220`)

`flashinfer/experimental/deepgemm_mega_mhc/mega_mhc.py` registers two new `sm_107a` hidden-4096 programs (`cake_mega_mhc_a940c94296afad2b2497` and `cake_mega_mhc_1c61fed11caf93d1b762`), expanding `sf_layout` to three states `("bf16", "fp8_group128", "nvfp4")`:

- **Bitwise-Identical Base Outputs**: Both quantized variants preserve the four base outputs—`new_residual` (`BF16 [T, 4096]`), `y_bf16` (`BF16 [T, 4096]`), `new_post_mix` (`FP32`), and `new_comb_res_mix` (`FP32`)—**bitwise identical** to the `sf_layout="bf16"` program.
- **`sf_layout="fp8_group128"` Contract**: Computes per-128-column group scales in registers and stores `y_fp8` (`FP8-E4M3 [T, 4096]`) alongside `y_fp8_sf` (`FP32 [T, 32]`). With `fp8_sf_ue8m0=True` (default), scales are rounded to exact powers of two:
  $$s_{t,g}^{\text{UE8M0}} = 2^{\lceil \log_2(\max_{j \in g} |y_{t,j}| / 448.0) \rceil}, \quad g \in \{0, \dots, 31\}$$
  matching vLLM's `per_token_group_quant_fp8(y_bf16, 128, use_ue8m0=fp8_sf_ue8m0)` bit-for-bit.
- **`sf_layout="nvfp4"` Contract**: Given `fp4_global_scale`, writes packed E2M1 `y_fp4` (`uint8 [T, 2048]`) and linear-layout `y_fp4_sf` (`FP8-E4M3 [T, 256]`, one scale per 16 columns), matching vLLM's `scaled_fp4_quant(y_bf16, fp4_global_scale, is_sf_swizzled_layout=False)`.

### 2.2 Frost Low-Latency MoE Single TVM-FFI Entry (`c0e51ae29b`)

In `flashinfer/fused_moe/backends/frost_low_latency/pipeline.py` and `runtime.py`:
- `Pipeline.launch` is decorated with `@cute.jit` and compiled once per tactic via `cute.compile(..., options="--enable-tvm-ffi")` under cache key `"mx-tc-ffi-v1"`.
- Pointer recasting (`cute.recast_ptr`) and tensor view formation happen inside the compiled TVM-FFI host function rather than in Python via `from_dlpack` / `make_ptr`.
- The cache stores only executable code and immutable specialization metadata (`geometry, tile_k, pdl, wd, narrow_weights, finalize`); every call binds its current tensors and CUDA stream dynamically.

| Tactic (`T=1, H=2048, I=1408, E=64, top_k=6`, B300) | Public API Before → After (µs) | Public Speedup | Prepacked Runner Before → After (µs) | Prepacked Speedup | GPU Kernel Time (µs) |
|---|---:|---:|---:|---:|---:|
| `BF16 FMA` | `106.04 → 35.81` | **2.96×** | `83.25 → 14.64` | **5.69×** | `26.63 → 26.65` |
| `MXFP4 TC128` | `260.85 → 61.85` | **4.22×** | `217.82 → 24.90` | **8.75×** | `10.33 → 10.33` |
| `MXFP4 FMA` | `154.30 → 47.30` | **3.26×** | `117.98 → 14.33` | **8.23×** | `17.25 → 17.25` |
| `MXFP8 TC128` | `256.21 → 60.98` | **4.20×** | `213.17 → 25.74` | **8.28×** | `10.50 → 10.50` |
| `MXFP8 FMA` | `152.79 → 47.38` | **3.22×** | `116.25 → 15.13` | **7.68×** | `24.52 → 24.52` |
| `NVFP4 FMA` | `162.43 → 48.37` | **3.36×** | `125.88 → 16.02` | **7.86×** | `13.23 → 13.23` |

### 2.3 TileLang Transitive Pipeline Copy Dependencies & Thread-Domain Expansion (`4b2032a6ce` & `1bafce1a37`)

1. **Transitive Copy Closure (`pipeline_planning.cc`)**: `BufferRegionCollector::VisitExpr_(const BufferLoadNode *op)` preserves and restores `is_global_read_` when visiting index expressions so indirect loads `A[idx_smem[i]]` record `idx_smem` as an address dependency, and a unified backward worklist propagates transitive copy producers to retain required asynchronous copy `wait` statements.
2. **Thread-Domain Expansion (`CollectPipelineThreadDomains` & `PipelineRegionsMayConflict`)**: Each `BufferRegion` is independently expanded over the union of its materialized thread domains before intersecting intervals:
   $$\text{Conflict}(R_1, R_2) \iff \forall d,\ \text{Eval}(R_{1,d}, \mathcal{D}_{\text{thread}}) \cap \text{Eval}(R_{2,d}, \mathcal{D}_{\text{thread}}) \neq \emptyset$$
   preventing false algebraic cancellation of shared thread variables while keeping disjoint shared-memory slices pipelined.
3. **Ascend PTO Predicate Deduplication (`1bafce1a37`)**: Checking object identity and structural equality before constructing `S || S` eliminates exponential $\mathcal{O}(2^N)$ predicate growth across $N$ mutable guards.

---

## 3. Deterministic Verification (`experiments/flashinfer-mhc-h4096-frost-tvm-ffi-tilelang-pipeline`)

Running `python experiments/flashinfer-mhc-h4096-frost-tvm-ffi-tilelang-pipeline/run.py --check` validates all 8 architectural invariants:
- **Mega-mHC Hidden-4096 ($T=128$)**: Eliminates `1,048,576 B` (`1.0 MiB`) of redundant `y_bf16` HBM re-reads; `nvfp4` achieves a **3.5556×** payload compression ratio over BF16.
- **Frost TVM-FFI**: Collapses 4 FFI launches into 1, reaching **4.22×** public API speedup and **8.75×** prepacked runner speedup on B300.
- **StepFun MXFP8 Tile-128**: Switches $T=1024$ plan selection from `tile=64` to `tile=128`, delivering `1.047×` on GB300 and `1.071×` on B200.
- **TileLang**: Captures transitive copy producers `[1, 0]`, detects cross-thread conflict `[1, 63]`, preserves disjoint slice pipelining, and collapses 32-guard PTO predicate AST nodes from $2^{32}$ to `1`.
