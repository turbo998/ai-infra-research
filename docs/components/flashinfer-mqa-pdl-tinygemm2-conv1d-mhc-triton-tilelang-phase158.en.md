# FlashInfer SM100a MQA Parallel-Schedule, PDL QK-RoPE-Append Overlap, SM107a TinyGEMM2 Routing, Causal Conv1D `bwd_grid_mult`, mHC Graph-Capture Guard & Triton/TileLang Compiler Primitives (Phase 158)

> **中文版**: [flashinfer-mqa-pdl-tinygemm2-conv1d-mhc-triton-tilelang-phase158.md](flashinfer-mqa-pdl-tinygemm2-conv1d-mhc-triton-tilelang-phase158.md)
> **Question ID**: `q-flashinfer-mqa-pdl-tinygemm2-conv1d-mhc-triton-tilelang-phase158` (`priority: 167`)
> **Evidence ID**: `ev-flashinfer-mqa-pdl-tinygemm2-conv1d-mhc-triton-tilelang-2026-phase158`

---

## 1. Architectural Context & Core Findings

On NVIDIA Blackwell (`sm_100a` / `sm_103a`) and Vera Rubin (`sm_107a` / `R200`) GPUs, tail-wave SM contention, inter-kernel drain bubbles, and unplayed CUDA Graph capture allocations directly govern P99 latency and numerical correctness. Auditing 10 primary upstream commits merged between `2026-10-02` and `2026-10-31`, we establish five concrete architectural conclusions:

1. **SM100a Cake DeepGEMM Dense MQA Parallel-Schedule Bucket (`flashinfer@9d55883`, `#8113`)**: Routes `fp8:h64:partial:qle128:kvsplit` shapes up to $K \le 8192$ to `cake_deepgemm_dense_mqa_1974e6db96c173903467`, achieving **1.0506x–1.0634x** speedup on B200 (median **1.0563x**, `4.83 -> 4.54 us` at `Q=3/7, prefix=4100`) with `106/106` bitwise equality.
2. **SM100a/SM103a Opt-In PDL for Fused QK RMSNorm + RoPE + KV Append (`flashinfer@b194ced`, `#8112`)**: Adds `enable_pdl: bool = False` using `cudaGridDependencySynchronize()` and `cudaTriggerProgrammaticLaunchCompletion()` to overlap upstream cuBLAS QKV projection GEMM tail waves with preamble metadata setup, delivering **1.1395x–1.1499x** FP8 geomean speedup (up to **1.3558x**, `6.768 -> 4.992 us` on GB200) and **1.1720x–1.1886x** BF16 geomean speedup (up to **1.3358x**, `6.520 -> 4.881 us` on GB300).
3. **SM107a (`R200`) TinyGEMM2 `tcgen05` 9–16 Row Routing Windows (`flashinfer@35be896`, `#8110`)**: Adds unbiased `sm_107a` dispatch rules for $M \in [9, 16], N \in [4097, 16384], K \in [1024, 16384]$, selecting `PACKED_M64_N16` when $N > 6784 \land K > 12288$ and `PACKED` otherwise.
4. **Cake Causal Conv1D Deterministic Backward `bwd_grid_mult` (`flashinfer@d29e16a`, `#8104`)**: Splits backward token-axis work by `k in {1, 2, 3, 4}` (`grid.x = G * k`) while retaining deterministic `dw` slot ownership (`blockIdx.x % G`), achieving **1.43x** speedup on R200 (`0.1589 -> 0.1114 ms` at `k=2`), **1.28x** on GB300 (`0.7072 -> 0.5526 ms` at `k=4`), and **1.27x** on B200 (`0.7813 -> 0.6175 ms` at `k=4`) under side-stream contention.
5. **Cake mHC CUDA Graph Capture Guard & Triton/TileLang Type Lowering (`flashinfer@fc6d231`, `triton@e6aa06c`, `triton@568e9d3`, `tilelang@b95ee4f`, `tilelang@c62ebe9`, `tilelang@256791f`)**: Forbids lazy default workspace creation (`torch.zeros`) during stream capture (`ValueError`), preventing unplayed capture zero-fills from leaving dirty ticket words in `_DEFAULT_WORKSPACES`.

---

## 2. Primary Upstream Evidence Matrix

| Commit | Path / Module | Target Hardware | Verified Mechanism & Metric |
| :--- | :--- | :--- | :--- |
| `flashinfer-ai/flashinfer@9d55883` (`#8113`) | `flashinfer/jit/cake_deepgemm_dense_mqa.py` | B200 (`sm_100a`, 148 SMs) | Parallel-schedule logits program up to $K \le 8192$; **1.0563x** median speedup (`4.83 -> 4.54 us`), `106/106` bitwise equal |
| `flashinfer-ai/flashinfer@b194ced` (`#8112`) | `csrc/cake_fused_qk_rope_append_kernel.cu` | B200 / GB200 / B300 / GB300 (`sm_100a`/`sm_103a`) | `enable_pdl=True` overlaps QKV GEMM tail: FP8 geomean **1.1395x–1.1499x** (peak **1.3558x**), BF16 geomean **1.1720x–1.1886x** (peak **1.3358x**) |
| `flashinfer-ai/flashinfer@35be896` (`#8110`) | `csrc/cake_tinygemm2_tcgen05.cu` | R200 (`sm_107a`) | Routes $9 \le M \le 16$ unbiased GEMMs to `PACKED` and `PACKED_M64_N16` ($N > 6784, K > 12288$) |
| `flashinfer-ai/flashinfer@d29e16a` (`#8104`) | `flashinfer/cake/causal_conv1d_det.py` | B200 / GB300 / R200 | `bwd_grid_mult` ($k \in \{1,2,3,4\}$) cuts 2nd-wave tail penalty: **1.43x** on R200, **1.28x** on GB300, **1.27x** on B200 |
| `flashinfer-ai/flashinfer@fc6d231` (`#8100`) | `flashinfer/cake/mhc.py` | CUDA Graph Stream | `_get_default_workspace()` raises `ValueError` on capture cache miss (`550/550` tests pass) |
| `triton-lang/triton@e6aa06c` (`#12456`) | `python/triton_kernels/triton_kernels/matmul.py` | Triton JIT Backend | Removes explicit `enable_fp_fusion` overrides from `matmul`, `reduce`, and `topk` |
| `triton-lang/triton@568e9d3` (`#12403`) | `python/triton/runtime/jit.py` | Triton AST Frontend | Adds `bool` (`tl.int1`) parameter annotation support lowering to `arith.select` |
| `tile-ai/tilelang@b95ee4f` (`#3361`) | `src/target/rt_mod_hip.cc`, `make_packed_api.cc` | TileLang TVM FFI | Casts `dtype.is_bfloat16()` scalar parameters via `Float(64)` in `MakePackedAPI` |
| `tile-ai/tilelang@c62ebe9` (`#3517`) | `testing/python/transform/*.py` | TileLang CUDA Test | Uses `tilelang_cuda_without_compile` for codegen inspection tests |
| `tile-ai/tilelang@256791f` (`#3463`) | `testing/python/language/test_tilelang_language_ldst.py` | TileLang CUDA LDGSTG | Gates nested `LDG128`/`STG128` tests on `is_cuda_FullySupported()` |

---

## 3. Reproducible Verification

```powershell
python experiments\flashinfer_mqa_pdl_tinygemm2_conv1d_mhc_triton_tilelang_phase158\run.py --check
```
