# TransformerEngine Bounded Symmetric All-Reduce Cache, NVFP4 `torch.compile` Metadata & DeepGEMM Mega-MHC CUDA Graph Epoch Barrier

> **中文版本**: [te-nvfp4-compile-symm-ar-deepgemm-mhc-epoch.md](te-nvfp4-compile-symm-ar-deepgemm-mhc-epoch.md) | **Back to Index**: [README.en.md](../../README.en.md)

## 1. Problem Statement & Architectural Context

In Blackwell (`SM100`/`SM120`) low-precision training and tensor-parallel (TP) inference/training pipelines, kernel-level optimizations face three subtle failure modes caused by **dynamic sequence-length scratch memory bloat**, **metadata drift between `torch.compile` / Dynamo `FakeTensor` tracing and C++ kernels**, and **hardware `%gridid` reuse across CUDA Graph replays**:

1. **Symmetric All-Reduce Scratch Cache Bloat vs. CUDA Graph Pointer Stability (`NVIDIA/TransformerEngine` `#3709`)**:
   - When `symmetric_all_reduce` is called on an input tensor that is not already allocated in symmetric memory, the previous implementation cached a dedicated symmetric buffer keyed by exact `tensor_numel`. Across variable row counts (`rows = 1..32, cols = 256`), 32 distinct buffers remained permanently allocated (`135,168` elements = **`16.50x`** the largest tensor of `8,192` elements).
   - Naively replacing the cache with a single growable buffer that frees the old allocation upon expansion invalidates device pointers already baked into captured `torch.cuda.CUDAGraph` instances.
2. **NVFP4 `torch.compile` / Dynamo `FakeTensor` Swizzled-Scale & SM120 RHT Drift (`NVIDIA/TransformerEngine` `#3628`, `#3669`)**:
   - NVFP4 quantization kernels emit GEMM-swizzled scales directly when shape and architecture alignment rules are met. Previously, `BiasBwdArgs.backward_compute_fake` omitted `with_gemm_swizzled_scales` on `TensorSpec`, `get_columnwise_shape` did not flatten 3D/4D shapes to 2D, `_amax_columnwise` remained `(1,)` instead of `(shape[-1],)` under `row_scaled_nvfp4=True`, and `NVFP4TensorStorage.size()` emitted a Python warning during Dynamo compilation—breaking `torch.compile(..., fullgraph=True)`.
   - In addition, `quantize_transpose` in `quantize_transpose_nvfp4.cuh` passed `nullptr` instead of `rht_matrix` to `quantize_transpose_impl` on `SM120`/`SM121` (`#3669`).
3. **DeepGEMM `sm100_mega_mhc` Split-Barrier Stale Reads Under CUDA Graph Replay (`deepseek-ai/DeepGEMM` `#478`)**:
   - `sm100_mega_mhc` synchronizes K-split producer CTAs and Mix/Norm consumer CTAs via global-memory `SplitBarrier` lines. Previously, barrier states were tagged using PTX `%gridid` (`encode_state(grid_idx, num_arrived_splits)`). However, **a CUDA Graph kernel node retains the exact same `%gridid` across every replay**. On replay $r+1$, if a consumer CTA reached `SplitBarrier::wait` before the producer init CTA executed `SplitBarrier::init`, it matched replay $r$'s completed state `encode_state(grid_idx, kNumSplits)` and read stale split partials.

---

## 2. First-Hand Code Evidence & Implementation Mechanics

### 2.1 TransformerEngine Quarter-Power-of-Two Symmetric Scratch Bucketing (`fd7d03b1fa`)

In `transformer_engine/pytorch/distributed.py` (blob `e90b17e4d9`, lines `2092–2102`), `symmetric_all_reduce` rounds `tensor_numel` up to one of four capacities per power of two:

```python
numel = max(1, tensor_numel)
step = 1 << max(0, numel.bit_length() - 3)
capacity = -(-numel // step) * step
msg = get_symmetric_memory_tensor(capacity, tensor_dtype, tensor_device, tp_group)
msg = msg[:tensor_numel]
```

- **Per-Allocation Padding Bound ($\le 25\%$)**: For $2^{k-1} \le \text{numel} < 2^k$, `numel.bit_length() == k` and $\text{step} = 2^{k-3} = 0.25 \times 2^{k-1}$. Thus, worst-case padding never exceeds $25\%$ (peaking at **`17.65%`** at `rows = 17` in the `1..32` sweep).
- **Bounded Total Retention & CUDA Graph Safety**: Across `rows = 1..32` (`cols = 256`), 32 tensor sizes map into 16 bucket capacities totaling **`49,152` elements** (**`6.0x`** max tensor, well below the `< 8x` test bound and **`63.64%` lower** than exact-size caching). Older buckets remain alive when a larger request (`rows = 128`, `capacity = 32,768`) arrives, and `rows = 23` and `rows = 24` share bucket `6,144`, allowing captured graphs to replay safely in any order.

### 2.2 NVFP4 `torch.compile` / Dynamo Metadata Alignment & SM120 RHT Forwarding (`f0b1096128`, `4a3a8a8e85`)

1. **Shared C++ / Dynamo Swizzled-Scale Alignment Helper**:
   - `transformer_engine/pytorch/csrc/quantizer.cpp` (blob `ba0410bcac`, lines `1937–1993`) introduces `NVFP4Quantizer::gemm_swizzled_scales_alignment` (bound via PyBind11 as `tex.nvfp4_gemm_swizzled_scales_alignment`), returning `(64, 128)` (or `(128, 128)` for grouped kernels) when `with_rht=True` on `SM100..SM110`, `None` on `SM120/SM121` (which uses compact 1D TMA scales plus post-quantize swizzle), and `(128, 128)` for plain 2D quantization when `!row_scaled_nvfp4 && !nvfp4_use_4over6`.
   - `transformer_engine/pytorch/dynamo/fake.py` (blob `5f856d6221`, lines `19–76`) queries this helper under `@torch.compiler.assume_constant_result` and computes `rows % row_alignment == 0 and cols % col_alignment == 0` so Dynamo fake propagation sets `spec.with_gemm_swizzled_scales` identically to eager C++ kernels.
2. **Columnwise 2D Flattening, `_amax_columnwise` Shape, and RHT Matrix Forwarding**:
   - `nvfp4_tensor.py` (blob `938d9a77d4`) flattens `get_columnwise_shape(shape)` to `(shape[-1], math.prod(shape[:-1]))` and sets `_amax_columnwise` to `(shape[-1],)` when `row_scaled_nvfp4=True`.
   - `quantize_transpose_nvfp4.cuh` (blob `f83564b35f`, line `2065`) forwards `rht_matrix` instead of `nullptr` in `quantize_transpose_impl`.

### 2.3 DeepGEMM `sm100_mega_mhc` In-Line `Epoch` + `exit_count` Split-Barrier State Machine (`5081f8513a`)

In `deep_gemm/include/deep_gemm/layout/mega_mhc.cuh` (blob `f8a103dbac`) and `scheduler/mega_mhc.cuh` (blob `cd9afb6fab`):

- **Zero Buffer Growth via First-Line Tail Packing**: Each split barrier occupies a 128-byte line (`kSplitBarrierLineBytes = 128`) whose state uses only the first 8 bytes. `kSplitBarrierEpochOffset = 64` (`uint64_t`) and `kSplitBarrierExitCountOffset = 72` (`uint32_t`) place the execution epoch and CTA exit counter in the unused second half of the very first barrier line (`64..76B`).
- **Unsigned Wrap-Around in `wait_init`**: `encode_state(epoch, s) = (epoch + 1) * 128 + s`. Non-initializing producer split CTAs spin on `while (ld_acq_gpu(ptr) - current_barrier_base >= kNumSplits)`. Before the init CTA writes `current_barrier_base`, unsigned 64-bit subtraction wraps around to a huge integer $\ge \text{kNumSplits}$; once initialized, any arrival count $0 \le s < \text{kNumSplits}$ immediately passes.
- **Last-CTA `Epoch::retire`**: At kernel exit (`impls/sm100_mega_mhc.cuh:697-699`), `threadIdx.x == 0` of each CTA atomically increments `exit_count`; the last CTA (`atomic_add_rel == num_ctas - 1`) issues `__threadfence()`, resets `*exit_count_ptr = 0`, and increments `*epoch_ptr` via `st_rel`.

---

## 3. Deterministic Offline Experiment & Quantitative Findings

Verified via [`experiments/te_nvfp4_symm_ar_deepgemm_mhc_epoch/run.py`](../../experiments/te_nvfp4_symm_ar_deepgemm_mhc_epoch/run.py) (`own-offline`):

| Evaluation Dimension | Pre-Fix Baseline | Verified Implementation | Architectural Impact |
|---|---|---|---|
| **TE Symmetric All-Reduce Retention (`rows=1..32, cols=256`)** | 32 buffers / `135,168` numel (`16.50x` max tensor) | 16 buckets / **`49,152` numel** (**`6.00x`** max tensor) | **`63.64%`** less retained symmetric memory; max padding **`17.65%`** ($\le 25\%$) |
| **CUDA Graph Replay (`rows=23,24` then `128` growth)** | Single-buffer realloc invalidates **`2/2`** graphs | Retains buckets; `23/24` share `6,144` (**`2/2` graphs valid**) | Preserves CUDA Graph pointer stability with bounded memory |
| **TE NVFP4 Dynamo `FakeTensor` Alignment (4 cases)** | **`2/4`** swizzle mismatches; 3D colwise unflattened | **`0/4`** mismatches; 2D colwise `[256, 64]`; `rht_matrix` forwarded | Full `torch.compile(fullgraph=True)` parity with eager C++ |
| **DeepGEMM `sm100_mega_mhc` 8 CUDA Graph Replays** | Constant `%gridid=42` causes **`7/7` stale reads** | In-line `Epoch` (`64B`/`72B`) yields **`0/7` stale reads** (`epoch=8`) | Eliminates replay split-barrier races with zero extra workspace |

---

## 4. Architect Takeaways & Falsifiable Questions

1. **Bucket-and-Retain for Graph-Captured Comm Buffers**: Symmetric memory buffers referenced by captured CUDA Graphs must never be freed on capacity growth; quarter-power-of-two bucketing (`step = 1 << max(0, bit_length - 3)`) bounds per-allocation slack to $\le 25\%$ and total retention to $< 8\times$ the largest tensor.
2. **Never Use `%gridid` as a Replay Epoch**: Because CUDA Graph kernel nodes preserve `%gridid` across replays, persistent kernels with global barriers must maintain an explicit global-memory `Epoch` advanced by the last retiring CTA (`atomic_add_rel(exit_count_ptr, 1) == gridDim.x - 1`) after `cudaGridDependencySynchronize`.
