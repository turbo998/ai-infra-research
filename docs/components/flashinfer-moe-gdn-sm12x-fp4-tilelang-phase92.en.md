# Phase 92 Track 1: FlashInfer MoE Activation Overlap, SM12x/SM100 GDN Matrix Pruning, NVFP4 Padding Zeroing & TileLang Compiler Verification

## Executive Summary

This document covers 10 first-hand upstream commits across `flashinfer-ai/flashinfer` and `tile-ai/tilelang` (`priority: 101`):

1. **`flashinfer-ai/flashinfer@215f83cb27` (`#7368`)**: Overlaps Cutlass MoE grouped activation staging on Ampere (`tile_k == 128` using 4-warp 128-bit `cp.async` guarded by a 16-byte alignment check `((ptr | (k * sizeof(T))) & 15) == 0` with scalar fallback) and Hopper (4-warp activation gather; `BF16`/`FP16` issue weight TMA loads before gathering activations so weight TMA overlaps with indexed activation gather).
2. **`flashinfer-ai/flashinfer@801fcd0a52` (`#6371`)**: Skips 12 all-zero `M16xN8xK16` MMA tiles below the main diagonal (`16 * k > 8 * n + 7`) in SM12x GDN prefill `64x64` upper-triangular `NewV`/`O2` multiplications, reducing MMA tile count from `32` to `20` (`37.5%` reduction).
3. **`flashinfer-ai/flashinfer@55b2e33697` (`#5492`)**: Preserves `state_stride_divisibility` (largest power-of-two `<= DK=128` dividing `slot`, `head`, and `V` strides) via `cute.assume(..., divby=state_stride_divisibility)` in SM100/SM103 indexed GDN prefill, restoring 128-bit vectorized loads and eliminating up to a `3.2x` slowdown relative to packed state.
4. **`flashinfer-ai/flashinfer@f45e1d6fdd` (`#6337`)**: Relaxes GDN decode gate and parameter alignment checks (`a`, `b`, `A_log`, `dt_bias`) from `16B`/`32B` to element size (`elem_bytes`), allowing zero-copy `gates.chunk(2, dim=-1)` views when `HV % 8 != 0` (such as `HV = 6`).
5. **`flashinfer-ai/flashinfer@063604bfc1` (`#7433`)**: Preserves all `7` positive E4M3 subnormal scale factors (`< 1/64`, or `< 2.0` after multiplying by `128`) in SM120 `FP16` W4A16 `_process_nvfp4_packed_scales` when `a_dtype == torch.float16`.
6. **`flashinfer-ai/flashinfer@966dcfd6b6` (`#7430`)**: Inserts `fence.proxy.async.shared::cta` before SM120 `cake_nvfp4_attention` TMA stores and barrier handoffs so generic-proxy shared-memory writes are visible to the asynchronous TMA engine.
7. **`flashinfer-ai/flashinfer@b2b2bfe892` (`#7008`)**: Zeroes out-of-bounds `128x4` swizzled scale-factor padding (`padded_M = ((M + 127) // 128) * 128`, `padded_K_blocks = ((K // block_size + 3) // 4) * 4`) inside `rmsnorm_fp4quant` and `add_rmsnorm_fp4quant` between `griddepcontrol.wait` and `griddepcontrol.launch_dependents`.
8. **`flashinfer-ai/flashinfer@90ac2eeb7e` (`#7050`)**: Stages full rectangular cuDNN page tables (`len(spans) == batch_size == rows` and uniform `width`) via a single `out[:, :width].copy_(indices[s0:sN].view(batch_size, width))` while preserving CUDA Graph pre-allocated buffer pointers.
9. **`tile-ai/tilelang@2573d47079` (`#3307`)**: Preserves signed `T1` (`int64`) in `cuda::atomic_ref<T1>` for `AtomicMax`/`AtomicMin` instead of casting to unsigned `NT1` (`unsigned long long`), and preserves destination types (`int64`, `bf16`, `fp16`) in `AtomicStore`.
10. **`tile-ai/tilelang@cb26ee0faf` (`#3345`)**: Rejects non-unit loop steps (`!op->step.has_value() || is_one(op->step.value())`) in `VectorizeLoop` instead of silently dropping `step` and visiting every index in `[0, extent)`.

---

## 1. Upstream Commit Analysis

### 1.1 Cutlass MoE Grouped Activation Overlap & SM12x/SM100 GDN Kernel Optimizations

In `flashinfer-ai/flashinfer@215f83cb27` (`#7368`), Cutlass MoE grouped GEMM activation staging is parallelized across 4 warps:
- **Ampere (`SM80`)**: When `tile_k == 128`, all 4 warps stage grouped activations via 128-bit `cp.async` whenever `((reinterpret_cast<uintptr_t>(src_ptr) | (k * sizeof(ElementA))) & 15) == 0`, falling back to scalar loads on unaligned rows.
- **Hopper (`SM90`)**: Activation gathering expands from 1 warp to 4 warps. For `BF16`/`FP16`, weight TMA loads are issued before gathering activations so HBM-to-SMEM weight transfers overlap with indexed activation gathering.

For Gated Delta Rule (GDN) attention kernels:
- **`801fcd0a52` (`#6371`)**: In SM12x GDN prefill, the `64x64` right operand of `NewV` and `O2` is upper-triangular. Across `k in [0, 3]` and `n in [0, 7]` (`32` `M16xN8xK16` tiles), tiles satisfying `16 * k > 8 * n + 7` lie strictly below the main diagonal and contain only zeros. Skipping those `2 + 4 + 6 = 12` tiles reduces MMA count by `37.5%`.
- **`55b2e33697` (`#5492`)**: In SM100/SM103 indexed GDN prefill (`state_indices` provided), passing `state_stride_divisibility` to `cute.assume()` restores 128-bit vectorized loads and eliminates up to a `3.2x` regression compared to packed `state`.
- **`f45e1d6fdd` (`#6337`)**: Relaxing GDN decode gate pointer alignment to `elem_bytes` allows non-16B-aligned `gates.chunk(2, dim=-1)` views when `HV % 8 != 0` (e.g., `HV = 6`).

### 1.2 SM120 NVFP4 Subnormals, Proxy Fence, PDL Padding Zeroing & TileLang Guards

- **SM120 FP16 W4A16 E4M3 Subnormals (`063604bfc1`, `#7433`)**: E4M3 has 7 positive subnormal values ($m \times 2^{-9}$ for $m \in \{1..7\}$, all $< 1/64$). When `a_dtype == torch.float16`, `_process_nvfp4_packed_scales` now preserves all 7 subnormal scale factors instead of flushing values whose `* 128` product is `< 2.0`.
- **SM120 NVFP4 Attention Proxy Fence (`966dcfd6b6`, `#7430`)**: Inserts `fence.proxy.async.shared::cta` prior to TMA stores and barrier synchronization in `cake_nvfp4_attention`.
- **`rmsnorm_fp4quant` Swizzled Padding Zeroing (`b2b2bfe892`, `#7008`)**: Zeroes out-of-bounds rows and column blocks in the `128x4` swizzled scale tensor between `griddepcontrol.wait` and `griddepcontrol.launch_dependents`.
- **TileLang Signed Atomics & VectorizeLoop Guard (`2573d47079` `#3307`, `cb26ee0faf` `#3345`)**: Uses `cuda::atomic_ref<T1>` so negative `int64` values do not wrap above positive values in `AtomicMax`/`AtomicMin`, and rejects `step != 1` in `VectorizeLoop`.

---

## 2. Deterministic Simulation & Quantitative Verification

Validated via `experiments/flashinfer_moe_gdn_sm12x_fp4_tilelang_phase92/run.py`:

| Module / Commit | Metric | Baseline | Fixed / Verified | Architectural Impact |
|---|---|---:|---:|---|
| `flashinfer@215f83cb27` (`#7368`) | Hopper BF16 4-warp gather + weight TMA lead | `680` cycles | `410` cycles | `1.6585x` staging speedup |
| `flashinfer@801fcd0a52` (`#6371`) | SM12x GDN `64x64` upper-triangular `M16xN8xK16` tiles | `32` tiles | `20` tiles | Skips `12` zero tiles (`-37.5%`) |
| `flashinfer@55b2e33697` (`#5492`) | SM100 GDN indexed `state` stride divisibility | `divby = 1` (`2B` load) | `divby = 128` (`16B` load) | Eliminates `3.2x` slowdown |
| `flashinfer@063604bfc1` (`#7433`) | SM120 FP16 W4A16 E4M3 positive subnormals kept | `0 / 7` | `7 / 7` | Prevents small-scale block flushing |
| `flashinfer@b2b2bfe892` (`#7008`) | `M=1, K=256` NVFP4 `128x4` swizzle zeroed pad slots | External memset / dirty | `2032` slots zeroed in-kernel | PDL-safe with zero extra kernel launch |
| `flashinfer@90ac2eeb7e` (`#7050`) | `B=64` rectangular cuDNN page-table copy launches | `64` slice copies | `1` `view(64, 32)` copy | `64x` fewer launches, preserves CUDA Graph ptr |
| `tilelang@2573d47079` (`#3307`) | `int64` `AtomicMax` / `AtomicMin` on `[-12, 3, -5, 19, -1]` | `-1` / `3` (unsigned wrap) | `19` / `-12` | Restores signed 64-bit atomic ordering |

---

## 3. Architectural Insights & Design Exercises

1. **Upper-Triangular Tile Pruning Condition**: An `M16xN8xK16` tile at `(k, n)` accesses rows $r \in [16k, 16k+15]$ and columns $c \in [8n, 8n+7]$ of the right-hand matrix. Because an upper-triangular matrix is zero whenever $r > c$, the entire tile is zero whenever $\min(r) > \max(c) \iff 16k > 8n + 7$.
2. **PDL Ordering of In-Kernel Padding Initialization**: Under Programmatic Dependent Launch (`griddepcontrol`), dependent GEMM kernels can begin reading global memory as soon as `griddepcontrol.launch_dependents()` executes; zeroing the `128x4` swizzled scale padding before `launch_dependents()` guarantees deterministic visibility without host-side `cudaMemsetAsync`.

---

## 4. Verification & Falsifiable Follow-Up Questions

- **Verification Command**: `python experiments/flashinfer_moe_gdn_sm12x_fp4_tilelang_phase92/run.py --check` and `python -m unittest tests/test_phase92_experiments.py`.
- **Falsifiable Follow-Up Question**: If the SM12x GDN prefill chunk size increases from `64` to `128` using the same `M16xN8xK16` tiling, does the below-diagonal tile pruning ratio increase from $12/32 = 37.5\%$ to $56/128 = 43.75\%$?
