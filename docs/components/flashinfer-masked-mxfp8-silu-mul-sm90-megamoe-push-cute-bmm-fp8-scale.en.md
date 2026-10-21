# FlashInfer SM100+ Masked MXFP8 Experts Quantize & Fused SiLU-and-Mul, SM90 FP4 MegaMoE Push Dispatch, and CuTe-DSL `bmm_fp8`/`mm_mxfp8` One-Wave Priority

> **First-Hand Upstream Evidence (2026-10-21)**:
> - `flashinfer-ai/flashinfer` PR `#6437` (`140f9fc4c5fea322de9de2e520e88cc1ef976d7a`): `feat(quantization): add CUDA backend for mxfp8_grouped_quantize and fused silu_and_mul_mxfp8_experts_quantize`
> - `flashinfer-ai/flashinfer` PR `#6517` (`da992ed9919604c5eb6dc255c63f5e89a466452c`): `[MegaMoE] Fix SM90 FP4 fused kernel, enable H200 push dispatch, and tune tile selection`
> - `flashinfer-ai/flashinfer` PR `#6800` (`bf044ba86e8ec7ff5af11034378c6ff6568e4a8d`): `perf(gemm): fuse scale tensor multiply into Blackwell bmm_fp8 epilogue`
> - `flashinfer-ai/flashinfer` PR `#6848` (`617845d1a3aeadf2b95fb0e5d6a12383b192a963`): `fix(gemm): gate mm_mxfp8 warp split-K priority on one-wave output fit`

---

## 1. SM100+ Masked Expert-Major MXFP8 Quantization & Fused `silu_and_mul_mxfp8_experts_quantize` (PR `#6437`)

In DeepEP low-latency MoE execution, the dispatch collective produces fixed-capacity `[E, M, K]` expert buffers paired with a device-resident valid-row count tensor `mask[E]`. Previously, the `cutile` and `cake` backends for `mxfp8_grouped_quantize` either quantized padded rows unnecessarily or required a separate `silu_and_mul` activation launch before GEMM2 quantization.

PR `#6437` introduces the SM100+ (`__CUDA_ARCH__ >= 1000`) CUDA kernel `cvt_fp16_to_mxfp8_expert<Type, USE_SILU_AND_MUL>` in `csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh`:

1. **Prefix-Sum Flattened Valid-Row Worklist & Register/SMEM Binary Search**:
   - Only the first `min(mask[e], numRows)` valid rows of each expert are read and quantized, flattened into a single contiguous worklist across the grid so skewed expert routing still saturates all SMs.
   - For `numExperts <= 32`, each warp computes the prefix-sum `expertFirstRow` table in registers via butterfly `__shfl_xor_sync` and binary-searches the owning `(expertIdx, rowIdx)`; for `numExperts > 32`, the CTA stores `(numExperts + 1) * sizeof(int32_t)` in dynamic shared memory (`260 B` at `E=64`).
2. **Two-Lane Half-Block Vector Loads & `128x4` Swizzled UE8M0 Scales**:
   - Each `32`-element MXFP8 scale block is split across two adjacent lanes (`HalfBlockT = PackedVec<Type, 16>`, one `32 B` vector load per lane), reduced via warp shuffle, and written directly into the `grouped_gemm_nt_masked` swizzled layout `[32, 4, padded_M // 128, 4, scale_K // 4, E]`.
   - When `USE_SILU_AND_MUL=true`, each lane loads both the gate half-block and up half-block (`inRow + numHalfBlocks + colIdx`), fuses `silu_and_mul` in registers, and emits quantized FP8 E4M3 + UE8M0 scales in a single pass.

### B200 Benchmarks (`E = 32, CAP = 1024`, BF16 Input, `μs`)

| Workload | Valid Rows/Expert `mask[e]` | `cutile` (`μs`) | `cake` (`μs`) | `cuda` (PR `#6437`, `μs`) | Speedup vs `cutile` | Speedup vs `cake` |
|---|---:|---:|---:|---:|---:|---:|
| `mxfp8_grouped_quantize` (`K=7168`, GEMM1) | `16` | `59.17` | — | **`5.90`** | **`10.03x`** | — |
| `mxfp8_grouped_quantize` (`K=7168`, GEMM1) | `128` | `94.58` | `38.21` | **`20.18`** | **`4.69x`** | **`1.89x`** |
| `silu_and_mul_mxfp8_experts_quantize` (`2x2048`, GEMM2) | `16` | `91.22` (unfused) | — | **`4.88`** | **`18.69x`** | — |
| `silu_and_mul_mxfp8_experts_quantize` (`2x2048`, GEMM2) | `128` | `99.81` (unfused) | `83.65` (unfused) | **`11.70`** | **`8.53x`** | **`7.15x`** |

---

## 2. SM90 FP4 MegaMoE Workspace Cleanup Fix, H200 Push Dispatch & Scale Rebalancing (PR `#6517`)

PR `#6517` resolves three critical production issues in the Hopper (`SM90`, H20/H200) fused NVFP4/MXFP4 MegaMoE kernel:

1. **Inactive Dispatch Warp Workspace Cleanup Bug Fix**:
   - Plans using `single_active_dispatch_warp=True` (`BLOCK_M=16` or `24`) configure `kNumDispatchWarps = 2` and `kNumActiveDispatchWarps = 1`. On the pull path (`!kPushDispatch`), warp `1` skipped the pull loop and never called `scheduler.fetch_expert_recv_count()`.
   - During `cleanup_workspace`, all `kNumDispatchWarps = 2` warps stride by 2 across each expert's used pool blocks. Because warp `1` held zeroed expert counts, it computed `start_block = 1` with `count = 0`, leaving pool blocks `>= 32` of hot experts (`112` of `128` blocks when a single expert receives `128` tokens) dirty across iterations.
   - **Fix**: Explicitly execute `if (warp_idx >= kNumActiveDispatchWarps) scheduler.fetch_expert_recv_count();` before `cleanup_workspace`, reducing dirty blocks from `112` to `0`.
2. **H200 EP8 Push Dispatch & Symmetric Buffer Prewarming**:
   - Enables H200 (`141` SMs, EP8) push dispatch for `M <= 32` across `DeepSeek-V4-Flash` (`4096, 256, 6`), `MiMo-V2.5-Pro` (`6144, 384, 8`), and `DeepSeek-V4-Pro` (`7168, 384, 6`), plus `M = 33..64` with rotated pool on `DeepSeek-V4-Pro`, saving `~10 μs/call` (**`1.03x–1.16x`** speedup).
   - Adds `MegaMoESymmBuffer.prewarm()` (faulting peer IPC page mappings before the first timed launch to eliminate a `~20 ms` first-call spike) and `commit_staged()` (making staged buffer copies visible on the caller's stream for CUDA Graph capture).
3. **NVFP4 / MXFP4 In-Register E4M3 Decode Scale Range Contract**:
   - Because the SM90 kernel decodes `FP4 * scale` directly into a single E4M3 byte (max `448.0`), NVFP4 scales must obey `6 * scale <= 448` (`scale <= 74.67`) and MXFP4 nonzero scales must stay within `[2^-5, 2^6]`. Factoring out a per-expert power-of-two multiplier into `l1_global_scales` / `l2_global_scales` keeps weights exact while preventing silent E4M3 saturation (which affected ~51% of `DeepSeek-V4-Flash-NVFP4` blocks).

---

## 3. Blackwell/Rubin (`sm_107`) CuTe-DSL `bmm_fp8` Epilogue Scale Product & `mm_mxfp8` One-Wave Priority (PR `#6800` / `#6848`)

1. **In-Epilogue Scale Tensor Multiplication in `bmm_fp8` (PR `#6800`)**:
   - Instead of computing `output_scale = a_scale * b_scale` via 1–2 separate PyTorch elementwise/cast kernels before every BMM, `SM107PersistentDenseGemmKernel` now accepts `a_scale_tensor` and `b_scale_tensor` directly and evaluates `output_scale = a_scale_tensor[0] * b_scale_tensor[0]` in the device epilogue (`120 -> 0` extra kernel launches across 60 layers).
2. **One-Wave Output-Tile Priority for `mm_mxfp8` (PR `#6848`)**:
   - The `warp` split-K runner uses `16`-wide output tiles (one CTA per output tile). Prioritizing `warp` split-K whenever `m <= 8 or k <= 2048` caused multi-wave CTA schedules on wide `N` (`N > 2368` on 148-SM B200/B300), where `griddepcontrol.wait` stalled second-wave CTAs during back-to-back PDL execution.
   - Gating `warp` split-K priority on `b.shape[1] // 16 <= get_device_sm_count(a.device)` (`N <= 2368` on B200/B300) eliminates **`63 / 150` PDL regressions on B300** (`1.059x` geomean, `1.51x` max speedup) and **`43 / 120` PDL regressions on B200** (`1.059x` geomean, `1.45x` max speedup).

---

## 4. Deterministic Verification

```bash
python experiments/flashinfer_masked_mxfp8_sm90_megamoe_bmm_fp8/run.py --check
```
