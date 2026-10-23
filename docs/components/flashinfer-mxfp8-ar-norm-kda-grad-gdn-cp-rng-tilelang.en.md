# Deep Dive into FlashInfer MXFP8 AllReduce+Residual+RMSNorm Fusion, Blackwell/Rubin Cake KDA Chunked Grad & SM107 GDN CP Precompute, Ragged Prefix Retention, Philox RNG Offsets & TileLang Guards

> **Phase 61 · Priority 70 · Track A (GPU Kernel / FlashInfer / TileLang / Quantization)**
> **Bilingual Edition**: [中文正文](./flashinfer-mxfp8-ar-norm-kda-grad-gdn-cp-rng-tilelang.md) | [English Version](./flashinfer-mxfp8-ar-norm-kda-grad-gdn-cp-rng-tilelang.en.md)
> **Reproducible Experiment**: `python experiments/flashinfer-mxfp8-ar-norm-kda-grad-gdn-cp-rng-tilelang/run.py --check`

---

## 1. Architectural Context & Problem Statement

When running low-latency speculative decoding (such as EAGLE3 with NVFP4/MXFP8) and ultra-long-context linear attention training/prefill (KDA and Gated DeltaNet) on NVIDIA Blackwell (`sm_100a` B200, `sm_103a` GB300) and Rubin (`sm_107a` R200) GPUs, the kernel stack encounters four critical performance and correctness boundaries:

1. **Standalone MXFP8 Quantization Launch Overhead & Padded Scale Zeroing After AllReduce + RMSNorm**: Separating TP AllReduce + Residual + RMSNorm from downstream MXFP8 quantization (E4M3 activations + 1D-32 UE8M0 block scales) incurs an extra `1.6–1.8 μs` kernel launch per layer. Worse, under the `SWIZZLED_128x4` scale layout, zeroing out padded rows `[token_num, round_up(token_num, 128))` consumes over half the kernel execution time at small decode batch sizes (`token_num = 4`).
2. **Packed Long-Sequence KDA Chunked Training Throughput & SM107 GDN CP Precompute Specialization**: Variable-length packed sequences (`S ≈ 240K–261K`) in KDA chunked forward and backward passes (`light`, `tape`, and `checkpoint-replay`) require tighter TMEM/register staging on SM100/SM103/SM107, while context-parallel (CP) GDN prefill `T-precompute` lacked a dedicated `sm_107a` (Rubin) target binary.
3. **CUDA Graph Ragged Prefix Pointer Invalidation & Speculative Sampling Philox RNG Collisions**: In `BatchPrefillWithRaggedKVCacheWrapper`, when `is_cuda_graph_enabled=True` under `causal + NHD + auto`, a subsequent `plan()` could drop or replace `ragged_prefix_buffers_`, leaving captured CUDA Graphs pointing at stale staging memory. Meanwhile, `get_philox_rng_offsets` incremented the counter before returning the offset and `chain_speculative_sampling` reserved only `B*(N+1)` offsets despite consuming `B*(N+2)` draws on rejection, causing adjacent calls to collide at offset `8`.
4. **TileLang Non-Unit Step Vectorization & NVRTC Disk Cache Library Path Loss**: TileLang's `LoopVectorizer` vectorized `For` loops without checking `step == 1`, silently miscompiling strided loops into contiguous vector instructions, and `TileLangNVRTCSourceWrapper` failed to populate `adapter.libpath` on disk-cache hits.

---

## 2. Upstream Source-Level Implementation & Formulation

### 2.1 FlashInfer Fused AllReduce + Residual + RMSNorm + MXFP8 Quantization (`2c89a4fefbd9`)

In `include/flashinfer/comm/trtllm_allreduce_fusion.h` and `csrc/trtllm_mnnvl_allreduce.cu`, FlashInfer adds two new fusion patterns and an MNNVL quantization enum:
- `kARResidualRMSNormMXFP8Quant = 13` (omits unquantized `norm_out`)
- `kARResidualRMSNormOutMXFP8Quant = 14` (writes both `norm_out` and MXFP8 `quant_out` + `scale_out`)
- `MNNVLQuantType::MXFP8 = 4`

**Mathematical Formulation & Hardware Invariants**:
1. **32-Element UE8M0 Block Scale via Warp Shuffles**: Each thread holds `kElems = 8` (for FP16/BF16) normalized FP32 values. Every `threadsPerSf = 32 / kElems = 4` consecutive threads reduce their local maximum across `delta ∈ {2, 1}` via `__shfl_xor_sync` to obtain $\text{amax}_{32}$, then compute:
   $$\text{scale}_{\text{ue8m0}} = \text{float\_to\_ue8m0\_ceil}\left(\frac{\max(\text{amax}_{32}, 10^{-12})}{448.0}\right), \quad s_{\text{inv}} = 2^{-( \text{scale}_{\text{ue8m0}} - 127 )}$$
2. **CTA & Cluster Width Divisibility Invariant**: For `hidden_dim = 1056` (e.g., MiniMax-M3), splitting the hidden dimension across a multi-CTA MNNVL cluster can leave `blockSize % threadsPerSf != 0` (e.g., `66 % 4 == 2`). The launcher narrows the cluster until every 32-element scale group resides within a single CTA:
   ```cpp
   while (blockSize % threadsPerSf != 0 && clusterSize > 1) {
     clusterSize /= 2;
     blockSize *= 2;
   }
   ```
3. **Writing Only Live Rows in `SWIZZLED_128x4` Layout**: The physical byte size of a `SWIZZLED_128x4` scale tensor is:
   $$\text{Bytes}_{\text{scale}} = \text{round\_up}(M, 128) \times \text{round\_up}\left(\frac{K}{32}, 4\right)$$
   At $M = 4, K = 1056$, $\text{num\_sf\_cols} = 33$, yielding $128 \times 36 = 4608\text{ B}$. The fused epilogue writes only live rows $m < M$ (padding columns `[33, 36)` with zeros) and leaves padded rows `[4, 128)` untouched—eliminating **124 padded row writes (96.88% reduction)** because downstream GEMM/MoE consumers only index `< token_num`.

### 2.2 Cake KDA Chunked Forward/Backward (`7b7db10941ad`) & SM107 GDN CP Precompute (`8ba1ba1d6b04`)

- **Cake KDA Chunked Grad (SM100 / SM103 / SM107)**: Optimizes variable-length packed sequences `[1, S, H, 128]` across `sm_100a`, `sm_103a`, and `sm_107a` for `light`, `tape`, and `ckpt` (checkpoint-replay) modes. Across 14 benchmark shapes on GB300 (`sm_103a`, 152 SMs), it delivers **1.435x–1.892x** speedups:
  - `P1 (H=4, S=239,718)`: forward `2.049 → 1.292 ms` (**1.586x**), backward `5.044 → 2.984 ms` (**1.691x**), `ckpt` `9.119 → 5.558 ms` (**1.641x**).
  - `P9 (H=1, S=261,393)`: forward `0.797 → 0.450 ms` (**1.771x**), backward `1.758 → 0.929 ms` (**1.892x**).
  - `P14 (H=64, S=261,393)`: `ckpt` end-to-end `129.050 → 79.415 ms` (**1.625x**, saving `49.635 ms` per step).
- **Rubin SM107 GDN CP `T-precompute` (`8ba1ba1d6b04`)**: Registers a dedicated `sm_107a` kernel (`kernel_cake_gdn_cp_ce1c179ef0e27da84479`) instead of falling back to `sm_100a`, achieving a **1.048x geomean speedup** (up to **1.066x**) across 48 shapes on R200 (`T=65536, Hk=4, Hsab=16`: `0.12048 → 0.11405 ms`).

### 2.3 Ragged Prefix Retention, Philox RNG Offset Fixes & TileLang Compiler Guards

1. **Ragged Auto Prefix Buffer Retention (`ad57e22e41ce`)**: Once `ragged_prefix_buffers_` is allocated during the initial `plan()` of a CUDA-graph-enabled `BatchPrefillWithRaggedKVCacheWrapper`, subsequent `plan()` calls unconditionally pass those owned staging buffers to `_plan()`, preventing pointer invalidation across graph replays.
2. **Philox RNG Pre-Increment Return & `B*(N+2)` Speculative Reservation (`0602a5b9929d`)**:
   - `get_philox_rng_offsets` now captures `uint64_t current = *offset_ptr;` before advancing `*offset_ptr = current + increment; return current;`.
   - On rejection, `chain_speculative_sampling` draws $N$ uniform values plus 2 Gumbel-max steps ($N + 2$ draws per batch row). Reserving $\text{round\_up}(B(N+2), 4) = 8$ instead of $\text{round\_up}(B(N+1), 4) = 4$ at $B=1, N=3$ places call 1 at $\{0..4\}$ and call 2 at $\{8..12\}$, eliminating the offset `8` collision.
3. **TRT-LLM MoE Unshuffled FP8 Workspace Query & Autotuner Rank Agreement (`75d4c6e61ddc`, `ba8bd0329958`)**: Preserves caller-specified `tile_N` when only `config == -1` so FP8 unshuffled MoE workspace queries return valid non-zero byte counts, and enforces rank-wide tactic agreement in Autotuner v2.
4. **TileLang Step-1 Vectorizer Guard & NVRTC Cache Binding (`cb26ee0fafe5`, `a2cf4c2b1e8e`, `35ee29507557`)**: Enforces `ICHECK(!op->step.has_value() || is_one(op->step.value()))` in `LoopVectorizer`, lowers `clamp` to `T.min(T.max(x, lo), hi)`, and binds `adapter.libpath` on NVRTC disk-cache hits.

---

## 3. Deterministic Verification Matrix

Verified via `python experiments/flashinfer-mxfp8-ar-norm-kda-grad-gdn-cp-rng-tilelang/run.py --check`:

| Evaluation Dimension | Baseline / Pre-Fix | Phase 61 Optimized / Fixed | Quantitative Gain / Architectural Invariant |
| :--- | :--- | :--- | :--- |
| **B300 TP4 MiniMax-M3 NVFP4 Decode TPOT (TRT-LLM AR)** | `2.241 ms` (standalone `mxfp8_quantize`) | **`2.208 ms`** (`kARResidualRMSNormMXFP8Quant`) | **-1.47% E2E TPOT** (`+0.5 μs` fused tail vs `1.7 μs` standalone) |
| **B300 TP4 MiniMax-M3 NVFP4 Decode TPOT (MNNVL AR)** | `2.190 ms` (standalone `mxfp8_quantize`) | **`2.142 ms`** (`MNNVLQuantType.MXFP8 = 4`) | **-2.19% E2E TPOT**, auto cluster narrowing ensures `blockSize % 4 == 0` |
| **MXFP8 `SWIZZLED_128x4` Scale Row Writes ($M=4, K=1056$)** | Zeroes all `128` rows (`4608 B`) | **Writes only `4` live rows** (`144 B`) | **Eliminates 96.88% (124 rows)** of redundant padded scale writes |
| **GB300 Cake KDA Chunked Grad (`P9: H=1, S=261,393`)** | Fwd `0.797 ms` / Bwd `1.758 ms` | **Fwd `0.450 ms` / Bwd `0.929 ms`** | **1.771x Forward / 1.892x Backward** speedup |
| **GB300 Cake KDA Chunked Grad (`P14: H=64, S=261,393`)** | Ckpt E2E `129.050 ms` | **Ckpt E2E `79.415 ms`** | **1.625x speedup** (`49.635 ms` saved per step) |
| **R200 (`sm_107a`) Cake GDN CP `T-precompute`** | Fallback `sm_100a` (`0.12048 ms`) | **Native `sm_107a` (`0.11405 ms`)** | **1.048x geomean speedup** across 48 shapes (up to `1.066x`) |
| **Speculative Sampling Philox RNG ($B=1, N=3$ Reject)** | Reserves `4`, collides at offset `[8]` | **Reserves `8` (`B*(N+2)`), starts at `0, 8`** | **Overlapping draws `1 → 0`**, zero RNG stream correlation |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Sizing Exercise**: For `token_num = 8, hidden_dim = 4096`, what is the physical byte size of the `SWIZZLED_128x4` MXFP8 scale tensor, and what percentage of scale row writes is saved by skipping padded rows?
   - *Solution*: $\text{num\_sf\_cols} = 4096 / 32 = 128$, $\text{padded\_rows} = 128$, total $= 16,384\text{ B}$. Writing only 8 live rows ($1,024\text{ B}$) saves $120 / 128 = 93.75\%$ of scale writes.
2. **Falsifiable Hypothesis**: If MNNVL two-shot splits `hidden_dim = 1056` (FP16, 8 elements/thread) into CTAs of `blockSize = 66` without narrowing `clusterSize`, will every 32-element scale group reduce within a single CTA?
   - *Verification*: False. `66 % 4 == 2`, splitting a 4-thread (32-element) scale group across CTA boundaries and corrupting the warp-shuffle UE8M0 scale reduction.
