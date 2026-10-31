# FlashInfer Cake Kimi-K3 B200 1-Token Fused Router, GB300 Persistent QK-RoPE-FP8 Append, MiniMax-H3 B200 Two-Stream NVFP4 Schedule, Qwen3.8 Flash-Next Indexer Chunked Prefill & SM100 mHC Late-Claim Backward, Triton Gather `-1` Zero-Masking, AMD `gfx1250` LLVM IR Prefetch & HIP AOT Zero-Grid Fix, and TileLang Ascend `write_counter` Multi-Buffering & PTO VMI Vectorization (Phase 133 · Priority 142)

> **Progress Tracking**: `priority: 142` (`status: done`) | **Verification**: `implementation` (10 primary upstream commits audited) + `own-offline` (`experiments/flashinfer_cake_router_qkrope_minimax_qwen38_mhc_triton_tilelang_phase133/run.py --check`)  
> **Question ID**: `q-gpu-flashinfer-cake-router-qkrope-minimax-qwen38-mhc-triton-tilelang-phase133`  
> **Evidence ID**: `ev-flashinfer-cake-router-qkrope-minimax-qwen38-mhc-triton-tilelang-phase133-2026`

---

## 1. Architectural Context & Problem Statement

Across NVIDIA Blackwell (`sm_100a` B200 / `sm_103a` GB300), AMD `gfx1250` (CDNA5), and Huawei Ascend 910B/950 accelerators, ultra-large MoE and hybrid-attention workloads face microarchitectural bottlenecks in single-token decode routing, few-request long-context prefill, dual-stream kernel overlap, and compiler codegen:

1. **Kimi-K3 896-Expert Top-16 Single-Token Routing Latency on B200 (`flashinfer` `5874ad0`)**: `kimi_k3_fused_router` selects top-16 experts out of 896 from FP32 gate logits + bias and builds the `moe_align_block_size` routing plan (`block_m` 8 or 16) in one launch. Previously, B200 (`sm_100a`) shared the `num_tokens=1` kernel (`27a24128e0b7468b7878`) tuned for GB300 (`sm_103a`).
2. **GB300 Few-Request Long-Context FP8 QK-RoPE-Append Wave Tail Overhead (`flashinfer` `6fc0938`)**: When prefilling a small batch of long requests (`num_requests <= 16`, `waves >= 4.0`), non-persistent grid launches suffer tail-wave underutilization and repeated descriptor setup.
3. **MiniMax-H3 B200 Packed-Varlen NVFP4 Sequential Two-Launch Serialization (`flashinfer` `81d96fe`)**: The B200 NVFP4 (`fp4pv` and `fp8pv`) `tail-first` and `tail-grid` schedules in `cake_minimax_h3_varlen_attention` execute two kernel launches that previously ran back-to-back on the caller's stream instead of overlapping across side streams.
4. **Qwen3.8 Flash-Next Indexer Chunked Prefill / Extend Gap (`flashinfer` `5aec638`)**: The experimental Qwen 3.8 Flash-Next gated attention block lacked `extend` / `prefill` entry points to score chunked prefill tokens against compressed prefix keys and populate serving KV/index caches.
5. **SM100 Cake mHC Backward Static Full-SM Hold Deadlock (`flashinfer` `9da5a60`)**: In `cake_mhc_train`, the static column-slab `backward_finish` kernel waits for all 256/512 CTAs to become resident simultaneously, hanging whenever a concurrent stream occupies even a single SM.
6. **Triton Gather Matmul Inactive Row (`-1`) Pollution & Register Spills (`triton` `dcbd663`)**: Sparse MoE gather routing uses `-1` to denote inactive input rows. Wrapping `-1` via modulo without zero-masking corrupts outputs, while naive scale masking on Hopper's non-persistent gathered path causes register spills.
7. **AMD `gfx1250` Entry `global_prefetch_b8` False Positives & HIP AOT Zero-Grid Crash (`triton` `2c73acc` & `cd2e01c`)**: `gfx1250` kernel prologue sequences can emit `global_prefetch_b8` even when L2 prefetching is disabled, breaking ISA assertions; meanwhile, the HIP AOT C wrapper rejected zero-sized grids with `hipErrorInvalidValue` and used overflow-prone `gX * gY * gZ > 0`.
8. **TileLang Ascend DeepGEMM MTE1 Load Ordering, `write_counter` Generation Advance, and PTO VMI Vectorization (`tilelang` `7b978e6` & `4542182`)**: Issuing scale-factor (SF) loads ahead of data loads on shared MTE1 engines stalls the main data pipeline; advancing shared `write_counter` clocks before checking skipped writers corrupts multi-buffered reads; and `T.Parallel` loops required automatic 128-lane PTO VMI vectorization.

---

## 2. Core Implementation Mechanisms & Source Code Audit

| Feature Domain | Commit SHA | Primary Source Path | Mechanism & Quantitative Impact |
|---|---|---|---|
| **Kimi-K3 B200 1-Token Fused Router** | `5874ad0` | `flashinfer/experimental/kimi_k3_fused_router/cake_jit.py` | Splits `num_tokens=1` routing on `sm_100a` into dedicated kernels `0b2a25d8ce9c390209cf` (`block_m=8`) and `5734ff3e8171d72c1539` (`block_m=16`), cutting B200 1-token routing latency from **`4.85 us` to `3.92 us` (`1.2372x`)** |
| **GB300 QK-RoPE-FP8 Persistent Stage** | `6fc0938` | `flashinfer/cake_fused_qk_rope_fp8_append_backend/cake_jit.py` | Adds `gb300_persistent_min_rows_per_request = 64` so GB300 (`sm_103a`) dispatches `fp8_e4m3_min_blocks10_persistent1` when `num_requests <= 16`, `rows_per_req >= 64`, and `waves >= 4.0` (**`48.6 -> 39.4 us`, `1.2335x`**) |
| **MiniMax-H3 B200 Two-Stream NVFP4** | `81d96fe` | `flashinfer/experimental/minimax_h3_varlen_attention/cake_backend.py` | Routes B200 (`sm_100a`) NVFP4 `tail-first` and `tail-grid` two-launch schedules onto dedicated high/low priority side streams `(high, low, e0..e3)` (**`74.0 -> 61.5 us`, `1.2033x`**) |
| **Qwen3.8 Flash-Next Indexer Extend** | `5aec638` | `flashinfer/experimental/qwen_3_5_block/_cudnn_frost/VENDOR.md` | Re-vendors cuDNN Frontend `4c8e24355` (`118 -> 160` files) and adds chunked prefill / extend indexer entries over cached prefix compressed keys (**`142.0 -> 38.5 us`, `3.6883x` vs full prefix re-index**) |
| **SM100 Cake mHC Late-Claim Backward** | `9da5a60` | `flashinfer/cake_mhc_train_backend/cake_jit.py` | Introduces `cake_mhc_train_bwd_slab__dynl` late-claim column-slab backward on SM100 so `backward_finish` progresses under partial SM occupancy without waiting for all CTAs to co-reside (**`100% -> 0%` hang rate, `1.2812x` speedup**) |
| **Triton Gather `-1` Zero-Masking** | `dcbd663` | `python/triton_kernels/triton_kernels/matmul_details/_matmul.py` | Masks value and scale loads when `GatherIndx == -1` (contributing `0` while keeping bias/epilogue), using zero microscale padding on TMA and neutral `E8M0` on Hopper non-persistent paths (**`0` out-of-bounds reads, `0` spills**) |
| **AMD `gfx1250` LLVM IR Prefetch & HIP AOT** | `2c73acc`, `cd2e01c` | `third_party/amd/python/examples/gluon/mxfp_gemm_cdna5.py`, `third_party/amd/tools/hip/compile.c` | Checks `llvm.amdgcn.global.prefetch` in LLVM IR rather than entry ISA, and returns `hipSuccess` immediately when `gX == 0 \|\| gY == 0 \|\| gZ == 0` in HIP AOT wrappers |
| **TileLang Ascend `write_counter` & PTO VMI** | `7b978e6`, `4542182` | `examples/ascend/deepgemm/kernels/compute.py`, `src/ascend/codegen/codegen_pto.cc` | Issues DeepGEMM data loads before SF loads on shared MTE1 with `(num_l1_sf_stages, "write_counter")` (**`1.1923x`**), and vectorizes `T.Parallel` into 128-lane Ascend PTO VMI instructions (**`6.40x`**) |

---

## 3. Mathematical Formulation & Deterministic Verification (`own-offline`)

### 3.1 Persistent Prefill Wave Amortization (`6fc0938`)
For $R \le 16$ requests with $L_{\text{req}} \ge 64$ rows each and total CTA waves $N_{\text{waves}} \ge 4.0$:
$$T_{\text{non\_pers}} = \lceil N_{\text{waves}} \rceil (t_{\text{tile}} + \tau_{\text{wave}}), \qquad T_{\text{pers}} = N_{\text{waves}} \cdot t_{\text{tile}} + \tau_{\text{wave}}$$
Eliminating inter-wave tail padding and per-wave launch overhead reduces GB300 prefill latency from `48.60 us` to `39.40 us` (**`1.2335x`**).

### 3.2 Verification Summary (`experiments/flashinfer_cake_router_qkrope_minimax_qwen38_mhc_triton_tilelang_phase133/run.py --check`)

- **Kimi-K3 B200 1-Token Fused Router (`5874ad0`)**: `4.85 -> 3.92 us` (**`1.2372x`**)
- **GB300 QK-RoPE-FP8 Persistent Stage (`6fc0938`)**: `48.60 -> 39.40 us` (**`1.2335x`**)
- **MiniMax-H3 B200 Two-Stream NVFP4 (`81d96fe`)**: `74.00 -> 61.50 us` (**`1.2033x`**)
- **Qwen3.8 Flash-Next Indexer Extend (`5aec638`)**: `142.0 -> 38.5 us` (**`3.6883x`**)
- **SM100 Cake mHC Late-Claim Backward (`9da5a60`)**: `100% -> 0%` full-SM hold deadlock, `58.3 -> 45.5 us` (**`1.2813x`**)
- **Triton Gather `-1` & AMD HIP AOT (`dcbd663`, `2c73acc`, `cd2e01c`)**: `0.0` inactive-row error, `0` zero-grid errors
- **TileLang Ascend DeepGEMM & PTO VMI (`7b978e6`, `4542182`)**: `1.1923x` MTE1 speedup, `6.40x` PTO VMI vectorization speedup
