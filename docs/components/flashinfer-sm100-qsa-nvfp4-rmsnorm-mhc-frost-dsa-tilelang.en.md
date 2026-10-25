# Deep Dive into FlashInfer SM100/SM103 `tcgen05` QSA NVFP4 Sparse Decode, RMSNorm/mHC Training Backward Kernels, FROST Grouped Tensor Core, and TileLang Compiler Guardrails

> **中文版**: [flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang.md](flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang.md)  
> **Phase**: 72 (`priority: 81` · Track A) · **Verified Date**: `2026-10-25` (UTC+08:00)  
> **Verification Script & Sealed Summary**: [`experiments/flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang/run.py`](../../experiments/flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang/run.py) · [`summary.json`](../../experiments/flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang/summary.json)

---

## 1. Architectural Context and Core Problems

On NVIDIA Blackwell (`sm_100a` B200 / `sm_103a` GB300 / `sm_120` RTX PRO 6000 Blackwell) and Vera Rubin (`sm_107a` R200), sparse attention decoding (QSA NVFP4 / DSA Indexer), training normalization and multi-stream hyper-connections (`cake_rmsnorm_train` / `cake_mhc_train`), and low-latency MoE routing (`frost_low_latency`) encounter four hardware-compiler boundary bottlenecks:

1. **Blackwell `tcgen05` Tensor Memory (TMEM) Occupancy vs. Split-KV Budgeting**: On `sm_80`/`sm_90a`, `cake_qsa_nvfp4` uses `mma.sync` with `2` resident CTAs per SM. On `sm_100a`/`sm_103a`, `tcgen05` tensor-memory allocation limits residency to `1` CTA per SM. Reusing Hopper's split heuristic causes wave tail quantization or excessive split-merge barrier overhead at small batch sizes.
2. **Redundant Floating-Point Multiplies in NVFP4 E2M1 Dequantization Loops**: On `sm_80`/`sm_90a`, `cake_qsa_nvfp4` previously multiplied every block's E4M3 scale by `rebias` inside the tile loop and read E2M1 magnitude tables from global symbols.
3. **Training Backward Register Pressure, Tail Chunking, and Rubin `min_cuda` Floors**: Backward RMSNorm and 4-stream mHC kernels across `H in [512, 6144]` suffer from non-uniform register address arithmetic, per-element BF16-to-FP32 conversion overhead, and strict CUDA 13.4 PTX compiler requirements on `sm_107a`.
4. **Ultra-Low-Batch (`T in {4, 8}`) MoE Routing and TileLang Lowering Soundness**: Scalar SIMT MoE kernels leave Tensor Cores idle at `T in {4, 8}`, while SM120 W4A16 CUDA Graph capture under worst-case distinct-expert routing overflows `packed_slots` scratch buffers. Meanwhile, TileLang's `VectorizeLoop`, `cuda::atomic_ref` codegen, and NVRTC cache loader carried subtle correctness bugs.

---

## 2. First-Hand Upstream Code Audit and Mechanics

### 2.1 FlashInfer `e73481617190` (`#7205`) & `cb93a30db500` (`#7175`): `tcgen05` QSA NVFP4 Sparse GQA Decode and E4M3 Rebias Fold

In [`flashinfer/comm/cake_qsa_nvfp4.py`](https://github.com/flashinfer-ai/flashinfer/commit/e73481617190444ab56ac258f6be5b4f673bc8b6) and `csrc/cake_qsa_nvfp4/`, upstream added native `tcgen05` QSA NVFP4 sparse paged GQA decode kernels for `sm_100a` and `sm_103a` (`num_qo_heads=24, num_kv_heads=2, head_dim=256, top_k=2051`, supporting both `NHD` and `HND` layouts):

- **Architecture-Aware Split-KV Rule**:
  $$\text{tiles} = \left\lceil \frac{\text{selection\_width}}{64} \right\rceil, \qquad \text{splits} = \max\left(1, \min\left(\text{tiles}, \left\lfloor \frac{\text{CTAS\_PER\_SM}[\text{arch}] \times \text{sm\_count}}{\text{rows} \times H_{kv}} \right\rfloor\right)\right)$$
  where `CTAS_PER_SM = {"sm_80": 2, "sm_90a": 2, "sm_100a": 1, "sm_103a": 1}`. For `selection_width = 2051` (`33` tiles) and `H_kv = 2`, B200 (`sm_100a`, `148` SMs) selects `splits = [33, 33, 18, 9, 4, 2, 1, 1]` across `rows = [1, 2, 4, 8, 16, 32, 64, 128]`, while GB300 (`sm_103a`, `160` SMs) selects `[33, 33, 20, 10, 5, 2, 1, 1]`.
- **Dual Dispatch Modes (Split + Merge vs. Fused Co-Resident)**:
  - Default execution launches `qsa_nvfp4_tcgen05_split` followed by `qsa_nvfp4_merge` (bypassing merge when `splits == 1`).
  - On Grace-Blackwell (`platform.machine() == "aarch64"`) outside CUDA Graph capture when `rows <= 4` and `splits > 1`, the dispatcher launches `qsa_nvfp4_tcgen05_fused`, co-residing `rows * 2 * splits` decode CTAs and `rows * 2` merge CTAs in a single grid launch.
- **E4M3 Block-Scale Rebias Fold (`cb93a30db500`, `#7175`)**:
  On `sm_80`/`sm_90a`, `#7175` folds the per-call $\text{rebias} = 2^{120 - \text{bias}}$ factor directly into the 8-entry E2M1 magnitude lookup table $[0, 0.5, 1, 1.5, 2, 3, 4, 6]$ held in registers per tile, and forms PV scale words from two token-pair scale bytes with a single `prmt.b32` instruction.

| Platform & Mode | Layout & Batch (`top_k=2051`) | Reference `qsa_nvfp4` ($\mu\text{s}$) | `cake_qsa_nvfp4` ($\mu\text{s}$) | Speedup | Provenance |
|---|---|---:|---:|---:|---|
| GB300 (`sm_103a`, Eager memset flush) | `NHD`, `rows=1` | `46.50` | `25.81` | **1.80x** | `author-benchmark` (`#7205`) |
| GB300 (`sm_103a`, Eager memset flush) | `NHD` / `HND`, `rows=128` | `191.09` / `192.02` | `71.94` / `71.87` | **2.66x / 2.67x** | `author-benchmark` (`#7205`) |
| GB300 (`sm_103a`, CUDA Graph replay) | `NHD`, `rows=1` / `rows=128` | `17.60` / `178.94` | `9.92` / `67.15` | **1.77x / 2.66x** | `author-benchmark` (`#7205`) |
| B200 (`sm_100a`, Eager memset flush) | `NHD` / `HND`, `rows=128` | `224.61` / `224.64` | `77.26` / `76.94` | **2.91x / 2.92x** | `author-benchmark` (`#7205`) |
| H100 (`sm_90a`, E4M3 rebias fold) | `NHD` / `HND`, `rows=4` | `19.65` / `19.52` | `18.98` / `18.98` | **1.035x / 1.028x** | `author-benchmark` (`#7175`) |

---

### 2.2 FlashInfer `03f82fddadae` (`#7200`), `0fac207b466d` (`#7199`) & `26d0d5dc906c` (`#7198`): Blackwell/Rubin Training RMSNorm and 4-Stream mHC Kernels

1. **`cake_rmsnorm_train` Backward Optimization (`#7200`)**:
   - **Uniform Register Descriptors (`shfl.sync.idx.b32`)**: Pins CTA-uniform base pointers and strides into uniform registers via lane-0 broadcast and separates branch-free interior loops from tail loops.
   - **Tail Chunking Rebalance**: Changes CTA row chunking from `max(1, (share + 15) // 16)` to `max(1, (share + 11) // 12)`. On B200 (`148` SMs) at `T = 65536` (`share = 443` rows/CTA), chunk size increases from `28` to `37` rows and loop iterations drop from `16` to `12`.
   - **Store-Side PDL Trigger**: Enables programmatic dependent launch completion right after `dx` stores for `1024 < T <= 4096`.
   - **Measured Speedups (`author-benchmark`)**: **1.18x–1.27x** at `H=512` on GB300; **1.332x** (`118.8 -> 89.2 us`) at `H=512, T=65536` on B200; and **1.331x** (`461.4 -> 346.6 us`) at `H=6144, T=16384` on Rubin R200 (`sm_107a`).
2. **Rubin `sm_107a` CUDA 13.4 Floor (`#7199`)**:
   - Raises `min_cuda` from `"13.0"` to `"13.4"` across 35 `sm_107a` entries in `cake_muse_spark_norm_backend` and 6 entries in `cake_tp_moe_finalize_rmsnorm_backend`.
3. **4-Stream `cake_mhc_train` (`H=4096`) Bit-Shift Unpacking (`#7198`)**:
   - Replaces element-wise BF16 conversions with `cake_bf16x2_to_f32x2(uint32_t w)` using `__uint_as_float(w << 16)` and `__uint_as_float(w & 0xffff0000u)` alongside specialized `bwd_reduce__pt32` / `pt64` / `s27` kernels (`256 B/token` saved stats), achieving **1.17x** backward and **1.15x** checkpointed speedups at `T=4096` on GB300 and **1.16x–1.17x** forward speedups on R200.

---

### 2.3 FlashInfer `94b5e266a966` (`#7194`), `5e4555094c32` (`#7195`) & `54459f0872e5` (`#7187`): FROST Low-Latency Tensor Core MoE and DSA Indexer Pipelining

1. **FROST `T in {4, 8}` Grouped BF16 Tensor Core Candidates (`#7194`)**:
   - Uses warp `ballot` in `route_members` to group tokens routed to the same expert at `H=2048, I=512, top_k=10, T in {4, 8}`, dispatching `wgmma` on SM90 and `warp_mma` on SM120 (**1.274x** warm at `T=4` on H100; **1.48x** at `T=4` and **1.37x** at `T=8` on SM120).
2. **SM120 W4A16 CUDA Graph Scratch Slot Sizing (`#7195`)**:
   - Sizes `gate_up_out` and `act_out` scratch slots via:
     $$\text{scratch\_slots} = \max\left(\text{packed\_slots}, \min(\text{routed\_rows}, 8 \times \text{topk}) \times \text{block\_size}\right)$$
     expanding `T=4` scratch capacity from `80` to `640` rows while keeping `route_slots = 80` unchanged so CUDA Graph capture never drops skewed experts.
3. **`cake_dsa_indexer` Candidate Floor & `mbarrier_test_wait` (`#7187`)**:
   - Doubles `cand_cap_floor` from `8192` to `16384` on `sm_100a`/`sm_103a` when tile sampling is active (`sample_tiles_max > 0`) and probes the next tile's completion (`mbarrier_test_wait`) immediately after `mbarrier_arrive(umma_empty_addr)` (**1.12x** on GB200 `Z2cp`, **1.16x** on GB300 `Z3cp`).

---

### 2.4 TileLang Compiler Correctness Guardrails (`#3345`, `#3307`, `#3193`)

1. **`VectorizeLoop` Non-Unit Step Rejection (`cb26ee0fafe5`, `#3345`)**: Enforces `ICHECK(!op->step.has_value() || is_one(op->step.value()))` so stepped loops (`step=2`) are never silently lowered into contiguous `Ramp(base, 1, lanes)` loads/stores.
2. **Signed `int64` and Half-Precision `cuda::atomic_ref<T1>` (`2573d470798a`, `#3307`)**: Removes the `unsigned long long*` pointer cast in `AtomicMax`/`AtomicMin`/`AtomicStore` that caused `atomic_max(-5, 2)` to evaluate `0xFFFFFFFFFFFFFFFB > 2` and return `-5`.
3. **NVRTC `from_database` Library Path Restoration (`35ee29507557`, `#3193`)**: Restores `adapter.libpath = adapter.lib_generator.libpath` when loading cached NVRTC kernels.

---

## 3. Deterministic Verification Experiment

Run the reproducible Python 3.12 standard-library verification script:

```bash
python experiments/flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang/run.py --check
```
