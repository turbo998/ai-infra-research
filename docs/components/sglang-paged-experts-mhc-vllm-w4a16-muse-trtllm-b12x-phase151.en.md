# SGLang Paged-Experts Exact Pinning, mHC MXFP8, DeepSelect 3-CTA, vLLM ROCm W4A16 64-Bit Offsets, Muse Spark Routing & TensorRT-LLM B12x MoE (Phase 151)

> **Evidence Classification**: Primary source implementation audit (`implementation`) + deterministic stdlib offline simulation (`own-offline`).

## 1. 10 Primary Commit Audit Matrix (Zero-Overlap)

| # | Repository | Commit (`SHA`) | Date | Core Engineering Breakthrough & Code Path |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `sgl-project/sglang` | [`730ed06`](https://github.com/sgl-project/sglang/commit/730ed062478cfbf96c87330df30b7bc750b41d46) | 2026-10-30 | `[Perf] paged-experts: exact size pinning` (`#46456`): Pins exact byte size in `paged_experts/store.py` instead of over-allocated power-of-two slabs. |
| 2 | `sgl-project/sglang` | [`302b78c`](https://github.com/sgl-project/sglang/commit/302b78c73dc0168db5585f0e61c586d6ec2b19fd) | 2026-10-29 | `[Perf] mhc: Split hc_combine_norm_mxfp8 rows by batch size and zero padding scales in the kernel` (`#46024`): Zeros padding scales inside the kernel. |
| 3 | `sgl-project/sglang` | [`da4ffac`](https://github.com/sgl-project/sglang/commit/da4ffac800d6229b9c8ef608120c994a86e8e4be) | 2026-10-29 | `[Perf] [NVIDIA] deep_select: Add a 3-CTA tier for the paged indexer top-k` (`#47328`): Adds 3-CTA dispatch tier in `deep_select/entry.cuh`. |
| 4 | `sgl-project/sglang` | [`b2367b1`](https://github.com/sgl-project/sglang/commit/b2367b1d4bd4c97d810cb1b785f0b3fb63cfce69) | 2026-10-30 | `[Fix] diffusion: LingBot QKV sharing, causal RoPE on no-cache path, uneven split guard` (`#46601`). |
| 5 | `vllm-project/vllm` | [`f81a414`](https://github.com/vllm-project/vllm/commit/f81a414b9e1947c5522087777e8170df262a2fd3) | 2026-10-30 | `[Bugfix][ROCm] Use 64-bit row offsets in the Triton W4A16 GEMM` (`#63054`): Promotes row stride offset arithmetic to `int64` in `triton_w4a16.py`. |
| 6 | `vllm-project/vllm` | [`36ed432`](https://github.com/vllm-project/vllm/commit/36ed432be4b5aa31a1f7b5d1cfba2ca6023a0256) | 2026-10-30 | `[Kernel] Add CUDA routing kernel for Muse Spark Open` (`#64553`): Dedicated `csrc/moe/muse_spark_routing.cu` kernel. |
| 7 | `vllm-project/vllm` | [`d492ee7`](https://github.com/vllm-project/vllm/commit/d492ee7851dc68edcd62db2217ef400f06b6ffde) | 2026-10-30 | `[Qwen4Exp] Remove FlashAttention dependency and standardize QSA naming` (`#64110`). |
| 8 | `vllm-project/vllm` | [`a17e79a`](https://github.com/vllm-project/vllm/commit/a17e79a8d812f53f438986b5e36337b56b0b6623) | 2026-10-30 | `[EPD] Metadata-only image and video inputs for GLM-5.3-Flash` (`#64071`): Eliminates redundant pixel payload copies on the prefill worker. |
| 9 | `NVIDIA/TensorRT-LLM` | [`5b6da9c`](https://github.com/NVIDIA/TensorRT-LLM/commit/5b6da9cf2b6ca8593c984488022b7d7fd891cb95) | 2026-10-29 | `Keep target page size for standalone DSpark/DFlash draft KV pool` (`#21046`). |
| 10 | `NVIDIA/TensorRT-LLM` | [`25f7221`](https://github.com/NVIDIA/TensorRT-LLM/commit/25f72219df4e2a17f5ff2320266b46cc6c5e8a33) | 2026-10-29 | `Refuse or honour every input on the B12x MoE path` (`#21057`): Eliminates silent parameter drops on SM120 Cutlass/CuTe DSL MoE. |

## 2. Deterministic Offline Verification (`own-offline`)

Run `python experiments/sglang_paged_experts_mhc_vllm_w4a16_trtllm_b12x_phase151/run.py --check`:
- **SGLang Paged-Experts Pinning Saving**: **`37.5%`** (`1024 -> 640 MB`); **DeepSelect 3-CTA Speedup**: **`1.3082x`**
- **vLLM ROCm W4A16 64-Bit Offsets**: Prevents 32-bit overflow at `4.29B` elements
- **Muse Spark CUDA Routing Speedup**: **`3.8222x`** (`86.0 -> 22.5 us`); **EPD Metadata Payload Reduction**: **`400.0x`**
- **TensorRT-LLM DSpark Page Table Reduction**: **`4.0x`** (`16 -> 64` tokens/page)
