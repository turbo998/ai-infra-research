# FlashInfer CAKE Det Loss FP32 Sparse dW Zero-Fill & SM90 WCX Mega-MoE, FlashAttention CuTe PackGQA Predication & FlashMLA Stable ABI (Phase 163 · Priority 172)

> **中文版本**: [`flashinfer-dw-fp32-wcx-fa-packgqa-flashmla-abi-phase163.md`](flashinfer-dw-fp32-wcx-fa-packgqa-flashmla-abi-phase163.md)

## 1. Primary Source Evidence & Classification

| Source ID | Repository & Commit | Release Date | Evidence Tier | Key Code Path & Architectural Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| `src-flashinfer-cake-det-loss-dw-fp32-words-2026` | `flashinfer-ai/flashinfer` `#8125` (`78420faf30`) | 2026-10-31 | `implementation` | `flashinfer/experimental/cake_lm_head_det_loss/cake_support.py`: Zero-fills global sparse $dW$ as 32-bit FP32 words instead of 16-bit BF16 elements, cutting store transactions by 50% |
| `src-flashinfer-cake-mega-moe-sm90-wcx-decode-2026` | `flashinfer-ai/flashinfer` `#8089` (`753db98b9a`) | 2026-10-31 | `implementation` | `benchmarks/bench_moe_ep_sm90_bf16_mega.py`: Adds SM90 BF16 opt-in compressed expert weights (WCX) for Mega-MoE decode rounds, halving HBM weight read traffic |
| `src-flashinfer-grouped-mm-cudnn-cublas-fix-2026` | `flashinfer-ai/flashinfer` `#8087` (`b08082f3f9`) | 2026-10-31 | `implementation` | `flashinfer/grouped_mm/cudnn/core.py`: Fixes cuBLAS/cuDNN grouped matrix multiplication kernel dispatch on boundary problem shapes |
| `src-flashinfer-cake-mega-mhc-sm107a-fast-coeff-2026` | `flashinfer-ai/flashinfer` `#8084` (`4eece64428`) | 2026-10-31 | `implementation` | `csrc/experimental/deepgemm_mega_mhc/generated/*`: Implements fast coefficient chains for two `sm_107a` tail programs, reducing dependency depth from 6 to 3 stages |
| `src-flashattn-cute-packgqa-padded-headdim-2026` | `Dao-AILab/flash-attention` `#2775` (`0dc2cb48e4`) | 2026-09-15 | `implementation` | `flash_attn/cute/pack_gqa.py`: Fixes CuTe forward `PackGQA` predication when logical head dimension is padded ($192 \to 256$) |
| `src-flashattn-setup-cpp20-pytorch213-2026` | `Dao-AILab/flash-attention` `#2879` (`f87af617fa`) | 2026-09-16 | `implementation` | `setup.py`: Builds CUDA extensions with C++20 on PyTorch 2.13+ |
| `src-cutlass-cutedsl-tvm-ffi-cpp-docs-2026` | `NVIDIA/cutlass` (`35e028f46e`) | 2026-08-05 | `specification` | `media/docs/pythonDSL/guides/tvm_ffi_compilation.rst`: Documents invoking `@cute.jit` compiled kernels directly from C++ via TVM-FFI |
| `src-cutlass-47-release-update-2026` | `NVIDIA/cutlass` `#3432` (`418ba45013`) | 2026-08-05 | `specification` | `CHANGELOG.md`, `README.md`: CUTLASS v4.7 release specification and CuTe updates |
| `src-triton-amd-llvm-08b05a33-bump-2026` | `triton-lang/triton` `#12523` (`a66cb2dcc0`) | 2026-10-27 | `implementation` | `cmake/amd-llvm-build-info.json`: Updates AMD CodeGen LLVM commit pin to `08b05a3352ee` |
| `src-triton-amd-codegen-mir-dump-swap-2026` | `triton-lang/triton` `#12122` (`54f03eb5c4`) | 2026-10-26 | `implementation` | `python/src/llvm.cc`, `third_party/amd/backend/codegen/CMakeLists.txt`: Adds Machine IR (MIR) dump and hot-swap hooks to AMD CodeGen LLVM |
| `src-flashmla-cpython-libtorch-stable-abi-2026` | `deepseek-ai/FlashMLA` `#230` (`0f11e178da`) | 2026-10-15 | `implementation` | `csrc/3rdparty/kerutils/include/kerutils/supplemental/torch_tensors.h`: Stabilizes CPython and LibTorch ABI across PyTorch minor releases |

## 2. Deterministic Verification (`experiments/flashinfer_dw_fp32_wcx_fa_packgqa_flashmla_abi_phase163`)

- **Sparse $dW$ FP32 Word Zero-Fill**: Cuts store transactions by **`50.0%`** (`2 -> 1` per 32-bit word) and lowers zero-fill latency from **`14.80 us`** to **`8.14 us`** (**`-45.0%`**).
- **SM90 Mega-MoE WCX Decode**: Halves HBM weight read traffic (**`-50.0%`**) and speeds up decode rounds from **`42.0 us`** to **`27.3 us`** (**`1.538x`** speedup).
- **CuTe PackGQA Predication & FlashMLA Stable ABI**: Eliminates all **64** out-of-bounds lanes on padded head dimensions (`192 -> 256`) and expands FlashMLA single-wheel compatibility from **1** to **4** PyTorch minor versions.
