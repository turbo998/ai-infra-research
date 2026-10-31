# FlashInfer CAKE Det Loss FP32 稀疏权重清零与 SM90 WCX 压缩专家、FlashAttention CuTe PackGQA 谓词修复与 FlashMLA 稳定 ABI（Phase 163 · Priority 172）

> **English Version**: [`flashinfer-dw-fp32-wcx-fa-packgqa-flashmla-abi-phase163.en.md`](flashinfer-dw-fp32-wcx-fa-packgqa-flashmla-abi-phase163.en.md)

## 1. 一手源码证据与分类表

| 证据编号 | 仓库与提交 | 发布日期 | 证据层级 | 核心改动路径 |
| :--- | :--- | :--- | :--- | :--- |
| `src-flashinfer-cake-det-loss-dw-fp32-words-2026` | `flashinfer-ai/flashinfer` `#8125` (`78420faf30`) | 2026-10-31 | `implementation` | `flashinfer/experimental/cake_lm_head_det_loss/cake_support.py`: 将全局稀疏梯度矩阵 $dW$ 的清零从逐元素 BF16 写入改为按 32-bit FP32 字（word）清零，降低 50% 写事务开销 |
| `src-flashinfer-cake-mega-moe-sm90-wcx-decode-2026` | `flashinfer-ai/flashinfer` `#8089` (`753db98b9a`) | 2026-10-31 | `implementation` | `benchmarks/bench_moe_ep_sm90_bf16_mega.py`: 为 SM90 BF16 Mega-MoE Decode 轮次引入可选压缩专家权重（WCX），降低 50% HBM 权重读取流量 |
| `src-flashinfer-grouped-mm-cudnn-cublas-fix-2026` | `flashinfer-ai/flashinfer` `#8087` (`b08082f3f9`) | 2026-10-31 | `implementation` | `flashinfer/grouped_mm/cudnn/core.py`: 修复分组矩阵乘（Grouped MM）cuDNN/cuBLAS 后端内核在边界形状下的调用异常 |
| `src-flashinfer-cake-mega-mhc-sm107a-fast-coeff-2026` | `flashinfer-ai/flashinfer` `#8084` (`4eece64428`) | 2026-10-31 | `implementation` | `csrc/experimental/deepgemm_mega_mhc/generated/*`: 为两个 `sm_107a` 尾程序实现快速系数链（Fast Coefficient Chain），阶段依赖从 6 级降至 3 级 |
| `src-flashattn-cute-packgqa-padded-headdim-2026` | `Dao-AILab/flash-attention` `#2775` (`0dc2cb48e4`) | 2026-09-15 | `implementation` | `flash_attn/cute/pack_gqa.py`: 修复填充注意力头维度（如 $D_{\text{logical}}=192 \to D_{\text{pad}}=256$）下的 CuTe `PackGQA` 越界谓词掩码 |
| `src-flashattn-setup-cpp20-pytorch213-2026` | `Dao-AILab/flash-attention` `#2879` (`f87af617fa`) | 2026-09-16 | `implementation` | `setup.py`: 在 PyTorch 2.13+ 构建中开启 C++20 编译标准 |
| `src-cutlass-cutedsl-tvm-ffi-cpp-docs-2026` | `NVIDIA/cutlass` (`35e028f46e`) | 2026-08-05 | `specification` | `media/docs/pythonDSL/guides/tvm_ffi_compilation.rst`: 规范化通过 TVM-FFI 在 C++ 宿主程序中直接调用 `@cute.jit` 编译函数的 ABI 契约 |
| `src-cutlass-47-release-update-2026` | `NVIDIA/cutlass` `#3432` (`418ba45013`) | 2026-08-05 | `specification` | `CHANGELOG.md`, `README.md`: CUTLASS v4.7 架构与内联 PTX/CuTe 更新 |
| `src-triton-amd-llvm-08b05a33-bump-2026` | `triton-lang/triton` `#12523` (`a66cb2dcc0`) | 2026-10-27 | `implementation` | `cmake/amd-llvm-build-info.json`: 升级 AMD CodeGen LLVM 基线至 `08b05a3352ee` |
| `src-triton-amd-codegen-mir-dump-swap-2026` | `triton-lang/triton` `#12122` (`54f03eb5c4`) | 2026-10-26 | `implementation` | `python/src/llvm.cc`, `third_party/amd/backend/codegen/CMakeLists.txt`: 为 AMD CodeGen LLVM 增加机器中间表示（MIR）导出与热替换（Dump/Swap）能力 |
| `src-flashmla-cpython-libtorch-stable-abi-2026` | `deepseek-ai/FlashMLA` `#230` (`0f11e178da`) | 2026-10-15 | `implementation` | `csrc/3rdparty/kerutils/include/kerutils/supplemental/torch_tensors.h`: 使 FlashMLA 兼容 CPython 稳定 ABI 与 LibTorch ABI，支持单 Wheel 跨多 PyTorch 次版本复用 |

## 2. 确定性实验结果（`experiments/flashinfer_dw_fp32_wcx_fa_packgqa_flashmla_abi_phase163`）

- **稀疏梯度 $dW$ FP32 字清零**：将 16-bit 逐元素写合并为 32-bit FP32 字写后，写事务减半（**`-50.0%`**），清零延迟从 **`14.80 us`** 降至 **`8.14 us`**（**`-45.0%`**）。
- **SM90 Mega-MoE WCX Decode 压缩权重**：Decode 轮次启用 8-bit 压缩专家权重（WCX）后，HBM 权重读取流量下降 **`50.0%`**，单轮 Decode 延迟从 **`42.0 us`** 降至 **`27.3 us`**（**`1.538x`** 加速）。
- **CuTe PackGQA 填充头维度谓词与 FlashMLA 稳定 ABI**：消除 $D=192 \to 256$ 下 **64** 个越界通道（`64 -> 0`），FlashMLA 单 Wheel 兼容的 PyTorch 次版本数从 **1** 提升至 **4**。
