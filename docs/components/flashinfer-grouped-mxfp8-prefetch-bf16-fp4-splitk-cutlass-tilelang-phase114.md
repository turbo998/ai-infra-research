# FlashInfer Grouped MXFP8 跨组预取、Blackwell BF16×FP4 Split-K OOB TMA、MiniMax-H3 Norm 共享内存暂存、CUTLASS SM75 与 TileLang Ascend SS/PTO（Phase 114）

> **证据分类**: `implementation` + `own-offline`（上游主干源码分析与可复现离线算例验证，非真实 GPU 硬件 profiling）

## 1. 核心源码改动与微架构机制

| Evidence ID | 上游提交 | 关键实现路径与微架构机制 |
|---|---|---|
| `flashinfer-cake-grouped-mxfp8-quantize-partial-group-prefetch-2026` | `flashinfer-ai/flashinfer@f23e264378` (`2026-10-27`) | `csrc/cake_grouped_mxfp8_quantize/*_kernel.cu` 中 per-K row-tiled 程序在遇到不满整 tile 的 partial group 时，提前 **2~3 个 group** 预取后续 group 的有效行，消除小专家尾块在 MXFP8 量化流水线中的暴露访存停顿（模型算例降低 `80.95%` 暴露停顿周期）。 |
| `flashinfer-cake-blackwell-bf16-fp4-small-m-splitk-2026` | `flashinfer-ai/flashinfer@88063fd695` (`2026-10-27`) | `csrc/blackwell_bf16_fp4/*_binding.cu` 为 SM100/SM103 小 `M` split-K 程序引入 `allow_oob_box[3] = {false, true, true}`，严格禁止最内层 `axis=0` TMA 盒子越界，同时允许 `axis=1,2` OOB zero-fill 并优化 `smk_part` 归约步长（`32` 元素）。 |
| `flashinfer-cake-batch-deepgemm-fp8-blockn48-8stage-sm103a-2026` | `flashinfer-ai/flashinfer@9b08bf90a3` (`2026-10-27`) | `csrc/cake_batch_deepgemm_fp8/cake_batch_deepgemm_fp8_896b65115baf37c01d60_kernel.cu` 在 `sm_103a` 上部署 `BLOCK_N=48` 八级流水线程序，将 `SMEM_EPI_STAGING_STAGE_BYTES` 从 `8192` 扩至 `12288`（`SMEM_TOTAL = 201728` 字节）。 |
| `flashinfer-cake-minimax-h3-nvfp4-norm-8dest-sm103-2026` | `flashinfer-ai/flashinfer@c1e28c1e16` (`2026-10-27`) | `flashinfer/diffusion_ops/cake_minimax_h3_nvfp4.py` 在 `sm_103a` 上不仅对 `P == 4` 启用共享内存行暂存版 `NORM_V5`，还对 `out_format="nvfp4"`、`P == 8` 且 `M >= 6096` tokens 启用 `NORM_V5`（`CUtensorMap` 按值传递，零描述符工作区）。 |
| `flashinfer-cake-deepgemm-mega-mhc-sm107a-tail-2026` | `flashinfer-ai/flashinfer@a50ec035cd` (`2026-10-27`) | `csrc/experimental/deepgemm_mega_mhc/generated/*` 新增 `sm_107a` FP8/NVFP4 尾形变体，并将 3 级 I/O mbarrier 寻址拆分为 `_stage_empty_io = seq % 3` 与 `_parity_empty_io = (seq / 3) % 2 ^ 1`。 |
| `flashinfer-fp4-quantization-grouped-nvfp4-mask-scale-doc-2026` | `flashinfer-ai/flashinfer@d8ce22c0fe` (`2026-10-27`) | `flashinfer/quantization/fp4_quantization.py` 明确 grouped NVFP4 量化输入形状为 `[B, M, 2 * K]`（前半 `gate`、后半 `up`），`mask` 为 `int32 [B]` 有效行数，输出为 `K` 列 `silu(gate) * up`。 |
| `cutlass-cute-sm75-16x8x8-f16-tn-mma-2026` | `NVIDIA/cutlass@6c68991985` (`2026-08-18`) | `include/cute/arch/mma_sm75.hpp` 与 `mma_traits_sm75.hpp` 新增 `SM75_16x8x8_F16F16F16F16_TN` Turing Tensor Core MMA 原子及 `CooperativeGemm2_Half_MMA` 单元测试。 |
| `cutlass-cute-print-tensor-smem-ptr-flag-bits-2026` | `NVIDIA/cutlass@80fe3cac71` (`2026-08-06`) | `include/cute/util/print_tensor.hpp` 补齐 `<cute/pointer_flagged.hpp>` 头文件依赖，修复独立包含 `print_tensor.hpp` 时 `cute::smem_ptr_flag_bits` 符号缺失。 |
| `tilelang-ascend-ss-buffer-support-2026` | `tile-ai/tilelang@7617c7d94d` (`2026-10-16`) | `src/ascend/codegen/codegen_ascend.cc` 支持 `T.alloc_ss`（`shared.ss` 作用域，发射 `volatile __ssbuf__`），在单个 AIC 与 2 个 AIV（`kMix && mix_aiv_count_ == 2`）间共享 `3072` 字节池（32 字节对齐）。 |
| `tilelang-ascend-route-pto-execution-codegen-2026` | `tile-ai/tilelang@c55eeb1c02` (`2026-10-15`) | `tilelang/ascend/` 在 `target="ascend", execution_backend="pto"` 时自动路由至 `PTO_CODEGEN` 与 `pto` 缓存分发。 |

## 2. 架构师计算与练习

1. **Grouped MXFP8 跨组预取**: 当专家平均有效行数 `< 32` 时，单 group 内部无法填满 TMA 流水线；跨 2~3 个 group 预取可将尾块停顿从 `420` 周期降至 `80` 周期。
2. **可证伪假设**: 若在 Ascend `kVector` 或 `mix_aiv_count_ != 2` 内核中声明 `shared.ss` 缓冲区，`codegen_ascend.cc` 必须在编译期触发 `ICHECK` 失败。
3. **复现命令**:
   ```bash
   python experiments/flashinfer_grouped_mxfp8_prefetch_bf16_fp4_splitk_tilelang_phase114/run.py --check
   ```
