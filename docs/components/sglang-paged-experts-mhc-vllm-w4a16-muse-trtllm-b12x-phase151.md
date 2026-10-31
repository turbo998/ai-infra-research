# SGLang Paged-Experts 精确锁页、mHC MXFP8、DeepSelect 3-CTA、vLLM ROCm W4A16 64 位偏移、Muse Spark 路由与 TensorRT-LLM B12x MoE 审计（Phase 151）

> **证据分类**：源码实现级审计 (`implementation`) + 本项目标准库确定性离线算例 (`own-offline`)。

## 一、 10 项核心公开提交审计矩阵（Zero-Overlap）

| 编号 | 仓库 | Commit (`SHA`) | 日期 | 核心工程突破与代码路径 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `sgl-project/sglang` | [`730ed06`](https://github.com/sgl-project/sglang/commit/730ed062478cfbf96c87330df30b7bc750b41d46) | 2026-10-30 | `[Perf] paged-experts: exact size pinning` (`#46456`)：按实际专家页大小精确锁页 (`store.py`)，消除对齐膨胀导致的 Host 锁页内存浪费。 |
| 2 | `sgl-project/sglang` | [`302b78c`](https://github.com/sgl-project/sglang/commit/302b78c73dc0168db5585f0e61c586d6ec2b19fd) | 2026-10-29 | `[Perf] mhc: Split hc_combine_norm_mxfp8 rows by batch size and zero padding scales in the kernel` (`#46024`)：内核内直接清零填充行缩放因子。 |
| 3 | `sgl-project/sglang` | [`da4ffac`](https://github.com/sgl-project/sglang/commit/da4ffac800d6229b9c8ef608120c994a86e8e4be) | 2026-10-29 | `[Perf] [NVIDIA] deep_select: Add a 3-CTA tier for the paged indexer top-k` (`#47328`)：为分页索引器 Top-K 引入 3-CTA 协作档位。 |
| 4 | `sgl-project/sglang` | [`b2367b1`](https://github.com/sgl-project/sglang/commit/b2367b1d4bd4c97d810cb1b785f0b3fb63cfce69) | 2026-10-30 | `[Fix] diffusion: LingBot QKV sharing, causal RoPE on no-cache path, uneven split guard` (`#46601`)：修复无缓存路径 Causal RoPE 与非均匀切分保护。 |
| 5 | `vllm-project/vllm` | [`f81a414`](https://github.com/vllm-project/vllm/commit/f81a414b9e1947c5522087777e8170df262a2fd3) | 2026-10-30 | `[Bugfix][ROCm] Use 64-bit row offsets in the Triton W4A16 GEMM` (`#63054`)：将行偏移提升为 `int64`，防止大 Batch 乘大隐藏层时 32 位索引溢出。 |
| 6 | `vllm-project/vllm` | [`36ed432`](https://github.com/vllm-project/vllm/commit/36ed432be4b5aa31a1f7b5d1cfba2ca6023a0256) | 2026-10-30 | `[Kernel] Add CUDA routing kernel for Muse Spark Open` (`#64553`)：新增原生 `muse_spark_routing.cu` CUDA 路由算子。 |
| 7 | `vllm-project/vllm` | [`d492ee7`](https://github.com/vllm-project/vllm/commit/d492ee7851dc68edcd62db2217ef400f06b6ffde) | 2026-10-30 | `[Qwen4Exp] Remove FlashAttention dependency and standardize QSA naming` (`#64110`)：移除 Qwen4Exp 对 FlashAttention 的硬耦合。 |
| 8 | `vllm-project/vllm` | [`a17e79a`](https://github.com/vllm-project/vllm/commit/a17e79a8d812f53f438986b5e36337b56b0b6623) | 2026-10-30 | `[EPD] Metadata-only image and video inputs for GLM-5.3-Flash` (`#64071`)：在编码-预填充分离 (EPD) 架构中支持仅传递图像/视频元数据。 |
| 9 | `NVIDIA/TensorRT-LLM` | [`5b6da9c`](https://github.com/NVIDIA/TensorRT-LLM/commit/5b6da9cf2b6ca8593c984488022b7d7fd891cb95) | 2026-10-29 | `Keep target page size for standalone DSpark/DFlash draft KV pool` (`#21046`)：当 Drafter 内核支持时保持 Target KV 页大小一致。 |
| 10 | `NVIDIA/TensorRT-LLM` | [`25f7221`](https://github.com/NVIDIA/TensorRT-LLM/commit/25f72219df4e2a17f5ff2320266b46cc6c5e8a33) | 2026-10-29 | `Refuse or honour every input on the B12x MoE path` (`#21057`)：在 `fused_moe_cute_dsl_b12x.py` 中严格校验每个输入参数，消除静默忽略。 |

## 二、 确定性离线算例验证 (`own-offline`)

运行 `python experiments/sglang_paged_experts_mhc_vllm_w4a16_trtllm_b12x_phase151/run.py --check`：
- **SGLang Paged-Experts 锁页内存节省**：`1024.0 MB -> 640.0 MB`（节省 **`37.5%`**）；**DeepSelect 3-CTA 加速比**：**`1.3082x`** (`41.6 us -> 31.8 us`)
- **vLLM ROCm W4A16 64 位偏移**：在 `262,144 x 16,384 = 4,294,967,296` 元素下消除 `int32` 溢出
- **vLLM Muse Spark CUDA 路由加速比**：**`3.8222x`** (`86.0 us -> 22.5 us`)；**EPD 元数据负载缩减**：**`400.0x`** (`48.0 MB -> 0.12 MB`)
- **TensorRT-LLM DSpark 页表项缩减**：**`4.0x`** (`16 -> 64` tokens/page)；**B12x MoE 静默忽略参数**：`0`
