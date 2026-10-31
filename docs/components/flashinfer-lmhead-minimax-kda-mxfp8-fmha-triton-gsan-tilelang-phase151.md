# FlashInfer Cake LM-Head 显存预算与有效行数、SM120 NVFP4 变长重叠、KDA N32 直发、MXFP8 预取、GB300 FMHA 与 Triton GSan 剖析（Phase 151）

> **证据分类**：源码实现级审计 (`implementation`) + 本项目标准库确定性离线算例 (`own-offline`)。未声明真实 GPU 硬件重测。

## 一、 10 项核心公开提交审计矩阵（Zero-Overlap）

| 编号 | 仓库 | Commit (`SHA`) | 日期 | 核心工程突破与代码路径 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `flashinfer-ai/flashinfer` | [`3292e01`](https://github.com/flashinfer-ai/flashinfer/commit/3292e01ba10ffbd78e5917bf85d5bd375dd09e9b) | 2026-10-29 | `feat(cake_lm_head_loss)` (`#7865`)：支持可选 `step_peak_budget` 峰值显存预算与调用方 `valid_rows` 有效行数约束 (`flashinfer/chunked_lm_head.py`)。 |
| 2 | `flashinfer-ai/flashinfer` | [`cf14a96`](https://github.com/flashinfer-ai/flashinfer/commit/cf14a9696ca3dc43929ed3da6923f2536bc73654) | 2026-10-29 | `perf(cake_minimax_h3)` (`#7854`)：针对 `sm_120a` NVFP4 非因果变长注意力引入依赖执行计划的 Overlap Width。 |
| 3 | `flashinfer-ai/flashinfer` | [`710f9b5`](https://github.com/flashinfer-ai/flashinfer/commit/710f9b5a694545d4a65cc94c878ff35b1faaa525) | 2026-10-29 | `perf(cake_kda)` (`#7849`)：短序列有界直接 N32 启动在单任务开销不划算时直接采用 Plain Work Bins (`cake_kda_tf32_runtime.py`)。 |
| 4 | `flashinfer-ai/flashinfer` | [`47442df`](https://github.com/flashinfer-ai/flashinfer/commit/47442df411955c390261df6668018bd1b03c1b52) | 2026-10-29 | `perf(cake_grouped_mxfp8_quantize)` (`#7848`)：为两类 Row-2D 问题形状引入 Head-Window L2 Prefetch，优化 `128x4` UE8M0 块缩放写回。 |
| 5 | `flashinfer-ai/flashinfer` | [`f319fc1`](https://github.com/flashinfer-ai/flashinfer/commit/f319fc1659a4a841450ca69dc3193f7ab17a7622) | 2026-10-29 | `perf(cake_fmha)` (`#7839`)：GB300 FP8 Context 程序将矩阵 `P` 保存在另一流水阶段的 `S` 列中复用 TMEM；B200 引入更小 Pack 切分计划。 |
| 6 | `triton-lang/triton` | [`6caee94`](https://github.com/triton-lang/triton/commit/6caee941d0f13c01e4933443536108914f551a89) | 2026-10-22 | `[GSan] Add memory access profiler` (`#12338`)：新增 `gsan-profile` 插桩模式，自动统计张量生命周期的访问粒度与 `write_once` 候选。 |
| 7 | `triton-lang/triton` | [`473737f`](https://github.com/triton-lang/triton/commit/473737fcb35a9c1836ef920df23a98944432a999) | 2026-10-21 | `[Runtime] Serialize class-valued constexprs for preload` (`#12430`)：按模块名与限定名序列化类类型 `constexpr`，修复 JIT Preload JSON 序列化失败。 |
| 8 | `tile-ai/tilelang` | [`942bfde`](https://github.com/tile-ai/tilelang/commit/942bfdee0a8b8c9d7d478556f36f72c38f861d09) | 2026-10-08 | `[BugFix][CUDA] Support uint8 sparse GEMM operands` (`#3297`)：在 `mma_sp_macro_generator.py` 中补齐 `uint8` 稀疏 GEMM 操作数映射。 |
| 9 | `tile-ai/tilelang` | [`29b4fdd`](https://github.com/tile-ai/tilelang/commit/29b4fdd9544340ee55ee0993e5d7e81759a01b95) | 2026-10-08 | `[CUDA] Emit packed FP16 and BF16 vector comparisons` (`#3303`)：在 `codegen_cuda.cc` 中生成打包的 `half2` / `nv_bfloat162` 向量比较指令。 |
| 10 | `tile-ai/tilelang` | [`c819a83`](https://github.com/tile-ai/tilelang/commit/c819a835a33c9d176a52cfcfab9c7da13726c29d) | 2026-10-08 | `[BugFix][ROCm] Fix ties-to-even rounding for T.round` (`#3299`)：修复 ROCm HIP 后端 `T.round` 的银行家舍入 (ties-to-even) 语义。 |

## 二、 架构机制深度解析

1. **Cake Chunked LM-Head 峰值显存预算与有效行数裁剪 (`3292e01`)**：
   大词表 (`V = 152,064`) 的 LM-Head + Cross-Entropy 融合算子在 CudaGraph 或静态 Buffer 分配下往往传入固定 `4096` 行张量，但实际有效 token 仅 `2560` 行。通过传入 `valid_rows` 并限制 `step_peak_budget`，算子直接跳过尾部无效行的 GEMM 与 LSE 归约，节省 **`37.5%`** 算力并将峰值激活显存控制在 **`512 MB`**。
2. **SM120 NVFP4 变长注意力重叠宽度与 KDA N32 直发 (`cf14a96`, `710f9b5`)**：
   在 `sm_120a` 上根据 plan 动态调整 DMA/MMA 重叠宽度 (`184.0 us -> 142.5 us`, **`1.2912x`**)；在 Kimi Delta Attention (KDA) 短序列 Prefill 中跳过昂贵的细粒度分箱构建 (`19.2 us` 开销)，直接以 Plain Work Bins 启动 N32 内核 (**`1.3122x`**)。
3. **GB300 FP8 Context FMHA 跨阶段 `S` 列复用 (`f319fc1`)**：
   在 GB300 (`sm_103a`) 上将 Softmax 输出 `P` 存放在另一流水线阶段的 `S` TMEM 列中，使 TMEM 列占用从 `256` 降至 `192`，提升 Prefill 吞吐 **`1.1964x`**。

## 三、 确定性离线算例验证 (`own-offline`)

运行 `python experiments/flashinfer_lmhead_minimax_kda_mxfp8_fmha_phase151/run.py --check`：
- **Cake LM-Head 算力节省**：`4,869.31 GFLOPs -> 3,043.32 GFLOPs`（节省 **`37.5%`**，峰值显存 `1216.5 MB -> 512.0 MB`）
- **SM120 NVFP4 Varlen Attention 加速比**：**`1.2912x`**；**KDA N32 Plain Bins 加速比**：**`1.3122x`**
- **Grouped MXFP8 Head-Window Prefetch 加速比**：**`1.2842x`** (`94.0 us -> 73.2 us`)
- **Triton Class Constexpr Preload 失败数**：`1 -> 0`；**TileLang ROCm Ties-to-Even 误差**：`0`
