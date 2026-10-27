# Phase 112 Track A：FlashInfer Rubin SM107 MegaMoE 掩码路由 NaN 暂存、SM120 MXFP8×MXFP4 TMA 128B L2 提升、Cake 256×448 MXFP8 FC1 与 CuTe IR / TileLang 流水线（2026-10-28）

> **英文对照 / English Version**: [`flashinfer-rubin-megamoe-mxfp8-l2-cutlass-cuteir-tilelang-phase112.en.md`](./flashinfer-rubin-megamoe-mxfp8-l2-cutlass-cuteir-tilelang-phase112.en.md)  
> **7-Card 4:5 视觉评测报告**: [`../reports/ai-infra-phase112-benchmark-cards.html`](../reports/ai-infra-phase112-benchmark-cards.html)  
> **可复现实验目录**: [`../../experiments/flashinfer_rubin_megamoe_mxfp8_l2_cutlass_cuteir_tilelang_phase112/`](../../experiments/flashinfer_rubin_megamoe_mxfp8_l2_cutlass_cuteir_tilelang_phase112/)

---

## 1. 证据分级与一手代码定位（10 项全栈算子与编译器提交）

| # | 仓库与 Commit | 日期 | 证据等级 | 关键代码路径与核心修复/优化机制 |
|---|---|---|---|---|
| 1 | [`flashinfer@7bce78b15d`](https://github.com/flashinfer-ai/flashinfer/commit/7bce78b15d6462e054bbd565fbadaf4daa65ac0d) | 2026-10-27 | `implementation` | `flashinfer/moe_ep/backends/mega/kernel/sm107/{routing_staging,routing_validation}.py`：Rubin (`sm_107a`) MegaMoE 允许掩码路由 (`expert == -1`) 携带 `NaN`/`±inf` 分数并通过 `RouteStager` 置零写入工作区，不改动调用方张量；新增 `FLASHINFER_SM107_MEGA_CHECK_ROUTES=0` 开关 |
| 2 | [`flashinfer@a1fbb4e0c4`](https://github.com/flashinfer-ai/flashinfer/commit/a1fbb4e0c4703c65a108bff1a17be5067a72adf0) | 2026-10-26 | `author-benchmark` / `implementation` | `flashinfer/fused_moe/backends/cudnn_frost/artifacts/mxfp8_mxfp4/sources/*.py`：SM120 cuDNN FROST `MXFP8 × MXFP4` 启用 TMA `TensorMapL2Promotion.l2_128b`，解决打包 E2M1 64B K-row 独立 DRAM 请求瓶颈（RTX PRO 6000 带宽 `0.85 → 1.19 TB/s`，耗时 `343 → 253 µs`） |
| 3 | [`flashinfer@fd0e4fdd96`](https://github.com/flashinfer-ai/flashinfer/commit/fd0e4fdd96b046b1433a44e9eec8f0c3d8fe9e70) | 2026-10-25 | `author-benchmark` / `implementation` | `csrc/cake_minimax_h3_mlp_sm10x.cu`、`flashinfer/diffusion_ops/minimax_h3_mlp.py`：Cake MiniMax-H3 MLP 新增 `256 × 448` MXFP8 FC1 错位累加器半区 (`staggered accumulator halves`)，N-tiles 从 `112` 降至 `64`（B200 `1.023x–1.060x`、B300 `1.033x–1.068x`） |
| 4 | [`flashinfer@cc8552a3a0`](https://github.com/flashinfer-ai/flashinfer/commit/cc8552a3a0158391f514571597b3b495090ce43c) | 2026-10-25 | `implementation` | `flashinfer/experimental/cake_qsa_nvfp4/{cake_backend,cake_jit}.py`：为 Rubin SM107 (`sm_107a`) 导出 Cake QSA NVFP4 稀疏分页 GQA decode `tcgen05` 算子，并在所有主机架构上保持 split (`decode + merge`) 形态 |
| 5 | [`flashinfer@4796d2effd`](https://github.com/flashinfer-ai/flashinfer/commit/4796d2effd9ea8511c32ffcd21609998c2896bc0) | 2026-10-25 | `author-benchmark` / `implementation` | `csrc/cake_minimax_h3_sm120_quant_out_proj_sm120a.cu`：SM120 NVFP4 输出投影在设备端以 FP64 计算 `alpha = 1 / (act_global_scale * o_weight_global_scale)`，消除每次调用 2 次主机同步，连续调用 GPU 空闲间隙从 `186.9–223.3 µs` 降至 `3.1–3.6 µs` |
| 6 | [`flashinfer@6185549792`](https://github.com/flashinfer-ai/flashinfer/commit/618554979285eaf2fa803e6803e1ccb014efd957) | 2026-10-27 | `implementation` | `flashinfer/{decode,prefill,sparse}.py`：`plan()`、`fast_decode_plan()` 与 `workspace_size()` 入口对 `indptr`/`indices`/`last_page_len`/`seq_lens` 强制 `.contiguous()` 归一化，消除非连续跨步视图引起的 CUDA/CPU 布局分裂与非法内存访问 (IMA) |
| 7 | [`cutlass@9d79de097b`](https://github.com/NVIDIA/cutlass/commit/9d79de097be048b67be8e527ceced4ba017e5e1d) | 2026-08-03 | `implementation` | `cutlass_compiler/README.md`：引入 CUTLASS Compiler `CuTe IR` MLIR 方言栈（`cute` 布局代数方言 + `Base` 目标伪方言边界 + `cute-to-base` 下沉通道） |
| 8 | [`cutlass@f94ec46f4f`](https://github.com/NVIDIA/cutlass/commit/f94ec46f4f63f96003d6cfdf2014731e7672c281) | 2026-07-29 | `implementation` | `include/cutlass/gemm/kernel/sm100_tile_scheduler_group.hpp`：移除 `PersistentTileSchedulerSm100Group` 中残留的私有 `load_query_response` 死代码包装 |
| 9 | [`tilelang@fc0fbf0e50`](https://github.com/tile-ai/tilelang/commit/fc0fbf0e50de55c8e3f04b4954c2ab1000b8a298) | 2026-10-16 | `implementation` | `src/transform/inject_pipeline.cc`：修复兄弟 `T.Pipelined` 循环共享同一 `alloc_shared` 缓冲区但使用非对称 `num_stages`（如 `2` 与 `4`）时的二次维度膨胀（2D→3D→4D 崩溃），通过 `PipelineBufferReconciler` 统一为 `max(stages)` 3D 分配 |
| 10 | [`tilelang@51bdf5467e`](https://github.com/tile-ai/tilelang/commit/51bdf5467e560de35fdc91e88821549480654180) | 2026-10-16 | `author-benchmark` / `implementation` | `src/ascend/transform/ascend_simdvf_lower_parallel.cc`：Ascend `SimdVF` `dim=1` 归约将写回指令从全向量 64-lane `NORM_B32` (`dst[rep * vreg_size_]`) 修正为单元素 `ONEPT_B32` (`vsts_1st` at `dst[rep]`)，消除跨行越界写与首行零覆盖 |

---

## 2. 核心架构机制与数学形式化深挖

### 2.1 Rubin SM107 MegaMoE 掩码路由非有限分数容忍与工作区暂存 (`flashinfer@7bce78b15d`)

在 Rubin (`sm_107a`) MegaMoE 中，上游路由算子对被掩码的槽位 (`expert == -1`) 常保留 `NaN` 或 `±inf` 分数。旧版 `routing_validation.py` 对所有槽位无条件执行有限性断言：

$$\text{valid}_{\text{legacy}} = (-1 \le e < E) \land \text{isfinite}(s) \land \neg\text{duplicate}(e)$$

修复后，校验器将有限性约束限定在有效激活路由上：

$$\text{valid}_{\text{fixed}} = (-1 \le e < E) \land \big((e = -1) \lor \text{isfinite}(s)\big) \land \neg\text{duplicate}(e)$$

同时，新增 CuTe-DSL `RouteStager` 内核将预量化输入（NVFP4 / MXFP8）的 `topk_ids` 与 `topk_weights` 拷贝至 MegaMoE 工作区：当 $e = -1$ 时将工作区权重写成 `0.0f`，**绝不原地修改调用方传入的张量**，从而保证 `late-weight combine` 归约不会因 $0 \times \text{NaN} = \text{NaN}$ 污染输出。

### 2.2 SM120 cuDNN FROST `MXFP8 × MXFP4` TMA 128B L2 提升 (`flashinfer@a1fbb4e0c4`)

在 SM120 (`sm_120a`, RTX PRO 6000 Blackwell) 的 `MXFP8 × MXFP4` (W4A8) 分组 MoE GEMM 中，每个 FP8 K-tile 包含 $K_{\text{tile}} = 128$ 个元素；对应打包 E2M1 权重的单行盒宽度仅为：

$$B_{\text{row, E2M1}} = 128 \times 0.5\text{ B} = 64\text{ B}$$

当 TMA 描述符未设置 L2 Promotion（cuDNN Frontend DSL 默认 `none`）时，每个 64B 盒行都会发起独立的 64B DRAM 事务，导致显存带宽被限制在 $0.85\text{ TB/s}$（12 专家 FC1 `7168×3072`、16 tokens 耗时 $343\text{ }\mu\text{s}$）。为 A/B TMA 数据描述符显式启用 `l2_promotion = TensorMapL2Promotion.l2_128b` 后，硬件自动将相邻 64B 提升为 128B 扇区合并读取，实测带宽跃升至 $1.19\text{ TB/s}$（$+40.0\%$），耗时降至 $253\text{ }\mu\text{s}$（**$1.3557\times$ 加速**）。

### 2.3 Cake MiniMax-H3 `256×448` MXFP8 FC1 与 SM120 NVFP4 设备端双全局缩放 (`flashinfer@fd0e4fdd96`, `flashinfer@4796d2effd`)

1. **`256×448` 错位累加器半区 (`fd0e4fdd96`)**：MiniMax-H3 中间层维度 $I = 14336$（SwiGLU gate+up 共 $28672$ 列）。2-CTA 对若采用 `256×256` tile（每对产出 $128$ 列 $y$），需 $N_{\text{tiles}} = 14336 / 128 = 112$ 个 N-tile；改用 `256×448` tile（112-row 权重缩放子块，每对产出 $224$ 列 $y$）并将 TMEM 累加器分为上下错位半区后，N-tile 数量减少至 $14336 / 224 = 64$（**减少 $42.86\%$**），在 B200 上取得 $1.023\times\text{–}1.060\times$（$M=4816$ 时 $0.9300\text{ ms} \to 0.8773\text{ ms}$，**$1.0601\times$**）、B300 上取得 $1.033\times\text{–}1.068\times$ 整块加速。
2. **SM120 NVFP4 输出投影消除主机同步 (`4796d2effd`)**：旧实现通过 `.item()` 在主机端读取 `act_global_scale` 与 `o_weight_global_scale` 计算反量化因子 $\alpha$，每次调用触发 2 次流同步，导致连续调用间出现 $186.9\text{–}223.3\text{ }\mu\text{s}$ 的 GPU 空闲泡沫；新内核直接在设备端按相同 FP64 公式计算并单次舍入为 FP32：

$$\alpha = \text{ float32}\!\left(\frac{1.0}{\text{float64}(s_{\text{act}}) \times \text{float64}(s_{W})}\right)$$

在保持逐比特一致的同时，将连续调用间的 GPU 空闲间隙压缩至 $3.1\text{–}3.6\text{ }\mu\text{s}$（中位数 **下降 $61.22\times$**）。

### 2.4 非连续索引归一化、CuTe IR MLIR 方言与 TileLang 兄弟流水线 / Ascend 归约 (`6185549792`, `9d79de097b`, `fc0fbf0e50`, `51bdf5467e`)

- **FlashInfer `plan()` 跨步视图归一化 (`6185549792`)**：当调用方传入跨步视图（如 `indptr_full[::2]`）时，CUDA 张量的 `.to(device)` 保持跨步，而 `.to("cpu")` 生成连续副本，导致主机规划器与 GPU 内核读取不同的物理布局（引发 FA2 IMA 或 FA3 误差高达 $4.6$）。在 `plan()` / `fast_decode_plan()` / `workspace_size()` 顶部统一执行 `.contiguous()` 彻底消除了该隐患。
- **CUTLASS `CuTe IR` MLIR 方言 (`9d79de097b`)**：建立 `cute` 方言到 `Base` 伪方言边界（`arith`/`scf`/`cf`/`ub`/`func`/`math`/`gpu`/`LLVM`/`NVVM`）的 `cute-to-base` 编译器分层。
- **TileLang 兄弟流水线共享缓冲区对账 (`fc0fbf0e50`)**：两个兄弟 `T.Pipelined` 循环（`num_stages = 2` 与 `4`）共享同一 `alloc_shared((BM, BK))` 时，旧版 `RewritePipeline` 将已扩为 `(2, BM, BK)` 的缓冲区二次扩为 4D `(4, 2, BM, BK)` 触发 `LayoutInference` 崩溃；新增 `PipelineBufferReconciler` 追踪 `pending_buffer_remap_` 并将其对账为单一 3D `(\max(2,4), BM, BK) = (4, 64, 32)` 分配。
- **TileLang Ascend `SimdVF` `dim=1` 单元素写回 (`51bdf5467e`)**：`vcadd` 将每行归约至向量寄存器 lane 0，旧版使用 `NORM_B32` 在 `dst[rep * 64]` 写入全部 64 个 lane，既用零覆盖了第 0 行后续输出（产生 `[-1.0672, 0, 0, 0]`），又使第 1..3 行越界写入偏移 `[64, 128, 192]`；改为 `ONEPT_B32` (`vsts_1st`) 写回 `dst[rep]` 后，Ascend 950PR 实测 `A(4,64)` 误差为 `0.0`。

---

## 3. 本地确定性复现实验结果 (`own-offline`)

运行 `python experiments/flashinfer_rubin_megamoe_mxfp8_l2_cutlass_cuteir_tilelang_phase112/run.py --check` 得到的密封结果（[`summary.json`](../../experiments/flashinfer_rubin_megamoe_mxfp8_l2_cutlass_cuteir_tilelang_phase112/summary.json)）：

| 检验维度 | 旧路径 / 基线表现 | 修复/优化后表现 | 关键量化指标 |
|---|---|---|---|
| Rubin MegaMoE 掩码路由校验 | `legacy_check_passed = False` | `fixed_check_passed = True` | 掩码权重全置 `0.0`，调用方张量零修改 |
| SM120 `MXFP8 × MXFP4` TMA L2 Promotion | `0.85 TB/s`, `343.0 µs` (`none`) | `1.19 TB/s`, `253.0 µs` (`l2_128b`) | 带宽提升 **`1.40x`**，延迟加速 **`1.3557x`** |
| Cake MiniMax-H3 `256×448` MXFP8 FC1 | `112` N-tiles (`0.9300 ms` @ M=4816) | `64` N-tiles (`0.8773 ms` @ M=4816) | N-tiles 减少 **`42.86%`**，加速 **`1.0601x`** |
| SM120 NVFP4 Out-Proj 设备端缩放 | 每次调用 2 次同步，间隙 `205.1 µs` | 设备端 FP64 算 $\alpha$，间隙 `3.35 µs` | 连续调用 GPU 空闲间隙缩小 **`61.22x`** |
| TileLang 兄弟流水线 & Ascend `dim=1` 归约 | 4D 膨胀崩溃 / 越界偏移 `[64, 128, 192]` | 3D `(4, 64, 32)` / `ONEPT_B32` `dst[rep]` | Ascend `A(4,64)` 最大绝对误差 **`0.0`** |

---

## 4. 解决方案架构师行动指南与可证伪问题

1. **架构师检查清单**：
   - 在 Rubin (`sm_107a`) 部署 MegaMoE 时，若上游保证路由索引合法，可在热身与 CUDA Graph 捕获前设置 `FLASHINFER_SM107_MEGA_CHECK_ROUTES=0` 跳过值校验，同时保留掩码 `-1` 槽位的 `NaN` 清零保护。
   - 在 Blackwell 工作站/服务器 (`sm_120a`) 上编写混合精度 `MXFP8 × MXFP4` TMA 内核时，凡 K-tile 盒行宽度为 64B 的一侧必须开启 `TensorMapL2Promotion.l2_128b`。
2. **后续可证伪问题**：
   - 当 `num_tokens` 扩展至 $>128$ 时，SM120 `MXFP8 × MXFP4` 的 `_swap_ab` 变体在开启 `l2_128b` 后是否仍受限于共享内存容量而非 L2 填充带宽？
