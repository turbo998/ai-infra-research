# FlashInfer SM107a SSD 精确扫描、MXFP8 8 区间分派、TinyGEMM2 L2 256B 提升、Kimi-K3 路由、Hopper VSA/QSA/KDA 与 TileLang Ascend 流水线（Phase 119 / Priority 128）

> **证据分类**：Upstream `implementation` + `author-benchmark`（10 个上游公开提交，2026-10-28）+ 本仓库 `own-offline` 确定性算术/拓扑验证（`experiments/flashinfer_ssd_combined_gated_act_mxfp8_tinygemm2_kimi_vsa_qsa_kda_tilelang_phase119/run.py`）。本文不包含未标注的真实 GPU 硬件重测声明。

## 1. 核心架构问题与背景

在 Blackwell (`SM100a`)、Blackwell Ultra (`SM103a`)、Rubin 过渡架构 (`SM107a` / `R200`)、Hopper (`SM90a`) 以及昇腾 Ascend NPU 的最新内核工程中，暴露出五类深层微架构与编译瓶颈：

1. **SM107a (`R200`) SSD 精确扫描与 KDA 反向尾部的双内核启动开销**：`cake_ssd_combined` 在 FP32 状态单块（超项界）及保留无环（retained no-ring）路径上存在冗余内核拆分，而 `cake_kda` 在 `SM107a` 的反向传递仍将 `glm_intra` 与 `gate_epilogue` 分成两次启动并占用额外的 `dg_int` / `db_int` 缓冲区。
2. **Gated-Act MXFP8 跨 $M \times K$ 规模的寄存器溢出与驻留退化**：在 `SM100a`/`SM103a` 上，单一 Tile 与固定 CTA 驻留配置无法同时兼顾 $< 2^{20}$ 的小规模低延迟与 $\ge 2^{26}$ 的高带宽，且 `SM103a` 反向按行（rowwise）程序在 12 CTAs/SM 下触发 8 字节局部栈溢出（Stack Spill）。
3. **TinyGEMM2 窄行矩阵乘的 L2 命中与 17–23 行 Tile 利用率不足**：在 `SM107a` 上窄 Tile TMA 映射未开启 256B L2 提升，且无偏置 17–23 行 GEMM（超越 16 行单块上限但远未填满 32 行块）因未走打包 Tile（Packed-Tile）路由而浪费 Tensor Core 吞吐。
4. **Kimi-K3 融合 MoE 路由器（896 专家 Top-16）跨容量边界的同步等待与排序开销**：3–32 tokens 的 16-CTA 集群程序在入口过早等待集群屏障；超过设备容量 $C$（GB300 为 592，B200 为 568）及 $2C$ 后的多行/每 Warp 路由对排序存在跨 CTA 共享内存竞争。
5. **TileLang Ascend 手工流水线配置寄存器覆写与 Bisheng 编译器升级缓存碰撞**：昇腾硬件配置寄存器（`HF32`、`MMAD` 方向、`PadValue`、`AtomicMode`）在跨 Stage 流水线中共享单一物理寄存器，缺乏周期级相位约束；且 `.aibin` 设备二进制缓存在 Bisheng 工具链升级但 Clang 版本号不变时发生缓存键碰撞。

---

## 2. 上游一手实现剖析（10 个零重叠 Commit）

| # | 仓库与提交 | 日期 | 核心实现路径与微架构机制 |
|---|---|---|---|
| 1 | `flashinfer-ai/flashinfer@372e9435f4` (`#7558`) | 2026-10-28 | `flashinfer/experimental/cake_ssd_combined/`：重新生成 17 个 `sm_107a` 程序（`exactpre` + `exactfused`），将超项界的单块 FP32 状态调用重路由至两内核 `exactpre`、将保留无环调用重路由至单内核 `exactfused`。R200 (`sm_107a`, 216 SMs) 上全部 44 个变更/重路由形状均加速（`nem_v8x64_float32` 达 **1.012x–1.562x**，`perf_varlen_128x32` 从 `449.52 us` 降至 `348.32 us` 即 **1.291x**），93 个形状总几何平均加速 **1.0487x**。 |
| 2 | `flashinfer-ai/flashinfer@1d74db1e9d` (`#7568`) | 2026-10-28 | `flashinfer/cake_gated_act_mxfp8/`：将 `SM100`/`SM103` 分派扩展为 8 个按架构划分的 $M \times K$ 区间（`sm_100a` 5 个、`sm_103a` 3 个，总程序数 `45 -> 50`）：$< 2^{20}$ 采用 `32x32` Tile（12 CTAs/SM）；`sm_100a` 在 $M \times K \ge 2^{26}$ 前向列方向启用 5 CTAs/SM Band-Path；`sm_103a` 反向行方向 $\ge 2^{26}$ 用 10 CTAs/SM 打包 `f32x2` Sigmoid 替代 12 CTAs/SM（消除 `8 B -> 0 B` 栈溢出），变更形状实现 **1.07x–1.74x** 加速。 |
| 3 | `flashinfer-ai/flashinfer@efa2600909` (`#7588`) | 2026-10-28 | `csrc/cake_tinygemm2_tcgen05/`：在 `sm_107a` 窄 Tile 权重与激活 TMA 描述符上启用 `CU_TENSOR_MAP_L2_PROMOTION_L2_256B`（R200 上稳定提速 **1.02x**，如 `1x128x14336` 从 `5.31 us` 降至 `5.22 us`）；并将无偏置 `17..23` 行 GEMM（输出 `1536..4096`、$\text{out}\times K \ge 6\times 10^6$、激活 $< 128\text{ KiB}$）路由至 Packed-Tile，78 个形状加速 **1.10x–1.46x**。 |
| 4 | `flashinfer-ai/flashinfer@95ab86637c` (`#7567`) | 2026-10-28 | `flashinfer/experimental/kimi_k3_fused_router/`：为 896 专家 Top-16 融合路由器引入行选择后延迟等待集群的 16-CTA 程序 `LCG_LW`、单行带下一行预取的 `Q4SB`（GB300 $C=592$ 至 1024 tokens）以及在 Owner CTA 自身共享内存中排序路由对的 `Q4SPW_LR`，GB300 与 B200 在 256 tokens 下达 **1.05x** 加速（`6.02 -> 5.73 us` / `6.24 -> 5.95 us`）。 |
| 5 | `flashinfer-ai/flashinfer@acce856763` (`#7580`) | 2026-10-28 | `flashinfer/cake_vsa_sm90_backend/`：流水化 Hopper (`sm_90a`) 变长块稀疏注意力（VSA）分组查询（GQA）程序：在 Sage-FP8 共享转置队列程序中拆分 K/V 就绪等待并将 K-scale 与下一队列字提前一拍加载，在 BF16 紧凑程序中使尾对 Tile 的最后 PV 等待及输出写回与下一 Tile 开头重叠。H100 上 37 个变更行几何平均加速 **1.0236x**（FP8 队列 29 行几何平均 **1.0274x**，`109632` 行从 `3869.0 us` 降至 `3767.2 us`）。 |
| 6 | `flashinfer-ai/flashinfer@3055feed86` (`#7578`) | 2026-10-28 | `flashinfer/experimental/cake_qsa_nvfp4/`：在 `SM103a` 持久化遍历解码程序中启用从尾声直接写出 Split 部分和（GB300 128 行加速 **1.03x**，`64.68 -> 62.94 us`），并在 `SM107a` 上统一启用精确 Warp-Redux Softmax 最大值、提前块表加载与直接部分和尾声（R200 64 行 NHD 加速 **1.06x**）。 |
| 7 | `flashinfer-ai/flashinfer@b17b12b2e5` (`#7576`) | 2026-10-28 | `flashinfer/experimental/cake_kda_train/`：在 `SM107a` (`R200`) KDA 分块训练反向中启用门控融合内块内核 `glm_intra_v18`（将全部命名屏障槽位编码为立即数，使 2 个 CTA 共享单个 SM），省去独立 `gate_epilogue` 启动与 `dg_int`/`db_int` 缓冲区。R200 反向提速 **1.0306x–1.0481x**（`P14 H=64, S=261393` 从 `33.5401 ms` 降至 `32.0011 ms`）。 |
| 8 | `flashinfer-ai/flashinfer@8cf7089cf6` (`#7583`) | 2026-10-28 | `csrc/cake_trtllm_moe_allreduce_union/`：在 `SM100`/`SM103` Two-Shot 融合程序中，利用等待 RMS 交换的窗口在循环内每步清理 2 个上一轮 Workspace 包（写入哨兵值 `0x80008000`），并在 `1536..2048` tokens 下提前 2 个 Token 预取 Reduce-Scatter（B200 提速 **1.017x–1.062x**，B300 提速 **1.026x–1.060x**）。 |
| 9 | `tile-ai/tilelang@dfe83e84ec` (`#3624`) | 2026-10-28 | `tilelang/ascend/transform/z3_scheduler.py`：在昇腾手工偏移流水线调度器中追踪硬件配置寄存器（`kHf32Mode`、`kMmadDirection`、`kPadValue`、`kAtomicMode`）读写效应，并向 Z3 求解器注入相位约束 $r[v] + \text{wrap} \cdot II \ge r[u] + \text{gap}$（$\text{wrap} \in \{0, 1\}$），防止跨 Stage 提前覆写。 |
| 10 | `tile-ai/tilelang@1c0f402e16` (`#3691`) | 2026-10-28 | `tilelang/contrib/bisheng.py` & `tilelang/cache/ascend_binary_cache.py`：每进程查询并缓存 `bisheng --version` 首行（含编译时间戳与 Clang 修订号，忽略 `InstalledDir`），写入 `.aibin` 缓存键的 `compiler` 字段，隔离具有相同 Clang 版本号的不同 CANN 构建。 |

---

## 3. 关键公式、数据移动与离线验证结果

### 3.1 Ascend 手工流水线配置寄存器相位约束与 MXFP8 栈溢出消除
1. **昇腾配置寄存器物理回绕约束**：设手工流水线启动间隔为 $II$，由于每类硬件配置寄存器（如 `PadValue`）在物理上跨所有流水线 Stage 共享单一实例，若逻辑 Stage 中的读取操作 $u$ 必须在下一轮物理写入 $v$ 之前完成消费，则 Z3 周期偏移变量 $r[\cdot]$ 必须满足：
   $$r[v] + \text{physical\_wrap} \cdot II \ge r[u] + \text{gap}, \quad \text{physical\_wrap} \in \{0, 1\}$$
2. **SM103a MXFP8 反向行方向寄存器拟合**：当 $M \times K \ge 2^{26}$（$67,108,864$）时，将每 SM 驻留从 12 CTAs 降至 10 CTAs 并采用打包 `f32x2` Sigmoid 计算，将每线程局部栈溢出从 `8 B` 降至 `0 B`，彻底消除高带宽区间的 Local Memory 往返流量。

### 3.2 离线模型验证汇总（`run.py --check`）

- `ssd_sm107a_regenerated_programs`: `17`（56 个总程序中更新 17 个 `sm_107a` 程序，44 个变更形状最高加速 `1.562x`，93 个形状几何平均 `1.0487x`）
- `mxfp8_total_mk_ranges`: `8`（`sm_100a` 5 区间 + `sm_103a` 3 区间，栈溢出 `8 B -> 0 B`，最高加速 `1.74x`）
- `tinygemm2_l2_promotion_bytes`: `256`（`sm_107a` 256B L2 提升 + `17..23` 行 Packed-Tile 路由最高加速 `1.46x`）
- `kimi_gb300_capacity_c`: `592`（152-SM GB300 4-CTA 集群容量界，256 tokens 加速 `1.0506x`）
- `vsa_changed_geomean_speedup`: `1.0236`（37 个变更行几何平均加速 `1.0236x`，FP8 队列 29 行达 `1.0274x`）
- `kda_sm107a_ctas_per_sm`: `2`（`glm_intra_v18` 立即数命名屏障实现 2 CTAs/SM，`P14` 反向加速 `1.0481x`）
- `ascend_reg_phase_constraint_satisfied`: `True`（Z3 配置寄存器周期相位约束验证通过）
- `bisheng_builds_separated` / `bisheng_relocation_invariant`: `True` / `True`（区分不同构建时间戳且对安装目录重定位保持缓存命中）

---

## 4. 面向云解决方案架构师的练习与可证伪问题

1. **计算练习**：在 `cake_tinygemm2_tcgen05` 的 `17..23` 行无偏置 Packed-Tile 路由中，准入条件要求激活工作集 $\text{batch} \times K \times 2\text{ B} < 128\text{ KiB}$（$131,072\text{ B}$）且 $\text{out} \times K \ge 6,000,000$。当 $\text{batch} = 22, \text{out} = 2048$ 时，求满足准入条件的合法 $K$ 范围（以 128 为步长）。（答案：由 $\text{out} \times K \ge 6\times 10^6$ 及 22 行起始门限得 $K_{\min} = 2944$（或输出更宽时从 $K=2816$ 起）；由 $22 \times K \times 2 < 131072$ 得 $K \le 2978$，故在 $\text{out}=2048$ 下 $K = 2944$；当 $\text{out}=4096$ 时合法范围为 $K \in \{2816, 2944\}$）。
2. **可证伪假设**：为何 TileLang Ascend 设备二进制缓存不能仅用 `clang version 15.0.5` 和 Git revision 作为编译器标识？因为不同 CANN 发行版（如 CANN 9.1 与 9.2 补丁版本）中的 Bisheng 编译器可能报告完全相同的 Clang 版本号与 revision，仅首行中的构建时间戳（如 `2026-07-01T09:55:44+08:00` vs `2026-08-27T22:23:38+08:00`）不同；保留首行构建时间戳并剔除 `InstalledDir`，既能避免升级后的脏二进制复用，又能保证编译器目录迁移时不触发无效重编译。
