# FlashInfer 统一 `frost_low_latency` MoE (`SM100/SM103/SM107`)、`cake_rmsnorm_train` 行窗口与固定序树归约、`cake_sage` 提前一步 Softmax 元数据、`sm_107a` 回溯块并行 SSD 与 `sm_90a` 紧凑静态 VSA 深度解析

> **语言 / Language**: [中文](flashinfer-frost-low-latency-moe-cake-rmsnorm-sage-vsa.md) | [English](flashinfer-frost-low-latency-moe-cake-rmsnorm-sage-vsa.en.md)  
> **Phase**: Phase 49 (`priority: 58`) · **日期**: 2026-10-22  
> **上游一手来源**:
> - `flashinfer-ai/flashinfer@2995461e872e030b364a1a84d8ca94181f857876` (`#6626`, 2026-10-21)
> - `flashinfer-ai/flashinfer@d892cd4197149d60c0e86ca0026a3df9dfe111fb` (`#6939`, 2026-10-21)
> - `flashinfer-ai/flashinfer@c21f621ae75ed3cf28cae7745281533e1bf78065` (`#6951`, 2026-10-21)
> - `flashinfer-ai/flashinfer@7601d0bb7e92390aae5276951f0e4b7c8a2b5c14` (`#6943`, 2026-10-21)
> - `flashinfer-ai/flashinfer@7808983872ee3961f142bce3227d7ecc2afa7253` (`#6949`, 2026-10-21)

---

## 1. 背景与核心问题

在 Blackwell (`sm_100a` B200 / `sm_103a` B300/GB300)、Rubin (`sm_107a` R200) 与 Hopper (`sm_90a` H100) 的最新算子演进中，FlashInfer 针对低延迟 MoE 解码、训练态 RMSNorm 确定性梯度归约、FP8 块稀疏注意力、Mamba2 SSD 状态扫描以及 Hopper 变长块稀疏注意力（VSA）落地了五项关键突破：

1. **统一 `frost_low_latency` MoE 后端 (`#6626`)**：原先 Frost 低延迟 MoE 拆分在 `frost_bf16_moe` 与 `frost_mxfp4_moe` 两个独立后端，不支持 `top-k > 8` 或非 SwiGLU 激活，且 Tensor Core 与标量 FMA 路径各自维护不兼容的预处理权重布局。
2. **`cake_rmsnorm_train` 行窗口反向与固定序树归约 (`#6939`)**：当反向通过多个持久 CTA 分担行区间时，原子累加 `dweight` 存在非确定性浮点舍入，且跨步栅栏（barrier）若依赖外部清零会引入额外 launch 开销。
3. **`cake_sage` Softmax 元数据提前一步流水 (`#6951`)**：Sage-FP8 块稀疏注意力在每个 Softmax 步都等待依赖链式的全局内存块索引与每 16-token `K` 缩放因子加载，而 CTA 内有两个 Warp 处于空闲状态。
4. **`cake_ssd_combined` `sm_107a` 回溯块并行 SSD (`#6943`)**：Rubin (`sm_107a`) 上 Mamba2 SSD combined chunk-scan 的回溯（look-back）块并行路径（`bf16/f16/f32 × batched/varlen` 共 6 个程序）仍有显著指令调度与访存余量。
5. **`cake_vsa_sm90` 持久核与紧凑静态程序 (`#6949`)**：Hopper (`sm_90a`) 变长块稀疏注意力在短 tile、高 `tiles/CTA` 的均匀对计划（uniform pair plans）下缺乏紧凑静态分发程序。

---

## 2. 架构与关键算子机制深挖

```
+---------------------------------------------------------------------------------------------------+
|         FlashInfer Phase 49 Kernel Stack (SM90a Hopper / SM100a-103a Blackwell / SM107a Rubin)     |
+---------------------------------------------------------------------------------------------------+
| 1. Unified frost_low_latency MoE (#6626)                                                          |
|    - FrostLowLatencyConfig(max_num_tokens=2, range 1..8)                                          |
|    - Shared MXFP4/MXFP8 prepared weights across TC128/TC256 (with PDL) + 16 FMA warp-width pairs  |
|    - 10 typed activations: SwiGLU, SiTU, GeGLU, GeGLUTanh, SwiGLUStep, GELU, ReLU, ReLU2, SiLU... |
+---------------------------------------------------------------------------------------------------+
| 2. cake_rmsnorm_train (#6939)                   | 3. cake_sage Softmax Metadata Staging (#6951)   |
|    - Row-window bwd (H=1536, 4096 on SM100-107) |    - 2 previously idle warps stage next step's  |
|    - Fixed-order tree dweight reduction         |      block metadata & per-16-token K scales     |
|    - Self-resetting flip barrier (no zero fill) |    - dynamic_smem_bytes: 119040 -> 120320 B     |
+---------------------------------------------------------------------------------------------------+
| 4. cake_ssd_combined sm_107a (#6943)            | 5. cake_vsa_sm90 18-Program Suite (#6949)       |
|    - 6 look-back chunk-parallel v3 programs     |    - 3 persistent (incl. compact static) +      |
|      (bf16/f16/f32 x batched/varlen on R200)    |      15 small-selection variants on H100        |
+---------------------------------------------------------------------------------------------------+
```

### 2.1 统一 `frost_low_latency` MoE 后端 (`#6626`)
- **统一配置与路由入口**：`FrostLowLatencyConfig(max_num_tokens=2)`（允许 `1 <= max_num_tokens <= 8`）取代了旧的 `frost_bf16_moe` / `frost_mxfp4_moe` 分离接口，并在 `fused_moe` 中当 `backend="auto"` 且请求满足 `1 <= num_tokens <= max_num_tokens` 时自动路由到 `frost_low_latency`。
- **共享预处理权重与双轨 Tactic 空间**：
  - 在 `SM100/SM103/SM107` 上，MXFP4/MXFP8 的 Tensor Core 路径（`TC128` 与 `TC256`，均启用 Programmatic Dependent Launch, PDL）与标量 FMA 路径（支持 `top-k <= 32`）**共享同一份 prepared weight 布局**。因此，FlashInfer 的 `AutoTuner` 可以在单次预热中同时探索 `2` 个 TC tactic 与 `16` 个 FMA tactic（`FC1/FC2` warp 输出宽度 `1, 2, 4, 8` 的笛卡尔积 `4 × 4 = 16`），总共 **18 个候选 tactic**（当 `top-k > 32` 时收敛为 2 个 TC tactic）。
  - BF16 FMA 路径扩展至 **10 种强类型激活函数**：`SwiGLU`、`SiTU`、`GeGLU`、`GeGLUTanh`、`SwiGLUStep`、`GELU`、`ReLU`、`ReLU2`、`SiLU`、`Identity`。
- **NVIDIA B300 (`sm_103a`) 实测端到端加速（对比 CUTLASS 基准）**：
  - **BF16**：`T=1, SwiGLU` 从 `27.736 µs` 降至 `15.460 µs`（**`1.794×`**）；`T=2, SwiGLU` 从 `44.144 µs` 降至 `31.872 µs`（**`1.385×`**）；`T=2, SiTU` 从 `60.424 µs` 降至 `46.056 µs`（**`1.312×`**）。
  - **MX Clamped-SwiGLU**：`W4/T1` 从 `21.616 µs` 降至 `13.440 µs`（**`1.608×`**）；`W4/T4` 从 `31.852 µs` 降至 `21.632 µs`（**`1.472×`**）；`W8/T4` 从 `40.008 µs` 降至 `31.816 µs`（**`1.257×`**）。

### 2.2 `cake_rmsnorm_train` 行窗口反向、固定序树归约与自复位翻转栅栏 (`#6939`)
- **行窗口（Row-Count Window）反向与确定性 `dweight`**：
  - 针对 `H=1536` 与 `H=4096`（覆盖 `sm_100a`、`sm_103a`、`sm_107a`），每个持久 CTA 负责一段连续行窗口并独立计算 `dx`（或融合残差梯度 `dx, du`）以及本地 `partial` 缓冲槽中的部分 `dweight`。
  - 所有 CTA 通过**自复位翻转栅栏（self-resetting flip barrier）**同步，无需在每次反向前发射清零计数器的辅助 kernel；随后按固定的 `chunk_id` 顺序执行**确定性树形归约（fixed-order tree reduction）**生成最终 FP32 `dw`，消除了原子加法导致的跨运行舍入漂移。
- **`sm_107a` 大行数前向 Bulk-Copy 生产者/消费者**：当 `H=1536` 且 `T > 12288` 时，Rubin (`sm_107a`) 切换至 bulk-copy producer/consumer 前向核，在 R200 上实现 **`1.035×–1.079×`** 前向加速；反向在 R200 上实现 **`1.007×–1.134×`** 加速（B200 为 **`1.010×–1.118×`**，GB300 为 **`1.008×–1.075×`**）。

### 2.3 `cake_sage`、`cake_ssd_combined` 与 `cake_vsa_sm90` (`#6951`, `#6943`, `#6949`)
- **`cake_sage` 提前一步 Softmax 元数据暂存 (`#6951`)**：在 `sm_100a` (B200) 与 `sm_103a` (GB300) 的 Sage-FP8 块稀疏注意力前向核（`head_dim=128`，64-token 块）中，利用原先空闲的 2 个 Warp 提前一步将下一 Softmax 步的块元数据与每 16-token `K` 缩放因子预取至共享内存（`dynamic_smem_bytes` 从 `119040 B` 增至 `120320 B`，净增 `1280 B`），输出 `O` 与 `LSE` 保持逐位一致（bit-identical），在 B200 (`A1: B8 H32 sel=6`) 上获得 **`1.08×`** 加速（`0.3552 -> 0.3288 ms`），GB300 获得 **`1.03×`** 加速（`0.2922 -> 0.2824 ms`）。
- **`cake_ssd_combined` `sm_107a` 回溯块并行 SSD (`#6943`)**：升级 Rubin (`sm_107a`) 上 6 个 `mamba_ssd_chunk_parallel_v3_{bf16,f16,f32}_{batched,varlen}` 程序，在 R200 7 个基准形状上取得 **`1.0215×` 调用级几何平均加速** 与 **`1.0839×` 主核几何平均加速**；其中 `perf_batched_1x256` 调用级从 `92.1 µs` 降至 `77.0 µs`（**`1.196×`**），`cp_b4h8c16` 主核从 `33.3 µs` 降至 `26.2 µs`（**`1.268×`**）。
- **`cake_vsa_sm90` 18 程序矩阵与紧凑静态程序 (`#6949`)**：为 Hopper (`sm_90a` H100 SXM) 提供 **18 个 VSA 程序**（3 个持久核：`static_tile_lists`、`device_tile_queue`、新增针对短 tile 高密度的 `compact_static`；加上 15 个小选择数变体：4 个 direct、6 个 DSM-cluster、2 个 split-KV、3 个 split-cluster）。在 H100 上，`ext33` 的 `h1-m512-n8192-k48` 从 `13.472 µs` 降至 `10.680 µs`（**`1.2624×`**），`real39` 的 `h8-m128-n32768-k64` 从 `35.360 µs` 降至 `28.912 µs`（**`1.2237×`**）。

---

## 3. 标准库确定性复现实验验证

通过 `experiments/flashinfer-frost-low-latency-moe-cake-rmsnorm-sage-vsa/run.py --check` 验证：

| 验证维度 | 指标 / 不变量 | 实测 / 复现值 | 状态 |
|---|---|---|---|
| `frost_low_latency` 激活与 Warp 组合 | Typed Activations / FC1×FC2 Warp Pairs | `10` 种激活 / `16` 种组合 (`18` MX tactics) | `PASS` |
| `frost_low_latency` B300 峰值加速 | BF16 (`T=1` SwiGLU) / MX (`W4/T1`) | **`1.794×`** (`27.736→15.460 µs`) / **`1.608×`** (`21.616→13.440 µs`) | `PASS` |
| `cake_rmsnorm_train` 树归约与翻转栅栏 | 任意 CTA 到达序下的 `dw` 逐位一致性 & 自复位 | `tree_bitwise_deterministic = True`, R200 bwd **`1.134×`** | `PASS` |
| `cake_sage` 动态共享内存增量与加速 | `dynamic_smem_bytes` / B200 `A1` 加速 | `119040 -> 120320 B` (`+1280 B`), B200 **`1.08×`** | `PASS` |
| `cake_ssd_combined` & `cake_vsa_sm90` | R200 SSD 主核峰值 / H100 VSA 18 程序峰值 | SSD **`1.268×`** (`1.196×` call) / VSA **`1.2624×`** | `PASS` |
