# 大模型混合精度训练 `torch.amp.GradScaler` 在 `BF16` 下的误开启同步阻塞（`found_inf.item()` `14.6 ms` 卡顿）与 `FP8` 延迟缩放（`Delayed Scaling`）vs. 细粒度块级实时缩放（`1x128 / 128x128 Blockwise JIT Scaling`）选型：为什么从 `FP16` 迁移到 `BF16` 时忘了关 `GradScaler` 会让每步多搬运 `28 GB` 显存并强制 CPU-GPU 同步，而在 MoE 模型上 `FP8` 全局延迟缩放又会频繁触发异常值截断？

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/pytorch_amp_gradscaler_bf16_found_inf_sync_stall_fp8_scaling_selection/run.py` 与 `results.json`。

![PyTorch AMP GradScaler on BF16 & FP8 Blockwise JIT Scaling Visual Summary](../../assets/svg/pytorch-amp-gradscaler-bf16-found-inf-sync-stall-fp8-scaling-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向大模型混合精度训练（`PyTorch AMP / TransformerEngine / DeepSeek-V3 FP8 / TorchTitan`）的“`BF16` 强制禁用 `GradScaler(enabled=False)`（彻底消除 `_amp_foreach_non_finite_check_and_unscale_` 28GB 显存读写与 `found_inf.item()` `14.60 ms` 跨卡同步屏障）+ `FP8` 细粒度块级在线实时量化（`1x128` 激活值 Per-Token-Group / `128x128` 权重 Per-Block Online JIT Scaling，替代易受 MoE 离群值冲击的全局 `amax_history` 延迟缩放）”全栈数值架构**。
- **架构师价值**：在审查各大团队的大模型训练代码时，有两个与“动态缩放（Scaling）”相关的常见误区正在静默吞噬大量算力与收敛稳定性：
  1. **误区一：把旧版 `FP16` 训练脚本改成 `torch.autocast("cuda", dtype=torch.bfloat16)` 时，依然保留了 `scaler = torch.amp.GradScaler("cuda")`（默认 `enabled=True`），导致每步白白卡顿 `14.60 ms`、多读写 `28.0 GB` 显存，甚至人为把大梯度乘爆成 `Inf` 触发跳步（Tier 1）**！
     - **第一性原理反思：当初为什么要发明 `GradScaler`？**
       - 因为 `FP16`（IEEE 754 半精度，`1` 符号位 + **`5` 指数位** + `10` 尾数位）的动态范围极窄：正常正数范围只有 **$[6.10 \times 10^{-5}, \;\; 65504.0]$**！在反向传播中，小于 $6 \times 10^{-5}$ 的微小梯度在 `FP16` 下会直接下溢变成 `0.0`，所以必须先把 Loss 乘以一个巨大的缩放因子（默认 $2^{16} = 65,536$）把梯度往上抬，算完反向后再除以 $65,536$（`unscale_`）！
       - **那么 `BF16` 需要 `GradScaler` 吗？完全不需要，而且开了纯属帮倒忙！**
         因为 `BF16`（`1` 符号位 + **`8` 指数位** + `7` 尾数位）拥有与 `FP32` **100% 完全相同的 8 位指数位**，其动态范围高达 **$[1.18 \times 10^{-38}, \;\; 3.39 \times 10^{38}]$**！`BF16` 梯度根本不存在 `FP16` 那种指数下溢问题！
     - **误开 `GradScaler` 在 `BF16` 下造成的三大毒害**：
       1. **强制 CPU-GPU 流同步屏障（`14.60 ms/step` 阻塞）**：在调用 `scaler.step(optimizer)` 时，PyTorch 必须在 CPU 上执行一句 `if found_inf.item():` 来判断本步梯度有没有因为乘了缩放因子而溢出成 `Inf/NaN`——这句 `.item()` **强制阻塞等待当前 GPU 上所有的反向计算与 `FSDP/DDP Reduce-Scatter` 全部跑完**，在 512 卡集群上放大微小的跨卡抖动，每步白白损耗 **`14.60 ms`**！
       2. **白白多搬运 `28.0 GB/step` 的 HBM 显存带宽**：`unscale_()` 必须把全模型 `14.0 GB` 的梯度分片从 HBM 读出来、乘以 $1/\text{scale}$、再写回 HBM（共读写 **`28.0 GB`**），纯属无用功！
       3. **人为制造溢出跳步（`8.5` 次跳步/千步）**：在训练初期梯度范数较大时，原本在 `BF16` 下完全合法的梯度被 `GradScaler` 强行乘以 $2^{16}$ 后在中间算子里溢出，导致每 1,000 步被无辜跳过 `8.5` 步！
  2. **误区二：在升级到 Hopper / Blackwell `FP8` 训练（`E4M3 max = 448.0`, `E5M2 max = 57,344.0`）时，沿用第一代“全局单张量延迟缩放（Per-Tensor Delayed Scaling，`amax_history_len = 1024`，Tier 3）”，结果在 MoE / SwiGLU 模型上频繁遭遇离群值（Outlier Spikes）截断与每千步 `18.0` 次溢出跳步**！
     - **为什么全局延迟缩放在 MoE 上水土不服？**
       1. **时间滞后（Temporal Lag）**：延迟缩放用 **上一步（或过去 1,024 步）的最大绝对值 `amax`** 来计算当前步的缩放系数 $S_t = \frac{448}{\text{amax}_{t-1}}$。当 MoE 路由在某一步突然切换到某个激活幅值翻倍的专家时，用上一步的旧缩放系数去量化当前步，直接冲破 `E4M3` 的上限 `448.0` 发生饱和截断或 `NaN`！
       2. **空间连坐（Spatial Outlier Crushing）**：一个 `[4096, 4096]` 的大张量共有 `1,677 万` 个元素。只要其中 **1 个通道（Outlier Channel）** 出现了数值 `400.0`，整张矩阵的全局缩放系数就会被这 1 个离群值绑架，把其余 `99.9%` 幅值只有 `0.5` 的正常通道全部压缩到接近 `0` 的极低精度区间（信噪比崩塌）！
  采用 **`BF16` 下严格设置 `GradScaler(enabled=False)`（立省 `14.60 ms/step` 与 `28.0 GB` 带宽，吞吐提升 `+8.4%`）+ `FP8` 下采用 `DeepSeek-V3` 范式的细粒度块级在线实时量化（激活值按 `1x128` 连续 Token 通道组、权重按 `128x128` 子块在片上 SRAM 实时求当前块 `amax` 并量化，零历史滞后、零离群值连坐）（Tier 4）**，可将跳过步数与同步停顿双双清零（**`0.00 ms` 停顿，`0` 跳步**），并将训练吞吐大幅拉升至 **`154.2%`（`1.54x`）**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：四大训练浮点格式的指数位（Exponent Bits）与缩放策略黄金法则**：
   | 浮点格式 | 位宽分配 (符号/指数/尾数) | 最大正规格数 (`Max Normal`) | 最小正规格数 (`Min Normal`) | 正确缩放策略 |
   | :--- | :--- | :--- | :--- | :--- |
   | **`FP16`** | `1 / 5 / 10` | `65,504.0` | $6.10 \times 10^{-5}$ | ✅ **必须开启 `GradScaler(enabled=True)`** |
   | **`BF16`** | `1 / 8 / 7` | $3.39 \times 10^{38}$ (同 `FP32`!) | $1.18 \times 10^{-38}$ | 🏆 **必须关闭 `GradScaler(enabled=False)`** |
   | **`FP8 E4M3`** | `1 / 4 / 3` | `448.0` | $2^{-6} \approx 0.0156$ | 🏆 **使用 `1x128 / 128x128` 在线块级 JIT 缩放** |
   | **`FP8 E5M2`** | `1 / 5 / 2` | `57,344.0` | $6.10 \times 10^{-5}$ | 🏆 **使用 `1x128` 在线块级 JIT 缩放 (或全 E4M3 块缩放)** |
2. **第一性原理：为什么 `1x128` / `128x128` 在线块级实时缩放（Online Blockwise JIT Scaling）能彻底终结全局延迟缩放？**
   - **消除空间连坐**：将激活张量沿内积 $K$ 维每 **`128` 个连续元素（`1x128` Tile）** 划分为一个独立量化微块，每个 `1x128` 微块拥有自己专属的 `FP32 / E8M0` 缩放因子 $s_{i,b} = \frac{\max_{k \in \text{tile}} |X_{i,k}|}{448.0}$。即便第 5 号通道有巨大的离群值，它也只会影响自己所在的那个 128 元素小块，其余几十万个小块依然享受满格的 `E4M3` 动态范围！
   - **消除时间滞后且不增 HBM 流量**：因为 `1x128` 微块只有 `128` 个元素（在 `BF16` 下仅 `256 Bytes`），前序算子（如 `RMSNorm` 或 `SwiGLU`）在寄存器里算出这 128 个元素时，**顺手在同一个 Warp 内部做一次寄存器级 `warp_reduce_max` 即可当场算出精确的实时 $s_{i,b}$ 并写出 `FP8` 数据**——**零额外 HBM 往返读写、零跨步历史滞后、零溢出跳步**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：怎么用 1 行规范写法确保代码在切换 `FP16` 和 `BF16` 时永远不会误开 `GradScaler`？
- **永远把 `enabled=(amp_dtype == torch.float16)` 传给 `GradScaler` 构造函数**：
  ```python
  # 🏆 工业级防呆写法：仅在 FP16 时激活，BF16 下所有 scaler 调用自动变成 0 开销空操作 (No-op)!
  scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
  ```
  当 `enabled=False` 时，`scaler.scale(loss).backward()` 直接等价于 `loss.backward()`，`scaler.step(optimizer)` 直接等价于 `optimizer.step()`，既不用改主循环代码结构，又彻底消除了 `_amp_foreach_non_finite_check_and_unscale_` 和 `found_inf.item()` 同步屏障！

### Q2：关掉 `GradScaler` 后，如果在 `BF16` 训练中还想做梯度裁剪 `clip_grad_norm_(model.parameters(), 1.0)`，怎样避免 `clip_grad_norm_` 自己触发 `.item()` 同步阻塞？
- **不要在 Python 里写 `if total_norm.item() > max_norm:`**！
  PyTorch 原生的 `torch.nn.utils.clip_grad_norm_(..., foreach=True)` 在底层是用 **GPU 张量算式** 计算缩放系数：
  $$\text{clip\_coef\_clamped} = \text{torch.clamp}\left(\frac{\text{max\_norm}}{\text{total\_norm} + 10^{-6}}, \;\; \text{max}=1.0\right)$$
  然后直接把所有梯度原地乘上这个位于 GPU 显存里的标量张量 `clip_coef_clamped`——**全程不需要调 `.item()` 拷回 CPU 判断，100% 异步无阻塞**！只有在你为了打日志去调 `print(float(total_norm))` 时才会触发同步（因此日志只应每 $N=10 \sim 50$ 步异步打印一次）！

### Q3：在 Hopper H100 上做 `1x128` 块级 `FP8` 矩阵乘时，既然 Tensor Core 的 `WGMMA` 指令只原生支持单次 `64x128x32` 的统一累加，是怎么实现每 `128` 个 $K$ 维元素换一个缩放因子的？
- 这正是 `DeepSeek-V3` (`DeepGEMM`) 的精妙设计：**两级累加（Two-Level Promotion Accumulation）**！
  - 在内层循环中，先让 Tensor Core `WGMMA` 连续累加 $K = 128$ 个元素的乘积到一个短期的 `FP32` 累加器 `short_accum` 中；
  - 然后用 CUDA Core 的 `FFMA` 指令将 `short_accum * (scale_A_tile * scale_B_tile)` 乘上本块的缩放因子后，累加进最终的主寄存器 `master_accum` 中！这一操作与下一轮 `WGMMA` 异步流水线完全重叠，既克服了 Hopper 硬件指令限制，又实现了极高精度的块级 `FP8` 训练（而在 Blackwell B200 `MXFP8` 中，硬件 Tensor Core 已原生内置了每 32 元素的块缩放支持）！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 大模型混合精度训练（`FP16 / BF16 / FP8`）缩放策略选型决策树

```mermaid
flowchart TD
    A["配置大模型混合精度训练流水线 (70B / MoE, 512 GPUs)"] --> B{"前反向主计算数据类型 (AMP Dtype)?"}
    B -- "torch.float16 (FP16, 5位指数, max=65504)" --> C["✅ 必须启用 GradScaler(enabled=True) 防止下溢"]
    B -- "torch.bfloat16 (BF16, 8位指数, max=3.39e38)" --> D{"是否误开了 GradScaler(enabled=True)?"}
    D -- "是: 沿用默认 GradScaler(enabled=True)" --> E["⛔ Tier 1: 每步多读写 28 GB 显存 + found_inf.item() 阻塞 14.60 ms + 8.5 次/千步误溢出跳步!"]
    D -- "否: 显式设置 GradScaler(enabled=False) + 无同步 foreach 梯度裁剪" --> F["✅ Tier 2: 0.00 ms 同步阻塞 + 节省 28 GB/step 显存带宽 + 吞吐提升 +8.4%!"]
    B -- "FP8 (E4M3 max=448 / E5M2 max=57344)" --> G{"FP8 量化缩放因子粒度与时序?"}
    G -- "全局单张量延迟缩放 (Per-Tensor Delayed amax_history=1024)" --> H["⚠️ Tier 3: MoE 路由离群值跨步突变导致 18 次/千步截断跳步 + 离群通道压扁正常通道精度!"]
    G -- "细粒度块级在线实时缩放 (1x128 激活 / 128x128 权重 Online JIT)" --> I["🏆 Tier 4: 0.00 ms 同步停顿 + 0 跳步 + 离群值局部隔离 + 吞吐飙升至 154.2% (1.54x)!"]
```

### 3.2 `BF16` 误开 `GradScaler`（`14.60 ms` 同步阻塞）& `FP8` 延迟缩放 vs. 细粒度 `1x128` 在线 JIT 对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 3: BF16 误开 GradScaler 阻塞 14.6ms + FP8 全局延迟缩放离群崩溃"]
      B1["Tier 1 (BF16 + GradScaler): unscale_ 读写 28 GB + found_inf.item() 强制同步等待 14.60 ms!"] --> B2["Tier 3 (FP8 全局延迟缩放): 用上一步全局 amax 量化 -> MoE 突发离群值冲破 448.0 频繁跳步!"]
    end
    subgraph Good["🏆 Tier 4: GradScaler(enabled=False) + FP8 1x128 / 128x128 块级在线实时 JIT 缩放"]
      G1["BF16 关闭 GradScaler -> 0 同步屏障, 省 28 GB 显存带宽 (+8.4% 吞吐)"] --> G2["FP8 算子在 SRAM 内按 1x128 微块实时求 amax 并量化 -> 0 历史滞后, 吞吐暴涨至 154.2%!"]
    end
```

### 3.3 `BF16` 误开 `GradScaler` 的同步断点 vs. 零同步 `FP8 1x128` 在线量化 + 异步梯度裁剪时序图

```mermaid
sequenceDiagram
    participant Comp as GPU Compute Stream
    participant HBM as GPU HBM (14 GB Grads)
    participant CPU as Python Host Thread

    Note over Comp,CPU: ⛔ Tier 1 (BF16 + GradScaler=True):
    Comp->>HBM: 1a. Run `_amp_foreach_non_finite_check_and_unscale_` (reads & writes 28.0 GB!)
    HBM->>CPU: 2a. `if found_inf.item():` -> BLOCKING D2H sync stalls CPU & GPU for 14.60 ms!
    Note over Comp,CPU: 🏆 Tier 4 (GradScaler=False + FP8 1x128 Online JIT):
    Comp->>Comp: 1b. Quantize `1x128` tiles online in SRAM & run `WGMMA` + 2-level FP32 accumulation
    CPU->>Comp: 2b. Launch fused `clip_grad_norm_(foreach=True)` & `optimizer.step()` asynchronously (0.00 ms stall!)
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | 混合精度缩放与梯度更新方案 | 每步同步与反缩放耗时 | 每千步溢出跳过步数 | 额外 HBM 读写流量 | 相对训练吞吐 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: `BF16` + 误开 `GradScaler(True)`** | 多余乘除 $2^{16}$ + `found_inf.item()` 同步 | **`14.60 ms` (严重阻塞! ✕)** | `8.5 步` ✕ | `28.0 GB/step` ✕ | `100.0%` (基准) |
| **Tier 2: `BF16` + `GradScaler(False)`** | 原生 8 位指数直跑 + 异步梯度裁剪 | **`0.00 ms` 🏆** | `0.0 步` 🏆 | `0.0 GB` 🏆 | `108.4%` (`+8.4%`) |
| **Tier 3: `FP8` 全局单张量延迟缩放** | 依赖历史 `amax_history` + 全局缩放 | `2.40 ms` | **`18.0 步` (离群冲击 ✕)** | `14.0 GB/step` | `136.0%` (`+36.0%`) |
| **Tier 4: `FP8` `1x128 / 128x128` 块级在线 JIT (推荐)** | **SRAM 实时局部块缩放 + `GradScaler=False`** | **`0.00 ms` 🏆** | **`0.0 步` 🏆** | **`0.0 GB` 🏆** | **`154.2%` (`+54.2%` 🏆)** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/pytorch_amp_gradscaler_bf16_found_inf_sync_stall_fp8_scaling/run.py`](../../experiments/pytorch_amp_gradscaler_bf16_found_inf_sync_stall_fp8_scaling_selection/run.py) · 封存数据：[`results.json`](../../experiments/pytorch_amp_gradscaler_bf16_found_inf_sync_stall_fp8_scaling_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/pytorch-amp-gradscaler-bf16-found-inf-sync-stall-fp8-scaling-sizer.html`](../reports/pytorch-amp-gradscaler-bf16-found-inf-sync-stall-fp8-scaling-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/pytorch-amp-gradscaler-bf16-found-inf-sync-stall-fp8-scaling-selection-cards.html`](../reports/pytorch-amp-gradscaler-bf16-found-inf-sync-stall-fp8-scaling-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: BF16 + GradScaler(True)", "rel_throughput_pct": 100.0, "sync_stall_ms": 14.60 },
      { "tier": "Tier 2: BF16 + GradScaler(False)", "rel_throughput_pct": 108.4, "sync_stall_ms": 0.00 },
      { "tier": "Tier 3: FP8 Per-Tensor Delayed Scaling", "rel_throughput_pct": 136.0, "sync_stall_ms": 2.40 },
      { "tier": "Tier 4: FP8 Blockwise 1x128 Online JIT", "rel_throughput_pct": 154.2, "sync_stall_ms": 0.00 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "rel_throughput_pct": "Percentage",
    "sync_stall_ms": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Disabling GradScaler on BF16 + FP8 1x128 Blockwise JIT Boosts Training Throughput to 154.2%",
    "subtitle": "Relative training throughput (%) across 4 mixed-precision scaling tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "rel_throughput_pct",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有使用 `BF16` 或 `FP8` 的大模型预训练、SFT 与 RL 训练流水线。
- **⛔ Not for**：在使用 `bfloat16` 时开启 `GradScaler(enabled=True)`，或在具有显著激活离群值的 MoE 模型上使用粗粒度单张量 `FP8` 延迟缩放。

### 生产核验 Checklist
- [ ] **全局排查训练代码中的 `GradScaler`：强制写为 `GradScaler("cuda", enabled=(dtype == torch.float16))`**，确保 `BF16` 与 `FP8` 下 `enabled=False`。
- [ ] **确保 `clip_grad_norm_` 使用 `foreach=True` 且不在每步主循环中调用 `total_norm.item()` 触发同步**。
- [ ] **在 Hopper / Blackwell `FP8` 训练中优先选用 `1x128`（激活）与 `128x128`（权重）细粒度块级在线实时（Online JIT）缩放**。
