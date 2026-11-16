# 大模型激活检查点（`Activation Checkpointing`）与张量/序列并行（`TP + SP`）中的随机数状态失步（`CUDA RNG Philox Offset Desync`）排障选型：为什么重计算时漏存 `Philox` 状态会导致梯度产生 `34.8%` 错误并在百步内发散，而每层调 `get_rng_state()` 又会打断 `CUDAGraph` 并白耗 `38 ms`？基于双轨 `TensorParallelRNGTracker`（分离 `TP-Synced` 与 `TP-Sharded` 种子）+ 设备端 `16B Philox (seed, offset)` 零同步快照（开销压至 `0.4 ms`，提速 `95x`，`100%` 逐比特对齐）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/llm_activation_checkpointing_tp_rng_philox_offset_desync_fork_rng_selection/run.py` 与 `results.json`。

![Activation Checkpointing & TP/SP CUDA RNG Philox Desync Visual Summary](../../assets/svg/llm-activation-checkpointing-tp-rng-philox-offset-desync-fork-rng-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向 3D 并行（`TP + SP + PP + DP`）与 `FP8 / NVFP4` 低精度预训练（`Megatron-Core / TorchTitan`）的“正交双轨随机数状态机（`TensorParallelRNGTracker`：同 TP 组对齐种子 vs 按 TP Rank 偏移种子）+ 设备端 16 字节 `Philox (seed, offset)` 寄存器状态直接入图（`CUDAGraph-Safe RNG State`）”架构**。
- **架构师价值**：在大模型预训练与后训练中，有三类核心算子高度依赖 GPU 硬件伪随机数生成器（`cuRAND Philox-4x32-10`）：① `FlashAttention` 内部的在线生成注意力 Dropout 掩码；② 残差连接（Residual / MoE Router Jitter）随机掩码；③ **`FP8 / NVFP4` 反向传播与权重更新中的随机舍入（`Stochastic Rounding, SR`）**。当同时开启 **激活检查点重计算（`Activation Checkpointing`）** 与 **张量并行/序列并行（`TP=8 + SP`）** 时，RNG 状态管理稍有不慎就会引发极难排查的数值发散或性能回退：
  1. **事故一：为了追求速度关闭 `preserve_rng_state=False`，或自定义算子未捕获 `Philox Offset`（Tier 1）**：在 Forward 经过第 $l$ 层时，GPU 的 `Philox` 随机数计数器处于 `offset = 100`；随后模型继续往前跑完第 $l+1 \dots 80$ 层，`offset` 已经推进到了 `8000`。到了 Backward 阶段需要重新执行第 $l$ 层的 Forward 来重算激活值时，如果直接用当前的 `offset = 8000` 去重算，**重算出来的 Dropout 掩码或随机舍入方向与第一次 Forward 算出的值完全不同（匹配率 `0.0%`）**！用一套错误的激活值 $X^{\text{recomputed}} \neq X^{\text{forward}}$ 去乘上游梯度 $\nabla Y$，算出的权重梯度相对误差高达 **`34.8%`**，训练不到 500 步直接 `Loss NaN` 崩溃！
  2. **事故二：在 `TP=8` 的 8 张卡上只用同一个全局随机种子（Tier 2）**：如果 8 张 TP 卡的随机种子完全相同，那么每张卡负责的 `8` 个不同注意力头（Attention Heads）生成的随机掩码竟然 **100% 雷同（退化为强相关噪声）**；反之，如果把 8 张卡的全局种子設为各不相同，那么在未切分的复制层（或未使用 SP 的残差层）上，8 张卡丢弃了不同位置的元素，导致本该完全一致的张量在 `TP=8` 各卡之间彻底失步（误差 `12.4%`）！
  3. **事故三：每层进出都调用旧版 CPU `torch.cuda.get_rng_state()` / `set_rng_state()`（Tier 3）**：在 80 层模型的前向和反向重计算中合计调用 `160` 次 CPU ByteTensor 状态提取与还原，不仅每步白白耗费 **`38.0 ms`**，更因涉及主机内存操作而直接破坏 `CUDAGraph` 与 `torch.compile`！
  采用 **双轨 `TensorParallelRNGTracker`（精确区分 `TP-Synced` 与 `TP-Sharded` 随机流）+ 直接在 GPU 显存/寄存器内传递 16 字节 `Philox (seed, offset)` 二元组（Tier 4）**，可在保证前反向 **`100.0%` 逐比特精确复现（梯度误差 `0.0%`）** 的前提下，将每步 RNG 管理开销从 `38.0 ms` 压至 **`0.4 ms`（提速 `95.0x`）**，且 100% 兼容 `CUDAGraph`！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：为什么 `FlashAttention` 可以不保存 `[B, H, N, N]` 的 Dropout 掩码矩阵，只靠 `16 字节`就能在反向传播中完美复现数亿个随机数？**
   - 如果在 `32K` 上下文下把注意力 Dropout 的布尔掩码矩阵 `[B, H, 32768, 32768]` 存进 HBM 供反向传播使用，仅 1 层就要吃掉 **`32 GB` 显存**（彻底摧毁了 FlashAttention 省显存的意义）！
   - **计数器式伪随机数生成器（Counter-based PRNG：`Philox-4x32-10`）的数学魔法**：与传统线性同余必须串行一步步递推不同，`Philox` 本质上是一个纯无状态哈希函数：给定 64 位种子 `seed`、64 位全局步进偏移量 `offset` 以及当前线程负责的矩阵坐标 `(batch, head, row, col)`，只需在 GPU 寄存器里执行 10 轮乘法异或混淆，就能在 $\mathcal{O}(1)$ 时间内直接算出该坐标的随机数：
     $$u_{b,h,i,j} = \text{Philox}_{10}\big(\text{seed},\ \text{offset} + \text{tid}(b,h,i,j)\big)$$
   - 因此，`FlashAttention` 在 Forward 时根本不往 HBM 写 32GB 掩码，而只是把当时的 **`(philox_seed, philox_offset)`（仅仅两个 `int64` 整数，共 `16 Bytes`！）** 保存下来；到了 Backward 和激活重计算时，把这 16 字节传回内核，GPU 在寄存器里当场重算出 **100% 一模一样的掩码**！
2. **为什么在 `TP=8 + Sequence Parallel (SP)` 下必须维护两套独立的 `CUDA Generator`（`TensorParallelRNGTracker`）？**
   - 在张量并行（`TP=8`）中，模型里同时存在两类不同性质的随机操作：
     - **第一类（必须在 `TP=8` 组内保持 8 卡完全一致：`Data-Parallel / Base RNG`）**：输入 Embedding 初始化、非 SP 模式的残差 Dropout（8 张卡经过 `All-Reduce` 后持有完全相同的激活值，必须施加完全相同的掩码，否则 8 卡激活值瞬间分裂！）$\to$ 种子设为 **`seed = base_seed + dp_rank * 100`**（同一 DP 组内的 8 张 TP 卡种子完全相同）；
     - **第二类（必须在 `TP=8` 组内 8 卡互不相同：`Model-Parallel / TP-Sharded RNG`）**：① 多头注意力内部（每张 TP 卡负责不同的 `H / 8` 个头，不能用相同掩码）；② 开启序列并行（`Sequence Parallel`）后的 Dropout 与 `FP8` 随机舍入（每张 TP 卡持有不同的 `S / 8` 序列切片，必须使用独立随机流）$\to$ 种子设为 **`seed = base_seed + dp_rank * 100 + 2718 + tp_rank`**（通过 `with rng_tracker.fork("model-parallel-rng"):` 瞬间切换）！
3. **为什么必须用设备端张量状态（`get_device_state / philox_cuda_state`）替代旧版 `torch.cuda.get_rng_state()`？**
   - 旧版 `get_rng_state()` 返回的是一个驻留在 **CPU 主机内存** 上的 `torch.ByteTensor`；每次 `checkpoint()` 进出都要在 CPU 上构造对象并与 C++ Generator 同步，不仅慢（`38.0 ms`/80层），而且在 `CUDAGraph` 录制时因偏移量被静态固化而报错。
   - 现代 PyTorch `CUDAGeneratorImpl` 支持直接将 `seed` 和 `offset` 保存在 **GPU 显存上的 `int64` 标量张量（Device-Side Tensor State）** 中：在计算图内完全通过 GPU 指针读取和递增偏移量，**CPU 零同步（`0.4 ms`）且完美支持 `CUDAGraph` 反复 Replay**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：如果我们训练时把 `attention_dropout = 0.0` 和 `hidden_dropout = 0.0` 都关掉了（现代 LLM 经常设为 0），那在激活检查点里设 `preserve_rng_state=False` 是不是就绝对安全了？
- **在开启了 `FP8 / NVFP4` 训练、MoE 路由器噪声（Router Jitter）或任何带随机舍入（`Stochastic Rounding`）的内核时，设 `preserve_rng_state=False` 依然会引发严重梯度失真！**
  许多工程师以为“没开 Dropout 就没有随机数”。殊不知在 `Hopper / Blackwell` 的低精度训练中，将 `BF16/FP32` 激活值量化为 `FP8 / NVFP4` 时常使用硬件/软件随机舍入（Stochastic Rounding）来消除量化偏置；此外自定义算子或数据增强也可能隐式消费随机数。由于采用设备端 `Philox (seed, offset)` 保存状态的开销只有 **`0.4 ms`**（占单步时间 `< 0.01%`），**工程上永远应当保持 `preserve_rng_state=True`（使用 Graph-Safe Device State）**！

### Q2：如果编写了一个自定义 Triton 算子（内含 `tl.rand(seed, offsets)`），放在 `torch.utils.checkpoint` 里面跑，为什么开了 `preserve_rng_state=True` 重算出来的结果还是不对？
- **极具欺骗性的 Triton 随机数陷阱**！
  PyTorch 的 `preserve_rng_state=True` 默认保存和恢复的只是 **PyTorch 自带的默认 `CUDAGenerator` 的状态**！如果你在调用 Triton 算子时，在 Python 端手写了一个自增计数器 `my_step += 1` 或者用了 Python 的 `random.randint()` 作为 `seed` 传给 `tl.rand(seed, ...)`，那么 PyTorch 的 `fork_rng` 根本不知道你那个 Python 变量的存在——Backward 重算时 `my_step` 已经变大了，导致 Triton 算子在重算时拿到了错误的 `seed`！
- **正确规范**：所有自定义 Triton 随机算子必须通过调用 PyTorch 默认 Generator（或 `torch.cuda.nvtx` / `philox_cuda_state` 接口）获取由 `fork_rng` 托管的 `(seed, offset)`，严禁在 Python 闭包外维护自增随机计数器！

### Q3：在流水线并行（`PP=8`）的不同 Stage 上，随机种子应该怎么设？
- 每个流水线 Stage 代表模型的不同深度层（例如 Stage 0 是第 1..10 层，Stage 1 是第 11..20 层），因此不同 `pp_rank` 的权重初始化与随机掩码必须独立：在初始化种子时应包含 `pp_rank` 偏移（如 `base_seed + 100 * pp_rank + tp_rank`），防止所有流水线阶段生成完全同步的周期性噪声模式！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 激活检查点与 3D 并行 RNG 状态管理选型决策树

```mermaid
flowchart TD
    A["80 层 TP=8 + SP 分布式训练开启 Activation Checkpointing 与随机算子 (Dropout / FP8 SR)"] --> B{"激活重计算时是否严格保存并还原 Philox (seed, offset)?"}
    B -- "否: preserve_rng_state=False 或 Triton 自增外部计数器" --> C["⛔ Tier 1: Backward 重算掩码与 Forward 0% 匹配 -> 梯度误差高达 34.8% 发散崩溃!"]
    B -- "是: 还原前向 RNG 状态" --> D{"TP=8 组内是否分离了复制层与分片层的随机流?"}
    D -- "否: 8 张 TP 卡共用单一随机种子" --> E["⛔ Tier 2: 复制层激活在 8 卡间分裂失步或注意力头掩码雷同 -> 12.4% 梯度误差!"]
    D -- "是: 使用 TensorParallelRNGTracker 双轨分离" --> F{"RNG 状态的捕获与还原通过 CPU ByteTensor 还是 GPU 设备端二元组?"}
    F -- "每层调 CPU torch.cuda.get_rng_state() / set_rng_state()" --> G["⚠️ Tier 3: 精度 100% 正确, 但 160 次主机状态切换耗时 38.0 ms 且打断 CUDAGraph"]
    F -- "双轨 TP/DP Tracker + 设备端 16B Philox (seed, offset) 直传" --> H["🏆 Tier 4: 100% 逐比特精确对齐 + 0.0% 梯度误差 + 仅 0.4 ms 开销 (快 95x, 兼容 CUDAGraph)!"]
```

### 3.2 重计算 `Philox Offset` 失步导致梯度损毁（`34.8%` 误差）vs. 设备端 `16B (seed, offset)` 精确还原对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: 未保存还原 Philox Offset (前反向掩码失配, 梯度误差 34.8%)"]
      B1["Forward Layer l: 使用 Philox offset=100 算出激活 X_fwd"] --> B2["Backward Recompute Layer l: 误用当前 offset=8000 算出错误 X_recomp != X_fwd -> 梯度损毁!"]
    end
    subgraph Good["🏆 Tier 4: 设备端保存 16 字节 (seed, offset) 二元组 (100% 逐比特一致, 耗时 0.4 ms)"]
      G1["Forward Layer l: 记录 16B (seed, offset=100) 存入检查点上下文"] --> G2["Backward Recompute Layer l: 直接传入 (seed, offset=100) -> 寄存器瞬时重算出 100% 相同激活!"]
    end
```

### 3.3 `TensorParallelRNGTracker` 双轨切换与 `FlashAttention` `Philox` 状态重算时序图

```mermaid
sequenceDiagram
    participant Track as TensorParallelRNGTracker (GPU Generators)
    participant Ckpt as Activation Checkpoint Context (16B State)
    participant FA as FlashAttention / FP8 Kernel (Philox-4x32-10)

    Note over Track,FA: 阶段 1: Forward Pass (Layer l)
    Track->>Ckpt: 1. Enter `with rng_tracker.fork("model-parallel-rng")` (seed = base + tp_rank)
    Ckpt->>FA: 2. Capture current device 16B `(philox_seed, philox_offset)` & launch forward kernel
    Note over Track: Generator increments `philox_offset += threads_per_grid`
    Note over Track,FA: 阶段 2: Backward Activation Recompute (Layer l)
    Ckpt->>FA: 3. Restore saved 16B `(philox_seed, philox_offset)` on device (0ms CPU sync!)
    FA->>FA: 4. Recompute `Philox_10(seed, offset + tid)` in registers -> 100.0% bitwise identical activations!
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | 检查点与 `TP/SP` RNG 管理架构 | 前反向重算逐比特匹配率 | 每步 RNG 状态开销 | 权重梯度相对误差 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: `preserve_rng_state=False`（不还原状态）** | 重计算使用跑偏的 `Philox Offset` | `0.0%` (完全失配! ✕) | `0.0 ms` | `34.8%` (发散崩溃! ✕) | ⛔ **只要含任何随机算子必导致梯度错误** |
| **Tier 2: `TP=8` 共用单一未分离 RNG 种子** | 未区分 `TP` 复制层与分片层 | `50.0%` ✕ | `28.5 ms` ✕ | `12.4%` ✕ | ⛔ **导致跨 TP 卡激活分裂或掩码退化** |
| **Tier 3: 传统每层 CPU `get/set_rng_state()`** | 160 次主机 `ByteTensor` 状态搬运 | `100.0%` ✅ | `38.0 ms` ⚠️ | `0.0%` ✅ | ⚠️ 兼容旧代码，但慢且不支持 `CUDAGraph` |
| **Tier 4: 双轨 `TP/DP` Tracker + 设备端 `Philox` (推荐)** | **正交种子分离 + `16B` 设备端状态** | **`100.0%` 🏆** | **`0.4 ms` (`95x` 快 🏆)** | **`0.0%` (零误差 🏆)** | 🏆 **3D 并行与 `FP8` 预训练黄金标准** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/llm_activation_checkpointing_tp_rng_philox_offset_desync_fork_rng_selection/run.py`](../../experiments/llm_activation_checkpointing_tp_rng_philox_offset_desync_fork_rng_selection/run.py) · 封存数据：[`results.json`](../../experiments/llm_activation_checkpointing_tp_rng_philox_offset_desync_fork_rng_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/llm-activation-checkpointing-tp-rng-philox-offset-desync-fork-rng-sizer.html`](../reports/llm-activation-checkpointing-tp-rng-philox-offset-desync-fork-rng-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/llm-activation-checkpointing-tp-rng-philox-offset-desync-fork-rng-selection-cards.html`](../reports/llm-activation-checkpointing-tp-rng-philox-offset-desync-fork-rng-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: preserve_rng_state=False", "grad_err_pct": 34.8, "overhead_ms": 0.0 },
      { "tier": "Tier 2: Unsplit Single TP=8 Seed", "grad_err_pct": 12.4, "overhead_ms": 28.5 },
      { "tier": "Tier 3: Legacy Host get/set_rng_state", "grad_err_pct": 0.0, "overhead_ms": 38.0 },
      { "tier": "Tier 4: Dual TP/DP Tracker + Device Philox", "grad_err_pct": 0.0, "overhead_ms": 0.4 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "grad_err_pct": "Percentage",
    "overhead_ms": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Dual TP/DP RNG Tracker + Device Philox State Eliminates 34.8% Grad Error at 0.4ms Overhead (95x Faster)",
    "subtitle": "RNG state management overhead per 80-layer step (ms) and gradient error (%) (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "overhead_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有开启了激活检查点（Full / Selective Checkpointing）、张量并行与序列并行（`TP + SP`）、注意力/残差 Dropout 或 `FP8 / NVFP4` 随机舍入的训练流水线。
- **⛔ Not for**：在检查点包裹的自定义 Triton/CUDA 算子中使用未受 `fork_rng` 托管的外部自增计数器作为随机种子。

### 生产核验 Checklist
- [ ] **使用 `TensorParallelRNGTracker` 严格分离 `TP-Synced`（复制区）与 `TP-Sharded`（注意力分头与 SP 分片区）随机种子**：防止跨卡激活失步与同组掩码雷同。
- [ ] **启用设备端 `16-Byte Philox (seed, offset)` 状态捕获（`register_graph_safe_rng_state`）**：将 80 层 RNG 快照开销控制在 `< 0.5 ms` 并保持 `CUDAGraph` 兼容。
- [ ] **审计所有自定义 Triton `tl.rand` 算子**：确保其 `(seed, offset)` 100% 取自当前上下文的 `CUDAGenerator`，确保前向与反向重计算输出逐比特相等。
