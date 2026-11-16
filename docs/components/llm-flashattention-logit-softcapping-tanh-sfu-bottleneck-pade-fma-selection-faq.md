# 大模型注意力软截断（`Attention Logit Soft-Capping: Gemma-2/3 & Grok` `cap * tanh(S / cap)`）在 `FlashAttention` 中的超越函数单元（`SFU MUFU.TANH`）瓶颈与有理帕德近似（`Rational Padé FMA`）选型：为什么开启 `softcap=50.0` 会让 H100 `FlashAttention` 算力从 `640` 暴跌至 `275 TFLOPS`（降速 `2.33x`）？基于 `FP32 FFMA` 有理多项式卸载 `SFU` 拥堵 + 预折叠 `1/cap` 与 `cap * log2(e)` 缩放系数（算力恢复 `585 TFLOPS`，提速 `2.13x`，误差 `< 1.2e-5`）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/llm_flashattention_logit_softcapping_tanh_sfu_bottleneck_pade_fma_selection/run.py` 与 `results.json`。

![FlashAttention Logit Soft-Capping (tanh) SFU Bottleneck & Padé FMA Visual Summary](../../assets/svg/llm-flashattention-logit-softcapping-tanh-sfu-bottleneck-pade-fma-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向采用 `Attention Logit Soft-Capping`（$S' = \text{cap} \cdot \tanh(S / \text{cap})$，如 `Gemma-2 / Gemma-3 / Grok`）的大模型内核优化：“利用闲置 `FP32 CUDA Core (FFMA)` 运行有理帕德/极小极大逼近多项式（`Rational Padé / Minimax Polynomial`）替代表达式中的硬件 `MUFU.TANH` 指令（彻底解除超越函数单元 `SFU` 排队阻塞）+ 将 $1/(\text{cap}\sqrt{d})$ 预折叠进 `Q`、将 $\text{cap} \cdot \log_2(e)$ 预折叠进多项式分子”**。
- **架构师价值**：为了防止注意力分数 $S = \frac{QK^T}{\sqrt{d}}$ 过大导致训练梯度爆炸与推理注意力过度尖锐，`Gemma-2` 与 `Grok` 等主流架构在注意力 Softmax 之前引入了双曲正切软截断：
  $$P = \text{Softmax}\left(\text{cap} \cdot \tanh\left(\frac{Q K^T}{\text{cap} \cdot \sqrt{d}}\right)\right), \quad \text{cap} \in \{30.0, 50.0\}$$
  这个在数学公式上只多了一个 $\tanh(\cdot)$ 的改动，在 GPU 硬件底层却给 `FlashAttention-2/3` 带来了极其严重的性能断崖：
  1. **如果回退到未融合的 PyTorch `torch.tanh(S / cap) * cap`（Tier 1）**：必须先把 `[B, H, N, N]` 的 $N \times N$ 注意力分数矩阵写回 HBM 显存，在 `32K` 上下文下算力跌至 **`78.0 TFLOPS`**（耗时 `485.0 ms`）且直接爆显存！
  2. **如果在 `FlashAttention` 内部循环直接对每个寄存器元素调用硬件 `__tanhf()`（编译为 PTX `tanh.approx.f32` / SASS `MUFU.TANH` 指令，Tier 2）**：在 NVIDIA Hopper (`SM90`) 与 Blackwell (`SM100`) 架构中，`MUFU.TANH`（双曲正切）和 Online Softmax 必需的 `MUFU.EX2`（以 2 为底的指数 $2^x$）**争抢的是同一个带宽极其狭窄的特殊函数硬件单元（`SFU: Special Function Unit`）**！每个 SM 的 Tensor Core 吞吐高达每周期数千次运算，而 `SFU` 每周期只能处理极少数超越函数指令。原本单跑 `MUFU.EX2` 时 `SFU` 就已经是瓶颈，现在每个元素再多加一次 `MUFU.TANH`（SFU 指令量直接翻倍！），再加上打断了预缩放乘法，导致 H100 上的有效注意力算力从无软截断时的 `~640 TFLOPS` 暴跌至 **`275.0 TFLOPS`（降速超过一半，耗时 `137.5 ms`）**！
  采用 **在闲置的 `FP32 CUDA Core (FFMA 流水线)` 上用有理帕德近似计算 $\tanh(x)$（从而把宝贵的 `SFU` 硬件 `100%` 留给 `MUFU.EX2`！）+ 算子前后常数折叠（Tier 4）**，可将软截断 `FlashAttention` 的有效吞吐从 `275.0 TFLOPS` 狂飙至 **`585.0 TFLOPS`（提速 `2.13x`，耗时压至 `64.6 ms`）**，且最大绝对数值误差 `< 1.2\times 10^{-5}`（在 `BF16` 下与硬件 `MUFU.TANH` 100% 比特级等价）！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：为什么在 `FlashAttention` 里，用 7 条 `FFMA`（乘加）指令算多项式，反而比用 1 条硬件 `MUFU.TANH` 指令快 `2.13 倍`？**
   - 初学者常误以为“指令条数越少越快”。但在现代 GPU（如 H100/B200）的微架构中，决定性能的不是指令条数，而是**不同硬件执行管道的利用率均衡（Pipeline Balancing）**！
   - 在 `FlashAttention` 主循环中：
     - **Tensor Core (`WGMMA / tcgen05`)**：满负荷运转算 $Q K^T$ 与 $P V$；
     - **超越函数单元 (`SFU / MUFU`)**：已经被 Online Softmax 的 `MUFU.EX2`（$\exp_2$）塞得水泄不通（`100%` 饱和）；
     - **通用浮点算术单元 (`FP32 ALU / FFMA Pipe`)**：其硬件吞吐能力是 `SFU` 的 **`8 倍`（H100 每个 SM 每周期可执行 128 次 FP32 FMA，却只能执行 16 次 SFU 运算）**，而在纯矩阵乘期间大部分时间处于半闲置状态！
   - 因此，把 $\tanh(x)$ 从已经堵死的 `SFU` 管道挪到宽敞闲置的 **`FP32 FFMA` 管道**上执行，不仅消除了 `SFU` 翻倍拥堵，还能与下一轮 `WGMMA` 及 `MUFU.EX2` 实现指令级并行（ILP）重叠！
2. **如何设计零精度损失的 $\tanh(x)$ 有理帕德近似（Rational Padé / Estrada-Clausen Approximation）？**
   - 当 $|x| \ge 4.97$ 时，$\tanh(x) = \pm 1.0$（误差 $< 10^{-4}$）。先用 `fminf(fmaxf(x, -4.97f), 4.97f)` 对输入做对称截断；
   - 在 $[-4.97, 4.97]$ 区间内，利用奇函数对称性 $\tanh(x) = x \cdot \frac{P(x^2)}{Q(x^2)}$：
     - 经典 `[3/2]` 阶连分式（Lambert 帕德近似）：$\tanh(x) \approx \frac{x (27 + x^2)}{27 + 9x^2}$ 或 `[5/4]` / `[7/6]` 阶极小极大有理分式：
       $$\text{num} = x \cdot (c_1 + x^2 (c_3 + x^2 (c_5 + c_7 x^2))), \quad \text{den} = d_0 + x^2 (d_2 + x^2 (d_4 + d_6 x^2))$$
     - 全程只需调用嵌套的硬件融合乘加指令 **`fmaf(a, b, c)` (`FFMA`)**！
3. **常数折叠绝技：把软截断前后的两次额外乘法彻底消灭！**
   - 在朴素实现中，每个注意力矩阵元素都要算：
     ① $x = S \times \frac{1}{\sqrt{d}} \times \frac{1}{\text{cap}}$（2 次乘法）；
     ② $y = \tanh(x)$；
     ③ $z = y \times \text{cap} \times \log_2(e)$（为了喂给 `exp2f` 算 Softmax，又要 2 次乘法）！
   - **前后折叠（Pre/Post Scale Folding）**：
     - **前向折叠进 `Q`**：在把 `Q` 加载进片上共享内存（SMEM）之前，直接把 $\alpha_{\text{pre}} = \frac{1}{\sqrt{d} \cdot \text{cap}}$ 乘在 `Q` 上（复杂度从 $\mathcal{O}(N^2)$ 降为 $\mathcal{O}(N)$）——这样 `WGMMA` 算出的 $Q_{\text{scaled}} K^T$ **天生就已经是除以过 $\text{cap}$ 的 $x$ 了，内循环省掉 1 次全矩阵乘法**！
     - **后向折叠进帕德分子系数**：把 $\beta_{\text{post}} = \text{cap} \cdot \log_2(e)$ 直接预先乘进帕德多项式分子的常数系数 $(c_1', c_3', c_5', c_7') = \beta_{\text{post}} \cdot (c_1, c_3, c_5, c_7)$ 里——多项式一算出来直接就是乘好 $\text{cap} \cdot \log_2(e)$ 的值，立刻直接送入 `MUFU.EX2`，**内循环再省掉 2 次全矩阵乘法**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：能不能干脆用最简单的分段线性硬截断 `clamp(S, -50.0, 50.0)`（Tier 3）代替平滑的 `50.0 * tanh(S / 50.0)`？
- **在推理或微调已经用 `tanh` 预训练好的模型（如 `Gemma-2/3`）时，绝对不能换成硬截断 `clamp`！**
  因为当原始分数 $S = 25.0$（即 $S/\text{cap} = 0.5$）时：
  - 硬截断给出的值是 $25.0$；
  - 而 $50 \cdot \tanh(0.5) = 50 \times 0.4621 = \mathbf{23.105}$！
  两者相差近 **`1.9`（经 $\exp(1.9)$ 指数放大后，注意力相对权重相差整整 `6.6 倍`！）**，直接破坏模型预训练好的注意力温度分布（导致注意力输出产生 `0.042` 的巨大偏差，长文本困惑度显著劣化）！必须使用精度 `< 10^{-4}` 的帕德/多项式近似！

### Q2：帕德有理分式 $\frac{\text{num}(x)}{\text{den}(x)}$ 最后那一下除法，如果调了慢速除法岂不是又堵回 `SFU`（`MUFU.RCP`）了？
- **两种工程规避方案**：
  1. 在 `SM90/SM100` 上使用单周期近似倒数或将区间分段映射为**纯奇次契比雪夫/Remez 多项式（Pure Odd-Degree Minimax Polynomial on $|x| \le 3.5$ with Early Saturation）**，做到 **100% 零 `MUFU` 指令（纯 `FFMA`）**；
  2. 在 `SM100 Blackwell`（如 `FlashAttention-4`）中，利用双元素打包指令 **`FFMA2`（Packed FP32x2 FMA）** 在单周期内同时计算两个位置的霍纳（Horner）多项式，彻底跑满 ALU 吞吐！

### Q3：在反向传播（Backward Pass）中，软截断 $S' = \text{cap} \cdot \tanh(S / \text{cap})$ 的导数怎么高效融合？
- 根据双曲正切求导法则：
  $$\frac{\partial S'}{\partial S} = \text{cap} \cdot \frac{1}{\text{cap}} \left(1 - \tanh^2\left(\frac{S}{\text{cap}}\right)\right) = 1 - \tanh^2\left(\frac{S}{\text{cap}}\right)$$
  看到有多妙了吗？前后的 $\text{cap}$ 和 $\frac{1}{\text{cap}}$ **在求导时完美约掉了**！在反向传播重计算 $S'$ 时，算出的 $t = \tanh(S/\text{cap})$ 既用于重算正向注意力概率 $P$，同时只需 **1 条 `fmaf(-t, t, 1.0f)` 指令** 就能当场得出导数 $\text{d}S = \text{d}S' \cdot (1 - t^2)$，几乎零额外开销！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 `FlashAttention` 注意力软截断（`Soft-Capping`）算子选型决策树

```mermaid
flowchart TD
    A["部署/训练开启 attn_logit_softcapping (Gemma-2/3, Grok: cap * tanh(S / cap))"] --> B{"tanh 软截断在哪里执行?"}
    B -- "回退至未融合 PyTorch: 先算 Q@K^T 写回 HBM 再调 torch.tanh" --> C["⛔ Tier 1: 实例化 [N, N] 巨型矩阵 -> 算力跌至 78 TFLOPS 且长文必爆 OOM!"]
    B -- "融合在 FlashAttention 片上寄存器内执行" --> D{"片上寄存器如何计算 tanh(S / cap)?"}
    D -- "直接调用硬件超越函数指令 MUFU.TANH (__tanhf)" --> E["⛔ Tier 2: 与 Softmax MUFU.EX2 争抢同一狭窄 SFU 管道 -> 算力腰斩至 275 TFLOPS!"]
    D -- "偷懒改为分段线性硬截断 clamp(S, -cap, cap)" --> F["⛔ Tier 3: 在 S=25 处产生 6.6 倍指数权重偏差 (误差 0.042) -> 模型精度崩塌!"]
    D -- "FP32 FFMA 管道运行有理帕德/多项式近似 + 预折叠 1/cap 与 cap*log2(e)" --> G["🏆 Tier 4: 100% 释放 SFU 给 EX2 -> 算力飙至 585 TFLOPS (快 2.13x, 误差 < 1.2e-5)!"]
```

### 3.2 硬件 `MUFU.TANH` 争抢 `SFU` 管道（`275 TFLOPS`）vs. `FP32 FFMA` 帕德近似 + 常数折叠（`585 TFLOPS`）对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 2: 逐元素调硬件 MUFU.TANH (275.0 TFLOPS, 137.5 ms)"]
      B1["WGMMA 算出 S"] --> B2["乘 1/cap -> 调 SFU MUFU.TANH -> 乘 cap*log2e -> 再调 SFU MUFU.EX2"] --> B3["狭窄 SFU 管道双倍排队拥死!"]
    end
    subgraph Good["🏆 Tier 4: 预缩放 Q + FP32 FFMA 帕德多项式 + 折叠系数 (585.0 TFLOPS, 64.6 ms)"]
      G1["Q 预乘 1/(sqrt(d)*cap) -> WGMMA 直接产出 x = S/cap"] --> G2["闲置 FP32 FFMA 管道算带 cap*log2e 系数的帕德近似 -> SFU 100% 专供 MUFU.EX2!"]
    end
```

### 3.3 `SM90 / SM100` 三大硬件管道（`Tensor Core` + `FP32 FFMA` + `SFU EX2`）零冲突流水时序图

```mermaid
sequenceDiagram
    participant TC as Tensor Core Pipe (WGMMA / tcgen05)
    participant ALU as FP32 CUDA Core Pipe (FFMA Padé Tanh)
    participant SFU as Special Function Unit (MUFU.EX2 Only)

    Note over TC: Pre-Scale Q once in SMEM by 1 / (sqrt(d) * cap)
    TC->>ALU: 1. Tile k WGMMA completes -> outputs pre-scaled x = (Q_scaled @ K_k^T) in registers
    par Hardware Pipeline Overlap (Zero SFU Contention!)
        TC->>TC: 2a. Immediately launch Tile k+1 WGMMA on Tensor Cores
        ALU->>SFU: 2b. Evaluate Padé polynomial with folded (cap * log2e) on FP32 FFMA pipe
        SFU->>SFU: 2c. Execute Online Softmax MUFU.EX2 on Tile k-1 (100% SFU bandwidth!)
    end
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | `Logit Soft-Capping` 算子实现方案 | 有效注意力算力 (`TFLOPS`) | `32K` Prefill 耗时 | 最大输出绝对误差 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 未融合写回 `[N, N]` 矩阵 (`torch.tanh`)** | 落盘 HBM 再跑逐元素 `tanh` | `78.0 TFLOPS` ✕ | `485.0 ms` ✕ | `0.0` (基准) | ⛔ **长序列必爆显存且慢 `7.5x`** |
| **Tier 2: `FlashAttention` 内联硬件 `MUFU.TANH`** | `MUFU.TANH` 与 `MUFU.EX2` 争抢 `SFU` | `275.0 TFLOPS` ✕ | `137.5 ms` ✕ | `1.0e-5` | ⛔ **超越函数单元拥堵导致算力腰斩** |
| **Tier 3: 粗暴硬截断 `clamp(S, -cap, cap)`** | 丢弃平滑 `tanh` 曲率 | `615.0 TFLOPS` | `61.5 ms` | `4.2e-2` (严重失真! ✕) | ⛔ **破坏注意力分布，导致长文掉点** |
| **Tier 4: `FP32 FFMA` 帕德近似 + 系数折叠 (推荐)** | **卸载至 `FFMA` 管道 + 前后缩放折叠** | **`585.0 TFLOPS` (`2.13x` 快 🏆)** | **`64.6 ms` 🏆** | **`1.2e-5` (BF16 无损 🏆)** | 🏆 **`Gemma-2/3` 与 `Grok` 生产黄金标准** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/llm_flashattention_logit_softcapping_tanh_sfu_bottleneck_pade_fma_selection/run.py`](../../experiments/llm_flashattention_logit_softcapping_tanh_sfu_bottleneck_pade_fma_selection/run.py) · 封存数据：[`results.json`](../../experiments/llm_flashattention_logit_softcapping_tanh_sfu_bottleneck_pade_fma_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/llm-flashattention-logit-softcapping-tanh-sfu-bottleneck-pade-fma-sizer.html`](../reports/llm-flashattention-logit-softcapping-tanh-sfu-bottleneck-pade-fma-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/llm-flashattention-logit-softcapping-tanh-sfu-bottleneck-pade-fma-selection-cards.html`](../reports/llm-flashattention-logit-softcapping-tanh-sfu-bottleneck-pade-fma-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Unfused [N,N] torch.tanh", "tflops": 78.0, "latency_ms": 485.0 },
      { "tier": "Tier 2: Inline Hardware SFU MUFU.TANH", "tflops": 275.0, "latency_ms": 137.5 },
      { "tier": "Tier 3: Crude Hard Clamp (Distorted)", "tflops": 615.0, "latency_ms": 61.5 },
      { "tier": "Tier 4: Padé FMA + Folded Scale", "tflops": 585.0, "latency_ms": 64.6 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "tflops": "Quantity",
    "latency_ms": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Rational Padé FMA Emulation + Folded Scaling Boosts Soft-Capped FlashAttention 2.13x (275 -> 585 TFLOPS)",
    "subtitle": "Effective H100 attention throughput (TFLOPS) at 32K context with softcap=50.0 (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "tflops",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有部署或训练 `Gemma-2 / Gemma-3`、`Grok` 以及启用了 `attn_logit_softcapping` 的注意力内核（`FlashAttention-3 / FlashInfer / Triton FMHA`）。
- **⛔ Not for**：在 `FlashAttention` 内循环里直接调用硬件 `__tanhf()` (`MUFU.TANH`)，或为了省事禁用 FlashAttention 回退到 Eager Attention。

### 生产核验 Checklist
- [ ] **将 $\frac{1}{\sqrt{d} \cdot \text{cap}}$ 预缩放折叠进 `Q` 张量加载阶段**：消除内循环 $\mathcal{O}(N^2)$ 的除法/乘法缩放。
- [ ] **使用 `FP32 FFMA` 有理帕德/多项式近似替代 `MUFU.TANH`**：将 `SFU` 100% 留给 Online Softmax 的 `MUFU.EX2`。
- [ ] **将后置缩放系数 $\text{cap} \cdot \log_2(e)$ 直接乘入多项式分子的常数系数中**：一步输出 `exp2f` 所需尺度，并在反向传播中使用 $1 - t^2$ 融合求导。
