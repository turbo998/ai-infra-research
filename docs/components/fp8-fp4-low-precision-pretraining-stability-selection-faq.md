# 大模型低精度预训练与持续微调（FP8 / NVFP4 vs. BF16）数值稳定性与吞吐架构选型指南（`BF16 基线` vs. `粗粒度 Per-Tensor FP8` vs. `DeepSeek-V3 分块 FP8` vs. `Blackwell MXFP8/NVFP4`）

> **阅读指南（3 分钟速通 / 10 分钟白板宣讲）**：本文面向 AI 解决方案架构师（AI Solution Architect）与大模型预训练/后训练基础设施负责人，用第一性原理与通俗大白话拆解低精度训练（Low-Precision Pretraining & Continual Pretraining）最核心的三大工程谜题：**“为什么很多团队在 H100/H200 上直接打开初代 TransformerEngine 的 `Per-Tensor FP8` 训练，跑到几千亿 Token 就会突然遭遇 Loss Spike 甚至 NaN 崩溃回滚？为什么 Hopper GPU 硬件 Tensor Core 的 FP8 累加器（`WGMMA`）其实只有 13 位尾数精度，而 DeepSeek-V3 是怎么靠‘`1×128` 切片 + 每 128 步搬回 CUDA Core 做 FP32 累加’实现 `< 0.08%` 无损收敛并省下 149 万美元算力费的？到了 Blackwell B200/GB200 时代，原生 `MXFP8` 与 `NVFP4 + 随机舍入（Stochastic Rounding）` 又带来了什么降维打击？”**
>
> 🔗 **English Version**: [LLM Low-Precision Pretraining & Continual Pretraining (FP8 / NVFP4 vs. BF16) Selection FAQ (`fp8-fp4-low-precision-pretraining-stability-selection-faq.en.md`)](./fp8-fp4-low-precision-pretraining-stability-selection-faq.en.md) · **7-Card 评测报告**: [`fp8-fp4-low-precision-pretraining-stability-selection-cards.html`](../reports/fp8-fp4-low-precision-pretraining-stability-selection-cards.html) · **交互式工作台**: [`llm-inference-engine-pd-sizer.html#low-precision-train-sizer`](../reports/llm-inference-engine-pd-sizer.html)

![大模型低精度预训练 (FP8 / NVFP4 vs. BF16) 数值稳定性与吞吐选型一图流总结](../../assets/svg/fp8-fp4-low-precision-pretraining-stability-selection-faq-summary.svg)

---

## ① 📌 核心结论速览（Executive Summary）

1. **严禁在大模型预训练中裸跑 `粗粒度 Per-Tensor 延迟缩放 FP8 (Tier 2)`**：
   - 在 `671B MoE（37B 激活，10 万亿 Tokens 预训练）` 基准测试中，**Tier 2（整张矩阵共用 1 个历史 `amax` 缩放系数）** 虽然比 **Tier 1（BF16 基线）** 快 `1.35x`，但在 SwiGLU 激活层与 MoE 门控后期频繁遭遇 **`250x～1,000x` 通道级离群值（Activation Outliers）**，导致信噪比（SQNR）跌至 **`-1.4 dB`**，每训练 1T Tokens 发生 **`11 次` Loss 爆炸回滚**，最终验证集 Loss 劣化 **`+1.85%`**！
2. **Hopper (H100 / H200 / H800) 集群唯一生产级黄金标准：`DeepSeek-V3 细粒度分块 FP8 + CUDA 核 FP32 累加提升 (Tier 3)`**：
   - **Tier 3** 通过两大底层突破彻底根治了 FP8 训练不稳定性：① **细粒度在线分块量化**（激活与反向梯度按 **`1 × 128` Token 连续切片**、权重按 **`128 × 128` 子块** 独立计算缩放因子，将离群值锁死在 128 元素微笼子里）；② **周期性高精度累加提升**（每执行 4 次 `WGMMA` 即 $N_C=128$ 步，将 Tensor Core 内部仅 13 位尾数的部分和搬运至 **CUDA Core 的 23 位标准 FP32 寄存器**累加）。
   - 实测吞吐提升 **`1.48x`**（`4,366` vs. `2,950 tok/s/GPU`），激活与 MoE All-to-All 通信显存立降 **`-47.8%`**（`28.2 GB` vs. `54.0 GB`），10T Tokens 全程 **`0 次` 回滚**，相对 BF16 的 Loss 误差仅 **`0.07%`（无损收敛）**，直接省下 **`$1,490,000` 算力账单**！
3. **Blackwell (B200 / GB200) 新建集群首选 `硬件原生微缩放 MXFP8 / 混合 NVFP4 (Tier 4)`**：
   - **Tier 4** 利用第五代 Tensor Core 原生硬件支持的 **`1 × 32` 微块缩放（MXFP8）** 与 **`1 × 16` 微块缩放（NVFP4）**，配合**随机哈达玛变换（Random Hadamard Transform）抹平离群值**与**硬件随机舍入（Stochastic Rounding）消除梯度量化偏差**，将单卡吞吐推至 **`2.15x`（`6,342 tok/s/GPU`）**，10T 预训练成本从 `$4.59M` 砍半至 **`$2,135,000`（净省 `$245.5 万`）**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（FAQ-Driven）

### Q1：为什么初代 Per-Tensor FP8 训练到大模型中后期会突然 Loss 爆炸？什么是“离群值拉爆全局比例尺”？

- **通俗大白话（班级身高按姚明缩放的灾难）**：
  - `FP8 (E4M3)` 只有 8 个比特，最大能表示的正数是 **`448`**。
  - **粗粒度 Per-Tensor 量化（Tier 2）**：好比给全校 10,000 名学生量身高，全校共用**唯一一把比例尺**。突然校队来了一个身高 `1,000 米` 的巨人（**SwiGLU 激活函数在训练中后期必然出现的 Outlier 离群通道，数值高达 $10^3 \sim 10^4$**）。
  - 为了不让这个巨人溢出 `448`，Per-Tensor 算法被迫把整张矩阵同时除以 `3,000`！结果巨人被缩放到 `333` 保住了，但班里其他 **9,999 个正常身高 `1.7 米` 的普通学生除以 `3,000` 后全部变成了 `0.0005`——在 FP8 下直接下溢（Underflow）归零**！一整层梯度瞬间变成全 0，紧接着下一步权重更新失控触发 NaN 崩溃！
- **DeepSeek-V3 `1 × 128` 细粒度切片（Tier 3）是怎么破局的？**
  - 不再全校共用一把尺子，而是**每 128 个紧挨着的元素（`1 × 128 Tile`）单独配一把专属比例尺**！
  - 那个 `1,000 米` 的离群巨人最多只影响它所在的那个 128 人小分队，矩阵里其余 **`99.2%` 的分块完全不受污染**，信噪比从 `-1.4 dB` 瞬间恢复到 **`35.9 dB`**！并且因为动态范围已被微切片驯服，前向与反向传播可以**全链路统一使用精度更高的 `E4M3`（3 位尾数）**，彻底告别粗糙的 `E5M2`（仅 2 位尾数）。

### Q2：很多架构师都不知道的硬件秘密：为什么 Hopper H100/H200 的 FP8 Tensor Core 会发生“内积越长、误差越大”的漂移？DeepSeek-V3 的 `CUDA Core Promotion` 是怎么修好它的？

- **硬件底层真相（WGMMA 13-Bit 尾数截断缺陷）**：
  - 在标准 IEEE 754 `FP32` 中，小数尾数（Mantissa）有 **23 个比特**。
  - 但在 NVIDIA Hopper 架构的第四代 Tensor Core 中，为了省芯片面积和功耗，执行 `wgmma.mma_async`（FP8 矩阵乘累加）指令时，硬件内部加法树的累加器**最高只保留了 13 位尾数（相当于降级版 FP22）**！
  - 当大模型的隐藏层维度 $K = 4096 \sim 16384$ 时，上万个乘积在只有 13 位尾数的累加器里连加，**小数点后第 14～23 位的低位有效数字被连续无情截断丢弃**，产生系统性均值漂移（误差高达 `1%~2%`）！
- **DeepSeek-V3 / DeepGEMM 的神级软件补丁（每 $N_C=128$ 步提升至 CUDA Core）**：
  1. 让 Tensor Core 只连续跑 **4 条 `WGMMA` 指令（仅累加 $N_C = 128$ 个元素）**，此时 13 位累加器还没来得及发生长程截断漂移；
  2. 立即用一条寄存器搬运指令，把这 128 个元素的中间部分和导出到 GPU 旁边的通用 **CUDA Core（算术逻辑单元 ALU）** 中；
  3. 在 CUDA Core 内部使用**货真价实的 23 位尾数标准 `FP32` 寄存器**做主累加（`main_accum += partial_sum`），而且这一步与下一轮 Tensor Core 的 `WGMMA` 在硬件流水线上完全异步重叠（Overlap）！
  4. **效果**：几乎不损失任何算力吞吐（仍达 **`1.48x`**），却把矩阵乘相对误差从 `1.8%` 压到了 **`0.07%`**！

### Q3：到了 Blackwell B200 / GB200 (Tier 4)，为什么敢直接上 4-bit (`NVFP4`) 做预训练？`随机哈达玛变换 (RHT)` 和 `随机舍入 (Stochastic Rounding)` 是什么？

- **4-bit 预训练的两座大山与破局之道**：
  1. **第一座山：16 个元素的微块里还是有离群值怎么办？→ `随机哈达玛变换（Random Hadamard Transform, RHT）`**：
     - 在量化前，给激活和权重矩阵乘以一个正交的哈达玛旋转矩阵 $H$（因为 $H \cdot H^T = I$，矩阵乘结果 $(X H)(H^T W^T) = X W^T$ 数学上完全恒等！）。这个正交旋转就像把一座尖耸入云的孤峰**均匀摊平**到周围所有元素上，让原本尖锐的离群值瞬间变成平滑的高斯分布，完美契合 4-bit 的窄量程！
  2. **第二座山：4-bit 只有 16 个刻度，微小梯度更新在四舍五入（Round-to-Nearest）时永远被抹零怎么办？→ `硬件随机舍入（Stochastic Rounding）`**：
     - 假设梯度更新量是 `0.2`，刻度是 `0` 和 `1`。普通四舍五入每次都把它变成 `0`，模型彻底停止学习（梯度消失）！
     - 而 Blackwell 硬件原生支持的**随机舍入（Stochastic Rounding）**规定：数值 `0.2` 有 **`20% 的概率向上进位为 1`，`80% 的概率向下舍入为 0`**！在成千上万步训练的统计期望（Expectation $E[Q(x)] = x$）上，它是**完全无偏估计（Zero-Bias）**，让 4-bit 依然能精准感知十万分之一的微小梯度方向！

---

## ③ 🧭 直观可视化图解（Mermaid 决策树 / 累加器架构对比 / FP8 训练流水线时序图）

### 3.1 大模型低精度预训练与持续微调（CPT）选型决策树

```mermaid
flowchart TD
    A["🧮 大模型预训练 / 持续预训练 (CPT) 低精度架构选型"] --> B{"集群 GPU 硬件代际?"}
    B -- "NVIDIA Blackwell (B200 / GB200 NVL72)" --> C["🟩 Tier 4: 硬件原生微缩放 MXFP8 (1×32) / 混合 NVFP4 (1×16)<br/>+ 随机哈达玛旋转 (RHT) + 硬件随机舍入<br/>(吞吐 2.15x · 显存省 63.9% · 10T 省 $2.455M 🏆)"]
    B -- "NVIDIA Hopper (H100 / H200 / H800) 或 AMD MI300X" --> D{"训练框架是否支持细粒度 Tile/Block 缩放?"}
    D -- "支持 (DeepGEMM / Megatron-LM Block-FP8)" --> E["🟦 Tier 3: DeepSeek-V3 细粒度分块 FP8<br/>(1×128 Act Tile + 128×128 Weight Block + Nc=128 FP32 累加提升)<br/>(吞吐 1.48x · Loss 偏差仅 0.07% · 10T 省 $1.49M 🏆)"]
    D -- "仅支持初代 Per-Tensor Delayed Scaling" --> F["⬜ 退回 Tier 1: 标准 BF16 混合精度训练<br/>(切勿强开 Tier 2 Per-Tensor FP8，防止后期 Loss 爆炸回滚 ⚠️)"]
    B -- "严禁在生产千卡集群开启" --> G["🟥 Tier 2: 粗粒度 Per-Tensor 延迟缩放 FP8<br/>(SwiGLU 离群值拉爆全局 amax · 11次/1T 回滚 · Loss 掉点 +1.85% ✕)"]
```

### 3.2 初代 Per-Tensor FP8 (Tier 2) vs. DeepSeek-V3 分块缩放 + CUDA 核 FP32 提升 (Tier 3) 对比图

```mermaid
flowchart LR
    subgraph T2["🟥 Tier 2: 初代 Per-Tensor FP8 (全局 1 个 amax + 13位累加漂移 ✕)"]
        direction LR
        P1["[4096, 16384] 整张矩阵<br/>被单个 1000x 离群值拉爆比例尺"] --> P2["99% 正常梯度下溢变 0<br/>SQNR 跌至 -1.4 dB ✕"]
        P2 --> P3["WGMMA 仅 13位尾数连加 16384 次<br/>截断漂移触发 NaN 回滚 ✕"]
    end

    subgraph T3["🟦 Tier 3: DeepSeek-V3 细粒度分块 FP8 (1×128 Tile + FP32 提升 🏆)"]
        direction LR
        D1["激活 1×128 切片 / 权重 128×128 块<br/>离群值隔离在 128 元素微笼内"] --> D2["每 4 次 WGMMA (Nc=128)<br/>导出至 CUDA Core 23位 FP32 累加"]
        D2 --> D3["SQNR 稳定 35.9 dB · 0 回滚<br/>吞吐 1.48x · Loss 误差 0.07% 🏆"]
    end
```

### 3.3 单步前向 + 反向（Fprop / Dgrad / Wgrad）细粒度分块 FP8 与高精度通信时序图

```mermaid
sequenceDiagram
    autonumber
    participant Act as 激活张量 X (BF16)
    participant Quant as 在线分块量化核 (1×128 / 128×128)
    participant TC as Tensor Core (FP8 WGMMA)
    participant ALU as CUDA Core (23-bit FP32 累加器)
    participant Opt as FP32 主权重与 AdamW 优化器

    Act->>Quant: 1. MoE Dispatch 前直接在线量化为 1×128 E4M3 Tile (All-to-All 通信量立减 50%!)
    Quant->>TC: 2. 喂入 1×128 激活切片与 128×128 权重块执行 4 条 FP8 WGMMA (Nc=128)
    TC->>ALU: 3. 导出 13-bit 内部部分和至 CUDA Core 标准 23-bit FP32 寄存器累加 (零长程漂移)
    ALU-->>Act: 4. 反向传播 Dgrad/Wgrad 同样按 1×128 分块 E4M3 计算并输出 BF16 梯度
    Act->>Opt: 5. Embedding / Router / Norm 保持 BF16/FP32，更新 FP32 Master Weights
```

---

## ④ 📊 架构师选型决策矩阵（Architect Decision Matrix）

*基准环境：**671B MoE 大模型（37B 激活参数）**，完成 **10 万亿（10T）Tokens 预训练**，集群规模 **2,048 张 GPU**（Tier 1～3 为 H200 SXM，Tier 4 为 B200 SXM），算力单价 `$2.50 / GPU-hr`。*

| 评估维度 | ⬜ Tier 1: 标准 BF16 混合精度基线 | 🟥 Tier 2: 粗粒度 Per-Tensor 延迟 FP8 | 🟦 Tier 3: DeepSeek-V3 分块 FP8 (`1×128` + FP32 提升) | 🟩 Tier 4: Blackwell 原生 MXFP8 / 混合 NVFP4 |
| :--- | :--- | :--- | :--- | :--- |
| **量化缩放粒度** | 无量化 (`BF16 E8M7`) | 整张矩阵 1 个历史 `amax` | **激活 `1×128` Tile + 权重 `128×128` Block (全 `E4M3`)** | **硬件原生 `1×32` (`MXFP8`) / `1×16` (`NVFP4`) + 哈达玛旋转 🏆** |
| **矩阵乘累加器精度** | Tensor Core 标准 `FP32` (23-bit) | Hopper `WGMMA` 13-bit 尾数 ✕ | **每 $N_C=128$ 步提升至 CUDA Core 23-bit `FP32` 🏆** | **第五代 Tensor Core 原生微块缩放 + 硬件随机舍入 🏆** |
| **单卡训练吞吐 (`tok/s/GPU`)** | `2,950 tok/s` (`1.00x`) | `3,980 tok/s` (`1.35x`) | **`4,366 tok/s` (`1.48x` 加速 🏆)** | **`6,342 tok/s` (`2.15x` 加速 🏆)** |
| **验证集 Loss 相对误差** | `0.00%` (黄金基准) | **`+1.85%` (严重掉点 ✕)** | **`0.07%` (与 BF16 重合无损 🏆)** | **`0.14%` (无损收敛 ✅)** |
| **每 1T Tokens 崩溃回滚次数** | `0 次` | **`11 次` (被迫改学习率重训 ✕)** | **`0 次` (全程稳如泰山 🏆)** | **`0 次` 🏆** |
| **激活与 MoE 通信显存占用** | `54.0 GB / GPU` | `34.5 GB / GPU` | **`28.2 GB / GPU` (`-47.8%` 🏆)** | **`19.5 GB / GPU` (`-63.9%` 🏆)** |
| **10 万亿 Token 总算力成本** | `$4,590,000` | `$3,820,000` (含回滚重算浪费) | **`$3,100,000` (净省 `$1,490,000` 🏆)** | **`$2,135,000` (净省 `$2,455,000` 🏆)** |

---

## ⑤ 🧪 简明定量算例与 Benchmark 洞察（离群值冲击下信噪比对比）

> 📁 **封存数据源与可复现脚本**：[`experiments/low_precision_fp8_fp4_pretraining_stability/run.py`](../../experiments/low_precision_fp8_fp4_pretraining_stability/run.py) · [`experiments/low_precision_fp8_fp4_pretraining_stability/results.json`](../../experiments/low_precision_fp8_fp4_pretraining_stability/results.json) · 单测 [`tests/test_low_precision_fp8_fp4_pretraining_stability.py`](../../tests/test_low_precision_fp8_fp4_pretraining_stability.py)
> *(Provenance: `synthetic` + `anonymized_reported`, `independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)*

### 5.1 激活离群值（Outlier Ratio `10× → 1,000×`）冲击下的量化信噪比（SQNR dB）

| 训练阶段与层特征 | 粗粒度 Per-Tensor FP8 (Tier 2) SQNR | 细粒度 `1×128` Tile + FP32 提升 (Tier 3) SQNR | 数值稳定性诊断 |
| :--- | :--- | :--- | :--- |
| **常规 Attention 投影层 (`10×` 离群值, $K=4096$)** | `24.5 dB` | **`38.2 dB` 🏆** | 常规层勉强可用，但 13-bit 累加已引入轻微底噪 |
| **SwiGLU FFN Down-Proj (`250×` 离群值, $K=16384$)** | **`9.8 dB` (严重失真 ✕)** | **`36.8 dB` 🏆** | SwiGLU 门控相乘放大尖峰，Per-Tensor 丢失小梯度 |
| **MoE 训练后期路由尖峰 (`1,000×` 离群值, $K=16384$)** | **`-1.4 dB` (噪声大于信号，NaN 爆炸 ✕)** | **`35.9 dB` (全程零回滚 🏆)** | **`1×128` 分块将尖峰彻底隔离，信噪比领先 `+37.3 dB`！** |

### 5.2 配套 7-Card 4:5 视觉评测报告与交互式计算器

- 🃏 **7-Card 4:5 视觉评测卡片集（CSP 自包含，零外部依赖）**：[`docs/reports/fp8-fp4-low-precision-pretraining-stability-selection-cards.html`](../reports/fp8-fp4-low-precision-pretraining-stability-selection-cards.html)
- 🧮 **独立交互式低精度预训练稳定性与成本计算器**：[`docs/reports/fp8-fp4-low-precision-pretraining-stability-sizer.html`](../reports/fp8-fp4-low-precision-pretraining-stability-sizer.html)（已同步集成至总工作台 [`docs/reports/llm-inference-engine-pd-sizer.html#low-precision-train-sizer`](../reports/llm-inference-engine-pd-sizer.html)）

---

## ⑥ 🛠️ 附录：生产环境快速核验清单（Architect Checklist）

- [ ] **1. Hopper 集群训练强制开启 `1×128 Tile / 128×128 Block` 细粒度缩放与 $N_C=128$ FP32 累加提升**：使用集成 `DeepGEMM` 或 `TransformerEngine Block-FP8 Recipe` 的算子库，彻底禁用初代 `Per-Tensor Delayed Scaling`。
- [ ] **2. 死守“四大敏感模块不降精度（Keep in BF16/FP32）”铁律**：无论使用 FP8 还是 NVFP4，**① Embedding 词嵌入层与 Output LM Head、② MoE 门控路由器（Router Gating Logits 必须用 FP32）、③ RMSNorm / LayerNorm 与 Softmax、④ AdamW 主权重（Master Weights）** 严禁量化为 FP8/FP4！
- [ ] **3. 在 MoE Dispatch 前完成在线 FP8 量化以减半跨节点通信量**：在 `DeepEP` 发送 Token 前先转化为 `1×128 FP8 (E4M3)` + 4 字节缩放因子，直接将跨机 IB/RoCE 的 `All-to-All` 流量砍半。
- [ ] **4. 将缓存的激活值（Cached Activations for Backward）存为低精度以省掉重计算（Recomputation）开销**：利用 `1×128 FP8` 存储前向激活，显存立省 `47.8%`，从而关掉耗时的 Checkpointing 重计算，进一步白捡 `15%+` MFU。
- [ ] **5. Blackwell 集群开启 NVFP4 训练必配 `随机哈达玛变换 (RHT) + 随机舍入 (Stochastic Rounding)`**：同时在训练最后 `10%` 退火阶段（Cooldown Phase）切回 `MXFP8 / BF16` 进行高精度收尾微调，锁定满血下游评测分。
