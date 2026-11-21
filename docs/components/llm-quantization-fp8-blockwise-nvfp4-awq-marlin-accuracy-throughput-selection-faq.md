# 大模型低比特量化架构深度选型：`BF16` vs. 细粒度分块 `Block FP8 (1x128 & 128x128)` vs. `W4A16 AWQ Marlin` vs. `Blackwell NVFP4` 精度保持、吞吐拐点与硬件适配指南

> **语言 / Language**: **中文 (当前)** | [English](./llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect 与推理平台负责人的大模型量化技术栈（`BF16`、静态 `Per-Tensor FP8`、DeepSeek 细粒度 `Block FP8 1x128×128x128`、`W4A16 AWQ Marlin` 与 `Blackwell NVFP4`）硬件 TensorCore 适配、Batch Size 性能拐点与精度无损边界白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业模型部署精度与硬件选型）

![大模型低比特量化架构深度选型总结图](../../assets/svg/llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**大模型量化选型必须遵循“一看 GPU 硬件代际、二看并发 Batch Size、三看细粒度缩放因子（Scale Granularity）”三大铁律：① 在 `A100`（无原生 FP8 TensorCore）或极低并发（$B \le 16$，纯带宽受限）下，首选 **`W4A16 AWQ + Marlin 内核`**（权重压至 `0.55B/param`，单流 TPOT 仅 `11.8ms`）；② 在 `Hopper (H100 / H200)` 与 `Ada (L40S)` 中高并发（$B \ge 24$）下，首选 **细粒度分块 `Block FP8 W8A8KV8`（激活 `1x128` 逐 Token 分组 + 权重 `128x128` 分块缩放）**——既吃满 `2 倍` FP8 TensorCore 算力（`1,979 TFLOPS`）和 `2 倍` 显存/带宽收益，又彻底隔离 SwiGLU/MoE 异常值（Outliers），保持 **`99.8%` 无损精度** 与 **`2,140 Goodput tok/s/GPU`（较 BF16 提速 `2.18 倍`、降本 `60.9%`）**；③ 在新一代 `Blackwell (B200 / GB200)` 上，升级至硬件原生 **`NVFP4`（每 16 个元素共享 1 个 FP8 Micro-Block Scale，仅 `0.5625 Bytes/param`）**！**

- **交互式量化格式与 GPU 机型匹配计算器**：[打开交互式计算器 (`llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-sizer.html`)](../reports/llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-selection-cards.html`)](../reports/llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：为什么同样叫 `FP8`，早期的静态 `Per-Tensor FP8` 在数学/代码任务上会掉点 `5.8%`，而 `DeepSeek-V3/R1` 的细粒度 `Block FP8 (1x128 × 128x128)` 却能做到 `99.8%` 几乎无损？
- **根因（Activation Outlier 离群值截断灾难）**：
  - 大模型在经过 SwiGLU 门控激活和 MoE 专家路由时，极少数通道（Outlier Channels）的激活值会比普通通道大 `50 ~ 200 倍`；
  - 如果整个张量共用 **1 个全局缩放因子（Per-Tensor Scale）**，为了不让那几个极大值溢出 `FP8 E4M3` 的上限（`±448`），全局 Scale 会被迫拉得很大——结果导致占 99% 的正常小数值全部被下溢截断成 `0`（Underflow），在多步数学推理与代码生成上直接掉点 `5.8%`！
- **细粒度分块量化（Fine-Grained Tile/Block Scaling）的巧思**：
  - **激活值按 `1 × 128`（每 Token 每 128 个通道）在线动态计算 1 个 Scale**；
  - **权重按 `128 × 128` 子块静态存储 1 个 `FP32` Scale**（每 `16,384` 个字节仅多存 `4` 字节 Scale，显存额外开销仅为 $4 / 16384 = \mathbf{0.024\%}$，几乎为零！）；
  - 离群值被严格关在自己的 `1x128` 小格子里，完全不影响其他通道的动态范围，并在 TensorCore 每累加 128 次后用 CUDA Core 做高精度缩放累加，实现 **`99.8%` 精度保持率**！

### Q2：`W4A16 (INT4 权重 + FP16 激活)` 与 `W8A8 (FP8 权重 + FP8 激活)` 到底谁更快？为什么存在一个明确的 `Batch Size 拐点`？
- **`W4A16 (AWQ / GPTQ + Marlin Kernel)` 的工作原理**：
  - 权重以 `4-bit`（约 `0.55 Bytes/param`）存放在 HBM 中，从显存读入寄存器时**省了 `45%` 带宽（相比 FP8）**；但进入 TensorCore 做矩阵乘之前，必须先在寄存器里**反量化还原成 `FP16`**，调用的仍然是 **`FP16 TensorCore`（H200 上为 `989 TFLOPS`）**！
- **`W8A8 (FP8)` 的工作原理**：
  - 权重和激活都是 `8-bit FP8`（`1.0 Byte/param`），直接送入 **`FP8 TensorCore`（H200 上高達 `1,979 TFLOPS`，算力翻倍！）**。
- **Batch Size 黄金拐点定律**：
  - 当 **并发很低（$B \le 16$，纯显存带宽受限）** 时：算力绰绰有余，谁读的字节少谁就快——此时 **`W4A16 Marlin`（读 0.55B）比 `W8A8 FP8`（读 1.0B）的单流 TPOT 更快（`11.8ms vs 12.9ms`）**；
  - 当 **并发升高（$B \ge 24 \sim 32$）或处理长 Prompt Prefill（算力受限）** 时：`W4A16` 的反量化开销与减半的 `FP16` 算力成为瓶颈，而 **`W8A8 FP8 + FP8 KV` 凭借 `2 倍` TensorCore 算力反超 `W4A16`，系统总 Goodput 高出 `39%`（`2,140 vs 1,540 tok/s/GPU`）**！

---

## 3. 🧭 直观可视化图解（Mermaid 量化选型决策树 + 精度/字节对比）

### 3.1 企业大模型量化方案选型决策树
```mermaid
flowchart TD
    A["企业大模型量化方案选型"] --> B{"目标 GPU 硬件代际？"}
    B -- "NVIDIA A100 / A10G (Ampere)<br/>无原生 FP8 TensorCore!" --> C["首选: W4A16 AWQ + Marlin 内核<br/>(或 W8A8 INT8 SmoothQuant)<br/>• 严禁在 A100 上跑 FP8 (退化为软反量化)"]
    B -- "NVIDIA H100 / H200 / L40S<br/>(Hopper / Ada 原生 FP8)" --> D{"业务并发 Batch Size 与显存水位？"}
    D -- "中高并发生产主流 (B >= 24)<br/>或严苛代码/数学推理" --> E["🏆 首选: 细粒度 Block FP8 (W8A8KV8)<br/>(1x128 激活 × 128x128 权重)<br/>• 99.8% 无损精度 + 2.18x Goodput"]
    D -- "单卡极限塞大模型 / 低并发 (B <= 16)" --> F["可选: W4A16 AWQ Marlin + FP8 KV<br/>• 权重显存再省 45%, 低并发 TPOT 极快"]
    B -- "NVIDIA B200 / GB200 (Blackwell)" --> G["🏆 首选: 原生 NVFP4 (Micro-Block 16-Elem Scale)<br/>• 0.5625 Bytes/param + 9,000 TFLOPS 算力!"]
```

---

## 4. 📊 架构师多方案选型决策矩阵（主流精度格式全方位对比）

| 量化方案 | 每参数字节数 | 调用的硬件 TensorCore | 支持的 GPU 架构 | 基准精度保持率 | 单卡达标 Goodput (`H200`) | 最佳适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1：`BF16 (W16A16KV16)`** | `2.00 Bytes` | BF16 TensorCore (`1x` 算力) | A100 / H100 / H200 / B200 | **`100.0%`（基准）** | `980 tok/s` (`$0.92/1M`) | 医疗/金融高敏基线对照 |
| **Tier 2：粗放静态 `Per-Tensor FP8`** | `1.00 Byte` | FP8 TensorCore (`2x` 算力) | L40S / H100 / H200 / B200 | `94.2%` ✕（掉点严重） | `1,950 tok/s` (`$0.46/1M`) | ❌ 淘汰（被细粒度 Block FP8 取代） |
| **Tier 3：`W4A16 AWQ + Marlin`** | **`0.55 Bytes`** | FP16 TensorCore (`1x` 算力) | **A100** / L40S / H100 / H200 | `98.4%` ✓ | `1,540 tok/s` (`$0.54/1M`) | **`A100` 集群、低并发 ($B\le 16$) 或单卡塞大模型** |
| **Tier 4（推荐）：细粒度 `Block FP8 (W8A8KV8)`** | **`1.00 Byte`** | **FP8 TensorCore (`2x` 算力)** | **L40S / H100 / H200 / B200** | **`99.8%` ✓（近乎无损）** | **`2,140 tok/s` (`$0.36/1M`)** | **`H100/H200/L40S` 企业生产绝对主力** |
| **Blackwell 旗舰：`NVFP4 (W4A4KV8)`** | **`0.56 Bytes`** | **FP4 TensorCore (`4x` 算力)** | **B200 / GB200 NVL72** | **`99.1%` ✓** | **`4,150+ tok/s` (`B200`)** | **`B200 / GB200` 下一代超高吞吐底座** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **硬件与量化格式严格对齐**：`A100` 严禁部署 `FP8` 模型；`H100/H200/L40S` 优先部署 `Block FP8`；`B200/GB200` 优先评估 `NVFP4`。
- [ ] **优先选用带 `1x128 / 128x128` 细粒度 Block Scale 的 FP8 权重**：避免静态 Per-Tensor FP8 在复杂推理与工具调用 JSON 上出现隐性退化。
- [ ] **KV Cache 量化至 `FP8 E4M3` 时保留 Per-Head/Per-Channel Scale**：严禁在超长上下文（`>64K`）无校准直开 `INT4 KV`（易导致大海捞针召回率崩塌）。
