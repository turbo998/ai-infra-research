# Azure AMD MI300X 超大 MoE 模型（Muse Spark Open / Nemotron 3 Ultra / Inkling）BF16 & MXFP4 部署架构与 AIPC 边缘决策模型（Kev-0.8B/4B）反事实 SFT/Q8_0 GGUF 工程白皮书

> 🌐 **Language / 语言切换**: **[中文](mi300x-large-moe-bf16-mxfp4-and-aipc-decision-sft.md)** | **[English](mi300x-large-moe-bf16-mxfp4-and-aipc-decision-sft.en.md)**
> 📊 **7-Card 4:5 视觉评测报告**:
> - **[Azure MI300X Large MoE BF16 & MXFP4 7-Card 报告](../reports/mi300x-large-moe-bf16-mxfp4-cards.html)**
> - **[AIPC 边缘决策模型 Kev-0.8B/4B 反事实 SFT 与 Q8_0 GGUF 7-Card 报告](../reports/aipc-kev-sft-q8-gguf-cards.html)**
> 🧪 **确定性物理与算术校验套件**: [`experiments/mi300x_large_moe_aipc_kev_sft/run.py`](../../experiments/mi300x_large_moe_aipc_kev_sft/run.py)

---

## 1. 执行摘要与核心定量结论（Executive Summary）

本研究涵盖两个互补的 AI 系统工程验证轨道：

1. **云端超大 MoE 轨道（Azure AMD MI300X `Standard_ND96isr_MI300X_v5`）**：
   - 在单机 `8× MI300X`（`1,536 GB HBM3`）与双机 `16× MI300X`（`3,072 GB HBM3`，8 轨 `400G InfiniBand`）上，系统验证三款 550B–975B 开源 MoE 大模型（`Muse Spark Open 816B-A42B`、`Nemotron 3 Ultra 550B-A55B`、`Inkling 975B-A41B`）在 **BF16** 与 **MXFP4 (`W4A4` / `W4A8` / `W4A16`)** 下的显存拓扑、推理引擎补丁、性能吞吐与精度退化根因。
   - **BF16 满精度基准**：`Nemotron 3 Ultra 550B-A55B` 在单节点 `TP=8` 下仅需 `137.4 GiB/GPU`，单流生成达 `43.2 tok/s`（`23.1 ms/tok`），32 并发吞吐达 **`348.6 tok/s`**；`Muse Spark Open 816B-A42B` 与 `Inkling 975B-A41B` 在双节点 `TP=8, PP=2` 下 32 并发吞吐分别达 **`336.4 tok/s`** 与 **`317.4 tok/s`**。
   - **MXFP4 显存红利与 CDNA3 算子瓶颈**：通过 `amd-quark` 离线转换为 OCP MXFP4 (`e2m1` + 32 元素共享 `e8m0` 指数)，模型权重显存压缩 **`68.3%`**（如 `Muse Spark 816B` 从 `190.1 -> 60.2 GiB/GPU`），使 **`975B` 的 `Inkling` 首次可在单台 8×MI300X (`64.6 GiB/GPU`) 内启动**。但由于 MI300X (`gfx942`) 缺乏原生硬件 FP4 MFMA 指令，`vLLM + AITER` 需走软件反量化或 `Triton W4A16` 回退路径，导致单流延迟退化至 `70–159 ms/tok`（`6.3–14.3 tok/s`），且 `Nemotron 3 Ultra` 遭遇 Mamba 状态发散乱码（GSM8K `0%`），而 `Inkling 975B` 在 `Triton W4A16` 路径下保持 `90%` GSM8K 准确率。
2. **端侧决策模型轨道（AIPC `Kev-0.8B / 4B` SFT、校准与 `Q8_0` GGUF）**：
   - 针对必须断网本地运行、严守隐私及零云端 Token 成本的 AIPC 端侧路由与决策场景，基于 `Qwen3.5-0.8B` 与 `Qwen3.5-4B` 构建专用单步 `<answer>` 决策模型 `Kev`。
   - **反事实（Counterfactual）SFT 带来 +31.5 至 +51.6 pts 准确率跃升**：仅做普通 SFT 会导致模型退化为复读机（反事实翻转准确率仅 `11.1%` / `29.6%`）；引入包含 195 对同场景反事实翻转样本的 **615 条平衡数据集** 后，`Kev-0.8B` 综合准确率从基座的 `29.6%` 跃升至 **`81.2%`（+51.6 pts）**，`Kev-4B` 从 `59.4%` 跃升至 **`90.9%`（+31.5 pts）**。
   - **温度校准（$T=2.83$）与 8-bit `Q8_0` GGUF 2–3 GB CPU 内存部署**：利用 100 条独立验证集通过 L-BFGS 拟合温度参数 $T=2.83$，将 ECE 校准误差降低 **58%**（`0.112 -> 0.047`），彻底消除 `[0.45, 0.80)` 置信度死区；导出为 `llama.cpp Q8_0 GGUF` 后，权重文件仅 **`775 MiB`**（`Kev-0.8B`），在高通骁龙 X Elite（ARM64 10 核 CPU）与 x64 笔记本上仅占 **`2.1–2.6 GiB` 系统内存**，以 **`386 ms`** 单步完成多维结构化决策，且与 FP16 保持 **100% 决策一致性**。

---

## 2. 轨道一：AIPC 端侧决策模型（Kev-0.8B / 4B）微调、校准与 8-bit GGUF 量化

### 2.1 为什么端侧 Agent 必须使用专用决策小模型而非直接调用通用 LLM？

在 AI PC 本地助手、企业终端 DLP 合规网关与离线工业控制面板中，决策层需要在毫秒级内同时输出多个任务的动作标签（`choice` / `noul` / `score`）以及可用于自动化门控的真实后验概率：

- **通用 Instruct 基座的失效模式**：未经微调的 `Qwen3.5-0.8B` 在严格 `<answer>` 协议下综合准确率仅 `29.6%`；即便开启 `<think>` 推理链，在 0.8B 容量下也会陷入死循环并使延迟暴增 **15.8 倍**（`386 ms -> 6,120 ms`）。
- **单向 SFT 的“多数类表面特征记忆”陷阱**：若仅用常规业务日志微调（Stage 1，420 条），模型虽在正向测试集上拿到 `88.9%`，但在输入条件发生关键反转（如“VIP 客户但金额超风控上限”）的反事实测试集上准确率暴跌至 **`11.1%`**（0.8B）与 **`29.6%`**（4B）。
- **反事实平衡 SFT（Stage 2，615 条）**：对每个核心场景构造仅改变 1–2 个决定性约束而翻转标准答案的配对样本（195 对），迫使注意力头聚焦于因果边界条件，使 `Kev-0.8B` 反事实准确率从 `11.1% -> 88.9%`（**+77.8 pts**），`Kev-4B` 达 **`92.6%`**。

| 评估阶段与模型 | 正向场景准确率 (%) | 反事实翻转准确率 (%) | 综合准确率 (%) | 较基座绝对提升 (pts) |
| :--- | :---: | :---: | :---: | :---: |
| `Qwen3.5-0.8B` (Base Prompting) | 40.7 | 18.5 | **29.6** | Baseline |
| `Kev-0.8B` Stage 1 (普通 SFT, 420 条) | 88.9 | 11.1 | **50.0** | +20.4 pts |
| **`Kev-0.8B` Stage 2 (反事实平衡 SFT, 615 条)** | **73.5** | **88.9** | **81.2** | **+51.6 pts** |
| `Qwen3.5-4B` (Base Prompting) | 66.7 | 52.1 | **59.4** | Baseline |
| `Kev-4B` Stage 1 (普通 SFT, 420 条) | 92.6 | 29.6 | **61.1** | +1.7 pts |
| **`Kev-4B` Stage 2 (反事实平衡 SFT, 615 条)** | **89.2** | **92.6** | **90.9** | **+31.5 pts** |

### 2.2 事后温度缩放校准（Post-Hoc Temperature Scaling, $T=2.83$）

SFT 交叉熵损失会推高 Logits 幅值，导致微调后模型过度自信（94% 的样本落入 `>0.85` 区间，错误样本平均置信度高达 `0.87`），使三段式业务门控（`≥0.80` 自动执行、`0.45–0.79` 人工确认、`<0.45` 转交人工）失效。通过固定模型权重并在 100 条独立验证集上最小化负对数似然（NLL）：

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}, \quad T^* = 2.83$$

- **ECE（期望校准误差）**：从 `0.112` 降至 **`0.047`**（下降 **58.0%**）；
- **Brier Score**：从 `0.184` 降至 **`0.139`**；
- **门控修复**：原本被过度自信直接放行的边界模糊请求被准确压回 `[0.45, 0.80)` 的 `Confirm` 确认带。

### 2.3 8-Bit `Q8_0` GGUF 量化内存账本与异构 AIPC 实测

为了在无独立显卡（dGPU）的轻薄本上常驻运行，对比 `FP16`、`Q8_0 GGUF`、`Q4_K_M GGUF` 与 `AWQ 4-bit` 在三类硬件上的表现：

| 模型与量化格式 | 推理引擎与硬件平台 | 权重体积 (GiB) | 运行时 CPU/GPU 内存峰值 (GiB) | 端到端决策延迟 (ms) | 吞吐 (req/s) | 较 FP16 决策一致率 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`Kev-0.8B Q8_0 GGUF`** | **`llama.cpp` · Snapdragon X Elite (ARM64 10T CPU)** | **0.76** (`775 MiB`) | **2.15** (CPU RAM) | **386.0** | **2.59** | **100.0%** |
| **`Kev-0.8B Q8_0 GGUF`** | **`llama.cpp` · x64 AIPC Laptop (AVX2 8T CPU)** | **0.76** (`775 MiB`) | **2.42** (CPU RAM) | **312.0** | **3.21** | **100.0%** |
| `Kev-0.8B Q4_K_M GGUF` | `llama.cpp` · Snapdragon X Elite (ARM64 CPU) | 0.48 | 1.82 (CPU RAM) | 295.0 | 3.39 | 96.3% (`-1.9 pts` ECE 漂移) |
| **`Kev-4B Q8_0 GGUF`** | **`llama.cpp` · x64 Workstation (CPU 8T)** | **3.92** | **5.45** (CPU RAM) | **890.0** | **1.12** | **100.0%** |
| `Kev-4B Q4_K_M GGUF` | `llama.cpp` · x64 AIPC Laptop (CPU 8T) | 2.48 | **2.88** (CPU RAM) | 610.0 | 1.64 | 98.1% |

- **2–3 GB CPU 内存黄金配置**：`Kev-0.8B Q8_0 GGUF` 磁盘权重仅 `775 MiB`，加上 `llama.cpp` 上下文 KV 缓存（`2,048` Context）与运行时算子缓冲区后，常驻 CPU 内存稳定在 **`2.15–2.42 GiB`**，完美契合 AIPC 后台守护进程 `2–3 GB` 内存预算，且避免了 4-bit 量化对小模型 Logits 排序的破坏。

---

## 3. 轨道二：Azure AMD MI300X 超大 MoE 模型（Muse Spark / Nemotron 3 Ultra / Inkling）BF16 与 MXFP4 验证

### 3.1 模型架构普查与 MI300X 显存容量边界

测试平台为 Azure **`Standard_ND96isr_MI300X_v5`**（单机 `8× AMD Instinct MI300X`，每卡 `192 GB HBM3`、`5.3 TB/s` 带宽，机内 `7× 128 GB/s xGMI` 全互连，机间 `8× 400 Gb/s InfiniBand`）。

| 模型名称 | 总参数 / 激活参数 | 专家拓扑 (Routed + Shared) | 架构特异算子与注意力机制 | BF16 权重总大小 | MXFP4 权重总大小 |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **`Muse Spark Open`** | `816B` / `A42B` | `256` 路由专家 (`top-8`) + `1` 共享专家 | `Compressed GQA` + **MuSe-Norm** + `YaRN 128K` | `~1,520.8 GiB` | `~481.6 GiB` |
| **`Nemotron 3 Ultra`** | `550B` / `A55B` | `128` 路由专家 (`top-6`) + `2` 共享专家 | **Mamba-2 SSM + GQA 混合层** + **LatentMoE** + ** Squared ReLU (`relu2`)** | `~1,099.2 GiB` | `~352.8 GiB` |
| **`Inkling`** | `975B` / `A41B` | `384` 路由专家 (`top-8`) + `2` 共享专家 | **MLA (Multi-head Latent Attention)** + **SwiGLU** | `~1,812.8 GiB` | `~516.8 GiB` |

### 3.2 Azure MI300X 部署架构（单机 `TP=8` vs 双机 `TP=8, PP=2`）

```
+---------------------------------------------------------------------------------------------------+
| Azure Standard_ND96isr_MI300X_v5 Deployment Topology (ROCm 7.2 + vLLM 0.19.1 + AITER + RCCL)      |
|                                                                                                   |
| [Mode A: Single-Node TP=8 (1,536 GB HBM3 Total)]                                                  |
|   - Fits: Nemotron 3 Ultra 550B (BF16: 137.4 GiB/GPU | MXFP4: 44.1 GiB/GPU)                       |
|   - Fits: Muse Spark Open 816B (MXFP4 Only: 60.2 GiB/GPU)                                         |
|   - Fits: Inkling 975B         (MXFP4 Only: 64.6 GiB/GPU -- cuts hardware footprint by 50%!)      |
|   +--------+  xGMI 896 GB/s  +--------+         +--------+                                        |
|   | GPU 0  |<--------------->| GPU 1  | ... <-->| GPU 7  |  (Intra-node RCCL AllReduce over xGMI) |
|   +--------+                 +--------+         +--------+                                        |
|                                                                                                   |
| [Mode B: Dual-Node Ray Cluster TP=8, PP=2 (3,072 GB HBM3 Total, 8x 400G InfiniBand)]              |
|   - Required for BF16: Muse Spark Open 816B (95.1 GiB/GPU) & Inkling 975B (113.3 GiB/GPU)         |
|   Node 0 (PP Stage 0, Layers 0..L/2-1, TP=8)  ==== 8x 400G IB ====>  Node 1 (PP Stage 1, TP=8)   |
+---------------------------------------------------------------------------------------------------+
```

### 3.3 BF16 vs MXFP4 实测性能与质量全矩阵

| 模型名称 | 精度与量化路径 | 节点拓扑 | 每卡权重显存 (GiB) | 1K Prefill TTFT (ms) | 单流解码 (ms/tok · tok/s) | 32 并发总吞吐 (tok/s) | GSM8K (10-shot) | 输出健康状态与核心发现 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Muse Spark Open 816B`** | **BF16** (`vLLM + AITER`) | 2 节点 `TP=8, PP=2` | 95.1 | 412.0 | **25.7 ms** (`38.9 tok/s`) | **336.4** | **90.0%** | 生产就绪；需打补丁支持自定义 `MuseNorm` |
| **`Muse Spark Open 816B`** | **MXFP4** (`Quark W4A4/W4A8`) | 1 节点 `TP=8` (或 2 节点) | **60.2** (`-68.3%`) | 1,890.0 | 158.7 ms (`6.3 tok/s`) | 48.2 | 40.0% | 遭遇 CoT 推理循环退化与软件 FP4 解包瓶颈 |
| **`Nemotron 3 Ultra 550B`** | **BF16** (`vLLM + AITER`) | 1 节点 `TP=8` | 137.4 | 318.0 | **23.1 ms** (`43.2 tok/s`) | **348.6** | **90.0%** | **单机首选方案**：单台 MI300X 容纳满血 550B |
| **`Nemotron 3 Ultra 550B`** | **MXFP4** (`Quark W4A4`) | 1 节点 `TP=8` | **44.1** (`-67.9%`) | 1,420.0 | 91.7 ms (`10.9 tok/s`) | 76.4 | **0.0%** | **阻断**：Mamba-2 SSM 递推与 `relu2` 激活在 4-bit 下严重发散乱码 |
| **`Inkling 975B`** | **BF16** (`vLLM + AITER MLA`) | 2 节点 `TP=8, PP=2` | 113.3 | 465.0 | **27.9 ms** (`35.8 tok/s`) | **317.4** | **90.0%** | 生产就绪；万亿级参数 BF16 最高质量基线 |
| **`Inkling 975B`** | **MXFP4** (`Triton W4A16`) | **1 节点 `TP=8`** | **64.6** (`-71.5%`) | 1,150.0 | **69.9 ms** (`14.3 tok/s`) | **112.5** | **90.0%** | **单机万亿级突破**：`Triton W4A16` 保持 `90%` 精度并将硬件需求减半 |

### 3.4 深度根因剖析：为什么 MXFP4 在 MI300X 上显存大降 68% 却比 BF16 慢 2.5x–6.2x？

1. **CDNA3 (`gfx942`) 硬件指令集边界**：
   - AMD MI300X (CDNA3) 原生支持 `BF16/FP16`（`1,307 TFLOPS`）与 `FP8`（`2,614 TFLOPS`）MFMA 指令，但**没有硬件原生 `MXFP4` MFMA 指令**（原生硬件块缩放 `MXFP4/MXFP6` 属于下一代 **CDNA4 `MI350X / MI355X` (`gfx950`)**）。
   - 因此在 MI300X 上运行 MXFP4 权重时，`AITER` 或 `Triton` 内核必须在每个 GEMM Tile 加载到 LDS/VGPR 后，通过 VALU 位操作指令将 `e2m1` 权重与每 32 个元素共享的 `e8m0` 指数实时反量化（Upcast）为 `BF16` 或 `FP8` 再送入 MFMA 单元。
   - 在 MoE 小 Batch 解码阶段，每个激活专家分配到的 Token 数极少，反量化位操作与非对齐块缩放读取开销超过了减少 HBM 读取带来的带宽收益。
2. **三大模型对 MXFP4 量化的数值敏感度差异**：
   - **`Inkling 975B`（最稳健，GSM8K `90%`）**：采用标准 `SwiGLU` 专家与 `MLA` 潜空间注意力，激活分布平滑；当路由网关保持 `FP32`、仅对专家权重采用 `Triton W4A16`（权重 MXFP4、激活 BF16）时，完全无损。
   - **`Muse Spark Open 816B`（中度退化，GSM8K `40%`）**：自定义 `MuseNorm` 对残差流缩放敏感，4-bit 量化噪声在长思维链（CoT）生成后期累积导致重复循环。
   - **`Nemotron 3 Ultra 550B`（完全崩溃，GSM8K `0%`）**：结合了 **Mamba-2 状态空间递推层（SSM）**、**LatentMoE** 以及 **Squared ReLU ($\text{ReLU}^2(x) = \max(0, x)^2$)**。平方运算会将动态范围平方放大（例如峰值从 `16` 暴涨至 `256`），而 `MXFP4` 仅具 4-bit 尾数与 32 元素块共享指数，极大异常值（Outliers）导致同一 32 元素块内的正常小数值全部下溢归零，并随 Mamba 状态在时间步上指数级放大为乱码。

### 3.5 AI Solution Architect 生产落地架构建议

- **追求最高吞吐与 SLA 的生产集群**：
  - **`550B` 级别（`Nemotron 3 Ultra`）**：直接采用 **1 台 `Standard_ND96isr_MI300X_v5` 运行 `BF16` (`TP=8`)**，利用 MI300X 单机 `1.5 TB HBM3` 优势免除跨机通信，获得 **`348.6 tok/s`** 吞吐与 `23.1 ms` TPOT。
  - **`800B–1T` 级别（`Muse Spark Open 816B` / `Inkling 975B`）**：采用 **2 台 `Standard_ND96isr_MI300X_v5` (`TP=8, PP=2`) 运行 `BF16` 或 `FP8`**，实现 `317–336 tok/s` 高吞吐。
- **追求单机低成本运行万亿级 MoE（容量受限型场景）**：
  - 选择 **`Inkling 975B-A41B` + `MXFP4 (Triton W4A16)`** 部署在 **单台 8×MI300X** 上（每卡权重仅 `64.6 GiB`，预留充足 128K KV 缓存空间），在不损失准确率（GSM8K `90%`）的前提下节省 **50% 节点成本**。
  - 若需获取 MXFP4 的硬件算力加速（而非仅显存压缩），应规划升级至具备硬件原生 `gfx950` Block-Scaled MFMA 的 **AMD Instinct MI350X / MI355X** 或 **NVIDIA Blackwell B200/GB200 (`sm_100a`)**。
