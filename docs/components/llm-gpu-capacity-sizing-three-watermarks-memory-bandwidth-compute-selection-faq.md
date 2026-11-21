# 如何精确估算一个大模型“需要多少张 GPU 放得下 & 跑得动”？三道水位线（显存容量 × 带宽 TPOT × 算力与并发 KV 池）手算方法论与 `H100 / H200 / B200` 配平指南

> **语言 / Language**: **中文 (当前)** | [English](./llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect、容量规划与预售架构师的大模型 GPU 算力精确评估、三道水位线（`HBM 容量` × `显存带宽 TPOT` × `Prefill 算力与并发 KV 池`）手算公式与 `H100 (80GB) / H200 (141GB) / B200 (192GB)` 选型配平白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业云架构容量规划与成本估算）

![大模型 GPU 容量评估三道水位线手算方法论总结图](../../assets/svg/llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**回答客户“跑这个模型到底要买多少张 GPU？”时，绝对不能只用 `参数量 × 2 字节` 算权重显存！企业级生产容量规划必须同时跨过“三道水位线”并取最大值：$$N_{\text{GPU}} = \max\Big(\underbrace{N_{\text{mem}}}_{\text{第一道：放得下 (HBM 容量)}},\ \underbrace{N_{\text{bw}}}_{\text{第二道：吐得快 (带宽保 TPOT)}},\ \underbrace{N_{\text{compute}}}_{\text{第三道：扛得住 (算力保 TTFT/QPS)}}\Big)$$ 尤其注意两条架构黄金定律：① 对于 MoE 模型（如 `DeepSeek-V3/R1 671B`），算第一道“显存容量”必须用总参数量（`671B`），而算第二道“Decode 带宽 TPOT”和第三道“Prefill 算力”只用单 Token 激活参数量（`37B`）；②“放得下 $\neq$ 跑得快”——单张 `H200 (141GB)` 虽能装下 `70B FP8 (70GB)`，但在 32 并发下 TPOT 高达 `32.4ms`，必须扩到 `TP=2/4` 聚合多卡 HBM 带宽才能跨过 `<20ms` 交互红线！**

- **交互式三道水位线 GPU 容量手算器**：[打开交互式计算器 (`llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-sizer.html`)](../reports/llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-selection-cards.html`)](../reports/llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：第一道水位线（放得下 · 显存容量 $N_{\text{mem}}$）怎么精确手算？`MHA / GQA / MLA` 三种注意力机制的 KV Cache 差多少倍？
单副本总显存需求由 **四部分** 组成，且必须除以 **`0.85`（预留 15% 安全水位给 PyTorch 分配器碎片与突发缓冲）**：
$$N_{\text{mem}} = \left\lceil \frac{M_{\text{weights}} + M_{\text{KV}}(B, S) + M_{\text{act}} + M_{\text{cudagraph}}}{0.85 \times \text{HBM}_{\text{per\_GPU}}} \right\rceil_{\text{向上对齐至 } 1, 2, 4, 8, 16\dots}$$

1. **模型静态权重显存（$M_{\text{weights}}$）**：
   $$M_{\text{weights}} = P_{\text{total}} \times \text{bytes\_per\_param}$$
   - `BF16 / FP16`：`2.0 Bytes/param`（如 `70B = 140 GB`，`405B = 810 GB`）；
   - `FP8 (W8A8 / W8KV8)`：**`1.0 Byte/param`**（含 Scale 约 `1.02~1.05B`，如 `70B = 70 GB`，`405B = 405 GB`，**`DeepSeek-R1 671B = 671 GB`**）；
   - `INT4 (AWQ / GPTQ)`：**`0.55 Bytes/param`**（含 Group=128 量化零点与 Scale 开销，如 `70B ≈ 38.5 GB`）。
2. **动态并发 KV Cache 显存（$M_{\text{KV}}$）——最容易算漏导致上线即 OOM 的杀手！**：
   - **标准 `MHA` / `GQA` 精确公式**（$L$ 为层数，$n_{kv}$ 为 KV 头数，$d_h$ 为单头维度，$S$ 为平均序列总长，$B$ 为并发活跃请求数）：
     $$M_{\text{KV}}^{\text{GQA}} = B \times S \times \Big(2 \times L \times n_{kv} \times d_h \times \text{bytes}_{\text{KV}}\Big)$$
     - **`Llama-3.3-70B` ($L=80, n_{kv}=8, d_h=128$)**：每 Token KV 大小 = $2 \times 80 \times 8 \times 128 = 163,840$ 个元素 $\implies$ **`BF16 = 327.68 KB/tok`（`0.3125 MiB/tok`）**，**`FP8 = 163.84 KB/tok`**（若 $B=64, S=16\text{K}$，FP8 KV 需 **`167.8 GB`**！）；
     - **`Llama-3.1-405B` ($L=126, n_{kv}=8, d_h=128$)**：**`BF16 = 516.1 KB/tok`**，**`FP8 = 258.05 KB/tok`**。
   - **`MLA`（Multi-head Latent Attention，如 `DeepSeek-V3/R1 671B`，$L=61$）**：
     - 将每层 K 和 V 联合低秩压缩为潜向量 $d_c = 512$ + 解耦 RoPE 向量 $d_r = 64$（每层每 Token 仅存 **`576` 个元素**，而无需乘 $2 \times n_{kv} \times d_h$！）：
       $$M_{\text{KV}}^{\text{MLA}} = B \times S \times \Big(L \times (d_c + d_r) \times \text{bytes}_{\text{KV}}\Big)$$
     - **`DeepSeek-R1 671B` ($L=61, d_c+d_r=576$)**：每 Token KV 仅 $61 \times 576 = 35,136$ 个元素 $\implies$ **`BF16 = 70.27 KB/tok`**，**`FP8 = 35.14 KB/tok`**（比 `70B GQA` 还省 **`4.7 倍`**，比传统 `MHA` 省 **`28 倍`**！）。
3. **激活值 + CUDA Graph + NCCL 缓冲区（$M_{\text{act}} + M_{\text{cudagraph}}$）**：
   - 启用 PagedAttention + Chunked Prefill（`max_num_batched_tokens = 2048~4096`）后，激活值与 CUDA Graph 捕获图每张卡固定占用约 **`4 ~ 8 GB`**（整机 8 卡预留 `12 ~ 32 GB`）。

### Q2：第二道水位线（吐得快 · 显存带宽约束 $N_{\text{bw}}$）怎么手算？为什么“1 张卡装得下”却必须拆成 2~4 张卡跑？
- **物理定律**：Decode 阶段每生成 1 个新 Token，GPU 必须把**整份激活权重 $M_{\text{active\_weights}}$** 加 **当前活跃 Batch 的 KV Cache $M_{\text{KV\_active}}$** 从 HBM 完整读进 SM 寄存器一次！
- **单步 TPOT 下限公式**：
  $$\text{TPOT}_{\min} = \frac{M_{\text{active\_weights}} + M_{\text{KV\_active}}}{N_{\text{GPU}} \times BW_{\text{HBM}} \times \text{MBU}} \quad \Longrightarrow \quad N_{\text{bw}} \ge \left\lceil \frac{M_{\text{active\_weights}} + M_{\text{KV\_active}}}{\text{TPOT}_{\text{SLA}} \times BW_{\text{HBM}} \times \text{MBU}} \right\rceil$$
  （生产实测有效访存带宽利用率 $\text{MBU} \approx 0.68 \sim 0.78$，常取 **`0.72`**）。
- **典型反例（为什么 1 张 `H200` 装得下 `70B FP8` 却跑不快？）**：
  - `Llama-3.3-70B FP8` 权重 = `70 GB`，单张 `H200` 有 `141 GB HBM3e`（带宽 `4.8 TB/s = 4,800 GB/s`），第一道水位线 $N_{\text{mem}} = 1$ 完全放得下；
  - 但在 $B=32$ 并发、平均上下文 $S=8\text{K}$ 时，活跃 KV Cache = $32 \times 8,192 \times 163.84\text{ KB} \approx 42\text{ GB}$，每步总访存 = $70 + 42 = 112\text{ GB}$；
  - 单卡 `H200` 的 $\text{TPOT} = \frac{112\text{ GB}}{1 \times 4,800\text{ GB/s} \times 0.72} = \mathbf{32.4\text{ ms}}$（折合仅 `30.8 tok/s`），无法满足实时语音/代码补全 `<20ms` SLA！
  - 若切分为 **`TP=2`（2 张 H200，聚合带宽 `9.6 TB/s`）**，$\text{TPOT}$ 立即降至 **`16.2 ms`**（`61.7 tok/s`）！这就是**第二道水位线 $N_{\text{bw}} = 2 > N_{\text{mem}} = 1$** 的工程意义。

### Q3：第三道水位线（扛得住 · Prefill 算力 TTFT 与峰值并发 KV 池 $N_{\text{compute}}$）怎么手算？
1. **单请求 Prefill 首字延迟（TTFT）算力约束**：
   $$\text{TTFT}_{\text{compute}} \approx \frac{2 \times P_{\text{active}} \times S_{\text{in}} \times (1 - r_{\text{prefix\_cache}})}{N_{\text{GPU}} \times \text{TFLOPS}_{\text{peak}} \times \text{MFU}_{\text{prefill}}}$$
   - 例如 `Llama-3.1-405B FP8` 处理 $S_{\text{in}} = 16,384$ Token 长文档（无缓存），单次 Prefill 算力需求 = $2 \times 405 \times 10^9 \times 16,384 = \mathbf{13,271\text{ TFLOPs}}$；
   - 在 `8x H200`（FP8 峰值 `8 × 1,979 = 15,832 TFLOPS`，取 $\text{MFU}_{\text{prefill}} = 0.65$，有效算力 `10,290 TFLOPS`）上，首字耗时 $\text{TTFT} = 13,271 / 10,290 = \mathbf{1.29\text{ s}}$；若业务要求 $\text{TTFT} < 0.8\text{ s}$，则必须扩到 `16x H200` 或启用 Prefill Context Parallelism (`CP`)。
2. **集群总副本数（Replica Count）由峰值并发 KV 池与集群总 TPS 决定**：
   - 先用第一、二道水位线定出**单个模型实例（Replica）需要几张卡 $N_{\text{GPU\_per\_replica}}$**（如 `8x H200`），并算出该实例扣除权重后剩余可用 KV 显存能同时容纳多少个并发会话 $B_{\max}$；
   - 再根据业务峰值活跃并发数 $B_{\text{peak}}$（由 Little's Law $B_{\text{peak}} = \text{QPS} \times \text{E2EL}$ 算出），得出所需实例副本数：
     $$\text{Replicas} = \max\left(\left\lceil \frac{B_{\text{peak}}}{B_{\max\text{\_per\_replica}}} \right\rceil,\ \left\lceil \frac{\text{Target Cluster TPS}}{\text{Replica Goodput TPS}} \right\rceil\right)$$

---

## 3. 🧭 直观可视化图解（Mermaid 三道水位线决策树 + 显存瀑布拆解图）

### 3.1 大模型 GPU 数量“三道水位线”完整手算决策流
```mermaid
flowchart TD
    A["输入: 模型参数 (P_total / P_active)<br/>精度 (BF16/FP8/INT4) · 目标上下文 S · 并发 B · SLA"] --> W1["第一道水位线 (放得下 · HBM 容量):<br/>M_req = (M_weights + M_KV(B,S) + M_act) / 0.85<br/>N_mem = ceil(M_req / HBM_per_GPU)"]
    W1 --> W2["第二道水位线 (吐得快 · HBM 带宽保 TPOT):<br/>N_bw = ceil((M_active_weights + M_KV_step) / (TPOT_SLA * BW_HBM * 0.72))"]
    W2 --> W3["第三道水位线 (扛得住 · Prefill 算力保 TTFT):<br/>N_compute = ceil(2 * P_active * S_in / (TTFT_SLA * TFLOPS * 0.65))"]
    W3 --> MAX["单实例最小卡数:<br/>N_replica = next_power_of_2(max(N_mem, N_bw, N_compute))"]
    MAX --> REP["集群总卡数:<br/>N_cluster = N_replica * ceil(Peak_QPS_or_Concurrency / Capacity_per_Replica)"]
```

### 3.2 `8x H200 (1,128 GB)` 跑 `DeepSeek-R1 671B FP8` 显存水位瀑布拆解图
```mermaid
flowchart LR
    subgraph HBM["单台 8x H200 SXM 节点总物理显存: 8 × 141 GB = 1,128 GB"]
        W["1. 模型静态权重 (FP8)<br/>671 GB (59.5%)"] --> ACT["2. 激活值 + CUDA Graph + 通信 Buffer<br/>32 GB (2.8%)"]
        ACT --> KV["3. 可用 MLA FP8 KV Cache 并发池<br/>256 GB (22.7% -> 可同时塞下 192 个 16K 长会话!)"]
        KV --> SAFE["4. 15% 安全预留水位 (防碎片 OOM)<br/>169 GB (15.0%)"]
    end
```

---

## 4. 📊 架构师多方案选型决策矩阵（主流硬件参数表 & 三大模型配平总表）

### 4.1 主流云端推理/训练 GPU 硬件核心算力与显存参数速查表（手算必备常系数）

| GPU 型号 | 单卡 HBM 容量 | 单卡 HBM 带宽 ($BW_{\text{HBM}}$) | BF16 算力 (Dense) | FP8 算力 (Dense) | 机内互联带宽 (`NVLink`) | 最适合承载的模型水位 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA L40S** | `48 GB GDDR6` | `864 GB/s` | `362 TFLOPS` | `733 TFLOPS` | 无 NVLink (仅 PCIe 64GB/s) | `8B ~ 32B` FP8（单卡/双卡低成本推理） |
| **NVIDIA A100 SXM** | `80 GB HBM2e` | `2,039 GB/s` | `312 TFLOPS` | 不支持原生 FP8 | `600 GB/s` (NVLink 3) | 存量 `7B ~ 70B` BF16/INT4 |
| **NVIDIA H100 SXM** | `80 GB HBM3` | `3,350 GB/s` | `989 TFLOPS` | `1,979 TFLOPS` | `900 GB/s` (NVLink 4) | `70B FP8 (TP=2/4)`、Prefill 算力节点 |
| **NVIDIA H200 SXM** | **`141 GB HBM3e`** | **`4,800 GB/s`** | `989 TFLOPS` | `1,979 TFLOPS` | `900 GB/s` (NVLink 4) | **单机 8 卡装下 `671B MoE FP8`、长文本 Decode 甜点** |
| **NVIDIA B200 SXM** | **`192 GB HBM3e`** | **`8,000 GB/s`** | `2,250 TFLOPS` | `4,500 TFLOPS` | **`1,800 GB/s` (NVLink 5)** | **超高并发 `405B / 671B`、极低 TPOT (`<8ms`)** |

### 4.2 三大标杆模型在 `H100 (80GB)` vs. `H200 (141GB)` vs. `B200 (192GB)` 上的三道水位线配平总表（目标：`16K` 上下文、`TPOT <= 20ms`）

| 目标模型与精度 | 权重显存 ($M_{\text{weights}}$) | 每 `16K` 会话 KV 显存 | 在 `H100 (80GB)` 上的最低配平 | 在 `H200 (141GB)` 上的最低配平 | 在 `B200 (192GB)` 上的最低配平 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Llama-3.3-70B` (`FP8 W8KV8`, GQA)** | `70.0 GB` | `2.62 GB / 会话` | **`2x H100 (TP=2, 160GB)`**（剩 `54GB` KV $\approx 20$ 并发；要高并发选 `TP=4`） | **`2x H200 (TP=2, 282GB)`**（单卡虽装得下但带宽不够 `<20ms`；2 卡可跑 `60+` 并发） | **`1x ~ 2x B200`**（单卡 `8 TB/s` 即可跑进 `14ms` TPOT！） |
| **`Llama-3.1-405B` (`FP8 W8KV8`, GQA)** | `405.0 GB` | `4.13 GB / 会话` | **`8x H100 (TP=8, 640GB)`**（扣除权重与 0.85 水位仅剩 `115GB` KV $\approx 27$ 并发） | **`8x H200 (TP=8, 1128GB)`**（可用 KV 池高达 `520GB` $\approx 125$ 并发，TPOT `14.5ms`） | **`4x B200 (TP=4, 768GB)`**（半台机器即可跑 `55` 并发，TPOT `13.2ms`） |
| **`DeepSeek-R1 671B MoE` (`FP8`, MLA)** | `671.0 GB`（激活仅 `37B`） | **`0.56 GB / 会话`（MLA 压缩！）** | **`16x H100 (2台 8卡跨机)`**（单台 `640GB < 671GB` 根本装不下！） | **`8x H200 (单台 8卡, 1128GB)`**（**单机免跨机 RDMA！** 剩 `256GB` KV 可跑 **`192` 个 16K 并发**，TPOT `13.6ms`） | **`8x B200 (单台 8卡, 1536GB)`**（单机可跑 `450+` 个 16K 并发，TPOT **`7.8ms`**） |

---

## 5. 🧪 容量/成本拐点手算实例与 7-Card 报告

### 5.1 为什么跑 `DeepSeek-V3/R1 671B FP8` 选 1 台 `8x H200` 比选 2 台 `8x H100` 省一半成本且快 40%？（完整三步手算）
1. **第一道水位线（显存容量 $N_{\text{mem}}$）**：
   - `DeepSeek-R1 671B` 采用原生 FP8 权重，$M_{\text{weights}} = \mathbf{671\text{ GB}}$；
   - 若用 `H100 (80GB)`：单机 8 卡总显存仅 $8 \times 80 = 640\text{ GB} < 671\text{ GB}$，**连静态权重都装不下**，被迫拆到 **2 台 8 卡节点（`16x H100 = 1,280 GB`）** 跨机通信！
   - 若用 `H200 (141GB)`：单机 8 卡总显存 = $8 \times 141 = \mathbf{1,128\text{ GB}}$。按 `0.85` 安全水位可用显存为 $1,128 \times 0.85 = 958.8\text{ GB}$；扣除 `671 GB` 权重和 `32 GB` 激活/图开销后，还剩 **`255.8 GB` 纯净 KV Cache 池**！
   - 因为 MLA FP8 每个 `16K` 会话仅占 $16,384 \times 35.14\text{ KB} = \mathbf{0.56\text{ GB}}$（加上 MTP 草稿头约 `1.3 GB`），单台 `8x H200` 即可同时承载 **`192` 个 16K 并发活跃会话**！
2. **第二道水位线（带宽约束 $N_{\text{bw}}$）**：
   - Decode 每步只需读取 **`37B` 激活参数（`37 GB`）** + 当前活跃批次 KV（如 64 并发 × `0.56 GB` = `35.8 GB`），合计每步访存约 `72.8 GB`；
   - `8x H200` 聚合 HBM3e 带宽 = $8 \times 4,800\text{ GB/s} = \mathbf{38,400\text{ GB/s}}$；
   - 理论 + 通信开销后实测 P99 $\text{TPOT} = \mathbf{13.6\text{ ms}}$（单流生成速率 **`73.5 tok/s`**），远优于 `<25ms` SLA！

---

## 6. 🛠️ 架构评审与生产落地 Checklist

- [ ] **严禁只按 `Params × 2B` 报卡数**：所有容量评估表必须同时列出 $N_{\text{mem}}$（含 KV 与 $\div 0.85$）、$N_{\text{bw}}$（核验目标并发下的 TPOT）与 $N_{\text{compute}}$（核验长文本 TTFT）。
- [ ] **MoE 模型显存与算力分离核算**：MoE 权重显存按 $P_{\text{total}}$ 计算，Decode 带宽与 Prefill FLOPs 按 $P_{\text{active}}$ 计算。
- [ ] **注意力架构 KV 系数核对**：区分 `MHA`、`GQA-8`（如 Llama-70B FP8 = `163.8 KB/tok`）与 `MLA`（如 DeepSeek-671B FP8 = `35.1 KB/tok`），避免高估或低估 KV 显存池。
- [ ] **大参数模型优先选大显存高带宽机型（`H200 141GB / B200 192GB`）**：凡是单机 `8x H200` 能装下的模型（如 `671B FP8`），绝不用两台 `8x H100` 跨机拆分。
