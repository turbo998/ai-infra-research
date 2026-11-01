# DeepSeek DeepInfer 与“单卡时薪 $38 美元 / 单集群日吞吐 155 万亿 Token”深度研究报告：每卡 Token 生产效率、E/P/D 分离配比、100 PiB 3FS KV 缓存与单位成本账本全拆解

> 🌐 **Language / 语言切换**: **[🇨🇳 中文版 (Chinese)](ai-infra-phase167-deepinfer-v41-pd-economics-2026-11-01.md)** | **[🇬🇧 English Edition](ai-infra-phase167-deepinfer-v41-pd-economics-2026-11-01.en.md)** | **[📊 7-Card 4:5 视觉评测卡片 (`ai-infra-phase167-benchmark-cards.html`)](ai-infra-phase167-benchmark-cards.html)** | **[⚡ 交互式视觉工作台 (`ai-infra-phase167-visual-workbench.html`)](ai-infra-phase167-visual-workbench.html)** | **[🌐 在线预览 (`turbo998.github.io`)](https://turbo998.github.io/workshops/deepinfer-v41-economics/index.html)**

---

## 📌 Executive Summary（执行摘要与核心结论）

2026 年 10 月 20 日，DeepSeek 正式开源其内部生产级推理引擎 **[`deepseek-ai/DeepInfer`](https://github.com/deepseek-ai/DeepInfer)**（Apache 2.0）以及生产级多协议 Token 编解码库 **`deepseek-recipe`**（MIT），并罕见披露了支撑 **`DeepSeek-V4.1-Flash`**（总参数 `552B` MoE、Prefill 激活 `8B`、Decode 激活 `16B`、`1M` 上下文窗口）线上大规模推理的核心生产账本：
- **单集群日吞吐量**：**`155 万亿 Tokens/天`**（`155 × 10^12 tokens/day` ≈ **`17.94 亿 tokens/秒`**，比肩字节豆包 180T/天，约为 Google 日均约 107T/天的 `1.45 倍`）。
- **单卡每小时营收**：按**错峰谷电价（Off-Peak Valley Pricing，即周末全天及工作日非高峰半价时段）**与 **`1:1` 的 Prefill/Decode 卡数配比**折算，每张算力卡每小时产生 **`$38 美元`**（约 **`¥258.4 RMB/小时`**）营收；若按工作日白天高峰正价计算，单卡每小时产值高达 **`$76 美元`**（约 **`¥516.8 RMB/小时`**）。

结合行业深度拆解（包括【第四种黑猩猩】视频《DeepSeek 一张卡为什么能收入 38 美元》）、`deepseek-ai/DeepInfer` 官方文档（[`doc/inference_deployment_qa.md`](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md)）以及 vLLM 团队联合 SemiAnalysis `AgentX` 在 `GB300 NVL72` 上的独立复现报告（`2026-10-23`），本报告作为 **AI Infra 提升推理效率与单位 Token 经济学的阶段性核心样本**，从以下三个维度完成全链路定量解构与确定性仿真验证：

1. **每卡 Token 生产效率与算子级帕累托边界（Per-GPU Token Production & Pareto Frontier）**：
   - **Prefill 阶段（`EP4DP4` @ 4× SM100，128K 上下文，32 冗余专家，140-Token SWA 有界重放）**：冷启动（未命中缓存）单卡实测吞吐为 **`60,000` uncached tokens/s/GPU**（`batch=4,096` tokens/rank），在 `batch=65,536` 时冲高至 **`90,000` uncached tokens/s/GPU**；在线上 **`96%` 前缀缓存命中率** 加持下，单张 Prefill 卡的**等效总输入吞吐高达 `1,500,000` tokens/s/GPU**（**`25 倍`物理放大**，即每卡每小时处理 **`54 亿` 输入 Token**，全局平均 TTFT 仅 **`400 ms`**，P99 `< 1,000 ms`）。
   - **Decode 阶段（`EP64DP64` @ 64× SM100 超节点，128K 上下文）**：离线基准（`nextn=3`，接受率 `75%` $\rightarrow \tau = 3.25$）在 `256` 并发/rank 下突破 **`25,000` output tokens/s/GPU**（单流 `100` tok/s/user）；在 `8` 并发/rank 下单用户体验突破 **`300` tok/s/user**。线上生产甜点区（`conc=60`）稳定输出 **`14,000–14,466` output tokens/s/GPU**（单用户 **`230` tok/s**）。
   - **两大核心算法加速器**：**`DSpark` 置信度动态投机验证**（`DEEPINFER_ENABLE_MTP_VARLEN=1`）通过置信度头 + TileLang 算子动态截断低概率候选后缀，将平均验证草案 Token 数削减 **`~30%`**（`5.00` $\rightarrow$ `3.24–3.94`），几乎不损失接受长度（`3.21` vs `3.32`），使线上有效吞吐净增 **`+5.4% 至 +11.0%`**（线上均值 **`+6%`**）；**`SWA Bounded Replay`（有界滑动窗口重放，`KEEP_TOKENS=140`）** 利用 CED/YoCo 架构（0–19 层 Encoder，20–39 层 Decoder）将缓存命中或 SWA 缺失时的后缀重算严格锁定在最后 `128 + 6 = 134/140` 个 Token（重算复杂度降至 $O(1)$，Decoder 侧重算量削减 **`94.53%`**，总 Prefill 计算量近乎**减半**）。
2. **E/P/D 物理分离、100 PiB 3FS 分层 KV 缓存与四层闭源负载均衡（PD Disaggregation & 100 PiB 3FS）**：
   - **1:1 物理配比与最小经济规模**：生产环境采用 **Vision Encoder (`VE`, 1卡/实例) + Prefill (`PE`, 8卡通用节点 `EP2–8/DP2–8`) + Decode (`CE`, 64–128卡超节点 `EP64–128/DP64–128`)** 三相解耦，线上 **Prefill 与 Decode 卡数比例锁定为 `1:1`**，**最小高效起步规模为 `64 张 Prefill 卡 + 64 张 Decode 卡 = 128 张高端卡`**（SM100 Blackwell 或华为昇腾 Ascend 950），并支持跨机房部署（每 8 卡约需 `400 Gbps` 专线）。
   - **`890 Bytes/Token` 极致 KV 压缩与 100 PiB 3FS 存储池**：通过跨层 KV 共享、`CSA2`、主 KV `per-16 e4m3` FP4 与 Indexer KV `per-32 ue8m0 MXFP4`，全局 Block KV 显存占用压缩 **`75%`** 至 **`890 B/token`**（SSD 占用缩减 `87.5%`），持久化于独立的 **`100 PiB` 3FS 分布式全闪集群**（`10+2` 纠删码，查 11 返 10，每张在线 GPU 配比 **`14.4 TB`** 逻辑容量，峰值聚合读带宽 **`10 TiB/s`**、写 **`850 GiB/s`**，**Agent 流量保留 3 天、Chatbot 保留 7 天**，采用 **GDR RDMA** 零 SM 占用换入）；短期 SWA 状态（`20 层 × 134 tokens × 528 B = 1,415,040 B ≈ 1.415 MB/序列`）驻留主机内存（保留 `180 秒`，活跃会话命中率 **`>98%`**）。**计算与存储的彻底解耦，使任意 Prefill 节点均可零冲突命中全局任意前缀缓存**。
3. **“单卡时薪 $38 美元”精确财务还原与四大隐性护城河（Cost/Revenue Ledger & 4 Hidden Moats）**：
   - **单张 Prefill 卡每小时营收（`$47.95/hr` 谷电价）**：单卡每秒处理 `150 万`总输入 Token = **`54 亿`输入 Token/小时**。其中 **`96%` 缓存命中部分（`51.84 亿` Token）** 按白菜价 `$0.003/M`（`¥0.02/M`）计费，贡献 **`$15.55/小时`**（占输入收入 `32.4%`）；**`4%` 未命中重算部分（`2.16 亿` Token = `60,000` tok/s 真实 GPU 计算）** 按 `$0.15/M`（`¥1.00/M`）计费，贡献 **`$32.40/小时`**（占输入收入 `67.6%`）；合计 **`$47.95/小时`**。
   - **单张 Decode 卡每小时营收（`$28.05/hr` 谷电价）与 `1P:1D` 均值（`$38.00/hr`）**：单张 Decode 卡线上有效计费输出约 **`4,675 万` Token/小时**（约 `13,000–14,000` tok/s/GPU），按 `$0.60/M`（`¥4.00/M`）计费贡献 **`$28.05/小时`**。在 **`1P : 1D` 等比例分摊** 下，集群单卡平均时薪精确等于 $\frac{\$47.95 + \$28.05}{2} = \mathbf{\$38.00/\text{GPU-hr}}$（折合人民币 **`¥258.4/小时`**；高峰正价时段翻倍至 **`$76.00/GPU-hr`**）。
   - **为什么“买卡租云”赚不到 $38/小时？** 标准单节点 `vLLM` 在 **H200** 上仅能跑出 `16,600` tok/s/GPU，时薪仅 **`$0.79/小时`**（对比 `$5.40/hr` 租卡成本**巨亏 `-583%`**）；在 **B300** 上不开启 PD 分离与 Wide-EP 时薪仅 **`$14.88/小时`**（仅为 DeepSeek 的 `39%`）。唯有具备 **96% 缓存命中的海量长上下文流量 + 128 卡起步超节点 + 100 PiB 3FS 存储池 + 闭源四级负载均衡器（Prefill LB / Decode LB / EPLB / 20% 冗余扩缩容水位线）**，才能跨过从“亏本卖算力”到“80%+ 硬件毛利印钞机”的分水岭。

---

## 1. 一手信源与证据锚点（Primary Sources & Verification Ledger）

| 证据编号 | 来源类型 | 权威出处 / 仓库路径 | 核心核验内容 |
| :--- | :--- | :--- | :--- |
| **`SRC-01`** | `implementation` + `author-benchmark` | [`deepseek-ai/DeepInfer`](https://github.com/deepseek-ai/DeepInfer) · `README.md` & [`doc/inference_deployment_qa.md`](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md) (`2026-10-20`) | 官方披露 `155T tokens/day` 单集群吞吐、`$38/GPU-hour` 谷电价营收、`96%` 缓存命中率、`890 B/token` Block KV、`1.415 MB/seq` SWA KV、`100 PiB` 3FS、DSpark 吞吐表、SWA 140-token 有界重放及四大负载均衡机制 |
| **`SRC-02`** | `implementation` | [`deepseek-ai/DeepInfer/examples/sweep/plot_pareto.py`](https://github.com/deepseek-ai/DeepInfer/blob/main/examples/sweep/plot_pareto.py) & `run_sweep.sh` | 帕累托曲线与吞吐折算公式：$\tau = 1 + \text{nextn} \times a$，$\text{User Speed} = \text{avg\_SPS} \times \tau$，$\text{Output Throughput} = \text{avg\_SPS} \times B_{\text{req}} \times \tau / \text{cp}$ |
| **`SRC-03`** | `author-benchmark` (Independent Repro) | [vLLM Blog: *Serving DeepSeek-V4.1-Flash with PD Disaggregation and Wide-EP on GB300 NVL72*](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep) (`2026-10-23`) & SemiAnalysis `AgentX` | 独立复现：在 1 台 `GB300 NVL72` 机柜（`8P1D`/`7P1D`，Prefill `DEP2` + Decode `DEP32`，`MTP=5`）上实测达 **`954K` total tok/s/GPU**（P90 `196` tok/s/user），按 DeepSeek 谷电价折算达 **`$38.96/GPU-hour`**（单节点基线的 **`7.9 倍`**），并披露 H200 (`$0.79/hr`) 与单节点 B300 (`$14.88/hr`) 对照组 |
| **`SRC-04`** | `financial-breakdown` | 【第四种黑猩猩】视频《DeepSeek 一张卡为什么能收入 38 美元》（`https://v.douyin.com/8m31BovOgQs/`）及配套深度财务拆解报告 | 逐项还原 `54 亿` 输入 Token/卡/小时拆账（`96%` 命中 `$15.55` + `4%` 未命中 `$32.40` = `$47.95`）、`1P:1D` 均值 `$38/hr`、集群日入 `¥1,535万–¥3,069万 RMB` 及四大隐性成本与护城河 |
| **`SRC-05`** | `own-offline` | [`experiments/phase167_deepinfer_v41_economics/simulate_deepinfer_economics.py`](../../experiments/phase167_deepinfer_v41_economics/simulate_deepinfer_economics.py) | 本仓库纯 Python 3.12 零依赖确定性仿真套件（`--check` 100% 通过），完整复现帕累托曲线、DSpark 验证缩减、SWA 重放节省与 `$38/GPU-hr` 敏感度矩阵 |

---

## 2. 第一部分：每卡生产 Token 效果与算子级帕累托边界（Per-GPU Token Production）

### 2.1 `DeepSeek-V4.1-Flash` 的 CED/YoCo 异构激活架构与双平台原生支持

DeepInfer 并非通用兼容型推理框架，而是专为 **`DeepSeek-V4.1-Flash` (`deepseek-ai/DeepSeek-V4.1-Flash-1017`)** 深度定制的超大规模生产引擎：
- **非对称激活 MoE + CED (Cross-Encoder-Decoder / YoCo) 架构**：模型总参数量高达 **`552B`**（5,520 亿），支持 **`1M`（1,048,576）** 上下文窗口。前 20 层（Layer 0–19）为 **Encoder**（包含全局 MLA Block Attention 层与滑动窗口注意力 SWA 层），后 20 层（Layer 20–39）为 **Decoder / YOCO Consumer**（全部采用 `window=2560` 的 SWA，并直接消费第 19 层输出的共享全局 KV 状态 `ca_state / ica_state`）。这使得 **Prefill 阶段每 Token 仅激活约 `8B` 参数，而 Decode 阶段每 Token 激活约 `16B` 参数**。
- **英伟达 Blackwell (`SM100`) 与华为昇腾 (`Ascend 950`) 双一等公民**：
  - **NVIDIA 平台**：仅支持 **Blackwell `SM100`** 架构（B200 / B300 / GB200 / GB300，需 `CUDA 13`），深度集成 `FlashMLA`、`DeepGEMM`（含 `MegaMoE` 与 `MegaMHC` 流形超连接算子）、`DeepEP`、`TileKernels` 与 `TileLang` 算子。
  - **华为昇腾平台**：原生支持 **Huawei Ascend 950 (`dav-3510`)** 芯片（需 `CANN 9.2`，通过 `export DEEPINFER_ACCELERATOR=ascend` 一键编译），在 Ascend 950DT 上定制的 `DeepGEMM` 稠密矩阵乘达到了理论硬件峰值的 **`99.8%`**。
  - **明确放弃向下兼容**：开源版 DeepInfer **不支持 H100/H800 或消费级显卡**——其核心微格式（主 KV `per-16 e4m3` FP4、Indexer KV 与 MoE 权重 `per-32 ue8m0 MXFP4`）完全围绕新一代硬件原语设计。

### 2.2 Prefill 与 Decode 单卡吞吐帕累托曲线（Official SM100 128K Context Benchmarks）

根据 `deepseek-ai/DeepInfer` 官方披露的 128K（`131,072` tokens）长上下文实测数据及 `examples/sweep/plot_pareto.py` 中的折算模型：

$$\tau = 1 + \text{nextn} \times a, \qquad \text{User Speed (tok/s/user)} = \text{avg\_SPS} \times \tau, \qquad \text{Output Throughput (tok/s/GPU)} = \frac{\text{avg\_SPS} \times B_{\text{decode}} \times \tau}{\text{cp}}$$

我们整理出单张 SM100 GPU 在 **Prefill（`EP4DP4`，4 卡节点）** 与 **Decode（`EP64DP64`，64 卡超节点，`nextn=3`，接受率 $a=0.75 \Rightarrow \tau=3.25$）** 下的完整性能阶梯：

| 阶段与并行拓扑 | 批次/并发规模 (`per rank`) | 单步耗时 / 步频 (`Latency / SPS`) | 单卡物理计算吞吐 (`Uncached / Output tok/s/GPU`) | 96% 缓存命中等效总吞吐 / 单用户吐字速度 | 工程运行区间说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Prefill (`EP4DP4`)** | `4,096` tokens | `68.3 ms` | **`60,000` uncached tok/s/GPU** | **`1,500,000` total input tok/s/GPU** (`54亿/小时`) | **线上生产交互甜点区**（全局平均 TTFT `400 ms`，P99 `< 1,000 ms`） |
| **Prefill (`EP4DP4`)** | `16,384` tokens | `221.4 ms` | **`74,000` uncached tok/s/GPU** | **`1,850,000` total input tok/s/GPU** | 中等批次高吞吐区 |
| **Prefill (`EP4DP4`)** | `32,768` tokens | `399.6 ms` | **`82,000` uncached tok/s/GPU** | **`2,050,000` total input tok/s/GPU** | **Prefill LB 单引擎最大配额硬上限**（`32,768` tokens） |
| **Prefill (`EP4DP4`)** | `65,536` tokens | `~730.0 ms` | **`90,000` uncached tok/s/GPU** | **`2,250,000` total input tok/s/GPU** | 离线压测饱和峰值 |
| **Decode (`EP64DP64`)** | `8` reqs/rank | `95.38 SPS` | **`2,480` output tok/s/GPU** | **`310.0` tok/s/user** | 极致交互模式（单流突破 `300+ tok/s`） |
| **Decode (`EP64DP64`)** | `32` reqs/rank | `78.46 SPS` | **`8,160` output tok/s/GPU** | **`255.0` tok/s/user** | 高交互 SLA 均衡区 |
| **Decode (`EP64DP64`)** | `60` reqs/rank | `71.79 SPS` | **`14,000–14,466` output tok/s/GPU** | **`230.0–233.3` tok/s/user** | **线上生产主力运行点**（兼顾 `1.4万+ tok/s/卡` 与 `230 tok/s` 体验） |
| **Decode (`EP64DP64`)** | `128` reqs/rank | `49.23 SPS` | **`20,480` output tok/s/GPU** | **`160.0` tok/s/user** | 高吞吐批处理区 |
| **Decode (`EP64DP64`)** | `256` reqs/rank | `30.77 SPS` | **`25,600` output tok/s/GPU** | **`100.0` tok/s/user** | **离线吞吐极限**（突破 `2.5万 output tok/s/GPU` @ `100 tok/s/user`） |

> 💡 **关键洞察 1（25 倍缓存杠杆）**：单张 SM100 的冷启动 Prefill 物理计算极限是 `60,000–90,000 tokens/s`，但当 `96%` 的输入前缀直接从 3FS 存储池命中复用时，GPU 只需计算剩下 `4%` 的新增 Token，从而将单张 Prefill 卡对外交付的**有效计费输入吞吐放大 $1 / (1 - 0.96) = 25\text{ 倍}$，达到惊人的 `150 万 tokens/s/GPU`（`54 亿 tokens/小时`）**！

> 💡 **关键洞察 2（为什么 855K 超长上下文不会拖垮 Decode 速度？）**：vLLM 团队在 `GB300 NVL72`（`DEP32`，每 rank `16` 请求 × `6` 验证 token）上的 Profile 证实：由于 V4.1-Flash 的稀疏 MLA + Indexer 极大压缩了注意力访存，在 128K 上下文下 **MoE 专家计算+通信占单步耗时的 `62%`（`5.54 ms`），而 Attention + Indexer 仅占 `13%`（`1.16 ms`）**。因此当上下文从 **64K (`8.86 ms`) 暴增 `13.4 倍` 至 855K (`9.53 ms`) 时，单步解码耗时仅增加 `+6.7%`**！

### 2.3 `DSpark` 置信度动态投机解码：削减 30% 无效验证，线上净提速 +6%

在 `DeepSeek-V4.1-Flash` 中，轻量级投机草案模型（Draft Model）单步可并行提议 `5` 个候选 Token（`nextn=5`）。传统 MTP（Multi-Token Prediction）固定将全部 5 个候选送入主模型（Target Model）验证；一旦第 2 或第 3 个 Token 被拒绝，后续候选的昂贵主模型验证算力全部浪费。

DeepInfer 引入了 **`DSpark` 置信度调度验证（Confidence-Scheduled Verification，`DEEPINFER_ENABLE_MTP_VARLEN=1`）**：
1. **Confidence Head 预测**：草案模型内置的置信度头实时输出每个候选位置的接受概率 $p_i$。
2. **TileLang 全 GPU 动态截断算子**：在一个融合的 `TileLang` 内核中完成三步操作——① 计算每条请求前缀的累积接受概率 $\prod_{j \le k} p_j$；② 在全局 Batch 维度执行排序与 Top-$K$ 边际收益截断，为每条请求分配最优验证长度 $L_{\text{verify}, b} \in [1, 5]$；③ 通过 Fused Remap Scatter 将变长 Token 紧凑重排后送入主模型。

`doc/inference_deployment_qa.md` 披露的 `EP64/DP64` 实测对照表如下：

| 单 Rank 并发数 (`Concurrency`) | 固定验证 5 Token 吞吐 (`Fixed-5 tok/s`) | 固定验证平均接受长度 (`Accept Len`) | `DSpark` 动态验证吞吐 (`Dynamic tok/s`) | `DSpark` 平均送检长度 (`Verify Len`) | `DSpark` 平均接受长度 (`Accept Len`) | 验证 Token 节省率 (`Verify Saved`) | 净吞吐提升 (`Throughput Gain`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`16`** | `4,891` | `3.31` | `4,816` | `4.44` | `3.24` | `-11.2%` | `-1.5%` |
| **`32`** | `8,964` | `3.31` | `9,111` | `3.65` | `3.17` | **`-27.0%`** | **`+1.6%`** |
| **`48`** | `12,032` | `3.31` | `12,467` | `3.94` | `3.26` | **`-21.2%`** | **`+3.6%`** |
| **`60` (线上甜点)** | `13,730` | `3.32` | **`14,466`** | **`3.58`** | **`3.21`** | **`-28.4%`** | **`+5.4%`** |
| **`72`** | `14,962` | `3.29` | **`15,961`** | **`3.24`** | **`3.12`** | **`-35.2%`** | **`+6.7%`** |
| **`96`** | `16,817` | `3.30` | **`18,660`** | **`3.24`** | **`3.15`** | **`-35.2%`** | **`+11.0%`** |

- **核心效果**：在中高负载（`conc=60–96`）下，`DSpark` 将送入主模型验证的草案 Token 数量砍掉了 **`28.4%–35.2%`（约 `30%`）**，而实际接受的 Token 长度仅微降 `0.11–0.17` 个 Token，从而在相同硬件上白捡了 **`+5.4% 至 +11.0%`** 的总吞吐提升（线上全天综合提升 **`+6%`**）。

### 2.4 `SWA Bounded Replay`（有界滑动窗口重放）：将后缀重算锁死在 140 Tokens

在 V4.1-Flash 的 CED/YoCo 架构中，后 20 层 Decoder 采用滑动窗口注意力（SWA，`window=2560`），且其短期状态不写入 3FS 持久化存储。如果在命中全局 Block KV 缓存（或发生主内存 SWA 缓存未命中）时，为了恢复最后一步的 Decoder SWA 状态而重算整个 `2,560` 窗口，则需要在 20 层 Decoder 上重算 $20 \times 2,560 = 51,200$ 个 Layer-Tokens，长序列收益将被吞噬殆尽。

DeepInfer 通过 **`SWA Bounded Replay`（`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`）** 彻底消除了这一隐患：
- **原理**：由于前 20 层 Encoder 的第 19 层输出（`ca_state` / `ica_state`）已包含了完整的全局上下文语义，后 20 层 Decoder 只需要极短的局部近期窗口即可无损恢复生成质量。
- **128 对齐 + 6 Token 投机回退缓冲**：DeepInfer 将 SWA 截断窗口对齐至 `128` Tokens，并额外预留 `6` 个 Token 用于投机解码验证失败时的步退（Backtracking），即实际落盘/保活 `134` Tokens（环境变量默认配置 `140` Tokens），并将 SWA 注意力掩码严格限制在该重算后缀内。
- **定量收益**：Decoder 层的后缀重算工作量从 `51,200` Layer-Tokens 暴降至 `2,800` Layer-Tokens（**削减 `94.53%`**），重算开销从随序列增长变为 **$O(1)$ 常数级**，使整体 Prefill 计算量**直接减半（节省约 `50%`）**。

---

## 3. 第二部分：E/P/D 分离优化、100 PiB 3FS KV 缓存与四层闭源调度（PD Disaggregation & Systems）

```mermaid
flowchart TB
    subgraph ClientLayer["1. 多模态与长上下文流量入口 (deepseek-recipe · Rust+Python)"]
        Req["用户/Agent 请求 (平均 77K Context)\n分段并行 BPE (64KB/块) + 增量 KMP 流式解析"]
        Entry["Entrypoint 路由网关\n维护分区内存目录 (SWA 机器定位) + 全局前缀指纹"]
        Req --> Entry
    end

    subgraph EPDCluster["2. E/P/D 物理分离推理集群 (1P : 1D 配比 · 最小单元 64P + 64D = 128 卡)"]
        VE["Vision Encoder (VE 池)\n单实例 1 卡 · 按机器负载 + 最少图像 Patch 路由"]
        PE["Prefill 引擎池 (PE · 8卡节点 EP2-8/DP2-8)\nPrefill LB: 4 档长度分桶 (<16K/64K/256K/>=256K)\n二次耗时方程受限于 compute_quota_limit (<=32,768 tok)\nSWA Bounded Replay (仅重算末尾 134/140 tok)"]
        CE["Decode 超节点池 (CE · 64-128卡 EP64-128/DP64-128)\nDecode LB: DualPath 二维均衡 (活跃请求数 × 总 KV 长度)\nDSpark 动态投机验证 (削减 30% 验证量, +6% 吞吐)\nEPLB: NVLink symm_mem 在线专家重均衡 (moe_lb=1.090)"]
        Entry --> VE --> PE
        Entry --> PE
        PE -- "400G VLAN / RDMA\n跨机拉取 SWA KV (1.415 MB/seq)" --> CE
    end

    subgraph StoragePool["3. 100 PiB 3FS 全局 KV 缓存池与主机内存双层架构"]
        RAM["主机内存 SWA StateBlock (1.415 MB/seq)\n保留 20 层 Encoder × 134 tok × 528B (MXFP8)\nTTL ~180秒 · 活跃会话命中率 >98%"]
        ThreeFS["100 PiB 3FS 分布式全闪集群 (10+2 EC · 查11返10)\nBlock KV 仅 890 B/tok (压缩 75%) · 每卡配比 14.4 TB\nAgent 保留 3 天 / Chatbot 保留 7 天 · 全局命中率 96%\n峰值读 10 TiB/s · GDR (RDMA NIC) 零 SM 占用换入"]
        PE <--> RAM
        PE <--> ThreeFS
        ThreeFS -- "GDR RDMA Swap-In" --> CE
    end
```

### 3.1 为什么 Prefill 用 8 卡小并行，而 Decode 必须用 64–128 卡大 EP？

DeepInfer 在生产中采用 **E/P/D（Encoder / Prefill / Decode）三相物理分离**：
1. **Vision Encoder (`VE`)**：单实例仅需 1 张低成本 GPU，采用两级路由——先按机器级负载分流，再在单机内选择待处理图像 Patch 数 `(width / patch_size) * (height / patch_size)` 最少的实例。
2. **Prefill (`PE`，8 卡常规节点，`EP2–8 / DP2–8`)**：
   - Prefill 是典型的 **Compute-Bound（计算密集型）** 阶段，单条请求的未命中 Token 数即可跑满 Tensor Core，无需跨几十张卡切分专家。
   - **为什么更小的 EP 反而在 Prefill 更快？** vLLM 团队在 `GB300 NVL72` 上的实测揭示了关键物理机制：在 Attention DP + MoE EP 下，每个微批次必须等待所有 DP Rank 完成才能进入 All-to-All 专家通信。由于不同请求的上下文长度与命中率不同，**`DEP2` 的跨卡木桶等待损失（DP Lag Imbalance）仅为 `5.9%`（单卡吞吐达 `64.4K tok/s`），而扩大到 `DEP4` 时木桶等待损失翻倍至 `13.6%`（单卡吞吐跌至 `53.3K tok/s`）**！因此 Prefill 节点采用 `EP2` 或 `EP4` 最优。
3. **Decode (`CE`，64–128 卡超节点，`EP64–128 / DP64–128`)**：
   - Decode 是典型的 **Memory-Bandwidth & Interconnect Bound（显存带宽与互连受限）** 阶段。将 `552B` 参数的稀疏专家均匀切分到 `64–128` 张卡上（Wide-EP），每张卡仅需驻留约 **`7` 个路由专家**（含 1 个冗余专家），从而在单步 `< 9 ms` 内同时完成高并发 Token 的权重读取与 NVLink / UB 全互连通信。
4. **为什么线上 `P:D` 卡数配比是 `1:1`？**
   - 传统无缓存或低命中率系统往往需要 `3P:1D` 甚至 `4P:1D`。但在 DeepSeek 线上，`96%` 前缀缓存命中率 + `SWA Bounded Replay` 将单张 Prefill 卡的等效输入处理能力放大了 `25–50 倍`，使得 **1 张 Prefill 卡即可喂饱 1 张高吞吐 Decode 超节点卡**，形成高度对称的 **`1P : 1D` 黄金配比**（最小高效部署单元为 **`64P + 64D = 128 张卡`**）。

### 3.2 `890 Bytes/Token` KV 压缩与 100 PiB 3FS 存储池：彻底终结“前缀亲和 vs 负载均衡”悖论

在传统推理集群中，KV 缓存保存在计算节点的本地显存或本地 DRAM 中，调度器被迫面临两难：**若把请求路由给拥有缓存的节点（Prefix Affinity），会导致热点节点严重排队；若路由给空闲节点（Load Balancing），则丢失缓存被迫全量重算**。

DeepSeek 通过 **极致 KV 量化压缩 + 独立 100 PiB 3FS 存储集群** 彻底消除了这一矛盾：

| KV 缓存类型 | 物理格式与量化精度 | 单 Token / 单序列体积 | 存储介质与生命周期 | 跨节点传输与恢复机制 |
| :--- | :--- | :--- | :--- | :--- |
| **Global Block KV**（全局注意力块缓存） | 跨层 KV 共享 + `CSA2` + 主 KV `per-16 e4m3` FP4（无全局缩放） + Indexer KV `per-32 ue8m0 MXFP4` | **`890 Bytes / token`**（HBM 占用缩减 **`75%`**，SSD 占用缩减 **`87.5%`**） | **独立 `100 PiB` 3FS 全闪集群**（每张在线 GPU 配比 **`14.4 TB`**；`10+2` 纠删码，查 11 返 10；**Agent 保留 3 天，Chatbot 保留 7 天**） | **GDR (GPUDirect RDMA)**：通过 RDMA 网卡直接 DMA 写入目标 GPU HBM，**零 GPU SM 占用、零 CUDA Copy Engine 碎片瓶颈**；集群峰值聚合读 **`10 TiB/s`**、写 **`850 GiB/s`** |
| **SWA State KV**（滑动窗口短期状态） | 仅保存前 20 层 Encoder 的最后 `128 + 6 = 134` 个 Token（`512 FP8 + 16 scale = 528 B/layer/tok`），丢弃后 20 层 Decoder 状态 | **`1,415,040 Bytes / seq`**（约 **`1.415 MB / 序列`**） | **计算节点本地 Host 内存 (`StateBlock`)**（保留约 **`180 秒`**，活跃多轮会话命中率 **`>98%`**） | Entrypoint 维护分区内存目录；跨机调度时通过 **`400G VLAN`** 直接从源机器拉取 `1.415 MB` 状态；若缺失则触发 `140-token` SWA Bounded Replay |

### 3.3 未开源的四大生产级负载均衡器（实现离线理想吞吐的 `~80%`）

DeepSeek 明确指出，开源的 DeepInfer 仅包含单实例离线推理引擎，而将线上集群推至**离线理想基准 `~80%` 有效利用率**的四大核心调度器尚未开源：
1. **Prefill Load Balancer（四档长度分桶 + 二次耗时配额求解器）**：
   - 按未命中后需处理的上下文长度分为四档：`<16K`、`16–64K`、`64–256K`、`>=256K`，并根据上一轮各分桶实际消耗的算力配额动态调整各档位的 Engine 数量。
   - 调度器每隔几百毫秒向一个 Prefill Engine 下发一批请求：先按单请求预测耗时 $t = \text{cost}(q, c)$（其中 $q$ 为未命中长度，$c$ 为已命中长度；$q+c \le 16\text{K}$ 用线性模型，$>16\text{K}$ 用二次多项式模型）对 DP Rank 做负载均衡；再在每个 DP Rank 内部沿序列维度切分请求，通过求解二次方程 $\text{cost}(q, c) \le \text{compute\_quota\_limit}$（纳秒配额，且单步 Token 上限 `32,768`）确定本步 Chunk 大小，最后通过 DP 组内 `all-reduce MIN` 严格对齐各卡步长，消除木桶效应。
2. **Decode Load Balancer（DualPath 二维请求数 × KV 长度均衡）**：
   - 同时平衡每张 Decode 卡上的 **活跃请求并发数（决定 MoE GEMM 负载）** 与 **总 KV Cache 长度（决定 MLA Attention 负载）**——将短上下文高并发请求与长上下文低并发请求交错配对到同一 DP Rank 上。
3. **EPLB（基于 NVLink `symm_mem` 的在线专家并行负载均衡）**：
   - 利用 PyTorch/CUDA `symm_mem` 对称显存原语在后台无感迁移热点专家权重：Prefill 侧每卡配置 `4` 个冗余专家（100 步统计窗口、1,000 步重排周期，`EP4` 下 `moe_lb = 1.074`，`EP2` 下 `1.045`）；Decode 侧每卡配置 `1` 个冗余专家（`EP64` 取整后每卡驻留 `7` 个专家，1,000 步窗口、10,000 步重排周期，`moe_lb = 1.090`）。
4. **弹性扩缩容与 `0.8` 冗余水位护栏（Autoscaling & Failover）**：
   - 当 Prefill 或 Decode 池的配额利用率 **超过 `0.8`（80%）时严禁缩容（Scale-in）**，强制保留 **至少 `20%` 的在线热备容量缓冲** 以吸收突发流量。
   - Decode 实例缩容前设置 **1 分钟排空宽限期（Drain Grace Period）**：先停止分派新请求，1 分钟后将残余长尾请求的 KV Cache 一次性迁移至其他 Decode 实例；单卡硬件故障时支持动态缩小 EP 规模并由冗余专家平滑接管。

---

## 4. 第三部分：“单卡时薪 $38 美元 / 日吞吐 155 万亿 Token”成本与收入账本全拆解

### 4.1 `DeepSeek-V4.1-Flash` 官方分时 API 定价表

理解 `$38/GPU-hour` 的第一步，是明确官方披露口径采用的是 **错峰谷电价（Off-Peak Valley Pricing，即周末全天 + 工作日非高峰半价时段）**，而非白天高峰正价：

| 计费时段 | 输入（缓存命中 Cache Hit） | 输入（缓存未命中 Cache Miss） | 输出（生成 Output Tokens） | 价格倍率关系 |
| :--- | :---: | :---: | :---: | :--- |
| **谷电价（Off-Peak Valley）**<br>*(周末全天 + 工作日 00:00–09:00, 12:00–14:00, 18:00–24:00)* | **`¥0.02 / 1M`**<br>(`~$0.003 / 1M`) | **`¥1.00 / 1M`**<br>(`~$0.15 / 1M`) | **`¥4.00 / 1M`**<br>(`~$0.60 / 1M`) | **官方 `$38/GPU-hr` 基准口径**<br>（未命中单价是命中的 **`50 倍`**） |
| **高峰正价（Peak Standard）**<br>*(工作日 09:00–12:00, 14:00–18:00 CST)* | **`¥0.04 / 1M`**<br>(`~$0.006 / 1M`) | **`¥2.00 / 1M`**<br>(`~$0.30 / 1M`) | **`¥8.00 / 1M`**<br>(`~$1.20 / 1M`) | 全线严格为谷电价的 **`2.0 倍`**<br>（对应单卡时薪高达 **`$76/GPU-hr`**） |

### 4.2 逐项公式还原：一张卡如何每小时赚到 $38 美元？（【第四种黑猩猩】核心账本）

为什么媒体与视频中会出现 **“单卡每小时处理 54 亿输入 Token”、“缓存命中赚 $15.55 + 未命中赚 $32.40”、“与 Decode 卡 1:1 平摊后等于 $38/小时”** 这组精确数字？我们用确定性公式逐行还原：

#### 第一步：单张 Prefill 卡每小时的输入 Token 账本（`$47.95 / Prefill GPU-hr`）
1. **真实物理计算量（4% 未命中部分）**：
   如第 2.2 节官方帕累托曲线所示，单张 SM100 Prefill 卡在生产交互甜点区每秒可计算 **`60,000` 个未命中（Uncached）输入 Token**。
   $$\text{Uncached Tokens / Hour} = 60,000 \text{ tok/s} \times 3,600 \text{ s/hr} = \mathbf{216,000,000 \text{ tokens/hr}} \text{（即 } \mathbf{2.16\text{ 亿 Token/小时}}\text{）}$$
2. **3FS 缓存放大后的总计费输入量（100% 总输入 = 96% 命中 + 4% 未命中）**：
   线上全局前缀缓存命中率高达 **`96%`**，意味着这 `2.16 亿` 个真正在 GPU 上计算的 Token 仅占用户提交总输入的 **`4%`**：
   $$\text{Total Input Tokens / Hour} = \frac{216\text{M}}{1 - 0.96} = \frac{216\text{M}}{0.04} = \mathbf{5,400,000,000 \text{ tokens/hr}} \text{（即 } \mathbf{54\text{ 亿 Token/小时}} = \mathbf{150\text{ 万 tok/s}}\text{）}$$
   其中直接从 100 PiB 3FS 命中的 Token 数量为：
   $$\text{Cached Input Tokens / Hour} = 5,400\text{M} \times 96\% = \mathbf{5,184,000,000 \text{ tokens/hr}} \text{（即 } \mathbf{51.84\text{ 亿 Token/小时}}\text{）}$$
3. **单张 Prefill 卡每小时谷电价收入拆解（“96% 的白菜价流量 vs 4% 的高毛利计算”）**：
   - **96% 缓存命中收入**：
     $$R_{\text{prefill, hit}} = 5,184\text{M} \times \$0.003/\text{M} = \mathbf{\$15.552/\text{hr}} \approx \mathbf{\$15.55/\text{hr}} \quad (\text{占输入总收入的 } \mathbf{32.4\%})$$
   - **4% 缓存未命中收入**：
     $$R_{\text{prefill, miss}} = 216\text{M} \times \$0.15/\text{M} = \mathbf{\$32.400/\text{hr}} = \mathbf{\$32.40/\text{hr}} \quad (\text{占输入总收入的 } \mathbf{67.6\%})$$
   - **单张 Prefill 卡合计每小时创收**：
     $$R_{\text{prefill, total}} = \$15.55 + \$32.40 = \mathbf{\$47.95/\text{hr}} \quad (\text{折合人民币约 } \mathbf{¥326.06/\text{小时}})$$

> 🔥 **核心财务洞见（缓存经济学悖论）**：表面上看，高达 `96%` 的缓存命中只收 `$0.003/M` 的白菜价，似乎让 DeepSeek “少赚了钱”；但实际上，**正是这 96% 由 3FS 硬盘承接的命中流量，在不消耗 GPU Tensor Core 算力的前提下，每小时白送给每张 Prefill 卡 `$15.55` 的纯增量收入（足够直接付清整张 GPU 的电费与折旧！）**，同时把宝贵的 GPU 算力 100% 留给单价贵 50 倍（`$0.15/M`）的 `4%` 新增 Token，再赚 `$32.40/小时`！

#### 第二步：单张 Decode 卡每小时的输出 Token 账本与 `1P : 1D` 集群平摊（`$38.00 / GPU-hr`）
1. **单张 Decode 卡每小时输出创收**：
   在 `EP64DP64` 超节点上，单张 Decode 卡线上稳态生成速率约为 `14,000` tok/s（理想满载 `5,040 万` tokens/hr，对应 `$30.24/hr`）；扣除线上波动与调度损耗后，单张 Decode 卡每小时实际计费输出约 **`4,675 万` Output Tokens**（等效计费速率 `~12,986` tok/s/GPU）：
   $$R_{\text{decode, total}} = 46.75\text{M} \times \$0.60/\text{M} = \mathbf{\$28.05/\text{hr}}$$
2. **`1P : 1D` 集群全局单卡平均时薪**：
   由于生产集群中 Prefill 卡与 Decode 卡按 **`1:1`** 配比部署，将 1 张 Prefill 卡与 1 张 Decode 卡的收入相加并除以 2：
   $$R_{\text{cluster, avg}} = \frac{R_{\text{prefill, total}} + R_{\text{decode, total}}}{2} = \frac{\$47.95 + \$28.05}{2} = \mathbf{\$38.00 / \text{GPU-hour}} \quad (\text{约 } \mathbf{¥258.4\text{ RMB/小时}})$$
   *(注：若按理想 `14,000` tok/s Decode 速率并乘以线上负载均衡器相对于离线基准的 `~80%–95%` 折算系数，如 `daily.steinslab.io` 与 `vLLM GB300 NVL72` 独立测算，理论值分布在 **`$36.70–$38.96 / GPU-hour`** 极窄区间内，与官方披露的 `$38` 完美闭合！)*

#### 第三步：单集群 `155 万亿 Tokens/天` 规模与日营收还原
- **单集群算力规模推算**：在 `1P:1D` 下，每对（1P+1D）GPU 每小时合计处理 $5,400\text{M} + 46.75\text{M} = 5,446.75\text{M}$ Tokens，折合**平均每张 GPU 每天处理 `653.61 亿`（`65.361B`）Tokens**。要支撑单集群 **`155 万亿`（`155,000B`）Tokens/天**，该集群活跃工作的算力卡数量约为：
  $$N_{\text{GPU, active}} = \frac{155,000\text{B}}{65.361\text{B/GPU/day}} \approx \mathbf{2,371 \text{ 张 GPU}} \quad (\text{对应 } \mathbf{19\text{ 个 }128\text{ 卡最小单元}} = \mathbf{2,432\text{ 张卡}})$$
- **单集群日营收与年化产值**：
  - **全天按谷电价（下限）**：$2,371\text{ 卡} \times 24\text{ 小时} \times \$38/\text{hr} \approx \mathbf{\$216.2\text{ 万美元/天}}$（合 **`¥1,470万–¥1,535万 RMB/天`**，月入 **`~¥4.67亿 RMB`**，年化 **`~¥56亿 RMB`**）。
  - **全天按高峰正价（上限，`$76/GPU-hr`）**：日营收翻倍至 **`~¥3,069万 RMB/天`**（月入 **`~¥9.34亿 RMB`**，年化 **`~¥112亿 RMB`**）。

---

## 5. 第四部分：跨平台对比与“为什么买显卡赚不到 $38/小时”的四大隐性护城河

### 5.1 四种软硬件部署形态的单位经济效益大比拼（H200 vs B300 vs DeepInfer vs GB300 NVL72）

结合 SemiAnalysis `AgentX` 与 `vLLM` 在 2026 年 10 月 23 日发布的实测基准（统一按 DeepSeek-V4.1-Flash 谷电价与 `>=100 tok/s/user` 交互 SLA 核算）：

| 部署架构方案 | 硬件配置与并行拓扑 | 单卡综合总吞吐 (`Total tok/s/GPU`) | 单用户吐字速度 (`tok/s/user`) | 谷电价单卡时薪 (`Off-Peak $/GPU-hr`) | 高峰价单卡时薪 (`Peak $/GPU-hr`) | 市场租卡/折旧成本 (`$/GPU-hr`) | 硬件级毛利率 (`Off-Peak Margin`) | 商业可行性判定 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. 传统单机 vLLM (H200)** | 8× NVIDIA H200 SXM（无 PD 分离、无 Wide-EP） | `16,600` | `100` | **`$0.79 / hr`** | `$1.58 / hr` | `~$5.40 / hr` | **`-583.5%` (巨亏)** | ❌ **开机即亏损**：连电费与显卡租金的零头都付不起 |
| **2. 传统单机 vLLM (B300)** | 8× NVIDIA B300 SM100（无 PD 分离、无 Wide-EP） | `317,400` | `100` | **`$14.88 / hr`** | `$29.77 / hr` | `~$8.50 / hr` | `+42.9%` | ⚠️ **产能严重浪费**：仅发挥出 DeepSeek 产值的 **`39.2%`** |
| **3. DeepSeek 生产级 DeepInfer + 3FS** | 128+ SM100 / Ascend 950（`1P:1D`, `EP4` Prefill + `EP64` Decode, `DSpark`, `100 PiB 3FS`） | `~756,500` | **`230`** | **`$38.00 / hr`** | **`$76.00 / hr`** | `~$5.00–$8.00 / hr` | **`80.3%–86.8%`** | ✅ **工业级印钞机**：软硬协同榨干每瓦特算力 |
| **4. vLLM + Wide-EP + PD (GB300 NVL72)** | 72× NVIDIA GB300 NVL72（`8P1D`/`7P1D`, `DEP2` Prefill + `DEP32` Decode, `MTP=5`） | **`954,000`** | **`196`** | **`$38.96 / hr`** | **`$77.92 / hr`** | `~$9.00 / hr` | **`76.9%`** | ✅ **外部独立复现标杆**：较单节点基线提速 **`7.9 倍`** |

### 5.2 缓存命中率敏感度矩阵（当缓存命中率从 96% 跌落时会发生什么？）

如果第三方企业只部署了开源的 `DeepInfer` 离线引擎，却没有 `100 PiB` 3FS 存储池（3–7 天长效保留）或缺乏高密度连续 Agent/Chatbot 流量，当缓存命中率从 **`96%`** 下滑时，单张 Prefill 卡的物理算力被死死卡在 `60,000 uncached tok/s`（`2.16 亿` uncached tokens/小时），系统吞吐与单卡收入将发生断崖式衰减：

| 前缀缓存命中率 (`Cache Hit Rate`) | 单张 Prefill 卡等效总吞吐 (`Total Input tok/s/GPU`) | 每小时处理总输入 (`Total Input M/hr`) | 缓存命中输入量 (`Cached M/hr`) | 未命中重算输入量 (`Uncached M/hr`) | 用户侧等效平均输入单价 (`Blended ¥/1M`) | Prefill 单卡谷电时薪 (`Prefill $/hr`) | `1P:1D` 集群综合单卡时薪 (`Blended $/GPU-hr`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`96.0%` (DeepSeek 生产基线)** | **`1,500,000` tok/s** (`25.0x`) | **`5,400.0 M`** | `5,184.0 M` | `216.0 M` | **`¥0.0592 / 1M`** | **`$47.95 / hr`** | **`$38.00 / hr`** |
| **`90.0%`** | `600,000` tok/s (`10.0x`) | `2,160.0 M` | `1,944.0 M` | `216.0 M` | `¥0.1180 / 1M` | `$38.23 / hr` | `$33.14 / hr` |
| **`80.0%`** | `300,000` tok/s (`5.0x`) | `1,080.0 M` | `864.0 M` | `216.0 M` | `¥0.2160 / 1M` | `$34.99 / hr` | `$31.52 / hr` |
| **`50.0%`** | `120,000` tok/s (`2.0x`) | `432.0 M` | `216.0 M` | `216.0 M` | `¥0.5100 / 1M` | `$33.05 / hr` | `$30.55 / hr` |
| **`20.0%`** | `75,000` tok/s (`1.25x`) | `270.0 M` | `54.0 M` | `216.0 M` | `¥0.8040 / 1M` | `$32.56 / hr` | `$30.31 / hr` |
| **`0.0%` (无缓存裸跑)** | `60,000` tok/s (`1.0x`) | `216.0 M` | `0.0 M` | `216.0 M` | **`¥1.0000 / 1M`** | `$32.40 / hr` | `$30.23 / hr` *(注：因无法喂饱 Decode 且用户成本暴涨 16.9 倍，真实需求将直接归零)* |

### 5.3 总结：普通算力租赁商无法复制 `$38/小时` 的四大隐性成本与护城河

正如【第四种黑猩猩】视频所剖析，`$38/GPU-hour` 是 **极致全栈工程压榨出的系统级红利**，而非硬件自带属性，中间横亘着四道极高门槛：
1. **门槛一：规模化连续流量与 96% 缓存命中的网络效应**：没有日均百亿至百T级的连续多轮 Agent/Chatbot 流量，GPU 在夜间与波谷只能空转烧电；没有 96% 的缓存命中率，用户实际综合调用成本将从 `¥0.06/M` 暴涨至 `¥0.22–¥1.00/M`，丧失价格竞争力。
2. **门槛二：128 卡起步的异构超节点与 100 PiB 3FS 重资产投入**：DeepInfer 不支持消费级卡或老旧 Hopper 卡，且高效 E/P/D + Wide-EP 至少需要 **`64P + 64D = 128 张 SM100/Ascend 950`**、每卡配比 **`14.4 TB` 的高速 NVMe 3FS 存储池** 以及 **`400G` RDMA/VLAN 互联网络**。
3. **门槛三：核心控制面（KV 存储与四层负载均衡）暂未开源**：开源的 DeepInfer 提供了算子与单机引擎（相当于“F1 赛车发动机”），但决定集群能否跑出 `80%` 理想效率的 `kvcache-store`、E/P/D 编排与四大动态负载均衡器（相当于“车队遥测与进站调度系统”）仍掌握在 DeepSeek 内部。
4. **门槛四：20%+ 强制冗余缓冲、研发人力摊销与 SLA 故障赔付成本**：`$38/小时` 是满载算力卡的毛收入（Gross Run-Rate），尚未扣除 **低于 `0.8` 利用率才允许缩容的 `20%+` 强制热备冗余卡成本**、数百PB 3FS 存储集群折旧、顶尖 Infra 研发团队薪酬，以及高负载下的宕机赔付风险（例如 10 月 15 日线上 5 小时服务中断后的全额赔付）。

---

## 6. 本地确定性复现指南（Reproducibility Guide）

本报告配套的零依赖 Python 3.12 确定性仿真与校验脚本位于 [`experiments/phase167_deepinfer_v41_economics/simulate_deepinfer_economics.py`](../../experiments/phase167_deepinfer_v41_economics/simulate_deepinfer_economics.py)，可直接运行验证全部公式与账本断言：

```bash
python experiments/phase167_deepinfer_v41_economics/simulate_deepinfer_economics.py --check
```
