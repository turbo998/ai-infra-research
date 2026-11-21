# 企业大模型 `TPM (Tokens Per Minute)` 与 `TPS (Tokens Per Second)` 换算公式、`Little's Law` 并发容量正反推与云端 `PTU / GSU` 折算规划指南

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect、企业容量规划负责人与 FinOps 架构师的 `TPM / TPS` 三级定义厘清、Little's Law（$B_{\text{active}} = \text{RPS} \times \text{E2EL}$）并发槽位正推/反推模板、云端 `PTU / GSU` 输出 Token 加权折算与 `P80 预留底座 + P20 弹性溢出` 选型白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业云架构容量规划与成本估算）

![企业大模型 TPM vs TPS 换算与 Little's Law 容量规划总结图](../../assets/svg/enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**在做企业级大模型容量规划时，必须严格区分三个层级的吞吐指标：① 单用户流式生成速度（$\text{Per-User TPS} = 1 / \text{TPOT}$，管个人体验流畅度）、② 单实例/集群系统吞吐（$\text{System TPS} = B_{\text{active}} / \text{TPOT} = \text{RPS} \times \bar{N}_{\text{out}}$，管 GPU 总产能）与 ③ 云平台分钟级配额（$\text{TPM} = \text{TPS} \times 60$，且在采购 Azure OpenAI `PTU` / Vertex `GSU` 或自建 GPU 时，1 个输出 Token 消耗的硬件驻留时间与带宽相当于 `4 ~ 8` 个输入 Token，必须按加权 $\text{TPM}_{\text{weighted}} = \text{TPM}_{\text{in}} + w_{\text{out}} \times \text{TPM}_{\text{out}}$ 核算！）；同时，决定集群会不会爆显存或排队的不是日均请求量，而是由 **Little's Law（利特尔法则：$B_{\text{active}} = \text{RPS}_{\text{peak}} \times \text{E2EL}$）** 决定的早高峰在途活跃并发数！采用“Little's Law 峰值建模 + P80 预留底座（PTU/自建 GPU）+ P20 PAYGO 按量溢出”架构，可将早高峰 `HTTP 429` 限流率降至 `0.0%`、底座利用率提升至 `86.4%`，并较粗放规划降低 `65.2%` TCO！**

- **交互式 `TPM / TPS / Little's Law / PTU` 联动容量计算器**：[打开交互式计算器 (`enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-sizer.html`)](../reports/enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-selection-cards.html`)](../reports/enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：`Per-User TPS`、`System TPS`、`Raw TPM` 和云厂商 `Weighted PTU TPM` 到底怎么精确换算？为什么把 `Input TPM` 和 `Output TPM` 直接相加去买算力必踩大坑？
1. **第一层：单用户流式感知速率（Per-User / Single-Stream Output TPS）**：
   $$\text{TPS}_{\text{user}} = \frac{1}{\text{TPOT (sec)}} = \frac{1000}{\text{TPOT (ms)}}$$
   - 例如 $\text{TPOT} = 16\text{ ms}$，则单个用户在前端看到光标打字的速度为 $1000 / 16 = \mathbf{62.5\text{ tokens/s}}$（人类阅读速度约 `8 ~ 12 tokens/s`，实时语音合成需要 `>30 tokens/s`）。
2. **第二层：系统级每秒吞吐（System Input TPS & System Output TPS）**：
   - 假设集群峰值每秒进入 $\text{RPS}$ 个请求，平均每个请求输入 $\bar{N}_{\text{in}}$ 个 Token、生成 $\bar{N}_{\text{out}}$ 个 Token：
     $$\text{TPS}_{\text{sys, in}} = \text{RPS} \times \bar{N}_{\text{in}}, \qquad \text{TPS}_{\text{sys, out}} = \text{RPS} \times \bar{N}_{\text{out}} = \frac{B_{\text{active}}}{\text{TPOT (sec)}}$$
3. **第三层：分钟级 Token 吞吐（Raw `TPM` = Tokens Per Minute）**：
   $$\text{TPM}_{\text{in}} = \text{TPS}_{\text{sys, in}} \times 60 = \text{RPM} \times \bar{N}_{\text{in}}, \qquad \text{TPM}_{\text{out}} = \text{TPS}_{\text{sys, out}} \times 60 = \text{RPM} \times \bar{N}_{\text{out}}$$
   $$\text{TPM}_{\text{raw\_total}} = \text{TPM}_{\text{in}} + \text{TPM}_{\text{out}}$$
4. **第四层：云厂商预留吞吐（Azure OpenAI `PTU` / AWS Bedrock `Provisioned` / GCP `GSU`）加权等效 `TPM`**：
   - **为什么不能直接用 $\text{TPM}_{\text{in}} + \text{TPM}_{\text{out}}$？** 因为 Prefill 处理输入 Token 是并行矩阵乘（算力受限，1 张 H200 每秒可 Prefill 上万个 Token），而 Decode 生成输出 Token 是逐个自回归访存（带宽受限，生成 1 个输出 Token 消耗的硬件墙钟时间与功耗相当于处理 **`4 ~ 8` 个输入 Token**）！
   - 因此，在折算云端 `PTU` 或自建混部 GPU 负载时，必须使用 **加权等效 TPM（Weighted TPM）**：
     $$\text{TPM}_{\text{weighted}} = \text{TPM}_{\text{in}} + w_{\text{out}} \times \text{TPM}_{\text{out}} \quad (\text{通常取 } w_{\text{out}} = 4 \sim 6\text{；部分模型官方折算为 } 1\text{ Out} = 4 \sim 8\text{ In})$$

### Q2：如何利用 **Little's Law（利特尔法则）** 从业务日活（DAU）一步步正推出峰值并发会话数 $B_{\text{active}}$、KV Cache 显存池和所需 GPU 卡数？
**Little's Law（排队论第一定律）**：在一个稳态系统中，**系统内同时处于活跃状态的并发请求数 $B_{\text{active}}$，严格等于请求到达率 $\text{RPS}$ 乘以单个请求在系统内的平均驻留总时长 $\text{E2EL}$**：
$$B_{\text{active}} = \text{RPS}_{\text{peak}} \times \text{E2EL} = \text{RPS}_{\text{peak}} \times \Big[\text{TTFT} + (\bar{N}_{\text{out}} - 1) \times \text{TPOT}\Big]$$

- **致命工程启示（为什么 Reasoning 深度思考模型会让集群并发槽位瞬间爆炸？）**：
  - 以前普通对话模型只输出 $\bar{N}_{\text{out}} = 300$ Tokens，$\text{E2EL} \approx 0.6\text{s} + 300 \times 0.016\text{s} = \mathbf{5.4\text{ s}}$；在 `50 RPS` 下，系统同时只需保持 $B_{\text{active}} = 50 \times 5.4 = \mathbf{270}$ 个并发 KV Cache 槽位；
  - 切换到 `DeepSeek-R1 / OpenAI o3` 等深度推理模型后，思维链 `<think>` + 答案使 $\bar{N}_{\text{out}}$ 暴涨至 **`2,500 Tokens`**，单次请求驻留时长拉长到 $\text{E2EL} \approx 0.8\text{s} + 2,500 \times 0.016\text{s} = \mathbf{40.8\text{ s}}$！
  - 哪怕入口 **`50 RPS` 完全没变**，根据 Little's Law，系统内同时挂着的活跃并发会话数也会从 `270` 暴涨到 $B_{\text{active}} = 50 \times 40.8 = \mathbf{2,040}$ 个（且每个会话的序列长度还增加了 2,200 Tokens），**KV Cache 显存需求直接飙升 `8 ~ 12 倍`**！

### Q3：给出了明确的正向规划模板，如果反过来——“我已经买了 2 台 8x H200（16 张卡），能扛住多少 DAU、多少峰值 QPS 和多少 TPM？”怎么反推？
- **三步反推法（从硬件卡数 $\to$ 业务可承载容量上限）**：
  1. **由单实例显存池与带宽 SLA 定出单副本最大并发 $B_{\max}$ 与单副本吞吐 $\text{TPS}_{\text{replica}}$**：
     - 例如 1 台 `8x H200` 跑 `DeepSeek-R1 671B FP8`（见 `#300` 指南），在满足 $\text{TPOT} = 16\text{ms}$（即 $\text{TPS}_{\text{user}} = 62.5\text{ tok/s}$）及 `8K` 平均上下文下，稳态最优并发批次 $B_{\text{opt}} \approx 110$ 个活跃请求；
     - 单台 `8x H200` 的系统输出吞吐 $\text{TPS}_{\text{sys, out}} = B_{\text{opt}} \times \text{TPS}_{\text{user}} \approx 110 \times 62.5 \times \eta_{\text{prefill\_share}} \approx \mathbf{4,500\text{ Output tok/s}}$（即 `270K Output TPM`）。
  2. **由 Little's Law 反推单台节点可支撑的峰值 $\text{RPS}_{\max}$**：
     $$\text{RPS}_{\max\text{\_per\_node}} = \frac{B_{\text{opt}}}{\text{E2EL}} = \frac{\text{TPS}_{\text{sys, out}}}{\bar{N}_{\text{out}}}$$
     - 若平均输出 $\bar{N}_{\text{out}} = 500$ Tokens，则单台 `8x H200` 节点可稳态承载 $\text{RPS}_{\max} = 4,500 / 500 = \mathbf{9.0\text{ Peak RPS}}$（2 台即 **`18.0 Peak RPS`**）。
  3. **由峰均比 $k_{\text{peak}}$ 反推可支撑的企业 DAU**：
     $$\text{DAU}_{\max} = \frac{\text{RPS}_{\max\text{\_cluster}} \times 86,400}{N_{\text{turns/user/day}} \times k_{\text{peak\_to\_avg}}}$$

---

## 3. 🧭 直观可视化图解（Mermaid 正反推容量流水线 + P80/P20 混合弹性架构）

### 3.1 从业务 DAU 到 GPU 卡数 / 云端 PTU 的五步正向容量推演全链路
```mermaid
flowchart TD
    S1["Step 1: 业务流量画像<br/>DAU = 50,000 · 人均 18 次/天<br/>日总请求 = 900,000 · 峰均比 k = 4.8"] --> S2["Step 2: 峰值到达率 (Peak RPS/RPM)<br/>Avg RPS = 900,000 / 86,400 = 10.42 RPS<br/>Peak RPS = 10.42 × 4.8 = 50.0 RPS (3,000 RPM)"]
    S2 --> S3["Step 3: 系统 TPS 与加权 TPM 核算 (3.5K In / 500 Out)<br/>• Input TPS = 175K/s (10.5M In TPM)<br/>• Output TPS = 25K/s (1.5M Out TPM)<br/>• Weighted TPM (w=5) = 10.5M + 5×1.5M = 18.0M TPM"]
    S3 --> S4["Step 4: Little's Law 峰值活跃并发槽位 (B_active)<br/>• E2EL = 0.8s (TTFT) + 499 × 0.016s (TPOT) = 8.8s<br/>• B_active = 50 RPS × 8.8s = 440 个并发在途会话!"]
    S4 --> S5["Step 5: P80 预留底座 + P20 弹性溢出配平<br/>• P80 预留底座承载 40 RPS (352 并发) -> 3 台 8x H200 (或对应 PTU)<br/>• P20 尖峰 10 RPS (88 并发) -> AI Gateway 自动溢出至 Serverless PAYGO"]
```

### 3.2 `P80 预留吞吐底座 (PTU / 自建 GPU)` + `P20 Serverless PAYGO 溢出` 削峰填谷架构
```mermaid
sequenceDiagram
    autonumber
    participant User as 企业 50K 员工流量 (早高峰 50 RPS)
    participant GW as 企业 AI Gateway (令牌桶 + 队列水位监控)
    participant Base as P80 预留底座 (3x 8-H200 或 Committed PTU: 承载 40 RPS)
    participant Burst as P20 Serverless PAYGO 溢出池 (承载 0~10 RPS 尖峰)

    User->>GW: 发起并发推理请求 (50 RPS)
    alt 底座活跃并发 B_active <= 352 且 预留配额未满 (前 80% 流量)
        GW->>Base: 优先路由至固定成本预留底座 (利用率高达 86.4%!)
        Base-->>User: 稳定返回 (TTFT 0.8s, TPOT 16ms, E2EL 8.8s)
    else 早 10:00 突发尖峰 B_active > 352 或 HTTP 429 预警 (顶部 20% 流量)
        GW->>Burst: 零等待无感溢出至按 Token 计费 PAYGO 接口
        Burst-->>User: 吸收波峰流量 (全网 0.0% 429 报错，免买闲置底座!)
    end
```

---

## 4. 📊 架构师多方案选型决策矩阵（四档容量规划策略对比）

| 评估维度 | Tier 1：按日均 RPS 与未加权 TPM 粗算（忽略峰均比与 Little's Law） | Tier 2：100% 纯按量付费 Serverless PAYGO（无预留底座） | Tier 3：按 100% 最高尖峰静态买满 PTU / 自建 GPU | Tier 4（推荐）：Little's Law + 加权 TPM + `P80 预留底座 + P20 PAYGO 溢出` |
| :--- | :--- | :--- | :--- | :--- |
| **早高峰 `HTTP 429` 限流/超时率** | `58.4%` ✕（早 10 点全面瘫痪） | `12.6%` ✕（触发云区级 TPM 软限额） | `0.0%` ✓ | **`0.0%` ✓** |
| **早高峰 P99 端到端延迟 (`E2EL`)** | `34.2 s` ✕（排队雪崩） | `14.8 s` | `9.4 s` ✓ | **`8.8 s` ✓** |
| **预留算力全天平均有效利用率** | `31.0%` | `48.0%` | `38.5%` ✕（夜间/周末 61.5% 空转） | **`86.4%` ✓** |
| **50 Peak RPS 企业集群月度总成本** | `$385,000 / 月`（含故障损失与应急扩容） | `$298,000 / 月` | `$242,000 / 月` | **`$134,000 / 月`（节省 `65.2%`）** |

---

## 5. 🧪 容量/成本拐点手算实例与 7-Card 报告

### 5.1 架构师完整正推手算模板：5 万日活（DAU）企业知识库 Copilot 需要多少 TPM、多少并发槽位与多少张 GPU？
1. **输入业务参数**：
   - 企业活跃人数 $\text{DAU} = 50,000$ 人，人均每天提问 $18$ 次 $\implies$ 日总请求量 = $900,000$ 次/天；
   - 办公场景流量集中在早 09:30–11:30 与下午 14:00–16:30，实测 **峰均比 $k_{\text{peak}} = 4.8$**；
   - 每次提问包含 System Prompt + RAG 检索片段共 $\bar{N}_{\text{in}} = 3,500$ Tokens，模型平均回答 $\bar{N}_{\text{out}} = 500$ Tokens；
   - 目标体验 SLA：$\text{TTFT} \le 0.8\text{ s}$，$\text{TPOT} \le 16\text{ ms}$（即单用户流式速度 `62.5 tok/s`）。
2. **Step 1 — 算峰值 RPS 与 RPM**：
   $$\text{RPS}_{\text{avg}} = \frac{900,000}{86,400} = 10.42\text{ req/s} \implies \text{RPS}_{\text{peak}} = 10.42 \times 4.8 = \mathbf{50.0\text{ req/s}} \ (\text{RPM}_{\text{peak}} = 3,000\text{ req/min})$$
3. **Step 2 — 算系统 TPS、原始 TPM 与云端 PTU 加权 TPM**：
   - $\text{TPS}_{\text{sys, in}} = 50 \times 3,500 = \mathbf{175,000\text{ In tok/s}} \implies \text{TPM}_{\text{in}} = 175,000 \times 60 = \mathbf{10.5\text{M In TPM}}$；
   - $\text{TPS}_{\text{sys, out}} = 50 \times 500 = \mathbf{25,000\text{ Out tok/s}} \implies \text{TPM}_{\text{out}} = 25,000 \times 60 = \mathbf{1.5\text{M Out TPM}}$；
   - **原始总 TPM** = $10.5\text{M} + 1.5\text{M} = \mathbf{12.0\text{M Raw TPM}}$；
   - **PTU 加权等效 TPM**（取输出权重 $w_{\text{out}} = 5$）：$\text{TPM}_{\text{weighted}} = 10.5\text{M} + 5 \times 1.5\text{M} = \mathbf{18.0\text{M Weighted TPM}}$！
4. **Step 3 — 用 Little's Law 算峰值在途并发会话数 $B_{\text{active}}$**：
   - 单次请求端到端驻留时间：$\text{E2EL} = 0.8\text{ s} + (500 - 1) \times 0.016\text{ s} = \mathbf{8.784\text{ s}} \approx \mathbf{8.8\text{ s}}$；
   - 峰值同时在途活跃请求数：$B_{\text{active}} = \text{RPS}_{\text{peak}} \times \text{E2EL} = 50 \times 8.8 = \mathbf{440\text{ 个并发会话}}$；
   - 总活跃上下文 Token 池 = $440 \times (3,500 + 500 / 2) = \mathbf{165\text{ 万 Active KV Tokens}}$。
5. **Step 4 — `P80/P20` 硬件与成本配平**：
   - 按 `P80` 水位（`40 RPS`、$B_{\text{base}} = 352$ 并发、`20,000 Out tok/s`）配置预留底座：选用 **3 台 `8x H200`（共 24 张 H200）** 自建或购买对应 `P80` 档位的云端 Committed PTU，顶部 `20%` 短时尖峰（`10 RPS`）通过 AI Gateway 自动溢出至 Serverless PAYGO，月度总成本从 `$242K ~ $385K` 降至 **`$134,000 / 月`**！

---

## 6. 🛠️ 架构评审与生产落地 Checklist

- [ ] **严禁把日总请求量除以 `86,400` 当作容量规划目标**：必须乘上业务实测峰均比 $k_{\text{peak}}$（企业内部办公通常为 `4.0 ~ 6.0`，ToC 晚高峰为 `2.5 ~ 3.5`）。
- [ ] **强制使用 Little's Law ($B_{\text{active}} = \text{RPS} \times \text{E2EL}$) 校验 KV 并发槽位**：尤其在引入 Reasoning 思维链模型时，必须按增长后的 $\bar{N}_{\text{out}}$ 重算 $B_{\text{active}}$。
- [ ] **区分 `Input TPM` 与 `Output TPM` 权重**：采购云端 PTU/GSU 或评估混部 GPU 时，按 `1 Output Token = 4 ~ 8 Input Tokens` 进行等效折算。
- [ ] **落地 `P80 预留 + P20 按量溢出` 双池路由**：在 AI Gateway 配置基于活跃并发数与 `HTTP 429 / TTFT` 阈值的无感 Spillover 路由。
