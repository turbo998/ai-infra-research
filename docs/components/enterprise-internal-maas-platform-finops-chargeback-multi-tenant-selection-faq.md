# 多团队共享企业级模型服务中台（Internal Enterprise MaaS）与 FinOps 分账架构选型指南：`部门烟囱孤岛` vs. `K8s 静态占卡配额` vs. `Multi-LoRA 共享底座池` vs. `Token 计费分账 + 三级 QoS 潮汐混部`

> 🌐 **Language / 语言**: **中文 (Chinese)** | **[English (`enterprise-internal-maas-platform-finops-chargeback-multi-tenant-selection-faq.en.md`)](enterprise-internal-maas-platform-finops-chargeback-multi-tenant-selection-faq.en.md)**
> **阅读指南**：本文面向 **企业首席架构师（Chief AI Architect）、智算平台负责人与 FinOps 成本治理官**，深入拆解大型企业在推进 AI 规模化落地时必然遇到的组织与算力治理难题：**当集团内部有 16 个业务部门（零售、客服、法务、HR、供应链、研发等）各自需要运行业务定制的大模型与夜间文档批处理时，为什么顺从业务部门要求“每个部门分 2 张专属 GPU 卡（Tier 1）”会导致全集团采购 32 张顶级显卡、月烧 `$98,400` 却有 `85.5%` 的时间在空转闲置？为什么光建一个共享 K8s 集群按 Namespace 划卡（Tier 2）依然解决不了“部门囤卡不放”的顽疾？以及如何通过“共享基础模型 + `vLLM Multi-LoRA` 动态挂载 + AI Gateway 按 Token 内部结算（Chargeback）+ P0/P1/P2 三级 QoS 优先级抢占与夜间 3 折跑批激励（Tier 4）”，仅用 `8 张 GPU（缩减 4 倍）` 就无损服务全部 16 个部门，将集群利用率推至 `88.5%`、每月净省 `$75,800 (-77.0%)`？**

![多团队共享企业级模型服务中台与 FinOps 分账架构选型一图流总结](../../assets/svg/enterprise-internal-maas-platform-finops-chargeback-multi-tenant-selection-faq-summary.svg)

---

## 🔥 业界最新能力突破追踪（Daily Hottest Capability Breakthroughs）

- **`vLLM V1 / S-LoRA` 百级动态 LoRA 同 Batch 混合推理成熟落地**：支持在单套共享的 `70B/32B FP8` 基础模型实例上，同时在 GPU 显存与主机锁页内存中热缓存数十至上百个部门专属 LoRA 适配器（单个仅 `80~160 MB`），通过 `Punica / BGMV` 融合算子在同一个 Continuous Batching 步内同时计算不同部门的微调请求，切换开销 `< 2 ms`。
- **OpenTelemetry GenAI 语义规范 + Kubecost / Prometheus 驱动 Token 级内部核算（FinOps Chargeback）**：彻底改变传统按“物理 GPU 节点占用时长”粗放分摊的财务模式，结合 AI Gateway 拆解的 `input_tokens / output_tokens / cached_tokens` 与实时 GPU 功耗遥测（`DCGM_FI_DEV_POWER_USAGE`），自动生成精确到每个部门成本中心（Cost Center）的月度结算单。
- **VTC（Virtual Token Counter）公平队列与三级 SLO 抢占调度**：在多租户推理引擎层引入类似操作系统的虚拟令牌公平调度与优先级队列，确保高优实时客服请求（`P0`）到达时毫秒级挂起低优离线批处理请求（`P2`），实现在 `88.5%` 超高集群利用率下实时流量 `P99 TTFT <= 390 ms` 零劣化。

---

## ① 📌 核心选型结论速览（Executive Summary）

集团级 AI 中台建设的核心准则只有一句话：**“绝不向业务部门交付物理 GPU 卡或独立容器实例，只向业务部门交付带 SLA 承诺与 Token 账单的 표준 MaaS API！”** 四档演进架构对比如下：

1. **为什么严禁采用 `Tier 1（各部门烟囱式独立采购 / 独占 GPU 实例）`**：
   - **重灾区现场**：集团 16 个业务部门各自立项，每个部门都要求平台组给自己的微调 70B 模型分配 2 张专属 GPU（共 **32 张旗舰卡**，月成本 **`$98,400 / 月` ✕**）。
   - **两大灾难后果**：① **显存与算力极度浪费**——16 套实例在显存里重复加载了 **16 份一模一样的 70GB 基础模型权重（白白吃掉 `1,120 GB` 显存！）**，且各部门日常流量稀疏，全集群 24 小时平均 GPU 利用率仅有可怜的 **`14.5%`（`85.5%` 资金打水漂 ✕）**；② **忙闲无法互济**——当营销部搞大促流量冲高时，它自己的 2 张卡瞬间被打爆（**`16.8%` 请求排队报错 ✕**），而隔壁法务部完全空闲的 2 张卡却一分算力也借不过来！
2. **为什么 `Tier 2（K8s 共享集群 + 静态 Namespace 物理卡配额）` 只是“换汤不换药”**：
   - 平台团队虽然把机器收归统一的 Kubernetes 集群（缩减到 `24 张卡`，月费 `$73,800`），但依然按 `ResourceQuota (nvidia.com/gpu: 2)` 把物理卡静态划给各部门 Namespace，并按“占卡小时数”向部门收费。
   - **人性博弈陷阱**：因为申请新卡审批慢，各业务部门一旦拿到卡，哪怕周末和半夜 `0` 流量也 **绝不释放（GPU Hoarding 囤卡病）**，集群利用率依然卡死在 **`28.0%` ✕**！
3. **何时选 `Tier 3（统一基础模型池 + vLLM Multi-LoRA 动态适配器池化）`**：
   - **技术破局点**：砍掉 16 个独立模型 Pod！全集团共建一个统一的 **`8×H200` 高可用推理池**，显存里只存 **1 份共享的 `70B FP8` 基础权重（`70 GB`）**，而把 16 个业务部门各自微调的差异化参数编译为轻量级 **`LoRA 适配器`（每个仅 `160 MB`）** 动态按需注入每个请求的 `model="qwen-72b-lora-legal"` 参数中！
   - **定量飞跃**：所需 GPU 总卡数直接从 `32 张` 砍到 **`8 张`（硬件集约化 `4.0 倍` ✅）**，16 个部门共享全池并发弹性，波峰限流率从 `16.8%` 降至 **`1.2%` ✅**，月总成本从 `$98,400` 骤降至 **`$28,400 / 月`（立省 `71.1%` ✅）**！
4. **集团级 AI 中台终极标杆：`Tier 4（Multi-LoRA 共享池 + AI Gateway Token 计费分账 Chargeback + P0/P1/P2 三级 QoS 抢占 + 夜间 3 折潮汐混部）`**：
   - 在 **Tier 3** 基础上补齐两大运营与调度闭环：
     1. **FinOps 按 Token 计费分账 + 分时电价杠杆**：向业务部门彻底隐藏底层 GPU 概念，按各部门实际消耗的 Input/Output Tokens 内部结算（`100%` 成本精准归属）；同时推出 **“夜间跑批 3 折内部优惠价（22:00–07:00）”**，引导各部门主动把合同审查、质检跑批任务挪到夜间执行；
     2. **P0/P1/P2 三级 QoS 优先级抢占队列**：白天当 `P0（外部客户实时对话）` 与 `P1（内部员工 Copilot）` 流量涌入时，调度器自动挂起 `P2（离线批处理）` 任务，确保实时交互 **P99 TTFT 稳定在 `390.0 ms`、`0.0%` 限流报错 🏆**，而全天候 GPU 平均利用率冲上惊人的 **`88.5%` 🏆**，月度净成本压降至 **`$22,600 / 月`（比烟囱式孤岛每月净省 `$75,800`，降幅达 `77.0%` 🏆）**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（FAQ-Driven）

### Q1：集团 CIO 问：“业务部门总觉得‘别人跟我们共用一套大模型池，万一别的部门发大请求把我们部门的实时客服卡慢了怎么办’？架构上怎么彻底消除‘嘈杂邻居（Noisy Neighbor）’干扰？”

**架构师“三道防线”消除多租户互相干扰**：

1. **第一道防线（接入层）：AI Gateway 部门级保底配额 + 超额弹性借用（Guaranteed + Burstable TPM）**
   - 在 AI Gateway 中，每个部门被配置两个水位：**保底水位（Guaranteed TPM，如客服部保底 `80,000 TPM`）** 与 **弹性上限（Burstable Ceiling，如 `250,000 TPM`）**。当集群有空闲时，任何部门都可以免费向上突破保底水位借用全池算力；一旦集群总负载超过 `85%`，网关立即将超额借用部门压回各自的保底水位，确保核心部门的保底带宽 **100% 物理级不可侵犯**！
2. **第二道防线（调度层）：三级 QoS 优先级队列（`P0 实时客服 > P1 内部办公 > P2 离线跑批`）+ Chunked Prefill**
   - 在 `vLLM V1 / SGLang` 引擎内部开启 **分块预填充（`--enable-chunked-prefill`，单步预算 `2,048 Tokens`）** 与 **优先级调度（`--scheduling-policy priority`）**。哪怕法务部突然扔进一个 `64K Token` 的超长合同（P1/P2），它也会被切碎成 32 个微块分步执行，绝不会像传统引擎那样霸占 GPU 整整 2 秒钟卡死客服部（P0）正在流式吐字的对话！
3. **第三道防线（显存层）：租户级前缀缓存配额防止 KV 驱逐风暴**
   - 为高优业务固定系统提示词（Pin Prefix Block），防止某个低优部门的突发跑批把高优客服的常驻 KV Cache 挤出显存。

---

### Q2：16 个业务部门各自微调了自己的垂直模型，为什么用 `vLLM Multi-LoRA (Tier 3/4)` 能在同一张显卡里同时跑 16 个不同模型而不降低速度？

**通俗大白话 + 架构原理解析**：
- **为什么全量微调（Full Fine-Tuning）是共享中台的死敌**：如果 16 个部门各自做全量微调，就会产生 16 个互不兼容的 `70GB` 巨型权重文件（共 `1,120 GB`），哪怕只给每个部门起 1 个实例，光装权重就要买 16~32 张大显卡。
- **`Multi-LoRA` 的魔法（`99.8%` 共享干路 + `0.2%` 动态支路）**：
  - 所有部门统一基于同一个基座（例如 `Qwen-2.5-72B-FP8`）做 **LoRA（低秩适应，Rank $r=16\sim 64$）** 微调。
  - 在推理时，GPU 显存里 **永远只驻留 1 份 `70 GB` 的基座权重 $W_0$**；而 16 个部门的 LoRA 增量权重 $(A_i, B_i)$ 每个只有 **`~160 MB`**（16 个全部加起来才 **`2.56 GB`**，连显存的一个零头都不到）！
  - 当同一个 Batch 里同时有 10 个客服部请求（用 `LoRA_CS`）、5 个法务部请求（用 `LoRA_Legal`）和 8 个财务部请求（用 `LoRA_Fin`）时，GPU 先用 1 次大矩阵乘法统一算完所有 23 个请求的基座主干 $W_0 X$，随后用专门的 **`Punica / SGMV` 向量聚合算子** 顺带算完各自的 $B_i (A_i X)$ 并相加——**整批吞吐几乎与跑单一模型完全一致，却让 GPU 采购量当场缩减了 `4 倍`！**

---

### Q3：如何设计企业内部 MaaS 平台的 **FinOps 分账模型（Showback / Chargeback）**，才能让业务部门从“抢着囤卡”变成“主动省 Token、主动半夜跑批”？

**经过大型企业验证的“双轨制内部计价表”设计**：

1. **彻底废除“按 GPU 卡月租分摊”，全面推行“按实际消费 Token + 优先级系数结算”**：
   $$\text{部门月度分摊账单} = \sum \left( \text{Input\_Tokens} \times P_{\text{in}} + \text{Cached\_Tokens} \times 0.2 P_{\text{in}} + \text{Output\_Tokens} \times P_{\text{out}} \right) \times \text{QoS\_Multiplier}$$
2. **设置“三级 QoS 价格乘数（QoS Multiplier）”引导削峰填谷**：
   - **`P0（实时高保障，承诺 P99 TTFT <= 400ms，支持云端 PAYGO 溢出兜底）`**：乘数 **`1.2x`**；
   - **`P1（标准工作时间交互，08:00–22:00）`**：基准乘数 **`1.0x`**（且如果命中了 Prefix Cache 前缀缓存，输入 Token 只收 **`0.2x`**，倒逼研发团队规范编写固定前缀 Prompt！）；
   - **`P2（夜间异步批处理队列，22:00–07:00 闲时自动触发）`**：乘数仅收 **`0.3x`（3 折特惠！）**。
   - **立竿见影的效果**：内部定价表公布后第 1 周，各部门原本习惯在上午 10 点手动触发的百万级文档抽取、周报生成、评测集回归任务，**95% 以上都会主动改写为异步队列改到半夜 22:00 后跑**，不仅白天再也不挤占实时客服算力，还把夜间闲置的预留 GPU 吃到了 **`88.5%`** 的极限利用率！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 / 端到端 MaaS 中台架构图 / 潮汐混部时序图）

### 3.1 企业内部多团队大模型平台架构选型决策树（`flowchart TD`）

```mermaid
flowchart TD
    Start["🏛️ 多团队共享企业级模型服务中台 (Internal MaaS) 架构选型"] --> SiloCheck{"1. 是否按业务部门切分独占 GPU 实例或物理卡配额？"}
    SiloCheck -- "是：各部门独立采购/独占实例 (Tier 1)" --> T1["⛔ 严禁采用 Tier 1：部门烟囱式 GPU 孤岛<br/>• 16 个部门重复加载 16 份 70B 底座，需 32 张卡 ($98.4K/月)<br/>• 平均利用率仅 14.5% (85.5% 闲置)，波峰却有 16.8% 限流 ✕"]
    SiloCheck -- "是：K8s Namespace 静态绑卡 (Tier 2)" --> T2["⛔ 避开 Tier 2：静态 Namespace 物理卡配额<br/>• 按占卡时长收费导致各部门死守空闲显卡不放 (利用率 28%)<br/>• 缺乏模型层权重共享与 Token 级精细化分账 ✕"]
    SiloCheck -- "否：建设全集团统一共享 MaaS 推理池" --> FinOpsCheck{"2. 是否具备 Token 计费分账与早晚潮汐 QoS 混部？"}
    FinOpsCheck -- "仅做技术层池化 (Tier 3)" --> T3["✅ 选 Tier 3：统一基础模型池 + vLLM Multi-LoRA 动态加载<br/>• 1 套 70B FP8 底座动态挂载 16 个部门 LoRA (每个仅 160MB)<br/>• 显卡从 32 张缩至 8 张 (省 4x)，月成本降至 $28,400 (-71.1% ✅)"]
    FinOpsCheck -- "技术池化 + FinOps 分账 + 潮汐 QoS (Tier 4)" --> T4["🏆 首选 Tier 4：企业级 MaaS 中台 + Token 分账 + 三级 QoS 混部<br/>• 按实际 Token 消费内部结算 + 夜间 P2 跑批 3 折价格杠杆<br/>• 8 张卡利用率冲上 88.5%，0% 限流，P99 390ms，月成本仅 $22,600 (-77.0% 🏆)"]
```

### 3.2 `Tier 1 (部门烟囱 32 卡孤岛)` vs. `Tier 4 (8 卡共享 Multi-LoRA + 三级 QoS MaaS 中台)` 架构对比图（`flowchart LR`）

```mermaid
flowchart LR
    subgraph Bad["❌ Tier 1 传统烟囱孤岛：16 个部门 = 32 张 GPU ($98.4K/月 · 85.5% 闲置)"]
        direction TB
        S1["客服部独占 2 卡 (70GB 底座)<br/>法务部独占 2 卡 (70GB 底座)<br/>... 16 个部门共占 32 张卡 ✕"] --> S2["浪费 1,120 GB 重复底座显存<br/>忙的部门报 429，闲的部门空转<br/>平均利用率仅 14.5% ✕"]
    end

    subgraph Best["🏆 Tier 4 企业级 MaaS 中台：全集团共享 8 张 H200 ($22.6K/月 · 利用率 88.5%)"]
        direction TB
        M1["AI Gateway FinOps 收银台<br/>(部门 Virtual Key + OTel Token 计量 + P0/P1/P2 打标)"] --> M2["三级 QoS 优先级调度与潮汐队列<br/>(白天保 P0/P1 实时 <=390ms · 夜间 3 折放行 P2 跑批)"]
        M2 --> M3["8×H200 统一推理池 (vLLM Multi-LoRA)<br/>仅存 1 份 70B 底座 + 16 个 160MB 部门 LoRA 热切换 🏆"]
    end

    Bad -. "砍掉 24 张冗余显卡，每月净省 $75,800 (-77.0%)" .-> Best
```

### 3.3 企业内部 MaaS 中台白天实时抢占、Multi-LoRA 合批与夜间潮汐填谷时序图（`sequenceDiagram`）

```mermaid
sequenceDiagram
    autonumber
    participant BU_CS as 客服部应用 (P0 实时 · LoRA-CS)
    participant BU_Legal as 法务部应用 (P1 交互 · LoRA-Legal)
    participant BU_Fin as 财务部跑批 (P2 离线 · LoRA-Fin)
    participant GW as MaaS AI Gateway & FinOps 账本
    participant Engine as 共享 8×H200 vLLM Multi-LoRA 引擎池

    Note over BU_CS,Engine: 【白天 10:30 办公高峰期】P0 与 P1 实时请求涌入，P2 跑批自动挂起让路
    BU_CS->>GW: 1. 请求 model="qwen72b-lora-cs" (X-QoS: P0-Realtime)
    BU_Legal->>GW: 2. 请求 model="qwen72b-lora-legal" (X-QoS: P1-Interactive)
    BU_Fin->>GW: 3. 提交 50,000 份发票抽取任务 (X-QoS: P2-Batch-OffPeak)
    GW->>GW: 4. 将 P2 任务暂存至异步 Redis/Kafka 闲时队列；立即放行 P0 与 P1
    GW->>Engine: 5. 同一个 Continuous Batch 同时送入客服与法务请求
    Engine-->>GW: 6. 共享 70B 底座一次前向 + Punica 算子分别叠加 LoRA-CS / LoRA-Legal (P99 TTFT = 390 ms 🏆)
    GW--)GW: 7. 按实际消耗 Token 分别计入客服部 (1.2x) 与法务部 (1.0x) FinOps 成本中心

    Note over BU_CS,Engine: 【夜间 22:00 – 次日 07:00 低谷期】自动拉起 P2 离线跑批，吃满闲置算力
    GW->>Engine: 8. 检测到实时水位 < 25%，自动批量泵入财务部 P2 发票抽取任务 (LoRA-Fin)
    Engine-->>GW: 9. 整晚满负荷吞吐跑批 (将全天平均 GPU 利用率推高至 88.5% 🏆)
    GW--)BU_Fin: 10. 次日清晨交付跑批结果，财务部享受 0.3x (3折) 闲时 Token 优惠账单!
```

---

## ④ 📊 架构师多方案选型决策矩阵（Architect Decision Matrix）

*核算基准：拥有 **16 个业务部门** 的大型企业集团（各部门均需使用定制微调的 `70B` 行业模型 + 内部员工助手 + 每日离线文档批处理，全集团月总消耗 **`24B Tokens`**；SLA 目标：`实时交互 P99 TTFT <= 600 ms`，`高峰限流率 <= 0.5%`，`100% 部门成本归属分账`）。*

| 评估维度 | Tier 1: 部门烟囱式独占孤岛 (`Department GPU Silos`) | Tier 2: K8s 静态 Namespace 占卡 (`Static GPU Quotas`) | Tier 3: 共享底座 + 动态多 LoRA (`Multi-LoRA Pooling`) | Tier 4: 企业级 MaaS 中台 + FinOps (`Multi-LoRA + Chargeback + QoS`) |
| :--- | :--- | :--- | :--- | :--- |
| **核心适用场景** | 物理隔离涉密军工/极特殊监管孤岛（常规企业严禁采用 ✕） | 仅做基础设施主机托管、无统一模型平台层的传统 IT 部门 ✕ | **多业务线基于同一开源底座微调数十个垂直 LoRA 任务的研发中台 ✅** | **大型集团/金融政企全员 AI 中台终局标准（极致利用率 + 财务闭环 🏆）** |
| **承载 16 部门所需 GPU 数** | `32 张 GPU`（16 套独立 70B 实例 ✕） | `24 张 GPU`（静态配额切块，无法跨部门借卡 ✕） | **`8 张 H200`（共享 1 份 70B 底座 + 16 个 160MB LoRA，省 `4.0x` ✅）** | **`8 张 H200`（白天实时 + 夜间跑批同池混部，省 `4.0x` 🏆）** |
| **24×7 平均 GPU 利用率** | `14.5%`（**85.5% 算力空转蒸发 ✕**） | `28.0%`（部门囤卡不放，闲时照样空转 ✕） | `64.0%`（跨 16 个部门日间波峰错峰复用 ✅） | **`88.5%`（日间多部门复用 + 夜间 3 折批处理填谷 🏆）** |
| **业务波峰限流报错率** | `16.8%`（单部门 2 张卡极易被局部洪峰打爆 ✕） | `9.4%`（配额死板无法瞬时借用空闲卡 ✕） | `1.2%`（8 卡大池子吸收局部脉冲 ✅） | **`0.0%`（三级 QoS 挂起批处理优先保实时交互 🏆）** |
| **实时交互 P99 首字延迟** | `1,420.0 ms`（局部排队严重 ✕） | `960.0 ms` ✕ | **`480.0 ms`（Punica 同 Batch 融合推理 ✅）** | **`390.0 ms`（P0/P1 高优通道零阻塞 🏆）** |
| **FinOps 部门分账准确度** | `0.0%`（大锅饭或粗暴按买卡费平摊 ✕） | `25.0%`（仅按占卡时长收费，掩盖空跑浪费 ✕） | `85.0%`（网关基础请求统计） | **`100.0%`（精确到 Input/Output/Cached Token 与 QoS 等级 🏆）** |
| **新业务部门上线周期** | `2~4 周`（走采购/申请新 GPU 配额流程 ✕） | `3~5 天`（审批 Namespace Quota 与拉起独立 Pod） | **`< 5 分钟`（上传 160MB LoRA 权重即刻热加载上线 ✅）** | **`< 1 分钟`（自助签发部门 Virtual Key + 绑定 LoRA 路由 🏆）** |
| **全集团月度总成本 (`TCO`)** | `$98,400 / 月`（最昂贵的烟囱税 ✕） | `$73,800 / 月` ✕ | **`$28,400 / 月`（立省 `71.1%` ✅）** | **`$22,600 / 月`（每月净省 `$75,800`，降幅达 `77.0%` 🏆）** |

---

## ⑤ 🧪 容量/成本拐点算例与 7-Card 报告

### 5.1 复现实验与确定性验证底座

- **确定性评估脚本**：[`experiments/enterprise_internal_maas_platform_finops_chargeback_selection/run.py`](../../experiments/enterprise_internal_maas_platform_finops_chargeback_selection/run.py)
- **封存数表 (`results.json`)**：[`experiments/enterprise_internal_maas_platform_finops_chargeback_selection/results.json`](../../experiments/enterprise_internal_maas_platform_finops_chargeback_selection/results.json)
- **单元测试校验**：[`tests/test_enterprise_internal_maas_platform_finops_chargeback_selection.py`](../../tests/test_enterprise_internal_maas_platform_finops_chargeback_selection.py)
- **7-Card 4:5 视觉评测卡片**：[`docs/reports/enterprise-internal-maas-platform-finops-chargeback-multi-tenant-selection-cards.html`](../reports/enterprise-internal-maas-platform-finops-chargeback-multi-tenant-selection-cards.html)
- **交互式企业 MaaS 中台与 FinOps 收益计算器**：[`docs/reports/enterprise-internal-maas-platform-finops-chargeback-multi-tenant-sizer.html`](../reports/enterprise-internal-maas-platform-finops-chargeback-multi-tenant-sizer.html)
- **数据溯源声明（Provenance）**：`synthetic_analytical_model_calibrated_to_published_multi_tenant_maas_benchmarks`（`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`）。

---

## ⑥ 🛠️ 架构评审与生产落地 Checklist（Architect Checklist）

- [ ] **停止按部门划分独占推理 GPU（消灭烟囱孤岛）**：是否已制定集团架构红线——除极特殊合规强隔离场景外，所有基于通用开源底座（如 `Qwen-2.5 / Llama-3.3`）的部门微调任务一律交付 **标准 LoRA 权重包**，统一托管至共享 `vLLM Multi-LoRA` 池？
- [ ] **FinOps Token 级分账（Showback $\rightarrow$ Chargeback）切换**：是否已在 AI Gateway 为每个业务部门签发独立 `Virtual Key / OIDC AppID`，并通过 OpenTelemetry 采集 `prompt_tokens`、`cached_tokens` 与 `completion_tokens` 生成每月部门分摊账单？
- [ ] **P0/P1/P2 三级 QoS 与分时折扣定价**：是否在网关与推理队列层强制区分了实时交互（P0/P1）与离线跑批（P2），并通过夜间闲时（`22:00–07:00`）3 折内部计价将离线文档处理引导至夜间波谷？
- [ ] **多 LoRA 显存与 CPU 锁页缓存池定容**：在 `vLLM` 启动参数中，是否合理配置了 `--enable-lora`、`--max-loras 16`（单 Batch 并发活跃 LoRA 槽位）与 `--max-cpu-loras 64`（主机内存热备槽位），确保冷热 LoRA 切换零抖动？
