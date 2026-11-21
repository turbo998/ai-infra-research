# 企业大模型部署模式与云平台 TCO 选型对比：`Serverless API (MaaS PAYGO)` vs. `预留吞吐单元 (Azure Foundry PTU / Bedrock Provisioned / Vertex GSU)` vs. `容器化自建 (AKS/GKE + KAITO/vLLM)` 成本拐点与混合架构指南

> **语言 / Language**: **中文 (当前)** | [English](./cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect、CTO 与 FinOps 负责人的云端大模型三大部署模式（按量 `PAYGO`、预留吞吐 `PTU/GSU`、Kubernetes 容器化自托管 `AKS/GKE + vLLM/SGLang`）精确盈亏平衡点手算公式、SLA、数据隐私与运维负担对比白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业云架构选型与三年 TCO 评估）

![企业大模型部署模式 PAYGO vs PTU/GSU vs 容器化自建 K8s 选型总结图](../../assets/svg/cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**回答客户“我们到底该调按量 API（PAYGO）、买预留吞吐（PTU/GSU），还是自己租 GPU 在 K8s 上搭 vLLM？”时，架构师只需锁定两个定量盈亏拐点：① `PTU/GSU` vs. `PAYGO` 的利用率拐点为 **`35%`**（若开启 Prefix Caching 享受半价以上折扣，则拐点低至 **`22%`**——只要全天平均利用率超过 `35%`，买 `P80 PTU + P20 PAYGO 溢出` 既比纯 PAYGO 便宜，又能消灭共享池邻居干扰）；② `容器化自建 K8s (AKS/GKE + vLLM/SGLang)` vs. `全托管 API` 的月度流量拐点为 **`25 亿（2.5B）Tokens/月`（约合 `>= 8` 张 H100/H200）**——低于 `1B~2.5B Tokens/月` 时，自建 K8s 每月固定分摊的 `~$22,000` SRE 人力与跨可用区冗余卡成本会让真实单价反超 PAYGO（高达 `$4.85/1M`）；而当月耗量突破 `2.5B ~ 15B Tokens/月` 时，采用“70% 高频流量落自建 K8s 跑开源 FP8 模型 + 20% 核心闭源模型走 P80 PTU + 10% 尖峰溢出走 PAYGO”的三区混合架构，可将月度总 TCO 从 `$360K` 砍至 **`$102K`（立省 `71.7%`）**！**

- **交互式 `PAYGO / PTU / 自建 K8s` 盈亏平衡拐点计算器**：[打开交互式计算器 (`cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-sizer.html`)](../reports/cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-selection-cards.html`)](../reports/cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：第一道拐点——什么时候该从 `Serverless PAYGO`（按 Token 计费）切换到 `预留吞吐单元（Azure Foundry PTU / Vertex GSU / Bedrock Provisioned）`？
1. **精确盈亏平衡利用率公式（$\rho_{\text{break-even}}$）**：
   - 设包月购买一组预留吞吐单元的固定月费为 $C_{\text{PTU}}$（美元/月），该 PTU 满载每分钟可提供的等效吞吐上限为 $\text{TPM}_{\text{cap}}$，按量付费 PAYGO 的等效单价为 $p_{\text{PAYGO}}$（美元/百万 Token）：
     $$\rho_{\text{break-even}} = \frac{C_{\text{PTU}}}{43,200\text{ min/mo} \times \text{TPM}_{\text{cap}} \times (p_{\text{PAYGO}} / 10^6)} \approx \mathbf{32\% \sim 38\%}$$
   - **工程结论**：只要某个业务场景（或通过企业 AI Gateway 汇聚多个部门后的总底座）24×7 平均利用率 **$\rho \ge 35\%$**，买预留 `PTU/GSU` 的单 Token 成本就正式低于 `PAYGO`！更重要的是，PTU 提供**物理隔离的独享吞吐配额（P99 TTFT 稳定在 `490ms` 以内，彻底告别公共 PAYGO 池早高峰 `1,480ms` 抖动与 `HTTP 429` 限流）**。

### Q2：第二道拐点——为什么很多团队盲目自建 K8s + vLLM 反而比调云 API 贵一倍？自建容器化集群的真实盈亏平衡线在哪？
- **最容易漏算的隐性成本（Hidden SRE & HA Tax）**：
  - 很多工程师算自建成本时，只拿“1 张 GPU 租金除以理想满载吞吐”，却漏掉了三座大山：① **平台 SRE 与升级排障人力成本**（至少 0.5~1 名专职 AI Infra 工程师，折合 `$15,000 ~ $22,000 / 月`）；② **生产高可用（Multi-AZ N+1）最低起步卡数**（哪怕只有 1 QPS，跑 70B/MoE 也至少要开 2 个副本防单点故障）；③ **夜间与周末低谷期的 GPU 空转浪费（通常高达 50%~65%）**！
- **自建 K8s vs. 云托管 API 的月度 Token 量盈亏平衡公式（$V_{\min}$）**：
  $$V_{\min\text{ (Tokens/Mo)}} = \frac{C_{\text{fixed\_SRE\_and\_HA}}}{p_{\text{Managed\_API}} - p_{\text{Self\_Hosted\_GPU}}} \approx \frac{\$22,000}{(\$1.38 - \$0.48) / 10^6} \approx \mathbf{2.44 \times 10^9 \text{ (约 25 亿 Tokens / 月)}}$$
  - **当月调用量 $< 5\text{亿} \sim 15\text{亿}$ Tokens/月** 时：**千万别自建 K8s GPU 集群**（分摊固定成本后高达 `$4.85 / 1M`），老老实实用 `Serverless PAYGO`！
  - **当月调用量 $\ge 25\text{亿}$ Tokens/月（或因军工/金融合规强制要求权重不出内网 VPC）** 时：使用 `AKS / GKE + KAITO / Kueue + vLLM / SGLang` 自托管开源权重（如 `DeepSeek-V3/R1`、`Llama-3.3-70B`、`Qwen3`），边际成本骤降至 **`$0.35 ~ $0.48 / 1M Tokens`**！

---

## 3. 🧭 直观可视化图解（Mermaid 三阶段演进决策树 + 三区混合网关架构）

### 3.1 企业大模型部署模式三阶段演进决策树
```mermaid
flowchart TD
    A["企业大模型部署模式选型"] --> B{"是否有严格私有化/内网不出域合规红线<br/>或需要挂载自定义 LoRA/私有微调权重？"}
    B -- "是 (数据绝不出内网 VPC / 自研微调模型)" --> C["选方案 C: 容器化自建 (AKS/GKE + KAITO + vLLM/SGLang)<br/>• 权重与 KV 100% 驻留企业私有节点"]
    B -- "否 (可调用云厂商合规托管大模型)" --> D{"月度总 Token 消耗量 V 与 24×7 利用率 ρ？"}
    D -- "阶段 1: V < 10亿 Tok/月 或 ρ < 35%<br/>(PoC 孵化期 / 强波峰波谷应用)" --> E["🏆 选方案 A: Serverless MaaS (PAYGO 按量付费)<br/>• 零固定底租 · 零运维 · 秒级切换多模型"]
    D -- "阶段 2: 10亿 <= V < 25亿 Tok/月 且 ρ >= 35%<br/>(或强依赖闭源旗舰 GPT/Claude/Gemini)" --> F["🏆 选方案 B: P80 预留吞吐 (PTU/GSU) + P20 PAYGO 溢出<br/>• 锁定独享延迟 SLA · 比纯 PAYGO 省 35%+"]
    D -- "阶段 3: V >= 25亿 Tok/月 (>= 8~16 张 GPU)<br/>(规模化成熟期)" --> G["🏆 选方案 D (终极混合架构):<br/>• 70% 高频分类/摘要/代码分流至自建 K8s 开源 FP8 模型<br/>• 20% 复杂推理走闭源 P80 PTU + 10% 走 PAYGO 溢出"]
```

---

## 4. 📊 架构师多方案选型决策矩阵（三大部署模式 + 混合架构全方位对比）

| 评估维度 | 模式 A：`Serverless API (MaaS PAYGO)` | 模式 B：`预留吞吐单元 (PTU / GSU / Provisioned)` | 模式 C：`容器化自建 (AKS/GKE + KAITO/vLLM)` | 模式 D（推荐规模化架构）：`70% 自建 K8s + 20% P80 PTU + 10% PAYGO` |
| :--- | :--- | :--- | :--- | :--- |
| **最佳月度流量区间** | **`< 10 亿 Tokens / 月`**（或波峰溢出） | **`10 亿 ~ 50 亿 Tokens / 月`**（闭源主力） | **`>= 25 亿 Tokens / 月`**（开源/微调主力） | **`>= 30 亿 ~ 500 亿+ Tokens / 月`** |
| **计费模式与成本弹性** | 按实际 Input/Output Token 计费（0 底租） | 按小时/包月购买固定吞吐配额（1年/3年折扣） | 按 GPU 虚拟机实例时长计费（RI/Spot）+ SRE 成本 | 底盘享受自建/PTU 低单价，尖峰按 Token 溢出 |
| **延迟稳定性 (`P99 TTFT`)** | 较差（`1,480 ms`，受公共池限流影响） | **极佳（`490 ms`，独享预留算力）** | **极佳（`380 ~ 420 ms`，完全可控）** | **极佳（`420 ms`，`99.95%` 可用性）** |
| **数据隐私与定制自由度** | 共享多租户集群（受云商零留存协议保护） | 独享吞吐配额（仅限云商目录内模型） | **100% 物理/VPC 隔离，支持任意开源模型与多 LoRA** | 兼顾核心机密不出域与前沿闭源能力 |
| **工程运维负担（Ops）** | **零运维（0 SRE）** | **极低运维（仅需配网关溢出）** | 高（需管 GPU 驱动、K8s 调度、升级与压测，可借 `KAITO` 减负） | 中等（统一由企业 AI Gateway 纳管） |
| **150 亿 Tokens/月 总 TCO** | `$360,000 / 月` (`$2.40/1M`) | `$207,000 / 月` (`$1.38/1M`) | `$125,000 / 月` (`$0.83/1M`) | **`$102,000 / 月` (`$0.68/1M`，节省 `71.7%`）** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **月耗量低于 `15~25 亿 Tokens` 时严禁盲目立项自建 GPU K8s 集群**：先用 `Serverless PAYGO` 跑通业务闭环并采集真实峰均比与 I:O 比例。
- [ ] **当单一闭源模型全天平均利用率超过 `35%` 时，立即采购 `P80` 水位的 `PTU / GSU` 并配置 `P20 PAYGO` 溢出**。
- [ ] **自建 K8s 推理底座优先采用云原生托管算子（如 `AKS KAITO` 或 `GKE AI Hypercomputer Blueprints`）**：免除手工维护 NVIDIA GPU Operator 与数千 GB 模型镜像拉取的负担。
