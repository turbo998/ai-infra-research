# 企业级 AI 网关 Token 预算配额治理、多部门 Chargeback 精准分账与安全护栏选型：`Azure APIM (AI Gateway)` vs. `Kong AI Gateway` vs. `LiteLLM Enterprise` vs. `Envoy AI Gateway` 架构与 TCO 指南

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-faq.en.md)  
> **文档定位**：面向 Cloud & AI Solution Architect 与企业 FinOps 负责人的 AI 网关配额限流与跨部门成本分账（Chargeback / Showback）实战指南——彻底解决集团建设共享大模型底座后的三大治理难题：**① 为什么传统的 HTTP `RPM`（每分钟请求数）限流或按 `max_tokens` 单阶段预扣会把共享 PTU/GPU 池搞瘫痪？② 怎么设计“两阶段流式加权 Token 桶（入站按路由 P80 预扣 + SSE 流结束按四元组实结退差额）”？③ 怎么通过“保底配额 + 闲时弹性借用 + 超额自动溢出至本部门 PAYGO Key”将 10 个部门共池的算力利用率从 `52.5%` 推升至 `89.4%`？**  
> **标定声明**：本文所引用的分账计费误差率、共池有效利用率、吵闹邻居误伤率与月度算力净支出基于公开企业 AI 网关架构基准构建的合成定量模型（`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`），详见 [`experiments/enterprise_ai_gateway_token_quota_chargeback_apim_kong_litellm_envoy_selection/results.json`](../../experiments/enterprise_ai_gateway_token_quota_chargeback_apim_kong_litellm_envoy_selection/results.json)。

![企业级 AI 网关 Token 配额治理与部门分账选型总结图](../../assets/svg/enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-faq-summary.svg)

> 🔗 **关联交互式 H5 精读页与工具（点击即开，零外部依赖）**：
> - 📱 **[交互式 H5 精读页：两阶段流式加权 Token 桶、闲时弹性配额借用与四大网关分账沙盘 (`docs/h5/ai-gateway-token-quota-chargeback-apim-kong-litellm-lab.html`)](../h5/ai-gateway-token-quota-chargeback-apim-kong-litellm-lab.html)**
> - 🎓 **[AI Infra 全景可视化精学馆首页 (`docs/h5/index.html`)](../h5/index.html)**
> - 🧮 **[交互式网关 Token 配额与部门分账计算器 (`enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-sizer.html`)](../reports/enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-sizer.html)**
> - 🃏 **[7-Card 4:5 视觉评测报告 (`enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-cards.html`)](../reports/enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-cards.html)**

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**在服务企业内部 10 个业务部门的共享 AI 平台上，如果沿用传统微服务网关的“按请求次数 `RPM` 限流 + 平摊账单（Tier 1）”，某部门一个跑 `64K` 长上下文 + `4K` 思维链的 Agent 循环（单次算力消耗等于普通请求的 `420 倍`！）就会瞬间打穿全公司共享 PTU/GPU 池，导致其他 9 个部门遭受 **`29.4%` 的连坐限流误伤**；而如果把总池硬切成 10 份不可互借的静态配额并按 `max_tokens` 单阶段死扣（Tier 2），高达 **`47.5%` 的共享算力会被幽灵冻结闲置**！通过落地 **Tier 4（两阶段流式加权 Token 桶 [`入站按 P80 软预扣 + SSE 结束按四元组精确结算退差`] + 分层弹性配额借用 [`总池水位 <80% 免费借用空闲配额，>=85% 瞬时收回保底并溢出至超额部门自费 PAYGO 订阅`]）**，可将跨部门 FinOps 分账误差从 `38.5%` 压至 **`0.02%`**，吵闹邻居误伤率降至 **`0.0%`**，共享池有效利用率推高至 **`89.4%`**，月度算力净支出从 `$145,000` 降至 **`$56,500`（立省 `61.0%`）**！**

| 网关配额与分账档位（10 个业务部门共池） | 限流算法与跨部门配额调度机制 | 财务分账误差率 | 共池有效利用率 | 吵闹邻居连坐误伤率 | 月度算力净支出 | 架构师拍板结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **🟥 Tier 1：传统 `RPM` 限流 + 按次平摊** | 只限每分钟调用次数，不区分 100 词与 64K 词 | **`38.5%` ✕ (极不公平)** | `48.0%` ✕ | **`29.4%` ✕ (长文打挂全池)** | `$145,000 / 月` ✕ | **严禁直接拿普通 API 网关管 LLM** |
| **🟨 Tier 2：静态硬切分 + `max_tokens` 死扣** | 10 个部门静态切死不可互借，入站按 `max_tokens` 扣减 | `16.2%` | `52.5%` ✕ (闲置无法互借) | `1.2%` | `$118,000 / 月` | 用户传 `max=16384` 导致配额被幽灵锁死 |
| **🟦 Tier 3：两阶段流式 Token 桶 + 美元硬顶** | 入站预估软扣 + SSE 结束实结退差 + 部门月度 `$` 熔断 | `0.8%` ✅ | `76.0%` ✅ | `0.3%` ✅ | `$78,000 / 月` ✅ | 标准企业级网关（APIM / LiteLLM / Kong）基线 |
| **🟩 Tier 4：四元组加权实结 + 闲时弹性借用 + 部门自费溢出 (推荐)** | **区分缓存/思考词权重 + 空闲互借 + 超额定向溢出** | **`0.02%` 🏆** | **`89.4%` 🏆** | **`0.0%` 🏆 (零连坐)** | **`$56,500 / 月` 🏆 (`-61.0%`)** | **集团级 AI 中台 FinOps 治理终局标准** |

---

## 2. 🎯 架构师灵魂拷问与两阶段加权 Token 桶手算 Q&A

### Q1：为什么流式（`stream=True`）大模型请求不能只在请求开始时扣配额，也不能只在流结束时扣配额？
* **单阶段扣费的两难死结**：
  - **只在入站时按 `N_in + max_tokens` 扣**：很多开发框架默认把 `max_tokens` 设成 `8,192` 或 `16,384`，但实际只输出了 `250` 个词——单次请求多冻结了 `16,000` 个虚拟 Token，导致部门配额在 3 秒内被“幽灵锁死”报 429！
  - **只在流结束（`[DONE]`）时扣**：如果恶意或失控脚本在 1 秒内并发发出 200 个长思维链请求（每个要跑 40 秒才结束），在最初的 40 秒内网关账本读数全是 `0`，200 个请求全部放行直接打爆后端 GPU 显存！
* **两阶段流式加权 Token 桶（Two-Phase Weighted Streaming Token Bucket）精确公式**：
  1. **阶段一（Pre-Flight 入站软预扣）**：计算精确输入词数 $N_{\text{in}}$，并加上该路由历史 P80 平均输出词数 $\hat{N}_{\text{out\_p80}}$ 的加权值：
     $$\Delta Q_{\text{reserve}} = N_{\text{in}} + w_{\text{out}} \times \min\left(\text{max\_tokens},\; \hat{N}_{\text{out\_p80}}\right) \quad (w_{\text{out}} = 4)$$
  2. **阶段二（Post-SSE 流结束/断连毫秒级实结退差）**：从引擎返回的 `usage` 块读取精确的 **四元组 `(Uncached_In, Cached_In, Reasoning_Tokens, Output_Tokens)`**，按真实物理成本权重结算并立即退还多预扣的差额：
     $$\Delta Q_{\text{actual}} = N_{\text{uncached\_in}} + 0.5 \times N_{\text{cached\_in}} + 4 \times \left(N_{\text{reasoning}} + N_{\text{output}}\right), \qquad Q_{\text{bucket}} \leftarrow Q_{\text{bucket}} + \left(\Delta Q_{\text{reserve}} - \Delta Q_{\text{actual}}\right)$$

### Q2：四大主流企业 AI 网关（`Azure APIM`、`Kong AI Gateway`、`LiteLLM Enterprise`、`Envoy AI Gateway`）在配额与分账上各有什么绝活？
1. **`Azure API Management (AI Gateway)`**：内置原生 **`llm-token-limit` 与 `azure-openai-token-limit` 策略**（支持按 Subscription / JWT Claim / 部门 ID 实施流式 Token 预估与实结限流）、原生发射 `llm-emit-token-metric` 至 Azure Monitor / Application Insights，且支持 PTU 满载时 **按部门策略定向溢出至该部门自己的 PAYGO 订阅**。
2. **`LiteLLM Enterprise Proxy`**：提供最直观的 **Virtual Keys（虚拟密钥）+ Team / Organization / End-User 三级美元预算上限（`max_budget = $500/mo`）** 与实时成本看板，支持自动按模型单价表折算缓存输入与思考词成本，最受研发中台欢迎。
3. **`Kong AI Gateway`**：企业级高吞吐南北向网关，提供 **`AI Rate Limiting Advanced` 插件**（支持多云提供商 Token 计数与按成本权重限流）及完善的 RBAC 审计。
4. **`Envoy AI Gateway (CNCF / K8s GIE)`**：原生支持基于 **CEL 表达式与 HTTP Header/JWT Claim 的多维 Token 速率限制**，并与底层 K8s GPU Pod 的 KV 缓存水位调度无缝联动。

---

## 3. 🧭 直观可视化图解（两阶段流式扣费 + 闲时借用与定向溢出架构图）

```mermaid
flowchart TB
    Req["👤 部门请求 (携带 JWT: dept_id='marketing')"] --> P1["⚡ Phase 1: 入站软预扣\nReserve = N_in + 4 * P80_Out"]

    P1 --> Check{"检查本部门保底配额\n与总池实时水位?"}
    Check -->|"本部门保底配额充足\n或 (本部门超额 但 总池水位 < 80%)"| Shared["🟢 放行进入集团共享基线池\n(共享 PTU + 自建 K8s GPU 池 · 吃满 89.4% 利用率!)"]
    Check -->|"本部门已超保底配额\n且 总池繁忙水位 >= 85%"| Spill["🟨 定向溢出至 Marketing 部门自费 PAYGO 订阅\n(绝不挤占其他部门保底配额 · 0% 连坐误伤!)"]

    Shared & Spill --> P2["🧾 Phase 2: SSE 流结束 / Abort 断连\n按 (Uncached_In, Cached_In, Think, Out) 四元组实结\n秒级退差额 + 写入 OpenTelemetry FinOps 账本 🏆"]
```

---

## 4. 📊 四大主流 AI 网关配额治理与 FinOps 分账能力对比矩阵

| 评估维度 | `Azure APIM (AI Gateway)` | `LiteLLM Enterprise` | `Kong AI Gateway` | `Envoy AI Gateway (K8s GIE)` |
| :--- | :--- | :--- | :--- | :--- |
| **流式 Token 限流策略** | **原生 `llm-token-limit`（支持预估+实结）** 🏆 | 支持 RPM / TPM 与并发流限制 | **`AI Rate Limiting Advanced` 插件** 🏆 | **原生 Token 级速率限制滤镜** 🏆 |
| **美元预算上限 (`$ Cap`)** | 结合 Azure Cost Management 预算告警 | **原生内置 Key / Team / Org 级实时美元硬熔断** 🏆 | 支持插件扩展 | 结合外部计费控制面 |
| **缓存与思考词四元组分账** | 支持按响应 `usage` 明细发射自定义指标 | **原生解析 `cached_tokens` 与 `reasoning_tokens` 计价** 🏆 | 支持多云 Usage 明细提取 | 原生发射 OpenTelemetry GenAI 指标 |
| **最佳选型定位** | **Azure / 混合云集团南北向总网关** | **企业内部多团队自助式 API 分发与成本门户** | **跨多云高性能南北向 API 治理网关** | **K8s 自建 GPU 集群东西向推理网关** |

---

## 5. 🛠️ 生产落地核对清单（SA Production Readiness Checklist）

- [ ] **淘汰纯 `RPM` 限流，全面切换为“加权 `TPM`（`1 Out = 4 In`）”限流**：确保长思维链和大输出请求按真实占用的 GPU 解码时间扣减配额。
- [ ] **开启流式响应 `stream_options: {"include_usage": true}`**：网关转发 SSE 请求时强制注入该参数，确保流式最后一个 Chunk 必带精确的四元组 Token 统计用于第二阶段实结。
- [ ] **配置“保底配额 + `<80%` 闲时免费借用 + `>=85%` 超额定向溢出本部门 PAYGO”**：彻底终结部门间配额碎片化闲置与高峰期“吵闹邻居”连坐故障。
