# 企业级多智能体（Multi-Agent）生产平台与安全工具沙箱选型：`Azure AI Agent Service (Foundry)` vs. `LangGraph Platform` vs. `Semantic Kernel / AutoGen + AKS Dynamic Sessions` 架构、状态持久化与 TCO 指南

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-multi-agent-platform-foundry-langgraph-autogen-aks-tco-selection-faq.en.md)  
> **文档定位**：面向 Cloud & AI Solution Architect 的多智能体（Multi-Agent）生产架构与基础设施定容指南——直击当前企业从单 Agent 走向 5~10 个协作 Multi-Agent 时最痛的三大工程陷阱：**① 为什么用朴素“群聊广播（GroupChat Broadcast）”跑 5 个 Agent 协作会让单任务 Token 消耗从 `3.8 万` 暴涨到 `28.5 万`（膨胀 `7.5 倍`）？② 为什么多个不同角色的 Agent 轮流调用同一个 vLLM/SGLang 实例会导致 Prefix KV 缓存命中率从 `92%` 暴跌至 `18.4%`？③ `Azure AI Agent Service (Foundry)`、`LangGraph Platform` 与 `Microsoft Agent Framework (SK + AutoGen) on AKS` 该怎么选型？**  
> **标定声明**：本文所引用的单工作流 Token 消耗、Prefix KV 缓存命中率、20 步工作流完结率与每千次运行成本基于公开多智能体系统基准构建的合成定量模型（`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`），详见 [`experiments/enterprise_multi_agent_platform_foundry_langgraph_autogen_aks_tco_selection/results.json`](../../experiments/enterprise_multi_agent_platform_foundry_langgraph_autogen_aks_tco_selection/results.json)。

![企业级多智能体平台选型与 Prefix KV 粘性路由总结图](../../assets/svg/enterprise-multi-agent-platform-foundry-langgraph-autogen-aks-tco-selection-faq-summary.svg)

> 🔗 **关联交互式 H5 精读页与工具（点击即开，零外部依赖）**：
> - 📱 **[交互式 H5 精读页：多智能体 Token 膨胀手算、角色粘性 KV 路由与三大平台选型沙盘 (`docs/h5/multi-agent-foundry-langgraph-autogen-aks-tco-lab.html`)](../h5/multi-agent-foundry-langgraph-autogen-aks-tco-lab.html)**
> - 🎓 **[AI Infra 全景可视化精学馆首页 (`docs/h5/index.html`)](../h5/index.html)**
> - 🧮 **[交互式多智能体 Token 与 TCO 计算器 (`enterprise-multi-agent-platform-foundry-langgraph-autogen-aks-tco-selection-sizer.html`)](../reports/enterprise-multi-agent-platform-foundry-langgraph-autogen-aks-tco-selection-sizer.html)**
> - 🃏 **[7-Card 4:5 视觉评测报告 (`enterprise-multi-agent-platform-foundry-langgraph-autogen-aks-tco-selection-cards.html`)](../reports/enterprise-multi-agent-platform-foundry-langgraph-autogen-aks-tco-selection-cards.html)**

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**在构建包含 5 个专职角色（规划师、SQL 问数员、Python 沙箱程序员、安全审计员、报告合成师）的 15 步企业级多智能体工作流时，如果直接采用“全员群聊广播（Tier 1，把所有中间报错、10KB 原始 SQL 表与代码堆栈全量广播给每个 Agent）”且走随机轮询网关，不仅单次运行要烧掉 **`285,000 Tokens`**，而且由于 5 个角色各自不同的 `System Prompt + Tools Schema` 在 GPU 间交替冲刷，底层 Prefix KV 缓存命中率会崩塌至 **`18.4%`**！通过落地 **Tier 3（面向业务敏捷交付的 `Azure AI Agent Service + ACA Dynamic Sessions` 全托管底座）** 或 **Tier 4（`结构化 Artifact 摘要交接 [<800 Tok/步]` + `按 Agent_Role_ID 粘性路由的 Envoy GIE KV 池` + `步级 Checkpoint 事件溯源重放 + Firecracker/Hyper-V 隔离沙箱`）**，可将单次工作流 Token 消耗从 `28.5 万` 暴压至 **`38,000 Tokens`（立省 `86.7%`）**，Prefix KV 缓存命中率恢复至 **`92.4%`**，20 步长任务完结率从 `71.5%` 推升至 **`99.4%`**，每千次工作流成本从 `$420` 暴降至 **`$85`（节省 `79.8%`）**！**

| 多智能体平台架构档位（5 Agent × 15 步工作流） | 上下文传递与底层调度机制 | 单次工作流总 Token | Prefix KV 命中率 | 20 步任务完结率 | 每千次运行总成本 | 架构师拍板结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **🟥 Tier 1：内存态全员群聊广播 (`GroupChat`)** | 每个 Agent 接收全员完整历史对话与原始工具大输出 | **`285,000 Tok` ✕** | **`18.4%` ✕ (交替冲刷)** | `71.5%` ✕ (超时丢状态) | `$420 / 1K Runs` ✕ | **头号烧钱反模式**：Token 呈二次方爆炸 |
| **🟨 Tier 2：`LangGraph` 状态图 + 普通轮询网关** | 显式 DAG 状态机 + Postgres 落盘，但无 GPU 角色粘性 | `64,000 Tok` | `42.0%` | `96.2%` ✅ | `$165 / 1K Runs` | 解决了状态持久化，但 GPU 端仍频繁重算角色前缀 |
| **🟦 Tier 3：`Azure AI Agent Service (Foundry)` 全托管** | **托管线程 + Entra OBO 鉴权 + `ACA Dynamic Sessions` 沙箱** | `48,000 Tok` ✅ | `84.5%` ✅ | **`98.8%` ✅** | `$118 / 1K Runs` ✅ | **M365/Fabric/Azure 企业业务快速投产首选** |
| **🟩 Tier 4：结构化交接图 + 按角色 `GIE KV` 粘性路由 (推荐)** | **每步仅传 `<800 Tok` JSON 契约 + 按 `Agent_Role` 锁定 KV 池** | **`38,000 Tok` 🏆 (`-86.7%`)** | **`92.4%` 🏆** | **`99.4%` 🏆** | **`$85 / 1K Runs` 🏆 (`-79.8%`)** | **高并发自建/混合多智能体中台终局标准** |

---

## 2. 🎯 架构师灵魂拷问与多智能体容量手算 Q&A

### Q1：为什么单 Agent 跑得好好的 Prefix KV Cache，一旦改成 5 个 Agent 协作，GPU 缓存命中率就从 90% 暴跌到 18%？怎么根治？
* **“多角色前缀互相冲刷（Role Prefix Thrashing）”机理剖析**：
  - 每个专职 Agent 都拥有自己独特的 `4,000 Tokens` 角色定义与专属 MCP 工具集：
    - `Agent_SQL` 前缀：`[SQL 专家指令 + 30 张表 Schema 定义]`
    - `Agent_Coder` 前缀：`[Python 沙箱指令 + Pandas/Plotly API 规范]`
  - 在同一条工作流中，第 1 步调 `Agent_SQL`，第 2 步调 `Agent_Coder`，第 3 步调 `Agent_Reviewer`……如果你按 **`Session_ID`（用户会话 ID）** 做网关粘性路由，这 5 个完全不同前缀的请求会被送到**同一个 GPU 实例的同一个会话槽位**上！结果每一跳的开头第 1 个 Token（System Prompt）都不一样，导致 **上一跳的 Prefix KV Cache 被 100% 判为失效**！
* **两大根治法则（架构师必背）**：
  1. **网关路由键从 `Session_ID` 改为 `(Agent_Role_ID, Session_ID)`**：通过 `Envoy AI Gateway (K8s GIE EPP)` 将所有 `Agent_SQL` 的请求路由到缓存了 SQL Schema 的专属 Pod 组，将所有 `Agent_Coder` 的请求路由到缓存了代码工具的 Pod 组——这样每个角色在自己的专属 KV 池里始终保持 **`92.4%` 的极高前缀命中率**！
  2. **废除“全员共享聊天记录”，改用“结构化交接信封（Structured Handoff Artifact）”**：`Agent_Coder` 在沙箱里跑了 5 次报错调试并读了 50MB CSV，这些中间过程**100% 留在 `Agent_Coder` 的私有子线程内**；交接给下一个 `Agent_Reporter` 时，只传递一个 **`< 800 Tokens` 的经过校验的 Pydantic JSON 结论摘要 + 文件存储 URI 指针**！

### Q2：`Azure AI Agent Service (Foundry)`、`LangGraph Platform` 和 `Microsoft Agent Framework (SK + AutoGen)` 三大主流底座怎么选？
1. **`Azure AI Agent Service (Microsoft Foundry)` + `Azure Container Apps (ACA) Dynamic Sessions`**：
   - **核心优势**：全托管有状态线程（Threads）、原生集成 **Entra Agent ID & On-Behalf-Of (OBO) 用户级权限透传**、开箱对接 Azure AI Search / Microsoft Fabric / Logic Apps 1,400+ 企业连接器，且内置基于 **Hyper-V 硬件虚拟化隔离的 `ACA Dynamic Sessions`（预热池 `<150 ms` 分配安全 Python 代码解释器沙箱）**。
   - **最佳场景**：需要快速打通企业内部 M365、CRM、ERP 且对安全合规审计要求极高的企业业务团队。
2. **`LangGraph Platform (Stateful Graph Engine)`**：
   - **核心优势**：将多智能体协作建模为显式有向有环状态图（StateGraph），每一步节点执行完自动将状态序列化写入 PostgreSQL Checkpointer，完美支持 **跨天人工审批挂起（`interrupt()` 零内存占用）与任意历史节点时间旅行重放（Time-Travel Debugging）**。
   - **最佳场景**：复杂确定性 SOP 工作流、深度研究（Deep Research）与需要细粒度步级回滚的 Python AI 团队。
3. **`Microsoft Agent Framework (Semantic Kernel + AutoGen 统一底座) on AKS + Temporal/Dapr`**：
   - **核心优势**：融合了 AutoGen 的多智能体编排模式与 Semantic Kernel 的企业级 C#/.NET & Python 双语言类型安全、插件フィルター与 OpenTelemetry GenAI 原生遥测，配合 `Temporal / Dapr Workflow` 实现百万级并发持久化编排。

---

## 3. 🧭 直观可视化图解（结构化交接状态图 + 按角色粘性 KV 路由架构）

```mermaid
flowchart TB
    subgraph Orchestrator["🧭 持久化多智能体状态机编排层 (LangGraph / Foundry / SK + Temporal)"]
        O1["步级状态 Checkpointer (Postgres/Cosmos) · 人工审批 0 内存休眠\n跨 Agent 仅传递 <800 Tok 结构化 Handoff JSON + 对象存储 URI 指针"]
    end

    subgraph GIE["⚡ K8s GIE EPP 按角色亲和路由网关 (Route by Agent_Role_ID)"]
        R1["保持各专职 Agent 的 System Prompt + MCP 工具 Schema 100% 驻留对应 Pod HBM"]
    end

    subgraph Pools["🖥️ 角色亲和 GPU 推理池与安全微虚拟机沙箱"]
        P1["Pod Group A (Planner & Reviewer)\n常驻规划提示词 KV (命中 94%)"]
        P2["Pod Group B (SQL & RAG Analyst)\n常驻表结构 MDL KV (命中 92%)"]
        P3["Pod Group C (Coder) + ACA Dynamic Sessions / Firecracker\n140ms 硬件隔离代码执行 (原始 50MB CSV 不进 Prompt!) 🏆"]
    end

    Orchestrator --> GIE
    GIE --> P1 & P2 & P3
```

---

## 4. 📊 三大主流企业级多智能体平台全方位选型矩阵

| 对比维度 | `Azure AI Agent Service (Foundry)` | `LangGraph Platform` | `SK / AutoGen (MS Agent) + AKS` |
| :--- | :--- | :--- | :--- |
| **状态持久化与断点恢复** | 云端全托管 Threads 状态管理 | **原生步级 Checkpointer + 时间旅行重放** 🏆 | 对接 `Temporal / Dapr / Postgres` |
| **安全代码执行沙箱** | **内置 `ACA Dynamic Sessions` (Hyper-V `<150ms`)** 🏆 | 需外接 E2B / Firecracker / GKE Sandbox | 对接 `ACA Dynamic Sessions` 或自建 `Firecracker/Kata` |
| **企业身份鉴权 (`OBO`)** | **原生 Entra Agent ID + OAuth 2.1 OBO** 🏆 | 需自行在节点内实现 Token 传递 | 深度集成 Microsoft Entra 与企业网关 |
| **开发语言与生态** | Python / C# / REST API 全托管 | **Python / TypeScript 社区最活跃** 🏆 | **C# (.NET) 与 Python 双栈企业标准** 🏆 |

---

## 5. 🛠️ 生产落地核对清单（SA Production Readiness Checklist）

- [ ] **禁止多智能体共享全量聊天历史**：采用 **Supervisor / StateGraph 模式**，子 Agent 在独立上下文完成工具循环后，仅向主图返回经过 Schema 校验的 `< 800 Tokens` 结构化结论与数据文件 URI。
- [ ] **按 `Agent_Role_ID` 配置网关 KV 亲和路由**：确保同一角色的请求始终命中存有该角色 System Prompt 与 MCP 工具定义的 GPU Pod。
- [ ] **大型数据表与文件走“沙箱引用传递（Pass-by-Reference）”**：将 SQL 查询结果或 Excel 文件直接挂载进 `ACA Dynamic Sessions / Firecracker` 沙箱文件系统，严禁把几千行 CSV 序列化进 LLM Prompt。
