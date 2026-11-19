# 企业级 Copilot 扩展与自建选型指南（`Adopt vs. Extend vs. Build`）：何时用 `M365 Copilot 声明式代理 + Graph Connectors`，何时用低代码 `Copilot Studio`，何时必须上全代码 `Azure AI Foundry / LangGraph`？

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-m365-copilot-extensibility-foundry-studio-pro-code-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、企业 CIO / 数字化转型负责人与 AI 卓越中心（CoE）架构师的“面对业务部门提上来的 30 个 AI 需求，到底哪些该直接配 M365 插件、哪些用 Copilot Studio 拖拽、哪些由研发团队在 Azure AI Foundry 里写代码自建”全景拍板指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_m365_copilot_extensibility_foundry_studio_pro_code_selection/results.json`](../../experiments/enterprise_m365_copilot_extensibility_foundry_studio_pro_code_selection/results.json)。

![企业级 Copilot 扩展与自建选型（M365 Declarative vs. Copilot Studio vs. Azure AI Foundry）总结图](../../assets/svg/enterprise-m365-copilot-extensibility-foundry-studio-pro-code-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

当企业 CIO 和 AI 架构师收集完各业务部门（HR、IT 运维、法务、销售、采购、研发）提报的 **30 个企业级 AI 需求清单** 时，最常犯的两个战略级错误是：
1. **错误一：“手搓 30 个独立 React 网页聊天室”反模式（Tier 1，平均每个项目耗时 `16 周`，月度开发运维总成本高达 `$265,000`，且员工日活留存率仅 `24.5%`）**：不管什么需求，研发团队都从零写一套 `React 前端 + FastAPI + 自建向量库 + 手写 SharePoint 爬虫`。结果不仅把 SharePoint 原本精细的文档权限（ACL）全搞丢了（权限保真度仅 `62.0%`），而且**没有任何一个员工愿意在浏览器收藏夹里存 30 个不同的聊天网址**——上线三个月后全成了没人用的僵尸网站！
2. **错误二：“全塞进低代码画布”反模式（Tier 2，复杂模型与工作流适配度仅 `48.0%`）**：试图用纯拖拽的低代码画布去实现需要 20 步循环状态机、自研微调模型（如 `DeepSeek-R1 / 垂直 LoRA`）或 `< 200ms` 外部高并发客户服务的核心交易 Agent，导致后期逻辑缠成死结无法维护！

在 **30 个企业 AI 场景组合（50% 办公知识与简单查询 + 30% 部门业务流自动化 + 20% 核心复杂推理/自研模型）** 的企业基准下，四档架构策略的量化对决如下：

| 方案档位 | 核心企业 AI 组合策略 | 员工日常活跃与留存率 (`Adoption`) | `M365 / SharePoint` 权限保真度 | 单个场景平均上线周期 | 复杂模型与深度编排自由度 | 30 个场景月度总成本 (`TCO`) | 架构师核心选型结论 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：全部从零手写独立 Web 应用 (`30 Custom Apps`)** | 为每个部门单独开发一套前端 + 向量库 + 权限同步管道 | `24.5%` ✕ *(30个入口导致门户疲劳)* | `62.0%` ✕ *(自写爬虫极易漏同步ACL)* | `16.0 周` ✕ | `100.0%` | `$265,000 / 月` ✕ *(重复造轮子极贵)* | ⛔ 80% 算力与人力浪费在重写登录框和权限同步上 |
| **🟨 Tier 2：全部强行塞入低代码画布 (`100% Low-Code Only`)** | 所有需求一刀切只准用可视化拖拽构建 | `72.0%` | `98.0%` | `6.5 周` | `48.0%` ✕ *(搞不定自研模型与复杂图)* | `$148,000 / 月` | ⚠️ 适合中轻度流程，遇到深度定制场景撞上天花板 |
| **🟦 Tier 3：割裂双孤岛 (`M365` 与 `Foundry` 各玩各的)** | 办公用 M365 Copilot，业务用 8 个独立的 Foundry 网页 | `58.0%` ✕ *(用户不知道该去哪问)* | `94.0%` | `8.0 周` | `92.0%` | `$162,000 / 月` | ⚠️ 后端选对了，但前端入口割裂损害用户体验 |
| **🟩 Tier 4：“50/30/20 三环分层 + 统一入口联邦”架构 (`Declarative + Studio + Foundry`)** | **前端统一收敛至 `M365 Copilot / Teams` 单一入口；后端按复杂度分流：`50%` 声明式代理 + `30%` Copilot Studio + `20%` Foundry 全代码联邦引擎** | **`94.6%` 🏆** *(全员唯一工作台入口)* | **`100.0%` 🏆** *(原生继承Graph权限)* | **`2.4 周` 🏆** *(提速 85.0%)* | **`100.0%` 🏆** *(Foundry全代码无上限)* | **`$64,000 / 月` 🏆** *(-75.8%)* | 🌟 **企业级 AI 卓越中心 (CoE) 选型治理唯一黄金标准** |

> 💡 **架构师一句话选型铁律（50 / 30 / 20 三环分流定律）**：
> - **第一环（占企业需求 `50%`）：内部知识库问答 + 简单系统查询/提单 👉 选 `M365 Copilot 声明式代理 (Declarative Agents) + Graph Connectors + API Plugins`**——不写一行前端和向量库代码，直接复用 Microsoft Graph 语义索引与 100% 原生 SharePoint/Entra 权限，**1 周上线**！
> - **第二环（占企业需求 `30%`）：跨 ERP/CRM/ITSM 的多步确定性部门审批与业务自动化 👉 选 `Microsoft Copilot Studio`（低代码）**——直接调用 1,400+ 现成 Power Platform 企业连接器与自带托管环境，**2~3 周上线**！
> - **第三环（占企业需求 `20%`）：需自托管/微调模型（`vLLM / LoRA / DeepSeek`）、复杂多智能体循环图（`LangGraph / Semantic Kernel`）、或面向外部客户的嵌入式产品 👉 选 `Azure AI Foundry`（Pro-Code 全代码）**——并且对于对内场景，将其打包为 **`Custom Engine Agent` 联邦挂载回 Teams / M365 Copilot 统一对话框**，绝不让员工打开第 2 个网址！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击 3 大选型边界、Graph Connector vs. API Plugin 与统一入口痛点）

### Q1：客户问“既然我们已经有 Python 团队能在 `Azure AI Foundry` 里写代码做 RAG 了，为什么那 50% 的内部制度/Wiki 问答不自己写，非要用 `M365 Declarative Agent + Graph Connector`？”
- **大白话与第一性原理**：
  1. **最难的从来不是写 50 行 RAG 代码，而是“实时同步 SharePoint / Confluence 的复杂权限树（ACL）”**：在企业里，一份文档的权限可能嵌套了 5 层 AD 安全组、站点继承和临时分享链接。如果研发团队自己写爬虫把文档抽到外部向量库（Tier 1），一旦下午 3 点法务部把某份并购文件夹改成“仅限项目组可见”，自建向量库因为没能毫秒级同步复杂 ACL，下午 3 点零 5 分就会把并购机密泄露给普通员工（权限失真率高达 `38.0%`）！
  2. **第一环（Declarative Agent）的降维打击**：你只需写一份几十行的 JSON 声明清单（Manifest），把外部 Confluence / ServiceNow 知识库通过 **Microsoft Graph Connector** 注入 M365 语义索引——不仅免去了自建向量库、分块、Embedding 和前端 UI 的全部成本，而且 **`100.0%` 实时继承 Microsoft Graph 的零信任安全裁剪**，交付周期从 16 周缩短到 **`1 周`**！

### Q2：在扩展 M365 Copilot 或 Copilot Studio 时，什么时候该用 `Graph Connector`（图连接器），什么时候该用 `API Plugin / MCP Action`（实时接口插件）？
- **“静态非结构化文档进 Graph，实时交易状态与写操作走 API Plugin”黄金法则**：
  - **选 `Graph Connector` 的场景**：面向**读多写少、需要全文语义搜索的非结构化文档**（如 Confluence 研发规范、内部新闻、历史合同 PDF、静态产品 Wiki）。
  - **选 `API Plugin / OpenAPI / MCP` 的场景**：面向**每秒都在变的实时交易数据或需要执行写操作（Write Action）的业务系统**（如“查一下当前 SAP 里的实时库存余量”、“在 ServiceNow 里帮我建一张 P1 故障单”、“查询 Jira 当前迭代剩余工时”）。

### Q3：如果某个高难度场景（如投行量化风控、多步代码审查 Agent）必须在 `Azure AI Foundry` 里用 `LangGraph / Semantic Kernel` 和微调模型全代码开发，怎么避免员工又要去记一个新的独立网址？
- **统一门户表面，联邦后端引擎（One UI Surface, Federated Pro-Code Engine）**：
  - 使用 **`Microsoft 365 Agents SDK` 或 `Custom Engine Agent` 协议**（或通过 Copilot Studio 的 HTTP/A2A 技能委托），将你跑在 `Azure AI Foundry / AKS vLLM` 上的全代码 Agent 直接注册进员工每天必开的 **Microsoft Teams 与 M365 Copilot 侧边栏**！
  - 员工在同一个对话框里输入 `@RiskQuantAgent`，前端体验与原生 M365 完全一致（带 Adaptive Cards 交互卡片与 SSO 登录，**日活留存率高达 `94.6%`**），而背后的请求通过 `Entra OBO` 令牌无缝路由到你的 Pro-Code Foundry 集群执行！

---

## ③ 🧭 直观可视化图解（三环选型决策树 + 统一入口三环联邦架构图 + 跨环协同调用时序图）

### 3.1 企业级 AI 需求三环分流决策树（Adopt vs. Extend vs. Pro-Code Build）

```mermaid
flowchart TD
    A["企业新增 AI 业务需求立项评审<br/>(目标：最快上线、100% 权限合规、杜绝烟囱式重复造轮子)"] --> B{"1. 该场景是面向外部客户的产品嵌入，<br/>还是面向企业内部员工的工作流赋能?"}
    B -- "面向外部客户的 App / 网站高并发内嵌" --> C["🚀 第三环 (Ring 3): 全代码 Azure AI Foundry / AKS 自建<br/>(100% 掌控定制 UI、非微软生态模型、PTU/vLLM 弹性与 CI/CD)"]
    B -- "面向企业内部员工赋能" --> D{"2. 是否必须使用微调/开源专有模型 (如 DeepSeek/LoRA)、<br/>复杂循环多智能体图 (LangGraph) 或代码执行沙箱?"}
    D -- "是 (占内部复杂需求的 20%)" --> E["🧩 第三环联邦模式: 后端用 Azure AI Foundry 全代码开发，<br/>前端打包为 Custom Engine Agent 挂载进 M365 Copilot / Teams!"]
    D -- "否 (使用标准旗舰模型即可满足)" --> F{"3. 是否需要跨多个异构系统编排复杂多步审批/写操作流，<br/>或发布到企微/钉钉/外部网页等多渠道?"}
    F -- "是 (占 30% 部门级业务流自动化)" --> G["🛠️ 第二环 (Ring 2): 低代码 Microsoft Copilot Studio<br/>(直接调用 1,400+ Power Platform 连接器与可视化流程编排)"]
    F -- "否 (占 50% 企业知识库问答 + 单步 API 查改)" --> H["🟩 第一环 (Ring 1): M365 Copilot 声明式代理 (Declarative Agent) 🏆<br/>(Graph Connector 接文档 + API Plugin 接实时查改 | 1周上线 | 100% 原生权限)"]
    style H fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style G fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style E fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
```

### 3.2 Tier 4 “单一用户体验入口 + 50/30/20 三环分层后端”企业级参考架构图

```mermaid
flowchart LR
    subgraph Surface["1. 全企业唯一交互入口 (94.6% 日活留存)"]
        UI["👩‍💼 Microsoft 365 Copilot & Teams 统一工作台<br/>(零额外前端开发 | 统一 Entra SSO 身份 | 零门户切换疲劳)"]
    end

    subgraph Rings["2. 50 / 30 / 20 三环差异化能力底座"]
        UI -->|50% 需求: 知识问答与单步查询| R1["🟢 第一环: M365 Declarative Agents (1周交付)<br/>• Graph Connectors (Confluence/SharePoint 文档, 100% 原生 ACL)<br/>• API Plugins / MCP (实时查询 SAP/Jira 状态)"]
        UI -->|30% 需求: 部门跨系统流程自动化| R2["🔵 第二环: Microsoft Copilot Studio (2~3周交付)<br/>• 1,400+ Power Platform 企业连接器<br/>• 可视化状态流与人工审批卡片"]
        UI -->|20% 需求: 深度推理与专有模型| R3["🟣 第三环: Azure AI Foundry 全代码引擎 (Custom Engine Agent)<br/>• LangGraph / Semantic Kernel 多智能体编排<br/>• 自研微调 LoRA / DeepSeek / 专用向量库与沙箱"]
        R2 -.->|复杂子节点通过 A2A/HTTP 委托调用| R3
    end
```

### 3.3 员工在同一个 `M365 Copilot` 窗口内无缝调用“第一环知识库 + 第三环 Foundry 量化模型”时序图

```mermaid
sequenceDiagram
    autonumber
    participant Emp as 企业信贷经理 (处于 Teams / M365 Copilot)
    participant M365 as 第一环：M365 Copilot 协调器与 Graph 索引
    participant Studio as 第二环：Copilot Studio 审批工作流
    participant Foundry as 第三环：Azure AI Foundry 全代码风控 Agent

    Emp->>M365: 提问：“总结客户 Acme 集团最新的尽调报告（存在 SharePoint），并测算其违约概率后发起授信审批”
    M365->>M365: 1. 通过原生 Microsoft Graph 索引（自动校验该经理对尽调文件夹的 SharePoint ACL 权限）秒级提取尽调报告摘要
    M365->>Foundry: 2. 通过 Custom Engine Agent / A2A 协议（透传员工 Entra OBO 令牌）调用全代码 Foundry 风控 Agent
    Foundry-->>M365: 运行自研微调模型与蒙特卡洛现金流沙箱，返回结构化违约率测算卡片
    M365->>Studio: 3. 员工点击卡片上的“提交授信审批”，触发 Copilot Studio 连接器写入核心信贷系统（全程未离开同一个聊天窗口！）
```

---

## ④ 📊 三环架构（Declarative vs. Copilot Studio vs. Azure AI Foundry）横向决策矩阵

### 4.1 三大能力环选型边界速查表

| 评估维度 | 🟢 第一环：`M365 Declarative Agent` | 🔵 第二环：`Copilot Studio` (Low-Code) | 🟣 第三环：`Azure AI Foundry` (Pro-Code) |
|---|---|---|---|
| **最佳适用场景（占组合比例）** | **办公知识检索 + 轻量 API 查改（`50%`）** | **部门级多步工作流与连接器编排（`30%`）** | **复杂多智能体、自研微调模型、对客产品（`20%`）** |
| **底层模型选择自由度** | 固定使用 M365 Copilot 托管旗舰模型 | 支持平台内置与关联模型切换 | 🏆 **100% 自由（任意开源/微调/蒸馏/多模态模型）** |
| **知识库与权限 (`ACL`) 机制** | 🏆 **原生 Microsoft Graph + SharePoint ACL** | 支持 Graph、Dataverse 与内置知识源 | 完全自定义（`Azure AI Search / pgvector / Milvus`） |
| **交付团队画像与上线速度** | 业务分析师 / IT 管理员（**`~1 周`**） | 敏捷创客 / 低代码工程师（**`2~3 周`**） | 专业全栈与 AI 算法工程师（**`4~6 周`**） |
| **完整 GitOps 代码评测流水线** | 声明式 JSON Manifest 级管控 | 支持基础 ALM 解决方案导出 | 🏆 **完整支持 Git PR + `Promptfoo/Pytest` + 金丝雀** |

### 4.2 四档企业整体建设策略对比（30 个 AI 业务场景组合）

| 评估维度 | 🟥 Tier 1：手写 30 个独立 Web 应用 | 🟨 Tier 2：100% 纯低代码画布 | 🟦 Tier 3：前后端割裂的多入口孤岛 | 🟩 Tier 4：50/30/20 三环分层 + 统一入口联邦 |
|---|---|---|---|---|
| **员工日常活跃与留存率** | `24.5%` ✕（30个网址没人记） | `72.0%` | `58.0%` ✕（入口割裂） | 🏆 **`94.6%`（统一收敛至员工每天必开的 M365/Teams）** |
| **SharePoint / Graph 权限保真度** | `62.0%` ✕（自写爬虫同步滞后） | `98.0%` | `94.0%` | 🏆 **`100.0%`（第一环直接走 Graph 实时 OBO 裁剪）** |
| **单个场景平均上线周期** | `16.0 周` ✕ | `6.5 周` | `8.0 周` | 🏆 **`2.4 周`（50% 需求 1 周配置上线，提速 85%）** |
| **复杂模型与深度编排自由度** | `100.0%` | `48.0%` ✕（低代码天花板） | `92.0%` | 🏆 **`100.0%`（20% 硬骨头由 Foundry 全代码承接）** |
| **30 个场景月度总成本 (`TCO`)** | `$265,000 / 月` | `$148,000 / 月` | `$162,000 / 月` | **`$64,000 / 月` 🏆（节省 75.8%，年省 $241 万）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（30 个企业 AI 需求组合：50% 知识查询 + 30% 流程自动化 + 20% 深度定制）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级 Copilot 扩展与自建组合（Declarative vs Studio vs Foundry）定容计算器](../reports/enterprise-m365-copilot-extensibility-foundry-studio-pro-code-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-m365-copilot-extensibility-foundry-studio-pro-code-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_m365_copilot_extensibility_foundry_studio_pro_code_selection/run.py --check
  python -m unittest tests/test_enterprise_m365_copilot_extensibility_foundry_studio_pro_code_selection.py
  ```

### 5.2 核心量化对比图（员工日活留存率 % vs. SharePoint 权限保真度 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: 30 Custom Standalone Portals", "metric": "Employee Daily Adoption (%)", "pct": 24.5 },
      { "tier": "Tier 1: 30 Custom Standalone Portals", "metric": "SharePoint ACL Fidelity (%)", "pct": 62.0 },
      { "tier": "Tier 2: 100% Low-Code Canvas Only", "metric": "Employee Daily Adoption (%)", "pct": 72.0 },
      { "tier": "Tier 2: 100% Low-Code Canvas Only", "metric": "SharePoint ACL Fidelity (%)", "pct": 98.0 },
      { "tier": "Tier 3: Disconnected Multi-URL Silos", "metric": "Employee Daily Adoption (%)", "pct": 58.0 },
      { "tier": "Tier 3: Disconnected Multi-URL Silos", "metric": "SharePoint ACL Fidelity (%)", "pct": 94.0 },
      { "tier": "Tier 4: 3-Ring Unified in M365 UI", "metric": "Employee Daily Adoption (%)", "pct": 94.6 },
      { "tier": "Tier 4: 3-Ring Unified in M365 UI", "metric": "SharePoint ACL Fidelity (%)", "pct": 100.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "metric": "Category",
    "pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 achieves 94.6% employee adoption and 100% SharePoint ACL fidelity at $64K/mo",
    "subtitle": "Comparison across a 30-use-case enterprise AI portfolio (higher is better)",
    "encodings": {
      "x": "tier",
      "y": "pct",
      "group": "metric",
      "color": "metric"
    }
  }
}
```

---

## ⑥ 🛠️ 架构评审与生产落地核验清单（Production Readiness Checklist）

- [ ] **在 AI 卓越中心（CoE）立项入口强制执行“先声明式扩展（Ring 1）、再低代码编排（Ring 2）、最后全代码自建（Ring 3）”的三级漏斗评审**：凡是数据已经在 SharePoint / OneDrive / Teams 或可通过标准 Graph Connector 接入的内部问答需求，一律禁止立项从零手写独立 React 聊天前端与自建向量库。
- [ ] **严格区分 `Graph Connector` 与 `API Plugin / MCP` 的职责边界**：静态非结构化知识文档入库走 Graph Connector；实时库存/工单状态查询与写操作走 API Plugin。
- [ ] **将全代码开发的 `Azure AI Foundry / LangGraph` 内部助手通过 `Custom Engine Agent` 或 `Copilot Studio` 技能联邦统一挂载回 `Microsoft Teams / M365 Copilot` 界面**：保持全公司对内 AI 入口的唯一性。
- [ ] **全链路启用 `Microsoft Entra ID On-Behalf-Of (OBO)` 委托令牌传递**：确保无论是第一环、第二环还是第三环 Agent，访问后端业务系统时均严格执行当前登录员工本人的数据权限。
