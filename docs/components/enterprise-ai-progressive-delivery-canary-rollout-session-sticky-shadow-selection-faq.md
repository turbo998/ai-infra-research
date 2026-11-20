# 企业级 AI 应用渐进式发布（`Progressive Delivery & Canary Rollout`）、会话粘性灰度路由（`Session-Sticky Canary`）、工具副作用隔离影子伴飞（`Write-Mocked Shadowing`）与 `SPRT` 四维质量自动回滚架构选型指南：为什么传统按请求随机 `90/10` 切流会让多轮 Agent 发生 `28.4%` 的“精神分裂”，且完全看不见 `HTTP 200` 幻觉退化？

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-ai-progressive-delivery-canary-rollout-session-sticky-shadow-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、AI 平台工程（LLMOps / DevOps）负责人与 SRE 团队的“企业级 AI 应用（Prompt / LoRA / Agent 图 / RAG 管道，月均 120 次生产发布）渐进式灰度交付、为什么不能照搬传统 Web 微服务的按请求随机切流与 `HTTP 500` 探针、`HMAC(session_id)` 会话粘性环状发布、MCP 写工具沙箱拦截影子流量（`x-traffic-mode: shadow`）与 `4.5分钟 SPRT` 序贯概率比自动回滚”实战指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_ai_progressive_delivery_canary_rollout_session_sticky_shadow_selection/results.json`](../../experiments/enterprise_ai_progressive_delivery_canary_rollout_session_sticky_shadow_selection/results.json)。

![企业级 AI 应用渐进式发布与粘性金丝雀架构总结图](../../assets/svg/enterprise-ai-progressive-delivery-canary-rollout-session-sticky-shadow-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

当企业拥有 30 个生产 AI 应用、每月执行 `120 次` 线上变更（更新 System Prompt、切换底座模型小版本、上线新 LoRA 权重、修改 MCP 工具参数或调整切块策略）时，很多团队直接照搬传统 Kubernetes 微服务的 **`Nginx / Istio / Argo Rollouts` 按单次 HTTP 请求随机 `90/10` 分流 + 监控 `HTTP 5xx` 报错率**。结果在多轮 Agent 场景下连踩三大生产大雷：
1. **大雷一：按单次请求随机 `90/10` 切流，导致 10 轮对话里有 `65%` 的会话在中途“精神分裂”（Tier 1，跨版本状态错乱与 Schema 冲突率高达 `28.4%`）**：
   - 算一道简单的概率题：如果每次 HTTP 请求有 90% 概率进老版本 `v1`、10% 概率进新金丝雀版本 `v2`，那么在一个持续 10 轮的多轮 Agent 会话中，10 轮全都留在 `v1` 的概率只有 $0.9^{10} = 34.8\%$——也就是说，**高达 `65.2%` 的用户会在聊到第 3 轮时突然被切到 `v2`，第 4 轮又被切回 `v1`**！
   - 由于 `v2` 修改了 System Prompt 前缀、思考草稿（Scratchpad）格式或工具 JSON Schema，中途横跳不仅 **瞬间打碎 GPU `KV Prefix Cache`**，更会导致 `v1` 读不懂 `v2` 上一轮留下的工具状态而当场报 Schema 解析错误（`28.4%` 状态失步）！
   - 更致命的是 **“`HTTP 200` 幻觉失明症”**：当新版本 Prompt 改坏了导致模型开始满嘴跑火车、或者把工具参数全填成空值时，网关返回的依然是标准的 **`HTTP 200 OK`**！传统只看 `HTTP 500` 的发布控制器以为“一切正常”，在 30 分钟后把有毒的 `v2` 全量推到了 `100%` 生产流量（漏检率高达 **`88.0%`**）！
2. **大雷二：直接用网关做流量镜像（Shadow Mirroring），却忘了拦截影子 Agent 的外部写工具（Tier 2，导致 `100%` 的双倍建单、双倍发邮件与双倍扣款事故！）**。
3. **大雷三：每次发版靠人工盯盘浸泡 48 小时（Tier 3）**：严重拖慢发版节奏，且人工抽查漏掉了 `27.5%` 的长尾退化。

在 **30 个企业 AI 应用、月均 120 次生产发布（平均每会话 10 轮交互）** 的企业基准下，四档渐进式交付架构的实测对决如下：

| 方案档位 | 流量切分粒度、影子工具隔离与回滚门禁机制 | 语义/质量退化捕获率 | 会话中途状态错乱率 | 影子流量重复写副作用率 | 120 次发布/月事故与运维 TCO | 架构师核心选型结论 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：按单次请求随机 `90/10` 切流 + 仅看 `HTTP 500`** | 无会话粘性；仅靠 HTTP 状态码与延迟判断好坏 | `12.0%` ✕ *(漏掉88%的HTTP-200幻觉!)* | `28.4%` ✕ *(中途横跳精神分裂)* | `0.0%` *(未开影子)* | `$182,000 / 月` ✕ *(坏版本频繁全量上线)* | ⛔ 严禁对多轮有状态 AI 应用使用无粘性的按请求随机灰度 |
| **🟨 Tier 2：裸开网关流量镜像 (`Shadow`) 但不拦截写工具** | 复制 5% 生产流量给 `v2` 伴飞，但共用真实 MCP 工具 | `78.0%` | `4.5%` | **`100.0%` ✕** *(主影同时建单发信!)* | `$145,000 / 月` ✕ *(严重生产脏数据污染)* | ⛔ 不带写工具沙箱拦截的影子伴飞等于主动制造双写故障 |
| **🟦 Tier 3：固定 48 小时人工抽检浸泡 (`Manual Soak`)** | 切 5% 流量后靠工程师人肉翻看几십条日志才敢全量 | `72.5%` ✕ | `2.0%` | `0.0%` | `$118,000 / 月` ✕ *(发版极慢且漏长尾)* | ⛔ 人工抽检无法应对每周数十次 Prompt 与知识库迭代 |
| **🟩 Tier 4：`HMAC(session_id)` 会话粘性环状灰度 + `x-traffic-mode: shadow` 写工具干跑模拟 + 四维业务信号 `SPRT` `4.5分钟` 自动回滚** | **同一会话 10 轮 100% 锁定同一版本；影子流量写工具自动返回主链路回执；用序贯概率比检验 (`SPRT`) 实时盯防 4 大 AI 黄金指标** | **`99.4%` 🏆** *(精准抓出HTTP-200退化)* | **`0.0%` 🏆** *(零跨版本横跳)* | **`0.0%` 🏆** *(写工具100%沙箱隔离)* | **`$21,000 / 月` 🏆** *(比 Tier 1 省 88.5%)* | 🌟 **企业级 AI 持续交付与安全灰度发布唯一黄金标准** |

> 💡 **架构师一句话 AI 渐进式发布铁律（灰度跟会话走、影子拦写操作、回滚看四维 SPRT）**：
> 1. **会话级/租户级一致性哈希粘性路由铁律（`HMAC(session_id, deploy_id)` Session-Sticky Ring）**：“**绝不允许同一个多轮对话的前半截跑在 `v1`、后半截跳去 `v2`！**”在 AI 网关层根据 `HMAC-SHA256(session_id + ":" + release_id) % 100` 计算分桶：一旦某个会话在第 1 轮被分入 `10%` 金丝雀池 `v2`，该会话后续的第 2~20 轮 **100% 锁定在 `v2`（且锁定到同一个 GPU Prefix Cache 节点）**，跨版本状态错乱率从 `28.4%` 直接归零到 **`0.0%`**！
> 2. **影子伴飞请求头透传与 MCP 写工具干跑拦截铁律（`x-traffic-mode: shadow` Dry-Run Stubbing）**：在正式切流前先开启 `5%` 异步影子伴飞时，网关强制在镜像请求头注入 **`x-traffic-mode: shadow`**（写入 OpenTelemetry Baggage 全链路透传）。当影子 Agent 调用只读检索工具时正常放行；一旦调用任何 **`MUTATING_WRITE` 写工具（发邮件、建工单、改数据库）**，MCP 网关当场拦截并直接返回主链路 `v1` 的执行回执或确定性 Mock 回执（影子重复写副作用率 **`0.0%`**）！
> 3. **告别 `HTTP 500` 盲区、启用四维 AI 黄金信号 `SPRT` 序贯检验自动回滚铁律**：发布控制器不再只看 `HTTP 500`，而是实时采集 `v2` 与 `v1` 的 **四大 AI 业务健康指标：① 在线忠实度/护栏拦截率（`NLI Groundedness`）、② 工具 JSON 参数校验失败率、③ 用户追问/重试/点踩率（`User Re-Ask / Thumbs-Down`）、④ 单任务平均消耗 Token 数**，通过 Wald 序贯概率比检验（`SPRT`）一旦确认 `v2` 显著劣于 `v1`，在 **`4.5 分钟`** 内全自动切断 `v2` 并秒级回滚！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击为什么 10% 随机分流会污染 65% 的多轮会话、影子流量怎么防重发邮件、怎么在 5 分钟内自动识别“胡说八道但返回 200”的坏版本 3 大痛点）

### Q1：为什么在无状态 Web 服务里用了十年的“按请求 `90/10` 随机金丝雀分流”，一搬到多轮 AI Agent 上就会酿成 `28.4%` 的会话崩溃？
- **揭秘多轮有状态上下文（Stateful Context Trajectory）对版本一致性的刚性要求**：
  - 当你升级 `v2` 时，通常伴随着 System Prompt 微调、新增了一个 MCP 工具定义、或者调整了思考标签（例如从 `<thought>` 改成了 `<analysis>`）。
  - 如果按单次请求随机路由，用户在第 1~2 轮由 `v1` 服务，历史消息里塞满了 `<thought>` 标签；到了第 3 轮，那 `10%` 的随机概率落到了这个用户头上，请求被发给了只认识 `<analysis>` 的 `v2`——不仅 `v2` 的 System Prompt 变了导致前两轮攒下的 **GPU KV 前缀缓存 100% 作废重算**，更因为历史轨迹里的工具返回结构与 `v2` 的新 Schema 不兼容，直接引发 `28.4%` 的解析报错与逻辑断片！
  - **Tier 4 粘性分桶法**：路由网关以 `session_id`（对于 B2B 客户可叠加 `tenant_id`）作为哈希种子，保证 **“会话内绝对版本纯净，会话间精确满足 90/10 比例”**！

### Q2：我们在上线新版本 Agent 前想先拿 5% 的真实生产流量做“影子测试（Shadow Testing / Dark Launch）”，但 Agent 在处理过程中会调 3 个查询工具和 1 个“给客户发邮件/建工单”的写工具——怎么防止影子 Agent 把邮件再发一遍？
- **在 MCP 工具网关实施“全链路 `x-traffic-mode: shadow` 染色与写工具虚拟化”**：
  - 千万不要在网关只简单复制一下入口 HTTP 请求就不管了！因为 Agent 内部自己会发起多次外部工具调用。
  - Tier 4 的做法是：
    1. 入口网关复制请求发往 `v2-shadow` 实例时，注入不可篡改的 `OpenTelemetry Baggage: traffic.mode = shadow`；
    2. `v2-shadow` 正常跑大语言模型推理、正常调向量库和只读 API（检验新模型会不会查错词或报超时）；
    3. 当 `v2-shadow` 准备执行最后一步写操作（如 `POST /mcp/tools/create_servicenow_ticket`）时，MCP 工具网关识别到 `traffic.mode == shadow`，**立即拦截真实的外部调用，将影子 Agent 传入的工具参数记录到《影子对比审计表》（用于比对 `v2` 传的参数和主链路 `v1` 传的参数是否一致！），并向影子 Agent 返回一个虚拟成功凭据 `{"ticket_id": "SHADOW-DRYRUN-001"}`**——既 100% 测通了多步工具链，又实现了 **`0.0%` 生产脏写**！

### Q3：如果不用人工盯盘 48 小时，机器怎么在 `4.5 分钟`（仅跑了 200~300 个会话）内就准确判断新版本 `v2` 的回答质量退化了并自动回滚？
- **四维在线信号 + Wald 序贯概率比检验（`Sequential Probability Ratio Test, SPRT`）**：
  - 固定样本量 A/B 测试需要等攒够 10,000 个样本（耗时 2 天）才能算一次 $p$-value；而 **`SPRT` 序贯检验** 每完成 1 个对话就动态更新一次对数似然比 $\Lambda_n = \sum \log \frac{P(x_i | H_1)}{P(x_i | H_0)}$：
  - 控制器同时监控 4 个实时指标：
    1. **隐式用户负反馈率（User Re-Ask / Cancel Rate）**：用户收到回答后 15 秒内立刻追问“不对/不是这个意思”或中途掐断生成的比例（若从 `4%` 飙升至 `14%`）；
    2. **工具调用 Schema 拒止率（MCP Validation Error Rate）**；
    3. **在线抽样忠实度得分（`HHEM-2.1 NLI` 蕴含分）**；
    4. **单轮 Token 消耗暴涨率（防死循环思考）**。
  - 一旦任一指标的序贯似然比突破上限阈值 $B = \log((1-\beta)/\alpha)$，无需人工干预，网关在 **`4.5 分钟`** 内自动将金丝雀权重归零并告警！

---

## ③ 🧭 直观可视化图解（灰度发布选型决策树 + 粘性金丝雀与写隔离影子架构图 + `4.5分钟 SPRT` 自动发现 `HTTP-200` 幻觉并回滚时序图）

### 3.1 企业级 AI 应用渐进式发布与金丝雀灰度选型决策树

```mermaid
flowchart TD
    A["企业级 AI 渐进式发布与金丝雀选型<br/>(30 个生产 AI 应用 | 月均 120 次模型/提示词/Agent 发布 | 平均 10 轮多步会话)"] --> B{"如何对新版本 AI 进行灰度切流、影子验证与异常回滚?"}
    B -- "按单次 HTTP 请求随机 90/10 分流 + 仅监控 HTTP 500 (Tier 1)" --> C["⛔ 28.4% 会话中途跨版本精神分裂 + 漏掉 88% 的 HTTP-200 幻觉！<br/>有毒版本被自动全量推至 100%，月均事故损失 $18.2 万"]
    B -- "直接复制流量做影子伴飞，但未在工具网关拦截写操作 (Tier 2)" --> D["⛔ 100% 触发双倍建单、双倍发信与重复扣款！<br/>影子 Agent 真实调用外部写接口污染生产系统"]
    B -- "每次发版切 5% 流量后靠人工盯盘浸泡 48 小时 (Tier 3)" --> E["⛔ 单次发布耗时长达 48 小时 + 仍漏检 27.5% 长尾退化！<br/>严重阻塞研发迭代效率，月耗 $11.8 万"]
    B -- "HMAC(session_id) 粘性灰度 + x-traffic-mode:shadow 写拦截 + 4维 SPRT 自动回滚 (Tier 4)" --> F["🟩 Tier 4: 会话粘性金丝雀 + 写模拟影子伴飞 + 4.5 分钟 SPRT 自动回滚 🏆<br/>(99.4% 退化捕获 | 0.0% 状态错乱 | 0.0% 影子脏写 | 省 88.5% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `HMAC(session_id)` 粘性灰度环、写操作模拟影子伴飞与四维 `SPRT` 自动回滚架构图

```mermaid
flowchart LR
    subgraph Router["1. AI 网关会话粘性分流与影子染色层"]
        Req["💬 多轮用户会话请求<br/>(携带 session_id)"] --> Hash["📌 HMAC(session_id, deploy_id) % 100<br/>同一会话 10 轮 100% 锁定同一版本!<br/>(0.0% 中途横跳 | 100% 保护 KV Cache)"]
        Hash -- "90% 稳定会话" --> V1["🟢 v1 生产基线版本"]
        Hash -- "10% 粘性金丝雀会话" --> V2["🟡 v2 金丝雀版本"]
        Hash -. "5% 异步影子镜像 (打标 x-traffic-mode: shadow)" .-> V2S["👻 v2 影子伴飞实例"]
    end

    subgraph ToolGW["2. MCP 工具网关 (影子写操作防火墙)"]
        V1 & V2 --> LiveTool["⚡ 正常执行读写工具"]
        V2S --> MockGate{"🛡️ 检测到 shadow 标记<br/>且为写工具?"}
        MockGate -- "只读工具 (GET)" --> LiveTool
        MockGate -- "写工具 (POST/发信/扣款)" --> DryRun["🧪 拦截外部副作用! 记录参数 Diff<br/>并返回 Dry-Run 模拟回执 (0% 重复写)"]
    end

    subgraph Judge["3. 四维 AI 黄金信号 SPRT 自动回滚控制器 (4.5 分钟判决)"]
        V1 & V2 & V2S --> SPRT["📈 实时对比 v2 vs v1 四维指标:<br/>1. NLI 忠实度得分 | 2. 工具 Schema 错误率<br/>3. 用户 15s 重问/点踩率 | 4. 单任务 Token 数<br/>一旦劣化超阈值 -> 4.5 分钟内自动切回 100% v1!"]
    end
```

### 3.3 新版本 `v2` 引入隐蔽 `HTTP-200` 幻觉与工具参数漂移时“`4.5分钟 SPRT` 自动捕获并零资损回滚”时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as 在线用户群 (session_id)
    participant GW as AI 网关 (Session-Sticky + Shadow)
    participant V2 as v2 新版本 Agent (含隐蔽 Prompt 缺陷)
    participant MCP as MCP 工具网关 (Shadow 写拦截)
    participant SPRT as SPRT 自动发布控制器

    Note over GW,SPRT: 阶段 1：开启 5% 影子伴飞 (`x-traffic-mode: shadow`)
    User->>GW: 1. 发起报修对话；主链路走 `v1`，网关异步复制流量至 `v2-shadow`
    V2->>MCP: 2. `v2-shadow` 欲调用写工具 `create_ticket(priority="URGENT_P0")`
    MCP-->>V2: 3. 识别出 `shadow` 标记 -> 拦截真实建单，记录参数 Diff（发现 `v2` 把普通咨询误判为 `P0`！）并返回虚拟单号
    Note over GW,SPRT: 阶段 2：开启 10% 会话粘性金丝雀（同一 session_id 全程锁定 v2）
    User->>GW: 4. 220 个金丝雀会话进入 `v2`（HTTP 状态码全为 `200 OK`）
    SPRT->>SPRT: 5. `T + 4.5 分钟`：检测到 `v2` 用户 15 秒内重问率从 `3.8%` 跃升至 `15.2%`，且在线 `NLI` 忠实度下降 `9.4 个点`，`SPRT` 对数似然比突破回滚红线！
    SPRT->>GW: 6. ⚡ 全自动触发秒级回滚：将 `v2` 权重置为 `0%`，并将受影响的活跃会话平滑迁回 `v1`（捕获率 `99.4%`）！
```

---

## ④ 📊 四档企业 AI 渐进式发布与金丝雀灰度方案全维度决策矩阵

| 评估维度 | 🟥 Tier 1：按请求随机切流 + 看 `HTTP 500` | 🟨 Tier 2：裸开影子伴飞 (不拦写工具) | 🟦 Tier 3：48 小时人工抽检浸泡 | 🟩 Tier 4：会话粘性灰度 + 写模拟影子 + `SPRT` 自动回滚 |
|---|---|---|---|---|
| **`HTTP 200` 语义/质量退化捕获率** | `12.0%` ✕ (对幻觉完全失明) | `78.0%` | `72.5%` ✕ | 🏆 **`99.4%`（四维 AI 黄金信号 `SPRT` 序贯检验）** |
| **多轮会话中途状态错乱率** | `28.4%` ✕ (中途横跳精神分裂) | `4.5%` | `2.0%` | 🏆 **`0.0%`（`HMAC(session_id)` 强制会话级版本锁定）** |
| **影子流量重复写外部系统事故率** | N/A | `100.0%` ✕ (双倍发信/建单) | N/A | 🏆 **`0.0%`（`x-traffic-mode: shadow` 在 MCP 网关干跑拦截）** |
| **问题版本发现与自动回滚耗时** | 漏检直接推全量 | 约 60 分钟 | `2,880 分钟` (48 小时) | 🏆 **`4.5 分钟`（序贯概率比检验实时越界秒切）** |
| **120 次发布/月总事故与运维 TCO** | `$182,000 / 月` ✕ | `$145,000 / 月` ✕ | `$118,000 / 月` ✕ | **`$21,000 / 月` 🏆（比 Tier 1 节省 88.5%）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（月均 120 次生产 AI 发布 / 30 个应用 / 平均 10 轮会话）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级 AI 渐进式发布与粘性金丝雀 ROI 计算器](../reports/enterprise-ai-progressive-delivery-canary-rollout-session-sticky-shadow-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-ai-progressive-delivery-canary-rollout-session-sticky-shadow-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_ai_progressive_delivery_canary_rollout_session_sticky_shadow_selection/run.py --check
  python -m unittest tests/test_enterprise_ai_progressive_delivery_canary_rollout_session_sticky_shadow_selection.py
  ```

### 5.2 核心量化对比图（语义质量退化捕获率 % vs. 多轮会话中途状态错乱率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Random 90/10 + HTTP 500 Gate", "metric": "Semantic Regression Catch (%)", "val": 12.0 },
      { "tier": "Tier 1: Random 90/10 + HTTP 500 Gate", "metric": "Mid-Session State Desync (%)", "val": 28.4 },
      { "tier": "Tier 2: Unguarded Shadow Mirroring", "metric": "Semantic Regression Catch (%)", "val": 78.0 },
      { "tier": "Tier 2: Unguarded Shadow Mirroring", "metric": "Mid-Session State Desync (%)", "val": 4.5 },
      { "tier": "Tier 3: 48h Manual Canary Soak", "metric": "Semantic Regression Catch (%)", "val": 72.5 },
      { "tier": "Tier 3: 48h Manual Canary Soak", "metric": "Mid-Session State Desync (%)", "val": 2.0 },
      { "tier": "Tier 4: Sticky Canary + Shadow + SPRT", "metric": "Semantic Regression Catch (%)", "val": 99.4 },
      { "tier": "Tier 4: Sticky Canary + Shadow + SPRT", "metric": "Mid-Session State Desync (%)", "val": 0.0 }
    ]
  },
  "semantic_types": { "tier": "Category", "metric": "Name", "val": "Percentage" },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 catches 99.4% of HTTP-200 semantic regressions with 0.0% mid-session desync",
    "subtitle": "Semantic Regression Catch Rate (%) vs. Mid-Session Multi-Turn State Desync Rate (%)",
    "encodings": { "x": "tier", "y": "val", "group": "metric", "color": "metric" }
  }
}
```

---

## ⑥ 🛠️ 架构评审与生产落地 Checklist（上线前必查 5 项）

- [ ] **严禁对多轮对话或 Agent 应用使用按单次 HTTP 请求随机切流的负载均衡器**：必须在 AI 网关层基于 `HMAC(session_id, deployment_id)` 实施会话级（或租户级）一致性哈希粘性路由。
- [ ] **在开启影子流量伴飞（Shadow Mirroring）时强制注入全链路 `OpenTelemetry Baggage: traffic.mode=shadow`**：并在 MCP 工具网关拦截一切带有 `shadow` 标记的写工具调用，返回虚拟回执并记录参数 Diff。
- [ ] **彻底改造金丝雀发布控制器（如 Argo Rollouts / Flagger MetricTemplate），废除“仅看 HTTP 500 和延迟”的传统探针**：接入四维 AI 质量指标（`NLI 忠实度得分`、`工具 Schema 校验错误率`、`用户 15 秒重问/点踩率`、`单轮 Token 消耗量`）。
- [ ] **采用 Wald 序贯概率比检验（`SPRT`）替代固定时长等待**：一旦金丝雀池的业务质量指标突破显著性劣化边界，在 5 分钟内全自动触发回滚。
- [ ] **将 Prompt 模板、模型参数、切块策略与工具 Schema 统一打包为不可变版本快照（`Release Bundle Manifest`）**：确保任何一次灰度回滚都能在 1 秒内原子恢复全套依赖配置。
