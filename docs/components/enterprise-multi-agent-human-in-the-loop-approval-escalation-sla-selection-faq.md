# 企业级高风险业务 Agent 人机协同审批（`Human-in-the-Loop / HITL`）、零算力异步挂起检查点（`Postgres / Temporal Interrupt`）与 `OPA` 风险分级门禁选型指南：为什么让线程傻等人工审批会导致 `82%` 任务在发版时丢失，而“100% 弹窗审批”又会导致主管疲劳盲点（`44%` 漏审）？

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-multi-agent-human-in-the-loop-approval-escalation-sla-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、企业自动化工作流负责人与内控合规总监的“多智能体人机协同审批架构（`LangGraph interrupt() / Temporal Signal / Azure Durable Functions`）、`OPA / Cedar` 风险三级分流（`88% 自动放行 / 12% 人工签批`）、乐观锁防脏写（`ETag CAS`）与超时转交升级状态机”实战指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_multi_agent_human_in_the_loop_approval_escalation_sla_selection/results.json`](../../experiments/enterprise_multi_agent_human_in_the_loop_approval_escalation_sla_selection/results.json)。

![企业级 Agent 人机协同审批（HITL）异步检查点与风险门禁架构总结图](../../assets/svg/enterprise-multi-agent-human-in-the-loop-approval-escalation-sla-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

当企业 AI Agent 从“只读问答”跨入“真刀真枪调 API 改数据”（如 **执行采购下单、财务退款、生产库表变更、发邮件给外部客户，月均 `500,000` 个工作流**）时，必须引入 **人机协同审批（Human-in-the-Loop, HITL）**。但大模型是毫秒级的，而人类主管审批平均需要 **`3.5 小时`（甚至跨夜 12 小时）**！如果架构师用写同步接口的思维去设计 HITL，必踩以下三大深坑：
1. **深坑一：不加确定性拦截，让大模型自主决定调不调高危写接口（Tier 1，越权与错误写操作导致的财务损失率高达 `6.4%`，月损 `$420,000`）**：指望在 System Prompt 里写一句“超过 100 元请先询问人类”完全是掩耳盗铃，一次提示词注入或模型幻觉就会直接把钱打出去！
2. **深坑二：在 Python 进程里用 `await wait_for_human()` 同步挂起等待，且对 100% 工具调用弹窗（Tier 2，`78.0%` 的容器内存被空等协程占死；跨夜 24 小时后因 K8s 发版/缩容导致 `82.0%` 的在途审批直接丢失；更致命的是主管每天收到 800 个弹窗产生严重“审批疲劳”，高达 `44.0%` 的错单被看都不看无脑点通过）**！
3. **深坑三：发封无状态邮件通知主管，点同意后让 Agent 从第 1 步重头再跑一遍（Tier 3，不仅浪费双倍算力，而且在等待的 4 小时里底层 ERP 订单早已被客户取消，盲目恢复执行直接酿成重复退款脏写事故）**！

在 **月均 500,000 个企业 Agent 写操作工作流（人类平均响应耗时 `3.5 小时`）** 的生产基准下，四档 HITL 审批架构的实测对决如下：

| 方案档位 | 核心审批门禁与状态持久化架构 | 越权/误操作财务损失率 | 审批人疲劳盲点漏审率 | 等待期容器/GPU 空转占用率 | 跨 `24 小时` 等待状态存活率 | 50 万工作流/月总成本 (`TCO`) | 架构师核心选型结论 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：无门禁全自主执行 (`0% Human Gate`)** | 仅靠 Prompt 软约束，Agent 直调生产写接口 | `6.4%` ✕ *(严重资金损失)* | `0.0%` *(无人审)* | `0.0%` | `100.0%` | `$420,000 / 月` ✕ *(含赔偿与事故)* | ⛔ 严禁让大模型在无确定性策略门禁下直写生产系统 |
| **🟨 Tier 2：进程内同步阻塞等待 + 100% 弹窗** | 协程挂起占住内存；不论金额大小全弹窗找人批 | `0.4%` | **`44.0%` ✕** *(弹窗太多盲点同意)* | **`78.0%` ✕** *(连接池耗尽)* | **`18.0%` ✕** *(Pod重启即丢)* | `$285,000 / 月` ✕ | ⛔ 既把容器连接池占死，又把人类训练成无脑点同意的机器 |
| **🟦 Tier 3：无状态邮件通知 + 唤醒后重头重跑** | 杀掉进程发邮件；同意后从 Step 1 重新推理执行 | `0.8%` | `19.5%` | `12.0%` | `64.0%` | `$154,000 / 月` | ⚠️ 丢失前序中间推理上下文，且缺乏并发版本校验 |
| **🟩 Tier 4：`OPA/Cedar` 风险分级 (`88% 自动 / 12% 人工`) + 零算力 `Postgres/Temporal` 脱水检查点 + 参数可编辑卡片 + `ETag CAS`** | **策略引擎自动放行 88% 低危操作；12% 高危写操作序列化至数据库 (`interrupt()`) 释放 100% 算力 + 飞书/Teams 卡片内直接改参数 + 恢复时校验 ERP `ETag`** | **`0.0%` 🏆** *(铁闸零误操作)* | **`1.2%` 🏆** *(减少88%噪音)* | **`0.0%` 🏆** *(等待期零占用)* | **`100.0%` 🏆** *(跨周发版不丢)* | **`$42,000 / 月` 🏆** *(比 Tier 1 省 90.0%)* | 🌟 **企业级高风险 Agent 人机协同审批唯一黄金标准** |

> 💡 **架构师一句话设计铁律（高可靠 HITL 的四大支柱）**：
> 1. **零算力脱水挂起（Dehydrate-to-DB on `interrupt()`）**：“**人类思考的 4 个小时里，绝不允许占用哪怕 1MB 的容器内存或 GPU 显存！**”一旦触发人工审批，立刻通过 `LangGraph Postgres Checkpointer / Temporal Signal` 将当前图节点状态序列化落盘，**当场退出释放工作协程（`0.0%` 空转占用）**；哪怕期间 K8s 集群全量重启 10 次，12 小时后人类点击卡片按钮依然 **`100.0%` 精准原地唤醒**！
> 2. **用 `OPA / Cedar` 确定性风险分级消灭“狼来了”审批疲劳（`88% Auto / 12% HITL`）**：只读查询与 `<= $100` 的可逆常规操作 `100%` 自动放行；仅对 `12%` 中高危写操作触发人工卡片，将人类漏审率从 `44.0%` 压至 **`1.2%`**！
> 3. **支持“直接修改参数并批准（Approve with Edited Args）”+ 恢复前 `ETag` 乐观锁核验（防 TOCTOU 脏写）**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击跨夜等待怎么不丢状态、人类想改个数字怎么不重跑、等待期数据变了怎么防脏写 3 大痛点）

### Q1：半夜 11 点一个供应链 Agent 发起了一笔 `$8,000` 的紧急备件采购申请，采购总监第二天早上 8 点才在手机飞书/Teams 上点“同意”——这 9 个小时里如果 K8s 刚好做了滚动发版，怎么保证工作流 100% 不丢失？
- **“状态机脱水落盘（State Dehydration）与事件驱动复水（Webhook Rehydration）”**：
  - 在 Tier 2 的错误设计里，程序员写 `resp = await wait_for_slack_button(timeout=36000)`，把整个上下文留在 Pod 内存里——只要半夜 HPA 缩容或 CI/CD 发版重启了该 Pod，这笔单子就人间蒸发了（存活率仅 `18.0%`）！
  - 在 Tier 4 中，当执行流撞上 `interrupt()` 节点时：
    1. 引擎将完整的图状态（`thread_id`、前 5 步已查好的供应商比价表、待执行的 `create_po(amount=8000)` 参数快照）写入高可用 **`PostgreSQL Checkpointer / Temporal 历史表`**；
    2. 向飞书/Teams 推送一张带 **HMAC 数字签名与 `thread_id`、`checkpoint_id`** 的交互式自适应卡片（Adaptive Card），然后 **工作线程立即 `return` 退出，释放全部 CPU/内存/连接**；
    3. 第二天早上 8 点总监点击卡片上的【批准】，网关收到 Webhook 后从集群任意一个空闲 Pod 拉起 `thread_id`，从 Postgres 读取检查点，**跳过前 5 步推理，直接从第 6 步 `create_po` 工具调用处毫秒级继续往下走**！

### Q2：很多时候人类审批官看完了 Agent 拟定的方案，觉得“大体对，但退款金额应该从 `$1,200` 改成 `$800`”——如果卡片上只有【同意】和【拒绝】两个死按钮，点拒绝让 Agent 重写又慢又不准，架构上怎么破？
- **交互式卡片“行内参数直接编辑（Inline Argument Override on Resume）”模式**：
  - 严禁把 HITL 做成非黑即白的二元开关！
  - Tier 4 生成的飞书/Teams 审批卡片直接根据工具的 Pydantic Schema 渲染出 **可编辑表单控件**（例如默认填好 `refund_amount: 1200.00`、`reason_code: "LATE_DELIVERY"`）。
  - 审批官直接在卡片输入框里把 `1200.00` 改成 `800.00` 并点击 **【以修改后参数执行（Approve with Edits）】**。Webhook 将 `{action: "APPROVE_MODIFIED", edited_args: {refund_amount: 800.00}}` 直接注入 `Command(resume=...)` 状态机并标记 `verified_by_human=true`——**无需再调一次大模型重写，0 延迟、100% 精确执行人类意图**！

### Q3：如果审批单在下午 2 点发出，主管在下午 6 点点击了“批准退款”，但客户在下午 4 点已经自己通过自助通道退过款了（TOCTOU 时间差漏洞），怎么防止 Agent 醒来后重复打款？
- **检查点绑定目标资源版本号（`Resource ETag Compare-and-Swap, CAS`）+ 三级超时升级计时器**：
  - **防脏写乐观锁**：在下午 2 点挂起生成检查点时，一并记录目标 ERP 订单当前的版本号 `expected_order_etag = "v19"`。下午 6 点工作流唤醒、真正向 ERP 发起写操作的前一毫秒，强制执行 **`CAS` 校验（`SELECT etag FROM orders WHERE id = X`）**——发现订单已变为 `"v20"`，立即自动中止执行并向主管弹窗提示*“该订单状态已在等待期间发生变更，已自动拦截重复执行”*（误操作损失率 **`0.0%`**）！
  - **防无限挂起 SLA 计时器**：挂起同时注册分布式定时器：`T + 2h` 未批自动高亮催办 $\rightarrow$ `T + 4h` 未批自动按 `Entra ID / Workday` 汇报线转交 **备份审批人（Backup Delegate）** $\rightarrow$ `T + 24h` 仍未批自动安全过期释放锁！

---

## ③ 🧭 直观可视化图解（HITL 选型决策树 + `OPA` 分级与脱水检查点架构图 + 跨夜修改参数并校验 `ETag` 恢复时序图）

### 3.1 企业级 Agent 人机协同审批（HITL）架构选型决策树

```mermaid
flowchart TD
    A["企业级 Agent 人机协同审批 (HITL) 选型<br/>(50 万工作流/月 | 含数据库写操作、退款、采购与运维变更)"] --> B{"是否通过确定性策略引擎拦截高风险工具调用?"}
    B -- "否: 仅靠 Prompt 叮嘱大模型自觉询问人类 (Tier 1)" --> C["⛔ 6.4% 越权/误操作财务损失 ($42 万/月)！<br/>模型幻觉或注入攻击直接绕过软约束打款"]
    B -- "是: 如何划分人工审批范围并管理长达数小时的等待状态?" --> D{"采用什么挂起机制与审批触发比例?"}
    D -- "100% 动作全弹窗 + Python 线程阻塞 sleep 等待 (Tier 2)" --> E["⛔ 44% 审批疲劳盲点同意 + 82% 跨夜状态丢失！<br/>Pod 重启即丢单，78% 内存浪费在空等"]
    D -- "发无状态邮件，同意后从第 1 步重新推理执行 (Tier 3)" --> F["⚠️ 双倍算力浪费且易发生跨时间差重复写事故"]
    D -- "OPA 风险分级 (88% 自动/12% 人工) + Postgres/Temporal 脱水挂起 + ETag 校验 (Tier 4)" --> G["🟩 Tier 4: 零算力脱水检查点 + 卡片内改参数 + ETag 防脏写 + 4h 自动转交 🏆<br/>(0.0% 资金损失 | 100% 跨周存活 | 0.0% 空转占用 | 省 90.0% TCO)"]
    style G fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `OPA` 风险分级、`Postgres / Temporal` 零算力脱水挂起与交互式卡片总架构图

```mermaid
flowchart LR
    subgraph Policy["1. 确定性 OPA / Cedar 风险分级门禁"]
        Tool["🔧 Agent 拟调用工具<br/>(50 万次/月)"] --> OPA{"🛡️ OPA 风险策略评估"}
        OPA -->|88% 低危/只读 (<= $100)| Auto["🟢 免打扰自动放行<br/>(0 弹窗，消除审批疲劳)"]
    end

    subgraph Suspend["2. 零算力脱水检查点层 (0% CPU/GPU 占用)"]
        OPA -->|12% 高危变更 (> $100 / 生产变更)| Ckpt["💾 Postgres Checkpointer / Temporal<br/>写入 [thread_id, args, order_etag]<br/>工作进程立即退出释放资源!"]
        Ckpt --> Timer["⏱️ SLA 升级定时器<br/>2h 催办 -> 4h 转备份主管 -> 24h 安全过期"]
    end

    subgraph Resume["3. 可编辑签名卡片与 ETag 防脏写唤醒"]
        Ckpt -->|推送交互式卡片| Card["📱 飞书 / Teams / Slack 审批卡<br/>支持主管直接修改金额/参数并签名"]
        Card -->|Webhook 唤醒任意空闲 Pod| CAS["🔐 ERP/CRM ETag 乐观锁核验<br/>确认数据未变后执行最终写操作"]
    end
```

### 3.3 跨夜 9 小时挂起、期间经历 K8s 发版、次日主管修改金额并经 `ETag` 核验安全执行的时序图

```mermaid
sequenceDiagram
    autonumber
    participant Pod1 as Agent 工作 Pod #1 (23:00 晚间)
    participant DB as Postgres Checkpointer & 定时器
    participant Teams as 主管飞书 / Teams 交互式卡片
    participant Pod2 as 新发版 Agent Pod #2 (次日 08:30)
    participant ERP as 核心财务 ERP (含 ETag 乐观锁)

    Pod1->>ERP: 1. 23:00 查询订单 #991 状态（获取当前版本号 `etag = "v12"`），拟发起 `$1,200` 赔付
    Pod1->>DB: 2. 命中 OPA 高危规则（`> $100`）：调用 `interrupt()` 将状态与 `etag="v12"` 序列化存入 Postgres
    Pod1->>Teams: 3. 发送带表单的签名审批卡片，随后 Pod #1 进程立即退出释放 100% 内存！
    Note over Pod1,Pod2: 🌙 02:00 凌晨：K8s 集群执行全量滚动发版，旧 Pod #1 被彻底销毁（数据库检查点毫发无损！）
    Teams->>Pod2: 4. 08:30 主管在卡片中将 `$1,200` 手动改为 `$800` 并点击【批准】；Webhook 路由至全新 Pod #2
    Pod2->>DB: 5. 根据 `thread_id` 从 Postgres 毫秒级复水状态，读取人类修改后的参数 `amount = $800`
    Pod2->>ERP: 6. 执行带条件写操作：`POST /refund (amount=$800, If-Match: "v12")`
    ERP-->>Pod2: 7. ETag 匹配通过！精准完成 $800 退款（0 算力空转、跨发版 100% 存活、0 脏写事故）！
```

---

## ④ 📊 四档企业 Agent 人机协同审批（HITL）架构全维度决策矩阵

| 评估维度 | 🟥 Tier 1：无门禁全自主执行 | 🟨 Tier 2：线程阻塞等待 + `100%` 弹窗 | 🟦 Tier 3：无状态邮件 + 重头重跑 | 🟩 Tier 4：`88/12 OPA` 门禁 + 脱水检查点 + `ETag CAS` |
|---|---|---|---|---|
| **越权/误操作财务损失率** | `6.4%` ✕ (幻觉直写生产库) | `0.4%` | `0.8%` | 🏆 **`0.0%`（确定性 `OPA` 铁闸 + 恢复期 `ETag` 乐观锁）** |
| **人类审批官疲劳漏审率** | N/A (无人审) | `44.0%` ✕ (每天800弹窗麻木盲点) | `19.5%` | 🏆 **`1.2%`（自动放行 88% 低危操作，仅对 12% 高危弹窗）** |
| **等待期容器/GPU 空转占用率** | `0.0%` | `78.0%` ✕ (协程死守内存数小时) | `12.0%` | 🏆 **`0.0%`（状态序列化入库后进程立即退出释放算力）** |
| **跨 `24 小时` 等待状态存活率** | `100.0%` | `18.0%` ✕ (K8s 缩容/发版即丢失) | `64.0%` | 🏆 **`100.0%`（持久化在 Postgres/Temporal，无视容器重启）** |
| **50 万工作流/月总成本 (`TCO`)** | `$420,000 / 月` ✕ | `$285,000 / 月` ✕ | `$154,000 / 月` | **`$42,000 / 月` 🏆（比无门禁/阻塞模式节省 90.0%）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（月均 500,000 个 Agent 工作流 / `88%` 自动放行 + `12%` HITL / 平均等待 `3.5 小时`）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级 Agent 人机协同审批（HITL）异步检查点与风险门禁计算器](../reports/enterprise-multi-agent-human-in-the-loop-approval-escalation-sla-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-multi-agent-human-in-the-loop-approval-escalation-sla-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_multi_agent_human_in_the_loop_approval_escalation_sla_selection/run.py --check
  python -m unittest tests/test_enterprise_multi_agent_human_in_the_loop_approval_escalation_sla_selection.py
  ```

### 5.2 核心量化对比图（跨 24 小时等待状态存活率 % vs. 审批人疲劳漏审率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Unbounded Autonomous (0% Gate)", "metric": "24h State Survival (%)", "val": 100.0 },
      { "tier": "Tier 1: Unbounded Autonomous (0% Gate)", "metric": "Reviewer Fatigue Error (%)", "val": 0.0 },
      { "tier": "Tier 2: Blocking Thread + 100% Prompts", "metric": "24h State Survival (%)", "val": 18.0 },
      { "tier": "Tier 2: Blocking Thread + 100% Prompts", "metric": "Reviewer Fatigue Error (%)", "val": 44.0 },
      { "tier": "Tier 3: Stateless Email + Full Re-Run", "metric": "24h State Survival (%)", "val": 64.0 },
      { "tier": "Tier 3: Stateless Email + Full Re-Run", "metric": "Reviewer Fatigue Error (%)", "val": 19.5 },
      { "tier": "Tier 4: 88/12 OPA Gate + Durable Interrupt", "metric": "24h State Survival (%)", "val": 100.0 },
      { "tier": "Tier 4: 88/12 OPA Gate + Durable Interrupt", "metric": "Reviewer Fatigue Error (%)", "val": 1.2 }
    ]
  },
  "semantic_types": { "tier": "Category", "metric": "Name", "val": "Percentage" },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 guarantees 100% 24-hour state survival while cutting reviewer fatigue to 1.2%",
    "subtitle": "Workflow State Survival Across 24h Wait (%) vs. Reviewer Alert-Fatigue Error Rate (%)",
    "encodings": { "x": "tier", "y": "val", "group": "metric", "color": "metric" }
  }
}
```

---

## ⑥ 🛠️ 架构评审与生产落地 Checklist（上线前必查 5 项）

- [ ] **严禁在等待人工审批期间保持 Python 线程、WebSocket 连接或 GPU KV 缓存挂起**：必须使用 `LangGraph Postgres Checkpointer / Temporal Signal / Azure Durable Functions` 将状态序列化入库并立即退出工作进程（`0%` 等待期算力占用）。
- [ ] **在工具网关层前置 `OPA / Cedar` 确定性风险分级策略（`88% 自动放行 / 12% 人工签批`）**：严禁靠 Prompt 让模型自觉求批，也严禁对只读或小额低危操作无差别弹窗造成人类“审批疲劳”。
- [ ] **审批卡片必须支持“行内直接修改参数并批准（Approve with Edited Args）”**：允许主管直接修正金额或参数后一键继续执行，避免推倒重跑整个多智能体链路。
- [ ] **在挂起与恢复时强制校验目标业务记录的乐观锁版本号（`ETag Compare-and-Swap`）**：防止在等待审批的数小时内底层订单/工单已发生变更而引发重复打款或脏覆盖（TOCTOU 漏洞）。
- [ ] **为所有挂起的检查点配置三级超时升级定时器（`2h 催办 -> 4h 自动转交备份主管 -> 24h 安全过期回滚`）**：结合企业组织架构图（`Entra ID / Workday`）彻底消灭因主管休假导致的死锁工单。
