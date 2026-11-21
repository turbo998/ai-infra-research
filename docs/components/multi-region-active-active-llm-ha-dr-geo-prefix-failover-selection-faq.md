# 多区域高可用容灾（`Multi-Region Active-Active HA & DR`）与跨区流量调度架构选型：会话地域亲和性（`Geo-Sticky Prefix Affinity`）、防切换冷缓存风暴、三级逃生舱与 `1.8s RTO` 指南

> **语言 / Language**: **中文 (当前)** | [English](./multi-region-active-active-llm-ha-dr-geo-prefix-failover-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect 与企业高可用灾备负责人的跨区域双活大模型容灾架构、为什么传统“冷备临时买 GPU（`OutOfStock`）”与“切流触发 5 倍 KV 冷缓存风暴（`KV Cold-Storm`）”必翻车、以及“双活地域粘性 + 可抢占批处理缓冲池 + 跨云 PAYGO 逃生舱”白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业多区域高可用容灾与跨区流量调度设计）

![多区域双活大模型高可用容灾与防冷缓存风暴选型总结图](../../assets/svg/multi-region-active-active-llm-ha-dr-geo-prefix-failover-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**把传统无状态微服务的容灾套路照搬到大模型集群上会踩中两个致命深坑：① 妄想在主可用区故障时靠“灾备区临时按需扩容 GPU（Pilot-Light）”救场——一旦某云大区宕机，全网几千家企业同时抢购备区 `H100/H200`，**`88.5%` 概率会直接报错 `OutOfStock`（库存不足）**；② 哪怕你在备区常年空放一整套热备 GPU（白花 `$460K/月`），当主区 `120 RPS` 的多轮 `16K` 长会话瞬间切到备区时，由于备区没有这些会话的 **Prefix KV Cache（命中率从 `80%` 瞬间归零）**，会瞬間引爆 **`5.0 倍` 的 Prefill 算力风暴**，当场把备区也连环冲垮（`34.2%` 级联雪崩）！生产唯一正解是采用“**双区域 Active-Active 就近会话粘性路由（`Geo-Sticky Session Affinity`）+ 跨区异步预热 Top-100 公共前缀 + 两区各留 30% 跑可秒级抢占（`vLLM Wake 4.2s`）的离线任务做弹性缓冲 + 跨云 Serverless PAYGO 第三级逃生舱**”，实现 **`1.8 秒 RTO`**、**`0.0%` 切换雪崩丢包** 与 **`99.99%` 年可用性**，同时较热备空转方案节省 `58.3%` 成本！**

- **交互式多区域双活容灾与冷缓存风暴计算器**：[打开交互式计算器 (`multi-region-active-active-llm-ha-dr-geo-prefix-failover-sizer.html`)](../reports/multi-region-active-active-llm-ha-dr-geo-prefix-failover-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`multi-region-active-active-llm-ha-dr-geo-prefix-failover-selection-cards.html`)](../reports/multi-region-active-active-llm-ha-dr-geo-prefix-failover-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：什么是跨区域故障切换时的“KV 冷缓存连环雪崩风暴（Failover KV Cold-Storm）”？怎么手算它的算力放大倍数？
- **算一笔致命的切换冲击账**：
  - 正常稳态运行时，主区（Region A）承载 `60 RPS` 多轮 Agent 流量（平均上下文 $S = 16,384$ Tokens），由于同一会话始终粘在主区，**Prefix Cache 命中率高达 $r_{\text{hit}} = 80\%$**——即每次请求平均只需算 $16,384 \times (1 - 0.80) = \mathbf{3,277\text{ 个未缓存 Token}}$ 的 Prefill；
  - 当 Region A 突然断网、全部流量瞬间切往 Region B 时，Region B 显存里完全没有这批在途会话的历史 KV Cache（$r_{\text{hit}} \to 0\%$），每条请求被迫从头重算整整 **`16,384` 个 Token**！
  - **Prefill 算力瞬间放大倍数公式**：
    $$\text{Prefill Surge Multiplier} = \frac{1}{1 - r_{\text{hit}}} = \frac{1}{1 - 0.80} = \mathbf{5.0\times}$$
    也就是说，切流后的前 3 分钟里，Region B 承受的 Prefill 算力冲击不是原来的 2 倍，而是惊人的 **`1 + 5 = 6 倍`**！如果毫无防备，备区会在 15 秒内排队打满并触发连环雪崩！
- **三招彻底化解冷缓存风暴**：
  1. **常态异步双写预热 Top-100 公共前缀（System Prompt / 热门 RAG 块）**：确保切流时至少保底 `45%+` 的静态前缀命中率；
  2. **故障期自动激活 3 分钟“优雅降级裁剪（Brownout History Trimming）”**：AI Gateway 对刚从故障区漂移过来的会话自动启用轻量摘要或仅截取最近 `6K` 轮次，削平瞬时二次方算力尖峰；
  3. **第三级跨云 `Serverless PAYGO` 逃生舱分流**：将超出备区 `P85` 水位的突发 Prefill 请求自动溢出到多云按量 API 吸收。

### Q2：既然备区不能空转浪费钱，临时买卡又买不到（`OutOfStock`），怎么做到“平时零闲置浪费、故障时 4 秒变出双倍 GPU”？
- **双活 `60% 在线 + 40% 可抢占离线/微调缓冲池`（Preemptible Batch Buffer）架构**：
  - 在东区（East）和西区（West）各部署 1 套对等的预留 GPU / PTU 集群；
  - 平时每区用 `60%` 算力接本区域的就近在线流量（通过 **`hash(session_id)` 地域粘性路由** 保证多轮对话绝不跨洋乱跳），另外 **`40%` 算力用于跑本企业的离线文档抽取（`#326`）或 LoRA 微调，且在同机通过 `vLLM Sleep-Mode (level=1)` 预驻留在线模型权重于 CPU 主存**；
  - 一旦东区故障，西区网关在 **`1.8 秒`** 内接管流量，同时 `Kueue` 瞬间暂停西区的离线批处理任务并触发 `vLLM /wake_up`，**`4.2 秒` 内即可原地唤醒 `40%` 的热备 GPU**，零闲置成本扛下全量双区流量！

---

## 3. 🧭 直观可视化图解（Mermaid 双活地域粘性与三级逃生舱架构）

### 3.1 跨区双活 `Geo-Sticky` 路由、秒级抢占唤醒与三级逃生舱全链路
```mermaid
flowchart TD
    User["全球用户流量 (120 Peak RPS)<br/>带 Session_ID 亲和性 Header"] --> GSLB["全球边缘 AI Gateway (Anycast / Envoy)<br/>• 正常态: 按地域 + Session_ID 粘性锁死归属区<br/>• 故障探测: P95 TTFT > 1.5s 或连续 3 次 5xx/429 (1.8s 触发!)"]
    GSLB -- "东区正常 (60 RPS)" --> R1["Region A (East 主活区)<br/>• 60% 在线 vLLM/PTU (KV 命中 85%)<br/>• 40% 可抢占离线 Batch + Sleep 备用"]
    GSLB -- "西区正常 (60 RPS) + 东区故障接管" --> R2["Region B (West 主活区)<br/>• 1.8s 接管流量 + Top-100 前缀常驻预热<br/>• 4.2s 抢占离线任务唤醒 Sleep 权重!"]
    R1 & R2 -- "极端双区洪峰 / 冷缓存尖峰溢出" --> R3["Tier-3 跨云 Serverless PAYGO 逃生舱<br/>(Azure Foundry / Bedrock / Vertex 按量溢出)<br/>• 保障全网 0.0% 丢包与 99.99% SLA!"]
```

---

## 4. 📊 架构师多方案选型决策矩阵（`120 Peak RPS` 四档跨区容灾方案对比）

| 评估维度 | Tier 1：单区运行 + 故障时临时按需买卡（`Pilot-Light`） | Tier 2：主备模式（备区 100% 空闲热备 GPU + DNS 切换） | Tier 3：无状态双活 50/50 轮询（打散多轮会话） | Tier 4（推荐）：`双活地域粘性 + 40% 可抢占离线缓冲 + 前缀预热 + PAYGO 逃生舱` |
| :--- | :--- | :--- | :--- | :--- |
| **大区级故障恢复时间 (`RTO`)** | `2,400 s`（40 分钟 ✕） | `75.0 s`（受限于 DNS TTL） | `8.5 s` | **`1.8 s` ✓（网关毫秒级熔断切流）** |
| **切流瞬间级联过载丢包率** | `88.5%` ✕（云端 `OutOfStock`） | `34.2%` ✕（5倍冷缓存风暴） | `19.4%` | **`0.0%` ✓（前缀预热 + Brownout + 溢出）** |
| **折算全年有效可用性 (`SLA`)** | `99.15%` ✕ | `99.82%` | `99.91%` | **`99.99%` ✓（全年中断 `< 52 分钟`）** |
| **全球容灾底座月度净 TCO** | `$395,000 / 月`（含故障违约金） | `$460,000 / 月` ✕（备区白白空转） | `$310,000 / 月` | **`$192,000 / 月` ✓（较热备节省 `58.3%`）** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **彻底抛弃“出事再去云上临时开 H100/H200 实例”的幻想**：备区算力必须平时就以“Active-Active 在线 + 可抢占离线批处理”形态真实持有。
- [ ] **跨区双活严禁做每请求随机 Round-Robin**：必须按 `(tenant_id, session_id)` 实施地域粘性（Geo-Sticky）路由，保护本地 Prefix KV Cache。
- [ ] **每月执行一次自动化断网逃生演练（Chaos Game-Day）**：验证备区离线任务在 `<5s` 内让出 GPU 且冷缓存保护策略正常触发。
