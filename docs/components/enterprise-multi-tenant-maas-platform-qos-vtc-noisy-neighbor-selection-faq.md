# 多团队共享企业级模型服务中台（`Enterprise Multi-Tenant MaaS`）架构选型：三级 QoS 优先级隔离、虚拟 Token 公平排队（`VTC`）、弹性配额借用与防“吵闹邻居（`Noisy Neighbor`）”治理指南

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-multi-tenant-maas-platform-qos-vtc-noisy-neighbor-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect 与企业 AI 平台负责人的多部门共享 MaaS 中台架构、为什么传统先来先服务（`FIFO`）队列与部门独占物理孤岛必翻车、虚拟 Token 计数器公平调度（`Virtual Token Counter, VTC`）数学模型与三级 QoS 显存水位预留白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业内部多租户 MaaS 平台治理架构设计）

![企业级多租户 MaaS 中台三级 QoS 与 VTC 公平排队选型总结图](../../assets/svg/enterprise-multi-tenant-maas-platform-qos-vtc-noisy-neighbor-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**建设服务全公司 60 个业务部门的企业级模型服务中台（MaaS），最怕陷入两个极端：① 如果按部门拆成物理独立 GPU 孤岛，全集群 24 小时平均利用率只有 **`29.5%`**（月度成本高达 `$420K`）；② 如果把所有部门混进一个默认的先来先服务（`FIFO / FCFS`）共享 `vLLM` 池，只要有 1 个数据科学团队写个脚本并发打入 100 个 `32K` 长文本请求，就会瞬间塞满全部 KV Cache 槽位，让高管与外部客户在线对话（`Gold` 级）的 `P99 TTFT` 暴涨至 **`14.2 秒`（`68.4%` SLA 违约）**！最优解是构建“**三级 QoS 显存水位保护（预留顶部 `20%` KV 显存仅对 `Gold` 开放）+ 虚拟 Token 计数器公平排队（`VTC`，按加权消耗 $\Delta V_u = [N_{\text{in}} + 5 N_{\text{out}}]/w_u$ 选欠账最少的租户优先调度）+ 保工作弹性配额借用（Work-Conserving Borrowing）**”，在保持 **`88.6%` 极高共享利用率** 的同时，确保 `Gold` 业务 `P99 TTFT` 稳在 **`430ms`（零邻居干扰）**，并较物理孤岛方案节省 `67.1%` 成本！**

- **交互式多租户 MaaS `VTC` 公平排队与 QoS 容量计算器**：[打开交互式计算器 (`enterprise-multi-tenant-maas-platform-qos-vtc-noisy-neighbor-sizer.html`)](../reports/enterprise-multi-tenant-maas-platform-qos-vtc-noisy-neighbor-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`enterprise-multi-tenant-maas-platform-qos-vtc-noisy-neighbor-selection-cards.html`)](../reports/enterprise-multi-tenant-maas-platform-qos-vtc-noisy-neighbor-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：什么是 `VTC（Virtual Token Counter，虚拟 Token 公平排队）`？它如何从数学上根治“1 个大户发长文饿死 59 个小租户”？
- **为什么按请求个数轮询（Round-Robin per Request）依然不公平？**：
  - 租户 A 发 1 个请求只有 `200` Input + `100` Output Tokens；租户 B 发 1 个请求有 `64,000` Input + `4,000` Output Tokens！按请求轮询，租户 B 实际上霸占了 `150 倍` 的 GPU 时间！
- **`VTC`（基于加权 Token 的赤字公平排队算法）**：
  1. 为每个活跃租户 $u$（出资购买份额权重为 $w_u$）维护一个单调递增的 **虚拟 Token 账本计数器 $V_u$**：
     $$V_u \leftarrow V_u + \frac{1.0 \times \Delta N_{\text{in}} + 5.0 \times \Delta N_{\text{out}}}{w_u}$$
  2. **防休眠攒额度攻击（Idle Lift）**：当某个原本空闲的租户 $u$ 重新发起请求时，立即将其计数器对齐至当前活跃租户的最小值：
     $$V_u \leftarrow \max\Big(V_u,\ \min_{k \in \text{Active}} V_k\Big)$$
     （防止某租户闲了一整天，$V_u=0$ 远小于其他租户，突然发 1 万个请求把所有人饿死！）；
  3. **每次调度选 $\arg\min_{u} V_u$**：引擎每空出一个 KV 槽位或调度下一个 Prefill Chunk 时，永远优先挑**当前 $V_u$ 最小（即用得最少）的租户**！一旦“吵闹邻居”狂发 `64K` 长文，它的 $V_u$ 会在第 1 个请求后瞬间飙升到队尾，其他 59 个发短消息的正常租户全部插队优先秒回（`430ms`）！

### Q2：如何通过“三级 QoS 显存水位门禁 + 弹性配额借用”兼顾 `0%` 违约与 `88.6%` 利用率？
- **显存双水位线硬门禁（KV Cache Watermark Reservation）**：
  - 当共享池 `gpu_cache_usage_perc > 65%` 时：暂停准入 `Bronze（低优评测/批处理）` 新请求，并将其在途请求挂起（Preempt/Swap）；
  - 当 `gpu_cache_usage_perc > 80%` 时：暂停准入 `Silver（内部普通办公）` 超额借用请求，**最后 `20%`（`80% ~ 100%`）的黄金显存空间永远只对 `Gold（外部客户实时交易/语音）` 开放**！

---

## 3. 🧭 直观可视化图解（Mermaid VTC 公平调度器 + 三级显存水位门禁）

### 3.1 多租户 `VTC` 公平排队与三级显存水位门禁（`65% / 80% / 100%`）架构图
```mermaid
flowchart LR
    T1["Gold 租户 (实时客服/语音)"] --> Q["VTC 虚拟 Token 公平调度器<br/>• V_u += (N_in + 5*N_out) / Share_u<br/>• 永远优先调度 argmin(Priority, V_u)!"]
    T2["Silver 租户 (内部员工助手)"] --> Q
    T3["Bronze 租户 (跑批/评测脚本)"] --> Q
    Q --> Gate{"核查共享池实时 KV 显存水位"}
    Gate -- "KV 水位 < 65% (空闲期)" --> All["✅ Gold + Silver + Bronze 全部放行<br/>允许跨部门免费借用空闲配额 (利用率 88.6%!)"]
    Gate -- "65% <= KV 水位 < 80%" --> Med["⚠️ 拦截/抢占 Bronze 跑批请求<br/>仅放行 Gold + Silver 保底配额"]
    Gate -- "KV 水位 >= 80% (高峰期)" --> High["🛡️ 顶部 20% 显存专属保护区:<br/>仅放行 Gold 核心实时流量 (0.0% 违约!)"]
```

---

## 4. 📊 架构师多方案选型决策矩阵（`60` 个租户共享 `64x H200` 四档方案对比）

| 评估维度 | Tier 1：共享 `FIFO` 队列（无公平排队与水位隔离） | Tier 2：按部门拆物理 GPU 孤岛（零共享） | Tier 3：共享池 + 静态死配额硬限流（不许借用空闲卡） | Tier 4（推荐）：`三级 QoS 显存门禁 + VTC 虚拟 Token 公平排队 + 弹性配额借用` |
| :--- | :--- | :--- | :--- | :--- |
| **邻居突发时 Gold `P99 TTFT`** | `14,200 ms` ✕（被排死 14 秒） | `480 ms` ✓ | `690 ms` | **`430 ms` ✓（提速 `33.0 倍`）** |
| **吵闹邻居导致 SLA 违约率** | `68.4%` ✕ | `0.0%` ✓ | `6.4%` | **`0.0%` ✓** |
| **集群整体 GPU 算力利用率** | `74.0%` | `29.5%` ✕（70.5% 闲置浪费） | `51.2%`（有空卡也不让借） | **`88.6%` ✓（兼顾隔离与高利用）** |
| **60 租户 MaaS 平台月度总 TCO** | `$310,000 / 月` | `$420,000 / 月` ✕ | `$245,000 / 月` | **`$138,000 / 月` ✓（较孤岛省 `67.1%`）** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **共享 MaaS 推理池严禁使用裸 `FIFO` 队列**：必须启用基于加权 Token 消耗的公平排队（`VTC / WFQ`）与 `Idle Lift` 防攒额度机制。
- [ ] **设置 `65% / 80%` 三级 KV Cache 显存准入水位线**：确保低优批处理永远无法吃光核心实时业务的最后 `20%` 显存救命池。
- [ ] **推行“承诺保底配额（Guaranteed）+ 空闲可抢占借用（Burstable）”双层配额模型**。
