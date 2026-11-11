# 大模型推理集群全局 Prefix Cache 树（`RadixAttention` vs. `Block APC`）、KV 亲和路由与防污染淘汰（`SLRU`）架构选型指南（`无状态 Round-Robin 轮询` vs. `朴素粘性哈希路由` vs. `RadixAttention + 全局 KV 目录 + 纯 LRU` vs. `Dynamo 队列衰减有理路由 + RadixAttention + 双段 SLRU 租户隔离`）

> **阅读指南（3 分钟速通 / 10 分钟白板宣讲）**：本文面向 **AI 解决方案架构师（AI Solution Architect）与大模型推理平台负责人**。用第一性原理拆解多副本推理集群（如 16 台 8×H100 节点，共 128 张 GPU）在上线 Prefix Cache（前缀缓存）后最常见的三大诡异翻车——**为什么明明在每个 vLLM/SGLang 节点上都打开了 Prefix Cache，但前面挂了一层 Nginx/Envoy 无状态 `Round-Robin` 轮询负载均衡（Tier 1）后，全集群实际缓存命中率从单机的 `85%` 暴跌至 `18.4%`（`P99 TTFT` 恶化至 `1,420 ms`）？为什么改成按前缀哈希绑死节点的“朴素粘性路由（Tier 2）”后，一旦遇到全员共用的热门 System Prompt 或爆款 RAG 文档，就会把 1 台机器瞬间塞爆、其余 15 台机器闲死，`P99 TTFT` 反而飙升到 `2,760 ms`？为什么跑一个 1,000 条的一次性离线批量扫描任务会把线上多轮对话的缓存全部冲光（Cache Pollution）？以及如何通过“NVIDIA Dynamo 队列负载有理衰减路由（`Rational Overlap-Credit Decay`）+ `RadixAttention` 前缀树 + 双段式防冲刷 `SLRU`（试用区/保护区）”架构（Tier 4），将全集群有效命中率推至 `86.8%`、`P99 TTFT` 压低 `16.9 倍` 至 `163 ms`，并每年省下 `$148 万` GPU 算力？**

[![返回 README 导航](https://img.shields.io/badge/⬅️_返回_AI_Infra_架构选型主目录-0F172A?style=for-the-badge)](../../README.md)
[![English Version](https://img.shields.io/badge/🌐_English_Version-llm--inference--prefix--cache--radix--kv--routing--eviction--selection--faq.en.md-2563EB?style=for-the-badge)](./llm-inference-prefix-cache-radix-kv-routing-eviction-selection-faq.en.md)
[![7-Card 评测报告](https://img.shields.io/badge/📊_7--Card_视觉评测报告-HTML_直达-9333EA?style=for-the-badge)](../reports/llm-inference-prefix-cache-radix-kv-routing-eviction-selection-cards.html)
[![交互式选型工作台](https://img.shields.io/badge/🧮_交互式前缀缓存路由计算器-HTML_直达-059669?style=for-the-badge)](../reports/llm-inference-prefix-cache-radix-kv-routing-eviction-sizer.html)

![大模型推理集群全局 Prefix Cache 树、Dynamo 衰减路由与 SLRU 防污染淘汰一图流总结](../../assets/svg/llm-inference-prefix-cache-radix-kv-routing-eviction-selection-faq-summary.svg)

---

## ① 📌 核心结论速览（Executive Summary）

1. **为什么“单机开了 Prefix Cache + 网关用无状态 Round-Robin 轮询（Tier 1）”会让缓存形同虚设？**：
   - 在 **16 个推理副本（16 × 8-GPU H100 = 128 卡）** 组成的生产集群中，用户的一个多轮 Agent 会话（每轮携带 `24,000 Tokens` 的历史上下文与工具定义）第 1 轮被 `Round-Robin` 发到了副本 #1，第 2 轮被发到了副本 #7，第 3 轮被发到了副本 #12！
   - 结果是：**16 台机器各自重复计算同一段 24K 前缀，且每台机器的有限显存被 16 倍的乱序流量迅速冲刷淘汰**！全集群真实 KV Cache 命中率从单机压测时的 `85%+` 暴跌至 **`18.4%` ✕**，首字 P99 延迟（`P99 TTFT`）高达 **`1,420.0 ms` ✕**，相当于一半以上的 GPU 算力在无意义地重复算 Prefill！
2. **为什么“朴素粘性哈希路由（Tier 2：Sticky Hash）”比轮询死得更惨？以及何时用 `RadixAttention + 全局 KV 目录`（Tier 3）？**：
   - **Tier 2（按前 256 个 Token 固定哈希绑定节点，不看队列负载）的“爆款热点雪崩”**：在真实企业流量中，前缀分布遵循高度偏斜的 **齐夫定律（Zipfian $s=1.05$）**——例如全公司 5,000 名员工共用同一个“企业财报知识库前导文档”或“Coding Agent 系统提示词”。固定哈希会把这 40% 的流量 **100% 死死钉在副本 #3 一台机器上**！导致副本 #3 排队积压 80 个请求（**`P99 TTFT` 飙升至 `2,760.0 ms` ✕**），而旁边另外 15 台机器空空荡荡！
   - **单一交互业务集群选 Tier 3（`SGLang RadixAttention 前缀树` + `全局 KV 目录订阅路由` + `纯 LRU`）**：网关实时维护 16 台机器的 Radix 前缀树快照，将命中率提升至 **`79.2%`**、P99 TTFT 降至 **`385.0 ms` ✅**；但因为底层用单一 `LRU` 淘汰队列，一旦某个部门发起 1,000 条一次性离线批量文档抽取（One-Hit Scan），就会把常驻的高频多轮前缀全部挤出显存（**缓存命中率瞬间掉 `-28.5%` ⚠️**）！
3. **多副本、多租户推理集群终极黄金标准（Tier 4：`NVIDIA Dynamo 有理负载衰减路由` + `RadixAttention` + `双段式 SLRU 防扫描淘汰`）**：
   - **第一性原理双剑合璧**：
     1. **网关层——有理队列衰减打分（`Dynamo Rational Overlap-Credit Decay`）**：不再盲目追求 100% 路由到缓存最多的节点，而是用公式动态权衡“省下的 Prefill 时间”与“目标节点的排队等待时间”：
        $$\text{Logit}(w) = \frac{\text{OverlapBlocks}(w, q)}{1 + d \cdot \frac{\text{ExcessQueueBlocks}(w)}{\text{RequestBlocks}(q)}} - \beta \cdot \text{DecodeLoad}(w)$$
        当副本 #3 队列开始积压时，衰减项自动降低副本 #3 的得分，将第 10 个并发请求优雅溢出（Spillover）到空闲的副本 #8 上预热第 2 份副本！**用 1 次 `180 ms` 的冷预热换取免排 `2,760 ms` 的长队**！
     2. **引擎显存层——双段式防扫描淘汰（`Segmented LRU [SLRU]`：20% 试用池 Probationary + 80% 保护池 Protected + 租户配额）**：所有新进入的 KV 块（包括一次性批量扫描任务）只能先进入占 20% 容量的 **试用池（Probationary LRU）**；只有被**第二次命中（Hit Count >= 2）**的块才有资格晋升入占 80% 容量的 **保护池（Protected Radix Tree）**！一次性批量大扫描只能在 20% 的试用池里自己洗自己，**绝对动不了保护池里的核心多轮前缀一根汗毛**！
   - **量化收益**：全集群有效 KV 命中率高达 **`86.8%` 🏆**，热点突发下 `P99 TTFT` 压低 **`16.93 倍`（`2,760 ms → 163 ms`，比轮询快 `8.71 倍` 🏆）**，批量扫描冲击下缓存掉点从 `42.0%` 压至 **`1.8%` 🏆**，集群达标吞吐（Goodput）飙升 **`3.42 倍`（每年净省 `$148 万` GPU 成本 🏆）**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（FAQ-Driven First Principles）

### Q1：在引擎内部，SGLang 的 `RadixAttention（基数前缀树）` 和 vLLM 的 `Block-Hash APC（固定块哈希表）` 到底有什么区别？各自什么时候更优？

- **通俗大白话 + 数据结构第一性原理对比**：
  1. **vLLM `Automatic Prefix Caching (APC)` —— 固定块大小的链式哈希表（Block-Granular Hash Table）**：
     - **底层运作**：把 Token 序列按固定块大小（如 `block_size = 16` 个 Token）切分。第 1 块的哈希值由 `[Token 1..16]` 算出；第 2 块的哈希值由 `[Hash(块1) + Token 17..32]` 级联算出（像区块链一样）。用一张扁平哈希表查找 `BlockHash -> Physical GPU Block ID`。
     - **优势**：$O(1)$ 查表极简，与 PagedAttention 块分配器天然合一。
     - **短板**：必须满 16 个 Token 才能对齐命中；对于频繁在任意词位置分叉的树状搜索（如 `Tree-of-Thoughts`、`Best-of-N` 投机采样、Few-Shot 动态拼接），管理公共祖先与子树淘汰不如树结构直观。
  2. **SGLang `RadixAttention` —— 边压缩基数前缀树（Edge-Compressed Radix Tree）**：
     - **底层运作**：在 CPU 内存中维护一棵真正的 **基数树（Radix Tree）**，树的每条边可以代表任意长度的连续 Token 序列，每个节点指向对应的 GPU Paged KV 页。
     - **核心绝活**：当多个请求在任意位置产生分叉（例如共享 System Prompt → 分叉出 3 个不同 RAG 文档 → 每个文档再分叉出 4 条思维链采样），Radix Tree 会自动在分叉点裂变节点（Split Node），并维护每个节点的引用计数（`ref_count`）。淘汰时严格从引用计数为 0 的**叶子节点（Leaf Nodes）向上回溯淘汰**，天然保证公共祖先前缀永远最后被淘汰！

### Q2：为什么说在多副本集群里，“只要有 KV 命中就无脑往那台机器发（Tier 2 贪心粘性路由）”会引发灾难性的排队雪崩？`NVIDIA Dynamo` 的有理衰减公式是怎么算清这笔账的？

- **第一性原理拆解（“省 150ms 计算，却排了 2500ms 队”的贪心陷阱）**：
  1. **贪心路由的盲区**：
     - 假设一段 24K Token 的热门公共前缀，重算一次 Prefill 需要 `180 ms`，如果命中缓存只需要 `15 ms`（省下 `165 ms`）。
     - 当早上 9 点早高峰突然涌入 30 个携带该热门前缀的并发请求时，贪心粘性路由器（Tier 2）发现只有 Worker #0 有这段缓存，于是把 30 个请求全塞给 Worker #0！
     - 结果第 30 个请求在 Worker #0 的队列里足足等了 **`2,760 ms` ✕**！而此时 Worker #1～#15 全部闲着发呆——如果把第 2 个请求直接发给空闲的 Worker #1，哪怕多花 `180 ms` 现算一次，在 `180 ms` 时不仅早就返回了，还自动让集群多出了第 2 个拥有该热门缓存的副本！
  2. **`Dynamo KvRouter` 有理衰减打分公式（Rational Overlap-Credit Decay）的精妙平衡**：
     - 在本仓库验证的 `ai-dynamo/dynamo` 源码（`selector/reference.rs`）中，路由器对每个候选节点 $w$ 计算经过队列长度衰减后的有效重叠分：
       $$\text{EffectiveOverlap}(w, q) = \frac{\text{MatchedBlocks}(w, q)}{1.0 + d \cdot \left(\frac{\text{ExcessActivePrefillBlocks}(w)}{\text{RequestBlocks}(q)}\right)}$$
     - **物理直观**：当节点 $w$ 空闲（$\text{ExcessActivePrefillBlocks} = 0$）时，分母为 $1$，缓存命中获得 **100% 满分奖励**；一旦节点 $w$ 开始排队积压，分母线性增大，缓存奖励迅速衰减！此时空闲的节点 $w'$ 得分反超，热门前缀自动按热度扩散复制到 2～4 台机器上（自适应多副本扩增），将 `P99 TTFT` 从 `2,760 ms` 暴砍 **`16.93 倍` 至 `163 ms` 🏆**！

### Q3：为什么线上推理集群一旦混跑离线批量任务，普通 `LRU` 缓存就会瞬间失忆？`SLRU（双段式 LRU）` 是怎么用 10 行代码防住“一步游扫描污染（One-Hit Wonders）”的？

- **借鉴操作系统内核（Linux Page Cache）与数据库缓冲池（MySQL InnoDB Buffer Pool）的第一性原理**：
  - **纯 `LRU`（Tier 3）的致命弱点**：普通 LRU 只要谁最新进来就把谁放在队头，把最老的踢出队尾。当一个后台批处理脚本连续发来 1,000 篇各自不同、以后再也不会被访问第二次的长文档（One-Hit Wonders）时，这 1,000 篇只用一次的文档会像推土机一样把整个 LRU 队列从头洗到尾，把真正高频复用的多轮对话和 System Prompt 全部踢出显存（命中率暴跌 `-28.5%`）！
  - **Tier 4 的 `SLRU（Segmented LRU）+ 租户配额` 防护墙**：
    - 将 GPU 与 Host DRAM 的 KV 缓存池划分为两段：
      1. **`20%` 试用段（Probationary Segment）**：所有首次出现的全新前缀块（`hit_count == 1`）**一律只能进入 20% 的试用段**！如果它只是批处理扫过一次的“一步游”垃圾，它会在 20% 试用段内部很快被下一个批处理请求顶掉，全程被关在 20% 的隔离笼子里！
      2. **`80%` 保护段（Protected Segment）**：只有当某个前缀块在试用段存活期间**被第 2 次命中访问（`hit_count >= 2`）**，证明它是真正的多轮会话或共享知识库前缀，才会立刻晋升到 **80% 的保护段 Radix 树** 中！
    - 结果：即使后台同时狂扫 10,000 条一次性批处理文档，线上多轮交互用户的缓存命中率跌落也 **`< 1.8%` 🏆**！

---

## ③ 🧭 直观可视化图解（Mermaid 决策树 / 路由与淘汰对比 / Dynamo 溢出时序）

### 3.1 大模型推理集群 Prefix Cache 与 KV 亲和路由架构选型决策树

```mermaid
flowchart TD
    Start["🧭 推理集群 Prefix Cache 与 KV 路由选型<br/>输入：推理副本数 (N)、流量偏斜度与是否混跑批量任务"] --> Q1{"集群是否只有单台机器 (N = 1)<br/>且无多副本负载均衡？"}
    Q1 -- "是（单实例部署）" --> T0["⚪ 单机直接开启 SGLang RadixAttention 或 vLLM APC<br/>无需跨节点 KV 路由器，但仍建议开启 SLRU 防批量冲刷"]
    Q1 -- "否（生产多副本集群 N = 4 ~ 128 台）" --> Q2{"网关层是否仍在使用无状态<br/>Round-Robin / Least-Conn 轮询？"}
    Q2 -- "是（传统微服务网关直接照搬）" --> T1["🔴 Tier 1：无状态 Round-Robin + 节点独立 APC（立即整改！）<br/>⛔ 同会话被拆散到 16 台机器，命中率跌至 18.4%，P99 1,420ms"]
    Q2 -- "否（开启前缀感知路由）" --> Q3{"路由器是否根据目标节点排队深度<br/>动态衰减前缀匹配分 (Load-Decayed Scoring)？"}
    Q3 -- "否（死板一致性哈希/贪心绑定唯一节点）" --> T2["🟡 Tier 2：朴素粘性哈希路由 (无负载衰减)<br/>⚠️ 热门公共前缀将流量钉死在 1 台机器上，排队雪崩 P99 飙至 2,760ms！"]
    Q3 -- "是（Dynamo 有理衰减路由，自动溢出复制热前缀）" --> Q4{"集群是否混跑多租户流量或一次性批量扫描任务？"}
    Q4 -- "否（纯单一交互流量）" --> T3["🔵 Tier 3：RadixAttention + 全局 KV 目录 + 纯 LRU<br/>✅ 命中率 79.2%，P99 385ms（但遇批量扫描会掉点 28.5%）"]
    Q4 -- "是（企业多租户混部 / Agent + RAG + Batch）" --> T4["🟢 Tier 4：Dynamo 有理负载衰减路由 + RadixAttention<br/>+ 双段式 SLRU (20%试用/80%保护) + 租户配额<br/>🏆 命中率 86.8% | P99 TTFT 163ms (快16.9x) | 年省 $148万"]
```

### 3.2 传统轮询稀释 / 粘性热点雪崩 vs. Tier 4 Dynamo 衰减路由 + SLRU 防污染对比

```mermaid
flowchart LR
    subgraph Bad["❌ Tier 1/2：传统网关困局（要么轮询把命中率稀释到 18%，要么粘性把单机挤爆到 2.7s）"]
        direction TB
        B1["Tier 1：无状态 Round-Robin 轮询<br/>💥 多轮对话分散到 16 台机器重复算 Prefill<br/>💥 命中率仅 18.4%，P99 TTFT 1,420ms"] --> B3["集群有效吞吐仅 64 ~ 88 req/s<br/>年烧 $318万 ~ $365万 GPU 成本"]
        B2["Tier 2：死板粘性哈希 + 单池纯 LRU<br/>💥 爆款提示词把 40% 流量钉死在 Worker #3 (P99 2,760ms)<br/>💥 1次批量扫描把全池热缓存冲刷殆尽 (-35% 掉点)"] --> B3
    end

    subgraph Good["✅ Tier 4：Dynamo 有理负载衰减路由 + RadixAttention + 双段 SLRU 隔离"]
        direction TB
        G1["① 网关层：Dynamo 有理队列衰减打分<br/>⚡ Worker #3 排队微升时自动溢出至空闲 Worker #8 预热第 2 副本"] --> G2["② 树管理层：SGLang RadixAttention 基数树<br/>🌳 支持任意词粒度多分支共享，从引用为0的叶子节点向上淘汰"]
        G2 --> G3["③ 防污染层：双段 SLRU (20%试用池 / 80%保护池)<br/>🛡️ 一次性批量请求困在 20% 试用池自循环，80% 核心热前缀纹丝不动<br/>🏆 命中率 86.8% | P99 TTFT 163ms (-16.9x) | 年省 $148万"]
    end
```

### 3.3 突发热门前缀与批量扫描混部下 Tier 4 路由溢出与 SLRU 晋升时序图

```mermaid
sequenceDiagram
    autonumber
    participant C as 多租户请求流 (交互 Agent + 批量扫描)
    participant R as Dynamo KV 路由器 (有理负载衰减)
    participant W1 as Worker #1 (已缓存 24K 热门前缀)
    participant W2 as Worker #2 (空闲节点 · SLRU 池)

    Note over C,W1: 场景 1：热门共享前缀命中与过载自适应溢出 (Spillover Replication)
    C->>R: 并发涌入 20 个携带 24K 热门财报前缀的请求 (q1..q20)
    R->>W1: q1..q6 路由至 Worker #1 (队列空，100% 命中缓存，TTFT = 28ms)
    R->>R: 检测到 Worker #1 待处理 Prefill 队列上升，Overlap 得分按 1/(1+d*Queue) 衰减！
    R->>W2: 自动将 q7..q12 溢出路由至空闲 Worker #2 (180ms 预热第 2 副本后，后续请求全部 28ms 命中！)

    Note over C,W2: 场景 2：一次性批量扫描防冲刷保护 (SLRU Scan Resistance)
    C->>W2: 离线批处理注入 500 条从不重复的文档抽取请求
    W2->>W2: 新前缀仅写入 20% Probationary 试用池 (hit_count=1)，在试用池内循环淘汰
    W2-->>C: 80% Protected 保护池内的多轮 Agent 与热门财报前缀 0% 被踢出 (污染跌落仅 1.8%)！
```

---

## ④ 📊 架构师选型决策矩阵（Architect Decision Matrix）

| 评估维度 | Tier 1：无状态 `Round-Robin` + 本地 APC | Tier 2：朴素粘性哈希路由（无负载衰减） | Tier 3：`RadixAttention` + 全局目录 + 纯 LRU | Tier 4：`Dynamo 衰减路由` + `Radix` + `SLRU 隔离` |
| :--- | :--- | :--- | :--- | :--- |
| **核心路由与淘汰技术栈** | 随机/轮询分发 + 单机 16-tok 块 LRU | 固定前缀哈希绑死单节点 + 单池 LRU | **基数前缀树 + 全局 KV 目录订阅 + 纯 LRU** | **有理负载衰减打分 + Radix 树 + 20/80 双段 SLRU** |
| **16 副本集群有效 KV 命中率 (`Hit %`)** | **`18.4%` ✕（被稀释为单机的 1/16）** | `76.5%` ✅ | `79.2%` ✅ | **`86.8%` 🏆（自适应多副本热点扩散）** |
| **突发偏斜流量下首字尾延迟 (`P99 TTFT`)** | `1,420.0 ms` ✕（81.6% 请求全量重算） | **`2,760.0 ms` ✕（热点节点排队雪崩！）** | `385.0 ms` ✅ | **`163.0 ms` 🏆（比粘性哈希快 `16.93 倍`）** |
| **遭遇批量扫描时的缓存命中率掉点 (`Drop %`)** | `42.0%` ✕ | `35.0%` ✕ | `28.5%` ⚠️（纯 LRU 被一步游冲空） | **`1.8%` 🏆（20% 试用池完美隔离一步游垃圾）** |
| **16 节点集群 SLO 达标吞吐 (`Goodput`)** | `88.0 req/s` ✕ | `64.0 req/s` ✕（15 台闲置 1 台撑死） | `215.0 req/s` ✅ | **`301.0 req/s` 🏆（达轮询基线的 `3.42 倍`）** |
| **承载同等业务量的年化 GPU 成本 (`USD/Yr`)** | `$3,180,000 / 年` ✕ | `$3,650,000 / 年` ✕ | `$2,120,000 / 年` ✅ | **`$1,700,000 / 年` 🏆（比轮询净省 `$148 万/年`）** |
| **适用场景与架构师结论** | 仅限单副本测试；**多副本生产严禁使用** | 仅限完全无公共前缀的均匀 1v1 闲聊 | **无批处理干扰的纯单一交互服务集群** | **多副本、多租户推理集群（16～256卡）唯一黄金标准** |

---

## ⑤ 🧪 简明定量算例与 Benchmark 洞察（16 副本 128×H100 集群实测账单）

> **数据来源与透明度声明（Provenance Disclosure）**：本节与配套 7-Card 报告的所有量化数据均由本仓库归档的确定性仿真评估器 [`experiments/prefix_cache_radix_kv_routing_eviction_selection/run.py`](../../experiments/prefix_cache_radix_kv_routing_eviction_selection/run.py) 及密封输出 [`results.json`](../../experiments/prefix_cache_radix_kv_routing_eviction_selection/results.json) 生成（标记为 `independently_reproduced=false`、`physical_gpu_cluster_measurement=not_measured`，参数校准自本仓库源码审计验证的 `NVIDIA Dynamo KvRouter` (`selector/reference.rs`)、`SGLang RadixAttention`、`vLLM APC` 与操作系统 `SLRU` 防扫描淘汰模型）。

### 5.1 为什么说“换掉网关的 Round-Robin 和 Sticky Hash（升级 Tier 4），直接省下一半的 H100 机器”？

1. **打破“轮询没命中（Tier 1）”与“粘性排死队（Tier 2）”的死局**：
   - 在 16 台 8×H100 节点上，Tier 1（轮询）因为只有 `18.4%` 命中率，每秒只能跑 `88.0 req/s`；而 Tier 2（死板粘性哈希）虽然命中率有 `76.5%`，但因为爆款前缀全堵死在 1 台节点上，P99 TTFT 高达 `2,760 ms`，SLO 达标吞吐反而跌到 `64.0 req/s` ✕！
2. **Tier 4（`Dynamo 有理衰减路由 + SLRU`）的 `3.42x` 吞吐暴涨**：
   - 通过在打分公式分母引入队列积压衰减项 $1 + d \cdot (\text{ExcessQueue} / \text{ReqBlocks})$，热门前缀自动按实时流量热度在 2～3 台节点间弹性扩散，配合 `SLRU` 保护池锁死多轮热点，将达标吞吐推高至 **`301.0 req/s`（提升 `3.42 倍` 🏆）**，承载同等流量每年立省 **`$1,480,000` GPU 租金 🏆**！

### 5.2 📊 7-Card 4:5 视觉评测卡片集与交互式前缀缓存路由计算器

- **7-Card 4:5 视觉评测报告**：[`docs/reports/llm-inference-prefix-cache-radix-kv-routing-eviction-selection-cards.html`](../reports/llm-inference-prefix-cache-radix-kv-routing-eviction-selection-cards.html)
- **交互式全局 Prefix Cache 与 KV 路由选型工作台**：[`docs/reports/llm-inference-prefix-cache-radix-kv-routing-eviction-sizer.html`](../reports/llm-inference-prefix-cache-radix-kv-routing-eviction-sizer.html)

---

## ⑥ 🛠️ 附录：生产环境快速核验清单（Architect Checklist）

在上线多节点 vLLM / SGLang / TensorRT-LLM 推理集群的 Prefix Cache 与网关路由前，请逐项核验以下 6 条工程铁律：

- [ ] **1. 严禁在多副本推理集群前端使用无状态 `Round-Robin` 或纯 `Least-Connections` 负载均衡**：无状态分发会将缓存命中率稀释为 $1/N$（$N$ 为副本数），白白浪费 60%+ 的 Prefill 算力。
- [ ] **2. 严禁使用不看目标节点队列深度的“死板一致性哈希（Naive Sticky Hash）”**：必须采用 **`NVIDIA Dynamo / Mooncake` 式的队列负载有理衰减打分器（Rational Overlap-Credit Decay）**，当命中节点排队超过阈值时自动溢出至次优空闲节点复制热前缀。
- [ ] **3. 将共用前缀（System Prompt + 工具 Schema + 公共 RAG 文档）严格固定在 Prompt 最前端**：Prefix Cache 依赖从第 0 个 Token 开始的严格字节匹配；严禁在 System Prompt 开头插入动态时间戳（如 `Current time: 2026-11-12 03:20:01`），否则会导致后续 30,000 个 Token 的缓存 100% 失效！
- [ ] **4. 开启 `SLRU（双段式试用/保护 LRU）` 与多租户 KV 配额隔离**：将首次访问（`hit_count == 1`）的块限制在 `20%` 试用池内，仅允许二次命中（`hit_count >= 2`）的热点前缀进入 `80%` 保护池，彻底免疫离线批处理大扫描的缓存污染。
- [ ] **5. 针对多分支树状生成（Tree-of-Thoughts / Best-of-N / Multi-Turn Forks）优先选用 `RadixAttention` 基数树**：支持任意 Token 边界分叉共享与自底向上按叶子引用计数（`ref_count == 0`）精准回收。
- [ ] **6. 在短请求（如 `< 2,048 Tokens` 且本地前缀高命中）上启用 `IslBoundingPolicy` 条件旁路远程 PD 传输**：当 Decode 节点本地或二级缓存已命中大部分前缀时，直接在本地算完剩余短 Prefill，免去跨网 NIXL/RDMA 搬运开销。
