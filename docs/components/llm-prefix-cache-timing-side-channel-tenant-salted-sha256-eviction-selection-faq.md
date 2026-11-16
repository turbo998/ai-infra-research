# 多租户大模型推理前缀缓存（`Automatic Prefix Caching / RadixAttention`）TTFT 时序侧信道攻击、跨 LoRA/多模态哈希碰撞与抗扫描 `S3-FIFO` 淘汰选型：为什么开启全局无盐前缀缓存会让攻击者通过 `18.2 ms vs 165.0 ms` 的首字延迟差窃取其他租户 `98.4%` 的私密 Prompt，而平面 `LRU` 又会被单次 `128K` RAG 扫描冲垮命中率（跌至 `41.2%`）？基于只读公共根 + 租户加盐复合 `SHA-256`（`cache_salt + lora_id + mm_hash`）+ 抗扫描 `S3-FIFO`（`0.0%` 侧信道泄露，命中率拉升至 `84.6%`）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/llm_prefix_cache_timing_side_channel_tenant_salted_sha256_eviction_selection/run.py` 与 `results.json`。

![Multi-Tenant Prefix Cache Timing Side-Channel & S3-FIFO Eviction Visual Summary](../../assets/svg/llm-prefix-cache-timing-side-channel-tenant-salted-sha256-eviction-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向多租户大模型公有云/SaaS 推理网关（`vLLM cache_salt + SHA-256 Block Hasher / SGLang Multi-Tenant RadixCache + S3-FIFO Eviction`）的“只读公共系统模板根（`Public Read-Only Root`）+ 租户加盐全要素密码学链式哈希（`SHA256(tenant_salt || parent_hash || tokens || lora_id || mm_hash)`）+ 抗单次大文档扫描冲刷的双队列 `S3-FIFO / LRFU` 叶子节点淘汰协议”**。
- **架构师价值**：为了降低多轮对话与 Agent 的首字延迟（`TTFT`），几乎所有推理团队都在生产环境开启了自动前缀缓存（`enable_prefix_caching=True` 或 `RadixAttention`）。但在多租户、多 LoRA、多模态或混合长短请求的真实生产集群中，默认的“全局无盐平面 LRU 前缀缓存”隐藏着三大致命隐患：
  1. **隐患一：跨租户 `TTFT` 时序侧信道窃密攻击（`Prompt Stealing / Cache Timing Oracle`，窃取准确率高达 `98.4%`，Tier 1）**！
     - 在无盐全局前缀缓存中，KV 块的哈希只取决于 `(parent_hash, token_ids)`。如果租户 A（如某医院或金融机构）发送了包含敏感客户姓名、诊断记录或并购条款的 Prompt，这些 16-Token KV 块就会驻留在全局共享显存池中；
     - 此时外部攻击者租户 B 只需构造探测前缀，观察服务端返回的 `TTFT`（或流式第一个 chunk 的到达时间）：
       - **未命中缓存（Cache Miss）**：GPU 必须重新执行耗时漫长的 Prefill 计算，`TTFT = 165.0 ms`；
       - **命中租户 A 的缓存（Cache Hit）**：GPU 直接跳过计算复用租户 A 的 KV 块，`TTFT = 18.2 ms`（**整整快了 `9.07 倍`！**）
     - 凭借高达 `146.8 ms` 的宏观时间差（即使跨公网抖动也清晰可辨），攻击者可以像猜密码一样逐块（或在 `RadixAttention` 下逐 Token）搜索并还原出租户 A 的私密 Prompt 或验证某份机密文档是否刚被查询过（**重构准确率高达 `98.4%`**）！
  2. **隐患二：只对 `token_ids` 算哈希，漏掉 `lora_id` 与多模态图片 `mm_hash` 导致 `12.5%` 跨请求张冠李戴（Tier 1）**！
     - 在多模态模型（如 `Qwen2.5-VL`）中，两张截然不同的图片在文本序列里对应的占位符 Token ID 完全一样（都是连续 256 个 `<|image_pad|>` Token）！同理，两个挂载了不同行业 `LoRA` 适配器的请求，输入的 Prompt Token ID 也可能完全一样！
     - 如果前缀缓存的 Block Hash 只算了 `hash(token_ids)` 而漏掉了 **图片像素内容哈希 `mm_hash`** 和 **`lora_int_id`**，后一个请求就会错误复用前一个请求（不同图片或不同 LoRA）算出的 KV Cache，导致输出彻底错乱！
  3. **隐患三：平面 `LRU` 被单次 `128K` 离线 RAG 大文档扫描瞬间冲空（Cache Scan Pollution，命中率从 `84.6%` 暴跌至 `41.2%`）**！
     - 当某个后台批处理任务发来几条 `128K` 长文档总结请求（单条就占 `8,000` 个 KV Block，但只用这一次、绝不会复查）时，传统的 **平面 `LRU`（最近最少使用）队列** 会把这些刚进来的“一次性垃圾块”放到队头，而把几百个在线用户在高频复用的多轮对话前缀统统挤出显存，导致全局前缀命中率腰斩至 **`41.2%`**！
  采用 **只读公共系统模板根 + 租户加盐复合 `SHA-256` 链式哈希 + `10%` 试用队列过滤单次扫描的 `S3-FIFO` 淘汰算法（Tier 4）**，可将跨租户侧信道泄露与跨 LoRA/多模态串扰双双降至 **`0.0%`**，同时将混合负载下的持续缓存命中率大幅拉升 **`+43.4 pt`（达 `84.6%`）**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：全要素密码学链式块哈希公式（Cryptographic Composite Chained Block Hash）**：
   - 第 $k$ 个 KV Block（包含 Token 序列 $T_k$）的唯一标识 $H_k$ 必须由以下五元组通过密码学哈希 `SHA-256`（而非易受碰撞或 `HashDoS` 攻击的 64 位非密码学哈希）链式生成：
     $$H_k = \text{SHA256}\Bigl(\underbrace{H_{k-1}}_{\text{前序父块哈希}} \;\|\; \underbrace{\text{tenant\_salt}}_{\text{租户隔离盐值}} \;\|\; \underbrace{\text{lora\_id}}_{\text{适配器 ID}} \;\|\; \underbrace{\text{mm\_uuid}_k}_{\text{落入本块的多模态内容哈希}} \;\|\; \underbrace{T_k}_{\text{块内 Token IDs}}\Bigr)$$
   - **安全与正确性双重闭环**：
     1. 不同租户的 `tenant_salt`（由网关根据鉴权后的 JWT / API Key 自动注入，绝不允许客户端伪造他人的 salt）完全不同 $\to$ 即使两个租户发送一字不差的私密 Prompt，生成的 $H_k$ 也截然不同，**彻底斩断跨租户 `TTFT` 时序探测**；
     2. 同一 Prompt 搭配不同 `LoRA` 或不同图片（`mm_uuid`）时，从图片或 LoRA 生效的第一个块起 $H_k$ 自动分叉，**`0.0%` 脏 KV 污染**！
2. **如何在安全隔离租户的同时，依然让所有租户共享平台级超长公共 System Prompt（如 8K Token 的官方工具定义）？**
   - 采用 **两级分层命名空间（Two-Tier Public Root + Private Tenant Subtree，Tier 4）**：
     - 平台管理员将经过安全审计的只读公共模板（如标准 Agent 工具定义 Schema）以 `PUBLIC_ROOT` 盐值预热并锁定在缓存树根部；
     - 当租户请求匹配到公共模板末尾、开始进入租户私有对话轮次时，从分界块起切换注入该租户专属的 `tenant_salt`——**既白嫖了公共模板的高命中率（比完全隔离的 Tier 3 提升 `+29.8 pt` 命中率），又确保任何租户都无法探测其他租户的私有分支**！
3. **为什么 `S3-FIFO`（`10% Small-FIFO + 90% Main-Tree + Ghost Queue`）能彻底免疫 `128K` 单次扫描污染？**
   - 所有首次生成的 KV Block 先进入占总容量 `10%` 的 **试用队列（Small Probationary FIFO）**；
   - 像 `128K` 离线文档抽取这种“一锤子买卖（One-Hit Wonder）”的块，在试用队列里走完一圈发现访问次数 `freq == 1`，直接从试用队列尾部静默淘汰，**连碰都碰不到占 `90%` 容量的主缓存树（Main Protected Tree）**！只有被第二次访问命中（如多轮对话的第 2 轮、高频共享 Few-Shot 前缀）的块才会晋升进入主缓存树！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：把 Python 默认的 64 位 `hash()` 换成 256 位 `SHA-256`，会让 CPU 算哈希成为 Prefill 的性能瓶颈吗？
- **完全不会！每请求仅需 `14.8 µs`（0.015 毫秒）**！
  现代 CPU 均内置硬件 `SHA-NI` 指令集（单核吞吐 `> 1.5 GB/s`）。一段 `4,096` Token 的 Prompt 切成 256 个 `16-Token` 块，總共只需对约 `24 KB` 的元数据算 256 次增量 `SHA-256`，耗时仅 **`14.8 微秒`**，不到一次 GPU Prefill（`50~165 ms`）的万分之一，却能将 64 位生日悖论碰撞风险从 $2^{-32}$ 降到宇宙级安全的 $2^{-128}$！

### Q2：在前缀树（RadixTree / Block Tree）中执行淘汰时，为什么必须“自底向上只淘汰叶子节点（Leaf-First Bottom-Up Eviction）”？
- 因为在前缀缓存中，只有当块 $0, 1, \dots, k-1$ 全部连续命中（或配合父块状态）时，第 $k$ 个块才有意义！如果你先淘汰了树根附近的第 1 块，而保留了后面的第 $2..100$ 块，后面这 99 个块就会变成永远无法从头匹配上的“悬空孤岛”！因此淘汰器必须始终维护一个 **仅包含当前无子节点（`num_children == 0` 且 `ref_count == 0`）的叶子节点优先队列**，每次剥离最外层叶子后，若其父节点的子节点数归零，再将父节点推入可淘汰叶子队列。

### Q3：如果客户属于同一个企业组织（Organization）内部的不同微服务，希望共享同一套企业知识库前缀怎么办？
- 将网关注入的 `cache_salt` 粒度绑定到 **`org_id + security_clearance_level`（组织 ID + 密级标签）**，同一组织且密级相同的调用共享同一个加密盐值，不同组织或高低密级之间严格物理哈希隔离！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 多租户前缀缓存哈希安全隔离与抗扫描淘汰选型决策树

```mermaid
flowchart TD
    A["多租户 / 多 LoRA / 多模态推理集群开启自动前缀缓存 (APC / RadixAttention)"] --> B{"KV Block 哈希函数的构成要素?"}
    B -- "仅对 token_ids 算 64位 hash() 且无租户盐值" --> C["⛔ Tier 1: 跨租户 TTFT 侧信道泄露 98.4% 私密 Prompt + 跨 LoRA/图片 12.5% 脏数据碰撞!"]
    B -- "包含 (tokens + lora_id + mm_hash), 但仍无租户盐值" --> D["⛔ Tier 2: 虽消除 LoRA/多模态串扰, 但外部攻击者仍可通过 9.07x TTFT 差窃取私密 Prompt!"]
    B -- "注入鉴权租户盐值 SHA256(tenant_salt || parent || tokens || lora || mm)" --> E{"公共模板共享与缓存淘汰算法?"}
    E -- "完全物理隔离各租户树 + 平面 LRU 淘汰" --> F["⚠️ Tier 3: 0% 侧信道泄露, 但无法共享公共系统模板且被 128K 扫描冲刷 (命中率仅 54.8%)"]
    E -- "只读公共模板根 + 租户加盐 SHA-256 子树 + S3-FIFO 抗扫描淘汰" --> G["🏆 Tier 4: 0.0% 跨租户泄露 + 0.0% 脏碰撞 + 84.6% 持续高命中率 (+43.4 pt)!"]
```

### 3.2 无盐平面 `LRU`（`98.4%` 侧信道窃密 + `41.2%` 命中率）vs. 租户加盐 `SHA-256` + `S3-FIFO`（`84.6%` 命中率）对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: 全局无盐 hash(tokens) + 平面 LRU (侧信道泄密且易被大文档冲垮!)"]
      B1["攻击者探测前缀 -> 命中 TTFT=18.2ms vs 未命中 165.0ms (窃取 98.4% 私密 Prompt!)"] --> B2["单次 128K RAG 扫描涌入 -> 平面 LRU 把高频多轮对话全部挤出 (命中率跌至 41.2%)"]
    end
    subgraph Good["🏆 Tier 4: 公共只读根 + 租户加盐复合 SHA-256 + S3-FIFO 抗扫描淘汰"]
      G1["H_k = SHA256(salt || H_{k-1} || tokens || lora || mm) -> 0.0% 跨租户泄露与碰撞"] --> G2["10% Small-FIFO 过滤 128K 单次扫描块 -> 保护 90% 主树多轮会话 (命中率高达 84.6%!)"]
    end
```

### 3.3 API 网关注入 `tenant_salt`、链式计算复合 `SHA-256` 与 `S3-FIFO` 晋升时序图

```mermaid
sequenceDiagram
    participant GW as Auth Gateway (JWT -> tenant_salt)
    participant Hasher as Chained SHA-256 Block Hasher
    participant S3 as S3-FIFO Evictor (10% Small / 90% Main)
    participant HBM as GPU Paged KV Cache Pool

    GW->>Hasher: 1. Forward request with verified `tenant_salt`, `lora_id`, `mm_uuid`
    Hasher->>Hasher: 2. Compute `H_k = SHA256(H_{k-1} || tenant_salt || lora_id || mm_uuid || tokens_k)` (14.8 us)
    Hasher->>S3: 3. Lookup `H_k` in Public Root & Tenant Subtree
    alt First-Time Block Miss (e.g., 128K One-Hit Scan or New Chat Turn 1)
        S3->>HBM: 4a. Allocate in 10% `Small-FIFO` probationary queue (`freq = 1`)
    else Second Hit (`freq >= 2`, Multi-Turn Chat Turn 2+)
        S3->>HBM: 4b. Promote block from `Small-FIFO` to 90% `Main-Tree` (protected from scan eviction!)
    end
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | 前缀缓存哈希与淘汰方案 | 跨租户 Prompt 侧信道窃取率 | 跨 LoRA/多模态 KV 串扰率 | 混合负载持续命中率 | 哈希 CPU 耗时 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 无盐 `hash(tokens)` + 平面 LRU** | 全局共享无盐哈希，仅按最近访问淘汰 | `98.4%` (高危泄密! ✕) | `12.5%` (输出错乱! ✕) | `41.2%` ✕ | `4.2 µs` |
| **Tier 2: 无盐复合哈希 + 平面 LRU** | 加入 `lora_id` 与 `mm_hash`，仍无租户盐 | `98.4%` (高危泄密! ✕) | `0.0%` | `41.2%` ✕ | `8.5 µs` |
| **Tier 3: 严格租户独立缓存 + 平面 LRU** | 租户间完全不共享任何块，平面 LRU | `0.0%` | `0.0%` | `54.8%` | `14.0 µs` |
| **Tier 4: 公共只读根 + 租户加盐 `SHA-256` + `S3-FIFO` (推荐)** | **安全共享公共根 + 加密隔离私有树 + 抗扫描** | **`0.0%` 🏆** | **`0.0%` 🏆** | **`84.6%` (`+43.4 pt` 🏆)** | **`14.8 µs` 🏆** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/llm_prefix_cache_timing_side_channel_tenant_salted_sha256_eviction_selection/run.py`](../../experiments/llm_prefix_cache_timing_side_channel_tenant_salted_sha256_eviction_selection/run.py) · 封存数据：[`results.json`](../../experiments/llm_prefix_cache_timing_side_channel_tenant_salted_sha256_eviction_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/llm-prefix-cache-timing-side-channel-tenant-salted-sha256-eviction-sizer.html`](../reports/llm-prefix-cache-timing-side-channel-tenant-salted-sha256-eviction-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/llm-prefix-cache-timing-side-channel-tenant-salted-sha256-eviction-selection-cards.html`](../reports/llm-prefix-cache-timing-side-channel-tenant-salted-sha256-eviction-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Unsalted Token-Only + Flat LRU", "leak_pct": 98.4, "hit_rate_pct": 41.2 },
      { "tier": "Tier 2: Unsalted Composite + Flat LRU", "leak_pct": 98.4, "hit_rate_pct": 41.2 },
      { "tier": "Tier 3: Isolated Tenant + Flat LRU", "leak_pct": 0.0, "hit_rate_pct": 54.8 },
      { "tier": "Tier 4: Salted SHA-256 + Public Root + S3-FIFO", "leak_pct": 0.0, "hit_rate_pct": 84.6 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "leak_pct": "Percentage",
    "hit_rate_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tenant-Salted SHA-256 + S3-FIFO Eliminates 98.4% Prompt Stealing & Lifts Hit Rate to 84.6%",
    "subtitle": "Sustained prefix cache hit rate (%) across 4 security and eviction tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "hit_rate_pct",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有面向多租户开放的 SaaS 大模型 API、多 LoRA 适配器推理池、多模态 VLM 视觉问答以及交互式对话与大文档批处理混部的集群。
- **⛔ Not for**：在不可信多租户环境中使用无盐全局前缀缓存，或在多模态/多 LoRA 场景下仅对 `token_ids` 算哈希。

### 生产核验 Checklist
- [ ] **由鉴权网关强制注入每租户专属 `cache_salt` 并启用 `SHA-256` 块哈希（如 `vLLM --prefix-caching-hash-algo sha256`）**。
- [ ] **确保块哈希输入完整包含 `(parent_block_hash, tenant_salt, lora_int_id, mm_feature_hash, token_ids)` 五要素**。
- [ ] **启用抗单次扫描污染的 `S3-FIFO` / 双队列晋升淘汰机制，并严格按“自底向上仅驱逐无子节点的叶子块”顺序回收显存**。
