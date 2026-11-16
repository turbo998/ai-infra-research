# 大模型并行采样（`n=16` / `GRPO G=16` 组采样）共享 Prompt KV 引用计数与非对齐尾块写时复制（`Tail-Block Copy-on-Write, CoW`）选型：为什么一条 `4,005` Token 的 Prompt 在 `n=16` 并行采样时，要么白白浪费 `16 倍` Prefill 算力与 `31.36 GB` 显存，要么因为尾块未对齐（`4005 % 16 = 5`）导致 `93.75%`（15/16）的子序列在 Decode 第一步发生并发写踩踏？基于满块零拷贝引用计数（`ref_count=16`）+ 半满尾块写时复制（`CoW`，TTFT 提速 `15.6x`，KV 显存降至 `1.98 GB`，`0%` 踩踏）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/llm_serving_parallel_sampling_n_rollout_shared_prefix_cow_tail_block_selection/run.py` 与 `results.json`。

![Parallel Sampling Shared Prompt KV & Tail-Block Copy-on-Write Visual Summary](../../assets/svg/llm-serving-parallel-sampling-n-rollout-shared-prefix-cow-tail-block-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向强化学习组采样（`GRPO G=16..64 Rollout`）与推理并行采样（`OpenAI n > 1 / Best-of-N / Tree-of-Thought`）的“单次 Prefill 满块零拷贝共享（`ref_count = n`）+ 非对齐半满尾块 `O(1)` 写时复制分裂（`Tail-Block Copy-on-Write Fork`）+ `FlashInfer Cascade` 级联共享前缀解码”架构**。
- **架构师价值**：在 DeepSeek-R1 范式的 `GRPO` 强化学习训练（每道数学/代码题需要对完全相同的 Prompt 并行采样 $G = 16$ 甚至 $32$ 条思维链轨迹）以及在线 `n = 16` Best-of-N 推理中，围绕 Prompt KV Cache 的复用存在两个极易踩中的工程陷阱：
  1. **陷阱一：在调度器入口直接把 `n=16` 拆成 `16` 条独立请求去跑 Prefill（`15.6x` 算力浪费 + `31.36 GB` 显存爆炸，Tier 1）**！
     - 如果客户端循环发了 16 个并发请求（或网关把 `n=16` 展开成 16 条独立序列）且未做单次 Prefill 派生（Fork），GPU 会对这 `4,005` 个完全一模一样的 Prompt Token 重复计算 `16` 遍（共算 `64,080` 个 Token），导致首字延迟 `TTFT` 从 `95.0 ms` 暴涨至 **`1,480.0 ms`**，且原本只需 `1.96 GB` 的 Prompt KV Cache 被复制了 16 份，狂吞 **`31.36 GB` 显存**！
  2. **陷阱二：直接克隆页表 `block_table.clone()` 共享全部块，却忽略了“最后一个块没装满（`4005 % 16 = 5`）”引发的 `93.75%` 并发写踩踏事故（Tier 2）**！
     - 假设 PagedAttention 的块大小 `block_size = 16`。一条长度为 `4,005` Token 的 Prompt 在完成单次 Prefill 后，一共占用了 `251` 个物理块：
       - 前 **`250` 个块（Block `0..249`）**：每个块都装满了 16 个 Token（共 `4,000` 个 Token），在后续 Decode 阶段 **只读不写**；
       - **第 `251` 个块（尾块 Block `#250`）**：里面只装了 **`5` 个剩余的 Prompt Token（占用了槽位 `slot 0..4`）**，而后面的 **`slot 5..15` 还是空的**！
     - 如果你在派生 `16` 条子序列时，只是简单地把这 `251` 个物理块 ID 原封不动拷给 16 个子序列（令所有块的 `ref_count = 16`），那么当进入 **Decode 第 1 步** 时，灾难发生了：
       - 16 条子序列各自采样出了不同的第 1 个回答词，现在它们都要把新生成的第 `4,006` 个 Token（逻辑下标 `4005`）写进 **各自页表最后一个块（仍然是同一个物理块 `#250`！）的 `slot 5` 位置**！
       - **16 个 CUDA 线程束同时往同一个物理显存地址 `(phys_block=250, slot=5)` 写入各自不同的 KV 向量**！除了最后一个写进去的线程束外，其余 **15 条子序列（`15 / 16 = 93.75%`）在 `slot 5..15` 上的 KV 状态全部被相互覆盖污染**，生成的文本或对数概率彻底失真！
  采用 **前 `250` 个满块零拷贝引用共享（`ref_count = 16`，不复制一字节）+ 仅对最后 1 个未装满的尾块 `#250` 执行写时复制（`Copy-on-Write, CoW`：分配 15 个新物理块并仅拷贝前 `5` 个槽位的 KV 数据，耗时仅 `0.05 ms`）（Tier 4）**，可在保证 **`0.0%` 并发写踩踏** 的前提下，将 `TTFT` 从 `1,480.0 ms` 压降至 **`95.05 ms`（提速 `15.6 倍`）**，并将 `n=16` 的 Prompt KV 显存从 `31.36 GB` 压降至 **`1.98 GB`（节省 `15.8 倍` 显存，省下 `29.38 GB` HBM）**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：PagedAttention 物理块的“可变性分界定理（Mutability Boundary Theorem）”**：
   - 在块大小为 $B$（如 $B=16$）的分页 KV Cache 中，任意长度为 $L$ 的序列被划分为：
     $$N_{\text{full}} = \lfloor L / B \rfloor \text{ 个已封口满块（Immutable Full Blocks）}, \qquad r = L \bmod B \text{ 个尾部余数 Token}$$
   - **定理**：
     1. 凡是已经装满 $B$ 个 Token 的物理块，其内部所有槽位 `[0 .. B-1]` 均已写入完毕，在后续自回归生成中 **永远只读（Read-Only）**，因此可以被任意 $n$ 条子序列安全地 **100% 零拷贝共享（`ref_count = n`）**！
     2. 只要 $r = L \bmod B > 0$，最后一个物理块（尾块）就处于 **半满可写状态（Mutable Partial Block，已填 `[0 .. r-1]`，待写 `[r .. B-1]`）**！任何时候当某条序列准备向一个物理块的槽位 $r$ 写入新 Token KV 时，**若检查到该物理块的 `ref_count > 1`，必须立即触发写时复制（Copy-on-Write, CoW）**！
2. **写时复制（`CoW`）的开销有多小？为什么比全量克隆（Tier 3）强得多？**
   - 全量克隆（Tier 3）要把全部 `4,005` 个 Token 的 KV 复制 15 遍（搬运 `29.4 GB` 显存，耗时 `42.0 ms`，且永久占死 `31.36 GB` HBM）；
   - 而 **尾块 `CoW`（Tier 4）** 只复制最后那 **`r = 5` 个余数 Token**（仅占 `4,005` Token 的 `0.125%`！额外分配 15 个物理块仅占 **`0.02 GB`**，拷贝耗时仅 **`0.05 ms`**），随后当各子序列填满各自的尾块进入下一个新块时，直接分配独立空块即可！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：如果客户端没有传 `n=16`，而是由 16 个并发协程在同一毫秒发来了 16 个带着相同 Prompt 的 `n=1` 请求（例如 RL 框架里的并发异步 Rollout Worker），怎么防止这 16 个请求在还没算完 Prefill 时同时重复跑 16 遍 Prefill？
- **必须在调度器层实现“在途相同前缀合并（In-Flight Prefix Deduplication / Singleflight Prefill）”**！
  当第 1 个请求正在计算这 `250` 个满块的 Prefill 时，后到达的 15 个相同前缀请求在 RadixTree / BlockHash 表中发现这 `250` 个块处于 `COMPUTING_IN_FLIGHT` 状态，调度器直接将后 15 个请求挂起等待第 1 个请求的 Prefill 完成（或在本步直接降级为 1 次 Prefill + 派生 16 个子分支），绝不允许同一批里把相同的未缓存块重复算 16 遍！

### Q2：在 Decode 阶段，既然 16 条子序列共享了前 `250` 个物理块（`1.95 GB` KV 数据），为什么朴素 PagedAttention Decode Kernel 还是把这 `1.95 GB` 从 HBM 重复读了 16 遍？
- **好问题！物理地址共享解决了“HBM 容量浪费”，但要进一步解决“Decode 阶段的 HBM 带宽重复读取”，必须启用级联共享前缀注意力（`FlashInfer Cascade Attention / Multi-Level Shared-Prefix Decode`）**！
  - `Cascade Attention` 将 Decode 注意力拆成两半：
    1. **第一阶段（Shared Prefix Pass）**：将 16 条子序列的 Query 拼成一个 `[16, H_q, D]` 的小矩阵（算术强度直接提升 16 倍，从 Memory-Bound 变成 Compute-TensorCore 加速！），**只从 HBM 读取 1 遍** 那 `1.95 GB` 的公共前缀 KV，算出公共部分的部分和 $(O_{\text{prefix}}, \text{LSE}_{\text{prefix}})$；
    2. **第二阶段（Unique Suffix Pass）**：各子序列分别读取自己私有的尾块与生成块，算出 $(O_{\text{suffix}}, \text{LSE}_{\text{suffix}})$，最后用在线 Softmax LSE 归约公式合并！这能让 `GRPO G=16` 的 Decode 吞吐直接翻 **`3.5~5.2 倍`**！

### Q3：当其中一条子序列率先生成了 `<|endoftext|>` 结束释放显存时，共享的前 `250` 个块会被误删吗？
- **不会，只要严格遵守引用计数递减规则**：每当一条子序列结束时，遍历其 `block_table` 中持有的所有物理块，执行 `block.ref_count -= 1`；只有当 `block.ref_count == 0`（且若开启了前缀缓存则进入可淘汰队列被驱逐）时，该物理块才会归还给空闲池。

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 并行采样（`n > 1` / `GRPO G=16`）Prompt KV 共享与尾块 `CoW` 选型决策树

```mermaid
flowchart TD
    A["收到 L=4,005 Token 的并行采样请求 (n=16 或 GRPO G=16, block_size=16)"] --> B{"Prefill 阶段如何处理 n=16 个分支?"}
    B -- "拆成 16 条独立序列分别跑 Prefill" --> C["⛔ Tier 1: 算力浪费 16 倍 (TTFT 1,480 ms) + 显存暴涨 16 倍 (31.36 GB)!"]
    B -- "只跑 1 次 Prefill (95 ms), 然后派生 16 条 Decode 子序列" --> D{"如何处理页表 (250 个满块 + 1 个仅含 5 Token 的半满尾块 #250)?"}
    D -- "直接克隆页表, 所有 251 个块均不复制" --> E["⛔ Tier 2: Decode 第 1 步 16 个分支同时写 (block #250, slot 5) -> 93.75% 数据被踩踏覆盖!"]
    D -- "用 cudaMemcpy 把全部 251 个块复制 15 份" --> F["⚠️ Tier 3: 虽无踩踏, 但多花 42 ms D2D 拷贝且仍白白吃掉 31.36 GB 显存!"]
    D -- "前 250 个满块零拷贝共享 (ref_count=16) + 仅对半满尾块 #250 执行写时复制 (CoW)" --> G["🏆 Tier 4: TTFT 95.05 ms (快 15.6 倍) + 显存仅 1.98 GB (省 15.8 倍) + 0.0% 写踩踏!"]
```

### 3.2 半满尾块未做 `CoW` 引发 `93.75%` 并发写踩踏 vs. 满块共享 + 尾块 `CoW` 对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 2: 直接克隆页表未做尾块 CoW (16 个分支同时覆写物理块 #250 的 Slot 5!)"]
      B1["共享尾块 #250: Slot 0..4 存着 Prompt 最后 5 个词 (ref_count=16)"] --> B2["Decode Step 1: 16 个子分支全部把各自生成的第 1 个词写向 (Block #250, Slot 5)!"]
      B2 --> B3["结果: 15/16 (93.75%) 的分支 KV 被互相覆盖损坏!"]
    end
    subgraph Good["🏆 Tier 4: 满块 0..249 零拷贝共享 (ref_count=16) + 尾块 #250 写时复制 (CoW)"]
      G1["满块 0..249 (4,000 词): 只读不写, 16 分支安全共享 1.95 GB"] --> G2["尾块 #250 (ref_count=16 > 1): 触发 CoW 分配 15 个私有尾块, 仅复制前 5 个 Slot (0.05 ms)"]
      G2 --> G3["各分支在私有尾块的 Slot 5..15 安全写入生成词 -> 1.98 GB 总显存, 0% 踩踏!"]
    end
```

### 3.3 单次 Prefill 派生 `n=16` 子序列并在 Decode 第一步触发尾块 `CoW` 时序图

```mermaid
sequenceDiagram
    participant Sched as PagedAttention Scheduler
    participant Alloc as Block Allocator (ref_count)
    participant GPU as GPU Copy & Decode Engine

    Sched->>GPU: 1. Run single Prefill for L=4,005 tokens -> fills Blocks `0..249` (full) + Block `#250` (slots `0..4`)
    Sched->>Alloc: 2. Fork n=16 child sequences: clone `block_table`, set `ref_count = 16` on Blocks `0..250`
    Sched->>Alloc: 3. Decode Step 1 append at `pos=4005` (`slot=5` of tail Block `#250`): detect `ref_count == 16 > 1`!
    Alloc->>GPU: 4. Trigger CoW for children `1..15`: allocate 15 new blocks & copy slots `0..4` from `#250` (0.05 ms)
    GPU->>GPU: 5. All 16 children now hold `ref_count == 1` on their tail blocks and safely write `slot=5` in parallel!
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | `n=16` 并行采样 KV 管理方案 | 首字延迟 (`TTFT`) | Prompt KV 显存占用 | 尾块并发写踩踏率 | D2D 拷贝耗时 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 展开为 16 条独立序列跑 Prefill** | 无共享，重复计算并存储 16 份 | `1,480.0 ms` ✕ | `31.36 GB` ✕ | `0.0%` | `0.0 ms` |
| **Tier 2: 共享页表但尾块未做 `CoW`** | 直接克隆 `block_table` 不管尾块余数 | `95.0 ms` | `1.96 GB` | **`93.75%` (严重故障! ✕)** | `0.0 ms` |
| **Tier 3: 单次 Prefill + 全量深拷贝 15 份** | 算 1 次，但把全部 251 个块拷贝 15 份 | `137.0 ms` | `31.36 GB` ✕ | `0.0%` | `42.0 ms` ✕ |
| **Tier 4: 满块零拷贝 + 尾块 `CoW` (推荐)** | **前 250 块共享 + 仅对尾块 5 槽位做 CoW** | **`95.05 ms` (`15.6x` 🏆)** | **`1.98 GB` (`15.8x` 🏆)** | **`0.0%` 🏆** | **`0.05 ms` 🏆** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/llm_serving_parallel_sampling_n_rollout_shared_prefix_cow_tail_block_selection/run.py`](../../experiments/llm_serving_parallel_sampling_n_rollout_shared_prefix_cow_tail_block_selection/run.py) · 封存数据：[`results.json`](../../experiments/llm_serving_parallel_sampling_n_rollout_shared_prefix_cow_tail_block_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/llm-serving-parallel-sampling-n-rollout-shared-prefix-cow-tail-block-sizer.html`](../reports/llm-serving-parallel-sampling-n-rollout-shared-prefix-cow-tail-block-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/llm-serving-parallel-sampling-n-rollout-shared-prefix-cow-tail-block-selection-cards.html`](../reports/llm-serving-parallel-sampling-n-rollout-shared-prefix-cow-tail-block-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: 16x Independent Prefill", "prompt_kv_gb": 31.36, "ttft_ms": 1480.0 },
      { "tier": "Tier 2: Unprotected Page-Table Clone (93.75% Bug)", "prompt_kv_gb": 1.96, "ttft_ms": 95.0 },
      { "tier": "Tier 3: Single Prefill + Eager Full D2D Clone", "prompt_kv_gb": 31.36, "ttft_ms": 137.0 },
      { "tier": "Tier 4: Shared Full Blocks + Tail-Block CoW", "prompt_kv_gb": 1.98, "ttft_ms": 95.05 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "prompt_kv_gb": "Quantity",
    "ttft_ms": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Shared Full Blocks + Tail-Block CoW Shrinks n=16 Prompt KV HBM 15.8x (31.36 GB -> 1.98 GB) with 0% Race",
    "subtitle": "Prompt KV cache HBM footprint (GB) for L=4,005 and n=16 across 4 tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "prompt_kv_gb",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有 `GRPO / RLOO` 强化学习群组 Rollout（`G = 8..64`）、在线 `n > 1` Best-of-N 采样、束搜索（Beam Search）及树状思维链派生。
- **⛔ Not for**：把 `n > 1` 拆成独立 Prefill，或在克隆 `block_table` 后直接向 `ref_count > 1` 的非对齐半满尾块追加写入新 Token。

### 生产核验 Checklist
- [ ] **用非块大小整数倍的 Prompt 长度（如 `L = 4,005`，`4005 % 16 = 5`）做 `n=16` 并行采样回归测试**：核验每个子分支在槽位 `5..15` 的 KV 值未被兄弟分支覆盖。
- [ ] **在 `append_slot()` 中强制检查尾块 `if tail_block.ref_count > 1`**：立即触发写时复制（`CoW`）分配私有尾块并仅拷贝已填的 $L \bmod B$ 个槽位。
- [ ] **配合启用 `FlashInfer Cascade Attention`（级联共享前缀注意力）**：在 Decode 阶段将 16 个子分支的 Query 合并读取 1 次共享前缀块。
