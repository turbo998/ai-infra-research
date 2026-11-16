# 大模型变长序列注意力（`FlashAttention Varlen cu_seqlens`）长短序列偏斜（`1x 64K + 63x 512`）、空块早退风暴与持久内核动态瓦片调度（`Persistent LPT + Split-KV`）选型：为什么已经用了去填充 `flash_attn_varlen_func`，混入 1 条 `64K` 长序列仍会让 H100 的 SM 利用率暴跌至 `16.4%`（耗时 `18.50 ms`）？基于 `132-CTA` 持久工作窃取 + 因果工作量最长优先（`Causal-Cost LPT`）+ 自适应 `Split-KV`（SM 利用率升至 `89.2%`，提速 `5.44x`）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/llm_flashattention_varlen_sequence_length_skew_persistent_tile_scheduler_selection/run.py` 与 `results.json`。

![FlashAttention Varlen Sequence-Length Skew & Persistent Tile Scheduler Visual Summary](../../assets/svg/llm-flashattention-varlen-sequence-length-skew-persistent-tile-scheduler-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向混合长度推理与变长训练（`FlashAttention-3 Persistent Scheduler / FlashInfer Ragged Plan / Blackwell CLC`）的“固定 `num_SMs (132)` 持久线程块（`Persistent CTAs`）+ 按因果三角形真实 KV 步数降序排列（`Causal-Cost Longest-Processing-Time First, LPT`）+ 超长尾块自适应 `Split-KV`”动态负载均衡架构**。
- **架构师价值**：很多算法和系统工程师有一个根深蒂固的误区：**“只要我不用 `padding=True`，改用了一维拼接的 `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)`，长短混合 Batch 就不会有任何算力浪费了！”**
  殊不知，当一个 Batch 里同时包含 **`1` 条 `65,536`（64K）的长文档/长代码请求** 和 **`63` 条 `512` Token 的普通短请求** 时，传统 `FlashAttention-2` 的静态网格发射机制会在 GPU 硬件层面发生灾难性的 **“空块早退风暴”与“单 SM 孤岛长尾”**：
  1. **灾难一：按最大长度 `max_seqlen_q = 65536` 发射静态三维 CUDA Grid（Tier 1）**：因为 CUDA 硬件启动 Kernel 时三维 Grid 的每一维必须是固定矩形 `(ceil(max_seqlen_q / 128), batch_size, num_heads) = (512, 64, H)`，这意味着 GPU 硬件调度器会为那 63 条长度只有 `512`（其实只需要 `4` 个 Q-Block）的短序列，**每条也硬生生发射 `512` 个 CTA 线程块**！其中 `508` 个越界 CTA 一进 SM 读完 `cu_seqlens` 发现 `block_idx * 128 >= 512` 就直接 `return` 退出了（高达 **`98.4%` 的发射波次是空跑开销**）；更致命的是，当所有短序列在 `0.2 ms` 内跑完后，整张 H100 的 `132` 个 SM 中有 **超过 `110` 个 SM 彻底闲置空转**，干等剩下的少数几个 SM 吭哧吭哧跑完那 1 条 `64K` 序列的 512 轮内循环（平均 SM 占用率暴跌至 **`16.4%`**，耗时高达 **`18.50 ms`**）！
  2. **灾难二：只把有效瓦片展平按 FIFO 顺序分发，未考虑因果掩码（Causal Mask）下三角形工作量差异（Tier 2）**：即使消除了空 CTA，如果那条 `64K` 序列排在队列后面，或者 `64K` 序列最后一行 Q-Block（需要遍历全部 `512` 个 KV 块）被分在最后一个 Wave 执行，仍然会出现“1 个 SM 跑 512 步，其余 131 个 SM 干等”的 **`11.20 ms` 尾波停顿**！
  采用 **持久内核（固定发射 `132` 个常驻 CTA，100% 消除空块发射）+ 按因果内循环真实步数从大到小排序（`Causal-Cost LPT`：先跑最重的 `64K` 底部块，把 `512` 短序列的轻量块留作最后填平尾波的“细沙”）+ 对超长行启用自适应 `Split-KV` 切分（Tier 4）**，可将 H100 平均 SM 利用率从 `16.4%` 拉满至 **`89.2%`**，有效算力从 `108 TFLOPS` 飙升至 **`586.0 TFLOPS`**，耗时从 `18.50 ms` 骤降至 **`3.40 ms`（提速 `5.44x`）**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：为什么 `flash_attn_varlen_func` 必须传入 `max_seqlen_q`，而这个参数恰恰是传统静态 Grid 的阿喀琉斯之踵？**
   - 在传统非持久内核（Non-Persistent Kernel）中，CPU 在发射 CUDA 内核时无法读取位于 GPU 显存里的 `cu_seqlens` 数组（读回 CPU 会触发同步阻塞），因此只能根据用户传入的标量上限 `max_seqlen_q` 盲目发射一个矩形 Grid：$\text{Grid} = \big(\lceil \text{max\_seqlen\_q} / M_{\text{block}} \rceil,\ B,\ H\big)$。
   - 当 $\text{max\_seqlen\_q} \gg \text{mean\_seqlen\_q}$（例如 1 条 `64K` 搭配 63 条 `512`，最大长度是平均长度的 **`43 倍`**！）时，静态 Grid 不仅浪费了数十万次硬件 CTA 调度周期，而且把同一条长序列的不同分块绑死在稀疏的 Wave 上！
2. **因果注意力的“直角三角形陷阱”：为什么同样是一个 `128` Token 的 Q-Tile，耗时却相差 `512 倍`？**
   - 在带因果掩码（`causal=True`）的条带中：
     - 序列开头的第 `0` 号 Q-Tile（Token `0..127`）只需要和第 `0` 号 KV-Tile 做 **唯一 `1` 次** 矩阵乘，后面的 KV 块全部被因果掩码跳过；
     - 而 `64K` 序列最末尾的第 `511` 号 Q-Tile（Token `65408..65535`）需要从第 `0` 号一直算到第 `511` 号 KV-Tile，足足要跑 **`512` 次** WGMMA 矩阵乘循环（耗时是第 0 号块的 **`512 倍`**）！
   - **最长处理时间优先调度律（`Causal-Cost LPT: Longest-Processing-Time First`）**：
     如果我们按自然顺序（从第 0 号块到第 511 号块）分发任务，那么最重的第 `511` 号巨石块一定是在**最后关头**才被某个倒霉的 SM 领走——此时其他 131 个 SM 早就没活干了，只能陪它干等 512 轮循环！
     **正确做法（倒序/按成本 LPT 排布）**：把 `64K` 序列最末尾的第 `511, 510, 509 ...` 号重型块放在全局任务队列的**最前面**，让 132 个 SM 一开场就先合力啃掉大石头；而把 `512` 短序列和第 `0, 1, 2` 号只需 1 轮循环的微型块留在队列**最后面**，像“细沙”一样把 132 个 SM 最后收尾的进度差填平到 **50 微秒以内**！
3. **当单条长序列的 Head 数 × 重型块数还不够喂饱 `132` 个 SM 时：启用自适应 `Split-KV`（`FlashInfer / FA3 Split-KV`）**：
   - 在 GQA/MLA 模型（例如只有 8 个 KV 头）中，1 条 `64K` 序列的最重尾块如果只按 Head 切分，并发度不足以铺满 132 个 SM。将长序列底部的 KV 循环拆成 4~8 个 `Split-KV` 子任务分给不同 SM 并行归约，最后通过 `LSE` 合并，即可在任何极端偏斜下锁死 `> 89%` 的 SM 满载率！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：在持久内核（Persistent Kernel）中，132 个 SM 通过全局原子加法 `atomicAdd(&tile_counter, 1)` 抢任务，会不会引起严重的 L2 原子锁争用或破坏 L2 Cache 命中率？
- **不会争用（每次 Tile 耗时数十微秒，原子操作仅占 `< 0.1%`），但必须注意 L2 Cache 的 KV 块复用亲和性（Swizzled / Grouped Order）**！
  为了让共享同一个 KV 头的多个 Query 头（GQA 组）或者相邻 Q-Tile 尽可能同时命中 L2 Cache 里的同一份 KV 块，任务队列在按 `Causal-Cost LPT` 宏观降序排列的同时，在同成本档位内部按 **`(batch_id, kv_head_id)` 分簇交错排列（L2-Locality Swizzling）**，使同时在 132 个 SM 上运行的 CTA 共同消费 L2 中热驻留的 KV 切片！

### Q2：`FlashInfer` 的 `BatchPrefillWithRaggedKVWrapper.plan()` 为什么要先在 CPU 上跑一个 `plan()` 函数，而不是直接裸调 CUDA Kernel？
- 因为 `plan()` 干的正是本指南 **Tier 4 的核心工作**：它在发射 GPU 内核前，根据当前 Batch 的 `qo_indptr`（即 `cu_seqlens`）在微秒级内算出每个请求的真实长度，自动把长序列切分成均匀的虚拟定长工作瓦片（Work Tiles）并按成本构建好静态调度表（Work-Tile Descriptor Table），从而让 GPU 上的 132 个持久 CTA 零分支、零空块地跑满硬件峰值！

### Q3：在 `SM100 Blackwell` 架构中，硬件新增的 `CLC`（Cluster Launch Control）对变长偏斜调度有什么质的飞跃？
- 在 Hopper (`SM90`) 上，持久内核只能固定启动 132 个驻留 CTA 用软件 `atomicAdd` 模拟任务窃取；而在 **Blackwell (`SM100`)** 上，硬件原生提供了 **`Cluster Launch Control (CLC)` 指令（`clusterlaunchcontrol.try_cancel`）**：允许硬件调度器从尚未启动的 Grid 尾部直接动态窃取和取消 CTA Cluster，既有静态 Grid 的零软件调度开销，又有持久内核的 100% 动态负载均衡能力！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 变长偏斜序列（`Varlen Skew`）注意力调度器选型决策树

```mermaid
flowchart TD
    A["Varlen 变长 Batch 存在显著长度偏斜 (如 1 条 64K 长文 + 63 条 512 短文)"] --> B{"CUDA Grid 发射与工作瓦片分配方式?"}
    B -- "按 max_seqlen_q (64K) 发射静态 3D Grid" --> C["⛔ Tier 1: 短序列 98.4% 的 CTA 越界早退 + 83.6% SM 空转等待 -> 耗时 18.50 ms!"]
    B -- "展平有效瓦片, 但按自然 FIFO 顺序 (0..N) 分发" --> D["⛔ Tier 2: 64K 最重的三角形底部块留在最后执行 -> 11.20 ms 尾波慢卡!"]
    B -- "固定 132-CTA 持久内核动态窃取" --> E{"是否按因果 KV 循环成本倒序 (LPT) 并对超长行做 Split-KV?"}
    E -- "仅做 Q-Tile 粒度窃取, 不拆 KV 循环" --> F["✅ Tier 3: 消除空 CTA, SM 占用率升至 62.5% (耗时 4.85 ms)"]
    E -- "因果成本降序 (LPT: 底部重块优先) + 轻量块填尾 + 自适应 Split-KV" --> G["🏆 Tier 4: 89.2% SM 满载 + 586 TFLOPS + 耗时压至 3.40 ms (快 5.44x)!"]
```

### 3.2 传统静态三维 Grid（`16.4%` SM 利用率）vs. 持久内核 `Causal LPT + Split-KV`（`89.2%` SM 利用率）对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1/2: 静态 max_seqlen Grid 或 FIFO 顺序 (18.50 ms, 83.6% SM 空转)"]
      B1["63 条短序列在 0.2ms 跑完"] --> B2["110+ 个 SM 彻底闲置 18ms, 围观少数 SM 跑 64K 因果大三角底部!"]
    end
    subgraph Good["🏆 Tier 4: 132-CTA 持久内核 + Causal LPT 倒序 + Split-KV (3.40 ms, 89.2% SM)"]
      G1["开场: 132 个 SM 合力并行切分啃掉 64K 最重的底部因果块"] --> G2["收尾: 用 63 条短序列的 1-step 微型块像细沙一样填平所有 SM 尾波!"]
    end
```

### 3.3 `Causal-Cost LPT` 规划器与 `132-CTA` 持久工作窃取执行时序图

```mermaid
sequenceDiagram
    participant Plan as Tile Planner (Causal LPT + Split-KV)
    participant Queue as Device Work-Tile Queue (Sorted Descending Cost)
    participant SMs as All 132 H100 SMs (Persistent CTAs)
    participant Reduce as Split-KV LSE Merge Epilogue

    Plan->>Queue: 1. Inspect `cu_seqlens`; split 64K heavy causal rows & sort tiles by KV loop count (heaviest first!)
    loop Dynamic Work-Stealing Until Queue Empty
        SMs->>Queue: 2. CTA atomically pulls next tile `tile_id = atomicAdd(&head, 1)`
        SMs->>SMs: 3. Wave 1..K: all 132 SMs execute heavy 64K tiles concurrently
        SMs->>SMs: 4. Final Wave: SMs that finish early pull lightweight 512-token tiles (zero tail idle!)
    end
    SMs->>Reduce: 5. Merge Split-KV partial accumulators via log-sum-exp in 0.05 ms -> Done in 3.40 ms!
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | 变长 `Varlen` 瓦片调度架构 | 内核执行耗时 | H100 平均 SM 利用率 | 有效算力 (`TFLOPS`) | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 静态 3D Grid (`max_seqlen_q` 矩形)** | 为每条短序列发射 512 个块后早退 | `18.50 ms` ✕ | `16.4%` (严重饥饿! ✕) | `108.0` ✕ | ⛔ **长短混跑时算力浪费超 80%** |
| **Tier 2: 展平有效块 + 自然 FIFO 顺序** | 先跑短块/浅块，最后剩 64K 重块 | `11.20 ms` ✕ | `27.1%` ✕ | `178.0` ✕ | ⛔ **受困于严重的尾波木桶效应** |
| **Tier 3: `132-CTA` 持久内核 Q-Tile 窃取** | 动态分摊 Q 块，但未拆超长 KV 循环 | `4.85 ms` ✅ | `62.5%` ✅ | `411.0` ✅ | ✅ 中等长度偏斜场景 |
| **Tier 4: 持久内核 + 因果 `LPT` + `Split-KV` (推荐)** | **重块先行 + 细沙填尾 + KV 拆分** | **`3.40 ms` (`5.44x` 快 🏆)** | **`89.2%` 🏆** | **`586.0` 🏆** | 🏆 **Chunked Prefill 与变长训练黄金标准** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/llm_flashattention_varlen_sequence_length_skew_persistent_tile_scheduler_selection/run.py`](../../experiments/llm_flashattention_varlen_sequence_length_skew_persistent_tile_scheduler_selection/run.py) · 封存数据：[`results.json`](../../experiments/llm_flashattention_varlen_sequence_length_skew_persistent_tile_scheduler_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/llm-flashattention-varlen-sequence-length-skew-persistent-tile-scheduler-sizer.html`](../reports/llm-flashattention-varlen-sequence-length-skew-persistent-tile-scheduler-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/llm-flashattention-varlen-sequence-length-skew-persistent-tile-scheduler-selection-cards.html`](../reports/llm-flashattention-varlen-sequence-length-skew-persistent-tile-scheduler-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Static 3D Grid (max_seqlen_q)", "latency_ms": 18.50, "sm_util_pct": 16.4 },
      { "tier": "Tier 2: Unsorted FIFO Flattened Queue", "latency_ms": 11.20, "sm_util_pct": 27.1 },
      { "tier": "Tier 3: Persistent Atomic Q-Tile Stealing", "latency_ms": 4.85, "sm_util_pct": 62.5 },
      { "tier": "Tier 4: Persistent Causal-LPT + Split-KV", "latency_ms": 3.40, "sm_util_pct": 89.2 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "latency_ms": "Quantity",
    "sm_util_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Persistent 132-CTA Causal-LPT + Split-KV Scheduler Speeds Up Skewed Varlen Attention 5.44x",
    "subtitle": "Kernel latency (ms) and average H100 SM utilization (%) on 1x 64K + 63x 512 varlen batch (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "latency_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有存在长短请求混跑的在线推理引擎（`vLLM / SGLang` Ragged Prefill）、多模态动态分辨率 Patch 批次以及未截断长短混装的 SFT/预训练。
- **⛔ Not for**：在序列长度方差极大的 Batch 上使用基于 `max_seqlen_q` 静态 3D Grid 的旧版算子。

### 生产核验 Checklist
- [ ] **启用持久瓦片调度器（`FlashAttention-3 Persistent Scheduler` 或 `FlashInfer Ragged Plan`）**：固定启动 `num_SMs` 个常驻 CTA，杜绝 `max_seqlen_q` 引发的空块发射风暴。
- [ ] **按因果内循环成本降序排列工作瓦片（`Causal-Cost LPT`：大索引 Q-Tile 优先）**：先执行计算量最大的三角形底部块，用短序列轻量块填平末尾 Wave。
- [ ] **对长尾序列开启自适应 `Split-KV`（或 `Chunked Prefill` 限长切片）**：确保单个超大请求也能拆解并铺满全部 132 个 SM。
