# 大模型 PD 分离推理（`Disaggregated Prefill-Decode: vLLM KVConnector / SGLang Mooncake / NIXL`）逐层流水线异步 KV 推送（`Layerwise Pipelined KV Push`）与连续块聚合打包选型：为什么等 80 层全部算完再按分页块发 `16,000` 个小 RDMA 请求会在第 1 个词与第 2 个词之间产生 `148.0 ms` 的断层卡顿（网卡利用率仅 `35.8%`）？基于逐层 `cudaEvent` 流水线推送 + 单层连续缓冲聚合（暴露卡顿从 `148.0 ms` 缩至 `4.2 ms`，提速 `35.2x`，网卡利用率达 `94.6%`）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/llm_disaggregated_pd_layerwise_pipelined_kv_transfer_strided_block_packing_selection/run.py` 与 `results.json`。

![Disaggregated PD Layerwise Pipelined KV Transfer Visual Summary](../../assets/svg/llm-disaggregated-pd-layerwise-pipelined-kv-transfer-strided-block-packing-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向长上下文 PD 分离推理（`vLLM V1 KVConnector / SGLang Disaggregation + Mooncake TransferEngine / NVIDIA Dynamo NIXL`）的“逐层 `cudaEvent` 触发流水线异步 KV 推送（`Layer l` 算完 Attention 立即发往 Decode 节点，与 `Layer l+1` 计算 92.8% 重叠）+ 双缓冲单层连续块聚合打包（将 `16,000` 个细碎 Paged Block RDMA WQE 合并为 `80` 个大块连续传输，400GbE 网卡有效带宽利用率从 `35.8%` 跃升至 `94.6%`）”核心架构**。
- **架构师价值**：在上线 Prefill-Decode（PD）分离推理集群后，许多架构师发现了一个诡异的用户体验退化：**对于一条 `16,384` Token 的长 Prompt，Prefill 节点明明已经算出了第 1 个生成的 Token（`TTFT` 很正常），但在流式吐出第 1 个字之后、第 2 个字（Decode Step 1）出来之前，客户端画面会突然卡死整整 `148.0 ms`（在 `32K` 上下文下更是卡顿长达 `296.0 ms`）！而且在此期间，Prefill 节点上的 `2.68 GB` KV 显存一直被锁死无法释放给下一个请求！**
  拆解 PD 分离引擎的 KV Cache 传输层，会揪出两个致命的设计反模式：
  1. **反模式一：等全部 80 层 Transformer 的 Prefill 完全算完，才开始启动跨网 KV Cache 传输（End-of-Prefill Bulk Transfer，计算与通信重叠率 `0.0%`，Tier 1 & Tier 2）**！
     - 在 70B 模型（80 层）跑 `16K` Prefill 时，**第 0 层（Layer 0）的 KV Cache 其实在 Prefill 开始后的第 `3 毫秒` 就已经 `100%` 计算完毕且再也不会修改了**！
     - 然而，朴素实现让 400GbE RDMA 网卡在整整 80 层 Prefill 计算期间（约 `180~250 ms`）完全闲置发呆，非要等到第 79 层算完并采样出第 1 个 Token 后，才一口气把 80 层共 `2.68 GB` 的 KV Cache 推上网络——导致 **100% 的跨机传输时间全部裸露在关键路径（Token 1 到 Token 2 的交接窗口）上**！
  2. **反模式二：直接对着离散的分页表（PagedAttention Block Table）逐层、逐 K/V、逐 Block 下发 `16,000` 个微型 RDMA Write 请求（网卡利用率暴跌至 `35.8%`，Tier 1 & Tier 3）**！
     - 在 PagedAttention 中，每个请求的物理块号 `block_ids = [b_0, b_1, ..., b_99]` 在显存里是不连续的，且在標準 `[num_layers=80, 2 (K/V), num_blocks, block_size, H_kv, D]` 布局下，每一层的 K 块和 V 块分散在 80 个不同的张量里！
     - 如果不对离散块做聚合，传输一条 `16K` 请求（假设 100 个块）需要向 InfiniBand / RoCE 网卡提交 **`80 层 × 2 (K/V) × 100 块 = 16,000 个` 独立的细碎 RDMA Write WQE（每个仅几百 KB 甚至几十 KB）**！
     - **硬件瓶颈**：RDMA 网卡的门铃寄存器（Doorbell）、PCIe 事务开销与 WQE 解析速率在处理上万个细碎请求时会严重饱和，导致 400GbE 网卡的实际有效吞吐跌至 **`35.8%`**，传完 `2.68 GB` 硬生生花了 **`148.0 ms`**！
  采用 **逐层流水线异步推送（每当第 $l$ 层 Attention 写完 KV Cache，立即在计算流打下 `cudaEventRecord(evt[l])`，专用的传输流一收到事件即刻将第 $l$ 层的 100 个离散块打包进连续双缓冲区 `staging_buf[l % 2]`，并仅用 **`1 个`** `33.5 MB` 的大块 RDMA Write 发往 Decode 节点，与第 $l+1$ 层的 MLP 和 Attention 计算 `92.8%` 重叠）（Tier 4）**，当第 79 层在 Prefill 节点算完时，**前 79 层（`98.75%`）的 KV Cache 早已稳稳躺在 Decode 节点的显存里了**！暴露在关键路径上的交接延迟从 `148.0 ms` 暴降至 **`4.2 ms`（提速 `35.2 倍`）**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：逐层流水线重叠率公式（Layerwise Overlap Law）**：
   - 设模型共有 $L = 80$ 层，第 $l$ 层的计算耗时为 $t_{\text{comp}}$，单层 KV Cache 传输耗时为 $t_{\text{xfer}}$。
   - **批量尾传（Bulk End-of-Prefill，Tier 1/2）**：暴露在 Prefill 结束后的交接等待时间为全部 $L$ 层传输之和：
     $$T_{\text{exposed}}^{\text{bulk}} = L \cdot t_{\text{xfer}} \quad (\text{148.0 ms under uncoalesced, 58.4 ms under coalesced})$$
   - **逐层流水线（Layerwise Pipelined，Tier 4）**：只要网卡单层传输时间 $t_{\text{xfer}} \le t_{\text{comp}}$（在 `4K~64K` 长上下文下，Attention 计算复杂度是 $O(S^2)$，而 KV 传输复杂度只是 $O(S)$，因此计算时间天然大于传输时间！），前 $L-1$ 层的传输 **100% 被后续层的计算完全掩盖**，唯一暴露在最后的只有 **第 $L-1$ 层（最后 1 层）的传输与远端解包时间**：
     $$T_{\text{exposed}}^{\text{pipe}} \approx t_{\text{xfer}}^{(L-1)} + t_{\text{unpack}} = \mathbf{4.2\,\text{ms}} \quad (\text{比批量尾传快 } \mathbf{35.2\times}\text{!})$$
2. **为什么“GPU 侧双缓冲连续打包（Double-Buffered Contiguous Gather/Scatter）”比直接发 `16,000` 个散乱 RDMA WQE 快 `2.6 倍`？**
   - GPU 拥有高达 **`3.35 TB/s`**（H100）的内部 HBM 带宽，而跨机 400GbE 网卡只有 **`50 GB/s`** 带宽（相差 **67 倍**）！
   - 利用 GPU 充沛的内部带宽，在专用 `xfer_stream` 上用一个极轻量的 Triton Kernel 把本层 100 个不连续的物理块聚合成 1 块 `33.5 MB` 的连续内存（仅耗时 `0.02 ms`），即可让 RDMA 网卡每次只发 **1 个 `33.5 MB` 的巨型连续包（全模型 80 层共 80 个 WQE，减少 `200 倍`）**，将 400GbE 线速利用率拉满至 **`94.6%`**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：如果开启了 `Chunked Prefill`（例如把一条 `16K` 的请求拆成 4 个 `4K` 的 Chunk 分 4 步算完），逐层流水线 KV 传输应该在什么时候发？
- **两种精细策略**：
  1. **按已完成块实时流式推送（Block-Ready Streaming）**：每当一个 `4K` Chunk 算完，其对应的 25 个 PagedAttention 物理块就已经彻底封口（Immutable），立即在后台逐层把这 25 个块推往 Decode 节点，实现跨 Chunk 流水重叠；
  2. **最后一跳 Chunk 逐层推送**：若前几个 Chunk 期间 Decode 节点尚未分配目标槽位（为节省 Decode 显存租期），则在最后一个 Chunk（即生成第 1 个 Token 的那一步）启动时预先分配 Decode 槽位，并在该步的 80 层前向中全速流式推送全部已就绪层！

### Q2：如果 Prefill 节点的张量并行度是 `TP_prefill = 8`，而 Decode 节点的张量并行度是 `TP_decode = 4`（异构 TP 配比），连续打包缓冲怎么做点对点对齐？
- 在 GQA（如 `H_kv = 8` 个 KV 头）下：
  - `TP_prefill = 8` 时，Prefill 的每张卡持有 `1` 个 KV 头；
  - `TP_decode = 4` 时，Decode 的每张卡持有 `2` 个 KV 头（Rank $d \in \{0..3\}$ 持有第 $2d$ 和 $2d+1$ 号头）。
  因此 Prefill 的第 $2d$ 号卡和第 $2d+1$ 号卡只需各自将自己的连续块直接单向 `RDMA Write` 到 Decode 第 $d$ 号卡接收缓冲区的左半区与右半区，**零中转、零跨卡重排**！

### Q3：双缓冲连续打包区（`staging_buf[2]`）会占用多少额外 GPU 显存？
- 因为它是 **跨所有 80 层循环复用的双缓冲（Ring Buffer of Depth 2）**：当第 $l$ 层正在通过 GPUDirect RDMA 发往网卡时，第 $l+1$ 层写入另一个槽位 `(l + 1) % 2`。单层 `16K` 上下文的 KV 大小在 `TP=8` 下仅约 `4.2 MB`（全 8 卡合起来 `33.5 MB`），因此每张卡的双缓冲仅占 **`8.4 MB`（全机 `67 MB`）** 显存，几乎可以忽略不计！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 PD 分离推理 KV Cache 跨节点传输架构选型决策树

```mermaid
flowchart TD
    A["PD 分离推理集群传输长上下文 KV Cache (70B 80层, 16K Token = 2.68 GB, 400GbE RDMA)"] --> B{"KV Cache 传输的触发时机?"}
    B -- "等全部 80 层 Prefill 结束才开始传输 (Bulk End-of-Prefill)" --> C{"离散分页块是否先在 GPU 内聚合成连续块?"}
    C -- "否: 直接按 (层, K/V, Block) 提交 16,000 个小 RDMA WQE" --> D["⛔ Tier 1: 网卡门铃饱和 (利用率仅 35.8%) + 0% 计算重叠 -> 暴露卡顿长达 148.0 ms!"]
    C -- "是: 聚合成连续缓冲后批量发送" --> E["⛔ Tier 2: 网卡虽跑满 (91.2%), 但因 0% 重叠仍裸露 58.4 ms 交接停顿!"]
    B -- "逐层 cudaEvent 触发: 第 l 层算完立即异步推送 (与第 l+1 层计算重叠)" --> F{"单层内部的 100 个分页块如何发送?"}
    F -- "直接发 200 个细碎 SGL/WQE" --> G["⚠️ Tier 3: 重叠 74.3%, 但细碎 WQE 拖慢传输使尾部仍暴露 38.0 ms"]
    F -- "双缓冲 GPU Kernel 聚合成单块连续内存 + 每层仅 1 个大 RDMA Write" --> H["🏆 Tier 4: 92.8% 完美隐藏 + 94.6% 网卡利用率 -> 暴露交接停顿仅 4.2 ms (快 35.2 倍)!"]
```

### 3.2 批量尾传 + 散乱 WQE（`148.0 ms` 暴露卡顿）vs. 逐层流水线 + 连续打包（`4.2 ms`）对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: 等 80 层全算完再发 16,000 个散乱 WQE (暴露 148.0 ms 断层卡顿!)"]
      B1["Prefill Layer 0..79 计算期间: RDMA 网卡 100% 闲置!"] --> B2["Prefill 结束: 提交 16,000 个细碎块 WQE -> Decode 干等 148.0 ms 才吐第 2 个字!"]
    end
    subgraph Good["🏆 Tier 4: 逐层 cudaEvent 流水线推送 + 双缓冲单层连续打包 (仅暴露 4.2 ms!)"]
      G1["Layer l Attention 算完 -> 立即打包成 1 个连续块并发起 RDMA Write (与 Layer l+1 计算并行!)"] --> G2["Layer 79 算完时, Layer 0..78 早已到达 Decode 显存 -> 仅剩 4.2 ms 尾包即刻开跑 Decode!"]
    end
```

### 3.3 计算流（`compute_stream`）、打包流（`pack_stream`）与 RDMA 网卡三流流水线时序图

```mermaid
sequenceDiagram
    participant Comp as Prefill Compute Stream
    participant Pack as GPU Pack Stream (staging_buf[2])
    participant RNIC as 400GbE GPUDirect RDMA NIC
    participant Dec as Decode Node HBM

    loop Layer l = 0 .. 79
        Comp->>Comp: 1. Run Layer `l` Attention & write Paged KV Cache
        Comp->>Pack: 2. `cudaEventRecord(evt[l])` -> `cudaStreamWaitEvent(pack_stream, evt[l])`
        Note over Comp,Pack: Compute stream immediately starts Layer `l` MLP & Layer `l+1` Attention!
        Pack->>RNIC: 3. Gather Layer `l` blocks into `staging_buf[l%2]` (0.02 ms) & post 1 RDMA Write
        RNIC->>Dec: 4. Stream Layer `l` KV directly into Decode staging slab at 94.6% line rate!
    end
    Dec->>Dec: 5. Layer 79 arrives -> scatter into Decode Paged Pool & launch Decode Step 1 (4.2 ms gap!)
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | PD 分离 KV 传输与打包方案 | Prefill 后暴露交接卡顿 | 400GbE 网卡利用率 | 每请求 RDMA WQE 数 | 计算/通信重叠率 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 批量尾传 + 散乱单块直发** | 等 80 层算完，按块发 16,000 个 WQE | **`148.0 ms` (严重断层! ✕)** | `35.8%` ✕ | `16,000` ✕ | `0.0%` ✕ |
| **Tier 2: 批量尾传 + 连续缓冲聚合** | 等 80 层算完，打包后发 80 个 WQE | `58.4 ms` ✕ | `91.2%` | `80` | `0.0%` ✕ |
| **Tier 3: 逐层流水线推送 (未做块聚合)** | 边算边发，但每层仍发 200 个散乱 WQE | `38.0 ms` | `42.5%` ✕ | `16,000` ✕ | `74.3%` |
| **Tier 4: 逐层流水线推送 + 双缓冲连续打包 (推荐)** | **逐层事件触发 + 单层 1 个连续大 WQE** | **`4.2 ms` (`35.2x` 🏆)** | **`94.6%` 🏆** | **`80` (`200x` 精简 🏆)** | **`92.8%` 🏆** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/llm_disaggregated_pd_layerwise_pipelined_kv_transfer_strided_block_packing/run.py`](../../experiments/llm_disaggregated_pd_layerwise_pipelined_kv_transfer_strided_block_packing_selection/run.py) · 封存数据：[`results.json`](../../experiments/llm_disaggregated_pd_layerwise_pipelined_kv_transfer_strided_block_packing_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/llm-disaggregated-pd-layerwise-pipelined-kv-transfer-strided-block-packing-sizer.html`](../reports/llm-disaggregated-pd-layerwise-pipelined-kv-transfer-strided-block-packing-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/llm-disaggregated-pd-layerwise-pipelined-kv-transfer-strided-block-packing-selection-cards.html`](../reports/llm-disaggregated-pd-layerwise-pipelined-kv-transfer-strided-block-packing-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Bulk + 16,000 Small WQEs", "exposed_stall_ms": 148.0, "nic_util_pct": 35.8 },
      { "tier": "Tier 2: Bulk + Coalesced Slab", "exposed_stall_ms": 58.4, "nic_util_pct": 91.2 },
      { "tier": "Tier 3: Layerwise (Uncoalesced)", "exposed_stall_ms": 38.0, "nic_util_pct": 42.5 },
      { "tier": "Tier 4: Layerwise + Coalesced Pack", "exposed_stall_ms": 4.2, "nic_util_pct": 94.6 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "exposed_stall_ms": "Quantity",
    "nic_util_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Layerwise Pipelined KV Push + Coalesced Packing Cuts Exposed PD Handover Stall 35.2x (148ms -> 4.2ms)",
    "subtitle": "Exposed post-prefill KV transfer stall at 16K context (ms) across 4 tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "exposed_stall_ms",
      "color": "tier"
    }
  }
}
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有采用 PD 分离架构（`vLLM V1 KVConnector`, `SGLang + Mooncake`, `Dynamo + NIXL`）的长上下文在线推理服务。
- **⛔ Not for**：等整次 Prefill 全部跑完才开始发 KV Cache，或对每个 16-Token 分页块单独发起一次 RDMA Write。

### 生产核验 Checklist
- [ ] **在每层 Attention 写完 KV 后立即通过 `cudaEventRecord` 触发异步跨层传输**：确保前 $L-1$ 层传输与后续层计算重叠率 $> 90\%$。
- [ ] **启用 GPU 侧双缓冲连续打包 Kernel（`staging_buf[2]`）**：将单层离散的 Paged Blocks 聚合成 1 个大块连续 RDMA Write，杜绝万级小 WQE 门铃风暴。
- [ ] **监控 `Token-1 -> Token-2`（首字到次字）交接延迟**：确保 `16K` 上下文下交接附加耗时控制在 `< 10 ms` 以内。
