# 多 LoRA 推理服务 CPU-GPU 适配器换页（`vLLM max_loras vs max_cpu_loras / S-LoRA`）主计算流同步阻塞（`64.0 ms` Decode 冻结）与预置准入异步 DMA（`Stage-Before-Admit`）选型：为什么当 1 个新请求命中冷门 `r=64` LoRA（`320 MB`）时，在主计算流上换入权重会让同批次正在流式吐字的 `255` 个正常请求集体卡死 `64.0 ms`（P99 TPOT 飙升 `5.5x`）？基于锁页主机缓存（`max_cpu_loras`）+ 独立 `lora_h2d_stream` 预置就绪准入 + 亲和分组（Decode 停顿从 `64.0 ms` 缩至 `0.08 ms`，降低 `800x`）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/llm_multi_lora_cpu_gpu_paging_default_stream_stall_async_copy_selection/run.py` 与 `results.json`。

![Multi-LoRA CPU-to-GPU Adapter Paging & Stage-Before-Admit Visual Summary](../../assets/svg/llm-multi-lora-cpu-gpu-paging-default-stream-stall-async-copy-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向高密度多租户 LoRA 共享推理池（`vLLM LRUCacheWorkerLoRAManager / SGLang Multi-LoRA / S-LoRA`）的“三级存储分层（GPU 常驻槽位 `max_loras=8` + `cudaHostAlloc` 锁页内存池 `max_cpu_loras=64` + 异步磁盘预取）+ 独立 `lora_h2d_stream` 预置就绪准入状态机（`Stage-Before-Admit`: 权重未完成异步 PCIe DMA 前绝不提前塞入 Running Decode Batch 阻塞热请求）+ 固定显存指针原地更新保活 `CUDAGraph`”全链路架构**。
- **架构师价值**：在多租户企业大模型平台中，常在一台 8 卡 70B 底座模型上同时托管 `100 ~ 500` 个不同业务线的专属 `LoRA` 适配器（当 `rank = 64` 且覆盖全部 7 个线性层时，单个 LoRA 权重高达 **`320 MB`**）。由于 GPU 显存极其宝贵，通常只在 HBM 中预留 `max_loras = 8` 个活跃 GPU 槽位（占 `2.56 GB`）。
  然而，一旦线上不同业务线的请求交替到达（LoRA Churn），很多集群的流式生成延迟 `P99 TPOT` 会从平稳的 `14.2 ms` 突然飙升到 **`78.20 ms`（恶化 `5.5 倍`）**！排查原因会发现三大调度与内存传输深坑：
  1. **深坑一：把需要“冷门 LoRA”的新请求直接塞进正在高速运转的 Running Decode Batch，并在主计算流（`compute_stream`）上同步加载 `320 MB` 权重，导致车上原有的 `255` 个正常用户在原地干等 `64.00 ms`（队头阻塞 Head-of-Line Decode Freeze，Tier 1）**！
     - 设想 GPU 上正有 255 个请求（使用的是已经在 GPU 槽位 `0..6` 里的热 LoRA）在欢快地每 `14.2 ms` 吐一个字。
     - 突然，第 256 个请求到达，它指定要用 `LoRA #42`（此时不在 GPU 的 8 个槽位里，属于 **GPU LoRA Miss**）！
     - 朴素调度器直接把第 256 个请求合并进了本步的执行批次，然后在跑第 0 层前向之前，在当前默认的 `compute_stream` 上调用 `.to("cuda")` 从普通可分页内存（Pageable RAM，甚至磁盘）加载这 `320 MB` 权重！
     - **灾难后果**：可分页内存无法走硬件直接 DMA，CPU 驱动必须同步介入分块拷贝（吞吐仅 `5.0 GB/s`，耗时 **`64.00 ms`**），且因为拷贝排在 `compute_stream` 的最前面，**后面原本 100% 准备就绪的 255 个热请求被迫集体停工陪绑等待整整 `64.00 ms`**！
  2. **深坑二：就算配了锁页内存缓存 `max_cpu_loras=64`（Pinned Host RAM，DMA 带宽达 `27.8 ~ 54.0 GB/s`），如果依然在“准入当步”让 `compute_stream` 同步等待拷贝完成（Tier 2），255 个热请求还是会被硬生生卡住 `11.50 ms`（TPOT 接近翻倍）**！
  3. **深坑三：如果运行队列里同时接纳了 16 个不同 LoRA 的请求，而 GPU 只有 `max_loras = 8` 个槽位，FIFO 轮询调度会导致每一步 Decode 都发生 PCIe 换入换出抖动（Thrashing）**！
  采用 **“预置就绪后再准入（Stage-Before-Admit）”异步状态机（当新请求发生 GPU LoRA Miss 时，先将其置为 `STAGING_LORA` 挂起态，在独立的后台 `lora_h2d_stream` 上从 Pinned CPU 池异步发起 `320 MB` 的 PCIe DMA，同时放行原有的 255 个热请求在 `compute_stream` 上以 `0.08 ms` 零等待全速跑完当前步；等下一步非阻塞 `cudaEventQuery(lora_ready_evt)` 确认 DMA 完毕后，再将该请求无缝并入 Running Batch）+ 活跃批次 LoRA 种类上限门控（`|Active_LoRAs| <= max_loras`）（Tier 4）**，可将冷 LoRA 换页对在线 Decode 的干扰从 `64.00 ms` 压降至 **`0.08 ms`（降低 `800.0 倍`）**，`P99 TPOT` 稳如泰山保持在 **`14.28 ms`**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：为什么“Stage-Before-Admit（预置就绪后再准入）”能把 `11.5 ~ 64.0 ms` 的 PCIe 换页延迟从 Decode 关键路径上 `100%` 抹除？**
   - 在现代 GPU（H100 / B200）中，**PCIe H2D 硬件拷贝引擎（Copy Engine, CE）与 SM 计算核心（Compute Engine）是完全物理正交的两套硬件单元**！
   - 只要满足以下两个条件，PCIe 搬运 `320 MB` 新 LoRA 权重的过程就不会阻塞 SM 跑当前批次的 Decode：
     1. **源内存必须是锁页内存（Pinned Memory，`cudaHostAlloc`）**：使 PCIe Copy Engine 能够脱离 CPU 独立发起全速 `54.0 GB/s` DMA；
     2. **调度器实施一拍延迟准入（1-Step Deferred Admission）**：
        - 在 Step $t$ 调度时，发现新请求 $R_{\text{new}}$ 需要冷门 `LoRA #42`：调度器分配空闲（或 `ref_count == 0` 的 LRU）GPU 槽位 `slot = 7`，在专用 **`lora_h2d_stream`** 上启动异步拷贝并打下 `lora_ready_evt[7]`，但 **本步（Step $t$）暂不把 $R_{\text{new}}$ 放进 `compute_stream` 的执行列表**；
        - 在 Step $t$ 的那 `14.2 ms` 里，`compute_stream` 全速计算原有 255 个请求的 Decode，而 PCIe Copy Engine 同时在 `lora_h2d_stream` 上用 `~6~11 ms` 把 `LoRA #42` 搬进 `slot = 7`！
        - 到了 Step $t+1$（`14.2 ms` 后），调度器调一次 `lora_ready_evt[7].query()`（返回 `True`，耗时 `< 2 µs`），此时 `LoRA #42` 已经 `100%` 在 HBM 就位，$R_{\text{new}}$ 零等待直接上车！
2. **如何在频繁更换 GPU 槽位里的 LoRA 权重时，不破坏已录制的 `CUDAGraph`？**
   - 预分配形状固定的全局常驻槽位张量池 `gpu_lora_A[max_loras, L, r, D_in]` 与 `gpu_lora_B[max_loras, L, D_out, r]`；
   - 每次换入新 LoRA 时，**绝不调 `torch.empty` 重新分配张量，而是向固定槽位 `gpu_lora_A[slot]` 执行原地拷贝 `.copy_(pinned_cpu_A, non_blocking=True)`**！因为显存基地址指针分毫无变，所有录制好的 `SGMV / BGMV` `CUDAGraph` `100%` 永久有效！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：如果一个新来的 LoRA 连 Pinned CPU 池（`max_cpu_loras = 64`）里都没有，需要从磁盘/对象存储（S3 / NFS）拉取，应该怎么处理才不会卡住主进程？
- **严禁在调度器主线程里同步调 `safetensors.load_file()` 读盘**！
  将适配器状态分为三级：
  - **L1（GPU HBM 槽位，`max_loras=8`）**：直接参训/推理；
  - **L2（CPU Pinned RAM 槽位，`max_cpu_loras=64`，占 `20 GB` 锁页内存）**：通过 `lora_h2d_stream` 1 步（`~6 ms`）异步升入 L1；
  - **L3（NVMe / 远端存储）**：若发生 L2 Miss，由独立的 **后台磁盘 I/O 线程池** 将文件读入 L2 Pinned 槽位，期间该请求在等待队列保持 `LOADING_FROM_DISK` 状态，完全不触碰调度主循环与 GPU 流！

### Q2：如果某个请求用的是 `r = 16` 的小 LoRA，另一个请求用的是 `r = 64` 的大 LoRA，固定槽位池 `gpu_lora_A[max_loras, ...]` 怎么兼容不同的 `rank`？
- 按最大允许秩 `max_lora_rank = 64` 统一预分配 GPU 槽位张量：当加载一个 `r = 16` 的小适配器时，只需将其拷贝进槽位的前 16 行 `gpu_lora_A[slot, :, :16, :]`（并将剩余 `16..64` 行置零，或在 `Punica / FlashInfer SGMV` 算子中根据每个槽位的实际 `rank_table[slot] = 16` 动态控制内积循环长度），既零碎片复用同一个槽位池，又不会多算无效的零元素！

### Q3：怎么防止在并发请求极多时发生“GPU 槽位颠簸（Slot Thrashing）”？
- **实施“活跃 LoRA 集合粘性准入（Active-Set Sticky Admission）”**：
  只要当前 GPU 槽位 `slot` 中仍有正在生成的未完结请求（`slot.active_seq_count > 0`），该槽位就被标记为 **`PINNED_IN_USE`（不可驱逐）**。当 8 个槽位全部有活跃请求时，等待队列中属于这 8 个已加载 LoRA 的后续请求优先插队合并执行（Affinity Batching），直到某个 LoRA 的在途请求全部生成完毕（`active_seq_count == 0`），才将其槽位安全置换给下一个排队的 LoRA！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 多 LoRA 推理服务 CPU-GPU 换页与调度准入选型决策树

```mermaid
flowchart TD
    A["多 LoRA 推理服务收到请求 R_256 (需冷门 LoRA #42, 320 MB, 此时 255 个热请求正在 Decode)"] --> B{"CPU 侧 LoRA 权重存储在哪类内存中?"}
    B -- "普通可分页内存 (Pageable RAM) 或每次从磁盘读" --> C["⛔ Tier 1: 无法走直接硬件 DMA + 在 compute_stream 同步拷贝 -> 255 个热请求集体卡死 64.00 ms!"]
    B -- "预加载在 cudaHostAlloc 锁页内存池 (max_cpu_loras=64)" --> D{"PCIe H2D 拷贝在哪个 CUDA Stream 执行, 以及何时准入 R_256?"}
    D -- "本步立即准入 R_256, 在 compute_stream 上先拷贝再跑第 0 层" --> E["⛔ Tier 2: 255 个热请求仍被迫在每层前干等 PCIe 传输 11.50 ms (TPOT 恶化至 25.7ms)!"]
    D -- "在异步拷贝流逐层抢跑, 但未限制同批活跃 LoRA 种类数" --> F["⚠️ Tier 3: 当并发 LoRA 数 > max_loras 时每步发生换页抖动, 仍残留 2.40 ms 卡顿"]
    D -- "Stage-Before-Admit: 后台 lora_h2d_stream 异步预置 + 下一拍 cudaEventQuery 准入 + 亲和分组" --> G["🏆 Tier 4: 在线 Decode 停顿仅 0.08 ms (降低 800 倍) + P99 TPOT 14.28 ms + 100% 兼容 CUDAGraph!"]
```

### 3.2 主计算流同步换入（`64.00 ms` 冻结）vs. `Stage-Before-Admit` 后台异步 DMA（`0.08 ms`）对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: 先把冷请求拉上车再同步搬权重 (255 个热请求集体陪绑干等 64ms!)"]
      B1["Step t: R_256 未预置权重就直接并入 Running Batch"] --> B2["compute_stream 被 320MB LoRA 拷贝堵死 64.00 ms!"]
      B2 --> B3["车上 255 个热请求 P99 TPOT 瞬间飙至 78.20 ms (5.5x 尖刺)!"]
    end
    subgraph Good["🏆 Tier 4: Stage-Before-Admit 后台异步 DMA + 原地槽位覆盖"]
      G1["Step t: R_256 保持 STAGING 态; lora_h2d_stream 后台 DMA 拷贝 320MB 到空闲槽位 #7"] --> G2["同时 compute_stream 零等待跑完 255 个热请求的第 t 步 Decode (14.2 ms)!"]
      G2 --> G3["Step t+1: EventQuery 确认槽位 #7 就绪 -> R_256 零停顿 (0.08 ms) 并入批次!"]
    end
```

### 3.3 `compute_stream` 与 `lora_h2d_stream` 双流正交解耦的 `Stage-Before-Admit` 时序图

```mermaid
sequenceDiagram
    participant Sched as LoRA-Aware Scheduler
    participant H2D as Background lora_h2d_stream (PCIe DMA)
    participant Comp as Main compute_stream (255 Active Reqs)

    Sched->>Sched: 1. Step `t`: Req #256 needs cold `LoRA #42` -> allocate idle GPU `slot = 7`, set `STAGING_LORA`
    Sched->>H2D: 2. Launch `gpu_slot[7].copy_(pinned_cpu_lora[42], non_blocking=True)` + `record(evt[7])`
    Sched->>Comp: 3. Simultaneously launch Step `t` Decode for 255 warm requests (0.0 ms wait on `H2D`!)
    Note over H2D,Comp: PCIe Copy Engine transfers 320 MB in parallel while SMs execute Step `t` Decode!
    H2D-->>Sched: 4. Step `t+1`: `evt[7].query() == True` (0.08 ms) -> transition Req #256 to `RUNNING` on `slot = 7`!
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | 多 LoRA CPU-GPU 换页与准入方案 | 冷命中时在线 Decode 卡顿 | 换页期 P99 `TPOT` | 有效 PCIe 传输带宽 | `CUDAGraph` 兼容性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 可分页内存 + 主计算流同步加载** | 立即准入，在 `compute_stream` 读普通内存 | **`64.00 ms` (全员冻结! ✕)** | `78.20 ms` (`5.5x` ✕) | `5.0 GB/s` ✕ | **不兼容 ✕** |
| **Tier 2: 锁页内存 (`max_cpu_loras`) + 主流拷贝** | 走 DMA，但主流仍同步等待拷贝完才开跑 | `11.50 ms` ✕ | `25.70 ms` ✕ | `27.8 GB/s` | 兼容 (原地写) |
| **Tier 3: 异步流逐层抢跑 (无活跃集上限控制)** | 异步拷贝，但并发种类超限时每步颠簸 | `2.40 ms` | `16.60 ms` | `51.2 GB/s` | 兼容 |
| **Tier 4: 锁页池 + `Stage-Before-Admit` + 亲和分组 (推荐)** | **后台异步 DMA 就绪后再上车 + 原地写槽位** | **`0.08 ms` (`800x` 更低 🏆)** | **`14.28 ms` (零尖刺 🏆)** | **`54.0 GB/s` 🏆** | **100% 原生兼容 🏆** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/llm_multi_lora_cpu_gpu_paging_default_stream_stall_async_copy/run.py`](../../experiments/llm_multi_lora_cpu_gpu_paging_default_stream_stall_async_copy_selection/run.py) · 封存数据：[`results.json`](../../experiments/llm_multi_lora_cpu_gpu_paging_default_stream_stall_async_copy_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/llm-multi-lora-cpu-gpu-paging-default-stream-stall-async-copy-sizer.html`](../reports/llm-multi-lora-cpu-gpu-paging-default-stream-stall-async-copy-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/llm-multi-lora-cpu-gpu-paging-default-stream-stall-async-copy-selection-cards.html`](../reports/llm-multi-lora-cpu-gpu-paging-default-stream-stall-async-copy-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Pageable + Compute Stream", "decode_stall_ms": 64.00, "p99_tpot_ms": 78.20 },
      { "tier": "Tier 2: Pinned + Compute Stream", "decode_stall_ms": 11.50, "p99_tpot_ms": 25.70 },
      { "tier": "Tier 3: Layerwise Async (Unbounded)", "decode_stall_ms": 2.40, "p99_tpot_ms": 16.60 },
      { "tier": "Tier 4: Stage-Before-Admit + Affinity Group", "decode_stall_ms": 0.08, "p99_tpot_ms": 14.28 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "decode_stall_ms": "Quantity",
    "p99_tpot_ms": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Pinned Host LRU + Stage-Before-Admit Async DMA Cuts Cold-LoRA Decode Stall 800x (64ms -> 0.08ms)",
    "subtitle": "Active decode batch stall on cold r=64 LoRA miss (ms) across 4 tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "decode_stall_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有托管数十至数百个 LoRA 适配器的多租户推理池（`vLLM`, `SGLang`, `S-LoRA`），尤其是大秩（`r = 32..128`）全线性层适配器场景。
- **⛔ Not for**：在主计算流上直接从可分页内存/磁盘加载冷 LoRA，或在权重完成 H2D 拷贝前就将新请求并入活跃 Decode 批次。

### 生产核验 Checklist
- [ ] **配置充足的锁页内存二级缓存（`max_cpu_loras >= 4 * max_loras`）并将适配器张量显式 `.pin_memory()`**。
- [ ] **实施 `Stage-Before-Admit` 异步预置状态机**：在独立 `lora_h2d_stream` 上发起非阻塞拷贝，通过 `cudaEventQuery` 确认就绪后再将请求选入 Running Batch。
- [ ] **使用固定地址槽位原地 `.copy_()` 更新权重，并对活跃批次实施 `|Active_LoRAs| <= max_loras` 粘性准入防颠簸保护**。
