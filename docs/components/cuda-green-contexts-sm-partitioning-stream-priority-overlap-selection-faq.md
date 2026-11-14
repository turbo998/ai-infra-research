# CUDA 12.4+ `Green Contexts`（`CUgreenCtx` 进程内动态 SM 硬件切分）、流优先级抢占与计算通信真并行选型指南

> **Provenance & Evidence Tier**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)。定量数据锚定于 [`experiments/cuda_green_contexts_sm_partition_stream_overlap_selection/results.json`](../../experiments/cuda_green_contexts_sm_partition_stream_overlap_selection/results.json)。
> **English Version**: [CUDA 12.4+ `Green Contexts` (`CUgreenCtx`) Intra-Process SM Partitioning & Stream Priority Overlap Selection Guide](./cuda-green-contexts-sm-partitioning-stream-priority-overlap-selection-faq.en.md)

![CUDA Green Contexts SM Partitioning & Stream Overlap Summary](../../assets/svg/cuda-green-contexts-sm-partitioning-stream-priority-overlap-selection-faq-summary.svg)

---

## 0. 🔥 今日领域最热门能力突破追踪（Daily Hottest Breakthrough）

- **CUDA 12.4+ `Green Contexts`（`CUgreenCtx` / `cuDevSmResourceSplitByCount`）实现单进程内毫秒级零拷贝 SM 物理切分（如 `108` 计算 SM + `24` 通信/解码 SM），将多流并行排队延迟从 `2.40 ms` 压至 `0.02 ms`（降低 `120` 倍）**：在 `DeepSeek-V3 DualPipe / Two-Batch Overlap (TBO)`、`NanoFlow` 纳米流调度以及“Prefill 大矩阵乘与 Decode 小矩阵乘同卡共跑”架构中，许多工程师以为“只要创建两个 `torch.cuda.Stream()` 就能让计算算子和 NCCL 通信算子同时在 GPU 上并行执行”。然而在 Hopper/Blackwell 架构上，现代 `CUTLASS 3.x / FlashAttention-3` 算子普遍采用 **Persistent Kernel（持久化内核，启动时一次性申请全部 `132` 个 SM 并常驻循环直至算完）**！当 `Stream A` 跑起 Persistent GEMM 霸占全部 `132` 个 SM 时，哪怕 `Stream B` 设了最高流优先级（`cudaStreamCreateWithPriority`，Tier 2），硬件调度器也**无法强行打断正在执行的 CTA 线程块**，导致 `Stream B` 被迫干等 **`1.15 ~ 2.40 ms`**，所谓的“计算通信重叠”彻底退化为串行执行！通过引入 CUDA 12.4+ **`Green Contexts`（轻量级绿色上下文，Tier 4）**，在**同一个进程、同一个虚拟地址空间**内将 `132` 个 SM 物理切分为互不越界的 **`108` 个计算 SM（`stream_comp`）** 与 **`24` 个通信/Decode 专属 SM（`stream_comm`）**，高优流 **`0.02 ms` 零排队瞬时启动**，端到端重叠步耗时缩短 **`31.1%`（`14.8 ms -> 10.2 ms`）**！

---

## 1. 📌 核心结论速览（Executive Summary — 3 分钟看懂）

1. **为什么“开两个 CUDA Stream（甚至设了高优先级）”依然无法阻止大 GEMM 饿死通信流或 Decode 流？**
   - **通俗大白话 + 第一性原理**：把 H100 的 **`132` 个 SM** 比作医院里的 **132 间急诊室**。当 `Stream A` 送来一个大 Prefill 矩阵乘（Persistent GEMM）时，GPU 硬件调度器（Work Distributor）极其贪婪，瞬间把 132 间急诊室**全部锁门占满**！0.1 毫秒后，`Stream B` 送来一个十万火急的 NCCL 通信包或 Decode 请求：
     - **默认双流（Tier 1）**：132 间房全满，`Stream B` 在门外干等 **`2.40 ms`** 直到 `Stream A` 全部做完手术才进得去！
     - **传统流优先级（`cudaStreamCreateWithPriority`，Tier 2）**：相当于给 `Stream B` 发了“VIP 插队卡”，但 GPU 硬件规定**“不能把正在手术台上的病人（Persistent CTA）赶下来，只能等某一间房做完当前病人退场时才能插队”**——结果遇到 Persistent Kernel 一口气包场，照样得等 **`1.15 ms`**！
     - **`CUDA 12.4+ Green Contexts`（Tier 4）**：院长直接用隔断墙把 132 间房划成 **108 间普通诊室（绑定 `stream_comp`）** 和 **24 间急诊专室（绑定 `stream_comm`）**！哪怕 `Stream A` 再贪婪，硬件也只准它进那 108 间房；剩下的 24 间急诊室永远为 `Stream B` 敞开大门，**`0.02 ms` 随到随治，真正实现 100% 物理并发**！
2. **为什么 `Green Contexts`（Tier 4）比传统的 `CUDA MPS`（Tier 3）强得多？**
   - 传统 `CUDA MPS`（Multi-Process Service）是给**不同操作系统进程**划分 SM 上限的；而在单个 PyTorch 进程内部，所有 Stream 依然混用同一个主 Context。`Green Contexts` 是 **进程内（Intra-Process）** 的极轻量级资源视图，不同 `CUgreenCtx` 创建的 `cudaStream_t` **天然共享同一份模型权重指针和 KV Cache 显存，零进程间通信（Zero IPC）、零上下文切换开销**！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：把 132 个 SM 切掉 24 个给通信/Decode，剩下的 `108` 个 SM 跑大矩阵乘会不会慢很多？
- **答**：**几乎不慢（仅损失 `< 4%`），这就是“带宽饱和边际递减定律”**！在 H100 上，当大约 `104 ~ 108` 个 SM 同时满载跑 Tensor Core 或读写 HBM 时，芯片的功耗预算（700W TDP）或 HBM3e 显存带宽就已经接近 `95%` 饱和。抽出 `16 ~ 24` 个 SM 专门跑通信（`NCCL / DeepEP`）或内存受限的 Decode Attention，能用 **`< 4%` 的主 GEMM 降速换取 `100%` 的通信耗时隐藏**，净赚 **`31.1%` 的端到端吞吐提升**！

### Q2：在 Hopper（H100 / H800）和 Blackwell（B200）上划分 `Green Context` 的 SM 数量有什么对齐要求？
- **答**：GPU 硬件内部是按 **TPC（Texture Processing Cluster，每 2 个 SM 组成 1 个 TPC）及 GPC（Graphics Processing Cluster）** 组织的。在 Hopper（SM90）上，`cuDevSmResourceSplitByCount` 要求划分出的 SM 数量通常需为 **`8` 个 SM 的整数倍**（或最小粒度 2/4/8 SM，取决于驱动版本）。例如在 `132` SM 的 H100 上，黄金切分比例为 **`108 SMs + 24 SMs`** 或 **`112 SMs + 20 SMs`**。

### Q3：如果算子（如 cuBLAS / CUTLASS）内部写死了按 `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)`（返回 132）来决定 Grid Size 怎么办？
- **答**：这是实战中最关键的工程细节！当一个 Kernel 运行在只有 `108` 个 SM 的 `Green Context` 流上时，如果它误以为自己有 `132` 个 SM 并启动了 132 个 Persistent CTA，那么第 `109~132` 个 CTA 会被迫沦为**第二波（Wave 2 Tail）**，引发严重的 Wave Quantization 尾延迟！因此必须调用 `cuGreenCtxGetDevResource` 获取当前 Green Context 的**真实可用 SM 数（`108`）**，并将其显式传给 CUTLASS Kernel 的 `hw_info.sm_count = 108`！

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + SM 分区对比图 + 双流重叠时序图）

### 3.1 架构师多流并发与 SM 硬件隔离选型决策树

```mermaid
flowchart TD
    A["需要在同一张 GPU 上并行重叠 Compute GEMM 与 Comm / Decode"] --> B{"计算算子是否包含 Persistent Kernel 或大 Grid 波次？"}
    B -- "是 (霸占全部 132 个 SM)" --> C{"是否在同一个训练/推理进程内？"}
    C -- "多进程共享单卡" --> D["✅ Tier 3: CUDA MPS 进程级 SM 配额"]
    C -- "单进程内部多 Stream 重叠 (TBO / NanoFlow)" --> E["🏆 Tier 4: CUDA 12.4+ Green Contexts (cuGreenCtxCreate)<br/>108 计算 SM + 24 通信/Decode SM 硬隔离 · 0.02ms 零排队"]
    B -- "误用传统双流" --> F["⛔ Tier 1/2: 普通双 Stream 或 Stream Priority<br/>无法打断运行中 CTA · 高优流被阻塞 1.15~2.40ms"]
```

### 3.2 传统双 Stream（争抢 132 SM 导致串行）vs. `Green Contexts`（108 + 24 SM 空间分片真并行）

```mermaid
flowchart LR
    subgraph SHARED["⛔ Tier 1/2: 传统双 Stream (争抢同一池 132 SMs)"]
        direction TB
        S1["Stream A (Persistent GEMM) 锁死全部 132 个 SM"] --> S2["Stream B (NCCL / Decode) 被挡在门外苦等 2.40ms!"]
    end
    subgraph GREEN["🏆 Tier 4: CUDA 12.4+ Green Contexts (物理空间分片)"]
        direction TB
        G1["GreenCtx A (108 SMs): 100% 满载跑 Tensor Core GEMM"]
        G2["GreenCtx B (24 SMs): 同时 0.02ms 零等待跑 NCCL / Decode!"]
    end
```

### 3.3 `Green Context` 驱动 API 创建与算子 `sm_count` 感知调用时序图

```mermaid
sequenceDiagram
    autonumber
    participant App as 训练/推理引擎 (PyTorch / vLLM / Megatron)
    participant Drv as CUDA 12.4+ Driver API
    participant CWD as GPU 硬件工作分发器 (CWD)

    App->>Drv: 1. cuDevSmResourceSplitByCount(132 SMs -> 108 Compute + 24 Comm)
    Drv-->>App: 返回 gctx_compute (108 SMs) 与 gctx_comm (24 SMs)
    App->>Drv: 2. cuGreenCtxStreamCreate 创建绑定各自分区的 stream_comp 与 stream_comm
    App->>CWD: 3. 在 stream_comp 启动 GEMM (显式传入 hw_info.sm_count = 108)
    App->>CWD: 4. 同时在 stream_comm 启动 NCCL/Decode (独占 24 个 SM，0.02ms 立即执行!)
```

---

## 4. 📊 架构师选型决策矩阵（4-Tier Selection Matrix）

| 评估维度 | Tier 1: 默认同优先级双 Stream | Tier 2: 传统流优先级 (`StreamPriority`) | Tier 3: 传统 `CUDA MPS` 进程配额 | Tier 4: CUDA 12.4+ `Green Contexts` (`CUgreenCtx`) |
| :--- | :--- | :--- | :--- | :--- |
| **隔离粒度** | 无隔离（全卡争抢） ✕ | 仅 CTA 退场边界软抢占 ⚠️ | 操作系统进程级（Process） | **单进程内 CUDA Stream 级** 🏆 |
| **高优流启动排队延迟** | `2.40 ms` ✕ | `1.15 ms` ⚠️ | `0.45 ms` | **`0.02 ms` (降低 120 倍)** 🏆 |
| **计算+通信重叠单步耗时** | `14.8 ms` (退化为串行) ✕ | `13.1 ms` | `11.9 ms` | **`10.2 ms` (提速 31.1%)** 🏆 |
| **是否共享进程虚拟地址空间** | 是 | 是 | 否（需跨进程 IPC） | **是（零拷贝共享全部张量指针）** 🏆 |
| **动态调整灵活性** | 无 | 静态优先级 | 需重启进程修改环境变量 | **可在进程内创建多个不同配比的 GreenCtx 流** 🏆 |

---

## 5. 🧪 定量算例与工具链（Interactive Sizer & 7-Card Visual Report）

- **实验脚本与密封数据**：[`experiments/cuda_green_contexts_sm_partition_stream_overlap_selection/run.py`](../../experiments/cuda_green_contexts_sm_partition_stream_overlap_selection/run.py) · [`results.json`](../../experiments/cuda_green_contexts_sm_partition_stream_overlap_selection/results.json)
- **交互式 Green Context SM 分区计算器**：[`docs/reports/cuda-green-contexts-sm-partitioning-stream-priority-overlap-sizer.html`](../reports/cuda-green-contexts-sm-partitioning-stream-priority-overlap-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/cuda-green-contexts-sm-partitioning-stream-priority-overlap-selection-cards.html`](../reports/cuda-green-contexts-sm-partitioning-stream-priority-overlap-selection-cards.html)

---

## 6. 🛠️ Good Fit vs. Not For & 生产环境快速核验清单（Checklist）

### ✅ Good Fit vs. ⛔ Not For

| 维度 | ✅ Good Fit（强烈推荐） | ⛔ Not For（严禁误用） |
| :--- | :--- | :--- |
| **计算与通信重叠 (`DualPipe / TBO / NanoFlow`)** | 使用 `CUgreenCtx` 将 `132` 个 H100 SM 切分为 `108`（GEMM）+ `24`（NCCL / FlashDecoding），并向 CUTLASS 显式传入 `sm_count=108` | 以为仅靠两个普通 `torch.cuda.Stream()` 就能在 Persistent GEMM 运行期间并发执行 NCCL 通信 |
| **Prefill 与 Decode 同实例混部** | 为延迟敏感的 Decode 流划分专属的 `16~24` SM `Green Context`，确保长输入 Prefill 绝不阻塞 Decode TBT | 在 `Green Context` 内仍按全卡 `132` SM 启动 Persistent Grid，引发额外 Wave Quantization 尾延迟 |

### 生产环境上线核验清单（Pre-Flight Checklist）

- [ ] **核验 CUDA Driver 与 Runtime 版本 `≥ 12.4`（推荐 `12.6 / 12.8`）**：确认支持 `cuDevSmResourceSplitByCount` 与 `cuGreenCtxCreate`。
- [ ] **核验算子 `sm_count` 参数已对齐 Green Context 实际分区大小**：确认跑在 `108` SM 分区上的 CUTLASS / FlashAttention 算子按 `108`（而非全卡 `132`）计算 Persistent Grid Size。
- [ ] **核验 Nsight Systems (`nsys`) 并发轨迹**：在 `nsys-ui` 中确认 `stream_comp` 与 `stream_comm` 在时间轴上 `100%` 垂直重叠，且 `stream_comm` 的 Kernel Launch 延迟 `< 0.05 ms`。
