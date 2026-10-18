# Prefill/Decode 分离架构与跨节点 KV Cache 传输/分层缓存路由（Mooncake、Dynamo/NIXL 与 SGLang HiCache 源码级深挖）

面向解决方案架构师（Solution Architect），本篇组件文档基于 **Mooncake (`kvcache-ai/Mooncake` commit `7ff6bf7f72f15aba7289bc25c87cf491e04efc6f`)**、**NVIDIA Dynamo NIXL (`ai-dynamo/nixl` commit `c40f1b29b4989765385059196fadcc7bedc37ffb`)** 与 **SGLang (`sgl-project/sglang` commit `4ec19ca27f08f7ce44bbe1da70ce964731940eb2`)** 的一手 C++/Python 源码，系统拆解 Prefill/Decode (PD) 分离架构中跨节点 KV Cache 零拷贝传输、步长描述符压缩、截止时间感知（Deadline-Aware）网卡仲裁、三级分层缓存（L1 HBM / L2 Host DRAM / L3 NVMe & Remote Store）重叠路由与异常中止（Abort Drain）安全边界，并配套可复现标准库实验 `experiments/disaggregated_kv_routing/run.py`（`own-offline`）。

---

## 1. 工程问题定义：为什么 PD 分离的瓶颈会从算力转移到“描述符展开 + 网卡队头阻塞 + 尾块放大”？

在长上下文（`32K–128K` tokens）与多轮 Agent 工作负载下，将计算密集的 Prefill 节点与访存带宽密集的 Decode 节点解耦，能够消除 Prefill 注入对 Decode TPOT（Time Per Output Token）的干扰。但在工程落地上，跨节点 KV Cache 传输面临三大底层系统瓶颈：

1. **Paged KV 碎片化与描述符爆炸（Descriptor Explosion）**：PagedAttention 将 KV Cache 切分为固定大小的非连续页（如 `16` 或 `64` tokens/page），且每一层（`num_layers = 61..80`）独立存储。一次 `33K` tokens 的 `Llama-3-70B` 请求在 `16 tokens/page` 下跨越 `2,063` 个页 $\times$ `80` 层 = **`165,040` 个离散内存块**。若逐块构造 RDMA SGE/WR 描述符并执行区间合并，仅 CPU 侧描述符准备延迟即可达毫秒级。
2. **多请求并发下的网卡队头阻塞（Head-of-Line Blocking on NIC）**：当多个 Prefill/Decode KV 传输流共享同一张 `400GbE`（有效约 `388 Gbps`）网卡时，按 FIFO 顺序排队或按流均分带宽（Equal-Split）会导致携带紧迫 TTFT SLO（如 `1.5 ms` 语音/交互 Agent 增量 KV）的小流被 `100+ MiB` 的长文档 RAG KV 传输拖垮。
3. **跨层前缀缓存（L1/L2/L3 HiCache）与远端 Prefill 传输的竞态与回退悖论**：当 Decode 节点在本地 L2 Host DRAM 或 L3 NVMe 存储中命中历史前缀时，如何与远端 Prefill 的增量计算 + RDMA WRITE 并行重叠？何时从 L3 存储加载反而比远端重算更慢？

```
+---------------------------------------------------------------------------------------------------+
|                     Disaggregated Prefill -> Decode KV Transfer & 3-Tier HiCache                  |
|                                                                                                   |
|  [Prefill Node]                                                    [Decode Node]                  |
|  Prompt [0 : T]                                                    DecodePrefixMatch:             |
|   |-- Prefix [0 : L_dec] (Skipped on Prefill!)                      |-- L1 GPU HBM  [0 : L1]      |
|   +-- Suffix [L_dec : T] ---> Compute Suffix KV                     |-- L2 Host DRAM [L1 : L1+L2] |
|                                  |                                  +-- L3 Storage   [L1+L2 : L_d]|
|                                  v                                         |                      |
|                    +---------------------------+                           v                      |
|                    | NIXL nixlStrideDesc       |               Local Restore Branch:              |
|                    | or Mooncake planRdmaSlices|               L3 -> L2 Prefetch (if L3 > 0)      |
|                    | + OrderByUrgency (MLU)    |               L2 -> L1 PCIe/C2C load_back        |
|                    +-------------+-------------+                           |                      |
|                                  | RDMA WRITE (Multi-Rail)                 |                      |
|                                  +--------------------+--------------------+                      |
|                                                       |                                           |
|                                                       v                                           |
|                                      HiCacheRestoreGatedKVReceiver.poll()                         |
|                              Wait for BOTH: (RDMA KVPoll.Success) AND (HiCache READY)             |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. 一手源码深挖：NIXL、Mooncake 与 SGLang 的核心实现机制

### 2.1 NVIDIA Dynamo NIXL：统一五段内存抽象与 `nixlStrideDesc` 步长描述符压缩

在 `ai-dynamo/nixl`（commit `c40f1b29b4989765385059196fadcc7bedc37ffb`）中，NIXL 为分离式推理定义了统一的跨介质传输抽象：

- **五种内存段类型（`src/api/cpp/nixl_types.h:47-55`）**：
  `enum nixl_mem_t { DRAM_SEG, VRAM_SEG, BLK_SEG, OBJ_SEG, FILE_SEG }`，将 GPU HBM (`VRAM_SEG`)、主机内存 (`DRAM_SEG`)、块设备 (`BLK_SEG`)、对象存储 (`OBJ_SEG`) 与文件系统 (`FILE_SEG`) 纳入同一套描述符列表 `nixlDescList<T>`。
- **步长描述符 `nixlStrideDesc`（`src/api/cpp/nixl_descriptors.h:148-192` & `src/api/cpp/nixl.h:201-237`）**：
  - 基础描述符 `nixlBasicDesc`（`nixl_descriptors.h:32-115`）包含 `(uintptr_t addr, size_t len, uint64_t devId)`（24 字节）。
  - 为避免对每个 `(layer, page)` 块逐一展开 `nixlBasicDesc`，NIXL 引入 `struct nixlStrideDesc : public nixlBasicDesc { size_t stride; size_t count; }`，并在 `nixlAgent::prepXferDlist(agent_name, const nixl_stride_dlist_t &descs, dlist_hndl)`（`nixl.h:201-237`）中直接消费步长描述符：
    - 只要区间 `[addr, addr + (count - 1) * stride + len)` 位于同一个已注册 Memory Region (MR) 内，单个 `nixlStrideDesc` 即可表达 `count` 个相距 `stride` 字节、每块长 `len` 字节的规则 KV 布局，**无需展开为 `count` 个 `nixlBasicDesc` 再做 $O(N)$ 相邻块合并**。
- **PCIe Switch BAR1 与 Grace-Blackwell C2C 拓扑分流（`src/utils/libfabric/libfabric_topology.h:99-121, 314-367`）**：
  - `nicSharesPcieSwitchWithAccel(efa_device)` 通过 `hwloc` 检查网卡与加速器是否共享同一颗 PCIe Switch（经典 x86 + PCIe Switch 拓扑下走 GPUDirect RDMA BAR1 窗口）。
  - 在 `GB200`（Grace-Blackwell NVL72）等無独立 PCIe Switch 的架构上，网卡挂载于 Grace CPU Root Complex，GPU HBM 通过 NVLink-C2C（`450 GB/s` 单向 / `900 GB/s` 双向 cache-coherent 互连）映射入主机物理地址空间；`libfabric_topology.h:340-366` 专门沿 `hwloc` 向上回溯至 `Package` 层级（严禁跨越 Socket 边界），精准匹配同封装内的 NIC 与 GPU。

### 2.2 Mooncake Transfer Engine：尾块折叠切分（`planRdmaSlices`）、MLU 截止时间仲裁与多轨熔断器

在 `kvcache-ai/Mooncake`（commit `7ff6bf7f72f15aba7289bc25c87cf491e04efc6f`）的下一代传输引擎 TENT（`mooncake-transfer-engine/tent/`）中，针对多网卡 RDMA 传输实现了四项关键机制：

1. **尾块折叠切片算法 `planRdmaSlices`（`tent/include/tent/transport/rdma/slice.h:69-90`）**：
   给定请求字节数 `length`、对齐基块 `base_block`、最大切片数 `max_slices` 与尾块合并阈值 `merge_ratio = 0.25`：
   $$\text{count}_0 = \min\left(\text{max\_slices}, \left\lceil \frac{\text{length}}{\text{base\_block}} \right\rceil\right), \quad \text{per\_slice} = \left\lceil \frac{\text{length}}{\text{count}_0} \right\rceil$$
   $$\text{block\_size} = \left\lceil \frac{\text{per\_slice}}{\text{base\_block}} \right\rceil \cdot \text{base\_block}, \quad \text{count} = \left\lceil \frac{\text{length}}{\text{block\_size}} \right\rceil$$
   若 $\text{count} > 1$ 且尾块字节数 $\text{tail} = \text{length} - (\text{count} - 1)\cdot \text{block\_size} < \text{merge\_ratio} \cdot \text{block\_size}$，则执行 `--count`，将短尾块直接折叠进倒数第二个切片（最后一个切片长度变为 $\text{block\_size} + \text{tail}$），从而**省去 1 个 RDMA Work Request (WR)、1 个完成队列事件 (CQE) 和 1 次多轨路径选择开销**。
2. **最大链路利用率（MLU）截止时间紧迫度函数 `DeadlineMlu`（`tent/include/tent/runtime/deadline_mlu.h:56-66`）**：
   $$\text{MLU}(\text{bytes\_ahead}, L, t_{\text{dl}}, t_{\text{now}}, B_{\text{bps}}) = \frac{(\text{bytes\_ahead} + L) / B_{\text{bps}}}{(t_{\text{dl}} - t_{\text{now}}) / 10^9}$$
   - 当 $\text{MLU} \le 1.0$ 时，该流在考虑前方已排队字节 $\text{bytes\_ahead}$ 后仍能在截止时间 $t_{\text{dl}}$ 前完成；当 $\text{MLU} > 1.0$（或 $t_{\text{dl}} \le t_{\text{now}}$ 时返回 $+\infty$）则预测违约。
3. **贪心累积排队的紧迫度排序 `OrderByUrgency`（`tent/include/tent/transport/rdma/bw_arbitration.h:77-116`）**：
   - 在每个调度周期对至多 `kGreedySlots = 64` 个槽位进行贪心选择：第 $k$ 步遍历所有未选中的流 $i$，用**当前累积的 `cur_ahead`** 计算 $\text{MLU}$，选出 $\text{MLU}$ 最大的最紧迫流 `best` 放入队首，并立即执行 `cur_ahead += flows[best].length`！
   - **准入控制（Admission）与网卡仲裁（Arbitration）的有意非对称设计（`bw_arbitration.h:19-46`）**：准入控制使用更宽的 `dispatching_bytes_`（包含尚在软件工作队列中的切片）以保守拒绝会导致全局过载的流；而网卡仲裁仅传入硬件 QP 已投递的 `getPostedBytes()` 作为初始 `bytes_ahead`，并在 `OrderByUrgency` 内部累加被提权流的字节数，确保高优小流能精准抢占大流。
4. **逐条目老化提权 `DecidePromotionPerEntry`（`tent/include/tent/transport/rdma/promotion_policy.h:74-89`）与单轨熔断器 `RailMonitor`（`tent/include/tent/transport/rdma/rail_monitor.h:125-147`）**：
   - 旧版 `DecidePromotionHeadOnly` 只检查队首 `enqueue_ts.front()`，一旦队首超时就把整条低优队列全部提权（导致刚入队的新请求搭便车造成优先级反转）；新版 `DecidePromotionPerEntry` 仅提权满足 `current_ts - ts >= promotion_timeout_ns` 的老化条目。
   - `RailMonitor` 对每个 `(local_nic, remote_nic)` 维护滑动窗口（默认 `error_window_ = 10s` 内达到 `error_threshold_ = 3` 次错误即触发熔断），初始冷却 `cooldown_ = 30s` 并按指数退避倍增至 `kMaxCooldown = 300s`，恢复期进入 `half_open` 并由 `probe_in_flight` 布尔锁保证每次仅放行单次探测流。

### 2.3 SGLang：异构状态传输类型、Decode 三级 HiCache 重叠与 Abort 释放安全边界

在 `sgl-project/sglang`（commit `4ec19ca27f08f7ce44bbe1da70ce964731940eb2`）中：

- **异构注意力与混合状态传输分类（`python/sglang/srt/disaggregation/base/conn.py:29-58, 96-119`）**：
  `StateType` 枚举显式区分 `MAMBA`（Mamba 线性状态）、`QSA_PENDING` / `QSA_COMPRESSED`、`SWA` / `SWA_RING`（滑动窗口注意力）、`DSA` / `DSA_TAIL`（DeepSeek 稀疏注意力）、`DSV4_REQUEST_STATE` 以及 `BLOCK_SCALE` / `BLOCK_SCALE_SWA`（FP8/FP4 块级缩放因子），并支持指定传输目标 `KVTransferDestination.DEVICE`（直写 GPU HBM）或 `KVTransferDestination.HOST`（写入主机 DRAM）。
- **Decode 端三级前缀匹配与并发恢复门控（`python/sglang/srt/disaggregation/decode_hicache_mixin.py:26-51, 166-181`）**：
  - `DecodePrefixMatch` 计算：
    $$\text{decode\_prefix\_len} = L_{\text{L1,HBM}} + L_{\text{L2,DRAM}} + L_{\text{L3,Storage}}, \quad \text{restore\_token\_count} = \text{decode\_prefix\_len} - L_{\text{L1,HBM}}$$
  - Prefill 节点仅需计算并通过 RDMA 传输后缀增量区间 $[\text{decode\_prefix\_len}, T_{\text{prompt}})$；与此同时，Decode 节点在本地异步启动 `_start_hicache_prefetch`（L3 $\to$ L2）与 `_try_hicache_queue_load_back`（`python/sglang/srt/mem_cache/l2_transfer.py:55-99`，通过 `storage_backend.ShouldUseZeroCopy()` 选择零拷贝或分批聚合 H2D 拷贝将 L2 Host DRAM 页恢复至 L1 GPU HBM）。
  - `HiCacheRestoreGatedKVReceiver.poll()`（`decode_hicache_mixin.py:166-181`）在 `kv_receiver.poll() == KVPoll.Success` 返回后，必须继续检查 `hicache_restore_status == HiCacheRestoreResult.READY`（若仍在 `IN_PROGRESS` 则返回 `KVPoll.WaitingForInput`），确保远端后缀 RDMA 写入与本地前缀恢复双双就绪后才进入 Decode 迭代。
- **异常中止（Abort）时的内存延迟释放契约（`conn.py:173-198, 250-264`）**：
  当请求因超时或客户端断开被 Abort 时，`is_abort_release_safe()` 明确指出：**轮询失败（failed poll）或本地超时绝不等于远端 Prefill 发起的单边 RDMA WRITE 已经排空（drained）**！若立即释放并复用 Decode 端 HBM/DRAM KV 槽位，飞行中的 RDMA WRITE 会静默覆写新请求的 KV Cache（Silent Data Corruption）。因此必须通过 `supports_deferred_decode_kv_release` 等待 Prefill 端返回 `ABORT_ACK` 确认不再发起或已完成 RDMA 传输后，才可回收 KV 页。

---

## 3. 原创 CPU/Offline 实验结果与反例分析（`experiments/disaggregated_kv_routing/run.py`）

### 3.1 KV 布局、NIXL `nixlStrideDesc` 描述符压缩与 Mooncake `planRdmaSlices` 尾块折叠

实验配置见 `experiments/disaggregated_kv_routing/input.json`，结果摘自 `experiments/disaggregated_kv_routing/results.json`（`own-offline`）：

| Case ID | 架构与精度 | 序列长度 / 页大小 | 单 Rank 传输量 (MiB) | NIXL 展开 `nixlBasicDesc` 数 (耗时 $\mu\text{s}$) | NIXL `nixlStrideDesc` 数 (耗时 $\mu\text{s}$) | 压缩比 | Mooncake 切片数 (`planRdmaSlices`) | 尾块折叠 (`tail_folded`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `llama3_70b_gqa_bf16_32k_tp8` | Llama-3-70B GQA BF16 (`TP=8`, 80L) | `33,000` / `16` | `1,289.062 MiB` | `165,040` (`14,028.40 us`) | `80` (`6.80 us`) | **`2,063x`** | `64` (`block=20.19 MiB`) | `0` |
| `qwen3_32b_gqa_fp8_32k_tp4` | Qwen3-32B GQA FP8 (`TP=4`, 64L) | `33,000` / `16` | `1,031.250 MiB` | `132,032` (`11,222.72 us`) | `64` (`5.44 us`) | **`2,063x`** | `32` (`block=32.25 MiB`) | `0` |
| `deepseek_v3_mla_fp8_32k_tp1` | DeepSeek-V3 MLA FP8 (`TP=1`, 61L) | `32,512` / `64` | `1,089.422 MiB` | `30,988` (`2,633.98 us`) | `61` (`5.18 us`) | **`508x`** | **`15`** (`block=72.00 MiB`, 末片 `81.42 MiB`) | **`1` (16 $\to$ 15)** |

**关键工程发现**：
1. **MLA 架构将单 Token 全模型 KV 降至 `576 B/layer`（`35.14 KiB/token`）**，使得 `DeepSeek-V3` 在 `TP=1` 下传输 `32,512` tokens 的单 Rank KV 体积（`1,089.42 MiB`）甚至低于 `Llama-3-70B` 在 `TP=8` 切分后的单 Rank 体积（`1,289.06 MiB`，全模型达 `10.07 GiB`）。
2. **`nixlStrideDesc` 消除毫秒级 CPU 描述符准备开销**：在 `Llama-3-70B` `16 tokens/page` 下，逐页逐层展开 `165,040` 个 `nixlBasicDesc`（按每描述符 `85 ns` 校验/合并计）耗时 **`14.03 ms`**，而 `nixlStrideDesc` 按层步长压缩为 `80` 个描述符后仅需 **`6.80 us`**（压缩 **`2,063x`**）。
3. **Mooncake `planRdmaSlices` 尾块折叠验证**：在 `deepseek_v3_mla_fp8_32k_tp1`（`1,142,341,632 B`）中，`block_size = 75,497,472 B`（`72 MiB`）。若按常规向上取整需 `16` 个切片，但第 16 片尾块仅 `9,879,552 B`（占 `block_size` 的 `13.09% < 25%` `merge_ratio`），因此被折叠进第 15 片（`85,377,024 B`），消除小尾块 WR。

### 3.2 Mooncake `OrderByUrgency` 截止时间感知网卡仲裁对比

在单张 `388 Gbps`（`48.5 GB/s`）RDMA 网卡上对比 **FIFO 顺序排队**、**多流均分带宽（Equal-Split）** 与 **Mooncake `OrderByUrgency`（`bw_arbitration.h:77-116`）**：

| Case ID | 并发流数 | FIFO 违约流数 (`MLU > 1.0`) | FIFO 最大 MLU | 均分带宽 (`B/N`) 违约流数 | Mooncake `OrderByUrgency` 违约流数 | Mooncake 最大 MLU | 贪心重排结果 (`ordered_flow_ids`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `4flows_388gbps_nic_burst_contention` | `4` | `1` | `4.4970` | `1` | **`0`** | **`0.4324`** | `flow3 (16MB, 1.5ms) -> flow0 -> flow1 -> flow2` |
| `6flows_mixed_pd_kv_transfers_388gbps` | `6` | `3` | `6.3899` | `5` | **`0`** | **`0.6594`** | `f3 (12MB, 0.9ms) -> f4 (20MB, 1.8ms) -> f2 -> f0 -> f1 -> f5` |

- **为什么均分带宽（Equal-Split）在 6 流混合场景下反而造成 `5/6` 违约？** 当 6 条流平分 `388 Gbps`（每流仅 `64.67 Gbps = 8.08 GB/s`）时，既让 `0.9 ms / 1.8 ms / 3.5 ms` 的紧迫中小流因带宽不足超时，又拖慢了 `100 MiB` 大流的排空速度。Mooncake `OrderByUrgency` 通过贪心累积 `cur_ahead` 优先串行排空高 MLU 紧迫小流，将两组的 SLO 违约数均从 `1–5` 降至 **`0`**（最大 MLU 分别控制在 `0.4324` 与 `0.6594`）。

### 3.3 SGLang Decode 三级 HiCache 重叠加速与“L3 慢速存储负优化”反例

针对 `65,536` tokens 长上下文请求（每 Token 全层 KV `320 KiB`），对比无 Decode HiCache（仅复用 Decode L1 HBM 命中部分）与启用 SGLang `DecodePrefixMatch` + `HiCacheRestoreGatedKVReceiver` 并发分支：

| Case ID | 命中分布 (`L1 / L2 / L3` tokens) | 远端 Prefill 增量 tokens | 无 HiCache TTFT (`ms`) | 远端 Prefill+RDMA 分支 (`ms`) | 本地 HiCache 恢复分支 (`ms`) | 门控合成 TTFT (`ms`) | TTFT 加速比 | RDMA 流量节省 | 主导瓶颈 (`bottleneck`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `agentic_64k_l2_host_hit_overlap` | `8K / 40K / 0` | `16,384` | `648.09 ms` | `185.17 ms` | `255.65 ms` | **`255.65 ms`** | **`2.54x`** | `71.43%` | `local_hicache_restore` |
| `degraded_rail_2of8_failed_with_hicache` | `8K / 40K / 0` (2/8 轨熔断，`291 Gbps`) | `16,384` | `777.23 ms` | `222.07 ms` | `255.65 ms` | **`255.65 ms`** | **`3.04x`** | `71.43%` | `local_hicache_restore` |
| `multiturn_64k_l2_plus_l3_nvme_hit` (**反例**) | `4K / 24K / 24K` (`100 Gbps` L3) | `12,288` | `694.38 ms` | `138.88 ms` | **`951.03 ms`** | **`951.03 ms`** | **`0.73x` (劣化!)** | `80.00%` | `local_hicache_restore` |

- **反例与架构护栏（何时 L3 命中反而劣化 TTFT？）**：
  在 `multiturn_64k_l2_plus_l3_nvme_hit` 中，虽然 L2+L3 合计省去了 `80%` 的跨节点 RDMA 流量，将远端分支缩短至 `138.88 ms`，但本地恢复分支必须先以 `100 Gbps`（`12.5 GB/s`）从 L3 存储读取 `24,576` tokens（耗费 `644.25 ms`），再通过 PCIe H2D 将 `49,152` tokens 拷回 L1 HBM（`306.78 ms`），导致 `HiCacheRestoreGatedKVReceiver.poll()` 被本地恢复分支阻塞在 **`951.03 ms`**（比远端重算+RDMA 的 `694.38 ms` 慢 **`27%`**）。
  **结论**：只有当本地层级有效恢复带宽 $B_{\text{restore,eff}}$ 满足 $\frac{\Delta L_{\text{tier}} \cdot S_{\text{kv}}}{B_{\text{tier}}} < \frac{\Delta L_{\text{tier}}}{R_{\text{prefill}}} + \frac{\Delta L_{\text{tier}} \cdot S_{\text{kv}}}{B_{\text{RDMA}}}$ 时，启用该存储层恢复才是正收益；否则应限制单请求最大 L3 恢复 token 数并回退至远端 Prefill 重算。

---

## 4. 证据等级与已知边界

- **`implementation`（已核验一手源码）**：
  - `kvcache-ai/Mooncake` (`7ff6bf7f72f15aba7289bc25c87cf491e04efc6f`)：`topology.h`、`rdma_transport.h`、`slice.h`、`deadline_mlu.h`、`bw_arbitration.h`、`promotion_policy.h`、`rail_monitor.h`。
  - `ai-dynamo/nixl` (`c40f1b29b4989765385059196fadcc7bedc37ffb`)：`nixl.h`、`nixl_descriptors.h`、`nixl_types.h`、`libfabric_topology.h`。
  - `sgl-project/sglang` (`4ec19ca27f08f7ce44bbe1da70ce964731940eb2`)：`conn.py`、`decode_hicache_mixin.py`、`l2_transfer.py`。
- **`own-offline`（本项目可复现合成实验）**：
  - `experiments/disaggregated_kv_routing/run.py` 与 `tests/test_disaggregated_kv_routing.py`。
- **边界说明**：本实验为基于源码精确公式的离线确定性算例，不声称在真实多机 ConnectX-7 / GB200 集群上执行了硬件 RDMA 抓包压测。
