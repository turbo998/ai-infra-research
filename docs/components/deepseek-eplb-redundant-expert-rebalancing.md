# DeepSeek EPLB 分层冗余专家装箱与 SGLang 五级异步 P2P 权重重平衡 (`deepseek-ai/EPLB` & `sgl-project/sglang`)

- **Document Status**: Verified `implementation` (`deepseek-ai/EPLB` commit `d52c72d5b2f2fb4c41afbf8eb21366820239913d`, `2025-03-24`; `sgl-project/sglang` commit `642a671c300babc061b0200c69b1c9882465696d`, `2026-10-19`) + `own-offline` (`experiments/deepep_eplb_expert_rebalancing/run.py`)
- **Primary Sources**:
  - `deepseek-eplb-hierarchical-packing`: `deepseek-ai/EPLB`, `eplb.py` (blob `26c3987ff019db83a46fd65d186c7ebc4e82913d`)
  - `sglang-eplb-5tier-p2p-updater`: `sgl-project/sglang`, `python/sglang/srt/eplb/eplb_manager.py` (blob `0388dbee15e36f2756eca410e696a573bdbc6da4`) & `python/sglang/srt/eplb/expert_location_updater.py` (blob `1fd0022efcd08afab405ac50ee3f4b36efe36c13`)
- **Related Questions**: `deepep-eplb-redundant-expert-online-rebalancing`, `deepep-low-latency-ibgda-nvlink-alltoall-dispatch`

---

## 1. 为什么大规模 EP 需要在线冗余专家重平衡 (EPLB)？

在 `DeepSeek-V3/R1` 等超大规模稀疏 MoE 模型中，每层包含 $E_{\text{log}} = 256$ 个路由专家（Routed Experts），分为 $G = 8$ 个专家组（每组 32 个逻辑专家），每个 Token 激活 Top-8 路由专家（且通过 Group-Limited Routing 限制每个 Token 最多命中 4 个专家组）。

当部署在 `EP = 32`（`4 Nodes × 8 GPUs/Node`）或 `EP = 64/144` 集群上时，由于自然语言、代码、数学推理流量的强 Zipfian 偏斜（加上跨时段 Topic Shift），静态均匀映射（每卡固定承载 $256 / 32 = 8$ 个专家）面临两大严峻瓶颈：
1. **最慢 GPU 木桶效应（Straggler Bottleneck）**：因为 `DeepEP` 的 `Dispatch -> Expert GEMM -> Combine` 存在全集群隐式同步，单步 MoE 延迟由负载最高的那张 GPU（$\max_g L_g$）决定。在 Zipfian ($s \approx 0.88$) 叠加组间热点偏斜下，静态无冗余部署的**峰值/均值负载比（$\max/\text{mean}$）高达 `4.337x`**，导致其余 31 张 GPU 超过 `75%` 的时间处于等待空转。
2. **盲目全局复制引发跨节点 IB 流量爆炸**：如果为了削平算力热点，忽略节点拓扑直接做全局贪心复制与装箱（Global Packing），原本属于同一专家组的 32 个专家会被打散到全部 4 个节点上。这会破坏 Group-Limited Routing 的节点亲和性，导致跨节点 InfiniBand/RoCE `All-to-All` 转发流量激增。

---

## 2. 一手源码剖析：`deepseek-ai/EPLB` 的三阶段分层装箱算法

在 `deepseek-ai/EPLB` (`eplb.py`, blob `26c3987ff019db83a46fd65d186c7ebc4e82913d`) 中，DeepSeek 设计了将 $E_{\text{log}} = 256$ 个逻辑专家映射到 $E_{\text{phy}} = 288$ 个物理专家槽位（即每张 GPU 额外增加 1 个冗余专家槽位，共 $32 \times 9 = 288$ 个物理副本）的完整双向索引与分层装箱算法。

### 2.1 三张核心映射表：`phy2log`、`log2phy` 与 `logcnt`

EPLB 在每层维护三张张量：
- `phy2log` (`[num_layers, num_physical_experts]`): 第 $p$ 个物理槽位存放的逻辑专家 ID $e \in [0, E_{\text{log}}-1]$；
- `logcnt` (`[num_layers, num_logical_experts]`): 逻辑专家 $e$ 当前拥有的物理副本总数 $c_e \ge 1$；
- `log2phy` (`[num_layers, num_logical_experts, max_replicas]`): 逻辑专家 $e$ 的各个物理副本对应的全局物理槽位 ID（未使用的槽位填充 `-1`）。当 Token 路由命中逻辑专家 $e$ 时，路由器通过均匀哈希或轮询从 `log2phy[layer, e, 0 : logcnt[layer, e]]` 中选取一个物理副本，将热点专家的 Token 流量均摊至 $c_e$ 个物理槽位：
  $$\hat{w}_e = \frac{w_e}{c_e}$$

### 2.2 核心原语 1：定容贪心装箱 `balanced_packing(weight, num_packs)`

给定 $N$ 个待装项（权重 $w_i$）与 $M$ 个容量严格为 $K = N / M$ 的包裹（Packs），`balanced_packing` 按权重降序排序后，每次将当前项放入**尚未满容（`< K`）且当前累计负载最小**的包裹中：
$$\text{pack}^* = \arg\min_{m \in \{0..M-1\},\, \text{cnt}[m] < K} \text{Load}[m]$$

### 2.3 核心原语 2：边际负载贪心复制 `replicate_experts(weight, num_phy)`

给定 $n$ 个逻辑专家与 $m = \text{num\_phy}$ 个物理槽位（其中 $m - n$ 为冗余副本预算），每次迭代挑选**人均副本负载最高**的逻辑专家增加 1 个副本：
$$e^* = \arg\max_{e \in \{0..n-1\}} \frac{w_e}{\text{logcnt}[e]}, \quad \text{logcnt}[e^*] \leftarrow \text{logcnt}[e^*] + 1$$
总共仅需 $m - n$ 次 `argmax` 迭代（在每节点 64 逻辑专家扩充至 72 物理专家的场景下仅需 `8` 次迭代）。

### 2.4 核心原语 3：Prefill 分层装箱 vs Decode 全局装箱 (`rebalance_experts_hierarchical`)

`rebalance_experts` 根据阶段切换两套策略：
1. **Hierarchical Policy（Prefill 阶段，`num_groups % num_nodes == 0`）**：
   - **Step 1（专家组级节点装箱）**：计算每个专家组 $g \in [0, 7]$ 的总负载 $W_g = \sum_{e \in g} w_e$，调用 `balanced_packing(W_g, num_nodes=4)`，将 8 个专家组严格两两配对放入 4 个节点（每节点固定承载 `2` 个完整专家组 = `64` 个逻辑专家）。这保证了任何发往该专家组的 Token 只需一次跨节点 IB 转发即可在节点内完成分发！
   - **Step 2（节点内冗余复制）**：在每个节点内部，调用 `replicate_experts(node_weights, num_phy_per_node=72)`，将本节点的 64 个逻辑专家扩充为 `72` 个物理专家副本。
   - **Step 3（节点内 GPU 装箱）**：在每个节点内部，调用 `balanced_packing(replicated_weights, num_gpus_per_node=8)`，将 72 个物理副本装入本节点的 8 张 GPU（每卡 `9` 个物理专家）。
2. **Global Policy（Decode 阶段）**：
   - 当进入小批量 Decode（延迟受限于单次通信跳数而非 IB 带宽饱和）或 `num_groups % num_nodes != 0` 时，直接调用 `rebalance_experts_hierarchical(weight, num_replicas, 1, 1, num_gpus)`：在全部 256 个逻辑专家上直接分配 `32` 个冗余副本（共 288 个物理槽位），并跨全集群 32 张 GPU 统一装箱。

```
+-----------------------------------------------------------------------------------+
|            DeepSeek EPLB Hierarchical Packing (256 Logical -> 288 Physical)       |
+-----------------------------------------------------------------------------------+
| Step 1: Group-to-Node Packing (8 Groups -> 4 Nodes, 2 Groups / Node = 64 Log/Node)|
|   Node 0: [Group 0, Group 7]   Node 1: [Group 1, Group 6]                         |
|   Node 2: [Group 2, Group 5]   Node 3: [Group 3, Group 4]                         |
|                                      |                                            |
| Step 2: Intra-Node Greedy Replication (64 Logical -> 72 Physical / Node, +8 Red.) |
|   e* = argmax(w_e / c_e) repeated 8 times inside each node independently          |
|                                      |                                            |
| Step 3: Intra-Node GPU Packing (72 Physical -> 8 GPUs / Node, 9 Physical / GPU)   |
|   Sorted descending by (w_e / c_e), placed into least-loaded GPU with < 9 slots   |
+-----------------------------------------------------------------------------------+
```

---

## 3. 一手源码剖析：`sgl-project/sglang` 的五级异步 P2P 权重迁移引擎

算出新的 `phy2log` 只是第一步：在生产推理服务中，每层 9 个 FP8 路由专家权重约占 $9 \times 44\text{ MiB} = 396\text{ MiB/GPU}$，58 个 MoE 层合计约 `22.4 GiB/GPU`。如果在重平衡时暂停服务、从磁盘或跨节点全量重载，会产生近 **500 ms** 的请求卡顿（Stop-the-World Stall）。

我们在 `sgl-project/sglang` (`python/sglang/srt/eplb/eplb_manager.py` blob `0388dbee15e36f2756eca410e696a573bdbc6da4` 与 `expert_location_updater.py` blob `1fd0022efcd08afab405ac50ee3f4b36efe36c13`) 中核验了 SGLang 的两大核心工程优化：

### 3.1 生成器分块逐层推进 (`EPLBManager._rebalance_layers_per_chunk`)

`EPLBManager.rebalance()` 被实现为一个 Python Generator（`yield`）：每次 Forward Step 仅推进一个 `layers_per_chunk`（如每次仅迁移 1–2 个 MoE 层），在单个 Chunk 迁移完成后立即归还控制权执行下一个 Decode Step，从而将单次重平衡的延迟分摊到数十个推理步中。

### 3.2 五级亲和度级联迁移 (`update_expert_weights_single_layer`)

在 `expert_location_updater.py` 中，SGLang 对比旧布局 `old_global_expert_ids` 与新布局 `new_global_expert_ids`，将每个待更新物理槽位按通信代价从低到高严格划分为**五级（5-Tier Affinity Cascade）**：
1. **Tier 0 (`unchanged`)**：新旧物理槽位存放同一个逻辑专家，**零拷贝（No-op）**。
2. **Tier 1 (`same_gpu`)**：新槽位所需的逻辑专家在旧布局中**本 GPU 已持有副本**，直接通过 GPU 本地 HBM 临时缓冲区 `buffer2` 拷贝（带宽 `~2,400+ GB/s`）。
3. **Tier 2 (`free_rider`)**：本 GPU 上有多个新物理槽位需要同一个逻辑专家，仅需向远端拉取 **1 次**，其余副本在本地顺风车复用（Free-Rider Deduplication）。
4. **Tier 3 (`same_node`)**：本 GPU 没有该专家，但**同节点内（`rank // local_world_size` 相同）的其他 GPU 持有该专家**，优先通过节点内高速 NVLink P2P (`batch_isend_irecv`) 拉取（带宽 `~640 GB/s`）。
5. **Tier 4 (`cross_node`)**：仅当整个节点内都没有该专家（例如发生了跨节点的专家组交换）时，才退化为跨节点 InfiniBand/RoCE RDMA P2P，且按 `SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE` 分批发送以防止撑爆 NCCL 内部缓冲区。

---

## 4. 定量实验与架构对比 (`experiments/deepep_eplb_expert_rebalancing/run.py`)

我们在标准库仿真器 `experiments/deepep_eplb_expert_rebalancing/run.py` 中对 `32 GPUs (4 Nodes × 8 GPUs)`、`256 -> 288 Experts (+32 Redundant)`、`58 MoE Layers` 进行了确定性仿真：

| 策略 / 指标 | Static Uniform (`256` 无冗余) | Global EPLB (`288` 全局装箱) | Hierarchical EPLB (`288` 分层装箱) | 架构意义 |
| :--- | :---: | :---: | :---: | :--- |
| **物理专家槽位数 ($E_{\text{phy}}$)** | `256` (`8/GPU`) | `288` (`9/GPU`) | `288` (`9/GPU`) | 仅增加 `12.5%` 专家显存 |
| **最大副本数 ($\max c_e$)** | `1` | `5` | `4` | 热点专家被拆分为 4–5 份 |
| **峰值/均值 GPU 负载比 ($\max/\text{mean}$)** | **`4.337x`** | **`1.005x`** | **`1.132x`** | 削减 **`73.9%`** 算力木桶偏斜 |
| **每节点跨专家组数量** | `2` | `8` (完全打散) | **`2`** (严格保序) | 分层装箱保持组级节点亲和性 |
| **跨节点 IB Dispatch 流量占比** | `50.00%` | `68.25%` | **`22.00%`** | 比 Global 降低 **`3.10x`** 跨机流量 |

同时，在 SGLang 五级异步 P2P 在线迁移测试中：
- **组内热点漂移（Intra-Group Shift，节点专家组归属不变）**：**`100%`** 的权重迁移落入 `same_gpu` (`7.29%`) 与 `same_node` NVLink (`92.71%`)，**跨节点 RDMA (`cross_node`) 流量严格为 `0.0%`**！配合 `chunk=2 layers` 生成器步进，单步停顿从全量阻塞的 `498.44 ms` 压降至 **`1.14 ms`**（提速 **`437.23x`**）。
- **组间剧烈反转（Inter-Group Shift，节点间发生专家组对调）**：即便发生跨节点组交换，`unchanged + same_gpu + same_node` 仍吸收了 `46.88%` 的槽位更新，`chunk=2 layers` 单步停顿仅为 **`9.65 ms`**（比全量阻塞降低 **`51.65x`**），完全满足在线推理 TPOT SLA。

---

## 5. 解决方案架构师设计准则与可证伪问题

1. **Prefill 与 Decode 差异化 EPLB 策略**：
   - 在 **PD 解耦架构**下，Prefill 集群受限于跨节点 IB `All-to-All` 带宽，**必须开启 `Hierarchical EPLB`**（要求 `num_groups % num_nodes == 0`），将跨节点 IB Dispatch 流量压制在 `22%`；
   - Decode 集群单步 Token 少、受限于最慢卡尾延迟而非带宽饱和，应开启 **`Global EPLB`**，将 GPU 负载比进一步压至 `1.005x`。
2. **后续可证伪问题**：
   - 当单专家权重采用 `NVFP4`（体积减半至 `22 MiB`）且配合 `Copy Engine` 异步流迁移时，是否可以在不开启 `layers_per_chunk` 暂停的情况下实现真正的零停顿双缓冲热切换？
