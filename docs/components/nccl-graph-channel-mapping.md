# NCCL 图搜索与 Channel 映射：可行边怎样变成 Ring、Tree 与 Rail NIC 绑定？

状态：2026-10-14，固定 `NVIDIA/nccl@3e46c089f4ae4732722a419eb6096aead7e47a30`
的有界静态研究。先修：[互连分层](interconnect-network.md)、
[RDMA 接口生命周期](rdma-lifecycle.md)、
[NCCL topology/transport](nccl-topology-transport.md)。未运行 NCCL、GPU、
NVLink、PCIe 或 RDMA，无实机 `NCCL_GRAPH_DUMP_FILE` / profiler 数据。

证据 ID：`nccl-graph-search-channels`、`nccl-rings-connect-layout`、
`nccl-rail-plane-pxn`、`rail-channel-cut-offline`。
精确源码路径、行范围、blob SHA 与限制见 [sources](../../research/sources.json)；
主张与证据状态（`implementation` / `own-offline`）见 [evidence](../../research/evidence.json)。

## 1. 数据布局：从 `ncclTopoSystem` 到 `ncclTopoGraph` 与 `ncclChannel`

固定实现把硬件拓扑图、算法搜索结果和通道连接拆成三层结构，不能把“物理存在某条链路”直接等同于“collective 实际走该链路”：

| 结构 | 固定定义位置 | 关键字段与布局含义 | 架构边界与反例 |
|---|---|---|---|
| `ncclTopoNode` / `ncclTopoLink` / `ncclTopoPath` | [`src/graph/topo.h:132-193`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/topo.h#L132-L193) | 节点类型含 `GPU/PCI/NVS/CPU/NIC/NET`；每对节点保存跳数 `count`、瓶颈带宽 `bw`、路径类型 `type`（`PATH_LOC`..`PATH_NET`..`PATH_DIS`）与 `edges[]` | `paths[i]` 是单跳边列表，路径带宽取沿途最小边 `bw`；有物理边不代表在当前 `maxType` / `minBw` 搜索窗口内可达 |
| `ncclTopoGraph` | [`src/include/graph.h:16-39`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/include/graph.h#L16-L39) | 输入/输出条件：`pattern`（Ring/Balanced Tree/Split Tree/NVLS/CollNet）、`nChannels`、`bwIntra`、`bwInter`、`typeIntra`、`typeInter`、`crossNic`、`sameChannels`；输出：`intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES]`（本机 GPU 序列）与 `inter[MAXCHANNELS*2]`（每 Channel 的 `[recvNet, sendNet]`） | `inter` 存的是每节点每 Channel 的入口与出口 `NET` ID（共 2 项/Channel），不是全集群端到端路由表 |
| `ringRecv / ringSend / ringPrev / ringNext` | [`src/graph/connect.cc:18-137`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/connect.cc#L18-L137)、[`src/graph/rings.cc:13-85`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/rings.cc#L13-L85) | `allGather3Data` 在全局交换每节点每 Channel 的 `ringRecv[c]`（首 GPU）、`ringSend[c]`（尾 GPU）与机内链 `ringIntra`；`ncclBuildRings` 再把各节点机内链首尾相接成全局 `prev[]/next[]` | 多节点 Ring 在每个节点内部先走完 `ngpus - 1` 跳机内链，只在 `ringSend[c] -> 下游节点 ringRecv[c]` 跨一次节点边界 |

## 2. 搜索状态机：`ncclTopoCompute` 如何扣减边带宽并放宽约束

[`src/graph/search.cc:880-1125`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/search.cc#L880-L1125)
中的 `ncclTopoCompute` 不是最短路单次求解，而是**带剩余带宽扣减的回溯搜索（backtracking search with bandwidth reservation）**：

1. **按架构选择候选速度阶梯 `speedArray`**：
   - `ccMin >= 100`（Blackwell SM100）使用 `sm100SpeedArray`（最高 `90.0 GB/s` per-channel 搜索档位）；
   - `ccMin == 90`（Hopper SM90）使用 `sm90SpeedArray`（最高 `60.0 GB/s`）；
   - `ccMin == 80`（Ampere SM80）使用 `sm80SpeedArray`。
2. **沿着 `folly`（路径类型上限 `typeIntra / typeInter`、`sameChannels`、`crossNic`）逐步放宽**：
   - 初始强制 `sameChannels = 1` 且 `crossNic = 0`（要求同一 Channel 在本机的入口 `NET` 与出口 `NET` 相同，即 `net->id == firstNet->id`，见 [`search.cc:656-672`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/search.cc#L656-L672)）。
   - 对每个候选 `(bwIntra, bwInter)`，调用 `ncclTopoSearchTryGpu` 递归枚举：
     - 多节点时，`step == 0` 先选 `NET` 作为本机入口并转到首块 GPU（`ringRecv`）；
     - `1 <= step < ngpus` 在本机 GPU 间寻找满足 `path->type <= typeIntra` 且每条底层 `link->remBw >= requiredBw` 的下一跳 GPU，调用 `followPath` 从经过的每条 `ncclTopoLink` 扣减 `remBw`（[`search.cc:50-108`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/search.cc#L50-L108)）；
     - `step == ngpus` 从尾块 GPU（`ringSend`）选出口 `NET` 并扣减 `GPU->NET` 路径剩余带宽。
   - 若回溯超时（`NCCL_SEARCH_GLOBAL_TIMEOUT = 1<<19` 或单通道超时）或 `nChannels == 0`，先尝试放宽 `sameChannels = 0`（[`search.cc:1039`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/search.cc#L1039)），再放宽 `typeIntra / typeInter`（例如从 `PATH_NVL/PATH_PIX` 退到 `PATH_PHB`、`PATH_SYS`），若 `crossNic == 2` 且仍未达到更优总带宽（`bw * nChannels`），则在 [`search.cc:1066-1071`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/search.cc#L1066-L1071) 将 `graph->crossNic` 切为 `1` 重搜。
3. **Channel 翻倍与后处理复制**：
   - 搜索结束后，若 `nChannels < maxChannels / 2`（且满足架构/带宽条件），[`search.cc:1096-1120`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/search.cc#L1096-L1120) 会将找到的 Channel 序列直接复制翻倍（`memcpy`），并将 `bwIntra` / `bwInter` 折半。这意味着**运行时看到的 `nChannels` 数量可能是物理搜索基础通道数的 2 倍**，不能把 `nChannels` 直接乘以单通道物理链路峰值当作硬件物理总带宽。

## 3. Rail Plane 排序、`mirrorBits` 与 PXN 改写

为什么有些多 NIC 节点不用 `crossNic=1` 也能把全部 NIC 跑满，而有些拓扑却必须开启 cross-rail 或 PXN？代码给出了三处确定性机制：

1. **对称拓扑镜像（`mirrorBits`）与 NIC 排序**：
   - [`src/graph/topo.cc:1278-1356`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/topo.cc#L1278-L1356) 的 `ncclTopoGetLocalNetType` 检测节点内是否存在跨 PCI/CPU 镜像对称的 `mirrorBits`。若满足对称切分，[`search.cc:503-521`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/search.cc#L503-L521) 在搜索不同 Channel 时通过 `index ^ mirror` 交替选择对称位置的 NIC，避免所有 Channel 挤在同一张 NIC。
   - [`src/graph/topo.cc:1187-1275`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/topo.cc#L1187-L1275) 的 `ncclTopoGetRailPlane` 与 `ncclTopoSortDevsByRailPlane` 根据 GPU 到各 NIC 的路径距离与平面编号排序，再由 `ncclTopoSelectNets` 结合 `NCCL_NETDEVS_POLICY`（`AUTO / ALL / MAX:n`）分配每块 GPU 可见的 `NET` 集合。
2. **PXN（PCI x NVLink）路径改写的单向性**：
   - [`src/graph/paths.cc:516-559`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/paths.cc#L516-L559) 的 `ncclTopoAddNetPaths` 在满足 `ncclPxnDisable(comm) == 0` 时，找一张与目标 `NET` 具备 `PATH_PXB` 或 `PATH_PIX` 的本机 peer GPU `g`，若 `g <-> Target GPU` 之间是 `PATH_NVL`，则把 `GPU -> NET` 路径改写为 `PATH_PXN`（经过 NVLink 到 peer GPU，再由 peer GPU 经 PCIe 到 NIC），但**明确不改写 `NET -> GPU` 接收路径**。
   - 架构含义：PXN 可以让跨 Rail 的远端发送在发送端内部通过 NVLink 汇集到同 Rail 的本地 NIC 再发出，从而在交换机侧保持同 Rail 通信，但它消耗机内 NVLink 带宽与代理 GPU 的队列/内存资源，不是零开销魔法。
3. **Tree 与 Ring 的全局拼接差异**：
   - [`src/graph/rings.cc:13-85`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/rings.cc#L13-L85) 的 `ncclBuildRings` 将节点 `n` 在 Channel `c` 的 `ringIntra[0..G-1]` 串接成链，并令 `prev[ringRecv[c]] = ringSend_of_prev_node[c]`、`next[ringSend[c]] = ringRecv_of_next_node[c]`。
   - [`src/graph/connect.cc:139-227`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/connect.cc#L139-L227) 将机内 GPU 串成线性链（`ringRecv -> child0 -> ... -> ringSend`），机间再用双二叉树（`tree0` 与镜像 `tree1`）连接各节点的 `ringRecv`/`ringSend`，使每个非叶节点承担约一半消息的机间上/下行。

## 4. 公式、单位与 `rail_channel_cut` 离线反证实验

单共享 NIC 的 `ring_cut` 模型只统计节点边界的切割边数，无法反映多 Channel、多 Rail NIC 分流、跨 Rail 偏斜和机内瓶颈。我们在 [experiments/rail_channel_cut](../../experiments/rail_channel_cut/README.md) 中实现了原创 Python 标准库算例（`own-offline`）：

设总 rank 数为 $p$，节点集合为 $V_{\text{node}}$，Channel 数为 $C$，单次 Ring AllReduce 输入消息大小为 $S\text{ (bytes)}$（要求 $p \cdot C \mid S$）。每个 Channel 内每条有向 Ring 边承载的理想 payload（不含协议头、LL/LL128 标志位或重传）为：

$$V_{\text{ch-edge}} = 2(p - 1)\frac{S}{p \cdot C}$$

对任意节点 $u$ 上的物理 NIC $m$（单向全双工发送/接收有效 payload 上限为 $B_{\text{nic}}$）以及节点内有向 GPU 边 $e_{\text{intra}} = (g_i \to g_j)$（机内瓶颈边有效 payload 上限为 $B_{\text{intra}}$）：

$$V_{\text{send}}(u, m) = \sum_{c=0}^{C-1} k_{\text{out}}(c, u, m) \cdot V_{\text{ch-edge}}, \quad V_{\text{recv}}(u, m) = \sum_{c=0}^{C-1} k_{\text{in}}(c, u, m) \cdot V_{\text{ch-edge}}$$

$$V_{\text{intra}}(e_{\text{intra}}) = \sum_{c=0}^{C-1} \mathbb{I}[e_{\text{intra}} \in \text{Ring}_c] \cdot V_{\text{ch-edge}}$$

$$T_{\text{nic}} = \frac{\max_{u, m} \max\bigl(V_{\text{send}}(u, m),\, V_{\text{recv}}(u, m)\bigr)}{B_{\text{nic}}}, \quad T_{\text{intra}} = \frac{\max_{e_{\text{intra}}} V_{\text{intra}}(e_{\text{intra}})}{B_{\text{intra}}}, \quad T_{\text{combined}} = \max(T_{\text{nic}}, T_{\text{intra}})$$

固定 $p = 8$（两节点 `A/B` 各 4 GPUs）、$S = 1\text{ GiB} = 1{,}073{,}741{,}824\text{ B}$、$C = 4$、$B_{\text{nic}} = 50\text{ GB/s} = 50 \times 10^9\text{ B/s}$。四个合成场景的可重算结果（见 [`experiments/rail_channel_cut/results.json`](../../experiments/rail_channel_cut/results.json)）：

| 场景 (`case_id`) | Channel 与 NIC / 机内配置 | 最忙单向 NIC 流量 | 最忙机内有向边流量 | $T_{\text{nic}}$ (s) | $T_{\text{intra}}$ (s) | $T_{\text{combined}}$ (s) | 主导瓶颈 |
|---|---|---:|---:|---:|---:|---:|---|
| `shared_single_nic_4ch` | 4 Channels 全部落到单张 `NIC0`，$B_{\text{intra}} = 400\text{ GB/s}$ | `1.75 GiB` | `1.75 GiB` | `0.03758096384` | `0.00469762048` | `0.03758096384` | `nic` |
| `rail_aligned_4ch_crossnic0` | 4 Channels 对齐 `NIC0..3`（`crossNic=0`），旋转起点分散机内边 | `0.4375 GiB` | `1.3125 GiB` | `0.00939524096` | `0.003288334336` | `0.00939524096` | `nic` |
| `cross_rail_skew_4ch_crossnic1` | `ch1` 跨 Rail 导致 `A:NIC0 -> B:NIC0` 叠加 2 个 Channel | `0.875 GiB` | `1.3125 GiB` | `0.01879048192` | `0.003288334336` | `0.01879048192` | `nic` |
| `intra_phb_clamp_4ch_rail` | 4 张 Rail NIC 均匀分担，但机内边降为 `32 GB/s`（`PATH_PHB`）且同环叠加 | `0.4375 GiB` | `1.75 GiB` | `0.00939524096` | `0.058720256` | `0.058720256` | `intra_node` |

**推翻单共享 NIC 简化模型的两条关键结论**：
1. 相同跨节点边数（每节点每 Channel 1 进 1 出）下，4 张独立 Rail NIC 对齐可将 NIC payload 下界从 `37.58 ms` 降到 `9.40 ms`（$4\times$）；而一旦 `crossNic` 或不对称拓扑让两个 Channel 共用同一出口 NIC，最忙 NIC 立即将网络下界拉回 `18.79 ms`。
2. 即便节点配齐 4 张 `50 GB/s` Rail NIC（聚合发送上限 `200 GB/s`），只要机内 `ringIntra` 被迫跨 `PATH_PHB`（`32 GB/s`）且多 Channel 叠加在同一机内瓶颈边，机内下界 `58.72 ms` 就会完全掩盖多 NIC 的收益，甚至比“单 NIC + NVLink”节点（`37.58 ms`）更慢。这正是 `ncclTopoCompute` 宁可减少 Channel 或开启 PXN 也要优先保持高 `bwIntra` 的原因。

## 5. 架构影响：质量、SLO、拓扑、成本与运维

- **拓扑与调度运维**：在 Kubernetes / Slurm 分配部分 GPU（如单节点 4/8 卡）或 MIG/虚拟化切片时，若容器可见的 GPU 与 NIC 不属于同一 PCIe Switch（`PATH_PIX`/`PATH_PXB`）或对称 Rail Plane，`ncclTopoCompute` 会退化到 `PATH_PHB`/`PATH_SYS` 或触发 `crossNic=1`，导致同规格作业出现数倍的 step time 抖动。
- **故障与降级边界**：单条 NVLink 或单张 NIC 掉线会改变 `ncclTopoSystem` 的可用边，使该节点搜出的 `nChannels` 或 `bwIntra` 低于其他节点；由于 `allGather3Data` 在全局取最小 `nChannels` 與最小 `bwIntra/bwInter`（[`connect.cc:69-88`](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/graph/connect.cc#L69-L88)），**单节点的局部拓扑降级会把整个 communicator 的 `nChannels` 拉低**。
- **成本与功耗**：盲目调大 `NCCL_MIN_NCHANNELS` 只会触发软件通道复制或跨过慢路径，占用更多 GPU SM CTA 与 CQ/QP 内存，却不会突破物理瓶颈边 `remBw`。

## 6. 架构师练习与后续可证伪问题

1. **解释练习**：根据 `src/graph/search.cc:1032-1075` 与 `src/graph/connect.cc:69-88`，解释为什么集群中只有 1 台机器掉了一张 NIC 或一条 NVLink，会导致整个多机训练 communicator 的 `nChannels` 或 `sameChannels` 发生变化？
2. **计算练习**：固定 `p = 16`（2 节点各 8 GPUs）、`S = 2 GiB`、`C = 8`、每张 Rail NIC 单向 `50 GB/s`。先手算 `crossNic = 0`（8 张 NIC 1:1 对齐）时的 $V_{\text{ch-edge}}$ 与 $T_{\text{nic}}$；再计算当 1 张 NIC 故障下线、其负责的 1 个 Channel 改走邻近 NIC 时的 $T_{\text{nic}}$ 退化比例。
3. **排障设计练习**：当线上多机 AllReduce 吞吐异常偏低时，如何仅用公开日志/拓扑导出变量（`NCCL_DEBUG=INFO`、`NCCL_DEBUG_SUBSYS=INIT,GRAPH`、`NCCL_GRAPH_DUMP_FILE`）区分“全局 `nChannels` 被单慢节点拉低”“退化到 `PATH_PHB/PATH_SYS`”与“触发 `crossNic=1` / PXN 禁用”？

练习状态：`not-assessed`（未收到用户作答反馈）。后续可证伪问题已同步至 [`research/queue.json`](../../research/queue.json)。
