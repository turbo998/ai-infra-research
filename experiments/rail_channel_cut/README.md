# Rail channel cut：多 Channel、Multi-Rail 与机内瓶颈怎样推翻单共享 NIC 模型

原创 Python 标准库、CPU/offline 算例；不运行 NCCL、GPU、NVLink、PCIe、RDMA 或模型。
固定的 NCCL `search.cc` / `rings.cc` / `connect.cc` 静态代码证据见
[NCCL 图搜索与 Channel 映射正文](../../docs/components/nccl-graph-channel-mapping.md)。

## 为什么需要扩展 `ring_cut`

`experiments/ring_cut` 假设 **单条 Ring（`C = 1`）** 与 **每节点单一共享发送 NIC**，
忽略了 `NVIDIA/nccl@3e46c089f4ae4732722a419eb6096aead7e47a30` 中 `ncclTopoCompute`
（`src/graph/search.cc`）与 `ncclBuildRings`（`src/graph/rings.cc`）的三个核心机制：

1. **多 Channel 分片（`C >= 1`）**：每个 Channel 承担总消息 `S` 的 `1 / C`，每条 Channel 内有向边发送量为
   `V_ch_edge = 2 × (p - 1) × (S / (p × C))`。
2. **每节点每 Channel 的 `ringRecv` / `ringSend` NIC 绑定与 `crossNic`**：
   - 当 `crossNic == 0` 且 4 个 Channel 均匀绑定到 4 张独立 Rail NIC（`NIC0..NIC3`）时，单张 NIC 的发送/接收负载降为单共享 NIC 的 `1 / 4`。
   - 若 `crossNic == 1` 或局部拓扑偏斜导致两个 Channel 共用同一出口 NIC（如 `A:NIC0` 发送两个 Channel，而 `A:NIC1` 发送 0 个 Channel），最忙单向 NIC 立即把网络下界拉回 `2×`。
3. **机内有向边瓶颈（`bwIntra`：NVLink vs `PATH_PHB` / `PATH_SYS`）的反证**：
   即便 4 张 Rail NIC 将跨节点网络下界压到 `9.395 ms`，若机内 `ringIntra` 某条有向边（如 `0 -> 1`）跨 PCIe Host Bridge（`PATH_PHB`，合成上限 `32 GB/s`）且被 4 个 Channel 共同穿过，机内有向边下界（`58.720 ms`）会直接主导总体下界，甚至慢于单 NIC + NVLink 节点（`37.581 ms`）。这对应 `search.cc` 中按 `speedArray` 递减搜索 `bwIntra` 的静态机理。

## 可重算结果（`p = 8, S = 1 GiB, C = 4, B_nic = 50 GB/s`）

| Case ID | 机制假设 | 最忙单向 NIC 负载 | 最忙机内有向边负载 | NIC 下界 (s) | 机内边下界 (s) | 联合下界 (s) | 主导瓶颈 |
|---|---|---:|---:|---:|---:|---:|---|
| `shared_single_nic_4ch` | 4 Channels 共用单张 `NIC0`，机内 `400 GB/s` | `1.75 GiB` | `1.75 GiB` | `0.03758096384` | `0.00469762048` | `0.03758096384` | `nic` |
| `rail_aligned_4ch_crossnic0` | 4 Channels 对齐 `NIC0..3`（`crossNic=0`），旋转起点分散机内边 | `0.4375 GiB` | `1.3125 GiB` | `0.00939524096` | `0.003288334336` | `0.00939524096` | `nic` |
| `cross_rail_skew_4ch_crossnic1` | `ch1` 在节点 A/B 跨 Rail（`A:NIC0 -> B:NIC0` 叠加 2 个 Channel） | `0.875 GiB` | `1.3125 GiB` | `0.01879048192` | `0.003288334336` | `0.01879048192` | `nic` |
| `intra_phb_clamp_4ch_rail` | 4 张 Rail NIC 均匀分担，但机内边降为 `32 GB/s`（`PATH_PHB`）且同环叠加 | `0.4375 GiB` | `1.75 GiB` | `0.00939524096` | `0.058720256` | `0.058720256` | `intra_node` |

## 严格限制

- 证据状态：`own-offline`（合成离线算例，非硬件实测，非 NCCL 运行时日志）。
- 网络发送量与机内有向边流量均**不等于 GPU HBM 读写流量**。
- 未包含 `Simple` / `LL` / `LL128` 协议开销、FIFO 握手延迟、PXN 跨卡转发双跳开销或交换机拥塞。

## 运行与校验

```powershell
python experiments\rail_channel_cut\run.py
python experiments\rail_channel_cut\run.py --check
python -m unittest tests.test_rail_channel_cut
```
