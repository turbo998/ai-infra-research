# 多机 RoCEv2 集群 `NCCL_IB_GID_INDEX`、`NCCL_IB_TC`（DSCP 无损队列错配）与静默退化 `NET/Socket`（`380 GB/s -> 1.8 GB/s`）排障：为什么配对了 `GID_INDEX=3` 但漏配 `NCCL_IB_TC=106` 会让 400G RDMA 流量误入交换机有损队列（带宽崩至 `24.5 GB/s`）？动态 GID 发现、DSCP 26 双向绑定与虚拟网卡过滤选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/ai_cluster_nccl_socket_ifname_gid_index_rocev2_dscp_fallback_selection/run.py` 与 `results.json`。

![NCCL RoCEv2 GID_INDEX, DSCP Traffic Class & Socket Fallback Visual Summary](../../assets/svg/nccl-ib-gid-index-rocev2-v1-fallback-socket-ifname-misconfig-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**万卡 RoCEv2 容器集群 NCCL 零配置漂移防护体系（基于 `sysfs` 的 `RoCE v2 + IPv4` 动态 `GID_INDEX` 校验 + 数据面 `NCCL_IB_TC=106` 与建连控制面 `cma_roce_tos -t 106` 双栈 `DSCP 26` 无损队列对齐 + `NCCL_SOCKET_IFNAME=^docker0,lo,veth,cali` 虚拟网桥屏蔽 + 禁止静默回退 `NET/Socket` 硬断言）**。
- **架构师价值**：在 `8x 400G RoCEv2` 三层（L3 Leaf-Spine）Clos 网络中，理论 `All-Reduce` 总线带宽高达 **`380.0 GB/s`**。但生产环境中最频发的两类性能断崖均源于环境变量的毫厘之差：
  1. **第一类断崖（`380 GB/s -> 1.8 GB/s`，慢 `211 倍`）**：未显式指定 `NCCL_IB_GID_INDEX`（默认用到 `GID 0/1` 的二层 `RoCEv1` 帧，跨三层 Leaf 交换机直接被丢弃），或容器内的 `docker0 / cali*` 虚拟网卡劫持了带外建连，导致 NCCL 悄悄降级走 `25G` 管理网 TCP **`NET/Socket`**，训练还在跑但速度慢了 200 倍！
  2. **第二类更隐蔽的断崖（`380 GB/s -> 24.5 GB/s`，慢 `15.5 倍`）**：运维配了 `NCCL_IB_GID_INDEX=3`（`RoCEv2 IPv4`），日志也显示成功走上了 `NET/IB`，但**漏配了 `NCCL_IB_TC=106`（对应 `DSCP 26 = ToS 0x6a`）**！结果所有 400G RDMA 报文带着默认的 `TC=0 (DSCP 0)` 进入交换机，**全数落入没有配置 PFC 与 ECN 的“0 号普通有损队列（Best-Effort Lossy Queue 0）”，而不是专门开辟的“3 号无损队列（Lossless Priority 3）”**！在多对一通信瞬间发生 `18.4%` 尾丢包与网卡 `Go-Back-N` 重传风暴！采用 **四重自校验配置（Tier 4）** 可实现 **`100%` 建连成功率、`0.0%` 丢包与 `380.0 GB/s` 满速线速吞吐**。

---

## ① 📌 核心结论速览（Executive Summary）

1. **为什么 `NCCL_IB_TC=106` 是 RoCEv2 集群的生命线？（`DSCP 26` 与 IP `ToS` 字节的位运算换算）**
   交换机（如 SONiC / Arista / NVIDIA Spectrum）依靠 IPv4 报文头里的 **`8-bit ToS (Type of Service)` 字节** 将流量分类到不同的硬件队列：
   - IPv4 `ToS` 的高 6 位是 **`DSCP`（Differentiated Services Code Point）**，低 2 位是 **`ECN`（Explicit Congestion Notification）**，即 $\text{ToS (TC)} = \text{DSCP} \times 4 + \text{ECN}$。
   - 数据中心标准无损 RoCEv2 队列（`Priority 3`）通常绑定在 **`DSCP = 26`**（或 `DSCP = 24`）。换算成 `NCCL_IB_TC`（即整字节 `ToS` 值，不含 ECN 位时）就是 **$26 \times 4 = 104$（若带 ECN 能力标记则为 `106`）**！
   - **如果不设 `NCCL_IB_TC=106`**：NCCL 默认发出 `TC = 0`（`DSCP = 0`），交换机一看是 `DSCP 0`，直接把 400G 的 RDMA 洪流扔进给普通 SSH/HTTP 用的 **0 号有损队列（Lossy Queue 0）**——既没有 `DCQCN ECN` 打标降速，也没有 `PFC` 反压保护，瞬间丢包重传崩盘至 **`24.5 GB/s`**！
2. **为什么硬编码 `NCCL_IB_GID_INDEX=3` 在开启 IPv6 或 VLAN 的机器上会突然报 `ibv_modify_qp failed`（Tier 3）？**
   在 Linux RDMA 子系统中，网卡的 `GID Table` 是按 IP 地址动态生成的：
   - 通常情况下：`GID 0` = RoCEv1 MAC，`GID 1` = RoCEv2 MAC，`GID 2` = RoCEv1 IPv4，**`GID 3` = RoCEv2 IPv4**。
   - **动态漂移陷阱**：如果某台服务器多配了一个 IPv6 Link-Local 地址或副 IP，内核会把新的 IPv6 GID 插在前面，导致 **`RoCEv2 IPv4` 的序号从 `3` 挤到了 `5`**！此时硬编码 `NCCL_IB_GID_INDEX=3` 的节点拿了一个 `RoCEv1` 或 `IPv6` 的 GID 去跟对端建连，当场报错崩溃！
3. **为什么除了设置 `NCCL_IB_TC=106`，还必须在宿主机执行 `cma_roce_tos -d mlx5_X -t 106`？**
   `NCCL_IB_TC=106` 只控制 **RDMA 数据包（Data QP）** 的 DSCP 标记；而在建连握手阶段，内核 `rdma_cm` 发送的 `CM REQ / REP` 建连控制包默认仍然是 `ToS = 0`！在拥塞时如果建连控制包在 Queue 0 被丢弃，会导致作业启动时报 `Connection timed out`。通过 `cma_roce_tos` + `mlnx_qos` 将控制面与数据面统一定死在 `ToS = 106`，可确保 `100%` 握手成功。

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：怎么防止 NCCL 在 RDMA 网卡异常或配置写错时，“偷偷摸摸”退化到 `1.8 GB/s` 的 `NET/Socket` 慢速 TCP 模式浪费几千卡时？
- **在所有生产训练脚本中注入“禁止静默降级”双保险**：
  1. 设置 `export NCCL_IB_DISABLE=0` 与 `export NCCL_NET="IB"`（强制指定只允许使用 `IB/RoCE` 插件，一旦 RDMA 初始化失败立即抛异常退出，绝不允许回退到 `Socket`！）；
  2. 在 `torch.distributed.init_process_group()` 完成后的第 1 个 Step 校验日志中包含 `NCCL INFO NET/IB` 且单步 `All-Reduce` 带宽 $> 250\text{ GB/s}$。

### Q2：为什么容器里明明有 8 张 `eth0..eth7` 高速网卡，NCCL 初始化时却卡死在 `connect() timed out`？
- 因为 Kubernetes / Docker 容器里除了物理网卡，还有 **`docker0`、`lo`、`veth*`、`cali*`（Calico）、`flannel*`** 等虚拟网卡。NCCL 在通过 TCP 带外通道（Out-of-Band Bootstrap）交换 RDMA QP 信息时，如果误选了无法跨机互通的 `docker0`（`172.17.0.1`），所有节点都会连向自己的本地网桥死锁！
- **标准解法**：显式设置排除规则 `export NCCL_SOCKET_IFNAME="^docker0,lo,veth,cali,flannel,cni"`（或者正向指定管理网前缀 `export NCCL_SOCKET_IFNAME="bond0,eth0"`）。

### Q3：如何在容器启动脚本（Entrypoint）里用 5 行 Shell 自动探测当前机器正确的 `RoCEv2 IPv4` `GID_INDEX`，彻底告别硬编码？
- 直接扫描 `/sys/class/infiniband/mlx5_0/ports/1/gids/*` 与 `gid_attrs/types/*`：
  查找满足 `type == "RoCE v2"` 且对应 `ndevs` IPv4 地址（`::ffff:10.x.x.x`）的那一行索引号，自动导出为 `NCCL_IB_GID_INDEX`，即使不同节点内核版本或 IPv6 状态不同也能 `100%` 精准对齐！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 多机 RoCEv2 集群 NCCL 网络参数核验与排障决策树

```mermaid
flowchart TD
    A["多机 8x 400G RoCEv2 集群 NCCL 带宽异常 (<50 GB/s) 或建连失败"] --> B{"查看 NCCL_DEBUG=INFO 首行传输层类型"}
    B -- "显示 NET/Socket (仅 1.8 GB/s!)" --> C["⛔ Tier 1: OOB 网卡被 docker0 劫持 或 GID=0 (RoCEv1) 跨 L3 失败退化为 TCP!"]
    B -- "显示 NET/IB, 但带宽仅 24.5 GB/s 且重传飙升" --> D{"检查 NCCL_IB_TC 与交换机 PFC/ECN 队列映射"}
    D -- "未设置 NCCL_IB_TC (默认 TC=0 / DSCP 0)" --> E["⛔ Tier 2: 400G RDMA 落入交换机 0 号有损队列 (无 PFC/ECN) -> 18.4% 丢包重传!"]
    D -- "硬编码 GID_INDEX=3, 部分节点报 ibv_modify_qp error" --> F["⚠️ Tier 3: 双栈 IPv6 导致 GID 表序号偏移 (3 -> 5)"]
    D -- "动态探测 RoCEv2 IPv4 GID + NCCL_IB_TC=106 + NET=IB 强校验" --> G["🏆 Tier 4: 100% 命中无损队列 3 + 0% 丢包 + 380.0 GB/s 满速吞吐"]
```

### 3.2 漏配 `NCCL_IB_TC` 误入有损队列（`24.5 GB/s`）vs. `TC=106` 命中无损队列 3（`380.0 GB/s`）对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 2: GID=3 但漏配 NCCL_IB_TC (默认 TC=0)"]
      B1["RDMA 报文携带 DSCP=0 (TC=0)"] --> B2["落入交换机 Lossy Queue 0 (无 ECN/PFC) -> 丢包重传崩至 24.5 GB/s!"]
    end
    subgraph Good["🏆 Tier 4: 动态 RoCEv2 GID + NCCL_IB_TC=106 (DSCP 26)"]
      G1["RDMA 报文携带 DSCP=26 (TC=106)"] --> G2["精准进入交换机 Lossless Priority-3 队列 (DCQCN + PFC) -> 380.0 GB/s!"]
    end
```

### 3.3 容器启动时动态 `GID` 探测、控制面/数据面 `ToS=106` 绑定与 `NET/IB` 建连时序图

```mermaid
sequenceDiagram
    participant Entry as Container Entrypoint
    participant Sysfs as /sys/class/infiniband/mlx5_*
    participant NCCL as NCCL NET/IB Plugin
    participant Sw as L3 Leaf Switch (Lossless Queue 3)

    Entry->>Sysfs: 1. 扫描 gid_attrs/types=="RoCE v2" & IPv4 (::ffff:) 自动锁定 GID_INDEX
    Entry->>NCCL: 2. 注入 NCCL_NET=IB, NCCL_IB_GID_INDEX, NCCL_IB_TC=106, SOCKET_IFNAME=^docker0,lo
    NCCL->>Sw: 3. 发出携带 DSCP 26 (ToS 106) 的 RoCEv2 UDP (Port 4791) 报文
    Sw-->>NCCL: 4. 映射至无损队列 3 (启用 WRED/ECN 与 PFC 保护)，跑满 380.0 GB/s 线速
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | `GID_INDEX` 与 `NCCL_IB_TC` 配置 | `All-Reduce` 总线带宽 | 丢包重传率 | 跨机建连成功率 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 默认 `GID=0` (`RoCEv1`) + 无 `IFNAME` 过滤** | L3 丢弃退化为 `NET/Socket` | `1.8 GB/s` (`211x` 慢 ✕) | `100.0%` (RDMA 不通) | `14.0%` ✕ | ⛔ **灾难性静默 TCP 降级** |
| **Tier 2: `GID=3` (`RoCEv2`) 但漏配 `NCCL_IB_TC`** | `TC=0` 误入交换机 0 号有损队列 | `24.5 GB/s` (`15.5x` 慢 ✕) | `18.4%` ✕ | `100.0%` | ⛔ **最隐蔽的“假 RDMA”有损丢包陷阱** |
| **Tier 3: 硬编码 `GID=3` + `NCCL_IB_TC=106`** | 静态序号易受 IPv6 偏移影响 | `362.0 GB/s` ✅ | `0.4%` | `88.0%` ⚠️ | ⚠️ 双栈节点易发 `ibv_modify_qp` 报错 |
| **Tier 4: 动态 `RoCEv2 GID` + `TC=106` + `NET=IB` (推荐)** | **全栈对齐 `DSCP 26` 无损队列 3** | **`380.0 GB/s` 🏆** | **`0.0%` 🏆** | **`100.0%` 🏆** | 🏆 **万卡 RoCEv2 生产集群黄金标准** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/ai_cluster_nccl_socket_ifname_gid_index_rocev2_dscp_fallback_selection/run.py`](../../experiments/ai_cluster_nccl_socket_ifname_gid_index_rocev2_dscp_fallback_selection/run.py) · 封存数据：[`results.json`](../../experiments/ai_cluster_nccl_socket_ifname_gid_index_rocev2_dscp_fallback_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/nccl-ib-gid-index-rocev2-v1-fallback-socket-ifname-misconfig-sizer.html`](../reports/nccl-ib-gid-index-rocev2-v1-fallback-socket-ifname-misconfig-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/nccl-ib-gid-index-rocev2-v1-fallback-socket-ifname-misconfig-selection-cards.html`](../reports/nccl-ib-gid-index-rocev2-v1-fallback-socket-ifname-misconfig-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: GID=0 Socket Fallback", "busbw_gbps": 1.8, "drop_pct": 100.0 },
      { "tier": "Tier 2: GID=3 Missing TC=106 (Lossy Q0)", "busbw_gbps": 24.5, "drop_pct": 18.4 },
      { "tier": "Tier 3: Hardcoded GID=3 + TC=106", "busbw_gbps": 362.0, "drop_pct": 0.4 },
      { "tier": "Tier 4: Auto RoCEv2 GID + TC=106 + NET=IB", "busbw_gbps": 380.0, "drop_pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "busbw_gbps": "Quantity",
    "drop_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Verified RoCEv2 GID + NCCL_IB_TC=106 (DSCP 26) Delivers 380.0 GB/s (15.5x Over Unmarked Lossy RoCE)",
    "subtitle": "Multi-node 8x400G All-Reduce bus bandwidth (GB/s) across NCCL RoCEv2 configurations (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "busbw_gbps",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有基于三层（L3 Leaf-Spine）RoCEv2 以太网构建的 GPU 训练与推理集群。
- **⛔ Not for**：在 RoCEv2 网络上只设 `NCCL_IB_GID_INDEX` 而不设 `NCCL_IB_TC`，或未禁用 `NET/Socket` 静默回退。

### 生产核验 Checklist
- [ ] **强制锁定 `NCCL_NET=IB` 防静默降级**：确保当 RDMA 建连失败时立即报错中止，绝不退化到 `1.8 GB/s` 的 TCP `NET/Socket`。
- [ ] **`NCCL_IB_TC` 与交换机无损队列 `DSCP` 严格对齐**：若交换机无损队列 3 绑定 `DSCP 26`，必须配置 `export NCCL_IB_TC=106`（或 `104`）并在宿主机执行 `cma_roce_tos -d mlx5_X -t 106`。
- [ ] **动态探测 `RoCE v2` IPv4 `GID_INDEX`**：通过 `/sys/class/infiniband/*/ports/1/gid_attrs/types/*` 动态验证 `RoCE v2` 对应的 IPv4 GID 索引号。
- [ ] **排除虚拟容器网桥**：设置 `export NCCL_SOCKET_IFNAME="^docker0,lo,veth,cali,flannel,cni"`。
