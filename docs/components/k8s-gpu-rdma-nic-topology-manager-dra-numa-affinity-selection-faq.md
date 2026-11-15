# Kubernetes 多卡/多网卡拓扑感知调度深水区（`Device Plugin` vs. `Topology Manager` vs. `DRA`）：为什么申请 `2` 张 GPU + `2` 张 400G RDMA 网卡时，K8s 竟把 `GPU 0` 与远端 `NUMA 1` 的 `mlx5_7` 配在了一起（跨 Socket `SYS` 绕行暴跌 `-76%`）？`single-numa-node` 误拒 8 卡 Pod 陷阱与 `DRA` 联合 `PIX` 绑定选型指南

> **Provenance & Evidence Tier**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)。定量数据锚定于 [`experiments/ai_cluster_k8s_dra_nri_topo_lstopo_nic_gpu_numa_affinity_selection/results.json`](../../experiments/ai_cluster_k8s_dra_nri_topo_lstopo_nic_gpu_numa_affinity_selection/results.json)。
> **English Version**: [Kubernetes Multi-GPU & Multi-RNIC Topology-Aware Allocation: Legacy `Device Plugin` Cross-NUMA (`SYS`) Mismatch, `Topology Manager single-numa-node` 8-GPU Rejection Trap & `DRA (Dynamic Resource Allocation)` `PIX` Co-Allocation Selection Guide](./k8s-gpu-rdma-nic-topology-manager-dra-numa-affinity-selection-faq.en.md)

![K8s Multi-GPU & Multi-RNIC Topology Manager vs DRA Summary](../../assets/svg/k8s-gpu-rdma-nic-topology-manager-dra-numa-affinity-selection-faq-summary.svg)

---

## 0. 🔥 今日领域最热门能力突破追踪（Daily Hottest Breakthrough）

- **揭秘为什么在公用 Kubernetes 智算集群上，当一个推理或中小规模训练 Pod 申请 `2` 张 GPU + `2` 张 400G RDMA 网卡（或 `4` 卡 + `4` 网卡）时，明明物理服务器的硬件拓扑是完美的 1:1 `PIX` 直连，进到容器里一跑 `nccl-tests` 跨机带宽却从 `47.5 GB/s` 暴跌至 `11.2 GB/s`（甚至开了 Kubelet `single-numa-node` 后不仅还慢在 `21.5 GB/s`，还把全网所有的 `8` 卡满机训练 Pod 100% 拒之门外报 `TopologyAffinityError`）？从第一性原理拆解 K8s `Device Plugin` 的双盲分配缺陷与 `Topology Manager` 仅认 `NUMA` 不认 `PCIe Switch` 的粒度盲区，并通过新一代 Kubernetes `DRA (Dynamic Resource Allocation)` / `NRI` 拓扑联合分配器（`Tier 4`），实现 `100%` `PIX` 交换芯片级 1:1 精准配对（恢复 `47.5 GB/s` 满速，`4.24x` 提速）且 `100%` 兼容 1/2/4/8 卡全规格 Pod**：在一台标准的 8 卡 H100/H800/H200 AI 服务器内部，物理硬件拓扑是严格分层的三层树：
  - **顶层（2 颗 CPU / 2 个 NUMA 节点）**：`NUMA 0` 管左半边（`GPU 0..3`, `mlx5_0..3`），`NUMA 1` 管右半边（`GPU 4..7`, `mlx5_4..7`）；
  - **中间层（4 颗 PCIe Gen5 Switch 芯片，每颗 NUMA 下挂 2 颗 Switch）**：
    - `Switch 0 (NUMA 0)` 下挂：`GPU 0, GPU 1` 与 `mlx5_0, mlx5_1`（彼此是 **`PIX`** 150ns 交换芯片直转！）；
    - `Switch 1 (NUMA 0)` 下挂：`GPU 2, GPU 3` 与 `mlx5_2, mlx5_3`；
    - `Switch 2 (NUMA 1)` 下挂：`GPU 4, GPU 5` 与 `mlx5_4, mlx5_5`；
    - `Switch 3 (NUMA 1)` 下挂：`GPU 6, GPU 7` 与 `mlx5_6, mlx5_7`。
  1. **灾难现场一：默认独立 `Device Plugin`（`topologyManagerPolicy: none`，Tier 1 — “盲婚哑嫁”配出跨 CPU 插槽 `SYS`，带宽暴跌至 `11.2 GB/s`！）**：
     - 当 Pod 声明需要 `nvidia.com/gpu: 2` 和 `rdma/hca: 2` 时，Kubelet 分别去问两个完全独立、老死不相往来的插件：
       - **NVIDIA GPU Device Plugin** 看了看自己的空闲列表，顺手分给你左半边 `NUMA 0 / Switch 0` 上的 **`GPU 0, GPU 1`**；
       - **RDMA Device Plugin** 看了看自己的空闲列表，顺手分给你右半边 `NUMA 1 / Switch 3` 上的 **`mlx5_6, mlx5_7`**！
     - **惊天惨剧**：容器启动后，里头的 `GPU 0` 要想通过 `mlx5_6` 发一个 RDMA 包，数据包必须从 **`GPU 0 -> Switch 0 -> CPU 0 Root Complex -> 跨 CPU 插槽的 UPI/XGMI 互联总线 -> CPU 1 Root Complex -> Switch 3 -> mlx5_6`（即 `nvidia-smi topo -m` 中最烂的 `SYS` 级别）**！原本 `47.5 GB/s` 的满速 GPUDirect RDMA 直接摔碎到 **`11.2 GB/s`（暴跌 `-76.4%`）** ✕！
  2. **灾难现场二：开启 Kubelet `Topology Manager` 的 `single-numa-node` 策略（Tier 2 — 既分不清同一 NUMA 下的两颗 PCIe Switch `PHB`，又把所有 8 卡大任务 100% 误杀拒载！）**：
     - 很多运维听说 Kubelet 有个 `Topology Manager`，赶紧把策略设成最严格的 `--topology-manager-policy=single-numa-node`。结果立刻炸出 **两个大雷**：
       1. **雷一（`NUMA` 粒度太粗，看不见内部的 `PCIe Switch`！）**：Kubelet 的 `Topology Manager` 只认识 **CPU `NUMA` 编号（`0` 或 `1`）**，根本不知道 `NUMA 0` 底下其实还分了 `Switch 0` 和 `Switch 1` 两颗 PCIe 交换芯片！于是它把同属 `NUMA 0` 的 **`GPU 0`（在 `Switch 0`）和 `mlx5_2`（在 `Switch 1`）** 配在了一起！虽然不用跨 CPU 插槽了，但流量依然要上送 CPU 0 的 Root Complex 跨桥折返（**`PHB` 级别**），带宽只有 **`21.5 GB/s`（腰斩！）**！
       2. **雷二（所有申请 `8` 张卡的整机训练 Pod 100% 报 `TopologyAffinityError` 启动失败！）**：当主力训练任务申请整机 `8` 张 GPU + `8` 张网卡时，因为 1 个 `NUMA` 节点只有 4 张卡，8 张卡必然跨越 `NUMA 0` 和 `NUMA 1`！而 `single-numa-node` 死板地规定“所有设备必须来自同一个 NUMA 节点”，直接把全网所有 8 卡大任务的 Pod **100% 拒绝准入（`Admission Rate = 0.0%`）** ✕！
  3. **终极正解：Kubernetes `DRA`（`Dynamic Resource Allocation` 动态资源分配）或 `NRI / CDI` 交换芯片级联合绑定（Tier 4 — `100%` `PIX` 满速 `47.5 GB/s` + `100%` 全规格 Pod 准入）**：
     - 彻底抛弃只能传递“整数个数”的旧版 `Device Plugin` API！
     - **方案 A（K8s 1.31+ 标准 `DRA ResourceClaim`）**：由统一的拓扑感知驱动将每张 GPU 和每张 RNIC 的完整属性（`pcie_switch_id`, `numa_node`）上报至集群，在单个 `ResourceClaim` 中通过 CEL 表达式强制约束 **`nic.attributes["pcie_switch_id"] == gpu.attributes["pcie_switch_id"]`**，100% 保证分配同一颗 PCIe Switch 下的 `(GPU_i, mlx5_i)` 黄金搭档！
     - **方案 B（轻量生产捷径：NRI / Mutating Webhook 附属网卡自动注入）**：Pod 只向 K8s 申请 `nvidia.com/gpu: N`，当容器运行时（Containerd NRI Hook）看到该容器分到了 `GPU 2` 和 `GPU 3`，自动查宿主机 `PIX` 映射表，把与它俩绑在同一颗 PCIe Switch 上的 **`mlx5_2` 和 `mlx5_3` 自动挂载进容器**——**`100%` `PIX` 1:1 匹配，单口跑满 `47.5 GB/s`（提速 `4.24x`），且 1/2/4/8 卡 Pod 全部 100% 丝滑准入** 🏆！

---

## 1. 📌 核心结论速览（Executive Summary — 3 分钟看懂）

1. **为什么说“在 8 卡 + 8 网卡的 AI 服务器上，Kubelet 原生的 `Device Plugin + Topology Manager` 从根子上就是残废的”？**
   - 因为旧版 `Device Plugin` 的 `GetTopologyInfo` 接口只能返回 **`NUMANode` ID**，完全没有 **`PCIeSwitch` ID** 的概念！而在 AI 服务器中，1 个 NUMA 节点下往往挂着 2 颗（甚至 4 颗）独立的 PCIe Switch。光靠 NUMA 对齐，有 `50%` 的概率会把不同 PCIe Switch 的 GPU 和网卡错配成 `PHB`（带宽掉一半）！
   - 更糟的是，Kubelet 的 `single-numa-node` 策略是节点全局静态配置，一旦开启，所有跨双 NUMA 的 `8 卡整机 Pod` 全部报 `TopologyAffinityError` 暴毙！
2. **两大生产破局选型（`Tier 4`）**：
   - 在 **K8s 1.31 / 1.32+** 集群上：启用 **Kubernetes `DRA (Dynamic Resource Allocation)`**，通过 CEL 拓扑约束实现跨设备（GPU + RNIC + 本地 NUMA CPU 核）的 `PCIe Switch (PIX)` 级联合分配；
   - 在 **存量 K8s** 集群上：采用 **“NRI / CDI 伴生网卡自动绑定（GPU-Affiliated NIC Injection）”**——业务只申请 GPU（由 GPU Plugin 按 PCIe Switch 成对分配 `gpu0, gpu1`），由宿主机 NRI 插件根据分到的 GPU 编号 1:1 自动注入同处 `PIX` 的 `mlx5_0, mlx5_1`！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：如果容器里分到的 2 张 GPU 和 2 张网卡在物理上已经是正确的 `PIX` 配对（比如 `GPU 2, GPU 3` 和 `mlx5_2, mlx5_3`），为什么有时候 `NCCL` 还是报错或选错了网卡？
- **答**：因为容器内部看到的设备序号被重映射了！在容器内部，`GPU 2` 和 `GPU 3` 被重命名成了 `cuda:0` 和 `cuda:1`；如果此时环境变量里写死了全局的 `NCCL_IB_HCA=mlx5_0,mlx5_1`（或者挂载了宿主机全量 `/dev/infiniband/uverbs*` 但没过滤），容器里的 `cuda:0`（物理 `GPU 2`）就会去用宿主机的 `mlx5_0`！
- **铁律**：在容器内，不要手动写死错误的 `NCCL_IB_HCA` 索引！让 `NCCL` 通过 `/sys/class/pci_bus` 自动探测与当前 GPU 物理 PCIe BDF 距离最近（`PATH_PIX`）的网卡，或者由 NRI 插件只把配对的那几张 `uverbs` 设备挂载进容器！

### Q2：如果暂时只能改 Kubelet 参数，怎么防止 `Topology Manager` 把 8 卡整机 Pod 拒载？
- **答**：千万不要在混合跑 2 卡、4 卡和 8 卡任务的节点上配 `single-numa-node`！应将 Kubelet 配置为 **`--topology-manager-policy=best-effort`**（或 `restricted` 配合 `--topology-manager-scope=container` 且不为 8 卡容器请求超过单 NUMA 上限的强绑定资源），并由上层 `GPU Operator / NRI / DRA` 接管细粒度 `PIX` 配对。

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + 盲配 `SYS/PHB` vs `DRA/NRI` `PIX` 1:1 绑定图 + 时序图）

### 3.1 架构师 Kubernetes 多 GPU + 多 RDMA 网卡拓扑感知调度选型决策树

```mermaid
flowchart TD
    A["K8s 8卡+8网卡 AI 节点需同时支持 1/2/4 卡切分 Pod 与 8 卡整机 Pod"] --> B{"如何协调 GPU 与 400G RDMA 网卡 (mlx5_0..7) 的分配？"}
    B -- "默认独立 Device Plugin (topology=none, Tier 1)" --> C["⛔ GPU 插件与 RDMA 插件盲目乱配 (GPU0 配 mlx5_7)<br/>跨 CPU 插槽 SYS 绕行 · GDR 带宽暴跌至 11.2 GB/s (-76%)!"]
    B -- "开启 Kubelet Topology Manager (single-numa-node, Tier 2)" --> D["⛔ 同 NUMA 内仍错配不同 PCIe Switch (PHB 21.5 GB/s)<br/>且 8 卡整机 Pod 因跨双 NUMA 100% 报 TopologyAffinityError 拒载!"]
    B -- "按交换芯片硬编码资源名 (gpu_sw0 + rdma_sw0, Tier 3)" --> E["⚠️ 业务 YAML 需绑定物理槽位号，引发严重调度碎片"]
    B -- "K8s DRA (ResourceClaim CEL) 或 NRI 伴生网卡 1:1 自动注入 (Tier 4)" --> F["🏆 100% 锁定同一颗 PCIe Switch (PIX: gpu_i <-> mlx5_i)<br/>单口跑满 47.5 GB/s (提速 4.24x) · 1/2/4/8 卡 Pod 100% 准入!"]
```

### 3.2 `Tier 1/2`（独立插件跨插槽 `SYS` / 同 NUMA 跨桥 `PHB`）vs. `Tier 4`（`DRA / NRI` 交换芯片级 `PIX` 1:1 绑定）对比图

```mermaid
flowchart LR
    subgraph BAD["⛔ Tier 1 & 2: Device Plugin 盲配与 NUMA 粗粒度错配"]
        direction TB
        B1["Tier 1 (无拓扑): GPU 0 (NUMA 0, Sw 0) 配上 mlx5_7 (NUMA 1, Sw 3) -> 跨 CPU 总线 SYS (11.2 GB/s)!"] --> B2["Tier 2 (single-numa): GPU 0 (Sw 0) 配上 mlx5_3 (Sw 1) -> 绕行 CPU Root Complex PHB (21.5 GB/s) + 8卡拒载!"]
    end
    subgraph GOOD["🏆 Tier 4: DRA / NRI 交换芯片级 PIX 1:1 绑定 (47.5 GB/s)"]
        direction TB
        G1["Pod 申请 2 套 (GPU + RNIC) 复合资源"] --> G2["DRA / NRI 按 pcie_switch_id 联合锁定 Switch 0 下的 (GPU 0, GPU 1) 与 (mlx5_0, mlx5_1)!"]
        G2 --> G3["100% PIX 交换芯片内部 150ns 直转 (47.5 GB/s) + 8 卡整机 Pod 无缝放行!"]
    end
```

### 3.3 `Tier 4` Kubernetes `DRA` / `NRI Hook` 联合分配 `PIX` `(GPU + RNIC + NUMA CPU)` 闭环时序图

```mermaid
sequenceDiagram
    autonumber
    participant SCHED as K8s Scheduler + DRA Controller
    participant KUBE as Kubelet + Containerd NRI Hook
    participant HW as PCIe Gen5 Switch k (GPU_2i, GPU_2i+1, mlx5_2i, mlx5_2i+1)

    SCHED->>SCHED: 1. 解析 Pod ResourceClaim: 要求 2 张 GPU 与 2 张 RNIC 满足 cel: gpu.pcie_switch == nic.pcie_switch
    SCHED->>KUBE: 2. 选中节点下空闲的 PCIe Switch 1，原子锁定 (GPU 2, GPU 3) 与伴生的 (mlx5_2, mlx5_3)
    KUBE->>HW: 3. NRI/CDI Hook 将 GPU 2,3、mlx5_2,3 及所属 NUMA 0 的 CPU 核亲和性一并注入容器
    HW->>HW: 4. 容器内 NCCL 自动识别 PATH_PIX (GPU2<->mlx5_2, GPU3<->mlx5_3)，GPUDirect RDMA 跑满 47.5 GB/s!
```

---

## 4. 📊 架构师选型决策矩阵（4-Tier Selection Matrix）

| 评估维度 | Tier 1: 默认独立 `Device Plugin` | Tier 2: `Topology Manager (single-numa)` | Tier 3: 硬编码 `sw0..3` 资源名 | Tier 4: **`K8s DRA` / `NRI` 伴生 `PIX` 绑定** |
| :--- | :--- | :--- | :--- | :--- |
| **单张 400G 网卡实测 `GDR` 带宽** | **`11.2 GB/s` (跨 CPU `SYS` 暴跌!)** ✕ | `21.5 GB/s` (同 NUMA 跨桥 `PHB`) ⚠️ | `47.5 GB/s` ✅ | **`47.5 GB/s` (提速 `4.24x`，100% `PIX` 满线速)** 🏆 |
| **整机 `8-GPU` 训练 Pod 准入成功率** | `100.0%` | **`0.0%` (`TopologyAffinityError` 全拒!)** ✕ | `100.0%` | **`100.0%` (自适应全规格 1/2/4/8 卡 Pod)** 🏆 |
| **`GPU <-> RNIC` 1:1 `PIX` 配对率** | **`12.5%` (随机盲配)** ✕ | `50.0%` (2 颗 Switch 随机混) ⚠️ | `100.0%` (但引发调度碎片) | **`100.0%` (硬件 `pcie_switch_id` 精确绑定)** 🏆 |

---

## 5. 🧪 定量算例与工具链（Interactive Sizer & 7-Card Visual Report）

- **实验脚本与密封数据**：[`experiments/ai_cluster_k8s_dra_nri_topo_lstopo_nic_gpu_numa_affinity_selection/run.py`](../../experiments/ai_cluster_k8s_dra_nri_topo_lstopo_nic_gpu_numa_affinity_selection/run.py) · [`results.json`](../../experiments/ai_cluster_k8s_dra_nri_topo_lstopo_nic_gpu_numa_affinity_selection/results.json)
- **交互式 K8s GPU+RNIC 拓扑感知调度（Device Plugin vs DRA）计算器**：[`docs/reports/k8s-gpu-rdma-nic-topology-manager-dra-numa-affinity-sizer.html`](../reports/k8s-gpu-rdma-nic-topology-manager-dra-numa-affinity-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/k8s-gpu-rdma-nic-topology-manager-dra-numa-affinity-selection-cards.html`](../reports/k8s-gpu-rdma-nic-topology-manager-dra-numa-affinity-selection-cards.html)

---

## 6. 🛠️ Good Fit vs. Not For & 生产环境快速核验清单（Checklist）

### ✅ Good Fit vs. ⛔ Not For

| 维度 | ✅ Good Fit（强烈推荐） | ⛔ Not For（严禁误用） |
| :--- | :--- | :--- |
| **支持 1/2/4/8 卡弹性混部的 K8s RoCE/IB 集群** | 采用 **Kubernetes `DRA (Dynamic Resource Allocation)`** 或 **Containerd `NRI / CDI` 伴生网卡注入器（按 `GPU_i` 1:1 自动挂载同 `PIX` 的 `mlx5_i`）** | 让独立的 GPU Plugin 与 RDMA Plugin 各自盲目分配设备，或在跑 8 卡训练任务的节点上硬开 `single-numa-node` |

### 生产环境上线核验清单（Pre-Flight Checklist）

- [ ] **在任意 2 卡 / 4 卡容器内运行 `nvidia-smi topo -m`**：确认分配进容器的第 $k$ 张 GPU 与第 $k$ 张 `mlx5` 网卡之间全部显示绿色的 **`PIX`**（绝不能出现 `SYS`、`NODE` 或 `PHB`）。
- [ ] **严禁在需运行 8 卡整机 Pod 的双路服务器上将 Kubelet 设为 `single-numa-node`**：避免跨双 NUMA 的 8 卡 Pod 被 `TopologyAffinityError` 拦截。
- [ ] **落地 `NRI / CDI` 伴生网卡绑定或 `K8s DRA` `ResourceClaimTemplate`**：将 `(GPU_i, mlx5_i)` 作为原子拓扑单元协同分配。
