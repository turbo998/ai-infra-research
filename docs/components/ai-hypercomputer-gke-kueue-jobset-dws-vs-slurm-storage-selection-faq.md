# AI Hypercomputer 集群控制面选型：`GKE (Kueue TAS + JobSet + DWS + Multislice)` vs. `Slurm (Cluster Toolkit)`、五种算力供给模式与三级训练存储（`Hyperdisk ML / GCS FUSE / Lustre + Orbax`）架构指南

> **适用读者**：负责千卡至万卡级 TPU（`v5p / v6e / TPU7x Ironwood / 8t & 8i`）与 GPU（`H200 / B200 / GB200`）集群编排调度、容量采买（Reservation / DWS / CUD）与高性能训练存储底座设计的 Cloud & AI Infrastructure Solution Architect、智算平台工程负责人与基础设施 SRE 总监。
> **核心课题**：在构建 **`1,024` 芯片级 AI Hypercomputer 训推平台（每 30 分钟保存一次 `4.2 TB` 分布式 Checkpoint）** 时，为什么直接用“原生 Kubernetes 默认调度器跑 Spot 实例 + 无缓存 `GCS FUSE` 同步写 Checkpoint（Tier 1）”会导致 **`28.0%` 的算力死锁空转（只拉起 `220/256` 个 Pod 卡死等待）、每次保存 Checkpoint 全集群冻结 `185 秒`、端到端有效吞吐（Goodput）跌至 `54.5%`（每月浪费 `$150 万` 算力）**？如何在 **`GKE (Kueue 四件套 + Topology-Aware Scheduling [TAS] + JobSet + XPK)`** 与 **`Slurm (Cluster Toolkit)`** 之间拍板？又如何通过 **“基荷 `Specific Reservation + COMPACT` 同子块紧凑放置 + 波峰 `DWS Flex-Start`（严守 `10 分钟` 消费窗口与 `7 天` 轮换）+ `Hyperdisk ML`（模型权重）+ `GCS FUSE 本地 NVMe 缓存`（训练分片）+ `Managed Lustre`（POSIX 热暂存）+ `Orbax` 两阶段异步与本地紧急 Checkpoint”** 四级架构，将 Checkpoint 停顿压缩至 **`2.4 秒`**、死锁浪费降至 **`0.0%`**、集群 Goodput 推升至 **`96.8%`**，并将月度综合 TCO 降低 **`60.5%`（`$2.48M -> $980K/月`）**？

![AI Hypercomputer 集群控制面选型：GKE (Kueue TAS + JobSet + DWS) vs. Slurm 与三级训练存储架构总结图](../../assets/svg/ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（3 分钟拍板）

在千卡级 AI Hypercomputer 中，**控制面与存储的选型直接决定了你买的昂贵芯片到底是在“算梯度”，还是在“等齐人（Gang Deadlock）、等拉镜像（DWS Timeout）、等写盘（Checkpoint Stall）”**：

| 控制面与存储选型层级 | 适用场景 | 端到端集群有效吞吐 (`Goodput`) | `4.2 TB` Checkpoint 停顿耗时 | 成组调度死锁/超时浪费率 | 月度 `1,024` 芯片算力与存储总成本 (`TCO`) | 最终拍板结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1：原生 K8s 默认调度器 (`Spot`) + 无缓存 `GCS FUSE` 同步写盘（反模式）** | ❌ **严禁用于任何多机分布式训练或 Multislice** | **`54.5%`（近半算力空转！）** | **`185.0 秒`** | **`28.0%`（凑不齐卡死锁）** | `$2,480,000 / 月` | ❌ **严禁上生产**：逐 Pod 调度引发死锁，同步写对象存储拖垮千卡集群 |
| **Tier 2：独立 `Slurm (Cluster Toolkit)` + 全天候常驻 `Managed Lustre`** | 100% 纯批处理科研 HPC 实验室（无在线推理微服务混部需求） | `86.4%` | `28.0 秒` | `4.5%` | `$1,790,000 / 月` | ✅ **纯科研 HPC 适用**：批作业简洁，但与 K8s 在线推理割裂且全天 Lustre 贵 |
| **Tier 3：`GKE JobSet + DWS Flex-Start`（无 `Kueue TAS` 拓扑钉死与紧急快照）** | 中小规模单 Slice 微调任务 | `81.2%` | `42.0 秒` | `6.2%`（镜像慢触发 10m 回收！） | `$1,460,000 / 月` | ⚠️ **跨块通信与超时隐患**：节点散落在不同交换机子块，且易踩 DWS 10 分钟红线 |
| **Tier 4：`GKE Kueue TAS + JobSet + Reservation/DWS` 分层 + `Hyperdisk ML` + `Orbax` 异步/紧急快照（推荐）** | **企业级训推共池、Multislice 大模型预训练与 RL 主力架构** | **`96.8%`** | **`2.4 秒`（提速 `77 倍`）** | **`0.0%`（全有或全无原子交付）** | **`$980,000 / 月`（省 `60.5%`）** | 🏆 **生产金标准**：Kueue TAS 锁死物理拓扑，三级存储各司其职，秒级容错 |

### 💡 架构师的三大黄金铁律
1. **彻底分清五种算力采买与交付机制（`Spot` vs. `DWS Flex-Start` vs. `DWS Calendar` vs. `Reservation + COMPACT` vs. `CUD`），绝不把“财务折扣（CUD）”当成“物理留座（Reservation）”**：
   * **财务承诺折扣（`1-Yr / 3-Yr CUD`）** 只管降单价，**不保证机房里有物理卡**！要锁死基荷物理产能，必须绑定 **指定预留（`Specific/Targeted Reservation`）** 并显式声明同可用区 **`COMPACT` 紧凑放置策略**（确保所有节点落在同一个物理 `Block / Subblock` 或完整 `4x4x4` TPU Cube 内）。
   * 对于弹性批处理与突发训练，使用 **动态工作负载调度器（`DWS Flex-Start`）**：它会排队等待直到 `100%` 的芯片（如整块 `256` 芯片 Slice）同时就绪才原子下发。**但架构师必须死守 `DWS Flex-Start` 的两条生死红线**：
     1. **`10 分钟` 消费窗口（10-Minute Consumption Window）**：一旦 DWS 批下节点，你的容器必须在 **10 分钟内完成镜像拉取并启动跑起来**，否则系统判定闲置直接收回算力！（必须开启 **GKE Image Streaming** 将 25GB 镜像启动压至 `<35s`）。
     2. **`7 天` 最大连续运行时长（`maxRunDurationSeconds <= 604800`）**：DWS Flex-Start 实例最长运行 7 天即强制回收，必须在 `JobSet` 中配置第 6 天 23 小时自动触发 `Orbax` 优雅保存并无缝重入队列续训！
2. **用“`Kueue` 四件套 + `Topology-Aware Scheduling (TAS)` + `JobSet`”打通 GKE 多切片（Multislice）成组拓扑调度，并锁死“7 层软件栈哈希对齐”**：
   * **`Kueue` 核心四对象**：`ResourceFlavor`（映射底层的 `Reservation` 保底池、`DWS Flex-Start` 弹性池、`Spot` 抢占池）$\rightarrow$ `ClusterQueue`（定义跨团队 `Cohort` 额度借用与高低优抢占）$\rightarrow$ `LocalQueue`（业务命名空间入口）$\rightarrow$ `Workload / ProvisioningRequest`。
   * **`JobSet` + `Kueue TAS`**：`JobSet` 将多个 TPU Slice（或 GPU 节点组）建模为带主从协调域的 `ReplicatedJobs`（支持单 Slice 故障全组协同重启）；`Kueue TAS` 读取节点上的物理层级标签（`cloud.google.com/gce-topology-block -> subblock -> host`），确保高带宽通信轴 `100%` 紧凑落在同一台 TOR 交换机/光交换子块内！
3. **训练存储严禁“一刀切”：按 I/O 形态拆分为“`Hyperdisk ML`（模型权重广播）+ `GCS FUSE + 本地 NVMe 缓存`（海量顺序语料）+ `Managed Lustre`（高频随机 POSIX）+ `Orbax` 异步与本地紧急 Checkpoint”**：
   * 把 `4.2 TB` Checkpoint 通过 `Orbax` 先在 **`2.4 秒`** 内拷贝至主机内存/本地 NVMe SSD，后台异步并发刷入 `GCS`，并在收到抢占信号（`SIGTERM`）时由 **`Emergency Checkpointer`** 在 `<5 秒` 内将状态落入本地 NVMe，彻底消灭 IO 等待！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A

### Q1：五种算力获取模式（`Spot` / `DWS Flex-Start` / `DWS Calendar` / `Reservation` / `CUD`）到底该怎么组合搭配？
* **一张表看懂 AI Hypercomputer 五种算力供给与计费组合拳**：

| 供给/计费模式 | 核心机制与物理交付保证 | 关键限制与生产踩坑红线 | 最佳适用工作负载 |
| :--- | :--- | :--- | :--- |
| **`Spot VMs / TPUs`** | 价格最低（省 `60%–70%`），有闲置即分配，随时可能被提前 `30 秒` 抢占回收 | 无成组原子保证；千卡作业中单机被抢占会导致全组重算 | 短周期探索实验、无状态离线批推理、具备 `<5s` 本地紧急快照的小规模容错训练 |
| **`DWS Flex-Start`** | 通过 `ProvisioningRequest` 排队，**攒齐 100% 所需整块拓扑后原子交付**，运行期抢占率远低于 Spot | ① **双注解必填**（`maxRunDurationSeconds` + `safe-to-evict: "false"`）；② **`10 分钟` 内必须拉起容器消费**；③ **单次最长运行 `7 天`（`604,800s`）自动轮换** | 1 小时至 7 天内的多机/多 Slice 微调、后训练（RLHF/GRPO）、大批量离线评测 |
| **`DWS Calendar Mode`** | 提前预约未来指定日期窗口（`1–90 天`）的整块算力，到点准时交付 | 需提前规划排期窗口 | 配合产品发版日历的定期大模型冲刺训练、季度全量重训 |
| **`Future / Specific Reservation + COMPACT`** | **最高物理容量保障**：在指定可用区锁定专属物理机柜，配合 `COMPACT` 放置策略确保最短网络跳数 | 即使空置也计费（因此必须用 `Kueue Cohort` 在训练空闲时塞满离线批处理/夜间推理） | 持续数周至数月的千卡/万卡旗舰基座预训练与核心在线推理保底池 |
| **`1-Yr / 3-Yr CUD`（承诺使用折扣）** | **纯财务计费折扣（省 `37%–55%+`）**，必须与 `Reservation` 绑定才能同时获得“有卡用 + 价格低” | 单独买 CUD 不绑定 Reservation 在热门可用区依然会报 `STOCKOUT`！ | 集群 24x7 常年稳态基荷容量 |

### Q2：控制面到底选 `GKE (Kueue + JobSet + XPK)` 还是 `Slurm (Cluster Toolkit)`？为什么多主机环境最怕“7 层软件栈版本错配”？
* **GKE vs. Slurm 拍板准则**：
  * **选 `Slurm (via Google Cloud Cluster Toolkit)`**：团队 100% 是传统 HPC/科研背景、拥有大量现成 `#SBATCH` 脚本、挂载共享 `Managed Lustre` `/home` 目录、且**没有任何在线微服务或网关混部需求**。
  * **选 `GKE + Kueue TAS + JobSet`（搭配官方 `xpk` [Accelerated Processing Kit] CLI 极速下发）**：企业需要将 **预训练、RL 后训练、在线 `vLLM` 推理与 CI/CD 评测统一纳管在同一个云原生资源池**，利用 `Kueue` 实现白班保推理、夜间借闲置卡跑训练的潮汐复用。
* **多主机 7 层软件栈版本兼容铁律（7-Layer Stack Compatibility Matrix）**：
  * 在多机 GPU 或多主机 TPU Slice 中，只要有 1 台节点的某一层版本与其他节点差一个小版本号，就会引发诡异的集合通信挂死（Hang）或 XLA 哈希不一致重编译：
    1. **L1 硬件与链路固件**：`BMC / BIOS / OCS / PCIe Switch` 固件
    2. **L2 宿主机内核与设备绑定**：`Container-Optimized OS (COS) / Ubuntu Kernel` + `VFIO / gVNIC` 驱动
    3. **L3 加速器运行时驱动**：`libtpu.so`（必须全 Slice 字节级一致！） / `NVIDIA GPU Driver (550/560+)`
    4. **L4 集合通信与网络插件**：`ICI / TCPXO / gIB / NCCL + GPUDirect RDMA` 插件
    5. **L5 编译器与内核库**：`XLA Compiler / Pallas` 或 `CUDA Toolkit / cuDNN / Triton`
    6. **L6 深度学习框架**：`JAX + jaxlib` 或 `PyTorch + torch_xla`
    7. **L7 模型与存储套件**：`MaxText / vLLM-TPU / JetStream + Orbax Checkpointer`

### Q3：同样是存数据，`GCS FUSE`、`Hyperdisk ML` 和 `Managed Lustre` 三者怎么分工才不会既慢又贵？
* **三级训练存储黄金分工表**：

| 存储产品 | 核心性能特征与优势 | 致命误用陷阱（不要干什么） | 架构师标准定位 |
| :--- | :--- | :--- | :--- |
| **`Hyperdisk ML`** | 支持只读挂载到**数百至数千台计算节点**，聚合读取吞吐高达 **`1.2 TB/s`**，成本远低于并行文件系统 | 不支持多节点并发随机写入（非共享读写 POSIX 文件系统） | **Tier-0 共享只读层**：存放百 GB 基础大模型权重、容器基础环境与高频共享词表，千卡秒级并发加载 |
| **`GCS + GCS FUSE (开启本地 NVMe SSD 缓存 + 并行复合上传)`** | 无限容量、单价最低；开启本地临时 NVMe 缓存后第 2 个 Epoch 起全走本地 `6 GB/s` SSD | ❌ **严禁不挂本地 NVMe 缓存直接跑随机小文件读取**，或用单线程同步写 Checkpoint | **Tier-1 主力语料与持久归档层**：存放 PB 级 `ArrayRecord / TFRecord / Parquet` 大分片训练语料与最终 `Orbax` 归档 |
| **`Google Cloud Managed Lustre`** | 原生 POSIX 语义、微秒级元数据延迟、支持千节点高并发随机读写（高达数百 GB/s） | 24x7 常驻保存冷历史数据成本高 | **Tier-2 高频 POSIX 热暂存层**：供 `Slurm` 共享目录、高频小文件预处理与超大规模同步/半异步热 Checkpoint 缓冲 |

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + 端到端参考架构 + 时序图）

### 3.1 AI Hypercomputer 控制面、算力供给与三级存储选型决策树

```mermaid
flowchart TD
    A["千卡级 AI Hypercomputer 集群构建需求"] --> B{"团队工作流与业务形态?"}
    B -- "纯科研批处理 / 强依赖传统 #SBATCH 与共享 POSIX /home" --> C["控制面: Slurm (Cluster Toolkit) + Managed Lustre 热存储 + GCS 冷归档"]
    B -- "企业训推一体共池 / 多团队配额借用 / Multislice 大模型" --> D["控制面: GKE + Kueue (四件套 + TAS) + JobSet + XPK (Tier 4)"]

    D --> E{"算力负载生命周期分类?"}
    E -- "24x7 核心基座预训练 & 在线推理基荷" --> E1["Specific Reservation + 同可用区 COMPACT 放置 + 1/3 年 CUD"]
    E -- "1 小时~7 天弹性微调 / RL / 评测" --> E2["DWS Flex-Start (开启 Image Streaming <35s 破 10m 窗口 + 第 7 天自动续训)"]
    E -- "可随时中断的短平快探索任务" --> E3["Spot 抢占池 + Kueue 空闲额度借用 (Cohort Borrowing)"]

    E1 & E2 & E3 --> F["三级存储与容错闭环:\n① Hyperdisk ML (1.2 TB/s 广播基础权重)\n② GCS FUSE + Local NVMe SSD Cache (分片语料)\n③ Orbax 两阶段异步 (2.4s) + 本地 NVMe 紧急快照 (<5s)"]
```

### 3.2 Tier 4 `Kueue TAS + JobSet` 四件套编排与三级存储端到端架构图

```mermaid
flowchart LR
    subgraph Kueue["🎛️ 1. GKE + Kueue TAS + JobSet 控制面"]
        LQ["LocalQueue (团队命名空间)\n提交 Multislice JobSet"] --> CQ["ClusterQueue (Cohort 额度借用)\n+ Kueue TAS 拓扑感知调度器"]
        CQ --> RF1["ResourceFlavor #1: 保底池\nReservation + COMPACT 子块亲和"]
        CQ --> RF2["ResourceFlavor #2: 弹性池\nDWS Flex-Start (双注解 + 10m 窗口)"]
    end

    subgraph Compute["⚡ 2. 拓扑对齐计算切片 (TPU Cubes / GPU Subblocks)"]
        RF1 & RF2 ==>|"全有或全无原子交付"| SLICE["JobSet ReplicatedJobs (Slice #1..#4)\n7 层软件栈哈希锁死 + Image Streaming (<35s)"]
    end

    subgraph Storage["💾 3. 三级训练存储与 Orbax 容错闭环"]
        HD["Hyperdisk ML (只读挂载)\n1.2 TB/s 秒级分发基础权重"] ==> SLICE
        NVME["节点本地 Local NVMe SSD\n① GCS FUSE 读缓存 ② <5s 紧急快照"] <==> SLICE
        NVME -.->|"Orbax 后台异步多线程刷盘\n(主线程仅停顿 2.4s)"| GCS["Google Cloud Storage (GCS)\n平行复合上传持久归档"]
    end
```

### 3.3 全链路 `DWS Flex-Start` 10 分钟极速拉起、`Orbax` 2.4s 异步存盘与第 7 天无缝轮换时序图

```mermaid
sequenceDiagram
    autonumber
    participant Team as 算法团队 (xpk / kubectl)
    participant Kueue as GKE Kueue TAS + DWS 控制器
    participant Nodes as 4x256 芯片 Multislice 节点池 (含本地 NVMe)
    participant Storage as Hyperdisk ML + GCS

    Team->>Kueue: 提交 1,024 芯片 `JobSet` (带 `maxRunDurationSeconds: "604800"` 与 `exclusive-topology: subblock`)
    Kueue->>Kueue: 通过 `ProvisioningRequest` 向 `DWS Flex-Start` 排队申请 4 个完整拓扑切片
    Kueue->>Nodes: `DWS` 凑齐 1,024 芯片原子下发! 启动 `10 分钟` 消费倒计时窗口
    Storage->>Nodes: `GKE Image Streaming` + `Hyperdisk ML` (`1.2 TB/s`) 在 `32 秒` 内完成容器启动与基础权重挂载 (完美跑赢 10m 窗口!)
    Nodes->>Nodes: 训练每 30 分钟触发 `Orbax`: 设备张量 `2.4 秒` 拷入主机 Pinned RAM/本地 NVMe，训练立即继续第 N+1 步
    Nodes--)Storage: 后台异步线程将 `4.2 TB` 分片并行上传至 `GCS` (零阻塞计算)
    Note over Kueue,Nodes: 突发单机硬件预警或到达第 6 天 23 小时 (7d 轮换前夕)
    Nodes->>Nodes: `Orbax Emergency Checkpointer` 在 `4.1 秒` 内将最新步状态写入本地/邻居 NVMe 并同步元数据
    Kueue->>Nodes: `JobSet` 自动重拉替换切片，从最新步无缝恢复续训 (`Goodput = 96.8%`)
```

---

## 4. 📊 架构师多方案选型决策矩阵

| 评估维度 | Tier 1：原生 K8s (`Spot`) + 无缓存同步 `GCS FUSE` | Tier 2：独立 `Slurm (Cluster Toolkit)` + 常驻 `Lustre` | Tier 3：`GKE JobSet + DWS`（无 `Kueue TAS` 与紧急快照） | Tier 4：`GKE Kueue TAS + JobSet + DWS/Res` + `Hyperdisk ML` + `Orbax` 异步/紧急快照（推荐） |
| :--- | :--- | :--- | :--- | :--- |
| **端到端集群有效吞吐 (`Goodput`)** | **`54.5%`（死锁 + 同步写盘吞噬算力）** | `86.4%` | `81.2%` | **`96.8%`（行业标杆级有效算力转化率）** |
| **`4.2 TB` Checkpoint 训练停顿耗时** | **`185.0 秒`（全集群空等对象存储）** | `28.0 秒` | `42.0 秒` | **`2.4 秒`（`Orbax` 两阶段内存/NVMe 异步落盘，提速 `77x`）** |
| **成组调度死锁与 `10m DWS` 超时浪费** | **`28.0%`（凑不齐卡白烧钱）** | `4.5%` | `6.2%`（冷拉巨型镜像超 10m 被收回） | **`0.0%`（`Kueue` 全有或全无 + `Image Streaming <35s`）** |
| **物理网络拓扑紧凑度 (`Subblock`)** | 随机打散（跨 Spine 拥塞） | 依赖人工指定拓扑脚本 | 跨子块随机分配（AllReduce 慢 `+22%`） | **✅ `Kueue TAS` + `COMPACT` 强制锁死同一物理子块/Cube** |
| **训推一体共池与跨团队额度借用** | ❌ 无队列借用与成组抢占 | ❌ 与 K8s 在线推理彻底割裂 | ⚠️ 仅基础队列 | **✅ `Kueue Cohort` 夜间自动借用推理闲置卡跑批处理** |
| **月度 `1,024` 芯片算力与存储总 TCO** | `$2,480,000 / 月` | `$1,790,000 / 月` | `$1,460,000 / 月` | **`$980,000 / 月`（节省 `60.5%`）** |

---

## 5. 🧪 容量/成本拐点算例与 7-Card 报告

### 5.1 基准算例假设（`1,024` 芯片规模集群 / 每 30 分钟保存 `4.2 TB` Checkpoint）
* **业务规模**：企业运营 `1,024` 颗旗舰加速器（4 个 256 芯片 TPU Slice 或 128 台 8-GPU 节点），运行多切片大模型预训练与 RL 后训练，每 30 分钟写入一次 `4.2 TB` 分布式状态快照。
* **工程实测与仿真代码**：见 [`experiments/ai_hypercomputer_gke_kueue_jobset_dws_vs_slurm_storage_selection/run.py`](../../experiments/ai_hypercomputer_gke_kueue_jobset_dws_vs_slurm_storage_selection/run.py) 与 [`results.json`](../../experiments/ai_hypercomputer_gke_kueue_jobset_dws_vs_slurm_storage_selection/results.json)（数据来源：`synthetic + anonymized_reported`，`independently_reproduced=false`）。
* **交互式容量与 TCO 计算器**：📊 [打开 AI Hypercomputer 控制面与三级存储容量计算器 (`ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-sizer.html`)](../reports/ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-sizer.html)
* **7-Card 4:5 评测报告卡**：🃏 [打开 7-Card 视觉评测报告 (`ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-selection-cards.html`)](../reports/ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Vanilla K8s Spot + Sync GCS", "goodput_pct": 54.5, "ckpt_stall_sec": 185.0, "deadlock_waste_pct": 28.0, "monthly_tco_m": 2.48 },
      { "tier": "Tier 2: Standalone Slurm + Lustre", "goodput_pct": 86.4, "ckpt_stall_sec": 28.0, "deadlock_waste_pct": 4.5, "monthly_tco_m": 1.79 },
      { "tier": "Tier 3: GKE JobSet + DWS (No TAS)", "goodput_pct": 81.2, "ckpt_stall_sec": 42.0, "deadlock_waste_pct": 6.2, "monthly_tco_m": 1.46 },
      { "tier": "Tier 4: Kueue TAS + DWS + Orbax Async", "goodput_pct": 96.8, "ckpt_stall_sec": 2.4, "deadlock_waste_pct": 0.0, "monthly_tco_m": 0.98 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "goodput_pct": "Percentage",
    "ckpt_stall_sec": "Quantity",
    "deadlock_waste_pct": "Percentage",
    "monthly_tco_m": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Achieves 96.8% Cluster Goodput, 2.4s Checkpoint Stalls, and Cuts Monthly TCO by 60.5%",
    "subtitle": "Monthly 1,024-chip effective compute and storage TCO ($ USD Millions)",
    "encodings": {
      "x": "tier",
      "y": "monthly_tco_m",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ 架构评审与生产落地 Checklist

- [ ] **`Reservation + COMPACT` 与 `CUD` 双绑定核验**：采购 1 年/3 年承诺使用折扣（`CUD`）时，是否已同时创建绑定同可用区 `COMPACT` 紧凑放置策略的 `Specific Reservation`，杜绝“有财务折扣但抢不到同子块物理卡”？
- [ ] **`DWS Flex-Start` 双注解、`10 分钟` 窗口与 `7 天` 轮换闭环**：使用 `DWS Flex-Start` 的节点池是否已开启 `GKE Image Streaming`（确保 `<35 秒` 拉起容器不超时 10 分钟消费窗口），并在 `JobSet` 中配置 `maxRunDurationSeconds <= 604800` 及第 7 天到期前的自动 Checkpoint 续训？
- [ ] **`Kueue TAS`（拓扑感知调度）与 `JobSet` 成组对齐**：是否在 `JobSet` 中配置了 `exclusive-topology: cloud.google.com/gce-topology-subblock`（或完整 TPU Cube），确保多机集合通信不跨远端交换机？
- [ ] **7 层软件栈容器镜像不可变锁死**：是否在 CI/CD 中将 `libtpu / NVIDIA Driver`、`NCCL / gIB`、`XLA`、`JAX / PyTorch` 与 `Orbax` 版本锁死在单一只读镜像 Digest 中，防止多主机节点间版本漂移？
- [ ] **三级存储（`Hyperdisk ML + 本地 NVMe 缓存 GCS FUSE + Orbax 异步/紧急快照`）验证**：是否已将基础权重迁移至 `Hyperdisk ML`、为 `GCS FUSE` 挂载本地 NVMe SSD 缓存目录（`--file-cache-max-size-mb`），并验证 `Orbax` 异步存盘停顿 `<3 秒`？
