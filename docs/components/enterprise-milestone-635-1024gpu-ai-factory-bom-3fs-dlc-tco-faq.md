# 🏆 第 635 篇里程碑：千卡级企业智算中心（`1,024-GPU AI Factory`）端到端硬件 BOM 清单、8 轨 `Rail-Optimized` 光铜网络、两阶段 `2.4 秒` 异步 Checkpoint + `DeepSeek 3FS` 本地 NVMe 池化、`120kW` 冷板液冷（`PUE 1.11`）与训推潮汐混部三年 TCO 决算圣经（`#635`）

> **Language / 语言**: **简体中文** | [English](./enterprise-milestone-635-1024gpu-ai-factory-bom-3fs-dlc-tco-faq.en.md) · [📱 交互式 H5 千卡智算中心 BOM 与三年 TCO 精算台](../h5/milestone-635-1024gpu-ai-factory-bom-3fs-dlc-tco-lab.html) · [🧮 交互式容量计算器 Sizer](../reports/enterprise-milestone-635-1024gpu-ai-factory-bom-3fs-dlc-tco-sizer.html) · [🃏 7-Card 评测报告](../reports/enterprise-milestone-635-1024gpu-ai-factory-bom-3fs-dlc-tco-cards.html) · [返回总目录](../full-catalog-zh.md)

![架构摘要图](../../assets/svg/enterprise-milestone-635-1024gpu-ai-factory-bom-3fs-dlc-tco-faq-summary.svg)

---

## 一、3 分钟核心结论卡片（Executive Summary）

1. **一句话结论**：
   - 当企业规划建设 **千卡级（`1,024 GPUs = 128 台 × 8 卡 H200/B200`）训推一体智算中心** 时，传统“训练与推理物理分池 + 三层胖树全 DSP 光模块 + 外采数百万美元独立全闪存储阵列 + 15kW 风冷机柜”的烟囱方案不仅导致 **三年 TCO 高达 `$7,420万`**，还会因同步写 Checkpoint 冻结集群（每小时停顿 `240 秒`，浪费 `13.3%` 算力）和夜间推理卡空转（24h 利用率仅 `46.5%`）将单百万 Token 成本推高至 **`$0.245 / 1M`**。
   - 采用 **`#635` 超融合液冷训推一体工厂参考架构**：
     1. **网络 BOM（省 `56.2%` 光模块）**：`32 台 Rail Leaf + 16 台 Spine = 48 台 64口 400G 交换机`，柜内 `<7m` 用 `DAC/ACC` 铜缆、柜间用 `LPO` 线性直驱光模块，实测 `busbw = 47.8 GB/s (95.6%)`；
     2. **存储与 Checkpoint BOM（外置阵列采购费 `$0`）**：通过 **`DeepSeek 3FS` 聚合 128 台计算节点内置的 `1,024 块 3.84TB PCIe Gen5 NVMe`（总容量 `3.93 PB`，`GDS` 聚合带宽高达 `3.58 TB/s`）**，配合 `PyTorch DCP` **两阶段异步快照（`GPU HBM → Host Pinned DDR5` 仅冻结 `2.4 秒`）**，快照停顿提速 **`100 倍`**；
     3. **供电与冷板液冷 BOM（`PUE 1.45 → 1.11`）**：采用 **`16 台 80–120kW 冷板直接液冷机柜（DLC）`** + Decode 池 **`520W` 功耗封顶**，将机房总功耗从 `1.92 MW` 压至 **`1.38 MW`（3 年省电费 `$142万`）**；
     4. **Kueue 训推潮汐共池（利用率 `46.5% → 94.2%`）**：夜间自动将 `448` 张闲置在线推理卡通过 `0.8s Sleep Mode` 切换为 `veRL GRPO` 后训练采样与离线 Batch ETL，使**三年总拥有成本从 `$7,420万` 压降至 `$4,580万`（净省 `$2,840万 / -38.3%`），折合每百万有效 Token 成本仅 `$0.048 / 1M`（降低 `5.10 倍`）**！

---

## 二、1,024-GPU 智算中心标准硬件 BOM 清单与手算公式

### 1. 千卡集群四层标准硬件 BOM 清单（128 台 × 8 卡）

| 子系统模块 | 传统高溢价烟囱方案 | `#635` 架构师最优 BOM 配置 🏆 | 数量规格 | 三年成本对比与净节省 |
| :--- | :--- | :--- | :--- | :--- |
| **1. GPU 计算节点** | 160 台（训推物理拆池含冗余） | **128 台 `8×H200/B200` 共池节点**（配 `2TB DDR5` + `8×3.84TB Gen5 NVMe`） | `1,024 GPUs` | 砍掉 32 台跨池冗余服务器，**省 `$1,120万`** |
| **2. RDMA 智算网络** | 96 台交换机（3层胖树 + 全 DSP 光模块） | **48 台 64口 400G 交换机**（`32 Rail Leaf + 16 Spine` + `DAC/LPO`） | `1,024×400G` 端到端无收敛 | 交换机减半、长距光模块省 `56.2%`，**省 `$410万`** |
| **3. 高性能并行存储** | 独立专有全闪存储阵列（`100–200 GB/s`） | **超融合 `DeepSeek 3FS` 池化内置 `1,024 块 Gen5 NVMe` (`3.58 TB/s`)** | `3.93 PB` 裸容量 | 外置存储阵列 CapEx **`$0`，省 `$680万`** |
| **4. 机柜、液冷与三年电费** | 96 个 `15kW` 风冷机柜（`PUE 1.45`，`1.92 MW`） | **16 个 `100kW+` 冷板直液冷机柜（`PUE 1.11` + `520W` 限电，`1.38 MW`）** | `16 Racks + N+1 CDU` | 机柜缩减 `83%`，加上潮汐增效合省 **`$630万`** |
| **合计（3-Year Total TCO）** | **`$74,200,000`（`$7,420万`）** | **`$45,800,000`（`$4,580万`）** | **`94.2%` 集群利用率** | **三年累计净省 `$28,400,000`（`-38.3%`）** |

```mermaid
flowchart TB
    subgraph Network["🌐 1. 8轨 Rail-Optimized + LPO/DAC 网络层 (48 台 64口 400G 交换机)"]
        SPINE["16 台 Spine 交换机 (无收敛 1:1 互联)"] <-->|"LPO 线性直驱光模块 (-50% 功耗)"| LEAF["32 台 Rail Leaf 交换机 (8 个 Rail × 每 Rail 4 台 Leaf)\n柜内 <7m 采用无源 DAC/有源 ACC 铜缆 · busbw = 47.8 GB/s"]
    end

    subgraph Compute["⚡ 2. 128 台 8-GPU 训推潮汐共池计算层 (1,024 GPUs · 16 个 100kW+ DLC 液冷机柜)"]
        LEAF <-->|"8×400G ConnectX-7 (PIX 同根直通)"| NODES["白天 (08:00–22:00): 640 卡动态 PD 在线推理 + 384 卡预训练/SFT\n夜间 (22:00–08:00): 192 卡在线值守 + 832 卡跑 GRPO 强化学习与百亿词 Batch ETL\n24h 算力有效利用率 = 94.2% · Decode 池开启 520W DVFS 甜点封顶"]
    end

    subgraph Storage["🗄️ 3. 超融合 DeepSeek 3FS + 2.4s 两阶段异步 Checkpoint 存储层 ($0 外置阵列)"]
        NODES -->|"阶段1: HBM ➔ 2TB Pinned DDR5\n(PCIe Gen5 仅冻结 2.4 秒!)"| DRAM["Host Pinned DDR5 内存缓冲池\n训练立即恢复继续跑 Step N+1 (0 算力空转)"]
        DRAM -->|"阶段2: 后台异步 RDMA + GDS 刷盘\n(3.58 TB/s 聚合写入)"| FS3["DeepSeek 3FS 池化 1,024 块本地 Gen5 NVMe (3.93 PB)\n同时作为推理 L3 全局持久化 KV Cache 池"]
    end
```

---

### 2. 两阶段异步 Checkpoint 停顿时间手算公式（`240 秒 → 2.4 秒`）
- 以千卡训练保存一次 `3.2 TB` 的分布式优化器与权重状态为例：
  - **传统同步直写外部 `13.3 GB/s` 共享存储**：全集群 1,024 张 GPU 必须同步挂起等待 $3,200\text{ GB} / 13.3\text{ GB/s} = \mathbf{240.0\text{ 秒}}$（若每 30 分钟存一次快照，光存盘就白白烧掉 **`13.3%`** 的千卡算力！）；
  - **`PyTorch DCP` 两阶段异步快照 + `DeepSeek 3FS`**：
    - **第一阶段（唯一阻塞 GPU 的阶段）**：128 台节点的 GPU 同时通过本地 `PCIe Gen5`（每机有效写入 `220 GB/s`，128 台聚合 **`28.1 TB/s`**）将各自的 `25 GB` 分片拷入本机 **Host Pinned DDR5**，加上元数据同步仅冻结 GPU **`2.4 秒`（提速 `100 倍`）**！
    - **第二阶段（后台无感刷盘）**：训练立即继续运行下一步，后台线程通过 `3FS` 将 DDR5 中的数据异步刷入全集群 `1,024` 块本地 NVMe，算力浪费率从 `13.3%` 骤降至 **`0.13%`**！

---

## 三、千卡智算中心建设与 Day-0 验收核对清单（SA Master Sign-Off Checklist）

- [ ] **1. 交换机收敛比与光铜混合介质核验**：64口 400G Leaf 下联 32 个同轨 GPU 端口、上联 32 个 Spine 端口（严格 `1:1` 无超订）；同柜 `<7m` 强制用 `DAC/ACC` 铜缆，跨柜 `<50m` 用 `LPO` 省电省钱；
- [ ] **2. 本地 NVMe 规格对齐 `3FS + GDS`**：每台 8 卡节点标配 `8 块 3.84TB/7.68TB PCIe Gen5的企业级高耐久 NVMe`，与 GPU/RDMA 网卡同挂在专用 PCIe Switch（`PIX`）下开启 `nvidia-fs (GDS)`；
- [ ] **3. 开启 `PyTorch DCP` 异步主机内存暂存与 `vLLM Sleep Mode` 潮汐切换**：确保每日 `22:00` 在线推理缩容释放的 GPU 在 `15 秒` 内无缝注入训练/GRPO 作业池。
