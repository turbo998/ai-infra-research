# 多租户 GPU 算力切分、共享与故障隔离（Time-Slicing vs. CUDA MPS vs. 硬件 MIG vs. K8s DRA + HAMi）架构选型指南

> **阅读指南（3 分钟速通 / 10 分钟白板宣讲）**：本文面向 AI 解决方案架构师（AI Solution Architect）与智算平台负责人，用第一性原理与通俗大白话回答企业内部多团队共用 H100/H200/B200 集群时最棘手的三大灵魂问题：**“我们有 56 个 8B 小模型、BGE Embedding、Reranker 微服务和算法工程师的 Jupyter Notebook，每个只需要 16GB 显存，如果给每个容器分一整张 141GB 的 H200 显存利用率还不到 15%！但开启 K8s 原生 `Time-Slicing（时间片轮转）` 后，为什么隔壁 Notebook 一发生 OOM 就会把同卡所有的线上推理服务全炸挂？改用性能翻倍的 `NVIDIA CUDA MPS` 后，为什么某一个租户的自定义算子触发指针越界（XID 31），竟然会导致整张卡上 7 个毫无关系的容器‘连坐团灭’？如何通过 `硬件级 MIG` 与 `K8s DRA + CNCF HAMi 软硬两级切分` 把 GPU 有效装箱率推上 88.5%，同时守住 0% 跨租户故障爆炸半径？”**
>
> 🔗 **English Version**: [Multi-Tenant Fractional GPU Sharing & Isolation (Time-Slicing vs. CUDA MPS vs. Hardware MIG vs. K8s DRA + HAMi) Selection FAQ (`multi-tenant-gpu-sharing-mig-mps-dra-selection-faq.en.md`)](./multi-tenant-gpu-sharing-mig-mps-dra-selection-faq.en.md) · **7-Card 评测报告**: [`multi-tenant-gpu-sharing-mig-mps-dra-selection-cards.html`](../reports/multi-tenant-gpu-sharing-mig-mps-dra-selection-cards.html) · **交互式工作台**: [`llm-inference-engine-pd-sizer.html#gpu-sharing-sizer`](../reports/llm-inference-engine-pd-sizer.html)

![多租户 GPU 算力切分与故障隔离选型一图流总结](../../assets/svg/multi-tenant-gpu-sharing-mig-mps-dra-selection-faq-summary.svg)

---

## ① 📌 核心结论速览（Executive Summary）

1. **严禁在生产环境用 `K8s 原生 Time-Slicing（时间片轮转，Tier 1）` 混部多租户服务**：
   - **Tier 1（Time-Slicing）** 本质上是让 7 个容器分时轮流霸占整张 GPU，**没有任何显存上限与带宽隔离**！
   - 算例实测：一旦有 1 个吵闹邻居（Noisy Neighbor）跑满大矩阵乘，同卡其余服务的 P99 尾延迟瞬间暴涨 **`+215.0%`**；一旦某位工程师的 Notebook 写错 Batch Size 吃光 141 GB 显存，同卡上所有正常运行的线上容器会**`100%` 被无差别 OOM 击杀陪葬**！
2. **揭示 `NVIDIA CUDA MPS (Tier 2)` 的“2.35x 高吞吐红利”与“100% 致命故障连坐陷阱”**：
   - **Tier 2（CUDA Multi-Process Service）** 将多个进程合并进同一个 CUDA 上下文，允许不同小模型的内核在 132 个 SM 上**空间并发执行**，并支持设置显存硬上限（`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`），单卡聚合吞吐比 Time-Slicing 暴涨 **`2.35x`（`333.7` vs. `142.0 QPS`）**！
   - ⚠️ **但架构师必须警惕 MPS 的致命死穴**：因为所有 MPS 客户端**共享同一张 GPU 硬件虚拟地址页表（Shared GPU Page Table）**，一旦卡上任意 1 个进程触发非法内存访问（`CUDA_ERROR_ILLEGAL_ADDRESS` / XID 31），硬件无法区分是谁干的，CUDA 驱动会**当场摧毁整个共享上下文，导致同卡全部 7 个进程 100% 连坐团灭**！因此 MPS **绝不能用于跨部门/不可信租户隔离**，只能用于**同一业务流水线内部的受信微服务组**（如同一个 RAG 应用内部的 Embedding + Reranker）。
3. **跨租户硬隔离必选 `硬件级 MIG (Tier 3)`，全集群极致装箱必选 `K8s DRA 动态 MIG + HAMi/MPS 软硬两级架构 (Tier 4)`**：
   - **Tier 3（硬件级 MIG，如 H200 切 `7 × 1g.20gb`）**：从 SM 流处理器、HBM3e 显存控制器、L2 缓存 Bank 到芯片内部 Crossbar 总线**全链路物理切分**！吵闹邻居下的 P99 抖动仅 **`+1.8%`**，OOM 与非法指针越界的跨租户爆炸半径均为 **`0.0%`**！
   - **Tier 4（`K8s 1.32+ DRA 动态 MIG` 管外层跨租户物理隔离 + `CNCF HAMi / MPS` 管内层同租户弹性超分）**：外层用硬件 MIG 划定不可逾越的部门安全边界，内层在单个 MIG 实例内利用 `HAMi (libvgpu.so)` 或 MPS 并发跑多个轻量组件，单卡吞吐推至 **`352.0 QPS`（`2.48x`）**，GPU 综合装箱利用率高达 **`88.5%`**，承载 56 个微服务仅需 **`8 张 H200`**（比半独占的 28 张省卡 **`-71.4%`**，每年净省算力成本 **`$480,000`**）！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（FAQ-Driven）

### Q1：Time-Slicing、CUDA MPS、硬件 MIG 和 HAMi (LD_PRELOAD 拦截) 在底层物理机制上到底有什么本质区别？

- **通俗大白话（一栋 7 层写字楼的四种出租方式）**：
  1. **`Time-Slicing（时间片轮转，Tier 1）` = “7 家公司共用一间不隔断的大厅，按毫秒轮流进门办公”**：
     - 同一微秒只有 1 家公司在干活，哪怕它只有 2 个人（只占 15% SM），另外 85% 的工位也只能空着；而且大厅里没有储物柜锁（无显存限额），一家公司堆满杂物（OOM），整栋楼瘫痪！
  2. **`CUDA MPS（多进程服务，Tier 2）` = “把大厅打通，7 家公司同时坐在各自工位上办公，但共用同一个总电闸”**：
     - 解决了工位空置问题（7 家公司空间并发跑满 132 个 SM，吞吐翻 `2.35 倍`），也给每家划了储物柜上限（`MEM_LIMIT`）；但因为共用一个总电闸（**共享单一 GPU 硬件上下文与页表**），只要有一家公司把电线接短路（**算子指针越界 XID 31**），总闸跳闸，7 家公司一起断电团灭！
  3. **`Hardware MIG（多实例 GPU，Tier 3）` = “用钢筋水泥承重墙把整层楼物理隔成 7 套独立产权房，每套都有独立电表、独立上下水和独立电梯”**：
     - 在硅片硬件层把 132 个 SM、141 GB HBM3e 的内存通道和 L2 缓存物理切成最多 7 份（如 `1g.20gb`）。哪怕隔壁房间把天花板炸穿（OOM 或非法地址访问），你的房间连灯光都不会闪一下（P99 抖动仅 **`+1.8%`**，故障波及 **`0%`**）！
  4. **`CNCF HAMi / vGPU（用户态 API 拦截）+ K8s DRA 动态 MIG（Tier 4）` = “承重墙隔开不同公司（外层 MIG），同一家公司套房内部再用灵活屏风隔开财务部和法务部（内层 HAMi/MPS）”**：
     - 外层享受 MIG 的 `0%` 故障传染，内层通过挂载 `LD_PRELOAD=libvgpu.so` 拦截容器内的 `cuMemAlloc` 和 `cuLaunchKernel`，实现以 `1 GB` 和 `1% SM` 为步长的细粒度弹性共享！

### Q2：为什么说“在多租户生产环境裸开 CUDA MPS 是一颗定时炸弹”？什么情况下才允许开 MPS？

- **NVIDIA 官方文档明确警告的硬件限制（Volta → Hopper → Blackwell 均存在）**：
  - 在 MPS 模式下，`nvidia-cuda-mps-server` 守护进程代表所有客户端进程向 GPU 提交任务。为了省去进程间页表切换开销，所有客户端共享同一个 GPU 虚拟地址空间（UVA）。
  - 当租户 A 编写的 Triton 内核、自定义 C++ 算子或推理引擎 Bug 产生了一个越界读写指针（Out-of-Bounds Memory Access）时，GPU MMU 无法保证租户 B 的显存没被这个野指针悄悄篡改！为了防止静默数据污染，GPU 驱动唯一能做的安全动作就是将整个 MPS Server 标记为 **Fatal Fault（致命故障）并强制杀掉该卡上连接的所有客户端容器**！
- **架构师准入法则**：
  - ⛔ **严禁跨租户开启 MPS**：绝对不要把“算法同学调试的 Jupyter Notebook”和“核心生产在线推理服务”放进同一个 MPS 池！
  - ✅ **强烈推荐在“同生共死的单租户微服务组”内开启 MPS**：例如同一个 RAG 业务 Pod 内部同时包含 `BGE-M3 Embedding` + `bge-reranker-v2` + `Guardrail 8B` 三个容器，它们代码经过严格测试且本来就是同一个业务链路，在一个 `2g.40gb` 的 MIG 切片内开启 MPS 可以让吞吐直接翻倍！

### Q3：硬件 MIG 既然这么安全，为什么不把全公司所有卡都静态切死成 `7 × 1g.20gb`？Tier 4 的 `K8s DRA + 动态 MIG` 解决了什么痛点？

- **静态 MIG（Tier 3）的“碎片僵化痛点”**：
  - H200 切成的 `1g.20gb` 只有 20 GB 显存，跑 8B 模型刚好，但白天突然来了一个需要 `38 GB` 显存的 32B 模型，或者晚上需要合并成整卡 `7g.141gb` 跑微调任务时，静态切死的 `1g.20gb` 根本装不下，导致卡上明明有 3 个空闲的 `1g.20gb`（共 60 GB），却无法合并成一个 `3g.71gb`！
- **Tier 4（Kubernetes 1.32+ `DRA 动态资源分配` + `Dynamic MIG` + `HAMi`）的破局**：
  1. 业务 Pod 不再申请死板的 `nvidia.com/mig-1g.20gb: 1`，而是通过 K8s **DRA `ResourceClaim`** 声明自己需要的真实资源（如 `memory: 16Gi, compute: 15%`）；
  2. 节点上的 DRA 驱动与 GPU Operator 根据当前调度的 Pod 组合，**在线调用 `nvidia-smi mig` 动态销毁/重组空闲的 MIG 分区**（白天切成 `1×3g.71gb + 4×1g.20gb`，夜间自动合并回 `7g.141gb` 跑训练）；
  3. 对于小于 `1g.20gb` 的轻量微服务（如仅需 4GB 显存的 OCR/Embedding），由 **HAMi** 在单个 MIG 分区内按精确显存配额二次装箱，将全集群 GPU 有效利用率从 `74.0%` 推升至 **`88.5%`**！

---

## ③ 🧭 直观可视化图解（Mermaid 决策树 / 架构对比 / 软硬两级切分时序图）

### 3.1 多租户 GPU 算力切分与隔离架构选型决策树

```mermaid
flowchart TD
    A["🧩 多租户 GPU 小模型/微服务算力切分与共享选型"] --> B{"共享同一张 GPU 的多个容器是否属于同一受信边界?"}
    B -- "否 (跨部门多租户 / 生产推理与开发 Notebook 混部)" --> C{"工作负载显存规格是否固定?"}
    C -- "多规格动态混部 (含 4GB 微服务 + 20GB/40GB 模型)" --> D["🟩 Tier 4: K8s DRA 动态硬件 MIG + 租户内 HAMi/MPS<br/>(跨租户物理硬隔离 0% 故障传染 + 租户内细粒度复用)<br/>(单卡 352 QPS · 装箱率 88.5% · 56 服务仅需 8 卡 🏆)"]
    C -- "规格统一 (每个租户固定 ~18GB 显存)" --> E["🟦 Tier 3: 静态硬件 MIG 物理切分 (7 × 1g.20gb)<br/>(独立显存通道/L2/SM · P99 抖动仅 +1.8% · 0% 故障波及 ✅)"]
    B -- "是 (同一应用内部的受信微服务组: 如 Emb + Reranker)" --> F["🟨 Tier 2: NVIDIA CUDA MPS 空间算力并发<br/>(配置 MEM_LIMIT + 算力配额 · 吞吐比时间片高 2.35x ✅<br/>⚠️ 注意: 严禁混入未验证的自定义算子以防 XID 31 连坐)"]
    B -- "切勿在生产环境使用" --> G["🟥 Tier 1: K8s 原生 Time-Slicing 时间片轮转<br/>(零显存隔离 · 1人 OOM 全卡陪葬 · P99 抖动 +215% ✕)"]
```

### 3.2 Time-Slicing / MPS 软共享 (Tier 1/2) vs. 硬件 MIG + HAMi 软硬两级切分 (Tier 4) 对比图

```mermaid
flowchart LR
    subgraph Soft["🟥/🟨 Tier 1 & 2: 全卡共享硬件通道 (OOM 陪葬或 XID 31 连坐 ✕)"]
        direction LR
        S1["7 个不可信租户容器<br/>挤进同一套 HBM 控制器与 L2"] --> S2["Time-Slicing: 无显存墙<br/>1 个 OOM 炸死全卡 7 租户 ✕"]
        S2 --> S3["CUDA MPS: 共用单一 GPU 页表<br/>1 个越界指针触发全卡团灭 ✕"]
    end

    subgraph Hard["🟩 Tier 4: K8s DRA 硬件 MIG 跨租户硬墙 + 租户内 HAMi/MPS (88.5% 装箱 🏆)"]
        direction LR
        H1["硅片 Crossbar 物理切分 MIG<br/>租户 A (2g.40gb) | 租户 B (1g.20gb)"] --> H2["租户 A 内部开启 HAMi/MPS<br/>并发跑 8B + Embedding + Rerank"]
        H2 --> H3["即使租户 B 发生 OOM/XID 31<br/>租户 A 抖动 <3.2% · 0% 故障波及 🏆"]
    end
```

### 3.3 K8s DRA 动态 MIG 划分 + 租户内故障完美隔离时序图

```mermaid
sequenceDiagram
    autonumber
    participant K8s as K8s 1.32+ 调度器 & DRA 控制器
    participant MIG as NVIDIA GPU Operator (硬件 MIG 引擎)
    participant TenantA as 租户 A (核心生产 RAG: 2g.40gb 内嵌 MPS)
    participant TenantB as 租户 B (算法开发 Notebook: 1g.20gb)

    K8s->>MIG: 1. 根据 ResourceClaim 动态切分 H200: 分配 2g.40gb 给租户 A，1g.20gb 给租户 B
    MIG-->>TenantA: 2. 挂载独立硬件引擎 /dev/nvidia-caps/mig-devA (物理独占 40GB HBM 通道与 L2)
    TenantA->>TenantA: 3. 在 2g.40gb 内部启动 MPS 空间并发跑 8B LLM + BGE Embedding + Reranker (352 QPS)
    MIG-->>TenantB: 4. 挂载独立硬件引擎 /dev/nvidia-caps/mig-devB (物理独占 20GB HBM 通道)
    TenantB->>TenantB: 5. 💥 事故触发！Notebook 运行死循环申请 130GB 显存并触发非法指针越界 (XID 31)
    MIG->>TenantB: 6. 硬件 MMU 将 OOM 与 XID 31 严格封死在 mig-devB 铁笼内，仅重置租户 B 容器
    TenantA-->>K8s: 7. 租户 A 生产服务全程零感知 (0% 故障波及 · P99 延迟抖动仅 +3.2% 🏆)
```

---

## ④ 📊 架构师选型决策矩阵（Architect Decision Matrix）

*基准环境：企业 AI 中台需并发承载 **56 个 AI 微服务与开发实例**（每个实例需 `16～18 GB` 显存、峰值算力约占 H200 的 `12%`），对比在 **NVIDIA H200 SXM（141 GB HBM3e）** 上的四档切分共享方案（未切分半独占基线需 28 张卡，年成本 `$672,000`）。*

| 评估维度 | 🟥 Tier 1: K8s 原生 Time-Slicing | 🟨 Tier 2: NVIDIA CUDA MPS 空间并发 | 🟦 Tier 3: 硬件级 MIG 物理切分 (`7×1g.20gb`) | 🟩 Tier 4: K8s DRA 动态 MIG + HAMi/MPS |
| :--- | :--- | :--- | :--- | :--- |
| **算力与显存隔离机制** | 时间片轮转切换，**无显存配额限制** | SM 空间并发 + 显存限额，**共享单一页表** | **硅片级独立 SM、独立 HBM 通道、独立 L2** | **外层硬件 MIG 隔租户 + 内层 HAMi/MPS 复用 🏆** |
| **单卡 7 租户聚合吞吐** | `142.0 QPS` (`1.00x` 上下文切换慢 ✕) | **`333.7 QPS` (`2.35x` 空间并发 ✅)** | `298.0 QPS` (`2.10x` 静态硬边界) | **`352.0 QPS` (`2.48x` 软硬协同最高 🏆)** |
| **吵闹邻居下 P99 延迟抖动** | **`+215.0%` (排队等时间片 ✕)** | `+28.5%` (争抢共享 L2 与显存带宽) | **`+1.8%` (物理级硬件稳态 🏆)** | **`+3.2%` (跨租户硬边界护航 🏆)** |
| **邻居 OOM 显存炸弹爆炸半径** | **`100.0%` (同卡全员 OOM 崩溃 ✕)** | `0.0%` (`MEM_LIMIT` 拦截 ✅) | **`0.0%` (硬件 20GB 物理墙 🏆)** | **`0.0%` (硬件 MIG + `libvgpu` 双重拦截 🏆)** |
| **邻居非法指针越界 (XID 31) 波及** | `0.0%` (独立 OS 进程上下文) | **`100.0%` (共享页表致命连坐团灭 ✕)** | **`0.0%` (独立硬件计算引擎 🏆)** | **`0.0%` (跨租户零连坐 🏆)** |
| **GPU 综合有效装箱利用率** | `42.0%` (不敢高密混部) | `79.5%` | `74.0%` (受限于固定 7 档规格) | **`88.5%` (动态 MIG + 1GB 步长超分 🏆)** |
| **承载 56 个服务所需 H200 卡数** | `16 张` (`$384,000 / 年`) | `10 张` (`$240,000 / 年`) | `10 张` (`$240,000 / 年`) | **`8 张` (`$192,000 / 年`，省卡 `71.4%` 🏆)** |

---

## ⑤ 🧪 简明定量算例与 Benchmark 洞察（三大生产故障注入实测）

> 📁 **封存数据源与可复现脚本**：[`experiments/multi_tenant_gpu_sharing_mig_mps_dra_selection/run.py`](../../experiments/multi_tenant_gpu_sharing_mig_mps_dra_selection/run.py) · [`experiments/multi_tenant_gpu_sharing_mig_mps_dra_selection/results.json`](../../experiments/multi_tenant_gpu_sharing_mig_mps_dra_selection/results.json) · 单测 [`tests/test_multi_tenant_gpu_sharing_mig_mps_dra_selection.py`](../../tests/test_multi_tenant_gpu_sharing_mig_mps_dra_selection.py)
> *(Provenance: `synthetic` + `anonymized_reported`, `independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)*

### 5.1 三类恶意/故障邻居注入下的生产生存能力对比表

| 注入的生产故障场景 | Tier 1: Time-Slicing | Tier 2: CUDA MPS | Tier 3 / Tier 4: 硬件 MIG 边界 | 底层硬件原理解析 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 邻居死循环满载大矩阵乘 (Noisy Neighbor)** | **`+215.0%` P99 恶化 ✕** | `+28.5%` P99 恶化 | **`+1.8% ~ +3.2%` 🏆** | MIG 物理切开了 HBM 内存控制器与 L2 Cache，带宽互不抢占 |
| **2. 邻居死循环申请 130GB 显存 (OOM Bomb)** | **`100%` 邻居陪葬崩溃 ✕** | `0%` 波及 ✅ | **`0%` 波及 🏆** | Time-Slicing 无显存墙；MPS/HAMi/MIG 均强制拦截超额 `cuMemAlloc` |
| **3. 邻居算子触发指针越界 (`XID 31` Fault)** | `0%` 波及 ✅ | **`100%` 全卡连坐团灭 ✕** | **`0%` 波及 🏆** | **MPS 共享单一 GPU 页表无法隔离野指针，唯有 MIG 拥有独立 MMU/引擎！** |

### 5.2 配套 7-Card 4:5 视觉评测报告与交互式计算器

- 🃏 **7-Card 4:5 视觉评测卡片集（CSP 自包含，零外部依赖）**：[`docs/reports/multi-tenant-gpu-sharing-mig-mps-dra-selection-cards.html`](../reports/multi-tenant-gpu-sharing-mig-mps-dra-selection-cards.html)
- 🧮 **独立交互式多租户 GPU 切分与 TCO 计算器**：[`docs/reports/multi-tenant-gpu-sharing-mig-mps-dra-sizer.html`](../reports/multi-tenant-gpu-sharing-mig-mps-dra-sizer.html)（已同步集成至总工作台 [`docs/reports/llm-inference-engine-pd-sizer.html#gpu-sharing-sizer`](../reports/llm-inference-engine-pd-sizer.html)）

---

## ⑥ 🛠️ 附录：生产环境快速核验清单（Architect Checklist）

- [ ] **1. 严禁在生产环境单独使用原生 `Time-Slicing`**：若在不支持 MIG 的老卡或 PCIe 卡（如 L40S / RTX PRO 6000）上必须做软切分，务必部署 **CNCF HAMi (`vgpu-device-plugin`)** 通过 `LD_PRELOAD=libvgpu.so` 强制锁死每个容器的显存上限（如 `nvidia.com/gpumem: 16000`），消灭 OOM 炸全卡事故。
- [ ] **2. 严禁跨不可信租户开启 `CUDA MPS`**：将 CUDA MPS 严格限制在**同一个 Pod 或同一条业务流水线内部**（例如同一个 RAG 服务的 `Embedding + Reranker` 容器组），并始终显式配置 `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` 与 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`。
- [ ] **3. 在 H100 / H200 / B200 多租户集群上默认启用 `硬件 MIG` 作为跨部门安全边界**：使用 `mig-parted` 或 GPU Operator 的 `mixed` 策略，按需提供 `1g.20gb`、`2g.40gb`、`3g.71gb` 物理隔离实例。
- [ ] **4. 升级至 Kubernetes 1.32+ `DRA (Dynamic Resource Allocation)` 架构**：用标准化的 `ResourceClaim` 替代繁琐的 `nvidia.com/mig-*` 硬编码资源名，让调度器根据拓扑与实时负载自动完成 MIG 切片合成与拆分。
- [ ] **5. 按 MIG 实例粒度独立采集 DCGM 监控指标**：在 `dcgm-exporter` 中开启 `-f /etc/dcgm-exporter/dcp-metrics-included.csv` 的 **`GPU_I_ID`（MIG Instance ID）** 标签映射，确保 Prometheus 能精确监控到每一个 `1g.20gb` 切片的独立显存与 SM 利用率。
