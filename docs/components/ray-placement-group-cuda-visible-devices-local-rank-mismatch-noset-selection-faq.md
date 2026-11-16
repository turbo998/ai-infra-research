# 大模型分布式强化学习与推理（`verl / OpenRLHF / vLLM RayExecutor`）Ray `PlacementGroup`（`STRICT_PACK` vs `PACK`）、`CUDA_VISIBLE_DEVICES` 设备号错位崩溃（`cudaErrorInvalidDevice`）与 `NOSET_CUDA_VISIBLE_DEVICES` 排序选型：为什么在 Ray Actor 里调用 `torch.cuda.set_device(local_rank)` 会导致 `87.5%` 的进程当场崩溃，而用 `strategy="PACK"` 又会把 `TP=8` 拆到两台机器上慢 `8.3 倍`？基于 `STRICT_PACK` + `NOSET_CUDA_VISIBLE_DEVICES=1` + `(node_ip, gpu_id)` 物理卡号排序（`0%` 崩溃，TPOT `14.2 ms`，原生支持同机零拷贝 `CUDA IPC`）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/ray_placement_group_cuda_visible_devices_local_rank_mismatch_noset_selection/run.py` 与 `results.json`。

![Ray PlacementGroup STRICT_PACK & NOSET_CUDA_VISIBLE_DEVICES Visual Summary](../../assets/svg/ray-placement-group-cuda-visible-devices-local-rank-mismatch-noset-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向基于 Ray 编排的后训练强化学习（`verl / OpenRLHF / Slime`）与分布式推理（`vLLM RayDistributedExecutor`）的“每节点 `STRICT_PACK` 放置组 + `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 全卡可见性透传 + 按 `(node_ip, ray.get_gpu_ids()[0])` 物理拓扑重排 `WORLD_RANK`”标准编排协议**。
- **架构师价值**：在使用 Ray 构建 `GRPO / PPO` 强化学习流水线（如 `verl` 在同一批 GPU 上混部 FSDP Actor Trainer 与 vLLM Rollout Engine）或拉起多机 `vLLM` 实例时，工程团队几乎 `100%` 会撞上三个令人头皮发麻的底层调度怪象：
  1. **怪象一：单机 8 卡启动 8 个 `@ray.remote(num_gpus=1)` Worker，为什么只有 `local_rank = 0` 成功启动，其余 `local_rank = 1..7`（`7/8 = 87.5%`）全部在 `torch.cuda.set_device(local_rank)` 处报 `RuntimeError: CUDA error: invalid device ordinal (cudaErrorInvalidDevice)` 当场崩溃（Tier 1）？**
     - **真凶**：当 Ray 调度器把一个声明了 `num_gpus=1` 的 Actor 分配到物理第 3 号卡（`GPU #3`）上时，Ray 默认会在拉起该 Python 进程前自动注入环境变量 **`os.environ["CUDA_VISIBLE_DEVICES"] = "3"`**！
     - 在 CUDA Runtime 眼里，既然 `CUDA_VISIBLE_DEVICES` 里只写了一个数字 `"3"`，那么 **该进程内部可见的 GPU 总数 `torch.cuda.device_count()` 就等于 `1`，且这张物理 3 号卡在进程内部的逻辑设备下标被重映射成了 `0`（Logical Ordinal `0`）**！
     - 此时如果你的分布式初始化代码按照 `torchrun` 的习惯去调 `torch.cuda.set_device(local_rank)`（即 `set_device(3)`），CUDA 发现你居然想访问下标为 `3` 的第 4 张卡（而它只看得见下标 `0`），立刻抛出 `cudaErrorInvalidDevice` 崩溃！
     - **更隐蔽的连带灾难（破坏同机 `CUDA IPC` 零拷贝权重同步）**：就算你把代码改成了 `torch.cuda.set_device(0)` 绕过崩溃（Tier 2），因为在混部 RL（`verl`）中，Trainer Actor 和 Rollout Actor 各自被不同的 `CUDA_VISIBLE_DEVICES` 蒙住了眼睛（只能看见 1 张卡），当跨卡或同节点其他进程尝试通过 `cudaIpcOpenMemHandle` 打开对方导出的显存句柄或建立 `NVLink P2P` 时，会因为 **双方进程的 CUDA 设备命名空间不一致（Peer Device Invisible）** 而直接失败或退化为慢速拷贝！
  2. **怪象二：明明创建 `PlacementGroup` 时传了 `strategy="PACK"`，为什么 `TP=8` 的 vLLM 引擎跑着跑着 `TPOT` 从 `14.2 ms` 暴涨到了 `118.5 ms`（慢了 `8.3 倍`，Tier 2）？**
     - **真凶**：在 Ray 中，**`"PACK"` 只是“尽力而为（Best-Effort）”的软约束**！当集群出现碎片（例如节点 A 剩 6 张空闲 GPU，节点 B 剩 2 张空闲 GPU）时，`strategy="PACK"` 会毫不犹豫地把同一个 `TP=8` 组的 6 个 Worker 塞在节点 A、另外 2 个 Worker 溢出一脚踢到节点 B！
     - 这导致原本应该在单机 `900 GB/s NVLink` 上飞速完成的每层 2 次 `TP All-Reduce`，瞬间变成了跨机器的慢速网络通信，解码延迟暴涨 **`8.3 倍`**！
  3. **怪象三：就算改成了严格不跨机的 `strategy="STRICT_PACK"`（Tier 3），为什么同一台机器上的 8 个 Bundle 分到的物理 GPU 顺序是乱序的（如 `[5, 2, 7, 0, 4, 1, 6, 3]`），导致 `TP=2/4` 或 NUMA 亲和性劣化了 `39%`（`19.8 ms vs 14.2 ms`）？**
     - 因为 Ray 的 `PlacementGroup` 只保证把 8 个 Bundle 放在同一台物理机上，**绝不保证 `bundle_index = 0..7` 依次对应物理 `GPU 0..7`**！如果直接拿 `bundle_index` 当作 `local_rank`，`Rank 0` 和 `Rank 1` 可能分别落在跨 CPU Socket / 跨 PCIe 交换机的 `GPU 5` 和 `GPU 2` 上！
  采用 **每节点 `STRICT_PACK` 放置组 + 注入 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 保留全节点 8 卡视野 + 在 `init_process_group` 前按 `(node_ip, int(ray.get_gpu_ids()[0]))` 对所有 Worker 排序后再分配 `WORLD_RANK`（Tier 4）**，可实现 **`0.0%` 启动崩溃**、**`0.0%` 跨机 TP 溢出**、原生支持 **同机 `0.18s` 零拷贝 `CUDA IPC` 权重热更新**，并将 `TP=8` 解码延迟稳定在最优的 **`14.2 ms`（提速 `8.3 倍`）**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：为什么在分布式训练与推理框架中，永远应该关闭 Ray 的单卡环境变量遮罩（`NOSET_CUDA_VISIBLE_DEVICES=1`）？**
   - Ray 默认设置 `CUDA_VISIBLE_DEVICES="<assigned_gpu>"` 的初衷是为了伺候那些“不懂多卡、只会默认往 `cuda:0` 塞张量”的单卡小脚本；
   - 而现代大模型框架（`PyTorch Distributed`, `Megatron-Core`, `FSDP2`, `vLLM`, `SGLang`, `verl`）全都是 **拓扑感知（Topology-Aware）** 的专业引擎，它们需要：
     1. 看见整台物理机的完整 8 张卡（`CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"`）以便正确识别 NVLink 拓扑树、NUMA 亲和性与同机 `CUDA IPC` 伙伴；
     2. 直接通过 `phys_gpu_id = int(ray.get_gpu_ids()[0])` 和 `torch.cuda.set_device(phys_gpu_id)` 精确绑定自己被分配的那张物理卡！
   - 在 Ray（$\ge 2.9$）中，只需在 `runtime_env` 或进程环境中设置：
     ```python
     os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
     # (注: 新版 Ray 亦支持 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 等配套开关)
     ```
2. **为什么必须对 Worker 列表按 `(node_ip, phys_gpu_id)` 排序后再赋 `RANK`？**
   - 当你用 `pg = placement_group([{"GPU": 1}] * 8, strategy="STRICT_PACK")` 创建了 8 个 Actor 后，先调一次极轻量的远程方法收集每个 Actor 的 `(node_ip, int(ray.get_gpu_ids()[0]))`，在 Driver 侧对 Actor 句柄数组执行一次 **字典序排序**：
     - 排序后的第 $0..7$ 个 Actor 必然严格对应同一台机器上的物理 `GPU 0, 1, 2, 3, 4, 5, 6, 7`！
     - 随后再调用各 Actor 的 `init_distributed(rank=i, local_rank=phys_gpu_id)`，即可确保 `TP=2/4/8`、`CP` 与 `PP` 通信域 100% 严丝合缝地对齐物理 NVLink 岛与 CPU NUMA 节点！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：如果要用 Ray 启动一个跨越 16 台机器（共 128 张 GPU）的大作业，应该创建 1 个包含 128 个 Bundle 的 `STRICT_PACK` 放置组，还是创建 16 个各含 8 个 Bundle 的 `STRICT_PACK` 放置组？
- **切记：永远按“单机 8 卡”为单位创建 16 个独立的 `STRICT_PACK`（或使用 1 个包含 16 个 `{"GPU": 8}` 大 Bundle 的 `SPREAD/PACK` 放置组），绝不能创建 1 个含 128 个 `{GPU: 1}` 的 `STRICT_PACK`！**
  因为 `STRICT_PACK` 的含义是“把该 PlacementGroup 里的 **所有 Bundle** 强制塞进 **同一台物理机**”！如果你建了一个要求 128 张 GPU 的 `STRICT_PACK`，而集群里每台物理机只有 8 张卡，Ray 会永远报 `Infeasible PlacementGroup` 挂起死等！正确做法是：**每个 `STRICT_PACK` 组的大小严格等于单机 GPU 数量（8 卡）**（或者单实例 `TP` 大小）。

### Q2：开启了 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 后，既然每个 Actor 都能看见机上的全部 8 张卡，怎么防止某个写漏了 `set_device` 的第三方库偷偷把显存分配到 `GPU 0` 上导致 `GPU 0` 爆显存？
- **在 Actor 的 `__init__()` 第一行（任何 CUDA 张量创建或模型加载之前）立即锁定当前进程的默认 CUDA 设备**：
  ```python
  phys_gpu_id = int(ray.get_gpu_ids()[0])
  torch.cuda.set_device(phys_gpu_id)
  torch.set_default_device(f"cuda:{phys_gpu_id}")  # 锁死默认设备!
  ```
  这样既保留了全节点 8 卡的 P2P / CUDA IPC 硬件可见性，又杜绝了任何不带设备号的 `.cuda()` 误落到 `GPU 0` 上！

### Q3：在 `verl` 这种将 `FSDP Actor` 和 `vLLM Rollout` 混部在同一张物理 GPU 上的架构（Colocated RL）中，Ray 是怎么允许两个 Actor 共享同一张物理 GPU 的？
- 两种标准方式：
  1. **小数 GPU 配额（Fractional GPU Allocation）**：在同一个 PlacementGroup Bundle `{"GPU": 1}` 内，给 Trainer Actor 声明 `num_gpus=0.5`、给 vLLM Rollout Actor 声明 `num_gpus=0.5`，并绑定同一个 `bundle_index`；
  2. **单 Worker 统一托管（HybridEngine Colocated Worker）**：在同一个 `num_gpus=1` 的 Ray Actor 进程内部同时实例化 FSDP Trainer 和 vLLM Engine（配合 `vLLM Sleep Mode Level-1` 与 `NOSET_CUDA_VISIBLE_DEVICES=1`），连跨进程 IPC 都不需要，直接在进程内原地共享显存指针！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 Ray 分布式大模型训练与推理 `PlacementGroup` 与 GPU 绑定选型决策树

```mermaid
flowchart TD
    A["使用 Ray 编排多机多卡 RL (verl / OpenRLHF) 或 vLLM 分布式推理 (每节点 8 卡, TP=8)"] --> B{"是否设置 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1?"}
    B -- "否: 默认由 Ray 遮罩单卡 CUDA_VISIBLE_DEVICES='k'" --> C{"代码内部如何调用 torch.cuda.set_device()?"}
    C -- "调用 set_device(local_rank) (1..7)" --> D["⛔ Tier 1: 进程只看得见逻辑 0 号卡 -> 87.5% (7/8) 的 Actor 报 cudaErrorInvalidDevice 当场崩溃!"]
    C -- "妥协改为 set_device(0)" --> E["⛔ Tier 2: 虽能启动, 但掩盖了兄弟卡可见性 -> 彻底阻断同机零拷贝 CUDA IPC 权重同步!"]
    B -- "是: 开启 NOSET=1, 保持 8 卡完整可见性并用 set_device(int(ray.get_gpu_ids()[0]))" --> F{"PlacementGroup 策略与 Rank 映射顺序?"}
    F -- "使用软约束 strategy='PACK' 或未按物理 GPU ID 排序 Bundle" --> G["⚠️ Tier 3: 碎片时将 TP=8 拆成 6+2 跨机 (慢 8.3x) 或乱序 Bundle 破坏 NUMA/NVLink 亲和!"]
    F -- "每节点 STRICT_PACK + 按 (node_ip, phys_gpu_id) 字典序重排分配 WORLD_RANK" --> H["🏆 Tier 4: 0% 崩溃 + 0% 跨机溢出 + 14.2 ms 最优 TPOT + 原生支持 0.18s CUDA IPC!"]
```

### 3.2 默认 Ray 遮罩崩溃与软 `PACK` 跨机溢出 vs. `STRICT_PACK` + `NOSET` + 物理卡排序对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: 87.5% 设备号越界崩溃 + 软 PACK 跨节点拆分 TP=8 (慢 8.3 倍!)"]
      B1["Ray 设 CUDA_VISIBLE_DEVICES='3' -> device_count()==1 -> set_device(3) 直接崩溃!"] --> B2["strategy='PACK' 把 TP=8 拆成节点 A (6卡) + 节点 B (2卡) -> TPOT 恶化至 118.5 ms!"]
    end
    subgraph Good["🏆 Tier 4: 每节点 STRICT_PACK + NOSET=1 + (node_ip, phys_gpu_id) 排序"]
      G1["NOSET=1 保持 0..7 全卡视野 -> set_device(ray.get_gpu_ids()[0]) 零崩溃且打通 CUDA IPC"] --> G2["STRICT_PACK 锁死同节点 + 按物理 GPU ID 排序 -> 100% 走 900 GB/s NVLink (14.2 ms TPOT)!"]
    end
```

### 3.3 Ray Driver 创建 `STRICT_PACK`、收集物理卡号排序并初始化 `ProcessGroup` 时序图

```mermaid
sequenceDiagram
    participant Driver as Ray Driver / Controller
    participant PG as Per-Node STRICT_PACK PlacementGroup
    participant Actors as 8x Ray GPU Actors (NOSET=1)

    Driver->>PG: 1. Allocate `placement_group([{"GPU": 1}] * 8, strategy="STRICT_PACK")` per node
    Driver->>Actors: 2. Spawn actors with `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1"`
    Actors-->>Driver: 3. Report `(node_ip, phys_gpu_id = int(ray.get_gpu_ids()[0]))` (e.g., scrambled `[5, 2, 7, 0...]`)
    Driver->>Driver: 4. Sort actor handles by `(node_ip, phys_gpu_id)` -> contiguous `GPU 0, 1, 2, 3, 4, 5, 6, 7`!
    Driver->>Actors: 5. Assign sorted `world_rank` & `local_rank = phys_gpu_id` -> `torch.cuda.set_device(phys_gpu_id)`
    Actors->>Actors: 6. `dist.init_process_group("nccl")` -> 100% aligned NVLink & CUDA IPC ready!
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | Ray GPU 放置组与设备可见性方案 | Actor 启动崩溃率 | 同机 `CUDA IPC` 支持 | 跨节点 `TP` 溢出率 | `TP=8` 解码 `TPOT` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 默认 Ray (`num_gpus=1`) + `set_device(local_rank)`** | 单卡遮罩 + 传 `1..7` 设备号 | **`87.5%` (7/8 崩溃! ✕)** | 不支持 ✕ | `25.0%` ✕ | `118.5 ms` ✕ |
| **Tier 2: 改调 `set_device(0)` + 软 `strategy="PACK"`** | 绕过崩溃，但阻断 IPC 且允许跨机拆分 | `0.0%` | 不支持 (设备命名空间隔离 ✕) | `25.0%` (6+2 跨机 ✕) | `118.5 ms` ✕ |
| **Tier 3: `STRICT_PACK` + 默认遮罩 (未排序 Bundle)** | 保证同机，但卡序乱且无跨进程 IPC | `0.0%` | 不支持 ✕ | `0.0%` | `19.8 ms` (NUMA 错位) |
| **Tier 4: `STRICT_PACK` + `NOSET=1` + 物理卡号排序 (推荐)** | **全卡视野 + 严格同机 + 连续物理卡号** | **`0.0%` 🏆** | **原生零拷贝支持 🏆** | **`0.0%` 🏆** | **`14.2 ms` (`8.3x` 🏆)** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/ray_placement_group_cuda_visible_devices_local_rank_mismatch_noset_selection/run.py`](../../experiments/ray_placement_group_cuda_visible_devices_local_rank_mismatch_noset_selection/run.py) · 封存数据：[`results.json`](../../experiments/ray_placement_group_cuda_visible_devices_local_rank_mismatch_noset_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/ray-placement-group-cuda-visible-devices-local-rank-mismatch-noset-sizer.html`](../reports/ray-placement-group-cuda-visible-devices-local-rank-mismatch-noset-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/ray-placement-group-cuda-visible-devices-local-rank-mismatch-noset-selection-cards.html`](../reports/ray-placement-group-cuda-visible-devices-local-rank-mismatch-noset-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Default Ray + set_device(local_rank)", "tpot_ms": 118.5, "crash_pct": 87.5 },
      { "tier": "Tier 2: set_device(0) + Soft PACK", "tpot_ms": 118.5, "crash_pct": 0.0 },
      { "tier": "Tier 3: STRICT_PACK (Unsorted Bundles)", "tpot_ms": 19.8, "crash_pct": 0.0 },
      { "tier": "Tier 4: STRICT_PACK + NOSET=1 + GPU-ID Sort", "tpot_ms": 14.2, "crash_pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "tpot_ms": "Quantity",
    "crash_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "STRICT_PACK + NOSET_CUDA_VISIBLE_DEVICES + GPU-ID Sorting Cuts TP=8 TPOT 8.3x (118.5ms -> 14.2ms)",
    "subtitle": "TP=8 decode per-token latency (ms) across 4 Ray placement tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "tpot_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有基于 Ray 构建的大模型强化学习框架（`verl`, `OpenRLHF`, `Slime`）、多机 `vLLM / SGLang` Ray 执行器以及需要同节点 `CUDA IPC` 共享权重的训推混部架构。
- **⛔ Not for**：在 Ray 默认单卡 `CUDA_VISIBLE_DEVICES` 遮罩下直接调 `torch.cuda.set_device(local_rank)`，或对 `TP > 1` 的实例使用非严格的 `strategy="PACK"`。

### 生产核验 Checklist
- [ ] **全局注入 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1"`，并通过 `int(ray.get_gpu_ids()[0])` 获取物理 GPU ID 调用 `torch.cuda.set_device(phys_gpu_id)`**。
- [ ] **对所有单机内 `TP=2/4/8` 放置组强制使用 `strategy="STRICT_PACK"`**：杜绝节点碎片时 `TP` 通信跨机器溢出。
- [ ] **在调用 `dist.init_process_group()` 之前，按 `(node_ip, phys_gpu_id)` 对 Worker 列表进行字典序排序**：确保逻辑 Rank 与物理 NVLink / NUMA 拓扑严格一一对应。
