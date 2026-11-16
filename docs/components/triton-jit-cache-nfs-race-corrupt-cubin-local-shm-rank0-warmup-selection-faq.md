# 大规模集群 `Triton / TorchInductor / DeepGEMM` JIT 编译缓存共享 NFS 并发写踩踏（`CUDA_ERROR_INVALID_IMAGE`）与节点本地 `LOCAL_RANK==0` 屏障预热选型：为什么 512 卡作业默认指向共享 NFS `~/.triton/cache` 会导致 `38.5%` 的启动崩溃与 `840 秒`（14 分钟）编译风暴，而按 Rank 隔离 `/tmp` 又会造成 `8 倍` 本地 CPU 重复编译？基于节点本地 `/tmp` + `LOCAL_RANK==0` 单路预编译屏障 + 原子重命名（启动缩至 `24.9 s`，提速 `33.7x`，`0%` 崩溃）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/triton_jit_cache_nfs_race_corrupt_cubin_local_shm_rank0_warmup_selection/run.py` 与 `results.json`。

![Triton JIT Cache NFS Race & Local-Rank-0 Warmup Visual Summary](../../assets/svg/triton-jit-cache-nfs-race-corrupt-cubin-local-shm-rank0-warmup-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向多机多卡 GPU 集群（`Slurm / Kubernetes` 512~4,096 卡运行 `@triton.jit`, `torch.compile`, `FlashInfer JIT`, `DeepGEMM`）的“容器镜像只读基线预置 + 节点本地高速盘（`/tmp` / NVMe）重定向 + `LOCAL_RANK == 0` 单进程独享 CPU 预热编译 + 节点内 `dist.barrier(node_pg)` 零竞争复用”全链路编译编排规范**。
- **架构师价值**：几乎每一个在 HPC / Slurm 或挂载了共享存储的 K8s 集群上跑大模型训练或推理的团队，都经历过这样一个灵异现场：**单机 8 卡跑得好好的代码，一扩到 64 台机器（512 张 GPU）启动，要么卡在 Step 1 前面整整 `14 分钟`（`840.0 s`）不动，要么有 `38.5%` 的概率在某几个 Rank 上突然抛出 `RuntimeError: Triton Error [CUDA]: invalid device image (CUDA_ERROR_INVALID_IMAGE)` 或 `json.decoder.JSONDecodeError: Extra data` 直接崩溃！**
  追根溯源，这是因为绝大多数集群把用户的 `$HOME` 目录挂载在 **共享网络文件系统（NFSv3 / NFSv4 / Lustre / CephFS）** 上，而 Triton（`~/.triton/cache`）、PyTorch Inductor（`/tmp/torchinductor_${USER}` 或 `$HOME/.cache`）和 `DeepGEMM` 在冷启动时会触发两层灾难：
  1. **灾难一：512 个进程同时往同一个共享 NFS 目录写同名 `.cubin` 引发非原子损坏（`38.5%` 崩溃率 + `840.0 s` IOPS 阻塞，Tier 1）**！
     - 当 512 个 GPU 进程同时执行到第一个 `@triton.jit` 算子时，它们算出的算子哈希目录 `<sha256_key>` **100% 完全相同**！
     - 512 个进程同时发现 NFS 上还没有编译好的 `.cubin` 文件，于是 **512 个进程同时拉起 `ptxas` 编译器，并同时向 NFS 上同一个文件路径发起创建、写入和覆盖**！
     - 在 NFS 的跨节点弱缓存一致性（Close-to-Open Coherency）下，当节点 A 的进程还在往 NFS 刷写半截 `.cubin` 二进制文件或 `.json` 元数据时，节点 B 的进程看到文件名已经存在，立即调用 `cuModuleLoadData` 读取该文件——**直接读到了只有一半的残缺 ELF 镜像，当场报 `CUDA_ERROR_INVALID_IMAGE` 或 `EOFError` 崩溃**！同时 512 个进程并发狂刷数万次小文件 `stat/open/rename`，把 NFS 元数据服务器（MDS）彻底打满，冷启动硬生生卡死 **`840.0 秒`（14 分钟）**！
  2. **灾难二：改成每卡独立本地目录 `TRITON_CACHE_DIR=/tmp/triton_${RANK}` 后，单机 8 卡把一模一样的算子重复编译 8 遍（`168.0 s` CPU 踩踏 + `14.40 GB` 内存盘浪费，Tier 2）**！
     - 很多网上的排障贴建议设 `export TRITON_CACHE_DIR=/tmp/triton_$SLURM_PROCID`（每个进程一个独立文件夹）。
     - 这虽然消除了跨进程写冲突（崩溃率降为 `0%`），但同一台物理服务器上的 **8 个 GPU Worker 拥有完全相同的 GPU 型号（如 H100 SM90）和完全相同的模型尺寸**！让 8 个进程各自独立编译 120 个 Triton / Inductor / DeepGEMM 算子，等于在同一台机器上瞬间拉起 **`8 × 16 = 128` 个并发 `ptxas` / C++ 编译进程** 疯狂争抢 CPU 核心与 L3 缓存，不仅让单机编译时间从 `24.5 s` 恶化到 **`168.0 s`（慢了 `6.7 倍`）**，还白白吃掉 `8 × 1.80 GB = 14.40 GB` 的 `/tmp` 内存盘空间！
  采用 **将缓存指向每台物理机共享的本地高速目录 `/tmp/triton_node_${JOB_ID}` + 仅由每台机器的 `LOCAL_RANK == 0` 独享整机 CPU 执行一次预热编译（`24.5 s`）+ 节点内其余 7 张卡在 `dist.barrier(node_pg)` 后仅花 `0.4 s` 直接命中加载（Tier 4）**，可将 512 卡冷启动编译耗时从 `840.0 s` 压降至 **`24.9 s`（提速 `33.7 倍`）**，崩溃率降至 **`0.0%`**，每节点磁盘占用降至 **`1.80 GB`（节省 `8.0 倍`）**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：JIT 编译产物的“节点内同构性（Intra-Node Homogeneity）”与“跨节点无共享性（Inter-Node Isolation）”**：
   - 算子编译产物（`.cubin` / `.so`）只与 `(GPU SM 架构, CUDA/Triton 版本, 算子代码与 Shape Specialization)` 有关，与 `GLOBAL_RANK` 编号毫无关系！
   - 因此在多机 8 卡集群中：
     - **跨节点（Inter-Node）绝不共享读写同一份网络文件系统目录**：彻底切断 NFS 元数据风暴与非原子写入损坏；
     - **节点内（Intra-Node）8 张卡应当 100% 共享同一份本地高速缓存目录（`/tmp` 或本地 NVMe）**，但必须遵循 **“1 写 7 读（Single-Writer, Multi-Reader）”** 的屏障时序！
2. **为什么连“8 卡共享 `/tmp/triton` 但不加 `LOCAL_RANK == 0` 屏障（Tier 3）”也会慢 `5.7 倍` 甚至偶发报错？**
   - 就算最新版本 Triton 在写文件时使用了临时文件名 `tmp.pid.uuid` + `os.replace()` 原子重命名，如果在冷启动时 8 个本地进程 **同时** 进入第 1 个 Step，8 个进程在检查缓存时都会看到 `Cache Miss`，从而 **依然同时启动 8 份 `ptxas` 编译一模一样的算子**（白白争抢 CPU 达 `142.0 s`），最后 8 个进程先后覆盖重命名同一个目标文件（在包含多文件联动的 `TorchInductor` C++ Wrapper 编译中仍有 `4.2%` 概率因头文件/动态库覆盖时机引发链接或加载竞态）！
   - 加上 **`if local_rank == 0: warmup() ; dist.barrier()`** 后，`LOCAL_RANK == 0` 可以毫无竞争地调动整机 CPU 飞速编完所有算子（`24.5 s`），其余 7 张卡过屏障后 `100%` 走纯内存文件读取（`0.4 s`）！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：除了 `TRITON_CACHE_DIR`，在大模型训练与推理（`vLLM / SGLang / TorchTitan`）中还有哪些 JIT 缓存环境变量必须一并重定向到本地 `/tmp`？
- **四大天王必须成套配置（缺一不可）**：
  ```bash
  export NODE_CACHE_ROOT="/tmp/ai_jit_cache_${SLURM_JOB_ID:-$$}"
  export TRITON_CACHE_DIR="${NODE_CACHE_ROOT}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${NODE_CACHE_ROOT}/inductor"
  export FLASHINFER_WORKSPACE_DIR="${NODE_CACHE_ROOT}/flashinfer"
  export DG_JIT_CACHE_DIR="${NODE_CACHE_ROOT}/deepgemm"
  ```
  如果只改了 `TRITON_CACHE_DIR` 而忘了改 `TORCHINDUCTOR_CACHE_DIR` 或 `FLASHINFER_WORKSPACE_DIR`，`torch.compile` 或 `FlashInfer` 依然会在共享盘或默认路径上发生多进程编译踩踏！

### Q2：如果不想每次作业重启都花 `24.9 秒` 重新编译，怎么做到“冷启动 `1.5 秒` 秒级开跑”？
- **采用“镜像内只读黄金缓存（Read-Only Golden Base Cache）+ 启动时 `cp -r` 到本地 `/tmp`”**：
  在 CI/CD 构建 Docker 镜像（或节点预拉取阶段）时，预先跑一次标准 Shape 的 Warmup，把生成的 `1.80 GB` 缓存打包进容器镜像的 `/opt/prebuilt_jit_cache`。容器启动脚本里由 `LOCAL_RANK == 0` 用 `0.8 秒` 将 `/opt/prebuilt_jit_cache` 拷入本地 `/tmp`（或设为 `TORCHINDUCTOR_PRECOMPILE_CACHE` 只读层），即可实现 `100%` 零编译秒级启动！

### Q3：如果线上模型用到了 `@triton.autotune`，在 `LOCAL_RANK == 0` 单独跑预热时需要注意什么？
- **注意通信算子与计算算子的分离预热**！如果被预热的函数内部包含 `dist.all_reduce`（需要所有 8 张卡同时参与），则不能让 `LOCAL_RANK == 0` 单独跑整个 `model.forward()`（否则会卡在集合通信上等待 Ranks 1..7）！
- **两种标准解法**：
  1. 在 `LOCAL_RANK == 0` 上仅针对纯本地算子（FlashAttention / RMSNorm / MoE Grouped-GEMM / SwiGLU / CrossEntropy）跑单卡 Dummy Tensor 预热；或者
  2. 如果必须在完整 `forward()` 中触发编译，则让 `GLOBAL_RANK == 0`（或第一个 TP 组 8 张卡，每卡设独立 `/tmp`）先跑 1 步把编译产物打包同步到各节点的 `/tmp/triton_node`，再释放全局屏障！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 大规模 GPU 集群 `Triton / Inductor / DeepGEMM` JIT 缓存架构选型决策树

```mermaid
flowchart TD
    A["512 卡 (64 节点 × 8 卡) 启动含 Triton / torch.compile / DeepGEMM 的作业"] --> B{"JIT 缓存目录 (TRITON_CACHE_DIR 等) 位于何处?"}
    B -- "默认共享网络存储 ($HOME/.triton/cache on NFS/Lustre)" --> C["⛔ Tier 1: 512 进程并发覆写同名 .cubin -> 38.5% 报 CUDA_ERROR_INVALID_IMAGE 崩溃, 卡死 840 s!"]
    B -- "重定向至节点本地高速盘 (/tmp 或本地 NVMe)" --> D{"同节点 8 个 GPU 进程如何组织缓存目录与编译时序?"}
    D -- "每卡独立目录 (/tmp/triton_rank_${LOCAL_RANK})" --> E["⚠️ Tier 2: 虽 0% 崩溃, 但单机重复编译 8 遍 -> 128 个 ptxas 争抢 CPU 耗时 168 s, 占 14.4 GB!"]
    D -- "8 卡共享同一个 /tmp/triton_node, 但无编译屏障" --> F["⛔ Tier 3: 冷启动时 8 卡仍同时触发 Cache Miss 重复编译 (142 s), 且 Inductor 存在 4.2% 链接竞态!"]
    D -- "8 卡共享 /tmp/triton_node + LOCAL_RANK==0 单路预热屏障" --> G["🏆 Tier 4: 冷启动仅 24.9 s (快 33.7 倍) + 0.0% 崩溃 + 每节点仅占 1.80 GB!"]
```

### 3.2 共享 NFS 编译风暴（`840s` / `38.5%` 崩溃）vs. 节点本地 `LOCAL_RANK==0` 屏障预热（`24.9s`）对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: 共享 NFS 写坏 .cubin (38.5% 崩) 或单机 8 倍重复编译 (168s)"]
      B1["Tier 1 (共享 NFS): 512 进程同时写 ~/.triton/cache -> 读到半截 .cubin 崩溃 + MDS 卡死 840s"] --> B2["Tier 2 (每卡独立 /tmp): 同节点 8 卡各自启动 ptxas 编完全相同的算子 -> 耗时 168s, 占 14.4GB"]
    end
    subgraph Good["🏆 Tier 4: 节点本地 /tmp/triton_node + LOCAL_RANK==0 屏障预热"]
      G1["LOCAL_RANK==0 独享整机 CPU 预热编译算子至 /tmp/triton_node (24.5 s)"] --> G2["dist.barrier(node_pg) 后 LOCAL_RANK 1..7 直接命中本地 .cubin (0.4 s, 0% 崩溃!)"]
    end
```

### 3.3 节点本地 `LOCAL_RANK == 0` 单路编译与节点内 `Barrier` 共享加载时序图

```mermaid
sequenceDiagram
    participant R0 as Node LOCAL_RANK 0
    participant TMP as Node-Local /tmp/triton_node (tmpfs/NVMe)
    participant R17 as Node LOCAL_RANKs 1..7

    Note over R0,R17: Set TRITON_CACHE_DIR & TORCHINDUCTOR_CACHE_DIR = /tmp/triton_node
    R17->>R17: 1. Enter `dist.barrier(node_local_pg)` and wait quietly (0% CPU usage)
    R0->>TMP: 2. Execute kernel warmup with 100% of node CPU cores -> write `.tmp.pid.uuid` & atomic `os.replace()` (24.5 s)
    R0->>R17: 3. Enter `dist.barrier(node_local_pg)` -> release Ranks 1..7!
    R17->>TMP: 4. Warmup hits 100% pre-compiled `.cubin` in `/tmp` -> load into CUDA context in 0.4 s!
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | JIT 缓存存储与编译并发策略 | 512 卡冷启动耗时 | `.cubin` 损坏崩溃率 | 每节点 `/tmp` 占用 | 单节点 `ptxas` 冗余度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 默认共享 NFS `~/.triton/cache`** | 512 进程跨网络争抢同一目录 | `840.0 s` (14m ✕) | **`38.5%` (频繁崩溃! ✕)** | `1.80 GB` (NFS) | `512x` 全网冗余 ✕ |
| **Tier 2: 每卡独立 `/tmp/triton_${RANK}`** | 进程间隔离，单机重编 8 份 | `168.0 s` ✕ | `0.0%` | `14.40 GB` ✕ | `8x` 单机冗余 ✕ |
| **Tier 3: 节点共享 `/tmp` (无预热屏障)** | 8 卡同时 Miss 同时编译同一目录 | `142.0 s` ✕ | `4.2%` (C++ 链接竞态 ✕) | `1.80 GB` | `8x` 单机冗余 ✕ |
| **Tier 4: 节点共享 `/tmp` + `LOCAL_RANK==0` 屏障 (推荐)** | **1 卡预编 + 原子替换 + 7 卡直接读** | **`24.9 s` (`33.7x` 🏆)** | **`0.0%` 🏆** | **`1.80 GB` (`8.0x` 🏆)** | **`1x` 零冗余 🏆** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/triton_jit_cache_nfs_race_corrupt_cubin_local_shm_rank0_warmup_selection/run.py`](../../experiments/triton_jit_cache_nfs_race_corrupt_cubin_local_shm_rank0_warmup_selection/run.py) · 封存数据：[`results.json`](../../experiments/triton_jit_cache_nfs_race_corrupt_cubin_local_shm_rank0_warmup_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/triton-jit-cache-nfs-race-corrupt-cubin-local-shm-rank0-warmup-sizer.html`](../reports/triton-jit-cache-nfs-race-corrupt-cubin-local-shm-rank0-warmup-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/triton-jit-cache-nfs-race-corrupt-cubin-local-shm-rank0-warmup-selection-cards.html`](../reports/triton-jit-cache-nfs-race-corrupt-cubin-local-shm-rank0-warmup-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Shared NFS ~/.triton/cache", "startup_sec": 840.0, "crash_pct": 38.5 },
      { "tier": "Tier 2: Per-Rank Isolated /tmp", "startup_sec": 168.0, "crash_pct": 0.0 },
      { "tier": "Tier 3: Shared /tmp (No Barrier)", "startup_sec": 142.0, "crash_pct": 4.2 },
      { "tier": "Tier 4: Shared /tmp + LOCAL_RANK==0 Barrier", "startup_sec": 24.9, "crash_pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "startup_sec": "Quantity",
    "crash_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Node-Local Cache + LOCAL_RANK==0 Warmup Barrier Cuts 512-GPU Cold Startup 33.7x (840s -> 24.9s)",
    "subtitle": "Cold startup JIT compilation time on 512 GPUs (seconds) across 4 tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "startup_sec",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有 `$HOME` 挂载在 NFS / Lustre / CephFS 上的多机 Slurm / Kubernetes GPU 集群，以及启用了 `@triton.jit`、`torch.compile`、`FlashInfer` 或 `DeepGEMM` 的训练与推理作业。
- **⛔ Not for**：把 `TRITON_CACHE_DIR` 留在共享网络盘上，或让同节点的 8 个进程无屏障并发编译同一批算子。

### 生产核验 Checklist
- [ ] **在作业启动脚本中强制将 `TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`FLASHINFER_WORKSPACE_DIR` 与 `DG_JIT_CACHE_DIR` 重定向至节点本地 `/tmp` 或本地 NVMe**。
- [ ] **实施 `LOCAL_RANK == 0` 阶梯预热屏障**：先由 `LOCAL_RANK == 0` 完成算子编译与 Warmup，通过 `dist.barrier(node_pg)` 后再放行同节点其余 7 张卡直接加载缓存。
- [ ] **在生产容器镜像构建阶段预置只读基线 JIT 缓存（`/opt/prebuilt_jit_cache`）**：将线上扩容冷启动时间进一步压缩至 `< 2 秒`。
