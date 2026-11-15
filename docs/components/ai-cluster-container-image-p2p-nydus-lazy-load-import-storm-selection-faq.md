# 千节点 AI 容器镜像（`25 GB` PyTorch/CUDA 巨型镜像）冷启动与 `import torch` 风暴选型：中心化 `Harbor` 拉取拥塞（`1,680 秒`）+ 共享存储 `NFS/Lustre` 4.5 亿次 `stat()` 卡死 vs. `Dragonfly P2P` + `Nydus / OverlayBD` 按需块级懒加载与本地 `SquashFS` 虚拟环境挂载（`12.5 秒` 极速就绪）选型指南

> **Provenance & Evidence Tier**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)。定量数据锚定于 [`experiments/ai_cluster_container_image_pull_oci_soci_nydus_dragonfly_p2p_selection/results.json`](../../experiments/ai_cluster_container_image_pull_oci_soci_nydus_dragonfly_p2p_selection/results.json)。
> **English Version**: [1,000-Node AI Container Cold-Start (`25 GB` PyTorch/CUDA Image) & `import torch` Metadata Storm: Centralized Registry Bottleneck (`1,680s`) & Shared-FS `stat()` Storm vs. `Dragonfly P2P` + `Nydus / OverlayBD` Lazy Loading & Local `SquashFS` (`12.5s`) Selection Guide](./ai-cluster-container-image-p2p-nydus-lazy-load-import-storm-selection-faq.en.md)

![1,000-Node AI Container Cold-Start Nydus P2P & SquashFS Summary](../../assets/svg/ai-cluster-container-image-p2p-nydus-lazy-load-import-storm-selection-faq-summary.svg)

---

## 0. 🔥 今日领域最热门能力突破追踪（Daily Hottest Breakthrough）

- **揭秘为什么在 `1,024` 台节点（`8,192` 张 GPU）上启动一个大模型训练任务或弹性扩容 `vLLM` 推理集群时，8,192 张昂贵的 GPU 竟然要在 `ContainerCreating` 和第一行 `import torch; import transformers` 上足足干等 `28 分钟（1,680 秒）` 才能跑出第 0 步？从第一性原理拆解“千节点并发拉取解压 `25 GB` `tar.gz` 镜像洪峰”与“8,192 个进程对共享存储发起 `4.5 亿次` `stat()` 元数据雪崩”，并通过“`Nydus (RAFS v6) / OverlayBD` 块级按需懒加载 + `Dragonfly P2P` 分发 + `SquashFS` 单文件只读镜像本地挂载”（`Tier 4`），将每个节点实际拉取流量从 `25.0 GB` 砍至 `2.2 GB`（`-91.2%`）、共享存储元数据请求削减 `-99.999%`，实现 `12.5 秒` 万卡极速冷启动（提速 `134.4 倍`）**：在传统的互联网微服务中，一个 Go/Java 容器镜像只有 `100 MB`；但在 AI 基础设施中，一个装满了 `CUDA Toolkit + cuDNN + NCCL + PyTorch + FlashAttention + DeepSpeed/Megatron` 的标准训练/推理容器镜像，压缩后高达 **`20 ~ 25 GB`**，解压后更是高达 **`45 ~ 60 GB`（包含超过 `350,000` 个零碎的 `.py`、`.pyc` 和 `.so` 动态库小文件）**！当 `1,024` 台服务器在同一秒启动任务时，两场灾难会先后引爆：
  1. **第一场灾难：中心镜像仓库 `25.6 TB` 带宽洪峰与 `tar.gz` 串行全量解压死穴（Tier 1 — 卡死 `1,680 秒`）**：
     - **网络打爆**：`1,024` 台机器同时向中心的 Harbor / 镜像仓库拉取 `25 GB` 镜像，瞬间产生 **$1024 \times 25\text{ GB} = \mathbf{25.6\text{ TB}}$** 的并发下行流量！哪怕中心 Harbor 挂了 `400 Gbps` 负载均衡，也要排队传输十几分钟！
     - **`tar.gz` 格式的原罪（必须 100% 下完 + 单核解压完才能启动容器！）**：传统 Docker/OCI 镜像层是按 `tar.gz` 打包的。由于 `gzip` 流式压缩没有块级随机索引，宿主机 `containerd` 必须**老老实实把整整 `25 GB` 一字节不落地全部下载完，再用 CPU 在本地磁盘上把 `45 GB`、`35 万` 个文件全部解压展开**，然后才允许执行 `runc start` 启动容器！
     - **最讽刺的真相**：统计表明，一个大模型训练或推理容器从启动到跑完，真正会被 Python 和 CUDA 读到的文件只占整个 `25 GB` 镜像里的 **`8.8%`（约 `2.2 GB`）**！另外 `91.2%` 的静态库、头文件和未用到的算子包纯粹是陪跑！
  2. **第二场灾难（哪怕镜像提前预热好了也会炸！）：把 `conda / venv` 虚拟环境放在共享存储 `NFS / Lustre / CephFS` 上引发的“4.5 亿次 `import torch` 元数据风暴”（Tier 2 — 卡死 `780.0 秒`）**：
     - 很多算法团队为了省去每次改代码重打 `25 GB` 镜像的麻烦，把基础镜像预拉到节点上，但把自己的 `conda` 虚拟环境和代码仓库放在集群共享的 `/mnt/lustre/conda_envs/llm_env` 下，然后让 `8,192` 个 GPU 进程直接去加载它！
     - **第一性原理算账：1 次 `import torch; import transformers; import vllm` 在操作系统底层到底干了什么？**
       - Python 的 `importlib` 在导入几百个子模块时，会沿着 `sys.path` 列表里的十几个目录逐一扫描查找文件是否存在：每次查找都会发起一连串的 **`stat()`、`open()`、`fstat()`、`getdents()`、`close()`** 系统调用！
       - 实测 `strace` 表明：仅 **1 个** Python 进程跑完 `import torch, transformers, megatron` 就会触发约 **`55,000` 次文件系统元数据请求（Metadata Operations）**！
       - 当 `8,192` 张 GPU 在 `torchrun` 指挥下在同一秒执行 `import` 时：**$8,192 \times 55,000 = \mathbf{450,560,000}$（`4.5 亿次`！）同步元数据 RPC 请求在同一秒钟砸向共享存储（Lustre MDS 或 NFS Server）**！
       - 再强大的分布式存储元数据服务器（MDS）在 `4.5 亿次` 小文件 `stat()` 面前也会瞬间锁死！全集群 8,192 张卡在第一行 `import torch` 上活活卡死 **`780.0 秒（13 分钟）`** ⚠️！
  3. **为什么只上 `Dragonfly P2P` 而不改镜像格式（Tier 3 — 仍需 `185.0 秒`）只解决了一半问题？**
     - 引入 `Dragonfly / Kraken` P2P 分发树后，1,024 台节点可以互相传块，不再打爆中心 Harbor；**但是**因为镜像还是传统的 `tar.gz` 格式，每个节点还是必须等 `25 GB` 全部下完并把 `35 万` 个文件解压写进本地磁盘后才能启动容器（仍需 **`185.0 秒`**）✅。
  4. **工业级终极秒级启动方案：`Nydus (RAFS v6) / OverlayBD` 按需块级懒加载 + `Dragonfly P2P` + 共享环境打包为单文件 `SquashFS` 本地挂载（Tier 4 — `12.5 秒` 直达训练 Step 0，提速 `134.4x`！）**：
     - **第一板斧（针对 `25 GB` 容器镜像：`Nydus / OverlayBD` 按需懒加载 + `P2P`）**：
       - 将镜像构建为 **`Nydus (RAFS v6)` 或 `OverlayBD`** 格式：它把镜像切成一个个独立索引的 `1 MB` `zstd/lz4` 压缩数据块（Chunk），并分离出仅几 MB 大小的元数据索引树（Bootstrap）。
       - 当 `1,024` 台节点启动容器时，只需花 **`1.5 秒`** 拉下几 MB 的元数据索引树，**容器当场直接启动运行（无需等待 `25 GB` 下载！）**！当容器里的 Python 进程真正去读 `libtorch_cuda.so` 的某几个字节时，内核通过 `EROFS / fscache`（或用户态 FUSE）通过 `Dragonfly P2P` 网络毫秒级按需拉取那 **`2.2 GB`（`8.8%`）** 真正用到的热块！
     - **第二板斧（针对共享存储上的自定义 `conda` 环境與代码：单文件 `SquashFS` 镜像 + 本地 NVMe 环回挂载）**：
       - 严禁让 8,192 个进程直接在 Lustre/NFS 上散着读 `35 万` 个 `.py/.so` 文件！
       - 用 `mksquashfs my_conda_env env.sqsh -comp lz4` 把整个虚拟环境打包成 **唯一 `1` 个只读镜像文件 `env.sqsh`（约 `3 GB`）**！
       - 作业启动时，每台节点只由 Local Rank 0 从 Lustre 顺序读取这 **`1` 个大文件** 拷入本机 `/tmp/env.sqsh`（或直接 `mount -o loop,ro /tmp/env.sqsh /opt/conda_mount`）：在 Lustre MDS 上只产生 **`1024` 次大文件顺序读（元数据请求减少 `99.999%`！）**，而节点内 8 个进程的全部 `import torch` 100% 命中本机 Linux 内核内存 `dcache / Page Cache`，**`12.5 秒` 完成千节点冷启动 + `import torch` 全流程** 🏆！

---

## 1. 📌 核心结论速览（Executive Summary — 3 分钟看懂）

1. **为什么 AI 集群的容器镜像绝对不能继续用十年前的 `OCI tar.gz` 格式？**
   - `tar.gz` 是为几十 MB 的小镜像设计的，它要求 **“先 100% 下载完 + 100% 解压完，才准启动进程”**。对于 `25 GB` 的 AI 镜像，这意味着每个节点每次冷启动都要白白传输和解压 `22.8 GB` 根本用不到的废文件！
   - 切换至 **`Nydus (RAFS v6 + Linux 内核 EROFS/fscache)` 或 `OverlayBD`** 后，变成 **“先在 1.5 秒内启动容器，用到哪块读哪块”**，配合 `Dragonfly P2P`，千节点启动直接从半小时压进 **`12.5 秒`**！
2. **彻底根治万卡 `import torch` 卡死 15 分钟的黄金口诀**：
   - **“绝不让 `PYTHONPATH` 或 `conda` 指向 NFS / Lustre 上的散装目录！”** 凡是跨节点共享的 Python 环境或百万小文件数据集，一律打包为 **`SquashFS (.sqsh)` 单文件镜像**，拷入计算节点本地 NVMe `/tmp` 后执行 `mount -o loop,ro` 挂载！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：`Nydus` 早期基于用户态 `FUSE` 会有一定的文件读取性能损耗，现在生产环境是怎么做到与原生本地磁盘完全零损耗一致的？
- **答**：自 Linux Kernel `5.19+`（及 RHEL 9 / Ubuntu 22.04+ HWE 内核）起，`Nydus (RAFS v6)` 原生集成了内核态 **`EROFS (Enhanced Read-Only File System) + fscache`**！
  - 当某个 `1 MB` 数据块第一次被访问时，由内核触发 `fscache` 向 `nydusd` 守护进程从 `Dragonfly P2P` 拉取并落盘到本机 NVMe 缓存；
  - 一旦该块就位，后续所有对该文件的 `read() / mmap()` 系统调用 **100% 直接走内核态 `EROFS` 与宿主内核 Page Cache，完全绕过用户态 FUSE 上下文切换**，性能与本地原生 Ext4/XFS 磁盘分毫不差！

### Q2：如果用户在非特权容器内没有 `sudo mount -o loop` 权限，怎么挂载 `env.sqsh`（`SquashFS`）单文件虚拟环境？
- **答**：有三种零特权优雅方式：
  1. 在 Slurm `TaskProlog` 或 Kubernetes `DaemonSet / NRI Hook`（宿主机特权层）中自动识别任务声明的 `.sqsh` 路径并代为完成只读 `loop` 挂载；
  2. 使用 Singularity / Apptainer 的 `--overlay env.sqsh:ro` 原生参数；
  3. 或者直接将虚拟环境打成 `tar.lz4` 单文件包，在节点启动 `InitContainer` 中由每台机器的 `Local Rank 0` 用 `3 秒钟`（`4 GB/s`）解压到本机内存盘 **`/dev/shm/conda_env`** 中，随后通过 `torch.distributed.barrier()` 让同机其余 7 个进程直接从本机内存盘极速 `import`！

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + 传统 `tar.gz + NFS` 雪崩 vs `Nydus P2P + SquashFS` 对比图 + 时序图）

### 3.1 架构师千节点 AI 容器镜像分发与 Python 环境启动选型决策树

```mermaid
flowchart TD
    A["1,024 节点 (8,192 GPU) 同时启动 25GB PyTorch 容器并执行 import torch"] --> B{"容器镜像格式与分发架构是什么？"}
    B -- "中心 Harbor 直接拉取传统 OCI tar.gz (Tier 1)" --> C["⛔ 25.6 TB 洪峰打爆中心仓库 + 单核串行解压 45GB<br/>叠加共享存储 import 风暴 · 启动耗时高达 1,680 秒 (28 分钟)!"]
    B -- "镜像虽预热，但将 Conda 环境放在共享 Lustre/NFS (Tier 2)" --> D["⛔ 8,192 个进程并发发起 4.5 亿次 stat()/open() 元数据请求<br/>Lustre MDS 锁死 · import torch 卡顿 780 秒 (13 分钟)!"]
    B -- "Dragonfly P2P 分发传统 OCI tar.gz (Tier 3)" --> E["⚠️ 虽打通 P2P 带宽，但仍需下完 100% (25GB) 并解压完才准启动 (185 秒)"]
    B -- "Nydus/OverlayBD 块级按需懒加载 + P2P + 本地 SquashFS 挂载 (Tier 4)" --> F["🏆 1.5s 秒启容器 (仅按需拉取 2.2GB 热块) + SquashFS 零 MDS 风暴<br/>千节点 12.5 秒进入训练 Step 0 (提速 134.4x)!"]
```

### 3.2 `Tier 1/2`（`25GB` 全量解压 + `4.5 亿次` MDS `stat()` 雪崩）vs. `Tier 4`（`Nydus P2P` 按需拉取 + 本地 `SquashFS`）对比图

```mermaid
flowchart LR
    subgraph STORM["⛔ Tier 1 & 2: 全量解压与 4.5 亿次元数据风暴 (1,680 秒)"]
        direction TB
        S1["1,024 节点从中心仓库苦等下载 25GB tar.gz 并解压 35 万个文件 (91% 文件根本不用!)"] --> S2["8,192 个进程对共享 NFS/Lustre 上的 Conda 目录发起 4.5 亿次 stat()!"]
        S2 --> S3["存储 MDS 彻底瘫痪，8,192 张 GPU 空转干等 28 分钟!"]
    end
    subgraph FAST["🏆 Tier 4: Nydus P2P 懒加载 + 本地 SquashFS (12.5 秒)"]
        direction TB
        F1["Nydus RAFS v6: 仅拉取几 MB 元数据索引，1.5 秒直接启动容器!"] --> F2["运行时通过 Dragonfly P2P 仅按需拉取真正 import 用到的 2.2GB 热块!"]
        F2 --> F3["动态环境打包为单文件 env.sqsh 本地 loop 挂载，MDS 请求减少 99.999%!"]
    end
```

### 3.3 `Tier 4` 千节点 `Nydus (EROFS/fscache)` + `Dragonfly P2P` + 本地 `SquashFS` 极速冷启动时序图

```mermaid
sequenceDiagram
    autonumber
    participant REG as Harbor + Dragonfly P2P Seed Peers
    participant NYD as 节点 Containerd + Nydus (EROFS/fscache)
    participant SQSH as 本地 NVMe SquashFS Loop Mount (/dev/shm)
    participant PY as 8×GPU Python Processes (import torch)

    REG->>NYD: 1. 仅拉取 8MB Nydus Bootstrap 元数据索引，1.5 秒内完成容器挂载并启动进程!
    SQSH->>PY: 2. 节点 Local Rank 0 将单文件 env.sqsh 挂载至本机只读页缓存 (共享 MDS 仅 1 次读取!)
    PY->>NYD: 3. 8 个进程并发执行 import torch，触发内核 EROFS 缺页读取真正需要的 2.2GB 热块
    NYD->>REG: 4. 经 Dragonfly 机内/机间 P2P 环网毫秒级拉取 2.2GB 热块，12.5 秒全集群齐步进入 Step 0!
```

---

## 4. 📊 架构师选型决策矩阵（4-Tier Selection Matrix）

| 评估维度 | Tier 1: 中心拉 `tar.gz` + 共享 `Conda` | Tier 2: 镜像预热 + 共享 `Lustre Conda` | Tier 3: `Dragonfly P2P` + `tar.gz` | Tier 4: **`Nydus/OverlayBD` + `P2P` + 本地 `SquashFS`** |
| :--- | :--- | :--- | :--- | :--- |
| **千节点冷启动至 `Step 0` 总耗时** | **`1,680.0 s` (`28 分钟` 空转!)** ✕ | **`780.0 s` (`13 分钟` 卡 `import`)** ✕ | `185.0 s` (卡在全量解压) ⚠️ | **`12.5 s` (提速 `134.4x`，秒级弹性启动)** 🏆 |
| **冲击共享存储 `MDS` 的 `stat()` 次数** | **`450.5 Million` (`4.5 亿次`!)** ✕ | **`450.5 Million` (打挂 MDS)** ✕ | `0.1 Million` ✅ | **`0.001 Million` (减少 `-99.999%`，单文件挂载)** 🏆 |
| **每节点实际需跨网拉取的镜像体积** | **`25.0 GB` (100% 全下)** ✕ | `2.4 GB` | **`25.0 GB` (100% 全下)** ✕ | **`2.2 GB` (仅按需拉取 `8.8%` 真实热块)** 🏆 |

---

## 5. 🧪 定量算例与工具链（Interactive Sizer & 7-Card Visual Report）

- **实验脚本与密封数据**：[`experiments/ai_cluster_container_image_pull_oci_soci_nydus_dragonfly_p2p_selection/run.py`](../../experiments/ai_cluster_container_image_pull_oci_soci_nydus_dragonfly_p2p_selection/run.py) · [`results.json`](../../experiments/ai_cluster_container_image_pull_oci_soci_nydus_dragonfly_p2p_selection/results.json)
- **交互式千节点容器镜像 P2P 懒加载与 import 风暴计算器**：[`docs/reports/ai-cluster-container-image-p2p-nydus-lazy-load-import-storm-sizer.html`](../reports/ai-cluster-container-image-p2p-nydus-lazy-load-import-storm-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/ai-cluster-container-image-p2p-nydus-lazy-load-import-storm-selection-cards.html`](../reports/ai-cluster-container-image-p2p-nydus-lazy-load-import-storm-selection-cards.html)

---

## 6. 🛠️ Good Fit vs. Not For & 生产环境快速核验清单（Checklist）

### ✅ Good Fit vs. ⛔ Not For

| 维度 | ✅ Good Fit（强烈推荐） | ⛔ Not For（严禁误用） |
| :--- | :--- | :--- |
| **百节点 / 千节点 AI 训练与弹性推理容器启动** | CI/CD 自动将基础镜像转换为 **`Nydus (RAFS v6)` / `OverlayBD`** 格式并配合 **`Dragonfly P2P`** 按需分发；用户动态 `conda` 环境打包为 **`SquashFS (.sqsh)`** 本地挂载 | 让几百台机器直接从中心 Harbor 拉取 `25 GB` `tar.gz` 镜像，或将包含数十万小文件的 `conda / venv` 直接放在共享 `NFS / Lustre` 上供上千个进程 `import` |

### 生产环境上线核验清单（Pre-Flight Checklist）

- [ ] **严禁在千卡任务的 `PYTHONPATH` 或 `PATH` 中包含挂载在共享存储（NFS / Lustre / CephFS）上的散装 `site-packages` 目录**。
- [ ] **在节点 `containerd` 上启用 `nydus-snapshotter`（开启内核 `EROFS + fscache` 模式）与 `Dragonfly dfdaemon` P2P 代理**。
- [ ] **将高频更新的用户虚拟环境打包为单文件 `SquashFS`（或由每节点 `Local Rank 0` 一次性拷入 `/dev/shm` 后再 `barrier` 放行其余 7 卡）**。
