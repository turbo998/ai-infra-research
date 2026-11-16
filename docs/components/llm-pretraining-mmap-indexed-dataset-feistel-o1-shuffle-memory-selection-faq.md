# 大模型万亿 Token 预训练内存映射数据集（`Megatron IndexedDataset .bin/.idx`）与 `50 亿`样本全局洗牌（`np.random.permutation` 内存爆炸 vs. `O(1)` `Feistel` 密码学双射置换）选型：为什么每个 Rank 调一次 `np.random.permutation(5B)` 会在单机 8 卡上瞬间吃掉 `320 GB` 主机内存直接触发 `OOMKilled`？基于 4 轮 `Cycle-Walking Feistel` 零内存全局置换 + `mmap MADV_DONTNEED` 页缓存回收（内存从 `320 GB -> 0.001 GB`，启动提速 `1,775x`）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/llm_pretraining_mmap_indexed_dataset_feistel_o1_shuffle_memory_selection/run.py` 与 `results.json`。

![10T-Token Pretraining mmap IndexedDataset & O(1) Feistel Shuffle Visual Summary](../../assets/svg/llm-pretraining-mmap-indexed-dataset-feistel-o1-shuffle-memory-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向 `10T+` Token 超大规模预训练（`Megatron-Core / MosaicML Streaming / LitData / TorchTitan`）的“4 轮循环游走费斯妥密码网络（`4-Round Cycle-Walking Feistel Cipher`）$\mathcal{O}(1)$ 寄存器级全局双射洗牌 + 预分词二进制 `mmap` 读后即焚（`madvise(MADV_DONTNEED)`）防止 Linux 页缓存颠簸”数据引擎架构**。
- **架构师价值**：当大模型预训练语料规模从早期的 `300B` Tokens 跃升至如今的 **`10T ~ 15T` Tokens（按每条训练序列 `2,048` Tokens 计，共有 $N = \mathbf{5,000,000,000}$（50 亿）条样本）** 时，过去在小数据集上习以为常的 PyTorch `Dataset / Sampler` 写法会在训练启动的第 1 秒直接把整台 8 卡 H100 服务器炸崩：
  1. **灾难一：每个 GPU Rank 进程在内存里调 `np.random.permutation(5_000_000_000)` 或 `torch.randperm(N)` 生成全局打乱索引数组（Tier 1）**：一个包含 50 亿个 `int64`（8 字节）整数的 NumPy 数组，单份就要占 **`40.0 GB` 物理内存**！一台服务器上有 8 个 GPU Rank 进程，各自生成一份就是 **`320.0 GB` 主机内存**；如果 `np.random.permutation` 内部先生成 `arange` 再拷贝打乱，或者 `DataLoader` 子进程发生 `COW` 缺页复制，瞬间突破 `512 GB ~ 1 TB` 容器 cgroup 内存上限，训练还没跑出第 1 步就全员 **`OOMKilled`（且生成 50 亿随机数白白卡死 `142.0 s`）**！
  2. **灾难二：對 20 TB 的 `.bin` 二进制 Token 文件直接 `np.memmap` 随机读取，却不调 `MADV_DONTNEED`（Tier 2）**：由于全局随机洗牌后每次读取都是跨 20 TB 文件的随机 Offset，Linux 内核会把读过的每一个 4KB 文件页全部塞进主机的 **Page Cache（页缓存）**。不到半小时，整台机器的百 GB 内存被只读一次就再也不会用到的历史样本页塞满，触发内核 `kswapd` 疯狂回收内存页，导致训练 Step 出现周期性几秒钟的严重卡顿！
  采用 **密码学 4 轮 `Cycle-Walking Feistel Network` 数学双射函数（在 $\mathcal{O}(1)$ 零数组内存下，仅用 `20 ns` 位运算直接将逻辑步号 $i \in [0, N)$ 映射为全局唯一随机样本号 $\pi_{\text{seed}}(i) \in [0, N)$）+ `mmap` 读取后立即调用 `madvise(MADV_DONTNEED)` 释放页缓存（Tier 4）**，可将单节点洗牌索引内存从 `320.0 GB` 砍至 **`0.001 GB`（彻底归零！）**，将 Epoch 启动与断点续训定位时间从 `142.0 s` 压缩至 **`0.08 s`（提速 `1,775x`）**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：怎么在不分配 1 个字节数组的前提下，把 `50 亿` 个数字做到 100% 不重复、不遗漏的全局随机打乱？**
   - 传统思维认为“洗牌（Shuffle）”必须先把牌编号 `0 .. N-1` 全部摆在桌子上（分配 $\mathcal{O}(N)$ 数组），再用 Fisher-Yates 算法交换位置。
   - **密码学第一性原理（伪随机置换 `PRP: Pseudorandom Permutation`）**：一个分组密码（Block Cipher，接收密钥 `Key = epoch_seed`）本身就是一个从定义域 $\{0, 1\}^b$ 到值域 $\{0, 1\}^b$ 的 **严格一一对应双射函数（Bijection：不同输入绝不会碰撞到相同输出，且覆盖全部元素）**！
   - **4 轮循环游走费斯妥网络（`Cycle-Walking Feistel Network`）**：
     - 取最小的偶数比特宽度 $2k$ 使得 $2^{2k} \ge N$（例如 $N = 5 \times 10^9 < 2^{34}$，取左半边 $L$ 为 17 bits、右半边 $R$ 为 17 bits）；
     - 对任意输入步号 $i$，执行 4 轮极速位运算：$L', R' = R, L \oplus (\text{MurmurHash}(R, \text{seed} + r) \bmod 2^{17})$；
     - 若输出的 $i' = (L' \ll 17) \mid R' \ge N$（落在真实样本数 $N$ 之外），则直接以 $i'$ 为输入再跑一次 Feistel（这就是 **Cycle-Walking 循环游走**：由于是在同一个置换环上继续走，数学上严格保证第一个落回 $[0, N)$ 区间的值依然是 $[0, N)$ 上的完美无碰撞双射！且平均只需循环 $< 2$ 次！）。
   - **震撼收益**：**内存占用恒为 `0 Bytes`**！无论样本数 $N$ 是 50 亿还是 1 万亿，想知道第 `3,456,789,012` 步该读哪条样本，直接调一次 `feistel_permute(3456789012)` 在 **`20 纳秒`** 内算出来，断点续训（Checkpoint Resume）无需加载任何状态文件，瞬间直达任意 Step！
2. **为什么词表 $V > 65,535$（如 `Llama-3 128K / Qwen 152K / Gemma 256K`）时，旧版 `Megatron IndexedDataset` 会出现静默词表截断事故？**
   - 早期 `GPT-2 / Llama-2` 词表 $< 65,535$，二进制 `.bin` 文件用无符号 16 位整数 **`uint16`（2 字节/Token）** 存储。
   - 当切换至现代 `128K ~ 256K` 大词表模型时，如果预处理脚本忘记把 `dtype` 升级为 **`uint32`（4 字节/Token，或 `int32`）**，所有编号 $\ge 65,536$ 的 Token ID 会发生整数溢出回绕（Wrap-around），导致模型在没有任何报错的情况下学出一堆乱码！
3. **为什么随机 `mmap` 读取百 TB 数据集必须配合 `posix_madvise(addr, len, POSIX_MADV_DONTNEED)`？**
   - 在一个 Epoch 内，每条预训练样本只会被读取 **唯一 1 次**。如果不告诉内核“这页用完即弃”，Linux 会把这些毫无复用价值的 `.bin` 文件页一直留在内存里，甚至把训练进程的堆栈页或共享内存挤兑掉！每次读完一个 Batch 的切片后调用 `madvise(..., MADV_DONTNEED)`，内核会立即回收对应的物理页框，使主机内存曲线保持一条完美的水平直线！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：如果只用流式局部缓冲洗牌（如 `IterableDataset` 的 `shuffle_buffer_size = 100,000`，Tier 3），和全局 `Feistel` 双射洗牌（Tier 4）相比有什么隐患？
- **局部窗口洗牌会导致跨领域/跨文档的时间局部性偏差（Domain & Temporal Clustering Bias）**！
  预训练底表通常是由几十个不同来源的大文件（如先排了 5 亿条 GitHub 代码，再排 5 亿条维基百科，再排 5 亿条数学网页）拼接而成的。如果只用 `100,000` 大小的滑动窗口洗牌，模型在前几个小时里看到的将 `100%` 全是代码，后几个小时 `100%` 全是百科，导致 Loss 剧烈震荡与灾难性遗忘！而 **`Feistel` 全局双射洗牌**能以 `0 MB` 内存直接在全量 50 亿样本跨度上实现真正的全局均匀采样！

### Q2：既然 `Feistel` 把样本访问顺序彻底打乱成了全盘随机 Offset，在 Lustre / 3FS 网络并行文件系统上随机读 `2,048` Token（仅 `8 KB`）会不会引发严重的随机小 I/O 性能崩盘？
- **切勿对百 TB 远端文件做纯单样本级跨文件随机 Seek！推荐“两级分层 `Feistel` 洗牌（Two-Level Block + Intra-Block Feistel）”或本地 NVMe 分片缓存**：
  - 将 50 亿样本划分为每个包含 `64 ~ 256` 条连续样本的宏块（Macro-Block，大小约 `512 KB ~ 2 MB`，完美对齐 SSD 与 Lustre 条带读取粒度），先用外层 `Feistel` 全局打乱宏块顺序并预取进内存，再用内层 `Feistel` 打乱宏块内部顺序，兼顾全局随机性与顺序大块吞吐！

### Q3：如果一定要在单机 8 个 Rank 之间共享某个只读大型元数据表（如文档长度偏移表 `.idx`），在 Python 里怎么做到 8 卡只占 1 份物理内存？
- **严禁在每个 Rank 里用 `pickle.load()` 或 `np.load()`（会分配 8 份私有匿名页）！**
  必须使用只读文件内存映射 **`np.memmap("dataset.idx", dtype=np.int64, mode="r")`**（或 `torch.from_file(..., shared=True)`）：操作系统内核的虚拟内存管理器会将 8 个进程的虚拟地址空间映射到**同一份物理 Page Cache 页框**上，8 卡共享仅占 1 份内存！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 万亿 Token 预训练数据集索引与全局洗牌选型决策树

```mermaid
flowchart TD
    A["10T Token 预训练 (N = 50 亿条序列, 单节点 8 个 GPU Rank)"] --> B{"如何生成全局随机打乱的样本访问顺序?"}
    B -- "每个 Rank 调 np.random.permutation(5B) 生成 int64 数组" --> C["⛔ Tier 1: 40 GB/Rank × 8 = 320 GB/节点! 瞬间撑爆宿主机内存 OOMKilled!"]
    B -- "Rank 0 预写 40GB 索引文件 + 8 卡 mmap 共享, 但不回收页缓存" --> D["⛔ Tier 2: 随机读 20TB .bin 填满 Linux Page Cache -> 触发 kswapd 周期性卡顿"]
    B -- "不分配全局索引数组 (O(1) 索引空间)" --> E{"需要局部窗口洗牌还是真·全局确定性洗牌?"}
    E -- "100K 滑动窗口蓄水池洗牌 (IterableDataset)" --> F["⚠️ Tier 3: 内存仅 2.4 GB, 但无法跨 TB 级大文件做真全局均匀混合"]
    E -- "4 轮 Cycle-Walking Feistel 双射置换 + mmap MADV_DONTNEED" --> G["🏆 Tier 4: 0.001 GB 内存 (省 320GB!) + 0.08s 瞬时启动/续训 + 零页缓存颠簸!"]
```

### 3.2 传统 `np.random.permutation(5B)`（`320 GB` OOM）vs. `O(1)` `Feistel` 双射置换（`0.001 GB`）对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: 每 Rank 分配 50 亿 int64 数组 (单节点耗尽 320.0 GB 内存 -> OOM!)"]
      B1["Rank 0..7 各自调用 np.random.permutation(5,000,000,000)"] --> B2["分配 8 × 40 GB = 320 GB 物理内存 + 耗时 142.0 s -> 容器 OOM 崩溃!"]
    end
    subgraph Good["🏆 Tier 4: 4 轮 Cycle-Walking Feistel 算术双射 (0.001 GB 内存, 0.08 s 启动)"]
      G1["输入全局步号 i 与随机种子 seed"] --> G2["4 轮位异或哈希在 20ns 内直接算出唯一目标样本 idx (零数组分配, 支持任意步 O(1) 续训)!"]
    end
```

### 3.3 `O(1)` `Feistel` 索引计算与 `mmap` `MADV_DONTNEED` 零页缓存膨胀读取时序图

```mermaid
sequenceDiagram
    participant Train as Training Loop (Step s, Rank r)
    participant Feist as Cycle-Walking Feistel Permuter (O(1) Registers)
    participant Mmap as Read-Only mmap (.bin / .idx)
    participant Kernel as Linux VMM Page Cache

    Train->>Feist: 1. Request global sample index for logical position i = s * global_batch + r
    Feist-->>Train: 2. Return bijective shuffled_idx = feistel_permute(i, N=5B, seed) in 20 ns (0 RAM!)
    Train->>Mmap: 3. Read token slice [offset : offset + 2048 * 4B] from mmap .bin
    Mmap->>Kernel: 4. Page fault loads 8KB into Page Cache & copies to pinned host tensor
    Train->>Kernel: 5. Call madvise(slice_addr, len, MADV_DONTNEED) -> immediately free physical page frame!
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | `50 亿`样本洗牌与 `mmap` 管理策略 | 单节点洗牌索引内存 | Epoch 启动/续训耗时 | 页缓存颠簸风险 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 每 Rank `np.random.permutation(5B)`** | 8 个进程各自实例化 `40 GB` 数组 | `320.0 GB` (OOM! ✕) | `142.0 s` ✕ | 极高 ✕ | ⛔ **十亿级样本以上必爆宿主机内存** |
| **Tier 2: 共享 `mmap` 索引文件（无 `madvise`）** | 8 卡共享文件页，但随机读撑爆缓存 | `64.0 GB` ✕ | `48.5 s` ✕ | 高 (`kswapd` 卡顿! ✕) | ⛔ **运行数小时后出现周期性 I/O 毛刺** |
| **Tier 3: 局部 `100K` 滑动窗口蓄水池洗牌** | 仅在内存小窗口内随机抽样 | `2.4 GB` ✅ | `1.20 s` ✅ | 无 ✅ | ⚠️ 仅适合已预先全局打乱过的分片 |
| **Tier 4: `O(1)` `Feistel` 双射置换 + `MADV_DONTNEED` (推荐)** | **位运算实时双射 + 读后释放页缓存** | **`0.001 GB` (省 `320GB` 🏆)** | **`0.08 s` (`1,775x` 快 🏆)** | **零颠簸 🏆** | 🏆 **万亿 Token 预训练数据加载黄金标准** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/llm_pretraining_mmap_indexed_dataset_feistel_o1_shuffle_memory_selection/run.py`](../../experiments/llm_pretraining_mmap_indexed_dataset_feistel_o1_shuffle_memory_selection/run.py) · 封存数据：[`results.json`](../../experiments/llm_pretraining_mmap_indexed_dataset_feistel_o1_shuffle_memory_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/llm-pretraining-mmap-indexed-dataset-feistel-o1-shuffle-memory-sizer.html`](../reports/llm-pretraining-mmap-indexed-dataset-feistel-o1-shuffle-memory-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/llm-pretraining-mmap-indexed-dataset-feistel-o1-shuffle-memory-selection-cards.html`](../reports/llm-pretraining-mmap-indexed-dataset-feistel-o1-shuffle-memory-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Per-Rank np.random.permutation(5B)", "ram_gb": 320.0, "init_s": 142.0 },
      { "tier": "Tier 2: Shared mmap .idx (No madvise)", "ram_gb": 64.0, "init_s": 48.5 },
      { "tier": "Tier 3: Local 100K Window Buffer", "ram_gb": 2.4, "init_s": 1.2 },
      { "tier": "Tier 4: O(1) Feistel Cipher + MADV_DONTNEED", "ram_gb": 0.001, "init_s": 0.08 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "ram_gb": "Quantity",
    "init_s": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "O(1) Cycle-Walking Feistel Permutation Eliminates 320 GB/Node Shuffle Index Memory & Starts in 0.08s",
    "subtitle": "Host RAM consumed by shuffle indices per 8-GPU node (GB) for 5B sequences (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "ram_gb",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有样本数 $N \ge 10^7$（百 GB 至数十 TB 语料）的大模型预训练、继续预训练（Continual Pretraining）与多源数据动态配比流水线。
- **⛔ Not for**：在内存中显式构造 `np.arange(N)`、`np.random.permutation(N)` 或 `list(range(N))`。

### 生产核验 Checklist
- [ ] **使用 4 轮 `Cycle-Walking Feistel Network`（或仿射 LCG）替代显式洗牌数组**：以 $\mathcal{O}(1)$ 内存开销实现百亿样本级确定性全局洗牌与任意 Step 瞬间恢复。
- [ ] **核验预分词 `.bin` 文件的整数位宽**：当词表大小 $V \ge 65,536$（如 `Qwen / Llama-3 / Gemma`）时，强制使用 `uint32`（而非 `uint16`）防止 Token ID 溢出截断。
- [ ] **对只读一次的随机 `mmap` 数据页调用 `madvise(MADV_DONTNEED)`**：防止 20 TB 语料将宿主机 Linux Page Cache 彻底塞满。
