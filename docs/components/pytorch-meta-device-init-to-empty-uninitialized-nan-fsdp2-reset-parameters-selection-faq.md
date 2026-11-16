# 大模型 `torch.device("meta")` 零内存初始化、`to_empty()` 未初始化 `NaN` 陷阱与 `FSDP2` 分片物化顺序选型：为什么单机 8 卡直接 `Model(config)` 会狂吞 `1,120 GB` 主机内存触发 OOM Killed，而用 `meta` 设备调 `to_empty("cuda")` 加载权重后又会在第 1 步输出 `100% NaN`（RoPE `inv_freq` 非持久化 Buffer 变野值）？基于 `meta` 建模 -> 先 `fully_shard()` -> 后 `to_empty("cuda")` -> 重算非持久化 Buffer（内存从 `1,120 GB` 降至 `0.20 GB`，启动缩至 `1.8 s`，提速 `102.8x`）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/pytorch_meta_device_init_to_empty_uninitialized_nan_fsdp2_reset_parameters_selection/run.py` 与 `results.json`。

![PyTorch meta-Device Init & to_empty() NaN Prevention Visual Summary](../../assets/svg/pytorch-meta-device-init-to-empty-uninitialized-nan-fsdp2-reset-parameters-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向超大模型分布式训练与推理（`PyTorch FSDP2 fully_shard / TorchTitan / HuggingFace Accelerate init_empty_weights`）的“`with torch.device('meta'):` 零内存拓扑构建 $\to$ 在 `meta` 张量上先行完成 `fully_shard()` / `parallelize_module()` 切分 $\to$ 调用 `model.to_empty(device='cuda')` 仅分配 1/N 物理分片 $\to$ 显式重算非持久化缓冲区（`persistent=False` 如 RoPE `inv_freq`）并执行分片加载/GPU 初始化”四步标准协议**。
- **架构师价值**：在启动 `70B`、`405B` 或 `671B` 大模型训练或推理时，几乎所有工程师都遇到过两个让人一头雾水的启动故障：
  1. **故障一：单机 8 张 H100（配了 `512 GB` 或 `1 TB` 系统内存），刚跑 `model = LlamaForCausalLM(config)`，还没调 `FSDP` 切分，进程就被 Linux 内核 `OOM Killer (SIGKILL, Exit Code 137)` 直接杀死了（Tier 1）！**
     - **算一笔账**：一个 `70B` 模型在 `BF16` 下占 `140 GB`（若默认 `FP32` 更是 `280 GB`！）。当你用 `torchrun --nproc_per_node=8` 启动 8 个进程，如果每个进程都在 CPU 上老老实实执行 `Model(config)`，**8 个进程会同时在主机内存里各自分配一份完整的 `140 GB` 模型（合计 `8 × 140 GB = 1,120.0 GB` = `1.12 TB` 主机 RAM！）**，并且 8 个进程还要在 CPU 上花 **`185.0 秒`** 跑高斯随机数初始化 `nn.init.normal_()`——而这些花 3 分钟算出来的随机数，下一秒钟就会被 `load_state_dict()` 全部覆盖扔掉！
  2. **故障二：改用 `with torch.device("meta"): model = Model(config)` 加上 `model.to_empty(device="cuda")` 和 `load_state_dict(ckpt)` 后，内存是不爆了，为什么模型第 1 个 Step 前向传播吐出来的 Logits 100% 全是 `NaN`（Tier 2）？**
     - **这是 `to_empty()` 最臭名昭著的“非持久化 Buffer 垃圾值陷阱”！**
     - 到底什么是 `model.to_empty(device="cuda")`？它在底层对模型的所有 `Parameter` 和 `Buffer` 调用的是 **`torch.empty_like(..., device="cuda")`**——注意：**`torch.empty` 只分配显存地址，绝不清零，里面装的全是上一次显存残留的随机二进制垃圾（包含大量 `NaN` 和 `1e38` 极值）**！
     - 紧接着你调用了 `model.load_state_dict(ckpt)`，把 Checkpoint 里的权重覆盖了上去。你以为所有参数都被覆盖干净了？**错！**
     - 在 `Llama / Qwen / DeepSeek / Mistral` 的源码里，旋转位置编码（RoPE）的频率张量是这样注册的：
       ```python
       self.register_buffer("inv_freq", inv_freq, persistent=False)
       ```
       注意那个 **`persistent=False`**！它的意思是：**`inv_freq` 不会被保存在 Checkpoint 的 `state_dict` 里**（因为作者认为它在 `__init__` 时算一次就够了）！
     - 结果：当你用 `meta` 设备运行 `__init__` 时，算出来的 `inv_freq` 被扔进了虚无的 `meta` 黑洞；随后你调 `model.to_empty(device="cuda")`，`inv_freq` 变成了一块 **装满显存未初始化垃圾（`NaN`）的 CUDA 张量**；最后你调 `load_state_dict(ckpt)`，因为 Checkpoint 里根本没有 `inv_freq`，**这块装满 `NaN` 的 `inv_freq` 原封不动地留在了显存里**！第 1 步一算 RoPE，所有 Query 和 Key 乘上 `NaN` 的 `inv_freq`，全网瞬间 **`100.0% NaN`**！
  采用 **第 1 步：`with torch.device("meta"):` 0 字节建模 $\to$ 第 2 步：在 `meta` 设备上直接执行 `fully_shard(model)`（变成 `1/8` 大小的 `meta DTensor`） $\to$ 第 3 步：调用 `model.to_empty(device="cuda")`（每卡只分配 `17.50 GB` 显存，主机内存占用仅 `0.20 GB`！） $\to$ 第 4 步：显式重新初始化所有非持久化 Buffer（如 `rope._rope_init()`）及未加载的新增模块参数（Tier 4）**，可将单节点主机内存峰值从 `1,120.0 GB` 压降至 **`0.20 GB`（节省 `5,600 倍`）**，初始化耗时从 `185.0 s` 压缩至 **`1.8 s`（提速 `102.8 倍`）**，且第 1 步 `NaN` 率彻底清零（**`0.0%`**）！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：为什么必须“先在 `meta` 上 `fully_shard(model)`，然后再调 `model.to_empty(device='cuda')`”，顺序绝对不能颠倒？**
   - 如果你 **先调 `model.to_empty(device="cuda")` 再调 `fully_shard(model)`**：那么在调 `to_empty` 的那一瞬间，模型还是一个完整的 `70B` 未分片模型！它会试图在单张 `80 GB` 的 H100 上一次性分配整块 `140 GB` 显存，**当场触发 `CUDA out of memory`**！
   - 而如果你 **先在 `meta` 设备上对每一层和根模型调用 `fully_shard(model)`**：`FSDP2` 会把每个 `meta` Tensor 替换成形状仅为原来的 $1/D$（如 $1/8$）的 **`meta` `DTensor`**（依然占 `0` 字节）！此时再调用 `model.to_empty(device="cuda")`，每张 GPU 只需要分配属于自己的那 $1/8$ 分片（即 **`17.50 GB`** HBM），丝滑无比！
2. **第一性原理：`to_empty()` 后的“三大必清零/必重算区域”**：
   调用 `model.to_empty(device="cuda")` 后，任何 **没有被 Checkpoint `state_dict` 100% 覆盖** 的张量都装着随机显存垃圾（`NaN/Inf`），必须按以下三类逐一修复：
   1. **所有 `persistent=False` 的 Buffer（如 RoPE `inv_freq`, `cos_cached`, `sin_cached`, 静态因果掩码 `causal_mask`）**：必须遍历所有子模块重新调用其初始化计算函数；
   2. **如果是从零开始预训练（From-Scratch Pretraining，不加载 Checkpoint）**：必须从底向上遍历所有子模块调用 `module.reset_parameters()`（在 GPU 上直接生成正态分布权重，比 CPU 快百倍）；
   3. **如果是加载预训练 Checkpoint 后新增了模块（如新挂的 `LoRA` 适配器、分类头 `score_head` 或多模态 `Projector`）**：必须对 `load_state_dict(..., strict=False)` 返回的 `missing_keys` 对应的子模块显式调用 `reset_parameters()`！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：传统的“只在 `LOCAL_RANK == 0` 的 CPU 上实例化完整模型，然后通过 `sync_module_states=True` 广播给其他 7 张卡（Tier 3）”有什么局限？
- 在 `14B / 32B / 70B` 规模下，Rank-0 CPU 方案（Tier 3）把主机内存从 `1,120 GB` 降到了 `140 GB`（1 份副本），勉强能跑（耗时 `34.0 s`）；
- 但一旦模型升级到 **`405B`（单份 `BF16` 占 `810 GB`）或 `671B`（DeepSeek-V3）**，哪怕只实例化 **1 份** CPU 副本也会直接撑爆 `512 GB` 主机内存！唯有 **Tier 4（`meta` 设备切分 + 分布式 `dcp.load` 直接按分片读入 GPU）** 能够以 **`0.25 GB` 主机内存** 在任意大小的模型上通用！

### Q2：如果子模块在自定义编写时没有实现标准的 `reset_parameters()` 方法，怎么安全地从 `meta` 物化？
- 在自定义 `nn.Module` 规范中，强制要求：
  1. `__init__()` 中只定义层结构与 `nn.Parameter(torch.empty(...))` 并调用 `self.reset_parameters()`；
  2. 所有 `register_buffer(..., persistent=False)` 的计算逻辑提取到独立的 `_init_buffers()` 或包含在 `reset_parameters()` 中，这样在 `to_empty(device="cuda")` 后只需一行 `for m in model.modules(): ...` 即可完美复原！

### Q3：使用 `meta` 设备 + `fully_shard()` + GPU 侧 `reset_parameters()` 从头预训练时，怎么保证不同数据并行副本（`DP` Replicas）初始化出来的随机权重是一致的？
- 在调用 `m.reset_parameters()` 前，将所有 `DP` 维度固定相同的随机种子 `torch.cuda.manual_seed(base_seed + tp_rank + pp_rank * 100)`：因为 `DTensor` 已经在 `DP` 维切分好了，只需用全局确定性种子（或在 `DTensor` 分发上下文中初始化）即可保证一致性。

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 大模型 `meta` 设备初始化与 `FSDP2` 分片物化选型决策树

```mermaid
flowchart TD
    A["单机 8 卡启动 70B / 405B 大模型 (主机内存 512 GB, 目标 FSDP2 分片)"] --> B{"模型结构在哪个设备上构建?"}
    B -- "8 个进程同时在 CPU 上调用 Model(config)" --> C["⛔ Tier 1: 8 × 140 GB = 1,120 GB 主机内存 -> 触发 Linux OOM Killer 当场被杀, 且白算 185s CPU 随机数!"]
    B -- "仅在 Rank 0 CPU 上构建 1 份, 再广播分片" --> D["⚠️ Tier 3: 70B 占 140 GB RAM (耗时 34s), 但遇到 405B (810 GB) 时依然主机内存 OOM!"]
    B -- "所有进程在 with torch.device('meta'): 下 0 字节构建" --> E{"fully_shard() 与 to_empty('cuda') 的调用顺序及 Buffer 处理?"}
    E -- "先 to_empty 再 shard, 或漏掉重算 persistent=False 的 RoPE inv_freq" --> F["⛔ Tier 2: 先 to_empty 会爆单卡显存; 漏重算 inv_freq 会让 Step 1 前向 100.0% 输出 NaN!"]
    E -- "先在 meta 上 fully_shard() -> 再 to_empty('cuda') -> 重算非持久化 Buffer + 分片加载" --> G["🏆 Tier 4: 主机内存仅 0.20 GB + 启动仅 1.8 s (快 102.8 倍) + 0.0% NaN!"]
```

### 3.2 8 进程 CPU 初始化（`1,120 GB` OOM）与未重置 `to_empty()`（`100% NaN`）vs. `meta` 分片物化对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: 要么 1,120 GB 撑爆主机内存, 要么 RoPE inv_freq 残留显存垃圾报 NaN!"]
      B1["Tier 1 (8x CPU Init): 8 进程分配 1,120 GB RAM -> OOM Killed (185.0 s)"] --> B2["Tier 2 (to_empty 未重算 Buffer): Checkpoint 不含 persistent=False 的 inv_freq -> 残留 NaN 毁掉全局输出!"]
    end
    subgraph Good["🏆 Tier 4: meta 建模 -> 先 fully_shard() -> 后 to_empty('cuda') -> 重算 Buffer + DCP 分片加载"]
      G1["1. meta 设备 0 字节建模 + 先行切分为 1/8 meta DTensor (0.15 s)"] --> G2["2. to_empty('cuda') 每卡仅分配 17.50 GB 分片 (主机 RAM 仅 0.20 GB!)"]
      G2 --> G3["3. 重算 RoPE inv_freq 等非持久化 Buffer + 并发加载分片权重 -> 1.8 s 就绪, 0% NaN!"]
    end
```

### 3.3 `FSDP2` 标准四步无损物化与非持久化 Buffer 重建时序图

```mermaid
sequenceDiagram
    participant Meta as torch.device("meta") (0 Bytes)
    participant FSDP as FSDP2 fully_shard (DTensor)
    participant GPU as GPU HBM (17.5 GB/Rank Shard)
    participant DCP as Distributed Checkpoint / Buffer Reset

    Meta->>Meta: 1. `with torch.device("meta"): model = Model(config)` (0.15 s, 0 GB RAM)
    Meta->>FSDP: 2. Apply `fully_shard(layer)` bottom-up -> converts params to 1/8th `meta` DTensors!
    FSDP->>GPU: 3. Call `model.to_empty(device="cuda")` -> allocates 17.50 GB/GPU uninitialized HBM
    GPU->>DCP: 4. Recompute `persistent=False` buffers (`rope._rope_init()`) on CUDA (eliminates NaN!)
    DCP->>GPU: 5. Load sharded weights via `dcp.load` (or run `reset_parameters()` for pretraining) in 1.8 s!
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | 70B 模型初始化与分片物化方案 | 节点主机内存峰值 | 初始化总耗时 | Step 1 前向 `NaN` 率 | `512GB` 节点兼容性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 8 进程直接在 CPU 上 `Model(config)`** | 8 份完整 CPU 模型 + CPU 随机初始化 | **`1,120.0 GB` (OOM! ✕)** | `185.0 s` ✕ | `0.0%` | ⛔ **直接被 OOM Killer 杀掉** |
| **Tier 2: `meta` + `to_empty("cuda")` (未重置 Buffer)** | 零内存建模，但漏掉 `inv_freq` 重算 | `0.20 GB` | `1.6 s` | **`100.0%` (必出 NaN! ✕)** | ⛔ **静默数值污染** |
| **Tier 3: Rank-0 CPU 实例化 + 跨卡广播** | 单份 CPU 副本，逐层广播至 GPU | `140.0 GB` | `34.0 s` | `0.0%` | ⚠️ **405B (`810GB`) 时仍会 OOM** |
| **Tier 4: `meta` + 先 `fully_shard` + `to_empty` + 重置 Buffer (推荐)** | **先切分再分配显存 + 重算非持久 Buffer** | **`0.20 GB` (`5,600x` 🏆)** | **`1.8 s` (`102.8x` 🏆)** | **`0.0%` 🏆** | 🏆 **通杀 70B / 405B / 671B** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/pytorch_meta_device_init_to_empty_uninitialized_nan_fsdp2_reset_parameters/run.py`](../../experiments/pytorch_meta_device_init_to_empty_uninitialized_nan_fsdp2_reset_parameters_selection/run.py) · 封存数据：[`results.json`](../../experiments/pytorch_meta_device_init_to_empty_uninitialized_nan_fsdp2_reset_parameters_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/pytorch-meta-device-init-to-empty-uninitialized-nan-fsdp2-reset-parameters-sizer.html`](../reports/pytorch-meta-device-init-to-empty-uninitialized-nan-fsdp2-reset-parameters-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/pytorch-meta-device-init-to-empty-uninitialized-nan-fsdp2-reset-parameters-selection-cards.html`](../reports/pytorch-meta-device-init-to-empty-uninitialized-nan-fsdp2-reset-parameters-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: 8-Rank CPU Model(config)", "host_ram_gb": 1120.0, "init_sec": 185.0 },
      { "tier": "Tier 2: meta + to_empty (100% NaN)", "host_ram_gb": 0.20, "init_sec": 1.6 },
      { "tier": "Tier 3: Rank-0 CPU + Broadcast", "host_ram_gb": 140.0, "init_sec": 34.0 },
      { "tier": "Tier 4: meta + Shard + to_empty + Reset", "host_ram_gb": 0.20, "init_sec": 1.8 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "host_ram_gb": "Quantity",
    "init_sec": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "meta-Device Sharding + to_empty() Cuts 70B Init Time 102.8x (185s -> 1.8s) & Saves 1,119.8 GB RAM",
    "subtitle": "Model initialization latency (seconds) across 4 materialization tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "init_sec",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有 $\ge 14\text{B}$ 大模型的 `FSDP2` (`fully_shard`)、`DeepSpeed ZeRO-3`、`TP/PP` 分布式训练与推理引擎冷启动。
- **⛔ Not for**：在 8 个本地进程中同时用 CPU 实例化完整大模型，或在调用 `to_empty(device="cuda")` 后忘记重新初始化 `persistent=False` 的 Buffer。

### 生产核验 Checklist
- [ ] **严格遵循“先在 `meta` 设备上执行 `fully_shard(model)`，再调用 `model.to_empty(device='cuda')`”的顺序**：严禁在分片前调用 `to_empty('cuda')`。
- [ ] **在 `to_empty('cuda')` 之后显式重新初始化所有 `persistent=False` 的 Buffer（如 RoPE `inv_freq`）以及 `missing_keys` 中的新增适配器参数**。
- [ ] **在 Step 1 前向开始前对全部 `model.buffers()` 执行一次 `assert not torch.isnan(buf).any()` 启动自检**。
