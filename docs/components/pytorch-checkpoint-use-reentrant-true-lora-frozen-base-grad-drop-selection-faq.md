# 大模型激活重计算（`torch.utils.checkpoint: use_reentrant=True vs False`）在冻结基座 `LoRA / VLM` 下丢失 `100%` 梯度与算子级选择性重计算（`Op-Level SAC`）选型：为什么开启 `gradient_checkpointing` 时使用 `use_reentrant=True` 会导致 `LoRA` 权重梯度全部变成 `None`（报错 `does not require grad`），而整层重计算又会白白增加 `35.8%` 的单步耗时？基于 `use_reentrant=False` + `create_selective_checkpoint_contexts` 算子级选择性保存（`100%` 梯度存活，节省 `72.0%` 激活显存，重计算开销压至 `4.8%`）选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/pytorch_checkpoint_use_reentrant_true_lora_frozen_base_grad_drop_selection/run.py` 与 `results.json`。

![PyTorch checkpoint(use_reentrant=False) & Selective SAC Visual Summary](../../assets/svg/pytorch-checkpoint-use-reentrant-true-lora-frozen-base-grad-drop-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向大模型微调与预训练（`TorchTitan / HuggingFace PEFT / FSDP2`）的“非重入式激活检查点（`use_reentrant=False`，基于 `saved_tensors_hooks` 原生保留计算图，彻底根除冻结输入 `LoRA / VLM` 梯度断裂与 `DDP` 双重标记崩溃）+ 算子级选择性激活检查点（`Op-Level Selective Activation Checkpointing, SAC`: `MUST_SAVE` 保存昂贵的 `Linear/MoE GEMM` 输出，仅重算高显存低算力的 `FlashAttention / SwiGLU / RMSNorm`）”架构**。
- **架构师价值**：凡是写过 `LoRA / QLoRA` 微调或多模态大模型（冻结 Vision Encoder 训练 Projector + LLM）的工程师，几乎都遇到过这个让人抓狂的经典报错：
  `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`（或者不报错，但跑完一整个 Epoch 发现 **`lora_A` 和 `lora_B` 的梯度全是 `None`，Loss 一动不动**）！
  网上无数的问答贴教你调一句 `model.enable_input_require_grads()` 来“修好”它，但很少有人告诉你 **为什么会出这个 Bug，以及 `enable_input_require_grads()` 背后付出了什么冤枉代价**：
  1. **机理一：为什么旧版 `checkpoint(layer, hidden_states, use_reentrant=True)` 只要看到输入 `hidden_states.requires_grad == False`，就会把层内所有 `LoRA` 权重的梯度 100% 全部掐断（Tier 1）？**
     - 在 `LoRA` 微调中，第 0 层之前的词表嵌入层 `embed_tokens` 是 **冻结的（`embed_tokens.weight.requires_grad == False`）**。因此，`embed_tokens(input_ids)` 算出来的输出张量 `hidden_states` 的属性自然是 **`hidden_states.requires_grad == False`**！
     - 当这个 `hidden_states` 传入第 0 层的 `checkpoint(layer_0, hidden_states, use_reentrant=True)` 时，旧版重入式检查点的底层实现是一个黑盒 `torch.autograd.Function`（`CheckpointFunction`）。
     - 根据 PyTorch Autograd 的老规则，一个 `autograd.Function` **只检查显式作为位置参数传进来的 `*args`（也就是 `hidden_states`）有没有 `requires_grad == True`**——它根本看不见藏在 `layer_0` 闭包内部具有 `requires_grad == True` 的 `lora_A.weight` 和 `lora_B.weight`！
     - 于是 `CheckpointFunction` 误以为：“既然传进来的唯一参数 `hidden_states` 不需要梯度，那这一整层都不需要算梯度！” 当场在 `torch.no_grad()` 下跑完 `layer_0`，导致输出的 `hidden_states_1` 依然是 `requires_grad == False`，一路多米诺骨牌式地把全模型 80 层的 `LoRA` 梯度 **100% 全部丢光**！
     - **为什么 `enable_input_require_grads()` 是个低效的创可贴（Tier 2）？** 它强行把冻结的 `embed_tokens` 输出设为 `requires_grad_(True)`，骗过 `CheckpointFunction`——但副作用是 **逼迫反向传播在第 0 层把完全没人要的激活梯度 `dX` 从头到尾全算一遍**，而且 `use_reentrant=True` 在反向重入时还会触发两次 Autograd Hook，导致 `DDP / FSDP` 报 `Expected to mark a variable ready only once` 崩溃，并直接打断 `torch.compile`！
  2. **机理二：为什么无脑对整个 Transformer 层做全量重计算（Full-Layer Checkpointing，Tier 2 & Tier 3）会白白增加 `31.2% ~ 35.8%` 的训练耗时？**
     - 一个 Transformer 层里的算子分为截然不同的两类：
       - **计算极重、但输出激活占显存较小或适中的算子**：线性投影矩阵乘 `aten.mm` / `_scaled_mm`（`QKV`, `O`, `Gate/Up/Down` 及 MoE `Grouped-GEMM`，占了全层 **`> 95%` 的 FLOPs**！）；
       - **计算极快（或内存带宽受限）、但中间激活占显存极大的算子**：`SwiGLU` 逐元素激活、`RMSNorm`、`RoPE` 以及 `FlashAttention`（它们合计占了全层 **`72.0%` 的激活显存**，但只占 **`< 5%` 的计算耗时**！）。
     - 传统整层重计算为了省显存，在反向时把那占 `95%` 算力的矩阵乘 `GEMM` 也全部重新算了一遍，导致每步耗时暴涨 **`31.2% ~ 35.8%`**！
  采用 **`use_reentrant=False`（通过 `saved_tensors_hooks` 在原生计算图上直接拦截激活存储，哪怕输入 `requires_grad=False` 也能 `100%` 保留内部 `LoRA` 梯度边，且完美兼容 `torch.compile`）+ `create_selective_checkpoint_contexts` 算子级选择性保存策略（对 `mm` 矩阵乘标记 `MUST_SAVE` 不重算，仅重算 `FlashAttention / SwiGLU / RMSNorm`）（Tier 4）**，可在 **节省 `72.0%` 激活显存（从 `56.40 GB` 降至 `15.80 GB`）** 的同时，将重计算时间代价从 `35.8%` 压降至仅 **`4.8%`（开销降低 `7.5 倍`）**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：`use_reentrant=False` 为什么从根源上终结了 `use_reentrant=True` 的四大顽疾？**
   - **`use_reentrant=True`（黑盒重入式）**：前向时关掉 `grad` 裸跑，把整个函数包成一个孤立的 `CheckpointFunction` 节点；反向时再在一个嵌套的 Autograd 引擎里重跑一遍前向建图。如果输入张量 `requires_grad=False`，整个节点直接不建图！
   - **`use_reentrant=False`（白盒钩子式）**：前向时 **保持 Autograd 正常开启**（因此只要层内有任何 `lora_A.weight` 具有 `requires_grad=True`，Autograd 就会自动建立完整的梯度边！），同时利用 **`torch.autograd.graph.saved_tensors_hooks`** 拦截中间激活张量的保存请求（将其替换为空占位符），在反向需要用到该激活时按需局部重算！
   - **四大收益**：① 冻结输入下 `LoRA / VLM` 梯度 `100%` 存活（无需 `enable_input_require_grads`）；② 反向不重触发参数 Hook，彻底根治 `DDP` 重复标记崩溃；③ 原生支持 `torch.compile` 联合图优化；④ 原生支持算子级细粒度选择性保存（`SAC`）！
2. **算子级选择性激活检查点（Op-Level `SAC`）的黄金 10 行代码**：
   ```python
   import functools
   import torch
   from torch.utils.checkpoint import (
       CheckpointPolicy,
       checkpoint,
       create_selective_checkpoint_contexts,
   )

   # 🏆 仅保存昂贵的矩阵乘 (GEMM) 输出, 重算便宜的 SwiGLU / RMSNorm / RoPE / Attention!
   SAVE_OPS = {
       torch.ops.aten.mm.default,
       torch.ops.aten.addmm.default,
       torch.ops.aten._scaled_mm.default,
   }

   def selective_policy_fn(ctx, op, *args, **kwargs):
       if op in SAVE_OPS:
           return CheckpointPolicy.MUST_SAVE
       return CheckpointPolicy.PREFER_RECOMPUTE

   # 在 Transformer Layer 调用处:
   hidden_states = checkpoint(
       layer,
       hidden_states,
       use_reentrant=False,
       context_fn=functools.partial(create_selective_checkpoint_contexts, selective_policy_fn),
   )
   ```

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：如果显存非常紧张（比如在单机上硬塞 `32K` 超长上下文，连保存全部 `mm` 矩阵乘的输出都不够显存），`selective_policy_fn` 可以怎么精细微调？
- **按“层内第奇数个 / 特定形状的 `mm`”做半量保存（Sub-Layer Selective Saving）**！
  在每个 Transformer 层中共有 7 个线性投影（`q, k, v, o` 以及 MLP 的 `gate, up, down`，其中 `gate` 和 `up` 的输出维度最大，为 `intermediate_size = 3.5 * D`）：
  - 你可以在 `selective_policy_fn` 里利用 `ctx.mm_count` 计数，只把输出维度较小的 `q, k, v, o, down` 设为 `MUST_SAVE`，而让占显存最大的 MLP 中间膨胀层 `up_proj` 设为 `PREFER_RECOMPUTE`——实现显存与算力在 `4.8% ~ 15%` 之间的任意无级调节！

### Q2：为什么在 HuggingFace `Transformers` 中调 `model.gradient_checkpointing_enable()` 时，必须显式传 `gradient_checkpointing_kwargs={"use_reentrant": False}`？
- 因为为了兼容几年前的老旧模型代码，部分旧版封装如果不显式传参，仍可能回退或未启用 `context_fn`！在所有现代训练脚本中，务必显式写明：
  ```python
  model.gradient_checkpointing_enable(
      gradient_checkpointing_kwargs={"use_reentrant": False}
  )
  ```

### Q3：`FlashAttention` 的输出在算子级 `SAC` 中应该设为 `MUST_SAVE` 还是 `PREFER_RECOMPUTE`？
- 在 `16K~128K` 超长上下文下，$O(S^2)$ 的 `FlashAttention` 计算耗时远大于 $O(S)$ 的线性层 `mm`！因此：
  - **短/中序列（$S \le 4,096$）**：`mm` 占主导，保存 `mm`、重算 `FlashAttention`；
  - **超长序列（$S \ge 16,384$）**：`FlashAttention` 占主导，将 `torch.ops.aten._flash_attention_forward.default` 加入 `SAVE_OPS`（`MUST_SAVE`），它的输出只有 `[B, S, H, D]`（不存 $S \times S$），保存它能省下极其可观的 $O(S^2)$ 重算时间！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 PyTorch 激活重计算（`use_reentrant` 与 `Op-Level SAC`）选型决策树

```mermaid
flowchart TD
    A["大模型训练开启 Activation Checkpointing (70B LoRA / VLM / 长序列预训练)"] --> B{"checkpoint() 的 use_reentrant 参数设置?"}
    B -- "use_reentrant=True (旧版重入式黑盒 Function)" --> C{"首层输入张量 hidden_states 是否具有 requires_grad=True?"}
    C -- "否 (如冻结 embed_tokens 的 LoRA 或冻结视觉塔的 VLM)" --> D["⛔ Tier 1: CheckpointFunction 判定无输入需梯度 -> 100% 丢弃层内 LoRA 梯度 (grad=None)!"]
    C -- "是 (或强开 enable_input_require_grads 补丁)" --> E["⛔ Tier 2: 白算冻结层 dgrad, 全量重算耗时增加 +35.8%, 且打断 torch.compile 与 DDP!"]
    B -- "use_reentrant=False (基于 saved_tensors_hooks 原生建图)" --> F{"重计算粒度策略 (Full-Layer vs Op-Level SAC)?"}
    F -- "整层无差别全部重算 (Full-Layer Recompute)" --> G["⚠️ Tier 3: 100% 保住梯度且兼容 compile, 但重算全部 GEMM 仍带来 +31.2% 时间开销"]
    F -- "create_selective_checkpoint_contexts (MUST_SAVE 矩阵乘, 仅重算 SwiGLU/Norm/Attn)" --> H["🏆 Tier 4: 100% 梯度存活 + 省下 72.0% 激活显存 + 重计算开销仅 +4.8% (降低 7.5 倍)!"]
```

### 3.2 `use_reentrant=True` 丢失 `100%` LoRA 梯度与全量重算（`+35.8%`）vs. `use_reentrant=False + SAC`（`+4.8%`）对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: use_reentrant=True 掐断冻结输入梯度 + 全量重算多花 35.8% 时间!"]
      B1["冻结 embed_tokens 输出 requires_grad=False -> CheckpointFunction 直接用 no_grad 跑全层!"] --> B2["层内 lora_A / lora_B 梯度全为 None (0% 存活)!"]
      B2 --> B3["打补丁后整层重算所有矩阵乘 -> 耗时暴涨 +35.8%, 且不支持 torch.compile!"]
    end
    subgraph Good["🏆 Tier 4: use_reentrant=False + Op-Level SAC 选择性激活检查点"]
      G1["saved_tensors_hooks 原生保留 lora_A/B 梯度图 -> 100% 梯度存活, 完美兼容 compile!"] --> G2["MUST_SAVE 保存 95% 算力的 GEMM 输出, 仅重算占 72% 显存的 SwiGLU/RMSNorm/Attn"]
      G2 --> G3["激活显存从 56.40 GB 压至 15.80 GB (-72%), 重算时间代价仅 +4.8% (降 7.5 倍)!"]
    end
```

### 3.3 `use_reentrant=False` 配合 `create_selective_checkpoint_contexts` 前反向钩子时序图

```mermaid
sequenceDiagram
    participant Fwd as Transformer Layer (LoRA / Full)
    participant SAC as Selective Checkpoint Hook (use_reentrant=False)
    participant HBM as GPU Activation HBM
    participant Bwd as Autograd Backward Engine

    Fwd->>SAC: 1. Run `mm` (QKV / MLP GEMM, 95% FLOPs) -> `policy_fn` returns `MUST_SAVE`
    SAC->>HBM: 2. Cache compact GEMM output in HBM (never recomputed during backward!)
    Fwd->>SAC: 3. Run `RMSNorm / RoPE / SwiGLU` (72% HBM, <5% FLOPs) -> `PREFER_RECOMPUTE`
    SAC->>SAC: 4. Discard intermediate tensor storage while preserving autograd `grad_fn` edges to `lora_A/B`!
    Bwd->>SAC: 5. Backward arrives -> recompute ONLY `SwiGLU/RMSNorm` (+4.8% time) & reuse cached GEMMs!
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | 激活重计算方案 | 冻结输入下 `LoRA` 梯度存活率 | 重计算单步时间代价 | 激活值显存 (`Raw=56.4GB`) | `torch.compile` 兼容性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: `use_reentrant=True` (默认无补丁)** | 黑盒 Function，仅检查输入参数 | **`0.0%` (梯度全丢! ✕)** | `+33.0%` ✕ | `12.40 GB` | **不兼容 (图断裂 ✕)** |
| **Tier 2: `use_reentrant=True` + `enable_input_require_grads`** | 强开输入梯度 + 整层全部重算 | `100.0%` | **`+35.8%` (极慢! ✕)** | `13.10 GB` | **不兼容 ✕** |
| **Tier 3: `use_reentrant=False` (整层全量重算)** | 原生钩子建图，但重算所有 GEMM | `100.0%` 🏆 | `+31.2%` ✕ | `12.40 GB` | 原生兼容 🏆 |
| **Tier 4: `use_reentrant=False` + 算子级 `SAC` (推荐)** | **保留 GEMM 输出，仅重算便宜算子** | **`100.0%` 🏆** | **`+4.8%` (`7.5x` 更低 🏆)** | **`15.80 GB` (`-72.0%` 🏆)** | **原生融合编译 🏆** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/pytorch_checkpoint_use_reentrant_true_lora_frozen_base_grad_drop/run.py`](../../experiments/pytorch_checkpoint_use_reentrant_true_lora_frozen_base_grad_drop_selection/run.py) · 封存数据：[`results.json`](../../experiments/pytorch_checkpoint_use_reentrant_true_lora_frozen_base_grad_drop_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/pytorch-checkpoint-use-reentrant-true-lora-frozen-base-grad-drop-sizer.html`](../reports/pytorch-checkpoint-use-reentrant-true-lora-frozen-base-grad-drop-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/pytorch-checkpoint-use-reentrant-true-lora-frozen-base-grad-drop-selection-cards.html`](../reports/pytorch-checkpoint-use-reentrant-true-lora-frozen-base-grad-drop-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: use_reentrant=True (0% LoRA Grad)", "recompute_overhead_pct": 33.0, "act_hbm_gb": 12.40 },
      { "tier": "Tier 2: Reentrant + Input Grad Hack", "recompute_overhead_pct": 35.8, "act_hbm_gb": 13.10 },
      { "tier": "Tier 3: use_reentrant=False (Full Layer)", "recompute_overhead_pct": 31.2, "act_hbm_gb": 12.40 },
      { "tier": "Tier 4: use_reentrant=False + Op-Level SAC", "recompute_overhead_pct": 4.8, "act_hbm_gb": 15.80 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "recompute_overhead_pct": "Percentage",
    "act_hbm_gb": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "use_reentrant=False + Op-Level SAC Cuts Recompute Overhead 7.5x (35.8% -> 4.8%) With 100% LoRA Grad Survival",
    "subtitle": "Step time recomputation overhead (%) across 4 activation checkpointing tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "recompute_overhead_pct",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有 `LoRA / QLoRA` 微调、多模态冻结视觉塔训练、以及开启 `torch.compile` 或 `FSDP2` 的大规模预训练。
- **⛔ Not for**：在任何新代码中继续使用已弃用的 `use_reentrant=True` 或盲目重算昂贵的矩阵乘算子。

### 生产核验 Checklist
- [ ] **全局强制显式指定 `use_reentrant=False`**：杜绝冻结输入层时内部可训练适配器梯度静默丢失。
- [ ] **配置 `create_selective_checkpoint_contexts` 算子级选择策略**：对 `aten.mm` / `_scaled_mm`（以及超长序列下的 `_flash_attention_forward`）设置 `CheckpointPolicy.MUST_SAVE`。
- [ ] **在微调第 1 步 `backward()` 后校验所有 `requires_grad=True` 参数的 `p.grad is not None` 且范数非零**。
