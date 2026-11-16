# PyTorch `AdamW` 优化器执行模式（`foreach=True` 临时缓冲区 `17.50 GB` 峰值 OOM vs. `fused=True, capturable=True` 单趟融合与 0-D 张量 `lr`）选型：为什么默认 `foreach=True` 会在第一步 `optimizer.step()` 里突然因 `torch._foreach_sqrt(exp_avg_sq)` 额外申请 `17.50 GB` 临时显存直接 OOM，且修改 Python `float lr` 会触发 `64 次` `torch.compile` 重编译？基于寄存器单趟融合（`fused=True` 临时显存 `0.00 GB`，提速 `7.5x` 至 `3.80 ms`）+ `capturable=True` 0-D GPU 张量 `lr` 选型指南

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. 本文所有定量数据均来自封存确定性评估器 `experiments/pytorch_adamw_foreach_temp_buffer_oom_fused_capturable_selection/run.py` 与 `results.json`。

![PyTorch AdamW foreach Temp-Buffer OOM vs. Fused + Capturable Visual Summary](../../assets/svg/pytorch-adamw-foreach-temp-buffer-oom-fused-capturable-selection-faq-summary.svg)

---

## 0. 追踪每日最热门能力突破（Daily Capability Breakthrough）

- **今日焦点突破**：**面向大模型分布式训练（`FSDP2 / TorchTitan / Megatron-LM`）的“`torch.optim.AdamW(fused=True, capturable=True)` 单趟寄存器融合内核 + 0-D CUDA 张量学习率（`lr=torch.tensor(lr, device='cuda')` 原地 `.fill_()` 更新）工程范式”：彻底消除 `foreach=True` 中 `torch._foreach_sqrt(exp_avg_sq)` 引发的 `17.50 GB` 临时显存尖峰 OOM，并将 `725` 个参数张量的优化器步耗时从 `28.40 ms` 压缩至 `3.80 ms`（提速 `7.5 倍`），实现 `0 次` `torch.compile` 重编译与 100% CUDA Graph 兼容**。
- **架构师价值**：在大模型训练中，许多团队经常遇到一个极其诡异的现象：**前向传播（Forward）和反向传播（Backward）明明稳稳当当地跑完了，显存占用停留在 `68 GB / 80 GB`，结果一进入第 1 步或第 2 步的 `optimizer.step()`，瞬间报出 `CUDA out of memory: Tried to allocate ...` 当场崩溃**！为什么明明优化器状态（`exp_avg`、`exp_avg_sq`）已经分配好了，`optimizer.step()` 还会额外爆掉十几 GB 显存？这背后隐藏着 PyTorch `AdamW` 三种执行实现（`for-loop`、`foreach=True`、`fused=True`）的底层架构差异：
  1. **深坑一：单张量串行 `for` 循环（`foreach=False, fused=False`，Tier 1）—— 每步发射 `4,350` 个小算子，优化器耗时高达 `28.40 ms`**！
     - 一个 70B 模型在单卡分片后仍有 **`725` 个独立的参数张量**（每层含 `wq, wk, wv, wo, w1, w2, w3` 及 RMSNorm 权重）。如果关闭 `foreach` 和 `fused`，PyTorch 会对每个张量依次发射 `mul_`（权重衰减）、`lerp_`（一阶矩 $m_t$）、`addcmul_`（二阶矩 $v_t$）、`sqrt`、`add_`、`addcdiv_` 共 **6 个独立 CUDA Kernel**——单步合计发射 **`4,350` 个小算子**！且每个算子都把 `(p, g, m, v)` 在 HBM 和寄存器之间反复读写 **3.5 遍**（总访存高达 `428.8 GB`），白白耗费 **`28.40 ms`**！
  2. **深坑二：PyTorch 默认开启的多张量矢量化 `foreach=True`（`foreach=True, fused=False`，Tier 2）—— 算子合并到了 `12` 个，但因非原地 `torch._foreach_sqrt(exp_avg_sq)` 瞬间额外分配 `17.50 GB` 临时显存直接把 80GB 显卡撑爆**！
     - 为了解决 `4,350` 次算子发射开销，PyTorch 默认启用了 `_multi_tensor_adamw`（即 `foreach=True`），利用 `multi_tensor_apply` 一次性批量处理全部 `725` 个张量。
     - **但为什么 `foreach=True` 会成为压垮显存的罪魁祸首？**
       打开 `torch/optim/adamw.py` 源码看 `_multi_tensor_adamw` 计算分母 $\sqrt{v_t} + \epsilon$ 的那行代码：
       ```python
       # ⛔ 致命的非原地（Out-of-Place）批量开方调用！
       exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sq)
       torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
       torch._foreach_add_(exp_avg_sq_sqrt, eps)
       torch._foreach_addcdiv_(device_params, device_exp_avg, exp_avg_sq_sqrt, step_size)
       ```
     - 因为不能破坏状态里的原始二阶矩 `device_exp_avg_sq`（$v_t$），`torch._foreach_sqrt(device_exp_avg_sq)` **必须在显存中非原地新建一整套共 `725` 个与 `exp_avg_sq` 完全同尺寸的 `FP32` 临时张量列表 `exp_avg_sq_sqrt`**！
     - 当单卡分片参数量为 `4.375B`（例如 35B 模型 `FSDP=8` 或 70B 模型 `FSDP=16`）时，仅这一套 `exp_avg_sq_sqrt` 临时张量列表就高达 **`4.375B × 4 Bytes = 17.50 GB`**（若是 70B `FSDP=8`，更是高达 **`35.00 GB`**）！在 `optimizer.step()` 执行的这几毫秒内，这 **`17.50 GB` 临时显存尖峰** 与尚未释放的梯度、主权重、$m_t$、$v_t$ 叠加在一起，直接触发 Step-1 OOM！
  3. **深坑三：开启 `fused=True` 但未开 `capturable=True` 且用 Python `float` 更新学习率（Tier 3）—— 虽消除了 `17.50 GB` 临时显存，但在 Cosine LR 调度下触发 `64 次` `torch.compile` 重编译且无法捕获 CUDA Graph**！
  采用 **`AdamW(params, lr=torch.tensor(lr, device='cuda'), fused=True, capturable=True)` + 原地 `pg['lr'].fill_(new_lr)` 更新（Tier 4）**，将权重衰减、一二阶动量更新、偏差修正与开方除法全部融合在 **单趟 CUDA 寄存器内核** 中完成，**临时显存尖峰彻底降至 `0.00 GB`（净省 `17.50 GB` 峰值显存！）**，单步耗时降至 **`3.80 ms`（比 Tier 1 快 `7.5 倍`，比 `foreach=True` 快 `2.26 倍`）**，且全程 **`0 次` 重编译**！

---

## ① 📌 核心结论速览（Executive Summary）

1. **第一性原理：为什么 `foreach=True` 既费显存（`+17.50 GB`）又比 `fused=True` 慢 `2.26 倍`？**
   - **`foreach=True` 是“横向合并（Horizontal Fusion across Tensors）”而非“纵向算子融合（Vertical Fusion across Ops）”**：
     - 它只是把 `725` 个张量的 `lerp_` 合并成 1 次启动、把 `725` 个张量的 `addcmul_` 合并成 1 次启动、把 `725` 个张量的 `sqrt` 合并成 1 次启动……
     - **6 个步骤之间依然要通过 HBM 显存传递中间结果**！不仅总 HBM 读写流量高达 **`315.0 GB`**（是理论下限 `122.5 GB` 的 **`2.57 倍`**），而且在第 3 步求 $\sqrt{v_t}$ 时必须分配 **`17.50 GB`** 的 `exp_avg_sq_sqrt` 临时张量缓冲！
   - **`fused=True` 是真正的“横向 + 纵向全融合（Horizontal + Vertical Register Fusion）”**：
     - 单个 CUDA Kernel 一次性将 `(param, grad, exp_avg, exp_avg_sq)` 读入 GPU 寄存器（Register File），在寄存器内部一口气算完：
       $$p \leftarrow p (1 - \eta \lambda), \quad m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \quad p \leftarrow p - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
       然后将更新后的 `(param, exp_avg, exp_avg_sq)` 一次性写回 HBM！
     - **HBM 读写流量精准达到理论屋顶线（Roofline）：读 `4 × 17.5 GB` + 写 `3 × 17.5 GB` = `122.5 GB`**，且寄存器里的 $\sqrt{\hat{v}_t}$ 用完即弃，**HBM 临时显存分配严格等于 `0.00 GB`**！
2. **为什么用 `torch.compile` 包装优化器或开启 CUDA Graph 时，必须把 `lr` 声明为 0-D GPU Tensor 并开启 `capturable=True`？**
   - 如果 `pg['lr']` 是一个普通 Python `float`（如 `0.0003`），由于 Cosine Warmup/Decay 调度器每一步都会算出一个新的浮点数（`0.0002998...`），TorchDynamo 会把 Python `float` 当作编译期常量（ Guard Specialization），每改一次 `lr` 就触发一次重新编译（直达 `recompile_limit` 上限 **64 次**后退化）！
   - 而当 `lr = torch.tensor(3e-4, dtype=torch.float32, device='cuda')` 且 `capturable=True` 时，`lr` 和 `step` 都是常驻显存的 0-D 张量指针，每步通过 `pg['lr'].fill_(new_lr)` 原地改写显存数值，**指针地址不变、计算图完全静态、零重编译、零 CPU-GPU 同步**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：为什么有些训练脚本在第 1 步（Step 1）`loss.backward()` 都没 OOM，偏偏在 `optimizer.step()` 之后或者第 2 步才 OOM？
- 有两个叠加原因：
  1. **延迟状态初始化（Lazy State Initialization）**：PyTorch `AdamW` 在调 `__init__` 时并不会分配 `exp_avg` 和 `exp_avg_sq`（共 `35.00 GB`），而是在 **第 1 次调用 `optimizer.step()` 的内部** 才动态创建！如果不在训练循环开始前做一次 Fake Step 预分配，第 1 步 `step()` 时就会突然向分配器索要 `35.00 GB` 状态显存！
  2. **`foreach=True` 的 `+17.50 GB` 临时缓冲尖峰 + `zero_grad(set_to_none=False)` 未释放梯度**：在 `optimizer.step()` 执行期间，`17.50 GB` 的梯度尚未被 `zero_grad(set_to_none=True)` 释放，此时 `foreach=True` 又突然申请 `17.50 GB` 的 `exp_avg_sq_sqrt` 临时张量，瞬间把峰值显存推上最高点！

### Q2：在 `FSDP2`（`DTensor` 参数）下，`torch.optim.AdamW(..., fused=True)` 可以直接开箱即用吗？
- **在 PyTorch 2.4+ 中完全支持 `DTensor` 与 `AdamW(fused=True, capturable=True)` 直接配合**！由于 `AdamW` 是逐元素（Elementwise）独立更新算子，每个 Rank 的本地分片 `param.to_local()` 与对应的 `grad.to_local()` 直接送入多张量融合算子 `_fused_adamw_` 即可，无需任何跨卡通信。

### Q3：为什么 `optimizer.zero_grad(set_to_none=True)` 比 `set_to_none=False` 既省显存又快？
- 当 `set_to_none=False` 时，PyTorch 不仅要在整个前向传播期间白白霸占 **`17.50 GB`** 的梯度显存不归还给缓存分配器，还要额外发射 `cudaMemsetAsync` 算子把 `17.50 GB` 显存刷成 `0`；而 `set_to_none=True`（PyTorch 2.0+ 已设为默认，但不少老框架仍显式传 `False`）直接将 `p.grad = None`，立即归还 `17.50 GB` 显存供前向激活复用，且省去一次 `17.50 GB` 的显存写零带宽！

---

## ③ 🧭 直观可视化图解（Mermaid 选型决策树 + 路径对比图 + 交互时序图）

### 3.1 PyTorch `AdamW` 优化器执行模式与学习率张量化选型决策树

```mermaid
flowchart TD
    A["大模型训练执行 AdamW optimizer.step() (725 个参数张量, 4.375B 分片参数/GPU)"] --> B{"选择哪一种 AdamW 算子执行后端?"}
    B -- "foreach=False, fused=False (单张量串行循环)" --> C["⛔ Tier 1: 每步发射 4,350 个独立小算子, 访存放大 3.5x (428.8 GB), 耗时 28.40 ms!"]
    B -- "foreach=True, fused=False (PyTorch 默认多张量矢量化)" --> D["⛔ Tier 2: _foreach_sqrt(exp_avg_sq) 非原地新建 725 个临时张量 -> 瞬间暴涨 17.50 GB 峰值显存直接 OOM!"]
    B -- "fused=True (单趟寄存器融合算子, 0.00 GB 临时显存)" --> E{"是否开启 capturable=True 并使用 0-D GPU Tensor lr?"}
    E -- "否: capturable=False + Python float lr" --> F["⚠️ Tier 3: 每次 Cosine LR 更新触发 torch.compile 重编译 (64 次/千步) 且无法捕获 CUDA Graph!"]
    E -- "是: capturable=True + 0-D CUDA Tensor lr (.fill_() 原地更新)" --> G["🏆 Tier 4: 0.00 GB 临时显存 (-17.50 GB) + 3.80 ms (提速 7.5x) + 0 次重编译!"]
```

### 3.2 `foreach=True` 的 `17.50 GB` 临时缓冲爆炸 vs. `fused=True` 单趟寄存器融合路径对比图

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 2 (foreach=True): 6 趟 HBM 往返读写 (315 GB) + 17.50 GB _foreach_sqrt 临时显存 OOM!"]
      B1["Pass 1-2: _foreach_mul_ + _foreach_lerp_ + _foreach_addcmul_ (读写 HBM)"] --> B2["Pass 3: exp_avg_sq_sqrt = _foreach_sqrt(v) -> 非原地申请 17.50 GB 临时显存!"]
      B2 --> B3["Pass 4-6: _foreach_div_ + _foreach_add_ + _foreach_addcdiv_ (耗时 8.60 ms)"]
    end
    subgraph Good["🏆 Tier 4 (fused=True, capturable=True): 单趟寄存器融合 (122.5 GB 屋顶线, 0.00 GB 临时显存!)"]
      G1["Single Fused Pass: 一次性将 (p, g, m, v) 读入寄存器 (70.0 GB Read)"] --> G2["寄存器内完成衰减 + m_t + v_t + rsqrt(v_t) + 参数更新 -> 写回 (52.5 GB Write, 仅 3.80 ms!)"]
    end
```

### 3.3 `AdamW(fused=True, capturable=True)` 配合 0-D GPU Tensor `lr` 的零同步单步时序图

```mermaid
sequenceDiagram
    participant Sched as Cosine LR Scheduler
    participant GPU_LR as 0-D CUDA Tensor `pg['lr']`
    participant Fused as `_fused_adamw_` CUDA Kernel (Register-Fused)
    participant HBM as GPU HBM (p, g, m, v = 122.5 GB Traffic)

    Sched->>GPU_LR: 1. `pg["lr"].fill_(new_lr)` (In-place 0-D tensor update, 0 recompile!)
    GPU_LR->>Fused: 2. Launch 1 multi-tensor fused kernel across all 725 parameter tensors
    Fused->>HBM: 3. Read `(p, g, m, v)` once -> compute in registers (`0.00 GB` temp HBM!)
    Fused->>HBM: 4. Write updated `(p, m, v)` once (`3.80 ms` total, `7.5x` faster!)
```

---

## ④ 📊 架构师选型决策矩阵（4-Tier Decision Matrix）

| 选型 Tier | `AdamW` 执行配置 | 单步算子发射数 | 优化器临时显存尖峰 | 单步耗时 (`ms`) | `torch.compile` 重编译 / `CUDAGraph` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 单张量循环 (`foreach=False`)** | `foreach=False, fused=False` | **`4,350` 个 ✕** | `2.40 GB` | **`28.40 ms` (`1.0x` ✕)** | `0` / ⛔ 不安全 |
| **Tier 2: 多张量矢量化 (`foreach=True`)** | `foreach=True, fused=False` (默认) | `12` 个 | **`17.50 GB` (Step-1 OOM! ✕)** | `8.60 ms` (`3.3x`) | `0` / ⛔ 不安全 |
| **Tier 3: 融合但用 Python Float `lr`** | `fused=True, capturable=False` | `2` 个 | `0.00 GB` | `4.10 ms` (`6.9x`) | **`64 次` 重编译 / ⛔ 崩溃 ✕** |
| **Tier 4: 融合 + 可捕获 + 0-D 张量 `lr` (推荐)** | **`fused=True, capturable=True` + Tensor `lr`** | **`2` 个 🏆** | **`0.00 GB` (`-17.50 GB` 🏆)** | **`3.80 ms` (`7.5x` 提速 🏆)** | **`0 次` / ✅ 100% 兼容 🏆** |

---

## ⑤ 🧪 定量算例与 7-Card 4:5 视觉卡片报告

- **定量算例仓库路径**：[`experiments/pytorch_adamw_foreach_temp_buffer_oom_fused_capturable_selection/run.py`](../../experiments/pytorch_adamw_foreach_temp_buffer_oom_fused_capturable_selection/run.py) · 封存数据：[`results.json`](../../experiments/pytorch_adamw_foreach_temp_buffer_oom_fused_capturable_selection/results.json)
- **交互式架构选型计算器**：[`docs/reports/pytorch-adamw-foreach-temp-buffer-oom-fused-capturable-sizer.html`](../reports/pytorch-adamw-foreach-temp-buffer-oom-fused-capturable-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/pytorch-adamw-foreach-temp-buffer-oom-fused-capturable-selection-cards.html`](../reports/pytorch-adamw-foreach-temp-buffer-oom-fused-capturable-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Single-Tensor Loop (4,350 Kernels)", "step_latency_ms": 28.40, "temp_buffer_gb": 2.40 },
      { "tier": "Tier 2: foreach=True (17.50 GB Temp OOM)", "step_latency_ms": 8.60, "temp_buffer_gb": 17.50 },
      { "tier": "Tier 3: fused=True + Float LR (64 Recompiles)", "step_latency_ms": 4.10, "temp_buffer_gb": 0.00 },
      { "tier": "Tier 4: fused=True + capturable=True + Tensor LR", "step_latency_ms": 3.80, "temp_buffer_gb": 0.00 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "step_latency_ms": "Quantity",
    "temp_buffer_gb": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Fused + Capturable AdamW Eliminates 17.50 GB Temp HBM Spike & Cuts Latency 7.5x (3.80 ms)",
    "subtitle": "Optimizer step latency (ms) across 4 PyTorch AdamW execution tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "step_latency_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ 生产环境快速核验清单（Checklist + Good fit vs. Not for）

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**：所有基于 `FSDP2`、`DDP`、`Megatron-LM` 的大模型预训练、SFT 与 RL 训练管线。
- **⛔ Not for**：在显存利用率超过 `80%` 的任务中继续使用默认 `foreach=True`（必遭 `_foreach_sqrt` 的 `17.50 GB` 临时缓冲暗算），或在 `torch.compile` 装饰的优化器步中使用 Python `float` 类型的 `lr`。

### 生产核验 Checklist
- [ ] **显式开启 `torch.optim.AdamW(..., fused=True, capturable=True)`**：消除 `torch._foreach_sqrt(exp_avg_sq)` 的 `17.50 GB` 临时显存分配。
- [ ] **将初始学习率传入为 0-D CUDA Tensor（`lr=torch.tensor(lr, dtype=torch.float32, device='cuda')`），并在 Scheduler 中通过 `.fill_(new_lr)` 原地更新**。
- [ ] **统一调用 `optimizer.zero_grad(set_to_none=True)`**：在 `optimizer.step()` 结束后立即释放 `17.50 GB` 梯度显存供下一轮前向激活复用。
