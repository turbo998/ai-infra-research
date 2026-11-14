# PyTorch `float32_matmul_precision`（`IEEE FP32` 慢速陷阱 vs. `TF32` Tensor Core vs. `Ozaki 3xBF16` 精确仿真）选型指南

> **Provenance & Evidence Tier**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)。定量数据锚定于 [`experiments/pytorch_tf32_fp32_matmul_precision_tensor_core_selection/results.json`](../../experiments/pytorch_tf32_fp32_matmul_precision_tensor_core_selection/results.json)。
> **English Version**: [PyTorch `float32_matmul_precision` (`IEEE FP32` vs. `TF32` Tensor Core vs. `Ozaki 3xBF16`) Selection Guide](./pytorch-tf32-fp32-matmul-precision-tensor-core-ozaki-selection-faq.en.md)

![PyTorch TF32 vs FP32 Matmul Precision Summary](../../assets/svg/pytorch-tf32-fp32-matmul-precision-tensor-core-ozaki-selection-faq-summary.svg)

---

## 0. 🔥 今日领域最热门能力突破追踪（Daily Hottest Breakthrough）

- **一行 `torch.set_float32_matmul_precision("high")` 破除 PyTorch 2.x 默认 `allow_tf32=False` 慢速陷阱（算力从 `38.2 TFLOPS` 飙升至 `365.0 TFLOPS`，提速 `9.55` 倍）+ cuBLAS `Ozaki Scheme (BF16x3)` 在 Tensor Core 上实现真 `23-bit IEEE FP32` 精度**：在 H100 / B200 大模型训练与推理代码中，哪怕主干开启了 `BF16 / FP8` 混合精度，模型中依然有大量算子为了数值稳定被显式强转为 `float32`（例如 **MoE 专家路由门控 `x.float() @ w_gate.float()`、强化学习 Value/Reward Head、DiT 扩散模型的时间步条件 MLP `adaLN` 以及部分 `lm_head`**）。然而，90% 的工程师不知道：**从 PyTorch 1.12 到 PyTorch 2.x，PyTorch 官方把 `torch.backends.cuda.matmul.allow_tf32` 的默认值改成了 `False`（即 `float32_matmul_precision = "highest"`，Tier 1）**！而 GPU 硬件中的 **Tensor Core 根本不支持原生 32-bit `IEEE FP32` 乘法**——一旦 `allow_tf32=False`，所有 `float32` 矩阵乘都会当场从 **`495 TFLOPS` 的 Hopper Tensor Core 摔下来，退化到峰值仅 `67 TFLOPS` 的传统 SIMT CUDA Core 上慢吞吞地跑（实测仅 `38.2 TFLOPS`，算力损失近 `90%`）**！通过显式设置 **`torch.set_float32_matmul_precision("high")`（启用 `TF32` Tensor Core，Tier 3）**，并在极端病态科学计算矩阵上结合 **cuBLAS `BF16x3 / 3TF32`（Ozaki 尾数三分法，Tier 4）**，即可彻底告别慢速 SIMT CUDA Core！

---

## 1. 📌 核心结论速览（Executive Summary — 3 分钟看懂）

1. **到底什么是 `TF32`（TensorFloat-32）？为什么它不用改任何 Tensor 类型就能让 `float32` 矩阵乘快 `9.55` 倍？**
   - **通俗大白话 + 第一性原理**：
     - 标准 **`IEEE FP32`** 有 32 位：`1` 位符号 + **`8` 位指数**（决定能表示多大的数） + **`23` 位尾数**（决定小数点后第 7 位有效数字）。但 Tensor Core 硬件乘法器嫌 23 位尾数电路面积太大，拒绝直接吃 32 位输入！
     - **`TF32`（19-bit 硬件内部格式）**：你的 PyTorch 张量在显存里**依然是标准的 32-bit `torch.float32`**；当 Tensor Core 读取它做乘法的瞬间，硬件自动把 23 位尾数四舍五入保留前 **`10` 位尾数**（保留完整的 **`8` 位指数**，因此动态范围和 FP32 完全一样，绝不溢出！），算完乘法后**立刻累加进完整的 `32-bit FP32`（23 位尾数）累加器**！
     - **结果**：显存格式零改动，仅需在脚本开头加一行 **`torch.set_float32_matmul_precision("high")`**，H100 上的 `float32` 矩阵乘算力瞬间从 **`38.2 TFLOPS` 暴涨至 `365.0 TFLOPS`（快 `9.55` 倍）**！
2. **为什么 PyTorch 2.x 里 `Conv2d` 默认是快（TF32）的，唯独 `Linear / matmul` 默认是慢（SIMT FP32）的？**
   - 这是一个极具迷惑性的历史遗留不一致：在 PyTorch 2.x 中，`torch.backends.cudnn.allow_tf32` 默认是 **`True`**（所以卷积默认走 TF32），但 `torch.backends.cuda.matmul.allow_tf32` 默认却是 **`False`**（所以所有 Transformer 里的 `Linear`、`matmul`、`bmm` 默认全走慢速 CUDA Core）！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：PyTorch 的 `set_float32_matmul_precision` 有三个档位：`"highest"`、`"high"`、`"medium"`，分别对应什么底层硬件行为？
- **答**：
  - **`"highest"`（PyTorch 默认值，Tier 1）**：禁用 Tensor Core，强制在 SIMT CUDA Core 上跑原生 23-bit `IEEE FP32`（H100 实测仅 `38.2 TFLOPS`）✕；
  - **`"high"`（强烈推荐黄金标准，Tier 3）**：启用 **`TF32` Tensor Core**（10-bit 尾数输入 + 32-bit FP32 累加器，实测 `365.0 TFLOPS`）✅；
  - **`"medium"`**：将 `float32` 输入强压为 **`BF16`**（仅 7-bit 尾数输入 + 32-bit 累加器），对于本就因为需要高于 BF16 精度才保留 `float32` 的 MoE Router Gate 来说，设为 `"medium"` 就失去了转为 `float32` 的意义。因此生产环境一律选 **`"high"`**！

### Q2：如果某个量子化学、CFD 流体仿真或极病态逆矩阵计算真的需要完整 `23-bit IEEE FP32` 精度，难道只能忍受 `38.2 TFLOPS` 的慢速 CUDA Core 吗？
- **答**：**不需要！这就是最新 cuBLAS / Blackwell 引入的 `Ozaki Scheme`（`BF16x3` / `FP8x4` 尾数切片仿真，Tier 4）**！
  - 原理极其巧妙：把一个 23-bit 尾数的 `FP32` 数字 `x` 拆成 **高 7 位、中 7 位、低 7 位三个不损失任何比特的 `BF16` 切片**：`x = x_high + x_mid + x_low`。
  - 然后调用 `989 TFLOPS` 的 `BF16` Tensor Core 做几次切片矩阵乘并高精度累加——**用 `989 TFLOPS` 的 Tensor Core 连乘 3~6 次，依然比 `67 TFLOPS` 的 CUDA Core 快 3~4 倍，且还原出 `100%` 完整 23-bit `IEEE FP32` 精度（相对误差 `1.2e-7`）**！

### Q3：为什么严禁开启 `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True`（Tier 2）？
- **答**：注意区分**“输入乘法精度”**和**“点积累加器精度（Accumulator）”**！Tensor Core 的灵魂在于“低精度相乘、**高精度（`FP32`）累加**”。一旦把累加器也降为 `16-bit FP16`（Tier 2），当隐藏层维度 `K = 4096 ~ 16384` 个乘积加在一起时，极易冲破 `FP16` 的 `65,504` 上限直接溢出为 `Inf / NaN`！

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + SIMT vs Tensor Core 路径图 + Ozaki 切片时序图）

### 3.1 架构师 PyTorch 矩阵乘精度配置决策树

```mermaid
flowchart TD
    A["训练/推理代码中存在 float32 矩阵乘 (MoE Gate / lm_head / DiT adaLN)"] --> B{"是否显式设置了 set_float32_matmul_precision？"}
    B -- "未设置 (Tier 1: 默认 'highest' / allow_tf32=False)" --> C["⛔ 掉下 Tensor Core! 在慢速 CUDA Core 上跑 38.2 TFLOPS<br/>浪费近 90% GPU 算力"]
    B -- "是" --> D{"任务对尾数有效位数的需求？"}
    D -- "AI 训练/推理常规高敏层 (10-bit 尾数足够)" --> E["✅ Tier 3: 设置 'high' (启用 TF32 Tensor Core + FP32 累加)<br/>零代码改动算力飙升 9.55 倍 (365 TFLOPS)"]
    D -- "主干网络 + 路由 + 严苛 23-bit 科学计算混合" --> F["🏆 Tier 4: 主干 BF16/FP8 (780 TF) + 路由 TF32 + 严苛层 cuBLAS BF16x3 Ozaki 拆分"]
    B -- "误开 16-bit 累加器 (Tier 2)" --> G["⚠️ allow_fp16_reduced_precision_reduction=True<br/>内积超过 65,504 直接溢出 NaN!"]
```

### 3.2 `IEEE FP32`（SIMT CUDA Core `38.2 TFLOPS`）vs. `TF32`（Tensor Core `365 TFLOPS`）位宽与硬件路径对比

```mermaid
flowchart LR
    subgraph SIMT["⛔ Tier 1: 默认 IEEE FP32 (allow_tf32=False)"]
        direction TB
        S1["1 Sign + 8 Exp + 23 Mantissa (32-bit)"] --> S2["Tensor Core 拒收 -> 退回慢速 SIMT CUDA Core (38.2 TFLOPS)"]
    end
    subgraph TF32["🏆 Tier 3: 显式开启 TF32 (set_float32_matmul_precision('high'))"]
        direction TB
        T1["硬件自动截取 1 Sign + 8 Exp + 10 Mantissa (19-bit)"] --> T2["直通 Hopper Tensor Core + 32-bit FP32 累加器 (365.0 TFLOPS · 快 9.55x!)"]
    end
```

### 3.3 Tier 4 分层精度路由与 `Ozaki BF16x3` 尾数无损拆分时序图

```mermaid
sequenceDiagram
    autonumber
    participant App as PyTorch 训练/推理主进程
    participant TC_FP8 as Tensor Core (BF16 / FP8 模式)
    participant TC_TF32 as Tensor Core (TF32 模式)
    participant Ozaki as cuBLAS Ozaki BF16x3 引擎

    App->>TC_FP8: 1. 骨干 QKV / FFN 线性层 -> 跑 BF16/FP8 + FP32 累加 (780 TFLOPS)
    App->>TC_TF32: 2. MoE Router Gate / Reward Head (float32) -> 自动走 TF32 (365 TFLOPS)
    App->>Ozaki: 3. 极端病态求逆矩阵 -> 拆分为 High/Mid/Low 3 个 BF16 尾数片在 Tensor Core 上合成 23-bit 精度!
```

---

## 4. 📊 架构师选型决策矩阵（4-Tier Selection Matrix）

| 评估维度 | Tier 1: 默认 `'highest'` (`IEEE FP32`) | Tier 2: 错误开启 `16-bit` 累加器 | Tier 3: 显式 `'high'` (`TF32` Tensor Core) | Tier 4: 分层精度 (`BF16/FP8` + `TF32` + `Ozaki BF16x3`) |
| :--- | :--- | :--- | :--- | :--- |
| **底层执行硬件单元** | **SIMT CUDA Core (慢)** ✕ | Tensor Core | **Tensor Core (`mma.sync.tf32`)** ✅ | **全线 Tensor Core** 🏆 |
| **输入尾数 / 累加器位宽** | `23-bit / 32-bit` | `10-bit / 16-bit` ⚠️ | **`10-bit / 32-bit`** ✅ | **`7~23 bit 按需 / 32-bit`** 🏆 |
| **H100 实测持续算力** | `38.2 TFLOPS` ✕ | `680.0 TFLOPS` (易溢出) | **`365.0 TFLOPS` (快 9.55 倍)** ✅ | **`780.0 TFLOPS` (主干满血)** 🏆 |
| **相对 L2 数值误差** | `1.0e-7` | **`2.5e-2` (严重失真/NaN)** ✕ | `4.8e-4` (AI 完美容忍) | **`1.2e-7` (Ozaki 精确还原)** 🏆 |

---

## 5. 🧪 定量算例与工具链（Interactive Sizer & 7-Card Visual Report）

- **实验脚本与密封数据**：[`experiments/pytorch_tf32_fp32_matmul_precision_tensor_core_selection/run.py`](../../experiments/pytorch_tf32_fp32_matmul_precision_tensor_core_selection/run.py) · [`results.json`](../../experiments/pytorch_tf32_fp32_matmul_precision_tensor_core_selection/results.json)
- **交互式 TF32 矩阵乘精度计算器**：[`docs/reports/pytorch-tf32-fp32-matmul-precision-tensor-core-ozaki-sizer.html`](../reports/pytorch-tf32-fp32-matmul-precision-tensor-core-ozaki-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/pytorch-tf32-fp32-matmul-precision-tensor-core-ozaki-selection-cards.html`](../reports/pytorch-tf32-fp32-matmul-precision-tensor-core-ozaki-selection-cards.html)

---

## 6. 🛠️ Good Fit vs. Not For & 生产环境快速核验清单（Checklist）

### ✅ Good Fit vs. ⛔ Not For

| 维度 | ✅ Good Fit（强烈推荐） | ⛔ Not For（严禁误用） |
| :--- | :--- | :--- |
| **所有 PyTorch 2.x 训练与推理入口** | 在 `main()` 第一行显式调用 `torch.set_float32_matmul_precision("high")`（开启 `TF32`） | 保持 PyTorch 2.x 默认值不管，导致 `float32` 的 MoE 路由门控与 `lm_head` 退化到慢 9.55 倍的 CUDA Core |
| **累加器精度保护** | 保持 `allow_fp16_reduced_precision_reduction = False` 且 `allow_bf16_reduced_precision_reduction = False` | 为了微不足道的提速把 Tensor Core 的 32-bit 累加器降级为 16-bit |

### 生产环境上线核验清单（Pre-Flight Checklist）

- [ ] **核验训练/推理启动脚本已开启 `TF32`**：确认 `torch.get_float32_matmul_precision() == "high"`（或 `torch.backends.cuda.matmul.allow_tf32 == True`）。
- [ ] **核验 Nsight Systems / Compute 中无意外的 `sgemm`（SIMT FP32）内核**：确认所有 `float32` 矩阵乘调用的底层 SASS 指令为 Tensor Core `HMMA / WGMMA (TF32)` 而非 SIMT `FFMA`。
- [ ] **核验 `32-bit` 累加器未被关闭**：确认 `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction` 为 `False`。
