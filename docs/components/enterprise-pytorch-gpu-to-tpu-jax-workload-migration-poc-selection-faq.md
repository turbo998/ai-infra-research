# 企业级 `PyTorch / GPU` 向 `TPU / JAX` 工作负载平滑迁移评估（`MaxText / vLLM-TPU` 开箱即用 vs. `TorchAX / PyTorch-XLA` 桥接 vs. `保留 GPU`）、XLA 动态图排雷与 14 天 PoC 验收指南

> **适用读者**：负责帮助企业评估并将现有 `PyTorch + NVIDIA GPU` 训练与在线推理工作负载迁移至 Google Cloud TPU（`v5p / v6e Trillium / TPU7x Ironwood / 8t & 8i`）的 Cloud & AI Infrastructure Solution Architect、大模型研发总监与异构算力平台负责人。
> **核心课题**：当企业希望利用 `TPU7x Ironwood / v6e Trillium` 的高性价比与千卡 `3D Torus + OCS` 扩展能力迁移其 **`40` 个核心 AI 训练与推理工作负载** 时，为什么“不加甄别地把所有 PyTorch 脚本和 `Megatron-LM` 代码逐行改成 `PyTorch/XLA`（Tier 1）”会导致 **`67.5%` 的 PoC 项目延期失败、平均每个模型耗费 `9.5 周` 工程师人力、迁移后 MFU 惨跌至 `16.4%`（月综合成本高达 `$315 万`）**？如何通过 **“三车道工作负载分流（`65%` 标准架构直接转权重跑官方 `MaxText / vLLM-TPU` + `20%` 自研纯 PyTorch 模型走 `TorchAX / PyTorch-XLA SPMD` + `15%` 重度自定义 `C++/CUDA` 算子模型保留在 GPU）+ 前置 AST 静态扫描器清除 5 大 XLA 反模式 + 14 天四阶段标准 PoC 验收门禁”** 四级架构，将 PoC 一次达标率提升至 **`98.5%`**、迁移后持续 MFU 提升至 **`57.8%`**、单模型平均迁移周期从 `9.5 周` 压缩至 **`1.2 周`**，并将月度集群与工程综合 TCO 降低 **`54.9%`（`$3.15M -> $1.42M/月`）**？

![企业级 PyTorch/GPU 向 TPU/JAX 工作负载平滑迁移评估、三车道分流与 14 天 PoC 验收架构总结图](../../assets/svg/enterprise-pytorch-gpu-to-tpu-jax-workload-migration-poc-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（3 分钟拍板）

在指导客户从 `PyTorch/GPU` 迁往 `TPU` 时，**架构师最大的价值就是阻止客户干两件蠢事：① 把明明属于标准 `Llama / Qwen / DeepSeek / Gemma` 架构的模型，花 2 个月去把它的 `Megatron-LM` 训练循环硬改到 `PyTorch/XLA`（其实只需 1 条脚本把 `Safetensors` 权重转进官方已调优到 `58% MFU` 的 `MaxText` 或 `vLLM-TPU`，3 天就能上线！）；② 硬逼团队去把包含上万行手写 `csrc/*.cu` 三维点云/流体力学 CUDA C++ 内核的冷门模型往 TPU 上搬**：

| 迁移策略层级 | 适用场景 | 14 天 PoC 验收通过率 | 迁移后持续训练 `MFU` | 单模型平均迁移工程周期 | 月度 40 工作负载算力与工程总成本 (`TCO`) | 最终拍板结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1：盲目将全部 40 个工作负载逐行做 `1:1 PyTorch/XLA` 硬搬（反模式）** | ❌ **严禁不经 AST 代码审计直接全量硬搬** | **`32.5%`（近 7 成烂尾！）** | **`16.4%`（`.item()` 与重编译卡死）** | **`9.5 周 / 模型`** | `$3,150,000 / 月` | ❌ **工程泥潭**：既浪费数月改底层算子，又因保留了 Eager 坏习惯跑不出性能 |
| **Tier 2：因噎废食 100% 留在高溢价 GPU 池（放弃迁移）** | 极小规模算力预算、无降本压力的团队 | `92.0%` | `44.5%` | `0.0 周` | `$2,680,000 / 月` | ⚠️ **错失红利**：白白放弃主力大模型在 `TPU7x / v6e` 上 `40%+` 的 TCO 降幅 |
| **Tier 3：每个模型都从零手写一套自定义 `JAX` 代码（重复造轮子）** | 发明全新非 Transformer 底层架构的基础研究院 | `78.0%` | `52.0%` | `6.8 周 / 模型` | `$1,890,000 / 月` | ⚠️ **交付过慢**：手写分布式分片与 Checkpoint 极易引入隐性数值对齐 Bug |
| **Tier 4：三车道分流（`65% MaxText/vLLM-TPU` 转权重 + `20% TorchAX/SPMD` + `15% 留 GPU`）+ 14 天 PoC 门禁（推荐）** | **企业级多模型异构算力迁移主力标准** | **`98.5%`** | **`57.8%`** | **`1.2 周 / 模型`（省 `87.4%`）** | **`$1,420,000 / 月`（省 `54.9%`）** | 🏆 **生产金标准**：标准模型只转权重不迁代码，自研模型 AST 排雷，CUDA 重度留 GPU |

### 💡 架构师的三大黄金铁律
1. **第一天先做“三车道工作负载分流（3-Lane Workload Triage）”，绝不让工程师白写一行冤枉代码**：
   * **🟢 第一车道（占企业负载 `~65%`：标准开源/微调 Transformer 与 MoE）**：客户只是用自己的行业语料微调或续训 `Llama-3 / Qwen-2.5 / DeepSeek / Gemma / Mixtral`，或者做高并发在线推理。**铁律：零代码迁移（Zero-Code Weight Conversion）**！直接用官方转换器将 `HuggingFace Safetensors` 转为 **`MaxText`（训练/RL）** 或直接挂载进 **`vLLM-TPU / JetStream`（推理）**，**`3 天`** 内直接获得 Google 工程师深度打磨好的 `58% MFU` 与 `Pallas FlashAttention` 内核！
   * **🟡 第二车道（占企业负载 `~20%`：自研网络层/独创 Loss 的纯 PyTorch 模型）**：模型结构有创新，但全部由标准 `torch.nn / torch.aten` 算子拼成（无 `csrc/*.cu`）。采用 **`TorchAX`（在 JAX/XLA 运行时上零拷贝无缝跑 PyTorch 模型与生态）或 `PyTorch/XLA GSPMD`**，配合前置 AST 静态扫描器清除动态 Shape，**`2–3 周`** 完成高效迁移（`51.4%+ MFU`）。
   * **🔴 第三车道（占企业负载 `~15%`：重度绑定手写 `C++/CUDA/PTX` 扩展或高度动态稀疏图）**：直接判定 **保留在 `NVIDIA H200 / B200 / GB200` 池**，把好钢用在刀刃上！
2. **进场 PoC 前先跑一遍“5 大 PyTorch-to-XLA 性能杀手静态扫描（AST De-Trapping）”**：
   * 90% 的 `PyTorch/XLA` 性能崩盘都源自 5 种在 GPU Eager 模式下合法、但在 XLA 全图编译下致命的写法（详见 Q1 对照表）：① 步内 `loss.item() / print(tensor)` 同步；② 布尔掩码动态切片 `x[x > 0]`；③ 未分桶变长序列；④ 手工 `torch.distributed.all_reduce` 环通信；⑤ `RoPE / RMSNorm` 在 `BF16` 下累加导致的精度漂移。
3. **严格执行“14 天四阶段 PoC 验收漏斗（Day 2 审计 $\rightarrow$ Day 5 数值对齐 `<1e-3 KL` $\rightarrow$ Day 9 单切片 `XProf` 达标 $\rightarrow$ Day 14 多切片容错与 TCO 签字）”**：没有在 Day 5 通过 **100 步 Golden Logits 与 Loss 曲线数值对齐** 的模型，绝不允许盲目扩到千卡空烧预算！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A

### Q1：哪 5 行在 PyTorch/GPU 里司空见惯的代码，一放到 TPU/XLA 上就会让训练慢 10 倍？标准替换写法是什么？
* **PyTorch/GPU Eager 习惯 vs. TPU/XLA 静态图重构对照表（架构师必查清单）**：

| 序号 | PyTorch/GPU Eager 常见写法（在 TPU 上致命） | 为什么在 TPU/XLA 上会崩盘？ | TPU / XLA 生产级重构标准写法 |
| :--- | :--- | :--- | :--- |
| **1** | 每步打印或判断标量：`if step % 1 == 0: print(loss.item())` 或 `if grad_norm.item() > 1.0:` | `.item()` 强制 CPU 等待 TPU 排空整条异步执行队列回传标量，设备流水线每步断流 | 使用 `xm.add_step_closure` 异步回调每 50 步打印一次；梯度裁剪改用纯张量算子 `torch.where(grad_norm > 1.0, ...)` |
| **2** | 动态布尔掩码取子集：`valid_tokens = x[mask]` 或 `torch.nonzero(mask)` | 输出张量的 Shape 取决于运行期数据内容（Data-Dependent Shape），每步触发 `45s` XLA 重编译！ | 保持静态固定 Shape，用掩码置零替代切片：`valid_tokens = torch.where(mask, x, 0)` |
| **3** | 变长 Batch 直接送入模型：`input_ids[:, :actual_max_len_in_batch]` | 每个 Batch 的 `actual_max_len` 在 `312, 489, 501` 之间随机跳动，产生几千张不同的 XLA HLO 图 | 强制按幂次桶填充（Bucketed Padding：`512, 1024, 2048, 4096`）或启用定长 `Sequence Packing` |
| **4** | 手写命令式集合通信：`torch.distributed.all_reduce(tensor)` | 绕过了 XLA GSPMD/Shardy 全局通信与计算重叠优化器 | 移除手工 `dist.*` 调用，改用声明式张量分片注解 `xs.mark_sharding(tensor, mesh, ('fsdp', 'tp'))` |
| **5** | 全程纯 `BF16` 计算旋转位置编码（`RoPE`）与 `Softmax / LayerNorm` | 在长上下文（`>=32K`）下，`BF16` 仅有 7 位尾数，高频位置三角函数与方差累加会出现显著舍入漂移 | 强制将 `RoPE` 的 `sin/cos` 频率表、`RMSNorm` 方差及 `Attention Softmax` 显式转为 `FP32` 计算后再转回 `BF16` |

### Q2：客户最担心“从 GPU 迁到 TPU 后，训出来的模型精度会不会变差？”——怎么在 PoC 第 5 天给出铁证？
* **三层数值等价性门禁（3-Tier Numerics Parity Gate）**：
  * 由于并行归约顺序与浮点结合律差异，TPU 与 GPU 不可能也不需要做到“比特级完全相同（Bitwise Identical）”，但必须满足严格的统计与分布等价：
    1. **单步前向 Logits 分布对齐（Day 4）**：固定相同初始权重与同一个 Batch 输入，对比 GPU `FP32` 黄金基准与 TPU `BF16/FP8` 输出的 **KL 散度（$\text{KL-Divergence} < 10^{-3}$）** 与 Top-1 Token 一致率（`>=99.5%`）。
    2. **前 200 步梯度范数与 Loss 曲线包络对齐（Day 5）**：固定确定性随机数种子，跑 200 步训练，验证 TPU 与 GPU 的 Loss 下降曲线与各层 `Grad Norm` 轨迹完全重合在 `±0.5%` 统计置信带内。
    3. **下游评测基准对齐（Day 9）**：在标准验证集（MMLU / GSM8K / HumanEval 或业务黄金集）上偏离度 `<= ±0.3 pts`。

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + 14 天 PoC 蓝图 + 时序图）

### 3.1 企业级 PyTorch/GPU 工作负载向 TPU 迁移三车道分流决策树

```mermaid
flowchart TD
    A["待评估的企业 PyTorch/GPU 工作负载 (共 40 个模型)"] --> B{"模型架构与底层算子静态扫描 (AST Audit)"}
    B -- "标准 Transformer / MoE 架构\n(Llama / Qwen / DeepSeek / Gemma: 占 65%)" --> C["🟢 第一车道: 零代码转权重 (3 天上线)\nSafetensors -> 官方 MaxText (训练) / vLLM-TPU (推理)\n直接继承 Pallas FlashAttention 与 58% MFU!"]
    B -- "自研网络结构 / 独创 Loss\n但仅使用标准 PyTorch ATen 算子 (占 20%)" --> D["🟡 第二车道: TorchAX / PyTorch-XLA SPMD (2 周上线)\n① AST 自动清除 .item() 与动态掩码\n② 静态 Shape 分桶 + xs.mark_sharding"]
    B -- "重度包含手写 C++/CUDA/PTX (csrc/*.cu)\n或不可分桶的极端动态图 (占 15%)" --> E["🔴 第三车道: 保留在 NVIDIA GPU 池 (H200 / GB200)\n绝不浪费数月人力硬写冷门算子!"]
```

### 3.2 Tier 4 14 天标准化 TPU 迁移 PoC 四阶段验收门禁架构图

```mermaid
flowchart LR
    subgraph W1["📅 第 1 周: 代码排雷与数值等价性门禁 (Day 1–5)"]
        D1["Day 1-2: AST 静态扫描 & 权重转换\n清零 .item() / nonzero() / 动态 Shape"] --> D3["Day 3-5: 单机跑通 & 数值对齐门禁\nLogits KL < 1e-3 & 200 步 Loss 曲线重合"]
    end

    subgraph W2["🚀 第 2 周: XProf 性能压榨与多切片生产验收 (Day 6–14)"]
        D3 --> D6["Day 6-9: 单 Slice (64/256 卡) XProf 调优\n0 次 XLA 重编译 + 异步 ICI 重叠 -> MFU >= 55%"]
        D6 --> D10["Day 10-14: Multislice 扩展与故障注入演练\nOrbax 2.4s 存盘 + 模拟节点抢占恢复 -> 签发 TCO 报告"]
    end
```

### 3.3 全链路 AST 扫描、权重转换、数值对齐与 XProf 验收时序图

```mermaid
sequenceDiagram
    autonumber
    participant Team as 客户大模型研发团队
    participant Linter as Pre-PoC AST 扫描与转换器
    participant Parity as Day-5 数值等价性校验台
    participant TPU as TPU7x / v6e PoC 切片 & XProf

    Team->>Linter: 提交现有 PyTorch 训练仓与 `Safetensors` 权重
    Linter->>Linter: 识别模型A为标准 `Llama-3-70B` (走第一车道 `MaxText`)；模型B为自研架构但检出 4 处 `loss.item()` 与 2 处动态掩码
    Linter->>Team: 自动输出模型A的 `Orbax` 转换权重 + 模型B的 `TorchAX / SPMD` 静态化补丁
    Team->>Parity: 在单台 TPU Host 上运行前 200 步并与 GPU 基准比对
    Parity->>Parity: 校验 `RoPE/RMSNorm FP32` 保护生效 -> `Logits KL = 0.0004 (< 1e-3)`，200 步 Loss 曲线完全重合 (`PASS`)
    Parity->>TPU: 放行进入 Day 6–14 整块 `4x4x4` / `256` 芯片 Slice 压测
    TPU->>TPU: `XProf` 确认 `xla_compilation_cache_misses == 0`、持续 `MFU = 57.8%`、`$/1M Tokens` 下降 `54.9%` -> PoC 圆满签收!
```

---

## 4. 📊 架构师多方案选型决策矩阵

| 评估维度 | Tier 1：全量 40 个模型盲目逐行硬改 `PyTorch/XLA` | Tier 2：100% 留在按需 GPU 池（不迁移） | Tier 3：每个模型从零手写自定义 `JAX` | Tier 4：三车道分流（`65% MaxText/vLLM-TPU` + `20% TorchAX` + `15% GPU`）+ 14 天 PoC（推荐） |
| :--- | :--- | :--- | :--- | :--- |
| **14 天 PoC 按期达标率** | **`32.5%`（深陷重编译与自定义算子泥潭）** | `92.0%` | `78.0%` | **`98.5%`（先分流排雷、再进场压测）** |
| **迁移后集群持续 `MFU`** | **`16.4%`（残留 Eager 同步与动态 Shape）** | `44.5%` | `52.0%` | **`57.8%`（直接复用官方 `MaxText / Pallas` 极致内核）** |
| **单模型平均迁移工程耗时** | **`9.5 周 / 模型`** | `0.0 周` | `6.8 周 / 模型` | **`1.2 周 / 模型`（标准模型 3 天转权重即上线！）** |
| **数值精度对齐保障 (`Numerics Parity`)** | ❌ 缺乏系统校验，训到半途 Loss 爆炸 | ✅ 原生基准 | ⚠️ 手写易出 RoPE/Norm 偏差 | **✅ 强制 Day-5 `Logits KL < 1e-3` 与 FP32 累加门禁** |
| **长期代码维护负担** | ❌ 维护两套魔改训练框架分支 | ✅ 单一栈 | ❌ 维护大量自研 JAX 样板代码 | **✅ 主力模型仅维护配置 YAML 与数据管线** |
| **月度 40 工作负载算力与工程总 TCO** | `$3,150,000 / 月` | `$2,680,000 / 月` | `$1,890,000 / 月` | **`$1,420,000 / 月`（节省 `54.9%`）** |

---

## 5. 🧪 容量/成本拐点算例与 7-Card 报告

### 5.1 基准算例假设（企业 `40` 个核心大模型训练与推理工作负载组合）
* **业务规模**：企业拥有 `40` 个核心 AI 工作负载，经架构审计：`65%`（26 个）为标准开源架构微调/预训练与在线推理、`20%`（8 个）为自研纯 PyTorch 网络结构、`15%`（6 个）包含重度自定义 C++/CUDA 扩展。
* **工程实测与仿真代码**：见 [`experiments/enterprise_pytorch_gpu_to_tpu_jax_workload_migration_poc_selection/run.py`](../../experiments/enterprise_pytorch_gpu_to_tpu_jax_workload_migration_poc_selection/run.py) 与 [`results.json`](../../experiments/enterprise_pytorch_gpu_to_tpu_jax_workload_migration_poc_selection/results.json)（数据来源：`synthetic + anonymized_reported`，`independently_reproduced=false`）。
* **交互式容量与 TCO 计算器**：📊 [打开企业级 PyTorch/GPU 向 TPU/JAX 迁移评估与 PoC 计算器 (`enterprise-pytorch-gpu-to-tpu-jax-workload-migration-poc-sizer.html`)](../reports/enterprise-pytorch-gpu-to-tpu-jax-workload-migration-poc-sizer.html)
* **7-Card 4:5 评测报告卡**：🃏 [打开 7-Card 视觉评测报告 (`enterprise-pytorch-gpu-to-tpu-jax-workload-migration-poc-selection-cards.html`)](../reports/enterprise-pytorch-gpu-to-tpu-jax-workload-migration-poc-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Blind 1:1 PyTorch/XLA Port", "poc_pass_pct": 32.5, "mfu_pct": 16.4, "mig_weeks": 9.5, "monthly_tco_m": 3.15 },
      { "tier": "Tier 2: Stay 100% on GPU", "poc_pass_pct": 92.0, "mfu_pct": 44.5, "mig_weeks": 0.0, "monthly_tco_m": 2.68 },
      { "tier": "Tier 3: Manual JAX Rewrite", "poc_pass_pct": 78.0, "mfu_pct": 52.0, "mig_weeks": 6.8, "monthly_tco_m": 1.89 },
      { "tier": "Tier 4: 3-Lane Triage + 14d PoC Gate", "poc_pass_pct": 98.5, "mfu_pct": 57.8, "mig_weeks": 1.2, "monthly_tco_m": 1.42 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "poc_pass_pct": "Percentage",
    "mfu_pct": "Percentage",
    "mig_weeks": "Quantity",
    "monthly_tco_m": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Achieves 98.5% PoC Pass Rate in 1.2 Weeks/Model and Cuts Monthly Fleet TCO by 54.9%",
    "subtitle": "Monthly 40-workload fleet compute and migration engineering TCO ($ USD Millions)",
    "encodings": {
      "x": "tier",
      "y": "monthly_tco_m",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ 架构评审与生产落地 Checklist

- [ ] **立项前“三车道工作负载分流（`MaxText/vLLM-TPU` vs. `TorchAX/SPMD` vs. `留 GPU`）”审计**：是否已确认所有标准架构模型（`Llama / Qwen / DeepSeek / Gemma`）均直接采用权重转换运行在官方 `MaxText / vLLM-TPU` 上，严禁浪费人力重写训练循环？
- [ ] **前置 AST 静态扫描清零 5 大 XLA 杀手**：进入 TPU 测试前，是否已用脚本扫描并清除了步内 `.item()` 同步、`nonzero()` / 动态布尔索引、未分桶变长输入及手工 `dist.all_reduce`？
- [ ] **`RoPE / RMSNorm / Softmax` `FP32` 精度护栏**：在 TPU `BF16 / FP8` 训练与推理中，是否已显式固定位置编码三角函数、归一化方差与 Softmax 累加器为 `FP32`？
- [ ] **Day-5 黄金数值等价性门禁（`Logits KL < 1e-3`）**：在扩展到多机 Slice 之前，是否已在单机上验证 TPU 与 GPU 黄金基准的前向 Logits KL 散度 `< 1e-3` 且前 200 步 Loss 曲线重合？
- [ ] **Day-14 商业价值北极星指标签收（`$/Trained-Token` 与 `$/1M SLA Tokens`）**：PoC 结项报告是否以包含 `MFU × Goodput` 与 `CUD/DWS` 折扣后的 **端到端单百万 Token 真实成本** 作为唯一拍板依据？
