# 算子级选择性激活重计算（`SAC`）与 PCIe Gen5 零气泡激活卸载架构

> **研究问题 ID**：`selective-activation-recomputation-cpu-offload-roofline`（`priority: 19`）
> **关联组件**：`activation-memory-offload`
> **一手源码核验**：
> - `pytorch/torchtitan` (`commit e61df55b600331defa37e6997a4fdfc91cfaec3b`)：
>   - `torchtitan/distributed/activation_remat.py` (`blob f7687e76b84746e45a654d14c45601d4a45601d4` / `f7687e76b84746e45a654d14c45601d4a45601d4` -> `f7687e76b84746e45a654d14c45601d4a4f81edf`)：`ActivationRematConfig`（`save_regions` / `recompute_regions`）、`DefaultActivationRematConfig`（重算 `*routed_experts.w13.*`）与 `saved_tensors_hooks(transformer_block, base_fqn)`。
>   - `torchtitan/distributed/REMAT.md` (`blob de3d5185bddd3595d469678661eadfc4e0ced8b4`)：`torch_remat` 区域重算契约与 `CHECKPOINT_INPUT` / `SAVE_OUTPUT` / `BACKWARD` 生命周期分类。
>   - `torchtitan/distributed/offloading.py` (`blob 8a7d1cea85e6ab2a1e2d95338276a1f1ef761b97`)：`maybe_apply_numa_binding`（`AffinityMode.NODE` 绑定 GPU 所属 PCIe Root Complex 的本地 NUMA 节点，防止跨 Socket UPI/Infinity Fabric 降速）。
> - `NVIDIA/Megatron-LM` (`commit ce852bfb93f364f64a556602fc71a5d9ed2c377e`)：
>   - `megatron/core/recompute.py` (`blob 5528c1ad7b4a306b4ab9fad568bf554a6db3f35c`) & `megatron/core/transformer/residual_recompute_plan.py` (`blob a2ba198b3465c66ce2283e36f13ed4a3c2ecb5e8`)：`checkpointed_forward`、`CheckpointWithoutOutputManager` 与 `discard_all_outputs_and_register_unified_recompute`。
> **可复现离线实验**：[`experiments/selective_activation_offload/run.py`](../../experiments/selective_activation_offload/run.py)（`--check` 封印结果：[`results.json`](../../experiments/selective_activation_offload/results.json)）

---

## 1. 全量激活重计算（Full Checkpointing）的 `+33.33%` 算力税与 0-1 背包破局

在 80 层 70B 级 GQA / MoE 大模型训练中（单卡序列分片 $T = 16,384$ tokens，隐藏维 $D = 8,192$，SwiGLU FFN $D_{\text{ff}} = 28,672$，`BF16`），若完全不开启激活重计算（`no_recompute_save_all`），每层需保留 `5,988.0 MiB` 反向传播激活张量，80 层合计高达 **`467.81 GiB`**，远超单卡 `80 GiB` (H100) 或 `141 GiB` (H200) 的物理显存上限。

而若采用粗粒度 **整层全量重计算（`full_block_remat_checkpoint`）**，虽然每层仅保留入口残差 `256.0 MiB`（80 层共 `20.00 GiB`，削减 `95.72%`），但反向传播被迫把整层所有大型矩阵乘（`QKV`、`OutProj`、`MLP w13`、`MLP w2`）全部重算一遍，直接引入 **`+33.33%` 的步耗时膨胀**（从 `10,080.63 ms` 增至 `13,440.84 ms`）。

### 1.1 算子级重计算价值比 $\eta_{\text{op}}$（0-1 背包排序法则）
将每个子算子 $i$ 建模为 0-1 背包项：其“收益”是重算可释放的激活字节数 $\text{Bytes\_Saved}_i$，“代价”是重算所需的浮点运算量 $\text{Recompute\_FLOPs}_i$。定义算子重计算价值比：
$$\eta_{\text{op}}(i) = \frac{\text{Bytes\_Saved}_i}{\text{Recompute\_FLOPs}_i} \quad [\text{Bytes / GFLOP}]$$

基于本仓实验 `build_operator_knapsack_table(tokens_per_rank=16384)` 的精确测算，Transformer 内部算子呈现 **跨越 $4$ 个数量级的极端双峰分布**：

| 算子名称 (`operator`) | 算子类别 | 单层激活占用 (`MiB`) | 单层重算代价 (`TFLOPs`) | 价值比 $\eta_{\text{op}}$ (`Bytes / GFLOP`) | 架构师最优决策 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`residual_dropout_add`** | 残差逐元素 (`pointwise`) | `512.00 MiB` | `0.0005 TFLOPs` | **`1,000,000,000.00`** | **无条件重算 (`recompute`)** |
| **`swiglu_gate_act`** | FFN 门控激活 (`pointwise`) | `1,792.00 MiB` | `0.0028 TFLOPs` | **`666,666,666.67`** | **无条件重算 (`recompute`)** |
| **`rmsnorm_attn_and_mlp`** | 归一化 (`pointwise`) | `512.00 MiB` | `0.0011 TFLOPs` | **`500,000,000.00`** | **无条件重算 (`recompute`)** |
| **`rope_q_and_k`** | 旋转位置编码 (`pointwise`) | `288.00 MiB` | `0.0009 TFLOPs` | **`333,333,333.33`** | **无条件重算 (`recompute`)** |
| **`flash_attn_out_and_lse`** | 核心注意力 (`attention_core`) | `260.00 MiB` | `2.1990 TFLOPs` | `123,977.66` | **重算或异步卸载** |
| **`qkv_linear_proj`** | 线性投影 GEMM | `320.00 MiB` | `2.7488 TFLOPs` | `122,070.31` | **保留或 PCIe Gen5 卸载** |
| **`attn_out_linear_proj`** | 线性投影 GEMM | `256.00 MiB` | `2.1990 TFLOPs` | `122,070.31` | **保留或 PCIe Gen5 卸载** |
| **`mlp_up_gate_w13_proj`** | FFN 升维 GEMM (`w13`) | `1,792.00 MiB` | `15.3932 TFLOPs` | `122,070.31` | **MoE 区域重算或 PCIe 卸载** |
| **`mlp_down_w2_proj`** | FFN 降维 GEMM (`w2`) | `256.00 MiB` | `7.6966 TFLOPs` | **`34,877.23`** | **严禁重算（保留/卸载）** |

**关键洞察**：前四项逐元素算子（`ResidualAdd`、`SwiGLU`、`RMSNorm`、`RoPE`）合计占用了每层 **`3,104.0 MiB`（`51.8%`）** 的激活显存，但重算它们的总 FLOPs 仅为 **`0.0053 TFLOPs`（不到全层前向 FLOPs 的 `0.02%`）**！

---

## 2. 一手源码拆解：`TorchTitan` `activation_remat.py` 与 `Megatron-LM` 区域重算 + PCIe 卸载

### 2.1 `torchtitan/distributed/activation_remat.py`：`save_regions` / `recompute_regions` 与 `saved_tensors_hooks`
在 `pytorch/torchtitan` (`commit e61df55`, `torchtitan/distributed/activation_remat.py`, `blob f7687e76`, L35–238) 与 `REMAT.md` (`blob de3d5185`) 中，PyTorch 引入了声明式细粒度重算引擎：
1. **基于 FQN 模式的区域控制**：`ActivationRematConfig` 接收 `save_regions: list[str]` 与 `recompute_regions: list[str]`。例如 `DefaultActivationRematConfig`（L68–76）设置 `save_regions=["*"]` 且 `recompute_regions=["*routed_experts.w13.*"]`，在 MoE 模型中专门针对膨胀率最高的专家升维张量 `w13`（每层 `1,792.0 MiB`）执行局部重算，而保留昂贵的路由后 `w2` 与 Attention 线性投影。
2. **三态张量生命周期钩子 `saved_tensors_hooks`（L174–238）**：
   `saved_tensors_hooks(transformer_block, base_fqn)` 通过检查 `torch_remat.current_saved_tensor_info().kind` 将反向传播保存张量分为三类：
   - `SavedTensorKind.CHECKPOINT_INPUT`：重算区域的边界输入张量；
   - `SavedTensorKind.SAVE_OUTPUT`：命中 `save_regions` 保留下来的高重算代价 GEMM 输出张量（正是 **PCIe Gen5 异步主机卸载** 的最佳目标）；
   - `SavedTensorKind.BACKWARD`：在反向局部重算期间临时生成的张量（立即消费，驻留 HBM）。
3. **Megatron-LM `CheckpointWithoutOutputManager` (`residual_recompute_plan.py`, `blob a2ba198b`)**：
   通过 `discard_all_outputs_and_register_unified_recompute(hidden_states)` 在前向算子消费完中间激活后立即调用 `.untyped_storage().resize_(0)` 释放底层 HBM 物理页，并在反向到达块边界时统一触发单次子图回放。

### 2.2 `torchtitan/distributed/offloading.py`：NUMA 亲和性绑定是零气泡卸载的生命线
为什么很多团队开启 CPU Activation Offload 后遭遇 `+20% ~ +80%` 的严重训练停顿？`torchtitan/distributed/offloading.py` (`blob 8a7d1cea`, L1–118) 揭示了根本原因：
- 必须调用 `maybe_apply_numa_binding( AffinityMode.NODE )` 将每个 GPU 进程严格绑定到其 PCIe Root Complex 直连的 CPU Socket 本地 NUMA 内存节点，并分配锁页内存（Pinned Memory `cudaHostAlloc`）。
- 若未开启锁页内存与 NUMA 绑定（`pageable_no_numa_gen4`），DMA 不仅退化为 OS 内核态二级拷贝（有效带宽跌至 `~18 GB/s`），还会跨 Socket 抢占 UPI/Infinity Fabric；而在 **NUMA 绑定 + Pinned PCIe Gen5 x16**（单向有效 DMA 带宽 `56.0 GB/s`）配合独立 `cudaStreamNonBlocking` 双缓冲流（前向第 $l$ 层计算重叠第 $l-1$ 层 `D2H`，反向第 $l$ 层计算重叠第 $l-1$ 层 `H2D` 预取）下，每层 `1,889.28 MiB` 的保留激活仅需 **`35.38 ms`** 单向 DMA，完全隐藏在 **`42.0 ms`** 的前向层计算与 **`87.1 ms`** 的反向层计算阴影之内！

---

## 3. 定量实验结果：5 种激活策略与 4 档主机互连对比

运行 `python experiments/selective_activation_offload/run.py --check`（80 层 70B GQA Transformer，$T=16,384$ tokens/rank），得到端到端显存与步耗时矩阵：

| 激活管理策略 (`policy`) | 单层保留激活 | 单层卸载激活 | 单层常驻 HBM | 80 层总 HBM 激活 | 显存削减率 | 重算耗时开销 | 暴露 PCIe 停顿 | 端到端步耗时 | 步耗时膨胀率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`no_recompute_save_all`** | `5,988.0 MiB` | `0.0 MiB` | `5,988.0 MiB` | `467.81 GiB` | `0.00%` | `0.00 ms` | `0.00 ms` | `10,080.63 ms` | `0.00%` |
| **`full_block_remat_checkpoint`** | `256.0 MiB` | `0.0 MiB` | `256.0 MiB` | `20.00 GiB` | `95.72%` | `3,360.21 ms` | `0.00 ms` | `13,440.84 ms` | **`+33.33%`** |
| **`megatron_core_attn_pointwise_sac`** | `2,624.0 MiB` | `0.0 MiB` | `2,624.0 MiB` | `205.00 GiB` | `56.18%` | `244.92 ms` | `0.00 ms` | `10,325.56 ms` | `+2.43%` |
| **`torchtitan_sac_recompute_w13_regions`** | `1,092.0 MiB` | `0.0 MiB` | `1,092.0 MiB` | `85.31 GiB` | `81.76%` | `1,710.94 ms` | `0.00 ms` | `11,791.58 ms` | `+16.97%` |
| **`sac_knapsack_plus_pcie_gen5_offload`** | `2,624.0 MiB` | **`1,889.28 MiB`** | **`734.72 MiB`** | **`61.09 GiB`** | **`86.94%`** | **`244.92 ms`** | **`10.61 ms`** | **`10,336.17 ms`** | **`+2.53%`** |

### 主机互连带宽与 NUMA 绑定敏感性（固定 `sac_knapsack_plus_pcie_gen5_offload`）
- **`pageable_no_numa_gen4` (`18.0 GB/s`)**：单层 DMA 耗时 `110.06 ms` > 层计算耗时，80 层累计暴露停顿 **`8,003.57 ms`**（步耗时劣化 **`+81.83%`**）。
- **`pinned_cross_numa_gen5` (`32.0 GB/s`)**：跨 NUMA 节点争抢导致单层 DMA `61.91 ms`，暴露停顿 **`1,879.81 ms`**（步耗时劣化 **`+21.08%`**）。
- **`pinned_numa_bound_gen5_x16` (`56.0 GB/s`)**：NUMA 亲和锁页内存使单层 DMA 降至 **`35.38 ms`**，80 层总暴露停顿仅 **`10.61 ms`**（步耗时仅增加 **`+2.53%`**，同时削减 **`86.94%`** 激活显存）。
- **`nvlink_c2c_grace_hopper_blackwell` (`420.0 GB/s` C2C)**：单层 DMA 仅 **`4.72 ms`**，暴露停顿 **`1.42 ms`**（总步耗时开销 **`+2.44%`**）。
