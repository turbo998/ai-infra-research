# PyTorch TorchTitan FlexShard (`ComputeLayout` / `BlockShard` / `Owned`) 异构优化器重分片、DistMuon 零拷贝视图与 GB200 NVLink72 HybridEP 非阻塞容量因子机制

> **English Version**: [torchtitan-flexshard-distmuon-hybridep.en.md](torchtitan-flexshard-distmuon-hybridep.en.md) | **关联实验**: [`experiments/torchtitan_flexshard_hybridep_sim/README.md`](../../experiments/torchtitan_flexshard_hybridep_sim/README.md)

### 一句话结论

PyTorch TorchTitan 引入的 **FlexShard（`torchtitan/distributed/flex_shard`）** 将参数的**持久化 DTensor 存储布局（Storage Placement）**与**优化器计算期临时分片布局（`ComputeLayout`）**彻底解耦：通过支持周期性异构分块的 `BlockShard(dim=0, block_sizes=(512, 64))`（如 Kimi 共享 `wkv_a` 投影中交替排列的 512 行 KV 潜矩阵与 64 行 RoPE 键矩阵）、`Owned` 动态单 Rank 归属、以及 `compute_dtype=torch.bfloat16` 的分桶（`BucketConfig`）通信-计算流水重叠，使 `DistMuon` 在保持完整矩阵正交化边界与长宽比学习率缩放 $\sqrt{\max(1, R/C)}$ 的同时，将重分片通信字节减半并获得 **1.716x 优化器步加速**；而在 MoE 前向/反向通信侧，针对 GB200 NVLink72 设计的 **`HybridEP`（`torchtitan/distributed/deepep/hybridep.py`）** 通过 `_num_permuted_tokens_for_non_blocking` 预分配机制消除了阻塞式 `cudaStreamSynchronize` D2H 同步，在 `capacity_factor = 0.25` 下将单卡预分配缓冲从最坏情况的 `16,132.5 MiB` 压缩至 `4,033.12 MiB`（**节省 75% 显存**）且在 P99 路由偏斜下保持零丢包（`_over_budget = False`）。

### 组件责任与上游主线位置

| 模块与文件 | 上游仓库与 Commit / Blob SHA | 核心工程职责 |
| :--- | :--- | :--- |
| `torchtitan/distributed/flex_shard/README.md` | `pytorch/torchtitan` commit `3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0` / blob `fe82e542c0acb07894312b1f64899685d1bcd4f4` | 定义 FlexShard 架构契约：`ComputeLayout`、`BlockShard`、`Owned`、`BucketConfig`、`shard_order_by_tensor_dim` 与 Kimi `wkv_a` `(512, 64)` 集成规范。 |
| `torchtitan/distributed/flex_shard/optimizer_reshard.py` | `pytorch/torchtitan` commit `3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0` / blob `cc4d63592b8bbc54e441191fff77fc86d6ad729c` | 实现不可变配置类型 `BlockShard`（含 `num_blocks` 与 `block_start` 周期边界计算）、`Owned`、`ComputeLayout` 与 `_FrozenConfigMapping`。 |
| `torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py` | `pytorch/torchtitan` commit `3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0` / blob `477bdaa818ad2aa435618158b718ba2b94d0734b` | 实现 `_bind_bucket_configs`、`_resolve_buckets`、`_TensorRegion` 矩形张量路由与跨 Rank 打包集合通信计划生成。 |
| `torchtitan/distributed/deepep/hybridep.py` | `pytorch/torchtitan` commit `3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0` / blob `b82c43862a59203545c08e0c68aa0b7a4ade59cf` | 实现 GB200 NVLink72 TMA 优化的 `hybridep::dispatch` / `combine` 自定义算子、`DispatchHandle` 不透明图类型、`_num_permuted_tokens_for_non_blocking` 与粘性 `_over_budget` 监控。 |

### 核心实现路径、数据布局与同步契约

1. **`BlockShard` 异构周期性矩阵分块契约（`optimizer_reshard.py:37-85`）**：
   - Muon 优化器的 Newton-Schulz 正交化必须在**每个独立的逻辑二维权重矩阵 $[R, C]$** 上完整执行，严禁在矩阵行中间切断，且每个矩阵需按各自长宽比 $\sqrt{\max(1, R/C)}$ 缩放更新步长。
   - 然而在大模型融合投影层（如 Kimi 的共享 `wkv_a` 投影）中，单个二维参数在维度 0 上交替拼接了不同行数的独立矩阵（如 512 行的 KV 潜向量矩阵与 64 行的 RoPE 键矩阵）。
   - `BlockShard(dim=0, block_sizes=(512, 64))` 以 `period = sum(block_sizes) = 576` 为周期，校验 `dim_size % period == 0`，计算总块数 `num_blocks = (dim_size // period) * len(block_sizes)`，并通过 `block_start(block_index)` 按块数（而非均分行号）将完整矩阵分配给不同计算 Rank。`DistMuon` 随后在各 Rank 本地基于连续存储直接构造零拷贝（Zero-Copy）跨步视图 `[M, R, C]`。
2. **FP32 存储到 BF16 计算的分桶重分片与流水重叠（`README.md:14-35`, `_optimizer_reshard_schedule.py:32-95`）**：
   - 主权重通常以 FP32 存储于 DTensor（如 FSDP `Shard(0)`），但 `DistMuon` 的 Newton-Schulz 迭代仅接受 `torch.bfloat16`。
   - `ComputeLayout(compute_dtype=torch.bfloat16)` 在发送重分片输入前先转换为 BF16，使存储到计算（Storage-to-Compute）与计算回写（Compute-to-Storage）的集合通信字节直接减少 **50%**；同时 `BucketConfig` 按 FQN 通配符将同传输网格（Transport Mesh）的参数打包进统一通信桶，实现桶 $i$ 的 Newton-Schulz 计算与桶 $i+1$ 的重分片通信重叠。
3. **GB200 NVLink72 `HybridEP` 阻塞 vs 非阻塞容量因子与 `torch.compile` 契约（`hybridep.py:10-120`）**：
   - 在 MoE All-to-All Token Dispatch 中，融合 Permute 内核需要知道输出缓冲区行数 `num_permuted_tokens`：
     - **阻塞模式（`non_blocking_capacity_factor=None`）**：先执行计数通信，调用 `cudaStreamSynchronize()` 将精确 Token 数同步回 Host CPU，再分配输出张量，这会打断 GPU 流水线并阻碍完整 `torch.compile` 图捕获。
     - **非阻塞模式（`non_blocking_capacity_factor in (0, 1]`）**：完全消除 D2H 同步，按公式预分配输出缓冲区并向上对齐至 `pad_multiple`（MXFP8 grouped GEMM 要求 128 对齐），同时用 `register_opaque_type(DispatchHandle, typ="reference")` 让底层 `deep_ep` 句柄安全穿过 `torch.compile` FX 图。若实际路由偏移超出预分配容量，内核丢弃超额 Token 并置位粘性标志 `_over_budget`。

### 关键公式、单位与工程假设

1. **`BlockShard` 块计数与周期边界定位公式**：
   $$P = \sum_{i=0}^{k-1} b_i, \quad N_{\text{blocks}}(D) = \frac{D}{P} \cdot k \quad (D \bmod P = 0)$$
   $$\text{block\_start}(m) = \left\lfloor \frac{m}{k} \right\rfloor \cdot P + \sum_{i=0}^{(m \bmod k)-1} b_i$$
2. **HybridEP 非阻塞置换缓冲区大小公式（`hybridep.py:73-99`）**：
   $$N_{\text{raw}} = \left\lfloor T \cdot \text{EP}_{\text{size}} \cdot \min(E_{\text{local}}, K_{\text{top}}) \cdot \alpha_{\text{cap}} \right\rfloor$$
   $$N_{\text{aligned}} = \left\lceil \frac{N_{\text{raw}}}{M_{\text{pad}}} \right\rceil \cdot M_{\text{pad}}$$
   其中 $T$ 为单卡输入 Token 数，$\text{EP}_{\text{size}}$ 为专家并行度（GB200 NVLink72 典型为 72），$E_{\text{local}}$ 为单卡本地专家数，$K_{\text{top}}$ 为每 Token 激活专家数，$\alpha_{\text{cap}} \in (0, 1]$ 为容量因子，$M_{\text{pad}}$ 为分组 GEMM 对齐粒度（如 MXFP8 取 128）。

### 确定性离线仿真算例与量化对比

通过运行标准库确定性仿真器 [`experiments/torchtitan_flexshard_hybridep_sim/run.py`](../../experiments/torchtitan_flexshard_hybridep_sim/run.py)（结果固化于 [`experiments/torchtitan_flexshard_hybridep_sim/results.json`](../../experiments/torchtitan_flexshard_hybridep_sim/results.json)），得到以下核验数据：

| 评估维度 | 指标 / 场景 | 基线方案 | TorchTitan FlexShard / HybridEP | 量化收益 / 工程结论 |
| :--- | :--- | :--- | :--- | :--- |
| **Kimi `wkv_a` 异构矩阵分片 (`D=4608`)** | `(512, 64)` 重复 8 组（共 16 个矩阵） | 6-Rank 朴素均分行号在 `768, 1536, 3072, 3840` 处**切断 4 个矩阵** | `BlockShard(0, (512, 64))` **0 次矩阵边界破坏** | 严格保护每个 512/64 矩阵独立 Newton-Schulz 正交化 |
| **DistMuon 分桶重分片优化器步耗时** | 4 桶 × 8 参数 `[4096, 4096]`, 8 Ranks | FP32 串行重分片 + 计算：`8.7570 ms` | BF16 `compute_dtype` + 分桶重叠：`5.1033 ms` | 通信流量减少 **50%**，优化器步取得 **1.716x 加速** |
| **GB200 NVLink72 HybridEP 显存开销** | `T=4096, EP=72, E_local=4, top_k=8` | Worst-Case `capacity_factor=1.0`：`16,132.50 MiB` (`1,179,648` tokens) | `capacity_factor=0.25`：`4,033.12 MiB` (`294,912` tokens) | **节省 75.0% 缓冲显存**，消除 D2H 同步且 P99 偏斜 **0 丢包** |

### 对质量、SLO、拓扑、成本与运维的架构影响

1. **收敛质量保护**：若不使用 `BlockShard` 而让 FSDP 默认 `Shard(0)` 切分融合投影矩阵，Newton-Schulz 会对半截矩阵或混入 RoPE 键行的矩阵做正交化，导致训练 Loss 劣化甚至发散；`BlockShard` 从机制上保证了数学等价性。
2. **Checkpoint 恢复契约**：`ComputeLayout` 是优化器构造期配置，**不保存在 `state_dict` 中**（`README.md:67-69`）。运维恢复训练 Checkpoint 时，必须使用一致的 `compute_sharding_by_fqn` 重建 `DistMuon`。
3. **NVLink72 MoE 内存-吞吐权衡**：在 `EP=72` 的 GB200 机柜上，完全最坏情况（所有 72 张卡的所有 Token 同时命中同一张卡的全部 4 个本地专家）需要 $4096 \times 72 \times 4 = 1,179,648$ 行缓冲（`16.13 GiB`），而均衡路由期望值仅为 $4096 \times 8 = 32,768$ 行。将 `non_blocking_capacity_factor` 设为 `0.125 ~ 0.25` 可提供 $4.5\times \sim 9.0\times$ 的安全裕度（容纳 `147,456 ~ 294,912` 行），同时省下 `12 ~ 14 GiB` 显存并监控 `_over_budget` 指标。

### 架构师解释、计算与设计练习

1. **解释练习**：为什么 `torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py` 中的 `_bind_bucket_configs` 禁止同一个 `BucketConfig` 包含跨越不同 `storage_mesh_axis` 的重分片参数？
   - *参考解答*：每个 Bucket 在运行时被打包为单次底层集合通信（Packed All-to-All / AllGather），而单次集合通信只能绑定一个确定的 ProcessGroup / DeviceMesh 子轴；若一个桶内混入不同传输轴的参数，无法在单次打包通信中完成路由，因此必须按传输轴拆分 `BucketConfig` 模式。
2. **计算练习**：对于 `T = 2048, EP = 72, num_local_experts = 4, top_k = 8, capacity_factor = 0.2, pad_multiple = 128`，`_num_permuted_tokens_for_non_blocking` 返回多少？
   - *参考解答*：$N_{\text{raw}} = \lfloor 2048 \times 72 \times \min(4, 8) \times 0.2 \rfloor = \lfloor 117,964.8 \rfloor = 117,964$。向上对齐至 128 的倍数：$\lceil 117,964 / 128 \rceil \times 128 = 922 \times 128 = 118,016$。
3. **设计练习**：在同时启用 Expert Parallel (`ep`) 与 Expert Data Parallel (`edp_shard`) 的 MoE 模型中，如何使用 `shard_order_by_tensor_dim` 配置 `ComputeLayout`？
   - *参考解答*：设置 `shard_order_by_tensor_dim={0: ("ep", "edp_shard")}`，先保持最外层 `ep` 轴的专家归属 `Shard(0)`，再在本地专家域内按 `edp_shard` 轴重分片供 `DistMuon` 并行计算。

### 可证伪后续问题与核验边界

1. **核验边界**：本分析基于 `pytorch/torchtitan` commit `3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0` 源码与本仓库离线仿真器（`own-offline`），未在物理 GB200 NVLink72 机柜上执行实机 `deep_ep` 通信压测。
2. **可证伪后续问题**：当 MoE 路由未启用强制负载均衡（如无 Aux-Loss 的 DeepSeek-V3 动态偏置路由在训练初期出现极端专家塌缩）时，`non_blocking_capacity_factor = 0.25` 触发 `_over_budget` 的真实步数比例是多少，是否支持自动回退到阻塞式重传？
