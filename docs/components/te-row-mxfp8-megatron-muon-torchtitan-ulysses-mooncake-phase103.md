# TransformerEngine 单向行缩放 MXFP8 主权重转换、Megatron-LM MFSDP v2 Muon 分布式正交化、TorchTitan Ulysses CP / 共享 RoPE 缓存及 Mooncake TENT RDMA 故障停机（Phase 103）

- **问题 ID**：`te-row-mxfp8-megatron-muon-torchtitan-ulysses-mooncake-phase103`（`priority: 112`，组件：`distributed-training`）
- **证据等级**：`own-offline`（基于上游源码补丁审计 + Python 3.12 标准库确定性算例 `experiments/te_row_mxfp8_megatron_muon_mooncake_phase103/run.py`）
- **配套英文版**：[`te-row-mxfp8-megatron-muon-torchtitan-ulysses-mooncake-phase103.en.md`](te-row-mxfp8-megatron-muon-torchtitan-ulysses-mooncake-phase103.en.md)

## 1. 一手上游信源（零重叠）

1. **TransformerEngine `3850986745` (#3488)**：`Support row-only MXFP8 distributed master-weight casts`
2. **TransformerEngine `0f8d92f5a7` (#3744)**：`[PyTorch] Set device during NCCL init in symmetric memory test`
3. **Megatron-LM `63f687dc83` (#8517)**：`Preserve GDP gate storage under chunkwise context parallelism`
4. **Megatron-LM `19895dafba` (#8512)**：`docs: Muon design for MFSDP v2`
5. **TorchTitan `b44774915d` (#5590)**：`[CP] Add Ulysses context parallelism for KDA and GDN`
6. **TorchTitan `e111f10d4d` (#5583)**：`[perf] Tied RoPE cache`
7. **LMCache `0395e677c4` (#5870)**：`[Token Dropping] [MVP 1/12] Define request-local token-drop state`
8. **Mooncake `1dd37f57ba` (#4383)**：`[Bugfix][TENT] Fail stop on terminal RDMA teardown errors`
9. **Mooncake `ddda75d351` (#4708)**：`[TENT][Bugfix] Refresh Local Rail Topology Before Slice Selection`
10. **verl `6b2b417a05` (#8315)**：`[tool] feat: rl-insight trace weight-sync stages and report trainer metrics by step`

---

## 2. 核心实现机制与定量分析

### 2.1 TransformerEngine 单向行缩放 (`Row-Only`) MXFP8 分布式主权重转换 (`3850986745`) 与对称显存 NCCL 绑定 (`0f8d92f5a7`)

在分布式混合精度训练中，当优化器后或前向仅需行方向（Row-wise）MXFP8 权重时，旧版 `mxfp8_scaling_compute_partial_amax_kernel` 与 `mxfp8_scaling_partial_cast_kernel` 仍无条件计算列方向（Column-wise）amax、读取列方向 `e8m0_t` 缩放因子并写出列方向 FP8 缓冲区。
- **编译期模板特化 (`template <..., bool kWithColumnwise>`)**：当传入空 `amax_colwise` / `output_colwise` `NVTETensor` 时，调度 `kWithColumnwise = false` 特化实例，完全跳过列方向 `fmaxf` 归约、`scales_colwise` 共享内存加载与 `output_colwise` 全局内存写入。
- **HBM 流量节省定量计算**：以 `test_partial_cast.py` 中的 $16384 \times 6144$（$100{,}663{,}296$ 元素）主权重分片为例，单方向 FP8 写出为 `96.0 MiB`、`e8m0_t` 缩放因子（每 32 元素 1 B）为 `3.0 MiB`、BF16 主权重读取为 `192.0 MiB`。启用 `Row-Only` 转换后，FP8 写出流量从 `192.0 MiB` 降至 `96.0 MiB`（**节省 50.0%**），总转换 HBM I/O 从 `390.0 MiB` 降至 `291.0 MiB`（**节省 25.38%**）。同时 `0f8d92f5a7` 在对称显存测试初始化 NCCL 时显式传入 `device_id=torch.device("cuda", rank)`。

### 2.2 Megatron-LM MFSDP v2 下的 Muon 分布式正交化架构 (`19895dafba`) 与 GDP 块级 CP (`63f687dc83`)

Megatron-LM MFSDP v2 将同一模块内相同 dtype 与 `requires_grad` 的参数打包进连续缓冲区并在 Rank 间切分。由于 Muon 优化器需要对每个完整逻辑矩阵执行 Newton–Schulz 正交化（直接对行分片或拼接缓冲区执行正交化会改变数学更新结果），`19895dafba` 对比并量化了两种核心放置策略（以 5 个参数 `P0=(2,6), P1=(4,4), P2=(4,4), P3=(1,2), P4=(1,6)` 共 **52 个元素**分布在 **5 个 Rank** 上为例）：
1. **`RowAtomic` / `BlockAtomic(32)` + `FsdpMuon` 点对点 Gather/Scatter**：
   - 保持每个 Rank 等长分片（每 Rank `12` 个元素，总缓冲区 `60` 个元素，含 `8` 个对齐/填充元素）以保证前向 All-Gather 与反向 Reduce-Scatter 完全负载均衡。
   - Reduce-Scatter 之后，`P0, P3, P4` 共 **3 个完整矩阵**已经完整落在单一 Rank 上，**优化器步网络通信量为 0**；仅有跨 Rank 边界的 `P1`（横跨 Rank 1 与 Rank 2）和 `P2`（横跨 Rank 2 与 Rank 3）需要重构。
   - `FsdpMuon` 通过 Owner Planner 为每个跨界矩阵选定一个已持有分片的计算 Owner（如 `P1 -> Rank 1`、`P2 -> Rank 3`），仅通过 P2P Gather/Scatter 搬移远端残余分片（每个跨界矩阵 Gather `4` 个元素 + Scatter `4` 个元素，合计仅搬移 **16 个元素**），并在单一 Owner 上执行一次 Newton–Schulz；相比对跨界矩阵做全组 All-Gather 并重复计算正交化（需搬移 $(16+16)\times 4 = 128$ 个元素），**优化器步通信量削减 8.0x**。
2. **`TensorAtomic` 整矩阵归属放置**：
   - 将每个完整矩阵的权重、梯度和动量持久分配给单一 Owner（如各 Rank 段长为 `[12, 16, 0, 16, 8]`，合计 `52` 个元素，`0` 填充），使 Muon 优化器步实现 **0 跨卡通信与 0 临时缓冲区**；代价是前向/反向通信段长不均（最大/均值负载不均衡比为 $16 / 10.4 = \mathbf{1.5385\times}$）。
3. **GDP 块级上下文并行 (`63f687dc83`)**：在 `GatedDeltaProductMixer._postprocess` 中，当 `rearrange(z, "b l d -> l b d")` 用于 CP1 或 Chunkwise CP 且 `batch_size == 1` 时，移除多余的 `.contiguous()` 拷贝以保留投影层原始存储步幅。

### 2.3 TorchTitan Ulysses CP (`b44774915d`)、共享 RoPE 缓存 (`e111f10d4d`)、LMCache Token-Drop (`0395e677c4`)、Mooncake TENT (`1dd37f57ba`, `ddda75d351`) 与 verl (`6b2b417a05`)

- **TorchTitan KDA/GDN Ulysses 上下文并行 (`b44774915d`)**：将 KDA 与 GDN 的 CP 切分从序列维通过打包 All-to-All 转换到头维（`head_shard_degree = tp * cp`，如 `TP=2, CP=4` 下 `head_shard_degree = 8`，64 头模型每 Rank 处理 `8` 个头的完整序列），并在本地切分逐头 `conv1d` 与门控参数而不改变 Checkpoint 键名。
- **TorchTitan 跨层共享 RoPE 缓存 (`e111f10d4d`)**：在每个流水线阶段（`Decoder`）内对配置相同的 Transformer 层共享同一个 `RoPE` 缓冲区。对于 80 层、128K 上下文（每层 FP32 cos/sin 缓存 `128.0 MiB`）的模型，单阶段 RoPE 显存从 `10,240.0 MiB`（`10.0 GiB`）降至 `128.0 MiB`（**80.0x 显存削减，节省 10,112.0 MiB**）。
- **LMCache 请求级 Token-Drop 状态契约 (`0395e677c4`)**：定义 `LMCacheMPTokenDropRequestState`，将调度器侧请求级 Token 丢弃策略与 Worker 侧批次行号（未分配前为 `None`）解耦。
- **Mooncake TENT RDMA 析构 Fail-Stop 与本地 Rail 拓扑刷新 (`1dd37f57ba`, `ddda75d351`)**：`~RdmaContext()` 在 `disable()` 失败（仍有存活 Endpoint 资源）以及 `~RdmaEndPoint()` 在 `deconstruct()` 失败（底层 QP/MR 尚未销毁）时立即触发 `LOG(FATAL)` 停机，彻底阻断静默忽略错误后的网卡 DMA 悬挂写后释放（Use-After-Free）；`Workers::selectOptimalDevice` 增加 `source.topo != rail.local()` 检查，确保本地 NIC/GPU 拓扑更新后自动刷新 Rail。
- **verl `rl-insight` 权重同步阶段追踪与按步指标上报 (`6b2b417a05`)**：为异步 SGLang Rollout 的 `wake_up` 与权重同步阶段挂载 `RLInsightLogger.trace_state_decorator`，并在检测到 `metric_log(name, value, step=)` 时按 `global_step` 记录训练标量（旧版 `<=0.3.0` 自动回退至 `metric_gauge`）。

---

## 3. 架构师设计练习与可证伪问题

1. **练习计算**：在 Megatron-LM MFSDP v2 `RowAtomic` 下，若一个跨界矩阵形状为 $4096 \times 4096$（BF16，`32 MiB`），其中 `24 MiB` 位于 Rank 1、`8 MiB` 位于 Rank 2，使用 `FsdpMuon` P2P Gather/Scatter（选 Rank 1 为 Owner）相比在 8 卡 DP 组内对该矩阵做 All-Gather 可减少多少跨卡通信量？（答案：P2P 仅需 Gather `8 MiB` + Scatter `8 MiB` = **16 MiB**；8 卡 All-Gather 需传输 $32 \times 7 = 224\text{ MiB}$，减少 **14.0x / 208 MiB**）。
2. **可证伪假设**：若 Mooncake `~RdmaEndPoint()` 在 `ibv_dereg_mr` 返回 `EBUSY` 时仅打印警告并继续释放宿主/设备缓冲区，在途 RDMA Write 将直接覆写已归还分配器的内存页。
