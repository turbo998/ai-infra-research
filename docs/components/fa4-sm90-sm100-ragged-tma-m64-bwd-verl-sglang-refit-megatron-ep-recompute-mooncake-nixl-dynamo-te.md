# FlashAttention-4 SM90/SM100 Ragged TMA 边界截断与 `M=64` 2-CTA 反向布局、verl SGLang MXFP8/MXFP4/NVFP4 在线权重重装、Megatron EP A2A 分段重计算及 Mooncake / NIXL / Dynamo / TE 协同优化

> **English Version**: [fa4-sm90-sm100-ragged-tma-m64-bwd-verl-sglang-refit-megatron-ep-recompute-mooncake-nixl-dynamo-te.en.md](fa4-sm90-sm100-ragged-tma-m64-bwd-verl-sglang-refit-megatron-ep-recompute-mooncake-nixl-dynamo-te.en.md)  
> **Evidence ID**: `ev-fa4-sm90-sm100-ragged-tma-m64-bwd-verl-sglang-refit-megatron-ep-recompute-mooncake-nixl-dynamo-te-2026`  
> **Experiment**: [`experiments/fa4_sm90_sm100_ragged_tma_m64_bwd_verl_sglang_refit_megatron_ep_recompute_mooncake_nixl_dynamo_te/run.py`](../../experiments/fa4_sm90_sm100_ragged_tma_m64_bwd_verl_sglang_refit_megatron_ep_recompute_mooncake_nixl_dynamo_te/run.py)

---

## 1. 一手源码审计范围与核心问题

本文基于 `Dao-AILab/flash-attention`、`volcengine/verl`、`NVIDIA/Megatron-LM`、`kvcache-ai/Mooncake`、`ai-dynamo/nixl`、`ai-dynamo/dynamo` 与 `NVIDIA/TransformerEngine` 近期合入的 10 项核心提交（`#3012`, `#3013`, `#2965`, `#8306`, `#6429`, `#4631`, `#4854`, `#2478`, `#16106`, `#3505`），深度剖析训练算子数值安全、RL 低精度在线 Refit、MoE 1F1B 激活重计算与分布式 KV 存储/控制面优化：

1. **FlashAttention-4 SM90/SM100 Ragged K/V TMA 边界零填充与 `M=64` 2-CTA TMEM 反向布局 (`#3012` `2bdeddd15a`, `#3013` `767386d814`, `#2965` `2b00d76f22`)**：根治 `seqused_k` / `cu_seqlens_k` 尾部未初始化 `NaN` 通过 `0 * NaN = NaN` 污染 `dQ` 的静默缺陷，并通过 `M=64` TMEM 半行布局将 `hdim=256` 反向的全部 5 个累加器压缩进 `448 <= 512` 列 TMEM。
2. **verl SGLang MXFP8 / MXFP4 / NVFP4 在线权重 Refit 与 QAT 对齐 (`#8306`, `5265d561a3`)**：构建分桶暂存（bucket staging）+ 末桶原地内核布局重建（保留 CUDA Graph 显存指针）的统一低精度 RL Rollout 权重同步管线。
3. **Megatron-Core 按层分段（Per-Layer-Segment）重计算 EP A2A Overlap 激活 (`#6429`, `da2c0ec614`)**：在每个重计算分段结束后立即调用 `reset_for_recompute()` 释放引用并保留底层 Storage，同时快照 `mtp_input_mask` 防止重放时二次移位。
4. **Mooncake HCA/NUMA 亲和性 (`#4631`, `868a6d05ab`) 与 `CursorSlice` 无锁分片驱逐 (`#4854`, `c81f4ad486`)**：将 DFS 文件删除、软固定过期与在途检查移出 `kNumShards = 1024` 元数据写锁。
5. **NIXL Repost/Release 遥测精确计数 (`#2478`, `70badd6aad`)、Dynamo `--freeze-gc-after-init` (`#16106`, `45cc9e8a1a`) 与 TE `torch.compile` FP8 权重缓存 (`#3505`, `9ea05cf2e8`)**。

---

## 2. 关键实现路径与架构推导

### 2.1 FlashAttention-4 Ragged K/V TMA 边界截断与 `M=64` 2-CTA TMEM 布局 (`#3012`, `#3013`, `#2965`)

- **`0 * NaN = NaN` 尾部梯度污染机理 (`#3012`, `#3013`)**：FA4 反向算子是 KV-Stationary 的，通过 TMA 按整块 `tile_n` 加载 K 与 V。当使用 Packed Varlen (`cu_seqlens_k`) 或 Dense `seqused_k`（如预分配 KV Cache 池）时，`[seqlen_k, tile_n)` 区间内的尾部行可能残留 `NaN`。尽管掩码使该区域的 `P` 与 `dS` 精确为 `0`，但在矩阵乘 `dP = dO @ V^T` 与 `dQ = dS @ K` 中，IEEE-754 算术 `0.0 * NaN = NaN` 会导致跨尾部的最后一个 K Tile 将整行 `dQ`（以及前向 `O = P @ V`）彻底污染为 `NaN`（而 `dK/dV` 仍为有限值）。`#3012` 与 `#3013` 通过 `create_ragged_tensor_for_tma_load` 将 TMA 描述符的逻辑边界严格限制在 `(offset_k, seqlen_k)`，利用硬件 TMA 越界自动零填充（OOB zero-fill）在不增加任何指令或 Barrier 开销的前提下彻底消除 `NaN` 污染。
- **SM100 `M=64` 2-CTA TMEM 累加器布局 (`#2965`)**：在 `hdim = 256` 下，若沿用常规 `tile_n = 128` 布局，`dK + dV + S + dP + dQ` 所需的 TMEM 列数高达 $256 + 256 + 128 + 128 + 256 = 1024$ 列，远超 SM100 单 CTA `512` 列的硬件上限。`#2965` 在 2-CTA 模式下采用 `tile_n = 64`（2-CTA 联合 MMA `M = 128`），使每个 TMEM Lane 持有半行（Lanes `0..63` 持有列 `[0, N/2)`，Lanes `64..127` 持有列 `[N/2, N)`），从而使每个累加器仅占 $N / 2$ 列，并将 `P^T` 与 `dS^T` 移至 SMEM 作为 `dV/dK` MMA 的 A 操作数。此时总 TMEM 列数压缩为：
  $$\text{Cols}_{\text{TMEM}} = \underbrace{128}_{dK} + \underbrace{128}_{dV} + \underbrace{32}_{S} + \underbrace{32}_{dP} + \underbrace{128}_{dQ} = 448 \le 512$$
  同时在后处理中通过统一的 `_dQaccum_tile_layout` 直接从 HBM 以每线程 `16B` 向量化加载进寄存器，裁撤了 `cp.async` SMEM 中转缓冲区。

### 2.2 verl SGLang MXFP8 / MXFP4 / NVFP4 在线权重 Refit (`#8306`)

在 `verl/utils/sglang/{sglang_quant_utils.py,sglang_fp4_utils.py,sglang_fp8_utils.py}` 中：
- 针对 Megatron 训练器向 SGLang Rollout 在线同步低精度权重的场景，设计了分桶暂存机制：各同步分桶（sync buckets）先暂存规范格式的 MXFP8 缩放因子与 MXFP4/NVFP4 专家张量，待最后一个分桶到达后，一次性原地重建 DeepGEMM / FlashInfer CUTLASS 后端所需的 Swizzled/Packed 内核布局，**不替换已被 CUDA Graph 捕获的底层显存地址**（并显式拒绝不完整序列的 `delta_flush`）。
- **Blackwell 实测 KL 散度（Qwen3.5-35B-A3B, Megatron EP=8 $\rightarrow$ SGLang TP=2）**：启用匹配 QAT 后，`BF16 Linear + MXFP4 MoE` 的训练-推演 `k3_kl` 从 `0.00571` 降至 `0.00200`（降低 **2.86x**），`BF16 Linear + NVFP4 MoE` 的 `k3_kl` 从 `0.00946` 降至 `0.00653`（降低 **1.45x**）。

### 2.3 Megatron-Core EP A2A Overlap 分段激活重计算 (`#6429`)

在 `megatron/core/models/common/model_chunk_schedule_plan.py` 与 `utils.py` 中：
- 先前开启 `overlap_moe_expert_parallel_comm` 全重计算时，各层的边界与 A2A 重叠激活在整个 Model Chunk 前向期间一直被引用锚定，无法及时释放。
- `#6429` 改为按层分段（per-layer segment）管理生命周期：分段前向一结束即调用 `reset_for_recompute()` 清空 `self.inputs / self.output` 引用（但保留底层 Tensor Storage 不做 `resize_(0)`，避免破坏反向重放重建的 Autograd 图），并对 `ModelChunkState`（含 `input_ids`、`position_ids` 与 `mtp_input_mask`）做分段前快照，防止 MTP 在反向重放时对已经移位过的 `mtp_input_mask` 二次调用 `roll` 导致掩码错位。在每流水线阶段 8 层 MoE 下，块内峰值激活显存降低 **4.27x**（`8.0 -> 1.875` 层等效单元）。

### 2.4 Mooncake HCA/NUMA 亲和与无锁分片驱逐 (`#4631`, `#4854`) 及 NIXL / Dynamo / TE 优化

- **Mooncake (`#4631`, `#4854`)**：`MC_STORE_HCA_AFFINITY` 在分配单副本 RDMA 内存段时优先匹配接收端本地 HCA/NUMA 节点；`#4854` 引入 `CursorSlice` 允许多个线程并发无冲突遍历同一租户对象，并通过 `CommitEvictionLogical` + `DeferDeletion` 将慢速文件系统 `DeleteFile` 移出 `kNumShards = 1024` 元数据分片写锁，使单次驱逐写锁持有时间从 `640 us` 降至 `50 us`（降低 **12.8x**）。
- **NIXL (`#2478`)、Dynamo (`#16106`) 与 TransformerEngine (`#3505`)**：NIXL 在 `postXferReq` 重投检查与 `releaseXferReq` 中补齐首次完成计数，且拒绝在途重投时不重置计时器；Dynamo 为替换 vLLM API Server 的 Worker 进程引入 `--freeze-gc-after-init`（`DYN_VLLM_FREEZE_GC_AFTER_INIT`），在模型注册前调用 `freeze_gc_heap()` 冻结约 **100 万个静态对象**，彻底消除请求循环中的 Gen-2 GC 全堆扫描毛刺；TransformerEngine 支持 `torch.compile` 下的 `is_first_microbatch` FP8 权重缓存（首微批创建新 FP8 Workspace 避免函数式算子原地修改违规，后续微批直接复用，4 层 4096 宽 Linear 编译态峰值显存稳定在 `539 MiB`）。

---

## 3. 核心指标汇总表

| 框架与特性 | 源码提交 | 核心硬件/架构参数 | 优化前基线 | 优化后结果 | 改善幅度 |
|---|---|---|---|---|---|
| FA4 SM90/SM100 Ragged K/V TMA | `#3012` / `#3013` | `seqlen_k` OOB zero-fill | `0 * NaN = NaN` 污染 `dQ` | `0.0%` `NaN` 梯度污染 | **100% 根治** |
| FA4 SM100 `M=64` 2-CTA Backward | `#2965` (`2b00d76f22`) | `tile_n=64`, `hdim=256` TMEM 列数 | $1024 > 512$ (溢出) | $448 \le 512$ 列 | **2.29x 列压缩** |
| verl SGLang MXFP4 在线 Refit + QAT | `#8306` (`5265d561a3`) | Qwen3.5-35B-A3B `k3_kl` | $0.00571$ | $0.00200$ | **2.86x KL 降低** |
| Megatron EP A2A 分段激活重计算 | `#6429` (`da2c0ec614`) | 8-Layer MoE Chunk 激活占用 | $8.0\ \text{units}$ | $1.875\ \text{units}$ | **4.27x 显存降低** |
| Mooncake DFS 分片写锁剥离 | `#4854` (`c81f4ad486`) | `kNumShards=1024` 写锁耗时 | $640.0\ \mu\text{s}$ | $50.0\ \mu\text{s}$ | **12.8x 锁时间缩短** |

---

## 4. 架构师设计练习与可证伪问题

1. **SM100 TMEM 列预算验算**：当 `hdim = 256`、`tile_n = 64` 且启用 2-CTA `M=64` 布局时，验证为何将 `P^T` (`64 x 128`) 与 `dS^T` (`64 x 128`) 放入 SMEM 是避免突破 `512` 列 TMEM 硬件上限的必要条件（若留在 TMEM 将额外占用 $64 + 64 = 128$ 列，$448 + 128 = 576 > 512$）。
2. **后续可证伪问题**：在 Megatron-Core `#6429` 中，为何 `overlap_moe_expert_parallel_comm` 的全重计算（`recompute_granularity=full`）目前在 `combined_1f1b.py` 中显式断言禁止与 `Megatron FSDP` 同时开启？
