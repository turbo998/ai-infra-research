# Megatron-LM MFSDP v2 无填充（Unpadded）分片与 HyperConnection 残差契约、NCCL GIN Socket GDRCopy 强校验与 nccl4py 完成事件、TransformerEngine 前向融合编译、FA4 SM100 块稀疏空 Tile 零填充、Mooncake DMA-BUF 页对齐导出、LMCache SGLang 盐值隔离、Dynamo GPT-OSS 推理通道与 NIXL Mooncake postXfer 内存泄漏修复深度解析（Phase 121）

> **模块定位**：面向负责超大规模分布式训练（Megatron-LM、TransformerEngine、FlashAttention-4、NCCL）与分离式 KV 传输底座（Mooncake、LMCache、Dynamo、NIXL）的解决方案架构师（Solution Architect）。本文档基于 8 个上游仓库在 2026-09-17 至 2026-10-28 合入的 10 个核心提交，系统拆解异构行宽无填充分片、块稀疏注意力 `0.0 * Inf = NaN` 陷阱、DMA-BUF 页对齐契约与 RDMA 描述符内存安全。
> **证据等级**：`implementation`（源码固定 commit 级核验）+ `own-offline`（本仓库 Python 标准库可复现建模验证）。

---

## 1. 核心架构问题与 10 项上游证据总览

| 序号 | 仓库与 Commit | 日期 | 核心机制与架构变更 | 证据分类 |
| :--- | :--- | :--- | :--- | :--- |
| C1 | `NVIDIA/Megatron-LM` [`47715bb`](https://github.com/NVIDIA/Megatron-LM/commit/47715bb029f59aef4c45d516379911729e6b375d) | 2026-10-27 | `megatron_fsdp`：新增 `ShardLayout.UNPADDED`（`megatron_fsdp_padded_shards=False`），按行或 32 行 MXFP8 块边界紧致切分并走非均匀 AllGather/ReduceScatter | `implementation` |
| C2 | `NVIDIA/Megatron-LM` [`31c7230`](https://github.com/NVIDIA/Megatron-LM/commit/31c723006edcc345c89416959f24b94b3d34eded) | 2026-10-27 | `transformer`：将 `HyperConnectionTransformerLayer`（$H_{\text{pre}}, H_{\text{post}}, H_{\text{res}}$）与 `WideResidual` / `residual_recompute` 拆分为独立模块并收敛契约 | `implementation` |
| C3 | `NVIDIA/nccl` [`d8347eb`](https://github.com/NVIDIA/nccl/commit/d8347eb9b864fab0e79e58c4ca1b9eba5618a136) | 2026-09-17 | `gin_host_socket.cc`：移除 `ncclRmaSocketProxyInit` 中的惰性 `ncclGdrInit()`，使 `NCCL_GDRCOPY_ENABLE=0` 时直接返回 `ncclInternalError` | `implementation` |
| C4 | `NVIDIA/nccl` [`373bb82`](https://github.com/NVIDIA/nccl/commit/373bb82d1f08be1227e896dae02a1d3f410c97cd) | 2026-09-17 | `nccl4py`：暴露 `NCCLCollConfig.launch_completion_event`（`NcclEventSpec`，拒绝 `0` 或 `bool`） | `implementation` |
| C5 | `NVIDIA/TransformerEngine` [`45223d7`](https://github.com/NVIDIA/TransformerEngine/commit/45223d764a28fdaae242de3f0077c34f538d00de) | 2026-10-08 | `fused_rope` / `fuser.py`：对 `ForwardLinearBiasActivation`、`BasicLinear`（按 TP 大小特化）与 `Bias` 启用 `torch.compile(fullgraph=True)`，并允许冻结前缀层后的反向融合 | `implementation` |
| C6 | `Dao-AILab/flash-attention` [`86dcb5d`](https://github.com/Dao-AILab/flash-attention/commit/86dcb5d60c4e3bfd8117cb71e8a209afc80edcf9) | 2026-09-24 | `flash_fwd_sm100.py`：块稀疏注意力空 Q Tile 在 `correction_epilogue` 中传入 `zero_fill=True` 直接写 `0.0`，消除读取未初始化 TMEM 导致的 `0.0 * Inf = NaN` | `implementation` |
| C7 | `kvcache-ai/Mooncake` [`058c1e8`](https://github.com/kvcache-ai/Mooncake/commit/058c1e8a0e872caf4c9175c737bba7ed60525e55) | 2026-10-28 | `rdma_context.cpp`：`exportDmabuf` 对完整分配（`offset == 0 && length == allocSize`）按 OS 页大小向上对齐 `exportSize`，修复非页对齐 `cudaMalloc` 的 `CUDA_ERROR_INVALID_VALUE` | `implementation` |
| C8 | `LMCache/LMCache` [`577cca9`](https://github.com/LMCache/LMCache/commit/577cca98ca22bc75416b085523217b4bcff3cd81) | 2026-10-22 | `test_unified_sglang_mp_connector.py`：验证统一 SGLang MP 连接器在 `submit_lookup`、`_create_key` 与 `ipc_key_to_object_keys` 中的 `cache_salt` 多租户键隔离 | `implementation` |
| C9 | `ai-dynamo/dynamo` [`532616d`](https://github.com/ai-dynamo/dynamo/commit/532616d12ddea4d9ebce30fbcb67807cc02b63e0) | 2026-10-28 | `http/service.rs`：升级 `parsers 0.7.35` 时保持 `gpt_oss` 模型族 `reasoning_disabled = false` 以保留多通道解析，并引入 `tracing_interest_guard()` | `implementation` |
| C10 | `ai-dynamo/nixl` [`7bd1d13`](https://github.com/ai-dynamo/nixl/commit/7bd1d1386ecb5c24ce51f16c9070d25dd93492e4) | 2026-10-12 | `mooncake_backend.cpp`：`nixlMooncakeEngine::postXfer` 用 `std::vector<transfer_request_t>` 替换 `new[]`/`delete[]`，消除描述符长度不匹配早退路径上的内存泄漏 | `implementation` |

---

## 2. 深度技术剖析与公式推导

### 2.1 Megatron-FSDP v2 `ShardLayout.UNPADDED` 异构行宽无填充分片（`47715bb`）

在 `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/layout_builder.py` 中：
- **默认填充布局（`ShardLayout.PADDED`）的 LCM 膨胀问题**：`RowAtomic`（`block_size=1`）或 MXFP8 `BlockAtomic`（`block_size=32` 行）要求任何 Rank 边界都不能切断同一行（或 32 行块）。默认 `build_for_row_atomic` 计算组内所有参数张量对齐粒度的最小公倍数：
  $$\text{chunk\_size} = \operatorname{lcm}_{i}\!\left(\text{block\_size} \cdot \text{row\_size}_i\right)$$
  并将每个张量起始偏移对齐到 $\text{chunk\_size}$、总组大小对齐到 $\text{chunk\_size} \cdot \text{dp\_size}$。当同一参数组混合了不同列宽的矩阵——例如 `[(128, 6144), (128, 4096), (128, 14336)]`，在 `dp_size = 8, block_size = 32` 下，真实元素总数为 $3,145,728$，但 $\operatorname{lcm}(32\times 6144, 32\times 4096, 32\times 14336) = 2,752,512$，导致填充后总缓冲区膨胀至 **$22,020,096$ 个元素（浪费 $18,874,368$ 个填充元素，膨胀 7.0x）**！
- **无填充布局（`ShardLayout.UNPADDED`）**：`build_for_row_atomic_unpadding` 紧致拼接全部张量（零填充，总长精确等于 $3,145,728$），随后在累积行/块终点序列 $\text{row\_ends}$ 上用二分查找定位每个 Rank $k$ 的起始偏移：
  $$\text{rank\_offset}_k = \text{row\_ends}\!\left[\operatorname{bisect\_left}\!\left(\text{row\_ends},\ \left\lfloor \frac{k \cdot \text{size}}{\text{dp\_size}} \right\rfloor\right)\right]$$
  并切换至 `dist.all_gather_into_tensor_uneven` 与 `dist.reduce_scatter_tensor_uneven`（同时显式拒绝要求等长分片的 `nccl_ub=True` 对称内存模式）。

### 2.2 FA4 SM100 块稀疏空 Tile 零填充与 Mooncake DMA-BUF 页对齐导出（`86dcb5d`, `058c1e8`, `7bd1d13`）

1. **FlashAttention-4 (`sm_100`) 块稀疏空 Tile 的 `0.0 * Inf = NaN` 根因（`86dcb5d`）**：
   - 当某个 Query Tile 在块稀疏掩码下没有任何可见的 K/V 块（`total_block_cnt == 0`）时，主循环不会向 Tensor Memory (TMEM) 写入任何累加值，且行和统计量保持为 $0$，导致缩放因子 $\text{scale} = 1 / 0 = +\infty$；
   - 先前 `correction_epilogue` 仍从 TMEM 读出未初始化残值并乘以 `scale`：若未初始化 TMEM 恰好为 `0.0`，IEEE-754 浮点乘法会计算出 **`0.0 * Inf = NaN`**！修复后当 `zero_fill=True` 时跳过 TMEM 加载与重缩放，直接执行 `tOrO_frg.fill(0.0)` 并将 LSE 写为 $-\infty$。
2. **Mooncake DMA-BUF 完整分配页对齐导出（`058c1e8`）与 NIXL RAII 修复（`7bd1d13`）**：
   - `cudaMalloc(&ptr, (4ULL << 20) + 12345)`（`4,206,649 Bytes`）在 CUDA 驱动内部按页粒度分配，但 `cuMemGetAddressRange(&allocBase, &allocSize, ...)` 返回的是未向上取整的请求字节数 `4,206,649`。直接将该 `allocSize` 传给 `cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_handle_TYPE_DMA_BUF_FD, ...)` 会因非 `4096` 页倍数被拒绝（`CUDA_ERROR_INVALID_VALUE`）。Mooncake 在 `offset == 0 && length == allocSize` 时将导出长度向上对齐到 OS 页边界：
     $$\text{exportSize} = (\text{allocSize} + \text{page} - 1) \ \& \ \sim(\text{page} - 1) = 4,210,688\text{ Bytes}$$
   - NIXL 的 `nixlMooncakeEngine::postXfer` 将裸数组 `new transfer_request_t[request_count]` 改为 `std::vector<transfer_request_t>`，消除了 `local[index].len != remote[index].len` 提前返回 `NIXL_ERR_INVALID_PARAM` 时的内存泄漏（2 个描述符即泄漏 `80 Bytes`）。

---

## 3. 本仓库离线建模验证（`own-offline`）

对应实验目录：`experiments/megatron_mfsdp_unpadded_hyperconnection_nccl_gin_te_compile_fa4_mooncake_lmcache_dynamo_nixl_phase121/`

- **运行命令**：`python experiments/megatron_mfsdp_unpadded_hyperconnection_nccl_gin_te_compile_fa4_mooncake_lmcache_dynamo_nixl_phase121/run.py --check`
- **关键输出校验**：
  - `megatron_mfsdp_and_hyperconnection.unpadded_saved_numel = 18874368`（从 `22,020,096` 降至精确 `3,145,728`）；
  - `nccl_te_and_fa4.fa4_empty_tile_buggy_is_nan = true`，`fa4_empty_tile_fixed_val = 0.0`；
  - `mooncake_lmcache_dynamo_nixl.mooncake_aligned_export_bytes = 4210688`（`4096` 整除），`nixl_fixed_leaked_bytes_per_mismatch = 0`。

---

## 4. 架构师决策指南与可证伪问题

1. **架构师计算练习**：对于 64 KiB 大页系统（`page = 65,536 B`），若 `cudaMalloc` 申请 `allocSize = 10,000,000 B`，求 Mooncake `exportDmabuf` 向上页对齐后的 `exportSize`。
   - *解答*：$\lceil 10,000,000 / 65,536 \rceil \times 65,536 = 153 \times 65,536 = 10,027,008\text{ Bytes}$。
2. **后续可证伪问题**：在 Megatron-FSDP v2 中启用 `ShardLayout.UNPADDED` 后，`all_gather_into_tensor_uneven`（基于 `all_gather_v` / 逐 Rank 分片拼接）在 NVLink 拓扑上的带宽效率相比基于 `nccl_ub` 对称内存的等长 `ShardLayout.PADDED` 会有多少差异？
