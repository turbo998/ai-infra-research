# FlashAttention-4 SM100 `hdim=256` 2CTA 反向统一、veRL VeOmni BF16 ReduceScatter、Mooncake 批量状态 C-API 与 NIXL/Dynamo/TE 深度审计

> **English Edition**: [fa4-sm100-hd256-2cta-bwd-verl-veomni-bf16-rs-mooncake-batch-capi-nixl-dynamo-te.en.md](fa4-sm100-hd256-2cta-bwd-verl-veomni-bf16-rs-mooncake-batch-capi-nixl-dynamo-te.en.md) | **关联问题**: `q-fa4-sm100-hd256-2cta-bwd-verl-veomni-bf16-rs-mooncake-batch-capi-nixl-dynamo-te` (`priority: 91`, Phase 82) | **可复现实验**: `experiments/fa4_sm100_hd256_2cta_bwd_verl_veomni_bf16_rs_mooncake_batch_capi_nixl_dynamo_te/run.py`

---

## 1. 架构背景与核心工程矛盾

在 Blackwell (`SM100`) 大头维 (`hdim=256`) 注意力反向传播、大模型 RL/FSDP2 混合精度训练（veRL + TransformerEngine）以及分离式推理 KV 传输控制面（Mooncake / NIXL / Dynamo）中，存在四项跨层架构挑战：

1. **Blackwell SM100 `hdim=256` 2CTA 反向传播的 TMEM 512 列预算与跨 CTA `dS` 交换开销**：当 `hdim=256` 采用 `2CTA, tile_m=128, tile_n=64` 时，单 CTA 的 Tensor Memory (TMEM) 满额为 **512 列**。若为跨 CTA 的 `dS` 交换额外分配共享内存暂存缓冲或插入 `dQaccum_empty` 等待屏障，将挤占 `sdS_dK` 空间并拉长 MMA 关键路径。
2. **FSDP2 训练反向梯度 ReduceScatter 的通信带宽瓶颈与 `GroupedTensor` 自定义派发下的 Autograd/Stream 断裂**：72B+ 模型在 FP32 下执行 ReduceScatter 会消耗双倍节点内/节点间带宽；而 TransformerEngine 的 `GroupedTensor` 在 `__torch_dispatch__` 内若直接构造带 `requires_grad=True` 的视图包装器，会阻止 PyTorch Autograd 引擎在派发后挂载 `ViewBackward0` 及向底层存储张量传播 `record_stream`。
3. **Mooncake C-API 逐任务轮询的 FFI 开销、NIXL 重投递 (`NIXL_ERR_REPOST_ACTIVE`) 遥测计时重置与 Dynamo 混合 Mamba/Attention 路由事件序号伪断层**：256 个并发传输任务若通过 `getTransferStatus` 逐个跨越 C FFI 轮询会产生 $O(N)$ 调用开销；NIXL 在 repost 活跃传输时若重置起始计时器会严重低估真实 RDMA 传输延迟；Dynamo KV 路由器若将 Mamba/SSM 的 `CacheOwner` (KVCR) 事件混入块几何校验，会引发 `next_event_id` 序列号不连续告警。

---

## 2. 第一手源码实现路径审计

### 2.1 FlashAttention-4 SM100 `hdim=256` 2CTA 反向统一与零暂存跨 CTA `dS` 读取 (`965c9212e6be`, `#2966`)

审计 `Dao-AILab/flash-attention` 的 `flash_attn/cute/flash_bwd_sm100.py`、`bwd_Schedule_builder.py` 与 `bwd_accum_tile.py`：

- **统一 `hdim=256` 2CTA 反向架构 (`FlashAttentionBackwardSm100`)**：将 `hdim=256` 反向统一至 `FlashAttentionBackwardSm100`（`cluster_n=2, tile_m=128, tile_n=64`，采用 `serial_2cta` 调度）。
- **512 列 TMEM 精确切分与 `231,424 / 232,448 B` SMEM 布局**：
  $$\text{TMEM}_{512} = \underbrace{128}_{dV} + \underbrace{128}_{dK} + \underbrace{64}_{S/P} + \underbrace{64}_{dP/dS} + \underbrace{128}_{dQ} = 512\text{ columns}$$
  非确定性模式共享内存为 **`231,424 B` (`226 KiB`)**，确定性 (`deterministic`) 模式为 **`232,448 B` (`227 KiB`)**。
- **直接从对端 `sdS_dK` 读取 `8 KB` 连续 `dS` 半块（零交换暂存、零 `dQaccum_empty` 等待）**：在 `128B` swizzle 下，`64 × 64` BF16/FP16 的对端 `dS` 半块（`64 × 64 × 2B = 8,192 B = 8 KB`）在 `sdS_dK` 中构成连续 K-block 切片。本端 CTA 直接从 `sdS_dK` 读取该 `8 KB` 切片馈入 2CTA `dQ` MMA，省去专用 exchange staging buffer，并消除了向 `sdS_dK` 写出 `dS` 前对 `dQaccum_empty` 的等待。
- **形式化 `AccumTileOrder` 契约 (`bwd_accum_tile.py`)**：定义 `ROW_MAJOR`、`COL_MAJOR`、`TMEM_FRAG_1CTA` 与 **`TMEM_FRAG_2CTA_M64`** 四种累加片顺序，解耦 TMEM 寄存器片段布局与外层循环。

### 2.2 veRL VeOmni `low_precision_reduce_scatter_comm` (`fb55b9288560`, `#8330`) 与 TE `GroupedTensor` Autograd 视图修复 (`2199b7166417`, `#3701`)

- **veRL VeOmni FSDP2 BF16 ReduceScatter (`verl` `#8330`)**：
  - 审计 `verl/models/veomni/monkey_patch.py` 与 `verl/workers/engine/veomni/transformer_impl.py`：暴露 `low_precision_reduce_scatter_comm: bool` 配置并透传至 VeOmni FSDP2 通信钩子。启用后，反向梯度在节点内以 **BF16 线速格式 (`2 B/elem`)** 执行 ReduceScatter，同时在接收端保持 **FP32 累加与输出**。
  - 对于 72B 参数模型 (`world_size = 8`)，单步每卡 ReduceScatter 流量从 $\frac{7}{8} \times 72 \times 10^9 \times 4\text{ B} = 234.6933\text{ GiB}$ 降至 **$117.3466\text{ GiB}$**（通信量精确减半 **`2.0x`**）。
- **TransformerEngine `GroupedTensor` 视图 Autograd 与 `record_stream` 传播 (`TransformerEngine` `#3701`)**：
  - 审计 `transformer_engine/pytorch/tensor/grouped_tensor.py`：在 `__torch_dispatch__` 拦截 `aten.view` / `aten._unsafe_view` / `aten.slice` 等视图算子时，内部构造的 `GroupedTensor` 包装器强制传入 `requires_grad=False`，交由 PyTorch Autograd 引擎在 `__torch_dispatch__` 返回后自动包裹可微视图节点 (`ViewBackward0`)；同时重写 `record_stream(stream)`，将 CUDA 流生命周期同步传播至全部 **4 类底层存储张量** (`row_splits`, `data`, `scale_inv`, `amax`)，防止异步通信/计算流提前回收显存。

### 2.3 Mooncake $O(1)$ 批量状态 C-API (`44094f2e01a0`, `#4910`)、NIXL 16B `treq` 标签与单次遥测 (`f3673f6be3f4` `#2487`, `70badd6aad36` `#2478`) 及 Dynamo KVCR 过滤 (`8bf042187a8f` `#15535`)

- **Mooncake `getBatchTransferStatus` C-API (`Mooncake` `#4910`)**：在 `mooncake-transfer-engine/include/transfer_engine_c.h` 与 `src/transfer_engine_c.cpp` 中导出 `getBatchTransferStatus(transfer_engine_t engine, batch_id_t batch_id, struct transfer_status_t *status)`，使 C/Rust/Go 绑定层只需 **1 次 C FFI 调用 ($O(1)$)** 即可汇总整个批次（如 256 个子任务）的完成状态与总传输字节数（替代此前 256 次 `getTransferStatus` 调用，FFI 开销降低 **`256x`**）。
- **NIXL 16-Byte Little-Endian `treq` 通知标签 (`nixl` `#2487`) 与单次完成遥测 (`nixl` `#2478`)**：
  - 在 `src/core/nixl_agent.cpp` 与 UCX 后端通知载荷中，在 `tctx` 字段后追加 **16 字节小端序 `treq` 标签**（8 字节 `request_id` + 8 字节 `execution` 计数），支持跨远端通知精确关联每次传输请求执行实例。
  - 在 `postXferReq`、`getXferStatus`、`repost` 与 `releaseXferReq` 四条路径中统一通过完成标志位保证每次传输执行仅发射 **1 次** 遥测事件，且当 repost 返回 `NIXL_ERR_REPOST_ACTIVE` 时**不重置起始计时器**，消除在途传输耗时被截断的偏差。
- **Dynamo KV Router 过滤 `CacheOwner` (KVCR) 事件 (`dynamo` `#15535`)**：在 `lib/llm/src/kv_router/` 中，在执行 KV 块几何校验 (`block_size` / token 计数检查) 之前先过滤掉 Mamba/SSM 混合模型的 `CacheOwner` (KVCR) 状态事件，使注意力 KV 块事件保持严格连续递增的 `next_event_id` 序列（如 `[101, 102, 103]`），消除伪断层告警。

---

## 3. 可复现实验与验证 (`own-offline`)

运行 `python experiments/fa4_sm100_hd256_2cta_bwd_verl_veomni_bf16_rs_mooncake_batch_capi_nixl_dynamo_te/run.py --check` 的验证结果如下：

| 组件 / 协议契约 | 优化前 / 旧实现 | 优化后 / 新实现 | 验证结果 |
|---|---|---|---|
| FlashAttention-4 SM100 `hdim=256` 2CTA Backward (`#2966`) | 独立路径 / 需交换暂存与 `dQaccum_empty` 等待 | 统一 `512` 列 TMEM + 直接读对端 `8 KB` `sdS_dK` | 交换暂存 **`0 B`**，SMEM `231,424 / 232,448 B` |
| veRL VeOmni 72B FSDP2 (`world_size=8`) 反向 RS (`#8330`) | FP32 RS 每卡 `234.6933 GiB` | BF16 线速 RS + FP32 累加，每卡 `117.3466 GiB` | 带宽需求降低 **`2.0x`** |
| TransformerEngine `GroupedTensor` 视图派发 (`#3701`) | 内部 `requires_grad=True` 阻断 `ViewBackward0` | 内部 `requires_grad=False` + 4 张量 `record_stream` | Autograd 视图链与多流安全完整保留 |
| Mooncake `256` 任务批次状态查询 (`#4910`) | `256` 次 `getTransferStatus` C FFI 调用 | `1` 次 `getBatchTransferStatus` C FFI 调用 | FFI 调用次数降低 **`256x`** |
| NIXL UCX 通知标签 (`#2487`) + 重投递遥测 (`#2478`) + Dynamo KVCR (`#15535`) | 缺执行级标签 / repost 重置计时 / KVCR 触发序号断层 | `16B` (`8B+8B`) 小端 `treq` + 单次遥测 + 预过滤 KVCR | 往返解码精确、单次发射、事件序号连续 `[101, 102, 103]` |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **Blackwell SM100 TMEM 512 列容量边界核算**：在 `FlashAttentionBackwardSm100` (`2CTA, tile_m=128, tile_n=64, hdim=256`) 中，单 CTA 分担 $dV(128) + dK(128) + S/P(64) + dP/dS(64) + dQ(128) = 512$ 列 TMEM，恰好占满硬件 512 列上限。请证明为何在此配置下无法在不溢回 SMEM 的前提下将 `tile_n` 翻倍至 `128`（提示：`S/P` 与 `dP/dS` 列数将翻倍至 $128 + 128$，总列数将达 $640 > 512$）。
2. **可证伪假设**：在混合 Attention + Mamba2 模型（如 Nemotron-H / Jamba 类架构）的分离式推理路由中，若 KV 路由器在验证注意力块几何尺寸后才过滤 `CacheOwner` (KVCR) 事件，则每当单个解码步同时产出一个 KVCR 事件与一个注意力块事件时，注意力路由器的 `next_event_id` 校验必然观察到非注意力块几何尺寸而报错。
