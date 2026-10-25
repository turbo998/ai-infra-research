# FlashAttention-4 SM90/SM100 Load-Safe 5D Ragged TMA `seqlen_k` 硬件清零（`0 * NaN` 防护）、`verl` Megatron Top-k 蒸馏 `quant_align_size` 与 `cp_layout` 打包对齐及 Mooncake TENT 寒武纪 MLU 异步主机-设备拷贝

> **文档状态**：Phase 77（`priority: 86`）第一手源码审计 + CPU/offline 合成验证  
> **面向读者**：深挖 Hopper/Blackwell TMA 描述符地址回绕、Megatron-LM Context Parallel（CP）+ MXFP8 在线蒸馏以及异构 AI 芯片（Cambricon MLU）RDMA/Staging 传输引擎的解决方案架构师  
> **English Companion**: [fa4-sm90-sm100-seqlenk-tma-nan-guard-verl-topk-cp-pack-mooncake-mlu.en.md](fa4-sm90-sm100-seqlenk-tma-nan-guard-verl-topk-cp-pack-mooncake-mlu.en.md)

---

## 1. 背景与核心工程挑战

在超长上下文变长注意力（FlashAttention-4 CuTe-DSL）、Megatron FP8/MXFP8 强化学习与在线 Top-k 蒸馏（`verl`）以及跨架构 KV 传输引擎（Mooncake TENT）中，存在三个极具代表性的底层契约问题：

1. **FlashAttention-4 变长/预分配 KV Cache 尾部的 `0 * NaN = NaN` 污染**：在增量解码或变长打包训练中，预分配的 `K` / `V` 缓冲区在有效长度 `seqlen_k`（由 `cu_seqlens_k` 或 `seqused_k` 指定）之后的未使用行常包含未初始化的 `NaN`。虽然在线 softmax 将越界位置的注意力权重掩码为 $P_{ij} = 0$（或反向中的 $dS_{ij} = 0$），但在 IEEE-754 浮点乘加中 $0 \times \text{NaN} = \text{NaN}$！一旦最后一个 `tile_n` 的 TMA 加载将这些越界 `NaN` 行搬入共享内存并参与 $O = P V$、$dP = V dO^T$ 或 $dQ = dS K$ 的 Tensor Core MMA，整行输出或梯度即被毒化为 `NaN`。此外，`quack` 现有的 `ptr_shift=True` 3D 描述符会将基址移出已映射显存区（仅对 TMA store 安全，对 TMA load 触发非法地址访问）且不支持 rank $> 3$ 的张量。
2. **`verl` Megatron Top-k 蒸馏在 MXFP8 与混合 `cp_layout` 下的崩溃与“形状一致但内容错位”的静默数据污染**：
   - 当启用 MXFP8（要求每序列 padding 对齐到 `32` 的倍数，而 blockwise FP8 为 `16`）时，旧蒸馏路径硬编码传递 `use_fp8_padding` 布尔值，导致 MXFP8 下 student 与 teacher 打包长度不一致而触发形状断言崩溃（`#8325`）。
   - 更危险的是（`#8332`），当 `dsv4_hybrid` student 配置 `cp_layout="contiguous"` 而 teacher 沿用默认的 `"zigzag"` 打包时，在 `CP=2`、`MXFP8`（`quant_align_size=32`）、16 条长度为 17 的序列下，两侧每个 CP rank 打包后的行数完全相同（均为 `256` 行），没有任何形状报错，但 **Rank 0 上 `128 / 136`（`94.1%`）个有效 token、Rank 1 上 `136 / 136`（`100%`）个有效 token** 的 student logits 被错误地与另一个 token 甚至 padding 行的 teacher Top-k 分布计算 KL 散度！
3. **Mooncake TENT 在寒武纪 MLU（Cambricon MLU）上的同步拷贝瓶颈与内存注销竞态（`#4720`）**：原先 TENT 的 MLU 传输层所有 H2D/D2H 均采用同步拷贝，无法将大块主机 staging 缓冲区的设备拷贝与 TCP/HP-TCP 网络传输流水化重叠；同时需要保证小包（$\le 64\text{ KiB}$）H2D 与可分页内存（pageable memory）在异步队列非空时的严格保序，以及在本地段元数据同步失败回滚时防止并发释放仍在飞行的缓冲区。

---

## 2. 第一手源码审计与数学推导

### 2.1 FlashAttention-4 Load-Safe 5D Ragged TMA 描述符（`#2983`、`#3012`、`#3013`）

在 `flash_attn/cute/copy_utils.py`、`flash_fwd_sm100.py`、`flash_bwd_sm100.py`、`flash_fwd_sm90.py` 与 `flash_bwd_sm90.py` 中，Dao-AILab 引入了 `create_ragged_tensor_for_tma_load` 与 `offset_ragged_tensor_for_tma_load`：

- **为什么不能用 `ptr_shift=True`**：`quack` 的 store 方案将 TMA 描述符基址负向偏移，使有效末尾对齐描述符上界。但硬件 TMA **load** 会校验描述符基址是否落在合法映射的虚拟地址空间内，偏移后的基址会触发 MMU 故障。
- **5D 无符号 64 位地址回绕（64-bit Unsigned Wraparound）构造**：
  设元素字节数为 $b \in \{1, 2, 4\}$。硬件 TMA 描述符要求任意维度的字节步长不超过 $2^{40}$ bytes。令步长单位为：
  $$\text{wrap\_stride} = \frac{2^{40}}{b} \text{ elements} \quad (\text{恰为 } 2^{40} \text{ bytes})$$
  $$\text{ragged\_rows} = \frac{2^{64}}{\text{wrap\_stride}} = 2^{24} \cdot b \text{ rows} \quad (\text{FP16/BF16 下为 } 2^{25} = 33,554,432 \text{ 行})$$
  构造 5D TMA 描述符形状为：
  $$\text{Shape}_{5\text{D}} = \left(\text{features},\; \text{ragged\_rows},\; 2^{30} + 1,\; 2^{31} - 1,\; \text{ragged\_rows} + 1\right)$$
  元素步长为 $\left(1,\; \text{row\_stride},\; \text{wrap\_stride},\; \text{wrap\_stride},\; 0\right)$。当需要加载有效行数为 $L = \text{seqlen\_k}$ 的序列时，将第 1 维起始坐标设为 $\text{ragged\_rows} - L$，并通过第 2、3 维的大步长偏移使总字节偏移精确达到 $2^{64}$ 的整数倍（在 64 位无符号算术下回绕为基址 $0$）：
  - 对于任意有效行 $0 \le r < L$：第 1 维坐标为 $\text{ragged\_rows} - L + r < \text{ragged\_rows}$，地址回绕后精确命中真实显存行 $r$；
  - 对于任意越界行 $r \ge L$：第 1 维坐标 $\ge \text{ragged\_rows}$，直接触发硬件 TMA 越界清零（OOB zero-fill），在进入共享内存和寄存器前即被硬件替换为 `0.0`，从根源上消灭 $0 \times \text{NaN} = \text{NaN}$！
- **跨内核全覆盖与性能**：
  - `#2983`：修复 SM100 前向 `V` TMA 加载（通过 194 个既有 B200 用例 + 146 个新增 `NaN` 尾部回归测试，最大延迟开销 $\le 2.23\%$）。
  - `#3012`：扩展 `create_ragged_tensor_for_tma_load(tensor, rows_dim=0)` 支持 `(rows, features)` 布局，并在 SM100 反向 `flash_bwd_sm100.py` 中对 `K`、`V` 及双 CTA `Kt` 施加 `seqlen_k` 边界，消除 $dP = V dO^T$ 与 $dQ = dS K$ 的 `dQ` `NaN` 污染。
  - `#3013`：推广至 SM90 Hopper 前向（`V`）与反向（`K`、`V`），用零开销硬件 TMA OOB 清零替代了 `#2407` 的共享内存尾部手动清零。

### 2.2 `verl` Megatron Top-k 蒸馏 `quant_align_size`（`#8325`）与 `cp_layout` 对齐（`#8332`）

在 `verl/models/mcore/model_forward.py`、`util.py`、`verl/utils/megatron/pipeline_parallel.py` 与 `verl/workers/megatron_workers.py` 中：

1. **配方无关的 `get_thd_quant_align_size(config)`（`#8325`）**：将布尔开关替换为精确粒度返回值——`mxfp8` 返回 `32`，`blockwise` 等其他 FP8 配方返回 `16`，BF16 返回 `None`；每条序列填充对齐至 $\text{lcm}(\text{quant\_align\_size}, 2 \cdot \text{CP})$，总 padded tokens 对齐至 $2 \cdot \text{CP} \cdot 128$。在 2×B200 上运行 Megatron Top-k 蒸馏 + MXFP8 连续 5 步全部通过（`rollout_corr/kl` 稳定在 `0.0018–0.0026`）。
2. **显式透传 `cp_layout` 与 `local_cp_size`（`#8332`）**：将 student 模型的 `cp_layout`（如 `"contiguous"` 或 `"zigzag"`）与 `local_cp_size` 一并传入 `compute_forward_kl_topk -> postprocess_packed_seqs_for_dict_output -> remove_left_padding`，确保 teacher 的 `[T, K]` top-k logprobs / indices 与 student 隐藏状态采用完全相同的 CP 切分拓扑。

### 2.3 `Mooncake@a7496800527b`（`#4720`）：TENT 寒武纪 MLU 异步拷贝与 `ProgressWorker` 轮询

在 `mooncake-transfer-engine/tent/src/transport/mlu/mlu_transport.cpp` 与 `src/runtime/progress_worker.cpp` 中：

- **按上下文与方向隔离的 `MluCopyQueue`（`CNqueue` + `CNnotifier`）**：
  - 锁页内存（`cnrtHostMalloc`）D2H 拷贝与 $> 64\text{ KiB}$ 的锁页 H2D 拷贝使用 `cnMemcpyAsync_V3` 提交至每上下文、每方向的 `CNqueue`，并放置 `CNnotifier` 异步跟踪完成状态。
  - $\le 64\text{ KiB}$ 的 H2D 拷贝在当前上下文 H2D 队列不存在或空闲时直接走同步内联 `cnMemcpyHtoD` 以规避队列开销；若前序 H2D 队列仍有未完成任务，则锁页拷贝排入同一队列，可分页（pageable）拷贝则先等待该 H2D 队列排空（drain）后再执行同步拷贝，严格保持单上下文内的 H2D 顺序。
- **`ProgressWorker::notifyBatchMaybeReady(batch_id, poll_until_terminal=true)`**：由于 `CNnotifier` 不会主动触发 OS 级完成中断，MLU `SubBatch::needsProgressPolling()` 返回 `true`，由 `ProgressWorker` 将其加入 `polling_` 集合周期性推进，直至所有子任务进入终态后才允许释放 caller 缓冲区；同时 `SegmentTracker` 通过递归互斥锁串行化注册与回滚，确保 `synchronizeLocal()` 失败回滚时不会与并发注销产生竞态。

---

## 3. 合成实验验证结果

通过运行 `experiments/fa4_sm90_sm100_seqlenk_tma_nan_guard_verl_topk_cp_pack_mooncake_mlu/run.py --check` 得到如下验证矩阵：

| 验证维度 | 核心参数 / 场景 | 修复前行为（Legacy） | 修复后行为（Phase 77 Audited） | 校验结论 |
|---|---|---|---|---|
| FA4 5D Ragged TMA (`#2983`, `#3012`, `#3013`) | FP16/BF16 (`b=2`), `seqlen_k=50`, `tile_n=128` 尾部含 `NaN` | `0.0 * NaN` 导致输出 `O` 与梯度 `dQ` 全为 **`NaN`** | 5D 形状 `(128, 2^25, 2^30+1, 2^31-1, 2^25+1)` 硬件 OOB 清零，输出精确为 **`1.0`** | 步长 `2^40 B` 合规，無 `NaN` |
| `verl` Top-k 蒸馏 CP 对齐 (`#8325`, `#8332`) | `CP=2`, `MXFP8` (`align=32`), 16 条 17-token 序列 (`256` 行/rank) | 形状静默匹配 (`256==256`)，但 Rank 0 **128 个**、Rank 1 **100%** 有效 token 错位 | 透传 `quant_align_size=32` 与 `cp_layout="contiguous"`，跨 rank 错位数为 **`0`** | 消除静默蒸馏毒化 |
| Mooncake TENT MLU 异步传输 (`#4720`) | `64 KiB` H2D 阈值 + `CNnotifier` + `SegmentTracker` 回滚 | 全同步阻塞；元数据同步失败存在注销竞态 | 空闲 $\le 64\text{ KiB}$ 走内联同步，其余走 `CNqueue` + `poll_until_terminal` | 保序与生命周期安全 |

---

## 4. 架构师设计练习与后续可证伪问题

1. **地址回绕推导练习**：当 FlashAttention-4 处理 FP8（`elem_bytes = 1`）或 FP32（`elem_bytes = 4`）张量时，根据公式 $\text{wrap\_stride} = 2^{40} / b$ 与 $\text{ragged\_rows} = 2^{64} / \text{wrap\_stride}$，分别计算 5D Ragged TMA 描述符所能支持的最大 `ragged_rows` 行数上限。
2. **可证伪问题**：
   - **H1**：在 SM100 反向内核中，对 2-CTA `Kt` 共享 TMA 加载启用 5D Ragged 描述符后，地址计算增加的整数指令是否会在极短序列（`seqlen_k <= 256`）下产生超过 `3%` 的延迟退化？
   - **H2**：在 Mooncake TENT MLU 传输中，当并发 staging 流数量超过 8 个时，将内联同步 H2D 阈值从 `64 KiB` 下调至 `16 KiB` 是否能通过提高 PCIe DMA 重叠度降低 P99 尾延迟？
