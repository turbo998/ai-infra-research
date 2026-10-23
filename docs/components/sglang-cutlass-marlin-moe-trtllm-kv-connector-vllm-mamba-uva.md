# SGLang CUTLASS/Marlin MoE 路由预算护栏、TRT-LLM 外部连接器 KV 复用记账与 vLLM Mamba/UVA 调度优化

> **文档状态**：2026-10-24 · Phase 62 (`priority: 71`, Track B)  
> **证据级别**：`implementation` + `own-offline`（`experiments/sglang-cutlass-marlin-moe-trtllm-kv-connector-vllm-mamba-uva/run.py` 确定性标准库仿真）  
> **关联问题**：`q-sglang-cutlass-marlin-moe-trtllm-kv-connector-vllm-mamba-uva`

---

## 1. 架构背景与核心问题

在生产级 LLM 推理引擎（SGLang、TensorRT-LLM、vLLM）中，MoE 稀疏路由下存在大量“未命中本地专家的无效路由槽位（inactive routes）”、跨轮次与跨节点 KV Connector 块复用边界、以及混合架构（Mamba + Attention）与主机内存卸载（UVA Weight Offloading）的调度退化。近期上游合并的 8 个核心 PR 精准解决了以下生产隐患：

1. **SGLang CUTLASS MoE 完整路由元数据与 Blockscale 128 对齐 (`ac3db731b3`, PR `#45917`)**：JIT 编译 `cutlass_prepare_moe_input` 与 `shuffle_weighted_sum`，确保所有未用映射槽位写满确定性哨兵且按槽位顺序执行无 FTZ 的 FMA。
2. **SGLang Marlin MoE 失活块归零与前缀预算 (`8b9b992e41`, PR `#45462`)**：引入 `kExpertIdsAreSorted`、`kZeroInactiveOutput` 与 `kHasInactivePrefixCount`，显式对 `expert_ids[block] == -1` 的路由输出行写 `{0, 0, 0, 0}`。
3. **SGLang DeepSeek-V4 尾部 MoE Token 计数所有权 (`1340044e97`, PR `#45915`) 与 AMD AITER FP8 DSA Prefill (`0931dfedc9`, PR `#39083`)**：64 位打包 `(global_rows << 32) | local_rows` 并约束 CUDA Graph 捕获时的 `moe_num_token_non_padded` 作用域；修复 AMD MI300X AITER FP8 KV Cache 预填充 `identity_scale`。
4. **TensorRT-LLM KV Cache Manager V2 外部连接器块复用记账 (`2706f4dfb6` `#20180`) 与跨轮次 Generation KV 复用 (`fac6d3f794` `#19710`)**：通过 `[divUp(tokenBegin, B), tokenEnd / B)` 精确统计完整填充的外部 Connector 块，并在解耦传输关闭请求前提前终结 V2 Context/Generation 资源。
5. **vLLM Mamba 已提交尾部块免重标 (`d561af5dfc` `#62841`) 与 UVA 主机驻留 MoE 强制排序 (`3b364baef5` `#62054`)**：消除共享前缀兄弟请求串行化退化（8 请求从 8 步降至 **1 步**），并将 UVA 卸载专家权重的 PCIe 重复读取削减 **3.2x**。

---

## 2. 源码级机制深挖与数学建模

### 2.1 SGLang CUTLASS MoE 路由准备与 Marlin 失活块归零 (`sglang@ac3db731b3`, `8b9b992e41`)

- **`cutlass_prepare_moe_input.cuh` (`#45917`)**：
  1. `count_expert_routes<kTransposeProblemSizes>` 统计每个专家 $e \in [0, E)$ 的路由数 $c_e$，并写入 `problem_sizes1[e]` 为 $[c_e, 2N, K]$（若转置则为 $[2N, c_e, K]$）与 `problem_sizes2[e]` 为 $[c_e, K, N]$。
  2. `compute_expert_offsets` 同时计算紧凑偏移 `expert_offsets[e+1] = expert_offsets[e] + c_e` 以及 FP8/FP4 块缩放所需的 **128 对齐偏移**：
     $$\text{blockscale\_offsets}[e+1] = \text{blockscale\_offsets}[e] + \left\lfloor \frac{c_e + 127}{128} \right\rfloor \times 128$$
  3. `complete_route_maps` 将越界路由（$e \notin [0, E)$，例如 EP 过滤后的异地专家）的 `output_permutation[route]` 设为 `-1`，并将 `input_permutation` 中超出 `expert_offsets[E]` 的尾部容量全部指向 Token `0`，保证向量化全量 Gather 不会越界读取未初始化显存。
  4. `shuffle_weighted_sum.cuh` 中的 `accumulate` 按固定槽位顺序执行 FMA（包含失活槽位的 $+0 \times +0$，禁用 FTZ），确保多副本数值完全一致。
- **`marlin_template.h` (`#45462`)**：新增模板参数 `<kExpertIdsAreSorted, kZeroInactiveOutput, kHasInactivePrefixCount>` 与 `zero_inactive_output_rows`。当未排序模式下 `expert_ids[block] == -1` 时，Kernel 遍历该块内所有 `route < num_routes`，将输出行向量显式写零 `{0, 0, 0, 0}`，避免复用的 Workspace 残留脏数据污染 MoE 归约结果。

### 2.2 SGLang DeepSeek-V4 尾部 MoE Token 计数所有权 (`sglang@1340044e97`)

在 `row_bucket_graph_runner.py` 与 `deepseek_v4_replay_tail.py` 中：
- 使用 `_pack_token_counts(local_rows, global_rows) = (global_rows << 32) | local_rows` 将本地与全局非填充 Token 数打包进单个 64 位整数。
- `DecoderReplayTail._can_capture_moe_bound(forward_batch)` 严格限制只有当 `bound` 为 `None`、`bound is forward_batch.num_token_non_padded`（本地计数），或者 **`forward_batch.attn_tp_sequence_sharded` 为真且 `bound is forward_batch.global_num_token_non_padded`** 时，才允许进入 CUDA Graph 尾部捕获——因为只有这两个计数域拥有 Graph 托管的 Replay Buffer。

### 2.3 TRT-LLM KV Cache Manager V2 外部连接器复用记账 (`trtllm@2706f4dfb6`, `fac6d3f794`)

在 `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/kvCache.cpp` 中，新增 `KvCache::recordConnectorReuse(int tokenBegin, int tokenEnd)`，用于记账从外部 KV 存储（如 Mooncake / LMCache / NIXL Connector）异步加载而非前向计算生成的 Prompt Token 区间 `[tokenBegin, tokenEnd)`：

$$\text{blockBegin} = \left\lceil \frac{\text{tokenBegin}}{B} \right\rceil = \text{divUp}(\text{tokenBegin}, B), \quad \text{blockEnd} = \left\lfloor \frac{\text{tokenEnd}}{B} \right\rfloor$$

- **核心不变量**：只有被外部 Connector **完全填满** 的块区间 $[\text{blockBegin}, \text{blockEnd})$ 才记入 `recordConnectorReuse(blockBegin, blockEnd)`（计为复用块）；任何仍需前向 Pass 写入部分 Token 的首尾半满块（Partial Boundary Blocks）必须保持为 **Miss**！同时，必须在异步传输完成前、接受加载请求时（即 `commitPendingStats()` 之前）记账。
- **跨轮次 Generation KV 块复用 (`#19710`)**：在 `py_executor.py` 的 `_handle_executed_batch` 中，在触发 `self._send_kv_async` 之前同时调用 `self._update_v2_context_resources(scheduled_requests)` 与 `self._update_v2_generation_resources(scheduled_requests)`，并在后续 `resource_manager.update_resources(..., skip_kv_cache_manager=True)` 中跳过重复更新，确保多轮对话中已生成的 KV 块在请求终止前被正确提交至 V2 基数树（Radix Tree）供下一轮命中复用。

### 2.4 vLLM Mamba 尾部块免重标 (`vllm@d561af5dfc`) 与 UVA 专家权重排序 (`vllm@3b364baef5`)

1. **Mamba 混合缓存共享前缀并发命中 (`#62841`)**：在 `vllm/v1/core/single_type_kv_cache_manager.py` 的 `_cache_partial_tail_block` 中，当多个兄弟请求（Sibling Requests）共享同一个已在先前步骤提交（Committed）的 Mamba 状态边界时，旧代码会再次调用 `cache_partial_block`，将该块哈希重新标记为“本步新缓存（cached this step）”，导致同批次其余所有兄弟请求被拒绝并推迟到后续 Step 串行执行（1 个请求/Step）。新增 2 行守卫：
   ```python
   if source_block.block_hash_num_tokens == num_tokens:
       return None
   ```
   使所有命中已提交边界的兄弟请求在 **单次 Engine Step** 内全部并发执行。
2. **UVA 主机驻留 MoE 权重强制走排序路径 (`#62054`)**：在 `vllm/model_executor/layers/fused_moe/fused_moe.py` 中，当启用 `--offload-backend uva` 时，权重驻留在主机内存（`_vllm_is_uva_offloaded = True`）。在小批量下（`num_tokens * top_k * 4 <= global_num_experts`），旧启发式会激活 `naive_block_assignment = True`（跳过 `moe_align_block_size`），导致 Triton Kernel 按每个 `(token, expert)` 对重复通过 PCIe 读取同一专家权重。修复加入 `and not weights_host_resident`，强制 UVA 权重走 `moe_align_block_size` 排序路径，使每个激活专家的权重仅跨 PCIe 读取 **1 次**。

---

## 3. 确定性实验验证 (`own-offline`)

运行 `python experiments/sglang-cutlass-marlin-moe-trtllm-kv-connector-vllm-mamba-uva/run.py --check` 的量化结果：

| 模块与 PR | 测试场景参数 | 修复前行为 | 修复后确定性结果 |
|---|---|---|---|
| SGLang CUTLASS (`#45917`) | `8` 路由（含 `2` 越界 `-1`），`4` 专家 | 尾部未初始化、无 128 对齐偏移 | `expert_offsets=[0,3,4,6,6]`, `blockscale=[0,128,256,384,384]`, 尾部全 `0` |
| SGLang Marlin (`#45462`) | `3` 块（中间块 `eid=-1`），`6` 路由 | 失活块保留 Workspace 脏值 | 失活路由 `[1, 4]` 精确写零，全集覆盖 `[0..5]` |
| SGLang 尾部计数 (`#45915`) | `local=16, global=128` | 非序列分片误捕获全局计数 | 仅当 `attn_tp_sequence_sharded=True` 时允许捕获全局边界 |
| TRT-LLM Connector (`#20180`) | `B=64`，加载 `[16, 240)` (`224` tokens) | 首尾半满块误计入复用块 | 仅中间块 `[1, 3)` (`2` 块) 计为复用，首尾 `2` 个半满块保持 Miss |
| vLLM Mamba (`#62841`) | `8` 个兄弟请求命中已提交 Mamba 边界 | 退化为 `8` 个串行 Engine Steps | **1 个 Engine Step** 全部并发完成（**8.0x** 步数缩减） |
| vLLM UVA MoE (`#62054`) | `4` tokens $\times$ `top-8` (`32` 对)，命中 `10` 个唯一专家 (`48 MiB`/专家) | Naive 路径跨 PCIe 读 `32` 次 (`1536 MiB`) | 强制排序路径跨 PCIe 仅读 `10` 次 (`480 MiB`，**3.2x** 带宽节省) |

---

## 4. 解决方案架构师行动指南

1. **PCIe / UVA 卸载推理调优**：在单卡显存受限、通过 `--offload-backend uva` 将超大 MoE（如 256 专家）部分驻留主机 DDR 时，必须确保包含 `#62054` 补丁，否则 Decode 阶段多 Token 命中同一热门专家时 PCIe 流量会放大 $2\text{–}4\times$。
2. **外部 KV 缓存块对齐设计**：在将 Mooncake 或 LMCache 接入 TRT-LLM V2 时，Connector 写入区间应尽量按 `mTokensPerBlock`（如 64 或 128 tokens）整数倍对齐，避免首尾非对齐块无法计入块级复用。
