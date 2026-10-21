# SGLang DeepSeek-V4.1 WindowLayout 与 Index-K 按压比缓存、请求窗口下稀疏 Prefill 恢复、RadixCache 双流同步卸载及 vLLM 单体 MoE 路由模拟守卫

> **一手上游来源**:
> - `sgl-project/sglang@f7d4f06242b76615a2cc95e87426b7123a90c7f6` (`[dsv4] Precompute WindowLayout history_src once per batch (#45579)`, 2026-10-21)
> - `sgl-project/sglang@bc412213c17cecfa826716e23d5ae00748bbe528` (`[dsv4] Cache V4.1 prefill index-k gather and plans per ratio (#45577)`, 2026-10-21)
> - `sgl-project/sglang@97945831f9ec29755099284243ab7d91336d314f` (`[dsv4] Restore sparse prefill when request window is active (#45573)`, 2026-10-21)
> - `sgl-project/sglang@620718c8ea67d8f1ef827f6be78a77b3a7830b41` (`[dsv4] Keep vision_topk off packed-ids under routing simulation (#45575)`, 2026-10-21)
> - `sgl-project/sglang@2c7a9e3b50828720951a5f7e71f42563d2df1b47` (`[RadixCache] Order cache linker offloads behind forward stream (#45559)`, 2026-10-21)
> - `vllm-project/vllm@b0e1f8052abb3de7c2fa8dc6234733104e829804` (`[MoE] Support simulated routing in monolithic MoE backends (#52780)`, 2026-10-21)

---

## 1. SGLang DeepSeek-V4.1 `WindowLayout` 预计算、按压缩比缓存 Index-K 与稀疏 Prefill 恢复

### 1.1 `WindowLayout.history_src` 批次级单次预计算 (`sglang@f7d4f06242`)
在 DeepSeek-V4.1 的 60 层长窗口推理中，`forward_prepare` 原本在全部 60 层逐层重复调用 `req_to_token[req_pool_indices[:, None], pos]` 执行相同的二维花式索引收集历史窗口 KV 槽位：
- 提交 `f7d4f06242` 在 `WindowLayout` 数据类中新增 `history_src: Optional[torch.Tensor] = None`，在 `build_window_layout()` 中仅当 `max_hist > 0` 时执行 **1 次**二维索引并将无效位置掩码置零（`torch.where(valid_mask, ..., 0)`）；
- 每层仅需通过 `_materialize_history_slots()` 将 `history_src.masked_fill(~window.valid_mask, layer_pad_slot)` 填入该层的 pad slot，将 60 层花式索引开销降低 **60x（`60 -> 1`）**。

### 1.2 `DSV4Metadata` 按压缩比缓存 `low_ratio_k_slots` 与 `FastIndexPlan` (`sglang@bc412213c1`)
DeepSeek-V4.1 的低压缩比层（48 层，共享仅 `r in {4, 8}` 两种压缩比）在 `forward_prepare` 中原本每层都把请求历史与扩展 Token 从 `req_to_token` 收集到临时缓冲区再按 `index_k_pool[slots // r]` 采样，并重建 `FastIndexPlan`：
- **整除单调步进恒等式**：因为请求窗口左边界与历史长度 `H` 均对齐到压缩比 `r` 的整数倍（`H % r == 0`），先将历史位置与扩展位置（步长 `r`）拼接后再取 `::r`，在数学上严格等价于直接对历史取 `::r`（得 `H // r` 个槽位）并拼接扩展块首槽位 `ext[::r]`；
- 提交 `bc412213c1` 在 `DSV4Metadata` 中引入 `low_ratio_k_slots: dict[int, torch.Tensor]` 与 `low_ratio_index_plans: dict[int, FastIndexPlan]`，按压缩比 `r` 仅构建 **2 次**（而非 48 次），消除 **24x（`48 -> 2`）** 重复 KV gather 与 `FastIndexPlan` 构建。

### 1.3 请求窗口激活时恢复稀疏 Prefill 门控 (`sglang@97945831f9`)
在早先重构中，`use_sparse_prefill` 被误写为要求 `window_layout is None`，导致一旦构建了 `request_window`（`window_layout is not None`），即使序列长度超过 `sparse_prefill_threshold`（如 `total_seq_len = 8192 >= 4096`），也会被迫回退到稠密 `forward_prefill_ragged`：
- 提交 `97945831f9` 移除 `window_layout is None` 限制，当 `is_extend and not is_Idle and max_total_seq_len >= sparse_prefill_threshold` 时恢复走稀疏 `forward_prefill_context_stage`。

---

## 2. SGLang RadixCache 双流同步卸载 (`sglang@2c7a9e3b50`)

### 2.1 异步调度重叠下的“未写先读”静默 KV 损坏
在 `UnifiedCacheLinker`（`MooncakeDirectLinker` 与 `UMBPDirectLinker`）中，当 `offload_Synchronize = False`（默认异步重叠模式）时，`offload()` 在专用 `offload_stream` 上发起 D2H / RDMA 拷贝。此前 `offload_stream` 仅调用 `wait_stream(torch.cuda.current_stream())`：
- 在 SGLang 的异步调度与 Overlap 模式下，`evict` / `inc_lock_ref` 等缓存管理操作常在控制流（`current_stream`）上触发，而真正的 Attention KV 写入仍在 `forward_batch.forward_stream` 上异步执行；
- 若 `offload_stream` 不等待 `forward_stream`，就会在 `forward_stream` 完成 KV 写入前（如 `t = 12 us < 25 us`）提前拷走未初始化或旧的 KV 缓冲。

### 2.2 `record_offload_ready()` 双流屏障
提交 `2c7a9e3b50` 在 `UnifiedCacheLinker` 中引入 `forward_stream` 绑定与 `_wait_for_offload_ready()` 屏障：
1. `offload_stream` 同时等待 `torch.cuda.current_stream()` 与记录在 `forward_stream` 上的 `_offload_ready_event`；
2. 保证任何 Mooncake / UMBP 异步卸载严格排在最后一次前向 KV 写入完成之后（`t >= 25 us`）。

---

## 3. vLLM 单体 MoE 与 SGLang DSV4.1 路由模拟守卫 (`vllm@b0e1f8052a`, `sglang@620718c8ea`)

| 框架与提交 | 核心痛点 | 契约化修复机制 |
| :--- | :--- | :--- |
| **vLLM 单体 MoE 路由模拟** (`vllm@b0e1f8052a`) | FlashInfer TRTLLM BF16 / FP8 / MXFP4 等**单体（Monolithic）MoE 内核**将路由计算融合在算子内部，无法直接调用外部 Python `RoutingSimulator`。 | 在 `FusedMoERouter` 新增 `kernel_routing_method_type`（模拟激活时返回 `SimulatedNoop`）、`supports_monolithic_kernels`（仅当 `DeepSeekV3` 路由配合 `moe_backend_LoadBalanced` 时允许单体内核通过 `prepare_monolithic_bias()` 传入合成静态偏置），其余模拟策略自动拒绝单体内核并回退到分解式模块化 MoE。 |
| **SGLang DSV4.1 `vision_topk` 守卫** (`sglang@620718c8ea`) | DeepSeek-V4.1 的 `DeepseekV4MoEGate` 将 `topk_indices` 编码为紧凑 `packed_ids`（`(expert_id << 16) \| token_idx`）供 FlashInfer `cutlass_fused_moe` 消费；但 `FunctionOverrideRouter` 在路由模拟下只重写 `topk_ids` 而不更新 `packed_ids`。 | 在 `DeepseekV4MoEGate.__init__` 中增加 `and not RoutingSimulationConfig.is_enabled()` 守卫，当启用 `SGLANG_MOE_ROUTING_SIMULATION_STRATEGY` 时禁用 `enable_packed_ids`，强制走标准 `topk_ids` 路径。 |

---

## 4. 本地确定性复现实验 (`experiments/sglang-dsv4-window-sparse-prefill-indexk-cache-radix-offload-vllm-moe/run.py`)

```bash
python experiments/sglang-dsv4-window-sparse-prefill-indexk-cache-radix-offload-vllm-moe/run.py --check
```
