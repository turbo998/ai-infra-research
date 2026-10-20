# SGLang DSpark CUDA 模拟接受快路径、Radix-Cache 可回收槽位长度解耦与双端池首尾优先分配，及 vLLM ROCm Gated RMSNorm 算子深度剖析

> **Lang / 语言**: **中文 (当前)** | [English](./sglang-dspark-sim-accept-radix-reclaimable-subpool-vllm-rocm-gated-norm.en.md)
>
> **证据等级 (Evidence Class)**：`implementation`（上游固定 Commit 源码核验）+ `own-offline`（本项目 Python 3.12 标准库确定性建模 `experiments/sglang_dspark_sim_accept_radix_reclaimable_subpool_vllm_rocm/run.py --check`）。

---

## 1. 架构背景与核心问题

在大模型投机解码（Speculative Decoding）、混合模型统一前缀缓存（Unified Radix Cache for Full + SWA + Mamba）与跨平台图编译融合（ROCm AITER Custom Op Fusion）的生产实践中，2026-10-20 上游合入的 4 项关键变更解决了以下核心工程痛点：

1. **DSpark 投机解码模拟接受（Simulated Acceptance）在 CUDA 上的冗余采样开销**：在压测投机解码吞吐或固定接受率基准时，SGLang DSpark 支持通过 `simulated_correct_len` 指定模拟接受长度。然而旧版代码将 `accept_sampling_info = None`、`simulate_fold` 与 `_simulate_acc_greedy` 全部用 `_is_hip`（ROCm）条件宏锁死。导致在 NVIDIA CUDA 上开启模拟接受时，每次校验步仍完整执行 `draft_probs` / `target_logits` 的词表级 Softmax 与拒绝采样核函数，计算结果随后又被 `simulated_correct_len` 直接覆盖丢弃。
2. **Unified Radix Cache 逻辑索引长度与物理可回收槽位长度漂移（Ledger Drift）**：在 `UnifiedTreeCore`（Python 与 Rust `sglang-radix-tree` 双栈）中，当组件存在半页共享回收（`rows_handed_back`）、附加侧边缓存（`side_backing`）或无设备物理页支撑的逻辑占位（Tombstone / Root）时，节点存储的设备张量逻辑长度 `value_len(node)` 与驱逐时真正归还给分配器的槽位数并不相等。旧版代码在 `mamba.py`、`swa.py` 及 Rust 端 `full.rs` / `mamba.rs` / `swa.rs` 的插入、加锁、解锁路径中混用了 `len(value)` 与 `reclaimable_tokens(node)`，导致 `component_evictable_size_` 与 `component_protected_size_` 账本漂移并在 `sanity_check` 时触发断言失败。
3. **Unified KV Pool 双端分配器（`MultiEndedAllocator`）内部碎片**：Full Attention 子池向上生长（`grow_direction="up"`，低地址端优先），SWA 子池向下生长（`grow_direction="down"`，高地址端优先）。旧版 `take_physical` 直接按释放顺序切片 `self._free_phys_pages[:n_drain]`，将刚好落在池中间区域的空闲页随机分配出去，打断了 H2D 异步加载所需的连续段并增加压缩搬移（Compaction Moves）次数。
4. **vLLM ROCm 线性注意力门控归一化（`RMSNormGated`）图匹配断层**：Qwen3.8-27B 等混合线性注意力模型依赖 Flash Linear Attention（FLA）的 `RMSNormGated`。旧版 `RMSNormGated.forward_cuda` 直接调用第三方 Python 函数 `rmsnorm_fn`，对 `torch.compile` 暴露展开后的原生算子子图而非稳定的不透明自定义算子（Custom Op），导致 ROCm AITER 的 `RMSNormGatedQuantPattern` 融合通行证无法统一匹配自定义算子与原生分解两条路径。

---

## 2. 一手源码实现路径剖析

### 2.1 SGLang DSpark CUDA 模拟接受快路径解锁（Commit `7d9bc10493`）

在 `sgl-project/sglang` Commit [`7d9bc10493`](https://github.com/sgl-project/sglang/commit/7d9bc10493275a371989bd03a13147588005f136)（PR `#45421`，2026-10-20）中，`python/sglang/srt/speculative/dspark_components/dspark_verify.py` 与 `dspark_worker_v2.py` 移除了 `_is_hip` 限制：

```python
# dspark_verify.py:
simulate = self._simulated_correct_len is not None
# Simulated acceptance overwrites correct_len below, so a sampling accept
# (draft/target softmax + rejection) would be computed only to be discarded.
accept_sampling_info = None if simulate else sampling_info

# dspark_worker_v2.py:
simulate_fold = envs.SGLANG_DSPARK_SIMULATE_FOLD.get()
self._simulate_acc_greedy = (
    self._simulate_acc_len > 0 and envs.SGLANG_SIMULATE_ACC_GREEDY.get()
)
```

对于批次大小 $B=16$、草稿步数 $K=5$、词表大小 $V=152{,}064$ 的单次校验步，旁路 `sampling_info` 在 CUDA 上每步直接省去 $2 \times B \times K \times V = 24{,}330{,}240$ 个元素的无效 Softmax 归一化与拒绝采样读取。

### 2.2 Radix-Cache `value_len` 与 `reclaimable_value_len` 严格解耦（Commit `0b536ca112`）

在 `sgl-project/sglang` Commit [`0b536ca112`](https://github.com/sgl-project/sglang/commit/0b536ca112652008d5fa8766df7258b0700109f3)（PR `#45238`，2026-10-20）中，Python `TreeComponent` 与 Rust `sglang-radix-tree` 统一了双层长度语义：
- `value_len(node)` / `device_value_len(ct)`：仅表示节点上存储的设备索引张量逻辑长度（用于寻址偏移和切片）。
- `reclaimable_value_len(node)` / `device_reclaimable_value_len(ct)`：表示驱逐该节点时分配器实际收回的槽位数（根节点或无设备值返回 `0`）。所有账本变更点（`_add_new_node`、`_unevict_node_on_insert`、`acquire_component_lock`、`release_component_lock`、`release_window_lock`、`evict_component`、`commit_hicache_transfer`、`set_component_device_value`）一律只读写 `reclaimable_value_len`。
- 同时，`adjust_component_size` 新增非负守卫：若 `updated = sizes[component_type] + delta < 0`，立即抛出 `ValueError` 且**绝不修改原账本计数器**；`evict_device_next_node` 的 `made_progress` 判定补齐了 `any(result.device_frees.values()) or any(result.host_frees.values())`。

### 2.3 Unified-Memory `MultiEndedAllocator` 惰性排序与池端优先分配（Commit `bab4a808d8`）

在 `sgl-project/sglang` Commit [`bab4a808d8`](https://github.com/sgl-project/sglang/commit/bab4a808d81fd47a039889a3f71514679d799136)（PR `#45222`，2026-10-20）中，`python/sglang/srt/mem_cache/allocator/unified_sub_pool.py` 引入了基于张量引用标识（Identity）的惰性设备端排序与端优先分配：

```python
def _sorted_free_phys_pages(self) -> torch.Tensor:
    free = self._free_phys_pages
    if free is not self._free_phys_sorted:
        if free.numel() > 1:
            free, _ = torch.sort(free)
            self._free_phys_pages = free
        self._free_phys_sorted = free
    return free

def _take_free_pages_from_end(self, n: int, end: str) -> torch.Tensor:
    n_free = int(self._free_phys_pages.shape[0])
    if n == n_free:
        taken, rest = self._free_phys_pages, self._free_phys_pages[:0]
    else:
        free = self._sorted_free_phys_pages()
        if end == "low":
            taken, rest = free[:n], free[n:]
        else:
            taken, rest = free[n_free - n :], free[: n_free - n]
    self._free_phys_pages = rest
    self._free_phys_sorted = rest
    return taken
```

- **零主机同步（Zero Host Sync）**：张量长度 `n_free` 属于主机侧元数据，`torch.sort` 仅在 `_free_phys_pages` 发生释放拼接（Rebind）后执行一次；后续连续切片后将 `rest` 同时赋给 `_free_phys_pages` 与 `_free_phys_sorted`，保持升序不变，无需再次排序。
- **双端紧凑性**：`grow_direction == "up"`（FULL）始终从 `"low"` 端取最小物理页号，`grow_direction == "down"`（SWA）始终从 `"high"` 端取最大物理页号，将中间连续空闲区完整保留给压缩器与大块传输。

### 2.4 vLLM ROCm `fla_rms_norm_gated` 自定义算子注册与默认启用（Commit `9da5f2d310`）

在 `vllm-project/vllm` Commit [`9da5f2d310`](https://github.com/vllm-project/vllm/commit/9da5f2d3103df460dfca53df4ab4884def5c935f)（PR `#60250`，2026-10-20）中：
1. `vllm/model_executor/layers/layernorm.py` 通过 `direct_register_custom_op("fla_rms_norm_gated", ...)` 将 FLA 的 `rmsnorm_fn` 封装为 `torch.ops.vllm.fla_rms_norm_gated` 并提供 `_fla_rms_norm_gated_fake` 元形状推导；
2. `vllm/compilation/passes/fusion/matcher_utils.py` 的 `RMSNormGatedMatcher.forward_custom` 改为匹配 `torch.ops.vllm.fla_rms_norm_gated`；
3. `vllm/platforms/rocm.py` 的 `RocmPlatform.apply_config_platform_defaults` 对未量化模型（`model_config.quantization is None`）默认追加 `"+rms_norm_gated"`。

---

## 3. 确定性实验结果验证（`own-offline`）

运行 `python experiments/sglang_dspark_sim_accept_radix_reclaimable_subpool_vllm_rocm/run.py --check` 验证了全部 4 项核心不变量：
- **DSpark CUDA 模拟接受**：单步消除 `24,330,240` 个冗余词表 Softmax 元素计算；
- **Radix-Cache 账本一致性**：在包含半页释放与侧边缓存的 4 节点树上，消除 `40 - 33 = 7` 个槽位的账本虚高漂移，且负数 `adjust_component_size` 严格抛出 `ValueError` 并保持原账本 `33` 不变；
- **双端池首尾优先分配**：在乱序空闲列表 `[10, 2, 14, 0, 6, 12, 4, 8]` 上，FULL 取 `[0, 2]`、SWA 取 `[12, 14]`（第二次分配零重排），完整保留内部连续区 `[4, 6, 8, 10]`；
- **vLLM ROCm 默认策略**：未量化模型自动开启 `+rms_norm_gated`，显式 `-rms_norm_gated` 或量化模型严格跳过。

---

## 4. 面向解决方案架构师的设计准则与可证伪问题

1. **缓存驱逐账本必须以“分配器回收粒度”而非“张量逻辑长度”记账**：在引入前缀块部分释放（Partial Row Hand-Back）或分层侧边缓存后，准入控制（Admission Control）若使用逻辑长度记账，会高估可驱逐空间并触发运行时 OOM。
2. **双端共享物理池必须按生长方向定向消耗空闲列表**：FIFO 顺序消耗空闲页会迅速在双向生长边界之间制造交错碎片。
3. **后续可证伪问题**：当 vLLM ROCm 在 Qwen3.8-27B 上启用 FP8 量化时，`fla_rms_norm_gated` 与 `+quant_fp8` 的融合内核在端到端在线服务下是否能保持与 BF16 相同的数值稳定性和 CUDA Graph 捕获兼容性？
