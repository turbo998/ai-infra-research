# TransformerEngine 零存储 Dummy `wgrad` 与 NaN 安全 TP Argmax、TorchTitan `GradientClipper` 与 HF 游标保持、LMCache 亲和性 RPC、NIXL UCX CPU Proxy 及 Mooncake 存储不变量深度解析

> **Phase 52 (`priority: 61`) · Track 3 一手源码深挖**  
> **上游提交**: `NVIDIA/TransformerEngine@fc48751409dd` (`#3670`), `39a8b879dfbf` (`#3671`), `e6afa9ddfa81` (`#3606`); `pytorch/torchtitan@9698a733623c` (`#5394`), `c0eb13609240` (`#5501`); `LMCache/LMCache@b28250e765c8` (`#5803`), `fcee51cc5e01` (`#5903`); `ai-dynamo/dynamo@4f42be7bebd9` (`#16974`); `ai-dynamo/nixl@3f6e1a2716e7` (`#2327`); `kvcache-ai/Mooncake@631e1019cf88` (`#4622`), `1a6d00d0f4dc` (`#4688`), `3cf058db6d42` (`#4673`), `d782a5998f6d` (`#4755`)  
> **确定性复现脚本**: `experiments/te-zero-wgrad-argmax-torchtitan-clip-hf-cursor-lmcache-nixl-mooncake/run.py --check`

---

## 1. TransformerEngine 零存储 Dummy `wgrad` (`#3670`)、NaN 安全 TP Cross-Entropy Argmax (`#3671`) 与 `GroupedLinear` 重构 (`#3606`)

1. **`zero_tensor_dummy_wgrad` 与设备级缓存 (`#3670`)**：
   - 在 `get_dummy_wgrad` 中新增 `zero_tensor_dummy_wgrad: bool = False`（由 `BasicLinear._get_dummy_wgrad` 在无需就地清零缓冲时自动开启），调用 `torch._efficientzerotensor(shape, dtype=dtype, device=device)` 创建**零物理显存分配（0 bytes storage）**的张量，并以 `(*shape, dtype, str(device))` 为键缓存在 `_dummy_wgrads_zero` 字典中。以 32 层 `4096 × 14336` BF16 权重为例，避免了 **`3.76 GB`** 的冗余零张量分配与清零内核开销。
2. **`parallel_cross_entropy(..., return_argmax=True)` 的 NaN 安全与 Tie-Breaking 语义 (`#3671`)**：
   - Triton `tl.max(..., return_indices=True)` 对 `NaN` 的比较全部返回 `False`，导致含 `NaN` 的行返回错误索引（如 `[8, -10, NaN, -10]` 返回 `1`）。
   - `_fold_block_argmax` 将 `NaN` 检测与常规最大值分离：在保持 Loss 运行最大值不变的前提下，若某分片含 `NaN` 则记录首个 `NaN` 的全局下标并置 `run_nan = 1`；在 Tensor Parallelism 跨卡 All-Gather 时，将局部索引以 `FP32` 原始比特搭载在已有统计量张量的额外一列中同步，严格复现 `torch.argmax` 在完整未切分 Logits 上的行为（**首个 NaN 优先，否则首个全局最大值胜出**），且 Loss 与梯度位级不变。
3. **`GroupedLinear` 前反向共享状态重构 (`#3606`)**：将 split-quantize 与 grouped-tensor 两条路径的反向状态抽取为 `y.grad_fn.backward_objects`，并在延迟融合 `wgrad` 回调中仅捕获累加器标志，从而在 `dgrad` 完成后立即释放转化后的权重显存。

---

## 2. TorchTitan 优化器级 `GradientClipper` (`#5394`) 与 HF Streaming 游标延迟应用修复 (`#5501`)

1. **`GradientClipper` 可配置扩展点 (`#5394`)**：将梯度裁剪抽象为由 `Optim` 拥有的 `GradientClipper(Configurable)` 组件，使标准 `Optim` 与容错 `TorchFTOptim` 共享同一裁剪契约（返回与各 Rank 等价的 0 维裁剪前全局 L2 范数并保留非有限值）。
2. **`_HuggingFaceCursorIterator._pending_hf_state` (`#5501`)**：
   - Hugging Face `IterableDataset.load_state_dict()` 仅在**下一次读取（`next()`）**时惰性生效；若文档打包器（Document-Aware Packer）在恢复后、首次读取前再次保存 Checkpoint，`get_state()` 会把旧游标位置（如 `0` 而非恢复的 `3`）写入快照，导致第二次重启从错误文档续读（Token 序列错位为 `[14..17]` 而非 `[24..27]`）。
   - 修复方案：在 `set_state` 时将字节流保存在 `self._pending_hf_state`，直至 `__next__` 成功产出一行数据后再清空；若在首次读取前调用 `get_state()` 或读取抛出异常，均原样返回 `_pending_hf_state`。

---

## 3. LMCache 亲和性插件 RPC (`#5803`, `#5903`)、Dynamo SGLang 2s 连接池超时 (`#16974`)、NIXL UCX CPU Proxy (`#2327`) 与 Mooncake 四项存储不变量 (`#4622`, `#4688`, `#4673`, `#4755`)

1. **LMCache `server_module_affinity_call` (`#5803`) 与禁用 `pickle` (`#5903`)**：
   - 新增 `@server_module_handler(..., requires_client_affinity=True)` 与 `server_module_affinity_call` RPC，使逐层（layer-wise）KV 检索插件与同一客户端的 `store`/`retrieve` 请求在同一亲和线程上严格保序执行；
   - 启用 Ruff `TID251` 规则禁止新增 `pickle` 导入，并将 gRPC `DeviceIpcWrapper` 字段从 `pickled_payload` 命名为 `descriptor_payload`（受控 MessagePack 白名单）。
2. **Dynamo SGLang 原生 HTTP 客户端 `pool_idle_timeout(2s)` (`#16974`)**：在 `reqwest::Client` 上设置 2 秒空闲连接淘汰，早于 SGLang 默认的 5 秒 Keep-Alive 关闭窗口，彻底消除复用半关闭连接引发的偶发请求失败。
3. **NIXL `nixl::ucx::proxyEngine` (`device_proxy=true`, `#2327`)**：
   - 为每个 `(channel, peer)` 环分配一个独立的共享 UCX Worker（如 `32 channels × 4 peers = 128 workers`），支持将环均匀划分给 `4 / 32 / 64 / 128` 个主机代理线程；
   - 在提交 `UCP_ATOMIC_OP_ADD` (`ucp_atomic_op_nbx`) 前对所属 UCX Worker 执行 Fence，确保远端计数器递增不会早于同环先序 `ucp_put_nbx` 数据落地；强制 `NIXL_THREAD_SYNC_RW` 与 `ucx_error_handling_mode=none`。
4. **Mooncake 存储与权重重分片安全 (`#4622`, `#4688`, `#4673`, `#4755`)**：
   - **跨 Store 租约拒绝 (`#4622`)**：`registered()` 校验 `pre_registered_lease._owner is not self`，拒绝外部 Store 的预注册租约且不消耗原租约；
   - **驱逐排序前快照 `statx` 创建时间 (`#4688`)**：`FileUtil::SortByCreationTime` 在 `std::sort` 前一次性采集 `TimeKey{!has_birth_time, seconds, nanoseconds}`，消除比较器内实时调用 `statx` 时因文件并发删除破坏严格弱序（Strict Weak Ordering）引发的未定义行为；
   - **同周期刷盘尾部分片桶 (`#4673`)**：`MOONCAKE_OFFLOAD_FLUSH_PARTIAL_BUCKETS=1` 在当前 Offload 周期末直接刷盘未满桶，防止任务跨周期丢失；
   - **DFS 缓存分片快速路径校验 (`#4755`)**：`GetOrOpenShard` 仅在 `shard->fd >= 0 && shard->path == descriptor.file_path` 精确匹配时跳过 `lexically_normal()` 路径归一化，兼顾零重开开销与防目录穿越安全。
