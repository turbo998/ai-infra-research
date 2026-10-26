# LMCache P2P Local-DRAM 隔离与异常引用释放、NIXL UCX InlinedVector 通知与 SharedCuObjClient、Megatron GDN 图安全分配、TE 变长尾维 MXFP8 缩放、TorchTitan 训练/验证共享 CUDA Graph 池、veRL Liger TP 融合交叉熵、Mooncake 1Hz 端口恢复与 Dynamo 直通流取消（Phase 97）

> **English Version**: [lmcache-p2p-dram-put-ref-nixl-ucx-cuobj-megatron-gdn-te-mxfp8-titan-verl-mooncake-dynamo-phase97.en.md](lmcache-p2p-dram-put-ref-nixl-ucx-cuobj-megatron-gdn-te-mxfp8-titan-verl-mooncake-dynamo-phase97.en.md)
> **优先级**: `106` · **问题 ID**: `lmcache-p2p-dram-put-ref-nixl-ucx-cuobj-megatron-gdn-te-mxfp8-titan-verl-mooncake-dynamo-phase97` · **证据类型**: `implementation` + `author-benchmark` + `own-offline`

---

## 1. 核心架构问题与结论速览

针对分布式 KV 存储、RDMA/UCX 传输、分布式训练（Megatron-LM、TransformerEngine、TorchTitan、veRL）以及流式推理控制平面（Mooncake、Dynamo），Phase 97 通过 10 个零重叠上游提交确立以下架构准则：

1. **LMCache P2P 查找严格限定于本地 DRAM 段 (`LMCache#6014`, `c90e6862ab`)**：`p2p_lookup_and_lock` 显式传入 `allowed_l1_memory_tags=frozenset({(MemorySegment.DRAM, MemoryResidency.LOCAL)})` 与 `allowed_l2_memory_tags=frozenset()`，防止对等节点 P2P 查找递归穿透至远端 P2P 或本地磁盘 L2 层引发级联风暴。
2. **LMCache `StorageManager.batched_put` 异常路径引用计数完整回收 (`LMCache#5940`, `562517ad9f`)**：将后端遍历包裹在 `try/except` 中，一旦某个存储在 `batched_put` 期间抛出异常，对尚未移交所有权的全部 `memory_objs` 执行 `ref_count_down()`，将异常路径下的固定内存块泄漏从 `N` 降至 `0`。
3. **NIXL UCX 通知路径 `absl::InlinedVector` 与零拷贝序列化 (`nixl#2412`, `0165676874`)**：在 `nixlUcxBackendReqH` 中用栈内联向量替代 `std::vector`，新增 `exportMoveStr` 移动语义导出并统一 `int_handle->flush()`，在 4 KiB × 8 批负载下将 P99 Prep 延迟从 `8.0 us` 降至 `6.0 us`（`1.3333x`），P99 Post 延迟从 `12.0 us` 降至 `9.0 us`（`1.3333x`），带宽从 `4.052 GB/s` 提升至 `4.169 GB/s`。
4. **NIXL `SharedCuObjClient` 进程内跨 Agent 共享 `cuMemObj` 句柄 (`nixl#2379`, `a486deaa23`)**：当单进程内多个 LMCache/NIXL Agent 共享同一个 1 GiB 锁页内存池时，直接重复调用 `cuMemObjGetDescriptor` 会在第 2 个 Agent 上报 `rc=1 (NIXL_ERR_BACKEND)`，且任一 Agent 注销会提前释放底层映射。`SharedCuObjClient` 按 `(ptr, size)` 维护引用计数，仅首次调用驱动注册、末次调用 `cuMemObjPutDescriptor`。
5. **Megatron-LM Gated DeltaNet 循环状态 CUDA Graph 安全分配 (`Megatron-LM#8322`, `e592c042d5`)**：`torch_chunk_gated_delta_rule` 将 `torch.zeros(...).to(value)`（在默认 CPU 设备创建再跨流拷贝，破坏 CUDA Graph 捕获）替换为 `value.new_zeros((batch_size, num_heads, k_head_dim, v_head_dim))`。
6. **TransformerEngine 变长尾维 Grouped MXFP8 交错缩放偏移 (`TransformerEngine#3599`, `76ae4b0981`)**：当 `varying_last` 或 `varying_both` 形状组开启 `optimize_for_gemm=True`（Swizzled Scale 要求每行对齐到 4 个 Scale 块 = 128 元素）时，Scale 偏移不再等于 `tensor_base / 32`。内核分离 `tensor_base` 与 `tensor_base_for_scales` 独立累加，消除越界写与精度损坏。
7. **TorchTitan 训练与验证共享 CUDA Graph 内存池 (`torchtitan#5241`, `3c7b0c9d94`)**：通过 `_CUDAGraphGradientState` 管理梯度开关，并在 Eager 预热后将 Validation Graph 捕获进与 Training Graph 共享的内存池。在 Muse Glimmer 30B 上，Train+Eval 双图峰值显存为 `61.530 GiB`，与无图 Eager Eval（`61.531 GiB`）持平，额外显存池开销为 `0.000 GiB`。
8. **veRL 集成 Liger-Kernel 0.8.4 TP 融合线性交叉熵 (`verl#7945`, `5b550b0c3d`)**：在 Megatron 引擎中支持 `impl_backend: "liger"`（含 SM90 `#1422` 与 SM100 Blackwell `#1458` 修复），避免物化 `[B*S, V/TP]` FP32 Logits（在 `8192` Tokens、`V=152064, TP=8` 下单卡节省 `594.0 MiB` 峰值显存）。
9. **Mooncake Classic RDMA 丢失 `PORT_ACTIVE` 事件后的 1 Hz 轮询自愈 (`Mooncake#4715`, `8c19b7286c`)**：`WorkerPool` 记录导致上下文禁用的异步事件类型；当且仅当因 `IBV_EVENT_PORT_ERR` 禁用时，以 `1 Hz` 轮询 `ibv_query_port` 并在端口恢复 `IBV_PORT_ACTIVE` 时触发延迟 GID 探测，避免单次事件丢失导致网卡轨道永久离线。
10. **Dynamo SGLang 流式生成消除逐 Chunk 取消任务竞速 (`dynamo#16351`, `a246c3f227`)**：不再对每个输出 Chunk 构造 `asyncio.wait({next_item_task, cancel_task})` 竞速集合，改为后台取消句柄直接停止底层流并以 `_STREAM_END` 哨兵收尾，在 512-Chunk 流上将异步等待任务调度开销降低 `1024.0x`。

---

## 2. 一手源码证据与定量指标对比

| 编号 | 证据 ID / 仓库提交 | 关键实现路径与硬件/系统契约 | 定量收益 / 正确性不变量 |
| :--- | :--- | :--- | :--- |
| 21 | `lmcache-limit-p2p-controller-lookup-local-dram-segment-pr6014-2026` (`c90e6862ab`) | `p2p_lookup_and_lock`: 仅允许 `(DRAM, LOCAL)` L1 标签 | 阻断 P2P 递归穿透远端/磁盘 L2 存储层 |
| 22 | `lmcache-release-storagemanager-refs-batched-put-error-pr5940-2026` (`562517ad9f`) | `StorageManager.batched_put`: 异常路径释放未提交 `memory_objs` | 后端报错时锁页内存块引用泄漏从 `N -> 0` |
| 23 | `nixl-ucx-improve-notification-performance-pr2412-2026` (`0165676874`) | `nixlUcxBackendReqH`: `absl::InlinedVector` + `exportMoveStr` | 4 KiB×8 P99 Prep `8->6 us` (`1.333x`)，Post `12->9 us` (`1.333x`) |
| 24 | `nixl-obj-share-cuobject-registration-across-agents-pr2379-2026` (`a486deaa23`) | `SharedCuObjClient`: 进程内按 `(ptr, size)` 引用计数共享句柄 | 4 Agents 共享 1 GiB 池仅 `1` 次驱动注册，`0` 次提前解绑 |
| 25 | `megatron-fix-torch-gdn-recurrent-state-alloc-cudagraph-pr8322-2026` (`e592c042d5`) | `torch_chunk_gated_delta_rule`: `value.new_zeros(...)` 原地设备分配 | 消除 GDN 循环状态在 CUDA Graph 捕获期间的 H2D 拷贝报错 |
| 26 | `te-fix-grouped-mxfp8-quant-varying-last-dim-swizzled-pr3599-2026` (`76ae4b0981`) | `group_quantize_mxfp8_kernel`: 独立累加 `tensor_base_for_scales` | 变长尾维 Swizzled MXFP8 缩放偏移精确对齐（如 `t1_scale_offset=1024`） |
| 27 | `torchtitan-share-cudagraph-memory-training-validation-pr5241-2026` (`3c7b0c9d94`) | `cudagraph.py`: 训练与验证共享 Graph Pool + Eager Warmup | Muse Glimmer 30B 峰值显存 `61.530 GiB`（相比 Eager `0.000 GiB` 额外开销） |
| 28 | `verl-megatron-add-liger-tp-fused-cross-entropy-pr7945-2026` (`5b550b0c3d`) | `liger_tp_ce.py`: `liger-kernel==0.8.4` TP 融合线性交叉熵 | `8192` Tokens、`V=152064, TP=8` 免物化 `594.0 MiB` FP32 Logits |
| 29 | `mooncake-recover-classic-rdma-contexts-port-active-missed-pr4715-2026` (`8c19b7286c`) | `worker_pool.cpp`: 针对 `PORT_ERR` 启用 `1 Hz` `ibv_query_port` 轮询 | 丢失 `PORT_ACTIVE` 事件后约 `1 s` 自动触发 GID 刷新并恢复轨道 |
| 30 | `dynamo-sglang-stop-racing-stream-chunk-cancellation-task-pr16351-2026` (`a246c3f227`) | `cancellation.py`: 移除逐 Chunk `asyncio.wait` 竞速，引入 `_STREAM_END` | 512-Chunk 输出流异步等待任务数从 `1024` 降至 `1`（`1024.0x` 缩减） |

---

## 3. 配套离线可复现实验

- **实验路径**: `experiments/lmcache_p2p_dram_put_ref_nixl_ucx_cuobj_megatron_gdn_te_mxfp8_titan_verl_mooncake_dynamo_phase97/run.py`
- **校验命令**: `python experiments\lmcache_p2p_dram_put_ref_nixl_ucx_cuobj_megatron_gdn_te_mxfp8_titan_verl_mooncake_dynamo_phase97\run.py --check`
- **单测覆盖**: `tests/test_phase97_experiments.py`
