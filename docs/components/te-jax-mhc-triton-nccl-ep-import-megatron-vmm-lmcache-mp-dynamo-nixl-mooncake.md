# TransformerEngine JAX Triton mHC 与 NCCL EP 句柄重建、Megatron-LM VMM 并发编译、LMCache MP 插件钩子及 Dynamo/NIXL/Mooncake 传输深挖

> **一手源码溯源** (`2026-10-14 ~ 2026-10-21`)：
> - `NVIDIA/TransformerEngine@53d1a3c71a8f54cbfcdbc5e9f548eba040e42523` (`#3680`): *[JAX][Triton] Integrate mHC kernels (#3680)*
> - `NVIDIA/TransformerEngine@6ca7b5aa88ef2e1497f7a896ca1d9bd80442222e` (`#3639`): *[JAX] Reconstruct ncclEp Handle via ncclEpImportHandle (#3639)*
> - `NVIDIA/Megatron-LM@6c8f14a3ecbd8ac4095c5d795f2661732381d702` (`#7494`): *fix: resolve concurrent build race in vmm_symm_allocator (#7494)*
> - `LMCache/LMCache@151c6c4f415ca3040c3370c673bb2b3a0ba4646b` (`#5805`): *feat(mp): add plugin transfer extension hooks (#5805)*
> - `LMCache/LMCache@4159596c25562ba5af0c4df66d629cc253aceff1` (`#5946`): *refactor: move health_monitor under lmcache/inprocess/v1/ (#5946)*
> - `ai-dynamo/dynamo@4d31cd2f18804311bd129e9b8dc580b3d4136388` (`#16742`): *fix: release GIL during FPM relay and publisher construction (#16742)*
> - `ai-dynamo/dynamo@7929f5370eef7f15095b16f0dd41012b920c334c` (`#13744`): *feat(operator): add lossless DGDR v1beta1 <-> v1beta2 conversion (#13744)*
> - `ai-dynamo/nixl@03d9b74648cc06daef8aa71f15a53c5a98be1319` (`#2479`): *feat(tracing): add transfer request and execution tracing (#2479)*
> - `kvcache-ai/Mooncake@4137ac00260da1690bed7151af98bd2cc63f4b7d` (`#4756`): *[TE] Back Ascend host segments with mmap so A5 RoCE can drive them (#4756)*

---

## 1. TransformerEngine JAX Triton 流形超连接 (`mHC`, `#3680`) 与无状态 NCCL EP 句柄重建 (`#3639`)

### 1.1 JAX Triton `mHC` 五阶段融合内核与确定性守护 (`#3680`)

流形超连接（Manifold Hyper-Connections, `mHC`）在残差干路上维护 $n=4$ 条并行残差流 (`N_STREAMS = 4`)，每步需生成 $2n + n^2 = 24$ 个混合系数 (`N_MIX = 24`，在张量末维零填充至 `H_PADDED_DIM = 32` 以满足 Triton 2 幂次块对齐)，并通过双随机矩阵投影稳定残差流范数。`#3680`（`+3,115 / -12` 行）在 `transformer_engine/jax/triton_extensions/mhc.py` 与 `flax/mhc.py` 中实现了完整的 JAX 原生 Triton 算子栈：
1. **五阶段前反向融合管线**：
   - `mhc_projection`：对展平输入 $X \in \mathbb{R}^{B \times nC}$ 计算 $\tilde{H} = X \Phi$ 及行平方和 $\|X\|_2^2$；当 `has_norm_weight=True` 或（`use_tf32=False` 且输入/`\Phi` 均为 BF16）时，自动将 $X\odot w$ 与 $\Phi$ 提升至 FP32，并把 `tl.dot` 精度从 `"ieee"` 切换为 **`"tf32x3"`**（3 次 TF32 外积拟真 FP32 精度）。
   - `mhc_scale`：融合 RMSNorm 倒数缩放、动态加权偏置与 Sigmoid 激活，输出 `h_pre`、`h_post = 2 * sigmoid(...)` 与 $4\times 4$ `h_res`。
   - `mhc_sinkhorn`：在 $4\times 4$ 寄存器数组上执行 **20 轮对数域 Sinkhorn-Knopp 迭代** (`f = -logsumexp(H + g, axis=1)`, `g = -logsumexp(H + f, axis=0)`)，将双随机行/列和误差收敛至 **`1.11e-7`**；反向传播前插入 `tl.debug_barrier()` 消除读后写 (`RAW`) 共享内存竞态。
   - `mhc_aggregate` 与 `mhc_expand_combine`：完成多流聚合输入归一化及残差流双随机混合 `x_next = H_res @ x + h_post * y`。
2. **分布式与确定性保证**：通过 `custom_partitioning` 与 Shardy `SdyShardingRule` 沿 Token 维 (`batch`) 零通信切分，并在 `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` 时显式禁止 Split-K/Split-M 原子累加 (`atomic_add`)。

### 1.2 基于 `ncclEpImportHandle` 的无状态 JAX EP 句柄重建 (`#3639`)

在 JAX `jax.lax.scan` 堆叠 MoE 层时，XLA 会在层间复用或迁移临时缓冲区，导致此前以 `handle_mem` 设备指针地址为 Key 的 C++ `EpHandleCache` 发生跨层指针别名（命中过期层的 `ncclEpHandle_t`）。`#3639` 在 C++ 侧新增 `nvte_ep_{dispatch,combine,dispatch_bwd,combine_bwd}_v2`：每次调用直接读取 `handle_mem` 中的序列化字节与 `EpLayerConfig`（`nccl_comm_ptr`, `num_Action_tokens`, `hidden_size`, `top_k`, `num_experts`, `data_type`），调用 `ncclEpImportHandle` 重建 `ScopedHandle(handle, /*owned=*/true)` 并在算子返回时立即调用 `ncclEpDestroyHandle` 销毁，同时在 C++ 入口强校验 `topk_idx_dtype` 必须为 `INT32` 或 `INT64`。PyTorch 侧则继续保留带 `NVTE_EP_HANDLE_CACHE_SIZE` 上限的 LRU 缓存。

---

## 2. Megatron-LM VMM 对称分配器并发编译修复 (`#7494`) 与 LMCache MP 插件钩子 (`#5805`, `#5946`)

1. **Megatron-LM 多卡并发 JIT 编译竞态修复与 IMEX 优雅降级 (`#7494`)**：
   - 旧版 `vmm_symm_allocator.py` 调用 `torch.utils.cpp_extension.load_inline`，该函数在获取 PyTorch 编译文件锁**之前**就会直接覆写共享 `build_directory` 下的 `main.cpp`，导致单机 8 卡同时初始化时高频出现截断源码或链接失败。
   - `#7494` 将 C++ 源码固化为包内静态文件 `megatron/core/allocator/csrc/vmm_symm_allocator.cpp`，改用先加锁再编译的 `cpp_extension.load`，同时缓存 `_allocator` 与 `_build_error`。
   - 当 GPU 驱动报告 `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED=1` 但集群未配置 IMEX daemon 时，`cuMemCreate` 首次尝试 `POSIX_FD | FABRIC` (`0x9`) 失败后自动剥离 `CU_MEM_HANDLE_TYPE_FABRIC` (`0x8`) 并以纯 `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` (`0x1`) 重试成功。
2. **LMCache 多进程 (`mp`) 传输插件扩展钩子 (`#5805`, `#5946`)**：
   - 新增 `create_lmcache_mp_worker_adapter`、`_create_transfer_context`、`_create_cache_context`、`BaseCacheContext.transfer_object_group` 及 `CUDA_CacheContext._make_temp_buffer(..., full_sw_kv=full_sw_kv)` 五个扩展钩子；在 `skip_first_n_tokens > 0` 的 `retrieve` 路径中用 `[None] * skip` 头部填充对齐 `memory_objs` 与 `slot_mapping_slices`；并将健康检查自动发现命名空间迁移至 `lmcache.inprocess.v1.health_monitor.checks`。

---

## 3. Dynamo FPM GIL 释放与 DGDR `v1beta2` 无损转换 (`#16742`, `#13744`)、NIXL 请求级追踪 (`#2479`) 与 Mooncake 昇腾 A5 `memfd` 映射 (`#4756`)

| 系统与 PR | 核心架构痛点 | 工程机制与确定性不变量 |
|---|---|---|
| **Dynamo `#16742`** (`4d31cd2f18`) | `FpmEventRelay::new` 与 `FpmDirectPublisher::new` 在持有 Python GIL 时调用 `rt.block_on(...)`，若 Tokio 工作线程回调 Python（日志/GC/信号）即死锁 | 将 ZMQ socket 绑定与后台接收循环移入 `py.allow_threads(\|\| ...)`，先解析 `py_callback.py()` 再释放 GIL |
| **Dynamo `#13744`** (`7929f5370e`) | `DynamoGraphDeploymentRequest` `v1beta2` 破坏性重构无法通过 `v1beta1` Hub 与 `v1alpha1` Spoke 无损往返 | 在 `v1beta1` 与 `v1alpha1` Spec/Status 中嵌入与原生字段互斥的 `v1beta2` 强类型信封，实现零 Annotation 膨胀的无损转换 |
| **NIXL `#2479`** (`03d9b74648`) | 多后端分片或重投递 (`postXferReq`) 时缺少跨 Span 关联 ID 与执行轮次计数 | 新增非零 64-bit 十进制 `nixl.xfer_request_id`（抽中 `0` 自动重抽）与仅在重投递/通知校验通过后递增的 `nixl.xfer_execution`，并在 `checkXferTelemetry` 完成瞬间发出 `nixl::xfer.complete` |
| **Mooncake `#4756`** (`4137ac0026`) | 昇腾 A5 在 `roce:host` 驱动 Host 段时不支持 `aclrtMallocHost`，且跨进程导入的 `memfd` 被识别为 `UNREGISTERED` 退化为逐 Slice 拷贝 | 新增 `host_memfd.h`（`2 MiB` THP 对齐 `memfd` / 私有 `mmap`）、`SharedHostMapper`（跨进程虚拟地址重定位 `local_base + (dest_addr - owner_base)`，越界 Slice 拒绝并回退 ADXL），并将 `UNREGISTERED` 映射归类为 Host 内存以启用 `aclrtMemcpyBatch` |
