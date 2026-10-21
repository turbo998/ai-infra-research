# Megatron-LM GDN 单次更新 CUDA Graph 重放、TorchTitan 优化器级 GradientClipper 与 LoRA TP SPMD 校验、LMCache L2 预取动态水位线与 VMM 填充池 IPC 及 Mooncake / NIXL / Dynamo 传输加固

> **一手上游来源**:
> - `NVIDIA/Megatron-LM@29e1a7b6697c61d184c95726865d417fc51b8574` (`fix(inference): model GDN prefill CUDA graph as single-update GDP (#8353)`, 2026-10-21)
> - `pytorch/torchtitan@9698a733623ca997ef824a03cb1f44706ee631e7` (`Move gradient clipping into OptimizersContainer as a GradientClipper component (#5394)`, 2026-10-21)
> - `pytorch/torchtitan@a242475253433725f10f39c18f5ccbfdf188e097` (`Fix LoRA compatibility with SPMD type checking under TP (#5390)`, 2026-10-21)
> - `LMCache/LMCache@a6accee6d84c77da8c8b3e3a68574b69247e9684` (`perf(mp): replace fixed max-in-flight with dynamic staging memory watermark for L2 prefetch (#5889)`, 2026-10-21)
> - `LMCache/LMCache@21aa6ac06c83e1bad460cc231c5168a6bc7defe6` (`fix: register dim-0-padded KV pools in VmmCudaIPCWrapper (#5855)`, 2026-10-21)
> - `LMCache/LMCache@45486917f13aa454a22435aa03e2dc28efda47f5` (`fix(c_ops): ensure primary CUDA context on current thread in stream_ops (#5929)`, 2026-10-21)
> - `kvcache-ai/Mooncake@7a0585567458a4aba6a58115888c54b7fd1a44e1` (`Allow CPU clients to select peer GPU TCP memory (#4410)`, 2026-10-20)
> - `ai-dynamo/nixl@03d9b74648cc06daef8aa71f15a53c5a98be1319` (`feat(telemetry): add xfer request ID and execution counter to transfer requests (#2479)`, 2026-10-21)
> - `ai-dynamo/dynamo@5004f3fa8093d99de68a8054ea238ea534d9daf4` (`fix(multimodal): propagate errors in count_all_image_tokens (#16947)`, 2026-10-20)

---

## 1. Megatron-LM GDN 单次更新 CUDA Graph 重放与 TorchTitan 训练核心契约 (`Megatron-LM@29e1a7b669`, `torchtitan@9698a73362`, `torchtitan@a242475253`)

### 1.1 Megatron-LM Gated DeltaNet (GDN) Prefill CUDA Graph 静态缓冲区建模 (`Megatron-LM@29e1a7b669`)
在 Megatron-LM 混合推理上下文中，GDN Prefill 通过 `chunk_gated_delta_rule` 以固定块大小 `GDP_CHUNK_SIZE = 64` 切分序列。对于 CUDA Graph 捕获与重放，底层内核必须从持久化的静态设备张量 `cu_seqlens` 读取块偏移：
- 此前代码未显式建模 `cu_seqlens` 的形状，而 GDN Prefill 在数学上等价于每个 `64-token` 块拥有 `num_householder = 1` 个 Householder 反射向量的广义 Delta 积（Generalized Delta Product, GDP）；
- 提交 `29e1a7b669` 将 GDN Prefill CUDA Graph 正式建模为 `num_householder = 1` 的单次更新 GDP：在 `max_tokens = 4096` 下预分配 `4096 // 64 = 64` 个持久化块描述符（`cu_seqlens` 长度 `65`），彻底消除 Graph Replay 期间的动态形状失配。

### 1.2 TorchTitan 优化器级 `GradientClipper` 与 LoRA TP SPMD 类型推导 (`torchtitan@9698a73362`, `torchtitan@a242475253`)
- **优化器内聚梯度裁剪 (`torchtitan@9698a73362`)**：将 `clip_grad_norm_` 封装为 `OptimizersContainer` 内部的一等 `GradientClipper` 组件，并在 `OptimizersContainer.step()` 的 `_update` 钩子中自动调用并记录 `grad_norm` 标量。这不仅消除了 `Trainer.forward_backward_step` 的显式裁剪耦合，更确保 `FTManager`（TorchFT 容错训练）在替换优化器 step 钩子时自动继承完全一致的跨 DP/PP/EP 分片梯度范数规约语义。
- **LoRA 在 TP 下的 SPMD 类型校验 (`torchtitan@a242475253`)**：当启用 Tensor Parallel (`RowwiseParallel` 产生产物 `Partial` 输出) 与 LoRA 时，基座线性层输出 `Partial` 进入 `lora_a`（`RowwiseParallel -> Partial`），再经激活量 `Replicate` 的 `lora_b` 权重相乘。原 `_LinearFunction.spmd_typecheck` 仅接受 `Replicate / Shard(0)` 输入，遇到 `Partial @ Replicate` 直接抛出类型错误；修复后将 `_LinearFunction.spmd_typecheck` 委托给 `einsum("...k,nk->...n")` 线性性推导规则，正确推导 `Partial @ Replicate -> Partial`，随后在基座加法处完成延迟 All-Reduce。

---

## 2. LMCache L2 预取动态暂存内存水位线、VMM 填充池 IPC 与主上下文绑定 (`LMCache@a6accee6d8`, `LMCache@21aa6ac06c`, `LMCache@45486917f1`)

### 2.1 `--l2-prefetch-inflight-watermark` 动态暂存内存水位线 (`LMCache@a6accee6d8`)
此前 `PrefetchController` 使用固定并发上限 `max_in_flight = 8` 限制同时处于 `L2_LOADING` 状态的预取请求数：
- 当面对小粒度（如 `8 MiB`）预取请求时，`max_in_flight = 8` 仅利用了 `64 MiB` 锁页暂存池（总容量 `1024 MiB` 的 `6.25%`）；而面对大粒度（如 `128 MiB`）请求时，若盲目调大固定并发数又会耗尽暂存池导致前台 Store 操作 OOM；
- 提交 `a6accee6d8` 用 `--l2-prefetch-inflight-watermark`（默认 `0.5`，即 `512 MiB`）替换固定计数：只要 `inflight_bytes + req_bytes <= pool_capacity * watermark`（或 `inflight_bytes == 0` 的单超大请求无死锁保底，只要 `req_bytes <= pool_capacity` 即放行），即可接纳预取。对 `8 MiB` 小请求并发度提升 **4.0x（`32 vs 8`）**，同时对 `128 MiB` 大请求严格封顶在 `4` 个（`512 MiB`），预留 `50%` 锁页内存给前台写入。

### 2.2 `VmmCudaIPCWrapper` Dim-0 填充池跨度修正与 `stream_ops` 主上下文绑定
- **VMM Dim-0 填充池 IPC (`LMCache@21aa6ac06c`)**：vLLM 跨层统一 KV Cache（Cross-Layer Uniform KV Cache）会对 KV 池的第 0 维做对齐填充（例如逻辑形状 `(2, 8, 64)`，但 `stride(0) = 1280 > 8 * 64 = 512`）。此前 `VmmCudaIPCWrapper` 仅按 `numel() * element_size() = 1024 * 2 = 2048 B` 查询 `cuMemGetAddressRange`，少算了末尾切片偏移，导致跨进程 IPC 映射越界；修复后按完整内存跨度 `1 + sum((shape[i] - 1) * stride[i]) = 1792 elements = 3584 B` 迭代物理分配块。
- **`stream_ops` 线程级主上下文绑定 (`LMCache@45486917f1`)**：`cuStreamWaitValue64` 与 `cuStreamWriteValue64` 是 CUDA Driver API，要求当前调用线程栈顶存在活跃的 `CUcontext`。当线程池工作线程首次调用 `stream_ops` 且未做過 Runtime API 调用时，直接返回 `CUDA_ERROR_INVALID_CONTEXT (201)`；修复后在入口处调用 `cudaGetDevice(&dev)` + `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent` 幂等绑定主上下文。

---

## 3. Mooncake 混合 TCP 传输、NIXL 传输追踪与 Dynamo 多模态 Token 异常传播

| 组件与上游提交 | 核心缺陷或能力缺口 | 架构修复与量化效果 |
| :--- | :--- | :--- |
| **Mooncake 混合 CPU↔GPU TCP 传输** (`Mooncake@7a05855674`) | 纯 CPU 客户端因未配置本地 GPU，其 `TransferEngine` 将 `caps.dram_to_gpu` 与 `caps.gpu_to_dram` 设为 `false`，拒绝选择远端 Peer 暴露的 `cuda:*` GPU TCP 内存段。 | 在 TCP 传输启用时对客户端无条件开启 `caps.dram_to_gpu = true` 与 `caps.gpu_to_dram = true`（因为远端 Peer 负责在其本地 GPU 执行 `cudaMemcpy`），打通纯 CPU 节点直读/直写远端 GPU KV 池。 |
| **NIXL 64-bit 请求 ID 与重发计数** (`nixl@03d9b74648`) | `postXferReq` 重用同一 `nixlXferReqH` 句柄重发时，遥测日志无法区分不同请求实例与第几次重发。 | 在 `nixlXferReqH` 增加进程内原子自增 64 位无符号 ID `nixl.xfer_request_id` 与每次 `postXferReq` 递增的 `nixl.xfer_execution`（`1 -> 2 -> 3`）。 |
| **Dynamo 多模态图片 Token 异常传播** (`dynamo@5004f3fa80`) | `count_all_image_tokens` 遇到损坏图片或不支持的 `grid_thw` 时吞掉错误并返回 `0`，将 `image_tokens = Some(0)` 喂给 KV 路由器造成严重欠估。 | 将签名改为 `Result<usize>` 并通过 `?` 向上传播，失败时记 `warn!` 并将 `image_tokens` 置为 `None`，杜绝伪造 `0` Token 误导调度器。 |

---

## 4. 本地确定性复现实验 (`experiments/megatron-gdn-cudagraph-titan-gradclip-lmcache-prefetch-vmm-mooncake-nixl/run.py`)

```bash
python experiments/megatron-gdn-cudagraph-titan-gradclip-lmcache-prefetch-vmm-mooncake-nixl/run.py --check
```
