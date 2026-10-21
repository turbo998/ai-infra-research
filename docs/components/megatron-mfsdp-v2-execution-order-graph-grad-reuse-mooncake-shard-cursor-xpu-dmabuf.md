# Megatron-LM MFSDP v2 `ExecutionOrder` 与全迭代 CUDA Graph 梯度转换内存池复用、TE 线性注意力编译及 Mooncake 分片游标与 `CUcontext` 精确恢复深挖

> **一手上游证据（2026-10-21）**：
> - `NVIDIA/Megatron-LM` PR `#8207` (`60547a3f419bd8027bb6ffb91793c2fa99ad97ed`)：`refactor(mfsdp): replace IndexedOrder with ExecutionOrder`
> - `NVIDIA/Megatron-LM` PR `#7775` (`66a6d3f71610ce209e0fbf7d16049d94f841dbff`)：`Reuse full-iteration CUDA graph storage for optimizer gradient conversion`
> - `NVIDIA/Megatron-LM` PR `#8230` (`db3a6cd8a03eb550e9fa4532a0db96778bb56fc5`)：`fix(mfsdp): support checkpointing for expert-parallel MoE models in MFSDP v2`
> - `NVIDIA/TransformerEngine` PR `#3685` (`5eeb62a2fcb020af9778e3d29b14b5dc5c5ad50a`)：`[PyTorch] Fix full-graph torch.compile of linear attention`
> - `kvcache-ai/Mooncake` PR `#4796` (`0ed2fdfc681efb4a692bdd5af89fa8b4c8fbaef7`)：`Walk tenants through shard-backed cursors in MasterService`
> - `kvcache-ai/Mooncake` PR `#4643` (`e3e9d70eea3c8ea9eb236b4c9c92f361fbbe533f`)：`fix(tent): restore caller's exact CUcontext in CudaPlatform::copy()`
> - `kvcache-ai/Mooncake` PR `#4378` (`042ab8a18b0e6ccfe87782261a5556a3dd4446e4`)：`feat(tent): support Intel XPU in TENT with GPUDirect RDMA`

---

## 1. Megatron-LM MFSDP v2 `ExecutionOrder`、全迭代 CUDA Graph 梯度转换内存复用与 EP Checkpoint（PR `#8207` / `#7775` / `#8230`）

### 1.1 `ExecutionOrder`：支持重复模块调用与无副作用前瞻预取（PR `#8207`）
在 Megatron-FSDP v2（`megatron_fsdp/experimental`）中，此前的 `IndexedOrder` 使用单一字典记录每个 `nn.Module` 的位置。当模型存在跨步重复调用的共享模块（如递归层或共享投影 `shared_mlp` 在 `["embed", "shared_mlp", "layer1", "shared_mlp", "head"]` 中出现两次）时，第二次插入会覆盖第一次的索引，导致第一次执行 `shared_mlp` 后的预取直接跳过 `layer1` 指向 `head`。

PR `#8207` 引入 `ExecutionOrder`（`execution_order.py`）：
- **唯一模块 O(1) 定位 + 重复模块游标校验**：`self._unique_index: WeakKeyDictionary[T, int | None]` 对仅出现一次的模块保存其确切索引，对出现多次的模块标记为 `None`。
- **`advance(item)` 与 `next_item(offset)` 解耦**：调用 `advance(item)` 时，若模块唯一则直接定位并推进游标 `self._position = (index + 1) % len`；若模块重复（`index is None`），则严格校验当前游标 `self._items[self._position]() is item`（若失序则立即抛出 `RuntimeError("FSDP module calls diverged from the expected prefetch order.")`）。随后的多步预取通过 `next_item(offset=1, 2, ...)` 纯函数前瞻，绝不污染游标状态。

### 1.2 `FullIterationGradCopy`：全迭代 CUDA Graph 激活临时池与 FP32 梯度转换池共享（PR `#7775`）
在启用 `--cuda-graph-impl full_iteration`（但未开启 `--optimizer-cuda-graph`）时，`DistribOptimizer._copy_model_grads_to_main_grads` 与 `LayerWiseOptimizer.ToMainGrad` 会在每次迭代末尾分配或转换 `main_param.grad`（FP32），在图内存池之外产生数 GB 的额外显存峰值。

PR `#7775` 在 `megatron/core/full_cuda_graph.py` 中新增 `FullIterationGradCopy`：
- 当检测到 `FullCudaGraphWrapper.cuda_graph.get("training")` 存在时，在训练图的共享捕获流（`get_shared_capture_stream()`）与**同一内存池（`pool=fb_graph.pool()`）**中将 `copy_fn` 捕获为独立的子 CUDA Graph。
- 由于前反向传播已完成，梯度转换生成的 `main_param.grad` 张量可以直接复用前反向 CUDA Graph 释放的激活临时缓冲（scratch memory），在 `14.5 GB` 图临时区与 `4.0 GB` FP32 梯度转换区场景下直接节省 **`4.0 GB`** 峰值显存，并在 `reset_cuda_graph()` 时自动感知 `fb_graph` 身份变化重建。

### 1.3 MFSDP v2 专家并行（Expert-Parallel）MoE Checkpoint 修复（PR `#8230`）
- 在 `FullyShardedOptimizer` 与 `fsdp_dtensor_checkpoint.py` 中，修复了专家并行（EP）MoE 参数与非 EP 参数拥有不同 `DeviceMesh` 时的状态字典保存/加载，并兼容 TE `< 2.18.0` 过滤空本地分片时的完整参数 FQN 映射。

---

## 2. TransformerEngine 线性注意力 `torch.compile(fullgraph=True)` 零图断裂修复（PR `#3685`）

在 `NVIDIA/TransformerEngine` 的实验性线性注意力模块（`GDN`、`GDN2`、`DeltaProduct`、`KDA`）中，此前每次 `forward -> _run` 都会调用 `@lru_cache` 包装的 `_import_linear_attention_op`，内部包含 `importlib.import_module("cudnn.linear_attention.ops")`。在 PyTorch Dynamo `torch.compile(..., fullgraph=True)` 追踪下，追踪 `importlib` 会直接引发 `Unsupported` 图断裂（graph break）。

PR `#3685` 在 `LinearAttentionBase.__init__` 构造阶段预先解析并缓存 `self._op = self._resolve_op()`（若构造时环境未安装 `nvidia-cudnn-frontend` 则静默跳过，推迟到首次 `_run` 时报错），使得 `torch.compile` 在 `forward` 期间仅看到普通属性读取 `self._op(...)`，将 Dynamo 图断裂次数从 **`1` 降至 `0`**。

---

## 3. Mooncake `MasterService` 分片游标、`CudaPlatform::copy` 精确 `CUcontext` 恢复与 Intel XPU RDMA（PR `#4796` / `#4643` / `#4378`）

1. **`MasterService` 分片支持游标 `TenantCursor<LockMode>` 与租户 Epoch 保护（PR `#4796`）**：
   - 消除了需要双写维护的全局 `tenant_object_index_`，改为通过 `ReadCursor(tenant_id)` / `WriteCursor(tenant_id)` 直接遍历 `kNumShards` 个元数据分片。
   - **关键并发不变量**：当游标通过 `route::ObjectRef` 重新加写锁（`MetadataAccessorRW(service, tenant_id, ref)`）执行 `RemoveTenantObjects` 时，**禁止**在该构造函数内触发 `CleanupInvalidMemoryReplicas`——因为无效副本清理的 KV 同步会递增租户 epoch，从而导致 `RemoveAll` 误判租户在清理期间发生了并发写入（`cleared` 标志被错误置否）。
2. **TENT `CudaPlatform::copy()` 精确 `CUcontext` 身份恢复（PR `#4643`）**：
   - `cudaMemcpyAsync` 要求当前线程绑定目标缓冲区的设备上下文，但仅用 `cudaGetDevice` / `cudaSetDevice` 备份恢复存在两大缺陷：① 若调用线程原本处于无上下文（bare thread）状态，`cudaGetDevice` 会隐式在 GPU 0 上创建主上下文，且退出后无法清空上下文；② 若调用线程绑定的是自定义非主 `CUcontext`，`cudaSetDevice` 会将其错误替换为设备的主上下文（primary context）。
   - PR `#4643` 通过 Driver API `cuCtxGetCurrent(&caller_ctx)` 保存确切上下文句柄，并在所有退出路径上先调用 `cudaSetDevice(caller_device)` 恢复 Runtime 绑定，再调用 `cuCtxSetCurrent(caller_ctx)` 精确还原自定义 `CUcontext` 或空上下文（`nullptr`）。
3. **Intel XPU Level Zero `dmabuf` GPUDirect RDMA 支持（PR `#4378`）**：
   - 新增 `XpuPlatform`，通过 `zeMemAllocDevice` 分配显存、`zeMemGetAllocProperties` 导出 `ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD`（dmabuf fd），并在 `VerbsTransport` 中对接 `ibv_reg_dmabuf_mr` 实现 Intel GPU 免主机中转的 GPUDirect RDMA。

---

## 4. 确定性复现实验验证

```bash
python experiments/megatron_mfsdp_v2_exec_order_grad_reuse_mooncake_cursor/run.py --check
```
