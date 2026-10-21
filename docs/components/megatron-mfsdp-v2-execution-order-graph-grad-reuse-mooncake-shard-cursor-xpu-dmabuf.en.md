# Megatron-LM MFSDP v2 `ExecutionOrder` & Full-Iteration CUDA Graph Grad-Copy Reuse, TE Linear-Attention `torch.compile`, and Mooncake Shard Cursors & Exact `CUcontext` Restore

> **First-Hand Upstream Evidence (2026-10-21)**:
> - `NVIDIA/Megatron-LM` PR `#8207` (`60547a3f419bd8027bb6ffb91793c2fa99ad97ed`): `refactor(mfsdp): replace IndexedOrder with ExecutionOrder`
> - `NVIDIA/Megatron-LM` PR `#7775` (`66a6d3f71610ce209e0fbf7d16049d94f841dbff`): `Reuse full-iteration CUDA graph storage for optimizer gradient conversion`
> - `NVIDIA/Megatron-LM` PR `#8230` (`db3a6cd8a03eb550e9fa4532a0db96778bb56fc5`): `fix(mfsdp): support checkpointing for expert-parallel MoE models in MFSDP v2`
> - `NVIDIA/TransformerEngine` PR `#3685` (`5eeb62a2fcb020af9778e3d29b14b5dc5c5ad50a`): `[PyTorch] Fix full-graph torch.compile of linear attention`
> - `kvcache-ai/Mooncake` PR `#4796` (`0ed2fdfc681efb4a692bdd5af89fa8b4c8fbaef7`): `Walk tenants through shard-backed cursors in MasterService`
> - `kvcache-ai/Mooncake` PR `#4643` (`e3e9d70eea3c8ea9eb236b4c9c92f361fbbe533f`): `fix(tent): restore caller's exact CUcontext in CudaPlatform::copy()`
> - `kvcache-ai/Mooncake` PR `#4378` (`042ab8a18b0e6ccfe87782261a5556a3dd4446e4`): `feat(tent): support Intel XPU in TENT with GPUDirect RDMA`

---

## 1. Megatron-LM MFSDP v2 `ExecutionOrder`, Full-Iteration CUDA Graph Grad-Copy Reuse & EP Checkpointing (PR `#8207` / `#7775` / `#8230`)

1. **`ExecutionOrder` for Repeated Module Calls & Side-Effect-Free Prefetch Lookahead (PR `#8207`)**:
   - Replaces `IndexedOrder` (which overwrote indices when a shared `nn.Module` appeared multiple times in an execution sequence) with `ExecutionOrder`.
   - Unique modules resolve their index in $O(1)$ via `WeakKeyDictionary[T, int | None]`, whereas repeated modules are marked `None` and verified against the live cursor `self._items[self._position]` in `advance(item)`. Multi-step prefetching queries `next_item(offset)` without mutating `_position`.
2. **`FullIterationGradCopy` Shared CUDA-Graph Pool for Optimizer Gradient Conversion (PR `#7775`)**:
   - Captures `DistribOptimizer._copy_model_grads_to_main_grads` and `LayerWiseOptimizer.ToMainGrad` as a secondary CUDA graph inside the training forward/backward graph's memory pool (`pool=fb_graph.pool()`), reusing forward/backward graph scratch for FP32 `main_param.grad` conversion (`4.0 GB` VRAM saved on a `14.5 GB` graph scratch / `4.0 GB` grad shard configuration).
3. **Expert-Parallel MoE Checkpointing in MFSDP v2 (PR `#8230`)**:
   - Enables `FullyShardedOptimizer` and `fsdp_dtensor_checkpoint.py` to save and load uneven DTensor shards across heterogeneous expert-parallel and data-parallel device meshes.

---

## 2. TransformerEngine Full-Graph `torch.compile` for Linear Attention (PR `#3685`)

- Resolves and caches the `cudnn.linear_attention.ops` callable on `self._op` inside `LinearAttentionBase.__init__` (`GDN`, `GDN2`, `DeltaProduct`, `KDA`) rather than calling `importlib.import_module` inside `_run` during `forward`, eliminating `torch.compile(..., fullgraph=True)` Dynamo graph breaks (`1 -> 0`).

---

## 3. Mooncake Shard-Backed `TenantCursor`, Exact `CUcontext` Restoration & Intel XPU GPUDirect RDMA (PR `#4796` / `#4643` / `#4378`)

1. **Shard-Backed `TenantCursor` & Epoch-Preserving `ObjectRef` Re-Locking (PR `#4796`)**:
   - Replaces the dual-maintained `tenant_object_index_` in `MasterService` with `ReadCursor` / `WriteCursor` across `kNumShards`. Re-locking an inspected `route::ObjectRef` via `MetadataAccessorRW(service, tenant_id, ref)` intentionally omits `CleanupInvalidMemoryReplicas` so its KV sync does not bump the tenant epoch that `RemoveAll` compares `cleared` against.
2. **Exact `CUcontext` Save/Restore in `CudaPlatform::copy()` (PR `#4643`)**:
   - Saves the caller's exact driver context via `cuCtxGetCurrent(&caller_ctx)` before `cudaSetDevice(device_id)` and restores both `cudaSetDevice(caller_device)` and `cuCtxSetCurrent(caller_ctx)` on every exit path—preventing primary-context leaks on bare threads and preserving custom non-primary `CUcontext` bindings.
3. **Intel XPU Level Zero `dmabuf` GPUDirect RDMA in TENT (PR `#4378`)**:
   - Adds `XpuPlatform` (`zeMemAllocDevice` + `zeMemGetAllocProperties` dmabuf fd export) integrated with `ibv_reg_dmabuf_mr` in `VerbsTransport`.

---

## 4. Deterministic Verification

```bash
python experiments/megatron_mfsdp_v2_exec_order_grad_reuse_mooncake_cursor/run.py --check
```
