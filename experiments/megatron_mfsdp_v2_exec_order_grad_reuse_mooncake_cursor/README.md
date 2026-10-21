# Megatron-LM MFSDP v2 `ExecutionOrder`, Full-Iteration CUDA Graph Grad-Copy Reuse & Mooncake Shard-Backed `TenantCursor`

Deterministic standard-library verification of:
1. `NVIDIA/Megatron-LM` PR `#8207` (`60547a3f41`), PR `#7775` (`66a6d3f716`), & PR `#8230` (`db3a6cd8a0`): MFSDP v2 `ExecutionOrder` supporting repeated module calls (`WeakKeyDictionary` with `None` sentinel for repeated modules + non-advancing `next_item(offset)`), `FullIterationGradCopy` sharing the forward/backward CUDA-graph memory pool for `DistribOptimizer`/`LayerWiseOptimizer` gradient conversion, and expert-parallel `FullyShardedOptimizer` DTensor checkpointing.
2. `NVIDIA/TransformerEngine` PR `#3685` (`5eeb62a2fc`): Constructor-time `self._op = self._resolve_op()` caching for `GDN`/`GDN2`/`DeltaProduct`/`KDA` linear attention to eliminate `torch.compile` (`fullgraph=True`) Dynamo graph breaks.
3. `kvcache-ai/Mooncake` PR `#4796` (`0ed2fdfc68`), PR `#4643` (`e3e9d70eea`), & PR `#4378` (`042ab8a18b`): `MasterService` shard-backed `TenantCursor<LockMode>` without dual-index maintenance or epoch-bumping `ref` re-locks, `CudaPlatform::copy()` exact `CUcontext` restoration (`cuCtxGetCurrent`/`cuCtxSetCurrent`), and Intel XPU `zeMemGetAllocProperties` + `ibv_reg_dmabuf_mr` GPUDirect RDMA.

## Usage

```bash
python experiments/megatron_mfsdp_v2_exec_order_grad_reuse_mooncake_cursor/run.py --check
```
