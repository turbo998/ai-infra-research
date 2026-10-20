# NIXL UCX VRAM Memtype Hint, Libfabric VMM Fallback, Empty Notifications & Triton CDNA5 Cluster Barrier Experiment

Deterministic Python 3.12 standard-library model verifying `ai-dynamo/nixl` commits `466ebb1ccc` (`#1536`), `1f4afe6b60` (`#2425`), `b55e79152f` (`#2455`), and `triton-lang/triton` commit `249c29ead3` (`#12342`):
- `nixlUcxContext::resolveMemoryTypeConfig` and `ucx_auto_detect_vram_hint` (`ze-device` auto-hinted only on ZE-only contexts; strict case-sensitive validation of `<auto|none|cuda|cuda-managed|rocm|ze-device>`).
- `nixlLibfabricCudaCtx::cudaUpdateCtxPtr` returns `-1` when `CU_POINTER_ATTRIBUTE_CONTEXT == nullptr` on context-independent `cuMemCreate()` VMM allocations, falling back to `cudaSetDevice(devId)` (`3/3` TRT-LLM KV cache V2 EFA transfers pass vs `3/3` `NIXL_ERR_BACKEND` failures).
- `check_remote_xfer_done` checks `if message is not None:` (`0/2` stuck empty `b""` notifications vs `2/2`).
- Triton AMD CDNA5 `TargetInfo::clusterBarrier` inserts a workgroup `s_barrier` before `ClusterBarrierArriveOp`, eliminating single-bit "Cluster Barrier Wave State" overwrite deadlocks across consecutive cluster barriers (`14/16 -> 0/16` deadlocked waves).

## Run & Verify

```powershell
python experiments\nixl_ucx_vram_hint_triton_cdna5_barrier\run.py --check
```
