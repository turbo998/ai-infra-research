# IBGDA / NCCL GIN GPU-Initiated RDMA Simulator (`ibgda_gpu_initiated_rdma`)

Deterministic CPU/offline analytical simulator grounded in `deepseek-ai/DeepEP` V2.5 (`commit b5a33eefda39485f438dc764695b71ba7b9249b6`):
- `csrc/kernels/comm/context.cpp`: `NCCL_GIN_TYPE_GDAKI`, `ginContextCount = num_allocated_qps`, `ginExclusiveContexts = true`, `NCCL_GIN_CONNECTION_RAIL` vs `FULL`.
- `deep_ep/include/deep_ep/comm/barrier.cuh`: `get_qp_mode` (`NCCL_GIN_RESOURCE_SHARING_CTA` vs `NCCL_GIN_RESOURCE_SHARING_GPU`) and `ptx::fence_acq_rel_sys()` before publishing world-team barriers.
- `deep_ep/include/deep_ep/impls/ep/dispatch.cuh`: `ncclGinOptFlagsAggregateRequests` doorbell batching and `cudaTriggerProgrammaticLaunchCompletion()` overlap.

## Run & Verify

```powershell
python experiments\ibgda_gpu_initiated_rdma\run.py
python experiments\ibgda_gpu_initiated_rdma\run.py --check
```
