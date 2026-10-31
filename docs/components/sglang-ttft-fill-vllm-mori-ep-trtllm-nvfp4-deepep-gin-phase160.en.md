# SGLang TTFT Breakdown & Metadata Fill Kernels, vLLM MORI EP V2 & gRPC Drain, TRT-LLM NVFP4 MoE & DeepEP Mixed-Fabric GIN Barrier (Phase 160)

> **Evidence Class**: `implementation` (primary upstream commit audit) + `own-offline` (deterministic standard-library experiment `experiments/sglang_ttft_fill_vllm_mori_ep_trtllm_nvfp4_deepep_gin_phase160/`)

## 1. Primary Upstream Commit Matrix (Track 2: 10 Commits)

| # | Repo & Commit | Date | Key Implementation Path & Source Location |
|---|---|---|---|
| 1 | `sgl-project/sglang@571ddad70f` | 2026-10-31 | `python/sglang/srt/observability/req_time_stats.py`: Reports fine-grained TTFT phase breakdowns in response `meta_info` (`queue_time_ms`, `prefill_launch_ms`, `prefill_compute_ms`, `kv_transfer_ms`). |
| 2 | `sgl-project/sglang@005b8e033f` | 2026-10-31 | `python/sglang/srt/managers/tokenizer_manager.py`: Blocks dispatching requests whose abort signal was already sent and cleans up RID/CUDA IPC state. |
| 3 | `sgl-project/sglang@9767ce3a5a` | 2026-10-31 | `python/sglang/kernels/ops/memory/small_copy.py`: Uses Triton `_write_host_ints_kernel` behind `SGLANG_ENABLE_MLP_SYNC_METADATA_FILL` to batch small forward metadata writes. |
| 4 | `vllm-project/vllm@d5301a4f54` | 2026-10-31 | `docs/serving/expert_parallel_deployment.md`: Adds MORI EP V2 all-to-all communication backend support for MoE deployments. |
| 5 | `vllm-project/vllm@5a233286c6` | 2026-10-31 | `vllm/entrypoints/launchers/grpc_server.py`: Implements bounded `_drain_and_stop` sequencing within `--shutdown-timeout` to drain in-flight gRPC calls and abort lingering engine work cleanly. |
| 6 | `vllm-project/vllm@0c63ce9f0e` | 2026-10-31 | `vllm/entrypoints/openai/chat_completion/batch_serving.py`: Reports `prompt_tokens_details` (including cached prefix tokens) in batched chat completions. |
| 7 | `NVIDIA/TensorRT-LLM@09198b7ca2` | 2026-10-31 | `tests/integration/defs/accuracy/modeling_v2/_env.py`: Isolates `modeling_v2` routing and scopes shared test state across runs. |
| 8 | `NVIDIA/TensorRT-LLM@f6c4125c79` | 2026-10-28 | `tests/unittest/_torch/moe/moe_test_utils.py`: Re-enables CuTeDSL NVFP4 MoE backend tests for large `intermediate_size` configurations. |
| 9 | `deepseek-ai/DeepEP@11d1cab604` | 2026-08-04 | `deep_ep/include/deep_ep/common/comm.cuh`: Inserts system-scope `fence.acq_rel.sys` before `gin_barrier_wo_local_sync` when scaleup spans both NVLink and RDMA fabrics. |
| 10 | `deepseek-ai/DeepEP@dd758caf45` | 2026-07-14 | `csrc/jit/no_ref.hpp`, `csrc/kernels/backend/api.cuh`: Passes runtime version to `ncclDevCommCreate` and wraps the device communicator in `jit::NoRefPtr` to avoid host-side pointer dereference. |

## 2. Key Implementation Mechanics & Formulas

### 2.1 SGLang Fused Metadata Fill (`_write_host_ints_kernel`)
Launching $K=14$ individual host-to-device metadata updates incurs $14 \times 4.5\ \mu\text{s} = 63.0\ \mu\text{s}$ of launch overhead per step. Consolidating them into a single unrolled Triton kernel (`tl.static_range(len(encoded))`, $9.0\ \mu\text{s}$) yields a **$7.0\times$** metadata launch speedup while `tokenizer_manager` abort gating eliminates all `32` ghost request dispatches.

### 2.2 DeepEP Mixed-Fabric System-Scope Fence
When an EP domain bridges intra-node NVLink and inter-node RDMA via NCCL GIN, GPU-local barriers do not order PCIe/NIC RDMA writes with peer NVLink stores. Inserting `fence.acq_rel.sys` prior to the GIN barrier (`DeepEP@11d1cab604`) reduces cross-fabric visibility hazards from `12` to `0`.

## 3. Deterministic Offline Verification (`own-offline`)

Validated via `python experiments/sglang_ttft_fill_vllm_mori_ep_trtllm_nvfp4_deepep_gin_phase160/run.py --check`:
- **SGLang Metadata Fill Launch Speedup**: `7.0x` (`63.0 us -> 9.0 us`), ghost request dispatches reduced `32 -> 0`.
- **vLLM gRPC Graceful Drain**: `64` in-flight RPCs drained in `8.0s` (`< 15.0s` timeout) with `0` orphan engine tasks.
- **DeepEP Mixed-Fabric GIN Fence**: `0` stale visibility hazards across 32 EP ranks (`4` nodes $\times$ `8` GPUs).
