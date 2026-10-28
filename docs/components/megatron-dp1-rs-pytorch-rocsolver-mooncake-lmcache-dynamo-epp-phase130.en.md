# Megatron-LM DP=1 FP32 Reduce-Scatter Fast Path, PyTorch ROCm `syevd` Caching-Allocator Workspace, Mooncake Multi-Node `tebench` & Store-RS, LMCache Stopped-Loop Guard & Dynamo EPP (Phase 130)

> **Chinese Edition**: [megatron-dp1-rs-pytorch-rocsolver-mooncake-lmcache-dynamo-epp-phase130.md](megatron-dp1-rs-pytorch-rocsolver-mooncake-lmcache-dynamo-epp-phase130.md)  
> **Phase 130 (`priority: 139`) · 130-Phase & 400-Experiment-Suite Milestone · Track 3 (Distributed Training, Compiler & Disaggregated KV Fabric)**  
> **Evidence Class**: `implementation` (first-hand upstream commits) + `own-offline` (deterministic CPU simulation; not a GPU hardware rerun)

---

## 1. Architectural Context & Problem Statement

Across Megatron-LM, PyTorch (ROCm & TorchDynamo), Mooncake, LMCache, and NVIDIA Dynamo, 10 upstream commits landed on 2026-10-28 address five critical systems bottlenecks:

1. **Redundant FP32 Accumulation & All-to-All at `DP size = 1` in Megatron-LM**: In model-parallel configurations where a group has `DP = 1` (`world_size == 1`), `reduce_scatter_with_fp32_accumulation` still allocated an FP32 buffer (**512 MiB** for 128M elements), upcast BF16 to FP32, ran a single-rank `all_to_all`, and copied back to BF16.
2. **PyTorch ROCm `rocSOLVER` `syevd` False OOM under Caching Allocator & TorchDynamo `functools.wraps` Regression**: `rocSOLVER` grew its `syevd` workspace via raw `hipMalloc`, failing with `rocblas_status_memory_error` whenever PyTorch's caching allocator held the device's free memory blocks; meanwhile, `UserFunctionVariable.get_name()` read `co_name` instead of `self.fn.__name__`, losing `functools.wraps` names under `torch.compile` (`#201123`).
3. **Mooncake Multi-Node `tebench` Orchestration, Store-RS `ERR_ADDRESS_OVERLAPPED`, and Intel XPU TENT VRAM**: Registering a slice of Store-RS's already-registered local scratch pool for zero-copy buffers triggered `ERR_ADDRESS_OVERLAPPED`; meanwhile, `tebench` needed SSH barrier orchestration across `fanout`, `incast`, `m2n`, and `all_to_all` topologies plus Intel XPU VRAM support (`--backend=tent --seg_type=VRAM`).
4. **LMCache 11-Second Hang on Repeated `PDBackendAsync.close()`**: Calling `close()` after `_recv_loop` had already stopped still submitted a pending-allocation drain coroutine via `run_coroutine_threadsafe`, blocking for `shutdown_timeout_sec + 1` (`11.0 s`) and emitting an unawaited-coroutine warning.
5. **NVIDIA Dynamo Gateway EPP Protocol Over-Constraint & Prefill Worker Discovery**: Dynamo's `ext-proc` Endpoint Picker (`EPP`) builds its `RequestHeaders` response only after receiving the full request body (to tokenize, rewrite LoRA model names, and set `x-dynamo-worker-instance-id`), so requiring Envoy's `send_body_without_waiting_for_header_response` flag was unnecessary; additionally, EPP now identifies prefill workers via `WorkerType::Prefill` and generates Python media protocols from Rust types.

---

## 2. First-Hand Upstream Evidence Matrix (10 Commits)

| # | Repository & Commit | Date | Mechanism & Quantitative Architectural Impact |
|---|---|---|---|
| 1 | `NVIDIA/Megatron-LM@79f5594` (`#8449`) | 2026-10-28 | Short-circuits `reduce_scatter_with_fp32_accumulation` when `world_size == 1` (`DP = 1`), copying directly to `output_tensor` and returning synchronous `_CompletedWorkHandle` (saves **512 MiB** FP32 temp buffer for 128M-element shard). |
| 2 | `pytorch/pytorch@b15a13a` (`#199196`) | 2026-10-28 | Queries `rocSOLVER` batched `syevd` workspace size via `rocblas_start_device_memory_size_query`, allocates from PyTorch's `getCUDADeviceAllocator()`, passes it to `rocblas_set_workspace`, and resets to `(nullptr, 0)` before freeing. |
| 3 | `pytorch/pytorch@3ce5d60` (`#201518`) | 2026-10-28 | Overrides `get_name()` and `get_qualname()` on `UserFunctionVariable` to return `self.fn.__name__` and `self.fn.__qualname__`, preserving `functools.wraps` names in `torch.compile`. |
| 4 | `kvcache-ai/Mooncake@78f9eb6` (`#4649`) | 2026-10-28 | Adds `run_tebench_multinode.sh` to orchestrate multi-node `fanout`, `incast`, `m2n`, and `all_to_all` runs with SSH start barriers, `write_seed`/`read_verify` consistency phases, and combined JSONL output. |
| 5 | `kvcache-ai/Mooncake@e03a7bf` (`#4899`) | 2026-10-28 | Adds `--store-backend` (`cpp` vs `store-rs`) to `store_kv_bench.py` and gives Store-RS zero-copy buffers a dedicated registered memory region to avoid `ERR_ADDRESS_OVERLAPPED`. |
| 6 | `kvcache-ai/Mooncake@047a215` (`#4901`) | 2026-10-28 | Adds Intel XPU device memory (`--backend=tent --seg_type=VRAM`) and `file://` target support to `tebench` under `USE_XPU=ON`. |
| 7 | `LMCache/LMCache@abc685b` (`#6116`) | 2026-10-28 | Skips pending-allocation drain in `PDBackendAsync.close()` when `_recv_loop` is already stopped, eliminating the `shutdown_timeout_sec + 1` (**11.0 s**) hang on repeated close. |
| 8 | `ai-dynamo/dynamo@8a9517c` (`#15605`) | 2026-10-28 | Generates Python media protocol models (`audio_protocol.py`, `image_protocol.py`, `video_protocol.py`) and JSON schemas directly from canonical Rust types. |
| 9 | `ai-dynamo/dynamo@9937486` (`#17125`) | 2026-10-28 | Stops requiring `send_body_without_waiting_for_header_response` in Dynamo EPP's `ProtocolConfiguration` validation while retaining bidirectional `FULL_DUPLEX_STREAMED`. |
| 10 | `ai-dynamo/dynamo@9087efc` (`#17325`) | 2026-10-28 | Recognizes disaggregated prefill workers in Dynamo EPP via explicit `WorkerType::Prefill` discovery metadata. |

---

## 3. Deterministic Verification & Architect Guidance

All 5 invariants in `experiments/megatron_dp1_rs_pytorch_rocsolver_mooncake_lmcache_dynamo_epp_phase130/run.py --check` pass (`megatron_dp1_eliminates_512mib_fp32_buffer`, `pytorch_rocsolver_caching_allocator_avoids_oom`, `mooncake_store_rs_avoids_err_address_overlapped`, `lmcache_repeated_close_eliminates_11s_hang`, `dynamo_epp_uses_worker_type_and_full_duplex`).

- **Falsifiable Follow-Up Question**: In a 16-node Mooncake `all_to_all` KV transfer benchmark driven by `run_tebench_multinode.sh`, what is the throughput and P99 tail-latency delta between `--store-backend=cpp` and `--store-backend=store-rs` with dedicated zero-copy buffer registration?
