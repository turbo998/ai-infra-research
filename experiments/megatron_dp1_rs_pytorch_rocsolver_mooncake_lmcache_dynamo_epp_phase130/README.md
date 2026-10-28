# Megatron-LM DP=1 FP32 Reduce-Scatter Fast Path, PyTorch ROCm `syevd` Caching-Allocator Workspace, Mooncake Multi-Node `tebench` & Store-RS, LMCache Stopped-Loop Guard & Dynamo EPP (Phase 130)

Deterministic CPU/offline analytical simulator covering 10 first-hand upstream commits in `NVIDIA/Megatron-LM`, `pytorch/pytorch`, `kvcache-ai/Mooncake`, `LMCache/LMCache`, and `ai-dynamo/dynamo`:

1. `NVIDIA/Megatron-LM@79f5594` (`#8449`): Bypasses redundant FP32 accumulation buffer allocation and single-rank `all_to_all` in `reduce_scatter_with_fp32_accumulation` when `DP size = 1`, returning synchronous `_CompletedWorkHandle`.
2. `pytorch/pytorch@b15a13a` (`#199196`): Allocates batched ROCm `rocSOLVER` `syevd` workspace via PyTorch's `getCUDADeviceAllocator()` instead of raw `hipMalloc` and resets `rocblas_set_workspace(handle, nullptr, 0)` before block release.
3. `pytorch/pytorch@3ce5d60` (`#201518`): Overrides `UserFunctionVariable.get_name()` and `get_qualname()` in TorchDynamo to read `self.fn.__name__` / `self.fn.__qualname__`, preserving `functools.wraps` metadata.
4. `kvcache-ai/Mooncake@78f9eb6` (`#4649`): Adds multi-node `tebench` orchestration (`fanout`, `incast`, `m2n`, `all_to_all`) with SSH launch, start barriers, `write_seed`/`read_verify` phases, and combined JSONL output.
5. `kvcache-ai/Mooncake@e03a7bf` (`#4899`): Adds `--store-backend` (`cpp` vs `store-rs`) to `store_kv_bench` and isolates Store-RS zero-copy buffers in their own registered memory region to prevent `ERR_ADDRESS_OVERLAPPED`.
6. `kvcache-ai/Mooncake@047a215` (`#4901`): Enables Intel XPU device memory (`--backend=tent --seg_type=VRAM`) and `file://` targets in `tebench`.
7. `LMCache/LMCache@abc685b` (`#6116`): Prevents `PDBackendAsync.close()` from scheduling receiver drain coroutines onto an already stopped asyncio loop, eliminating an `11 s` (`shutdown_timeout_sec + 1`) hang on repeated close.
8. `ai-dynamo/dynamo@8a9517c` (`#15605`): Generates Python OpenAI media protocol models (`audio_protocol.py`, `image_protocol.py`, `video_protocol.py`) directly from canonical Rust types.
9. `ai-dynamo/dynamo@9937486` (`#17125`): Stops requiring Envoy's `send_body_without_waiting_for_header_response` flag in Dynamo's Endpoint Picker (`ext-proc`) since `RequestHeaders` is returned only after full body receipt.
10. `ai-dynamo/dynamo@9087efc` (`#17325`): Recognizes disaggregated prefill workers in Dynamo EPP via explicit `WorkerType::Prefill` discovery metadata.

## Usage

```bash
python experiments/megatron_dp1_rs_pytorch_rocsolver_mooncake_lmcache_dynamo_epp_phase130/run.py --check
```
