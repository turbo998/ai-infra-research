# Phase 92 Track 3: TransformerEngine NVFP4 MLP, DeepGEMM Fences, TorchTitan Flux/Compile, Mooncake XPU EP & NIXL/Dynamo

Deterministic Python 3.12 standard-library simulator covering 10 upstream commits (`NVIDIA/TransformerEngine`, `deepseek-ai/DeepGEMM`, `pytorch/torchtitan`, `kvcache-ai/Mooncake`, `ai-dynamo/nixl`, and `ai-dynamo/dynamo`):

1. `NVIDIA/TransformerEngine@f2be56a50a` (`#3545`): Gather distributed BF16 weights before quantizing with per-expert quantizers in fused grouped MLP, move FC2 backward re-gather ahead of NVFP4 amax, and trim redundant NVFP4 saved state.
2. `deepseek-ai/DeepGEMM@78b6900079` (`#441`): Insert `fence.acq_rel.cta` before `task_info_empty_barriers.arrive(0u)` in SM100 MegaMoE `release_task_info()`.
3. `deepseek-ai/DeepGEMM@1f6f3f3789` (`#343`): Insert `tma_desc_commit_group()`, `tma_desc_wait_group()`, and `__syncwarp` before updating global tensormaps in SM90 1D1D FP8 GEMM.
4. `pytorch/torchtitan@d681f48b43` (`#5580`): Compute Flux `denoise` schedule shift after `pack_latents` using `latents.shape[1]` (`4x` smaller than unpacked `H*W`).
5. `pytorch/torchtitan@3521b50957` (`#5561`): Enable `rope` (`recompile_limit=32`) in DeepSeek-V4 and `fused_binary_activation` (SwiGLU) in Muse Spark `local_compile_regions` by default.
6. `kvcache-ai/Mooncake@7d0ff6c65c` (`#4900`): Add native Intel XPU (SYCL) single-node Mooncake EP using Level Zero IPC (`zeMemGetIpcHandle` + `pidfd_getfd`), fd-rewrite cache bypass, and `(boot_id, pid_ns)` validation.
7. `kvcache-ai/Mooncake@8d7fcd8885` (`#4633`): Guard Mooncake Store `RealClient` batch transfer metric observers with `client_ != nullptr` when uninitialized.
8. `ai-dynamo/nixl@603f70e996` (`#2506`): Give each sampled `postXferReq` repost a distinct `span_id` (`tracePostContext()`) while preserving the request's `trace_id`.
9. `ai-dynamo/dynamo@b20711f9c2` (`#16459`): Share `grpc_monitor_for_disconnects` across OpenAI, KServe, and Tensor gRPC streaming services to cancel contexts and drop `InflightGuard` on client disconnect.
10. `ai-dynamo/dynamo@a0de541ea8` (`#17163`): Add `DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION` opt-in migration policy on the Dynamo frontend.
