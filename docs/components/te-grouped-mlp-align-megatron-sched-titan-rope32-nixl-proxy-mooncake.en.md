# TE GroupedMLP Split-Alignment Fusion & ONNX 0-D Scale Metadata, Megatron SchedulerConfig & TorchTitan Llama-3.2 32x RoPE / Attention Gym Per-Module Guards, and NIXL EP 8x8 CPU Proxy & Mooncake Conductor Rust Client

> **Updated**: 2026-10-22 · **Phase**: Phase 53 (`priority: 62`) · **Category**: `distributed-training`  
> **Deterministic Experiment**: [`experiments/te-grouped-mlp-align-megatron-sched-titan-rope32-nixl-proxy-mooncake/run.py`](../../experiments/te-grouped-mlp-align-megatron-sched-titan-rope32-nixl-proxy-mooncake/run.py)

---

## 1. First-Hand Upstream Sources

| Repository & Commit | Date (UTC) | Core Engineering Contribution |
|---|---|---|
| [`NVIDIA/TransformerEngine@9d9bd9e5c4`](https://github.com/NVIDIA/TransformerEngine/commit/9d9bd9e5c4764633c252fdad905d8f60b6d95098) | 2026-10-15 | `[PyTorch]`: Deprecate `NVTE_CUTEDSL_FUSED_GROUPED_MLP` env gate in favor of explicit `GroupedLinear(split_alignment=256/128)` automatic `GroupedMLP_CuTeGEMMGLU` fusion (#3584) |
| [`NVIDIA/TransformerEngine@33f3b35619`](https://github.com/NVIDIA/TransformerEngine/commit/33f3b3561943fcd9dd2f544f67dc2dfd62efdc9e) | 2026-10-15 | `Fix ONNX custom-op metadata`: Fix `onnx_cs_quantize_fp8_op` fake-tensor scale output to 0-D `()` and validate with `torch.library.opcheck` (#3659) |
| [`NVIDIA/Megatron-LM@e4eb89b3bb`](https://github.com/NVIDIA/Megatron-LM/commit/e4eb89b3bb3bb55c31606ee9d672038e17caa2a0) | 2026-10-21 | `refactor(training)`: Migrate LR/WSD step resolution into `SchedulerConfig.validate_run()` and `PretrainConfigContainer._resolve_scheduler_step_fields()` (#8211) |
| [`pytorch/torchtitan@48bbce4434`](https://github.com/pytorch/torchtitan/commit/48bbce4434dfac2c38fedc9c7ef9c1998b04cf50) | 2026-10-19 | `[llama3]`: Set `scaling_factor=32.0` on Llama-3.2 1B and 3B flavors, eliminating `17/32` and `35/64` rotary frequency mismatches and `0.781 rad` angle error (#5528) |
| [`pytorch/torchtitan@efdde5edaf`](https://github.com/pytorch/torchtitan/commit/efdde5edaf2acad28782f5e8a851f9e9ecdb51d6) | 2026-10-19 | `[models]`: Report the missing backend per Attention Gym kernel via `_MissingKernel[<module>]` subclasses and `require_attn_gym(*kernels)` (#5500) |
| [`LMCache/LMCache@18651e8f19`](https://github.com/LMCache/LMCache/commit/18651e8f19c94f56c32bc88e8826e37760a726c3) | 2026-10-20 | `Refactor legacy transfer channel under inprocess`: Consolidate `CreateTransferChannel`, `NixlChannel`, and `PySocketChannel` under `lmcache.inprocess.v1.transfer_channel` (#5950) |
| [`ai-dynamo/dynamo@8e26089d3f`](https://github.com/ai-dynamo/dynamo/commit/8e26089d3f2ebddcad6c2117911c75d47a931ddf) | 2026-10-21 | `fix(sidecar)`: Forward `stop_token_ids_visible` in SGLang and TensorRT-LLM sidecars so visible tool delimiters (`<|call|>`) reach the frontend (#16759) |
| [`ai-dynamo/nixl@fda3a883c0`](https://github.com/ai-dynamo/nixl/commit/fda3a883c0aef00e3b42c4a0f08899aa0b49d3c6) | 2026-10-19 | `nixl_ep`: Run expert-parallel dispatch + combine over the 8-channel / 8-thread CPU proxy (`FENCE_MODE=ep_based`), reaching `100%–109%` of direct GPU RDMA on H100/GB200 (#2328) |
| [`kvcache-ai/Mooncake@a4fe83974b`](https://github.com/kvcache-ai/Mooncake/commit/a4fe83974b26e4565ee565d12db91f8af6e768c2) | 2026-10-19 | `[Conductor]`: Expose `ConductorClient` via C ABI (`conductor_c.h`) and safe `Send + Sync` Rust crate (`mooncake-conductor/rust`) (#4651) |

---

## 2. Deep-Dive Architectural Analysis

### 2.1 TransformerEngine GroupedMLP Split-Alignment & ONNX 0-D Fake-Tensor Metadata (`9d9bd9e5c4`, `33f3b35619`)

- **Automatic `GroupedMLP_CuTeGEMMGLU` Fusion via `split_alignment` (`9d9bd9e5c4`)**: Deprecates `NVTE_CUTEDSL_FUSED_GROUPED_MLP` and fuses `Sequential(GroupedLinear, ScaledSwiGLU(glu_interleave_size=32), GroupedLinear)` automatically when `split_alignment=256` (BF16/FP16) or `split_alignment=128` (`MXFP8`, `NVFP4`, `FP8 block scaling`), emitting a warning if fusion is skipped solely due to alignment.
- **ONNX FP8 Fake-Tensor 0-D Scale (`33f3b35619`)**: Fixes `onnx_cs_quantize_fp8_op.register_fake` to return a 0-D scalar scale `torch.ones((), dtype=torch.float32)` instead of 1-D `(1,)`, passing `torch.library.opcheck(..., test_utils=("test_faketensor",))`.

### 2.2 Megatron `SchedulerConfig` & TorchTitan Llama-3.2 32x RoPE / Attention Gym Guards (`e4eb89b3bb`, `48bbce4434`, `efdde5edaf`)

- **Megatron `SchedulerConfig` Migration (`e4eb89b3bb`)**: Centralizes iteration-based and sample-based LR warmup/decay/WSD step derivation into `PretrainConfigContainer._resolve_scheduler_step_fields()` and `SchedulerConfig.validate_run()`.
- **TorchTitan Llama-3.2 `scaling_factor=32.0` (`48bbce4434`)**: Fixes Llama-3.2 1B and 3B flavors that previously inherited Llama-3.1's `8.0` default, eliminating **`17/32` (1B) and `35/64` (3B)** rotary frequency mismatches, **`0.195 / 0.781 rad`** angle error at positions `2048 / 8192`, and reducing max absolute logit drift against HuggingFace from `3.2e-4` to **`7e-7`**.
- **Per-Module Attention Gym `_MissingKernel` Subclasses (`efdde5edaf`)**: Replaces a single global import error with cached `_MissingKernel[<module>]` subclasses and `is_missing_kernel()` `TypeGuard` checks so `KDAKernel`, `GDNKernel`, and `ContextParallelInnerGDN` report only their own missing backend (`cutlass` vs `triton`).

### 2.3 NIXL EP 8x8 CPU Proxy, Dynamo Visible Stop Tokens & Mooncake Conductor Rust API (`fda3a883c0`, `8e26089d3f`, `18651e8f19`, `a4fe83974b`)

- **NIXL EP 8-Channel / 8-Thread CPU Proxy (`fda3a883c0`)**: Enables `NIXL_EP_DEVICE_MODE=proxy` (`FENCE_MODE=ep_based`) for NICs/transports without GPU-initiated RDMA. With `8 channels × 8 threads` pinned per GPU NUMA node, network dispatch + combine reaches **`59.0 GB/s` (`103%` of direct) on 2x H100, `45.5 GB/s` (`104%`) on 4x H100, `41.6 GB/s` (`100%`) on 8x H100, `53.3 GB/s` (`109%`) on 2x GB200, and `50.7 GB/s` (`104%`) on 2x GB300**, while demonstrating that oversubscribing a 32-core H100 NUMA node with `16 spinning threads/GPU` collapses bandwidth to `1.2 GB/s` (`3%`).
- **Dynamo Visible Stop Token Forwarding (`8e26089d3f`) & Mooncake Conductor Rust Client (`a4fe83974b`)**: Preserves visible tool-call delimiters (`<|call|>`, merged sorted IDs `[2, 7, 200002, 200012]`) across SGLang (`return_text=None`) and TRT-LLM (`include_stop_in_output=Some(true)`) sidecars, and wraps Mooncake `ConductorClient` in `conductor_c.h` (`0` to `-2001` error codes) and a `Send + Sync` Rust crate (`mooncake-conductor/rust`).
