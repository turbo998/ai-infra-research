# Phase 112 Track C: TransformerEngine FusedAttention `torch.compile` & Deterministic dGLU, Megatron-LM MFSDP Unshard/MXFP8 Events, TorchTitan EMA Reseed, Mooncake TENT Context Restore & NIXL POSIX Rollback (2026-10-28)

> **中文对照 / Chinese Version**: [`te-compile-fused-attn-megatron-mfsdp-mooncake-tent-nixl-phase112.md`](./te-compile-fused-attn-megatron-mfsdp-mooncake-tent-nixl-phase112.md)  
> **7-Card 4:5 Visual Benchmark Report**: [`../reports/ai-infra-phase112-benchmark-cards.html`](../reports/ai-infra-phase112-benchmark-cards.html)  
> **Reproducible Experiment Suite**: [`../../experiments/te_compile_fused_attn_megatron_mfsdp_mooncake_tent_nixl_phase112/`](../../experiments/te_compile_fused_attn_megatron_mfsdp_mooncake_tent_nixl_phase112/)

---

## 1. Evidence Classification & First-Hand Commit Ledger (10 Training, Storage & Fabric Commits)

| # | Repo & Commit | Date | Evidence Tier | File Paths & Core Mechanism |
|---|---|---|---|---|
| 1 | [`TransformerEngine@7e7898fa82`](https://github.com/NVIDIA/TransformerEngine/commit/7e7898fa82719ebc1069cfa29b9ccd3b6eec4a24) | 2026-10-08 | `implementation` | `transformer_engine/pytorch/attention/dot_product_attention/backends.py`, `cpp_custom_ops.py`: Registers `te_fused_attn_fwd` and `te_fused_attn_bwd` custom ops with flat tensor/scalar signatures and `register_fake` meta kernels so `FusedAttention` compiles under `torch.compile(fullgraph=True)` without graph breaks |
| 2 | [`TransformerEngine@c25fa27525`](https://github.com/NVIDIA/TransformerEngine/commit/c25fa275259808f6f6cc4fa0b8b285cfd453287a) | 2026-10-08 | `implementation` | `transformer_engine/common/fused_glu/fused_glu.cu`, `pytorch/cpp_extensions/fused_attn.py`: Adds a deterministic 2D/3D (chunked) workspace reduction path for `dprob` in `nvte_ACT_and_dACT_and_mul_dprob`, enabling `torch.use_deterministic_algorithms(True)` for fused grouped MLPs |
| 3 | [`Megatron-LM@17fc7ba109`](https://github.com/NVIDIA/Megatron-LM/commit/17fc7ba10998cce2d2dba210e3a70eb09d604651) | 2026-10-27 | `implementation` | `megatron/core/distributed/fsdp/src/megatron_fsdp/{megatron_fsdp,param_and_grad_buffer}.py`: Makes `_ParameterGroup.model_weight_buffer_all_gather_finished` the single readiness source of truth, splits MXFP8 unshard completion into `all_gather_event` and `all_gather_colwise_event`, and guards `resize_storage` |
| 4 | [`Megatron-LM@74c9420e35`](https://github.com/NVIDIA/Megatron-LM/commit/74c9420e35cc3c1226e53421f5ff718653aa8399) | 2026-10-19 | `implementation` | `tests/unit_tests/dist_checkpointing/models/test_mfsdp_Checkpoint.py`: Wraps model-parallel state setup and teardown in `try...finally` to prevent state leaks on test failures |
| 5 | [`torchtitan@ba53205255`](https://github.com/pytorch/torchtitan/commit/ba53205255cc89e0479096cd061fccbb71e30112) | 2026-10-17 | `implementation` | `torchtitan/components/checkpoint.py`: Introduces `EMAStateAdapter` to trigger `ema.reseed_from_model()` whenever `ema` state is absent during load (base checkpoint, `--checkpoint.model-only`, or `--checkpoint.exclude-from-loading=ema`) across any checkpoint backend |
| 6 | [`Mooncake@edd323921d`](https://github.com/kvcache-ai/Mooncake/commit/edd323921d531d703f825bd042be552c878151a3) | 2026-10-22 | `implementation` | `mooncake-store-rs/src/{rpc_service,tenant_context}.rs`: Enforces control-plane tenant binding before dispatching Store-RS RPC handlers, rejecting unauthorized cross-tenant access with `-3250` (`RPC_UNAUTHENTICATED`) |
| 7 | [`Mooncake@e3e9d70eea`](https://github.com/kvcache-ai/Mooncake/commit/e3e9d70eea3c8ea9eb236b4c9c92f361fbbe533f) | 2026-10-21 | `implementation` | `mooncake-transfer-engine/tent/include/tent/runtime/platform/cuda.h`: Captures `cuCtxGetCurrent(&saved_ctx_)` in `CudaContextGuard` and restores `cuCtxSetCurrent(saved_ctx_)` on exit instead of forcing the device's primary context via `cudaSetDevice` |
| 8 | [`LMCache@e0e92b8356`](https://github.com/LMCache/LMCache/commit/e0e92b83568ab79eeee824cbbf19992ed7374f3e) | 2026-10-22 | `implementation` | `lmcache/v1/periodic_thread.py`: Checks `self._thread.is_alive()` in `PeriodicThread.start()` after a `stop(timeout)` timeout so a restart never spawns an overlapping worker thread |
| 9 | [`dynamo@492108cd43`](https://github.com/ai-dynamo/dynamo/commit/492108cd434efc46920ba230175c98faf5c0c117) | 2026-10-27 | `implementation` | `lib/llm/src/http/service/service_v2.rs`: Adds `Connection: close` to `503 Service Unavailable` responses returned during graceful drain so HTTP/1.1 keep-alive clients reconnect to healthy pods |
| 10 | [`nixl@f7533eb8a5`](https://github.com/ai-dynamo/nixl/commit/f7533eb8a547c4d2935cbcee6edf262eca7a78f6) | 2026-10-12 | `implementation` | `src/plugins/posix/{posix_backend.cpp,uring_queue.cpp}`: Implements two-phase rollback (unstage unsubmitted entries in reverse + reap already-submitted in-flight operations until `numCompleted == numSubmitted`) and treats short `io_uring_submit` counts as errors |

---

## 2. Architectural Mechanisms & Formal Formulations

### 2.1 TransformerEngine `FusedAttention` Zero-Break Custom Ops & Deterministic dGLU (`7e7898fa82`, `c25fa27525`)

1. **Flat Tensor/Scalar `torch.library.custom_op` Boundary (`7e7898fa82`)**: By decomposing `Quantizer` metadata into flat tensor/scalar arguments on `tex.te_fused_attn_fwd` and `tex.te_fused_attn_bwd` and registering `@torch.library.register_fake` meta functions, `FusedAttention` compiles under `torch.compile(fullgraph=True)` with **0 graph breaks** across BF16, FP16, `Float8CurrentScaling`, `Float8BlockScaling`, `MXFP8BlockScaling`, and `NVFP4BlockScaling` (6 of 7 modes; `DelayedScaling` remains excluded due to Python-side amax history state).
2. **Deterministic Workspace Reduction for `dprob` (`c25fa27525`)**: Instead of cross-block `atomicAdd` into `dprob[i]`, `nvte_ACT_and_dACT_and_mul_dprob` writes per-column-tile partial sums into a 2D/3D FP32 workspace and reduces them deterministically, cutting peak workspace memory by **$50\%$** at $R=8192$ via 4096-row chunking while satisfying `torch.use_deterministic_algorithms(True)`.

### 2.2 Megatron-LM MFSDP Dual MXFP8 Events, TorchTitan EMA Reseed, Mooncake TENT & NIXL POSIX (`17fc7ba109`, `ba53205255`, `e3e9d70eea`, `f7533eb8a5`)

- **MFSDP Dual Event Tracking (`17fc7ba109`)**: Assigns separate `all_gather_event` and `all_gather_colwise_event` handles per parameter group so MXFP8 rowwise and columnwise all-gathers never overwrite each other's completion event, and centralizes readiness under `model_weight_buffer_all_gather_finished`.
- **TorchTitan `EMAStateAdapter` (`ba53205255`)**: Invokes `ema.reseed_from_model()` directly inside `EMAStateAdapter.load_state_dict` whenever `state_dict` is empty, guaranteeing synchronized EMA weights across all checkpoint backends and `--checkpoint.model-only` / `--checkpoint.exclude-from-loading=ema` flows.
- **Mooncake TENT Exact `CUcontext` Restore (`e3e9d70eea`)** & **NIXL POSIX Two-Phase Rollback (`f7533eb8a5`)**: Restores the exact caller `CUcontext` via `cuCtxSetCurrent(saved_ctx_)`, and drains all already-submitted POSIX/io_uring/AIO operations before returning an enqueue error so no kernel IO writes into freed memory.

---

## 3. Deterministic Offline Verification Results (`own-offline`)

Validated via `python experiments/te_compile_fused_attn_megatron_mfsdp_mooncake_tent_nixl_phase112/run.py --check` ([`summary.json`](../../experiments/te_compile_fused_attn_megatron_mfsdp_mooncake_tent_nixl_phase112/summary.json)):

| Dimension | Legacy / Baseline | Fixed / Optimized | Key Quantitative Metric |
|---|---|---|---|
| TE `FusedAttention` `torch.compile` | `2` graph breaks (`FusedAttnFunc`) | `0` graph breaks (`te_fused_attn_fwd/bwd`) | **`85.7%`** (6/7 modes) `fullgraph=True` coverage |
| TE dGLU Deterministic `dprob` ($8192\times 4096$) | Runtime error (`atomicAdd`) | 2D/3D chunked reduction passes | **`50.0%`** workspace memory saved via 4096-row chunking |
| Megatron-LM MFSDP MXFP8 Unshard | `1` overwritten event/group | `2` independent events/group | Safe fallback when `resize_storage` is unavailable |
| TorchTitan EMA Load from Base Checkpoint | Stale random EMA (`-0.25`) | Reseeded to model weight (`1.75`) | Backend-agnostic across DCP and non-DCP loaders |
| Mooncake TENT & NIXL POSIX Recovery | Primary ctx leak / `EBUSY` hang | Restores `ctx_custom_7` / reaps `2` IOs | Zero dangling in-flight IOs on partial enqueue failure |
