# TransformerEngine Exit-Safe Static Cache, Megatron-LM FP8 Checkpoint GC Removal, Mooncake EGM + Local Staging Decoupling & NIXL Cross-Process UCX Trace Analysis (Phase 107 · Priority 116)

> **中文版**: [te-exit-dtor-megatron-fp8-gc-mooncake-egm-nixl-ucx-trace-phase107.md](te-exit-dtor-megatron-fp8-gc-mooncake-egm-nixl-ucx-trace-phase107.md)  
> **Evidence Boundary**: `implementation` (first-hand upstream commits) + `own-offline` (deterministic stdlib modeling: `experiments/te_exit_dtor_megatron_fp8_gc_mooncake_egm_nixl_ucx_trace_phase107/run.py --check`)

## 1. Primary Source Evidence & Implementation Paths

1. **`NVIDIA/TransformerEngine@dd628d3` (`#3733`: `transformer_engine/common/fused_attn/graph_cache.h`, `util/cuda_driver.h`, `util/rtc.cpp`) & `564e8b0` (`#3692`)**:
   - **Eliminating CUDA Driver/cuDNN Calls from C++ Static Destructors at Exit**: During process termination (`exit()`), static destruction order across translation units is unspecified and the CUDA driver may already be shutting down. Allocating `GraphCache`, `symbol_cache`, and `cache_mutex` as immortal heap references (`static auto &cache = *new ...`) reduces exit-time static destructor CUDA calls from `3` to **`0`**, preventing use-after-destroy crashes verified under glibc `MALLOC_PERTURB_=165`.
2. **`NVIDIA/Megatron-LM@29319fd` (`#8522`: `megatron/training/training.py`) & `a747bac` (`#8414`) & `pytorch/torchtitan@e3737d1` (`#5621`)**:
   - **Dropping FP8-Only `gc.collect()` After Checkpoint Saves**: Previously, `save_checkpoint_and_time` invoked a full Python `gc.collect()` whenever `args.fp8` was enabled right after per-rank-imbalanced checkpoint saving and immediately before the collective `timers.log` min/max barrier. Removing `gc.collect()` cuts post-checkpoint barrier tail latency on a 64-rank cluster (`420 ms` save skew, `1,850 ms` GC pause) from `2,270.0 ms` to **`420.0 ms` (`1,850.0 ms` saved per checkpoint, `5.4048x` tail reduction)**.
3. **`kvcache-ai/Mooncake@829d95f` (`#4859`: `mooncake-store/src/config/egm_store_pool_config.cpp`, `mooncake-integration/store/store_py.cpp`) & `LMCache/LMCache@10dd1e6` (`#5979`)**:
   - **Independent `local_buffer_size` Client Staging Alongside Mooncake EGM Store Pools**: Lifts the requirement that `local_buffer_size` must be `0` when `enable_egm_store_pool=true`. While the EGM DRAM pool is rounded down to a multiple of `max(slab_align, cuda_granularity)` across NUMA nodes (`128.0 GiB` total / **`64.0 GiB` per NUMA node** at `64 MiB` CUDA granularity), a nonzero `local_buffer_size` (`512 MiB`) uses the independent client-buffer allocator so hybrid Providers can serve EGM segments and invoke `put()`/`get_buffer()` simultaneously.
4. **`ai-dynamo/nixl@13a896b` (`#2404`: `test/gtest/ucx_trace_context_test.cpp`, `mocks/trace_recorder_plugin.cpp`) & `ai-dynamo/dynamo@95f1b12` (`#16540`)**:
   - **Two-Agent Cross-Process UCX Trace Context Verification & Dynamo Webhook Defaulting**: Spawns a separate `ucx_trace_receiver` process and records finished spans via `libtrace_backend_recorder.so` to assert that a single `trace_id` propagates intact across two NIXL agents over UCX (**`100%` trace continuity across `16/16` transfers**).

## 2. Architect Design Exercises & Falsifiable Questions

- **Calculation Exercise**: Given `requested_egm_bytes = 128 GiB + 32 MiB`, `numa_nodes = 2`, Store slab alignment `2 MiB`, and CUDA allocation granularity `64 MiB`, verify that `common_align = 64 MiB` rounds the total EGM pool to `128.0 GiB` (`64.0 GiB/node`) while leaving the `512 MiB` `local_buffer_size` staging pool independent.
- **Falsifiable Hypothesis**: Destroying a function-local `static GraphCache` that calls `cudnnBackendDestroyDescriptor` at process exit while `MALLOC_PERTURB_=165` is active deterministically faults if another destructor accesses the symbol map afterward (eliminated by `*new` immortal allocation in `#3733`).
