# TransformerEngine Sync-Free QB Bounds & DistributedWeight Hooks, Megatron-LM MFSDP v2 ExecutionOrder & Deterministic Mode, TorchTitan B200 FP Fusion, Mooncake XPU io_uring Staging & NUMA First-Touch, LMCache DAX Range Ownership, Dynamo Fetch-Credit Router & NIXL Trace Identity Getter (Phase 110)

> 中文版: [te-qb-bounds-dist-weight-megatron-mfsdp-torchtitan-mooncake-xpu-nixl-phase110.md](te-qb-bounds-dist-weight-megatron-mfsdp-torchtitan-mooncake-xpu-nixl-phase110.md) | Companion experiment: `experiments/te_qb_bounds_dist_weight_megatron_mfsdp_mooncake_nixl_phase110/run.py`

## 1. First-Hand Code Evidence (2026-10-19 to 2026-10-27)

| Repo | Commit | Date | Core Mechanism Change | Evidence Class |
|---|---|---|---|---|
| `TransformerEngine` | `96c0fb0e24` | 2026-10-27 | Removes `cudaMemcpyAsync + cudaStreamSynchronize` host validation of `bin_bounds` in `fused_topk_with_score_function`, relying on saturating `__float2int_rd` and `[0, num_bins)` clamping for memory safety | `implementation` |
| `TransformerEngine` | `fa4b30f09b` | 2026-10-27 | Preserves `DistributedWeight` identity across `saved_tensors_hooks` / non-reentrant checkpointing via `_SavedTensorRef` and cuts 32-expert `GroupedLinear` `prepare_for_saving` CPU overhead from `10.0 us` to `5.2 us` (`1.9231x` speedup) | `implementation` + `author-benchmark` |
| `Megatron-LM` | `60547a3f41` | 2026-10-19 | Replaces deduplicated `IndexedOrder` with occurrence-based `ExecutionOrder` in MFSDP v2 so repeated modules (`[m0, m1, m0, m0, m2]`) prefetch exact runtime successors | `implementation` |
| `Megatron-LM` | `89b28875a8` | 2026-10-21 | Runs Nemotron Lightning functional tests in deterministic mode (`--deterministic-mode: true`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `NCCL_ALGO=Ring`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and shared `TRITON_CACHE_DIR`) | `implementation` |
| `torchtitan` | `c1522ec208` | 2026-10-27 | Pins `TRITON_DEFAULT_FP_FUSION="1"` explicitly on B200 CI and refreshes Kimi K3 DistMuon 10-step full-precision golden values (step-10 loss `6.416478 -> 6.417246`) | `implementation` |
| `Mooncake` | `6b90ea5e6c` | 2026-10-27 | Stages `MTYPE_XPU` device buffers through `4096 B` page-aligned host memory in TENT `IOUringTransport::submitTransferTasks` when GDS is unavailable | `implementation` |
| `Mooncake` | `c3df64c745` | 2026-10-27 | Pins `tebench` allocating threads to a target NUMA CPU, first-touches DRAM pages without `CAP_SYS_NICE`, and registers the dominant node from `/proc/self/numa_maps` | `implementation` |
| `LMCache` | `8afb6da466` | 2026-10-22 | Adds `offset_gb` and range-based `owns_range(path, offset, size)` checks so L1 and DAX L2 or multiple servers can share disjoint byte ranges on one CXL DAX device | `implementation` |
| `dynamo` | `d3fd74f489` | 2026-10-26 | Credits peer-fetchable KV blocks (`fetchable_blocks * shared_cache_multiplier`) during worker selection and binds the winning fetch source into the router hint | `implementation` |
| `nixl` | `280c3a1cd8` | 2026-10-27 | Adds lock-free `nixlAgent::getXferTraceInfo()` returning a transfer handle's `request_id`, backend-post `execution_number`, and W3C `traceparent` | `implementation` |

## 2. Key Implementation & Architectural Takeaways

1. **Zero-Sync Router Bounds & Hook-Safe Distributed Weights**: Dropping host validation of `bin_bounds` removes an implicit stream sync on every MoE router call while `__float2int_rd` saturation + `[0, 32)` clamping prevents out-of-bounds writes; `_SavedTensorRef` preserves `DistributedWeight` dispatch under activation checkpointing and speeds up `prepare_for_saving` by `1.9231x`.
2. **Range-Shared CXL DAX & Fetch-Credited KV Routing**: Replacing whole-device ownership with `owns_range(path, offset, size)` allows `[0, 512 GiB)` L1 and `[512, 1024 GiB)` L2 mappings to coexist on one `1 TiB` CXL device, while Dynamo's `0.5x` fetchable-block credit reduces `100` raw prefill blocks with `60` peer-fetchable blocks to `70.0` adjusted blocks.
