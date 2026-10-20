# Mooncake Store HA Master `SIGTERM`/`SIGINT` 9-Stage Drain & 15s Watchdog, NIXL UCX `eventfd` Control Consolidation & LMCache SGLang Kimi Linear Mamba Zero-Copy Adapter

> **中文版**: [mooncake-ha-sigterm-drain-nixl-ucx-eventfd-lmcache-kimi-linear.md](mooncake-ha-sigterm-drain-nixl-ucx-eventfd-lmcache-kimi-linear.md)  
> **Evidence Class**: `implementation` (verified upstream commits) + `own-offline` (`experiments/mooncake_ha_drain_nixl_eventfd_lmcache_kimi/run.py --check`)

## 1. First-Hand Source Verification & Architectural Mechanics

1. **Mooncake Store HA Master `SIGTERM`/`SIGINT` 9-Stage Drain & 15s Lock-Free Watchdog (`kvcache-ai/Mooncake` commit `3db34ceb3a02113b002b3e36ac58da8d136b306e`, `mooncake-store/include/shutdown_request.h` & `mooncake-store/src/ha/leadership/master_service_supervisor.cpp`)**:
   - Arms `StartShutdownWatchdog(kHaShutdownDeadline = 15s)` and `InstallShutdownSignalHandlers()` (`sigaction` with `SA_ONSTACK` required by Go `c-shared` HA backends) in `main()` before any lease acquisition;
   - First `SIGINT`/`SIGTERM` sets a lock-free sticky flag (`ShutdownRequested()`) and starts an un-resettable 15s deadline; a second signal exits immediately via `_exit(128 + signo)` (`143` for `SIGTERM`); if drain or log flush blocks beyond 15s, the watchdog terminates via `_exit(EXIT_FAILURE)`;
   - Fences active sessions via `ServingStateGate`, drains `RPC listener -> worker pool -> leadership monitor -> OpLog writer -> serving state -> admin server -> standby replication` in strict order, and exits via `_exit(0)` without C++ stack unwinding so the coordinator destructor does not block revoking the etcd lease.
2. **NIXL UCX Shared Progress Thread `eventfd` Consolidation (`ai-dynamo/nixl` commit `8a78b338a367d935a7b1b20385042dbb227e7c71`, `src/plugins/ucx/ucx_thread_engine.cpp`)**:
   - Replaces the 2-FD `pipe(control_pipe)` in `nixlUcxSharedThread` with a single `eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)` (`controlFd_`), cutting progress-thread control file descriptors by **`50.0%` (`2 -> 1` FD per thread, `256` FDs saved at 256 threads)** with non-blocking 64-bit counter wakeups.
3. **LMCache SGLang Kimi Linear Mamba `slice_axis=1` Zero-Copy Adapter (`LMCache/LMCache` commit `e7a4f2380bc3717343ca06519a69187aa36745f9`, `lmcache/integration/sglang/unified_kv_adapter.py`)**:
   - Removes the `slice_axis != 0` `NotImplementedError` in `SGLangUnifiedKVAdapter._resolve_mamba_pool_tensors` because SGLang always exposes `mamba_pool` transfer tensors with `slot` as the leading dimension (`[slot, ...]`), while `slice_axis=1` (used by Kimi Linear `conv` state) only describes intra-slot TP slicing and preserves same-TP full-slot zero-copy wire views (`[5, 1, 1, 21]`) as long as `layer_tensor.is_contiguous()` holds.

## 2. Deterministic Offline Verification (`own-offline`)

Run `python experiments/mooncake_ha_drain_nixl_eventfd_lmcache_kimi/run.py --check`:
- **Mooncake HA SIGTERM Drain**: `9/9` ordered drain stages verified (`0` OpLog batches lost vs `3` under abrupt kill), 15s watchdog fires `_exit(1)` on hang, second `SIGTERM` exits with `143`.
- **NIXL UCX `eventfd`**: `50.0%` control FD reduction (`256` FDs saved at 256 threads), non-blocking coalesced counter wakeups verified.
- **LMCache Kimi Linear Mamba Adapter**: `2/2` contiguous states (`ssm` `slice_axis=0` and `conv` `slice_axis=1`) accepted with `[5, 1, 1, 21]` zero-copy wire view, `1/1` non-contiguous state rejected.
