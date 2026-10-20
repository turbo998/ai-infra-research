# Mooncake HA SIGTERM Drain, NIXL UCX `eventfd` & LMCache Kimi Linear Mamba Adapter (`Phase 32`)

Deterministic Python 3.12 standard-library simulator verifying:
1. `kvcache-ai/Mooncake` (`3db34ceb3a02113b002b3e36ac58da8d136b306e`): Lock-free `SA_ONSTACK` `SIGINT`/`SIGTERM` handlers, 15-second `StartShutdownWatchdog`, 9-stage ordered HA Master drain (`ServingStateGate` fence -> RPC listener -> worker pool -> leadership monitor -> OpLog writer -> serving state -> admin server -> standby replication -> `_exit(0)` without stack unwind), and immediate `_exit(128 + signo)` on a second signal.
2. `ai-dynamo/nixl` (`8a78b338a367d935a7b1b20385042dbb227e7c71`): Replacing `pipe(control_pipe)` (2 FDs) with `eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)` (1 FD) in `nixlUcxSharedThread`, reducing progress-thread control FDs by `50%` (`256` FDs saved at 256 threads).
3. `LMCache/LMCache` (`e7a4f2380bc3717343ca06519a69187aa36745f9`): Supporting Kimi Linear `conv` state (`slice_axis=1`) in `SGLangUnifiedKVAdapter._resolve_mamba_pool_tensors` while enforcing `layer_tensor.is_contiguous()` and zero-copy `[5, 1, 1, 21]` wire views.

## Run

```bash
python experiments/mooncake_ha_drain_nixl_eventfd_lmcache_kimi/run.py --check
```
