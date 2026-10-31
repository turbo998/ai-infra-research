# Megatron-LM Min-Memory DSA Attention & Async Routing Replay, Mooncake Replica Self-Heal, Dynamo Global Router & NIXL UCX SGL Split (Phase 152)

> **Evidence Classification**: Primary source implementation audit (`implementation`) + deterministic stdlib offline simulation (`own-offline`).

## 1. 10 Primary Commit Audit Matrix (Zero-Overlap)

| # | Repository | Commit (`SHA`) | Date | Core Engineering Breakthrough & Code Path |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `NVIDIA/Megatron-LM` | [`f87f092`](https://github.com/NVIDIA/Megatron-LM/commit/f87f0920c5c755779209f10b7b7d72b13f810771) | 2026-10-14 | `Support routing replay with asynchronous scheduling` (`#7310`). |
| 2 | `NVIDIA/Megatron-LM` | [`80781ab`](https://github.com/NVIDIA/Megatron-LM/commit/80781ab3d959a7203be3f8ef743d2df69c993c33) | 2026-10-14 | `fix(mtp): preserve metric reduction groups across graph replays` (`#7768`). |
| 3 | `NVIDIA/Megatron-LM` | [`1e3c0c6`](https://github.com/NVIDIA/Megatron-LM/commit/1e3c0c640431880cb5ee049e348573ec7d6336bc) | 2026-10-13 | `Add the min-memory DSA attention kernels` (`#7152`). |
| 4 | `pytorch/pytorch` | [`bc7e151`](https://github.com/pytorch/pytorch/commit/bc7e151be297e99850b09dc3be15c95c165fa536) | 2026-10-30 | `dynamo: Add extra guards on dict length for correctness` (`#202374`). |
| 5 | `pytorch/pytorch` | [`1dbd304`](https://github.com/pytorch/pytorch/commit/1dbd304877e1297dc2d3bb578cb17fb3d03bf094) | 2026-10-30 | `[precompile] A Dynamo artifact imports the modules that register its custom ops` (`#202258`). |
| 6 | `kvcache-ai/Mooncake` | [`09a570d`](https://github.com/kvcache-ai/Mooncake/commit/09a570da6c9fd865b73943c3568fc6b2e5e2b546) | 2026-10-14 | `[Store] Self-heal dangling LOCAL_DISK replicas on the read path` (`#3889`). |
| 7 | `kvcache-ai/Mooncake` | [`dae4828`](https://github.com/kvcache-ai/Mooncake/commit/dae4828071f2f0db49c4678791847df710663d48) | 2026-10-14 | `[Bugfix][Store] Skip replicas on unloaded segments during standby restore` (`#4553`). |
| 8 | `LMCache/LMCache` | [`2ef13a3`](https://github.com/LMCache/LMCache/commit/2ef13a3dc118fc4b7777d56a3dc77694fc541f46) | 2026-10-19 | `[Feat] Emit a per-request prompt trace span from the MP server` (`#5817`). |
| 9 | `ai-dynamo/dynamo` | [`9f88a47`](https://github.com/ai-dynamo/dynamo/commit/9f88a474f0e1a4a9573be5fb11f5dd1744019806) | 2026-10-29 | `feat(global-router): Global Frontend with load- and session-aware pool selection` (`#17524`). |
| 10 | `ai-dynamo/nixl` | [`7077d02`](https://github.com/ai-dynamo/nixl/commit/7077d0230a1e6c993db81d251500c1f1a33eb602) | 2026-10-02 | `PLUGINS/UCX: Set split_batch_size=4096 in threadpool when SGL is enabled` (`#2329`). |

## 2. Deterministic Offline Verification (`own-offline`)

Run `python experiments/megatron_dsa_routing_mooncake_heal_dynamo_ucx_phase152/run.py --check`:
- **Megatron Min-Memory DSA Activation Saving**: **`58.33%`** (`1536.0 -> 640.0 MB`)
- **Mooncake Dangling LOCAL_DISK Replica Self-Heal**: **`12/12`** healed; Standby restore unloaded crashes: `3 -> 0`
- **Dynamo Global Frontend Load/Session-Aware TTFT Speedup**: **`1.5814x`** (`340.0 -> 215.0 ms`)
- **NIXL UCX SGL `split_batch_size=4096` Speedup**: **`1.4844x`** (`285.0 -> 192.0 us`)
