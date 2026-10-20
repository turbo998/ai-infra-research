# SGLang Runtime PD Disaggregation Role Switching (`role_switch.py`), `PPConsensusStore` Zero-Pickle ZMQ Replication, and Decode Incremental KV Offload (`DecodeKVCacheOffloadManager`)

> **中文版**: [sglang-pd-role-switch-pp-consensus-offload.md](sglang-pd-role-switch-pp-consensus-offload.md) | **Experiment**: [`experiments/sglang_pd_role_switch_offload_sim/README.md`](../../experiments/sglang_pd_role_switch_offload_sim/README.md)

### Executive Summary

SGLang introduces three foundational control- and data-plane mechanisms for Prefill-Decode (PD) disaggregation: (1) **Zero-KV-Reallocation Runtime Role Switching (`role_switch.py::handle_pd_role_switch`)**, which keeps the role-independent token KV pool intact and flips an idle instance between `prefill` and `decode` in **~180 ms** (**233.33x faster** than a 42 s cold container restart) after enforcing a 7-rule precondition table (`_reject_reason`) and pre-validating Decode CUDA Graph memory headroom (`available_graph_gb >= decode_cuda_graph_memory_gb`); (2) **`PPConsensusStore` (`pp_consensus_store.py`)**, which replaces unsafe Python `pickle` serialization over unauthenticated ZMQ `PUSH`/`PULL` sockets with `msgspec.Struct(array_like=True, frozen=True)` 4-tuple MsgPack frames, eliminating remote code execution (RCE) exposure and shrinking wire payload size by **72.31%** (`65 B -> 18 B`); and (3) **`DecodeKVCacheOffloadManager` (`decode_kvcache_offload_manager.py`)**, which seamlessly recovers unaligned prompt tail tokens (`origin_len - prefill_offloaded_len`), offloads incremental decode KV chunks at `offload_stride` boundaries, and filters `UnifiedSWAKVPool` sink pages (`swa_indices > 0`).

### Upstream Source Responsibility & Locations

| Module & File | Upstream Repo & Commit / Blob SHA | Core Engineering Responsibility |
| :--- | :--- | :--- |
| `python/sglang/srt/disaggregation/role_switch.py` | `sgl-project/sglang` commit `df13d5dce31abfeff0e701a62e8c39a136758e6c` / blob `cc820a4e586c32d96032dd38735508812d8d0729` | Implements `run_event_loop_supervisor`, `handle_pd_role_switch`, 7-rule `_reject_reason` validation, Decode CUDA Graph memory headroom checks, and `teardown_disaggregation`. |
| `python/sglang/srt/disaggregation/pp_consensus_store.py` | `sgl-project/sglang` commit `df13d5dce31abfeff0e701a62e8c39a136758e6c` / blob `5f346a2ff0bc9390fa66d322b1849c688de917d0` | Implements background ZMQ metadata replication from `PP > 0` ranks to `PP0` using strict `msgspec.msgpack` `_ReplicationOp` structs. |
| `python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py` | `sgl-project/sglang` commit `df13d5dce31abfeff0e701a62e8c39a136758e6c` / blob `6ee0838bef13f78fe19f36d630ccd91a8a1824a8` | Manages decode-side incremental KV cache offloading, `WeakKeyDictionary` request tracking, unaligned prompt tail recovery, and `UnifiedSWAKVPool` SWA page translation. |
| `python/sglang/srt/disaggregation/checksum.py` | `sgl-project/sglang` commit `df13d5dce31abfeff0e701a62e8c39a136758e6c` / blob `74d5900c9774d61c5c0b42f1ec3c0b48e13b8564` | Provides KV page checksum verification utilities for diagnosing cross-node RDMA transfer corruption. |

### Core Implementation Path, Memory Layout & Synchronization Contracts

1. **Zero-KV-Reallocation Role Switching (`role_switch.py:37-195`)**:
   - Pre-teardown failures (disabled flag, non-idle instance, unsupported transfer backend, staging buffer enabled, or `available_graph_gb < decode_cuda_graph_memory_gb`) return `safe_to_restore=True` so the router can immediately restore traffic to the existing role.
   - Once `teardown_disaggregation` begins, any exception latches `scheduler._pd_role_switch_unhealthy = True` and returns `safe_to_restore=False` (no in-place rollback).
2. **`PPConsensusStore` MsgPack Replication (`pp_consensus_store.py:25-145`)**:
   - Encodes `_ReplicationOp(kind, rank, key, value)` on the caller thread and validates exact struct shape on `PP0`'s ZMQ `PULL` socket, preventing arbitrary code execution on routable unauthenticated sockets.
3. **Decode Incremental KV Offload & Unaligned Tail Recovery (`decode_kvcache_offload_manager.py:85-195`)**:
   - Prefill offloads `prefill_offloaded_len = (len(origin_input_ids) // page_size) * page_size`. Decode starts from `prefill_offloaded_len + state.inc_len`, seamlessly coalescing the unaligned prompt tail with generated tokens at `offload_stride` multiples while keeping GPU KV slots allocated until `_release_finished_req`.

### Key Equations, Units & Engineering Assumptions

$$L_{\text{prefill\_off}} = \left\lfloor \frac{L_{\text{prompt}}}{P_{\text{page}}} \right\rfloor \cdot P_{\text{page}}, \quad \Delta L_{\text{aligned}}(t) = \left\lfloor \frac{(L_{\text{prompt}} + t - 1 - L_{\text{prefill\_off}}) - L_{\text{inc\_done}}}{S_{\text{stride}}} \right\rfloor \cdot S_{\text{stride}}$$

### Deterministic Offline Simulation & Quantitative Comparison

Validated via [`experiments/sglang_pd_role_switch_offload_sim/run.py`](../../experiments/sglang_pd_role_switch_offload_sim/run.py) and [`experiments/sglang_pd_role_switch_offload_sim/results.json`](../../experiments/sglang_pd_role_switch_offload_sim/results.json):

| Dimension | Metric / Scenario | Baseline Approach | SGLang (`df13d5dce3`) | Quantitative Finding |
| :--- | :--- | :--- | :--- | :--- |
| **PD Role Switch Latency** | Prefill $\to$ Decode flip | Cold process restart + KV alloc: `42,000 ms` | In-place role flip (`kv_pool_reallocated=False`): `180 ms` | **233.33x faster** role transition |
| **CUDA Graph Headroom Gate** | Need `4.0 GB`, avail `2.2 GB` | OOM crash after tearing down old role | Pre-check rejects with `safe_to_restore=True` | Zero-downtime traffic restoration |
| **`PPConsensusStore` Wire Size** | Single `_ReplicationOp` message | Python `pickle` dict: `65 Bytes` (RCE risk) | `msgspec.msgpack` 4-element array: `18 Bytes` | **72.31% smaller** wire frame, RCE-free |
| **Unaligned Prompt Tail Offload** | `L_prompt=230, page=64, stride=128` | `38` tail prompt tokens stranded | Coalesced into chunks `[192, 320)` & `[320, 448)` at steps 91 & 219 | **100% prefix continuity** in HiCache |

### Architectural Impact on Quality, SLO, Topology, Cost & Operations

- **Elastic PD Pool Rebalancing**: Enables sub-second shift of GPU capacity between Prefill and Decode pools as workload prefill-to-decode ratios fluctuate.
- **Safe Lifecycle & Memory Accounting**: Pre-checking `decode_cuda_graph_memory_gb` before `teardown_disaggregation` prevents half-torn-down zombie instances, while `WeakKeyDictionary[Req, OffloadedState]` prevents `rid` reuse races during asynchronous D2H transfers.
