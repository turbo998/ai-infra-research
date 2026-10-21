# TensorRT-LLM CuTe DSL Uneven `next_n` DSA MQA Logits Split, KV Transfer Backend Contract & MPI Shutdown ACK, and vLLM WatchDog

> **Upstream Source Commits (2026-10-21)**:
> - `NVIDIA/TensorRT-LLM` PR `#19976` (`033f2ea0205341b2f3ad55bbeb5593427030ec78`): `[feat] Default-enable CuTe DSL paged MQA logits on SM100f and support uneven next_n split (#19976)`
> - `NVIDIA/TensorRT-LLM` PR `#20117` (`a9000fe10ecd62973dcf3670573419299b390acb`): `[Disagg] Define the formal KV cache transfer Backend contract (#20117)`
> - `NVIDIA/TensorRT-LLM` PR `#20608` (`16d26df5a7675b0aa13a213305c5dd59e504e9d5`): `[Fix] Reliability fix for RemoteMpiCommSessionServer shutdown (#20608)`
> - `vllm-project/vllm` PR `#55700` (`1c879919786497217b3b5fb0258d1d72f2442cb0`): `[Misc] Add process-wide ThreadedWatchDog for diagnosing worker/engine hangs (#55700)`

---

## 1. Architectural Problem Statement

Across multi-token prediction (MTP) sparse attention indexing, disaggregated KV cache transport, and multi-process worker orchestration, three reliability and efficiency challenges arise:

1. **Redundant Paged KV Re-Reads for Prime/Uneven `next_n` MTP Windows in DSA**: The CuTe DSL paged MQA logits kernel natively supports `kernel_atoms = (1, 2, 3)`. When `next_n` is a prime number such as `5` or `7`, caller-side exact division (`_pick_dsl_expand`) falls back to `atom = 1, expand_factor = next_n`, forcing the GPU to stream the entire paged KV cache $5\times$ or $7\times$ per verification step.
2. **Undelivered-Row Memory Reuse Hazards & Lost Shutdown Signals in Disaggregated Serving**:
   - In disaggregated KV transfer, reaching a logical `Outcome` (such as partial delivery alongside a peer cancellation fault) does not guarantee that in-flight RDMA/NVLink DMA engines have stopped touching undelivered destination slots;
   - `RemoteMpiCommSessionServer` previously stopped polling its ZMQ `PAIR` control socket while waiting several seconds for worker threads to tear down, causing `stop_server_main`'s fire-and-forget stop message to be dropped and leaving remote MPI sessions hung.
3. **Undiagnosed vLLM Worker Deadlocks vs. Idle Ring-Buffer False Alarms**: Hung worker processes need automatic all-thread Python stack dumps, but naive timeout polling triggers false-positive dumps whenever `MessageQueue.acquire_read` / `acquire_write` blocks on an idle shared-memory ring buffer or floods logs with repeated dumps for a single hang.

---

## 2. Mathematical Formulation & System Design

```
+--------------------------------------------------------------------------------------------------+
|  TRT-LLM CuTe DSL Uneven next_n (#19976) + KV Contract (#20117/#20608) + vLLM WatchDog (#55700)  |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|  [1] In-Kernel Uneven next_n Split (_pick_dsl_split & _atom_ctx_offsets, PR #19976)              |
|      When atom == 1 and next_n not divisible by {2, 3}:                                          |
|        num_atoms = ceil(next_n / 3),  in_kernel_atom = ceil(next_n / num_atoms)                  |
|        Taken iff waves_split < waves_atom1 and waves_atom1 >= 16                                 |
|        Causal stagger: offset_i = next_n - ((i+1)*base + min(i+1, rem))                          |
|        * next_n=5 -> num_atoms=2, atom=3, offsets=[2, 0] (reads KV 2x instead of 5x: -60.0%)     |
|        * next_n=7 -> num_atoms=3, atom=3, offsets=[4, 2, 0] (reads KV 3x instead of 7x: -57.14%) |
|                                                                                                  |
|  [2] KV Transfer Backend Contract (#20117) & MPI SHUTDOWN_ACK Handshake (#20608)                 |
|      SlotRegion / LayerGroupRows / View / Outcome(delivered_masks, fault) / Attempt.quiesced()   |
|      Undelivered row memory reusable IFF Attempt.quiesced() == True                              |
|      RemoteMpiCommSessionServer polls control socket every 0.1s during teardown & replies ACK    |
|                                                                                                  |
|  [3] vLLM ThreadedWatchDog (#55700)                                                              |
|      `with get_watch_dog().disable():` refreshes last_feed_time during idle shm_broadcast waits  |
|      One-shot dump per stall: triggers iff (now - last_feed > timeout) & (last_log < last_feed)  |
+--------------------------------------------------------------------------------------------------+
```

### 2.1 CuTe DSL In-Kernel Uneven `next_n` Split & Causal Stagger (PR `#19976`)

PR `#19976` default-enables `use_cute_dsl_paged_mqa_logits=True` on `sm_100f` and introduces `_pick_dsl_split`: when `next_n > 1` has no divisor in $\{2, 3\}$, it computes:

$$\text{num\_atoms} = \left\lceil \frac{\text{next\_n}}{\max(\text{kernel\_atoms})} \right\rceil = \left\lceil \frac{\text{next\_n}}{3} \right\rceil, \quad \text{atom} = \left\lceil \frac{\text{next\_n}}{\text{num\_atoms}} \right\rceil$$

With $\text{tiles} = \lceil \text{max\_ctx} / 256 \rceil$, the in-kernel split is taken iff $\text{waves}_{\text{split}} < \text{waves}_{\text{atom1}}$ and $\text{waves}_{\text{atom1}} \ge 16$. Each atom $i \in [0, \text{num\_atoms})$ receives a causal context-shortening offset:

$$\text{offset}_i = \text{next\_n} - \big(i \cdot \text{base} + \min(i, \text{rem}) + \text{base} + \mathbb{I}[i < \text{rem}]\big)$$

where $\text{base}, \text{rem} = \operatorname{divmod}(\text{next\_n}, \text{num\_atoms})$. For `next_n = 7`, $\text{num\_atoms} = 3, \text{atom} = 3$, and the per-atom offsets are **`[4, 2, 0]`**, cutting paged KV reads from $7\times$ to $3\times$ (**`-57.14%`**) and delivering **`1.04x–2.60x` speedup on GB300** when $\text{waves}_{\text{atom1}} \ge 24$.

### 2.2 Disaggregated KV Transfer `Backend` Contract & MPI Teardown ACK (PR `#20117` / PR `#20608`)

PR `#20117` defines the formal disaggregated KV transfer contract in `tensorrt_llm/_torch/disaggregation/base/backend.py`:
- **`SlotRegion`, `LayerGroupRows`, and `View`**: A backend registers contiguous slot regions `(address, nbytes)` for supported `MemoryKind`s and receives immutable `View` snapshots containing per-group `object_keys` and slot `addresses`. Any unrecognized row raises `InvalidSubmission` before any peer notification or memory access escapes.
- **`Outcome` vs. `Attempt.quiesced()`**: `Outcome(delivered_masks, fault)` reports read-only per-group boolean delivery masks alongside an optional `fault`. While delivered rows are immediately final and readable, **undelivered rows may have been partially written and their backing memory may be reused only after `Attempt.quiesced()` returns `True`**.

PR `#20608` makes `RemoteMpiCommSessionServer` poll its control socket every `PENDING_POLL_SECONDS = 0.1` seconds while waiting for worker threads to finish and reply with `SHUTDOWN_ACK`, while `stop_server_main` reconnects and retries (`STOP_ACK_TIMEOUT_SECONDS = 1.0`, `STOP_MAX_ATTEMPTS = 300`) until `SHUTDOWN_ACK` arrives.

### 2.3 vLLM `ThreadedWatchDog` One-Shot Dump & Idle Window Suspension (PR `#55700`)

PR `#55700` introduces `ThreadedWatchDog` (`timeout = 300s`, `check_interval = 10s`):
- **Idle Window Suspension (`with wd.disable():`)**: Wraps `MessageQueue.acquire_read` and `acquire_write` waits in `shm_broadcast.py`. While `active == False`, `_check_loop` continuously sets `last_feed_time = now` so idle waits never trigger false dumps and a fresh timeout window begins upon re-enabling.
- **One-Shot Dump per Stall**: Triggers `faulthandler.dump_traceback(file=f, all_threads=True)` only when `now - last_feed_time > timeout` and `last_timeout_log < last_feed_time`, guaranteeing at most one dump per stall until `WorkerProc._execute_worker_rpc` feeds the watchdog again.

---

## 3. Deterministic Verification & Benchmark Summary

Running `python experiments\trtllm_dkg_dsa_mqa_kv_contract_vllm_watchdog\run.py --check` validates all invariants:

| Dimension | Configuration | Baseline | New Mechanism | Verified Impact |
|---|---|---|---|---|
| **CuTe DSL `next_n=5` Split (`#19976`)** | GB300 (`148 SMs`), `B=32, max_ctx=16384` | `atom=1` reads KV `5x` (`139` waves) | **`num_kernel_atoms=2, atom=3, offsets=[2,0]`** (`56` waves) | **`-60.00%` KV cache reads** |
| **CuTe DSL `next_n=7` Split (`#19976`)** | GB300 (`148 SMs`), `B=32, max_ctx=16384` (`>=24` waves) | `atom=1` reads KV `7x` (`194` waves) | **`num_kernel_atoms=3, atom=3, offsets=[4,2,0]`** (`84` waves) | **`-57.14%` KV reads, `1.04x–2.60x` GB300 speedup** |
| **Low-Wave Guard (`#19976`)** | GB300 (`148 SMs`), `next_n=7, B=1, max_ctx=4096` (`1` wave `<16`) | Unconditional split (`0.87x` regression) | **`fallback_atom1_low_waves`** | **Prevents wide-tile overhead at small batch** |
| **KV Contract & MPI ACK (`#20117/#20608`)** | Partial delivery + `peer_cancelled` / 2.5s worker teardown | Premature slot reuse / dropped ZMQ stop | **`quiesced()` memory gate + `0.1s` `SHUTDOWN_ACK`** | **Zero memory corruption, `<= 0.1s` ACK latency** |
| **vLLM `ThreadedWatchDog` (`#55700`)** | 590s idle `shm_broadcast` wait + two `>300s` RPC stalls | False idle dumps or log flooding | **`0` idle dumps, `2` exact one-shot dumps** | **Eliminates false positives and repeated dump spam** |

---

## 4. Production Guidance & Boundaries

- **When to use**: DSA sparse attention on Blackwell (`sm_100f`) with prime/uneven speculative decoding lengths (`next_n = 5, 7`), custom RDMA/NIXL KV transfer backends in TRT-LLM, and production hang diagnostics in multi-GPU vLLM deployments.
- **Boundaries**: `_pick_dsl_split` activates the in-kernel uneven split only when `waves_atom1 >= 16` and `next_n` is not divisible by `2` or `3`; evenly divisible windows (such as `next_n = 6`) continue to use exact caller-side expansion (`expand_factor = 2, atom = 3`).
