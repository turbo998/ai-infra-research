# Megatron-LM DynamicInferenceEngine Async Payload Staging, Idle Rerun Strict Checkpoint Load & veRL HDO Optimizer Offload Preservation

> **中文版本**: [megatron-dynamic-engine-payload-staging-verl-hdo-offload.md](megatron-dynamic-engine-payload-staging-verl-hdo-offload.md) | **Back to Index**: [README.en.md](../../README.en.md)

## 1. Problem Statement & Architectural Context

In large-scale multimodal (VLM) inference and reinforcement learning post-training (RL rollout + Megatron training switching), three bottlenecks and correctness traps arise across **completed-request payload staging**, **strict distributed checkpoint loading**, and **`HybridDeviceOptimizer` (`HDO`) memory offloading**:

1. **`DynamicInferenceEngine` Synchronous Payload Staging Stalls Decode Loop & Retains VLM GPU Memory (`NVIDIA/Megatron-LM` `#8169`)**:
   - Upon request completion, `RequestPayloadStager.stage()` writes bulky request artifacts (logprobs, MoE `routing_indices`, token IDs, and VLM media tensors) to external storage so the ZMQ reply only carries compact `payload_stage_metadata`.
   - Running `stage()` synchronously inside the engine bookkeeping step stalls GPU decoding whenever slow network writes occur (e.g., two `45.0 ms` writes on one `12.0 ms` decode step inflate step latency to **`102.0 ms`**, an **`8.50x`** spike).
   - Conversely, offloading `stage()` to background threads while keeping the `DynamicInferenceRequest` alive leaks GPU memory if `request.prompt_tokens` and VLM `request.media_tensors` remain on CUDA, and calling ZMQ `send_multipart` from worker threads violates ZMQ socket thread-affinity.
2. **Strict Distributed Checkpoint Load Fails When `rerun_state_machine` Is `NOT_RUNNING_YET` (`NVIDIA/Megatron-LM` `#7222`)**:
   - Checkpoints saved with cached save plans include a `rerun_state_machine` shard even when the state machine is idle (`RerunState.NOT_RUNNING_YET`).
   - Because `rerun_state_machine.validate_state_dict(...)` returns `False` for an idle state machine, `checkpointing.py` previously omitted `rerun_state_machine` from `gen_sd`, causing strict checkpoint loading to fail on the unexpected shard key.
3. **veRL `offload_megatron_optimizer` Corrupts `HybridDeviceOptimizer` (`HDO`) Device Placement (`volcengine/verl` `#8286` / `#8237`)**:
   - Megatron-LM's `HybridDeviceOptimizer` (`HDO`) partitions its internal `sub_optimizers` into a CPU half and a GPU-resident half according to `optimizer_offload_fraction`.
   - Previously, `verl/utils/megatron_utils.py` iterated through `hdo.sub_optimizers` inside `offload_megatron_optimizer` and moved all state tensors (`exp_avg`, `exp_avg_sq`) to CPU, violating the GPU sub-optimizer's fused kernel device invariant on the next training step (`#8237`).

---

## 2. First-Hand Code Evidence & Implementation Mechanics

### 2.1 `DynamicInferenceEngine` In-Place Host Detach + Thread-Safe Async Staging (`894b5114e3`)

In `megatron/core/inference/inference_request.py` (blob `4d6acc5920`, lines `1398–1455`) and `megatron/core/inference/engines/dynamic_engine.py` (blob `5fab3275d8`, lines `583–590, 785–794, 1859–1958`):

1. **Synchronous D2H Replacement During Bookkeeping**:
   `OffloadedRequestPayload.from_request(request)` replaces `request.prompt_tokens` with `request.prompt_tokens.cpu()` and replaces every tensor in `request.media_tensors` with `tensor.detach().cpu()` before spawning the async task. Thus, a finished request kept alive while its payload is staged holds **`0 B` of GPU memory**, and `_serialize_finished_request` strips 8 VLM input keys (`imgs`, `image_embeddings`, etc.) from wire replies.
2. **Worker-Thread `asyncio.to_thread` Write + Event-Loop ZMQ Send**:
   `_stage_and_reply` awaits `asyncio.to_thread(stager.stage, ...)` so slow storage writes never pause GPU decoding, then resumes on the event-loop thread to call `self.socket_for_receiving_requests.send_multipart(...)`.
3. **Backlog Telemetry & `reset()` Drain Guard**:
   The engine tracks `self._staging_tasks` and `self._staging_tasks_peak` (logged via `_staging_log_suffix()`) and raises `RuntimeError` in `reset()` if `self._staging_tasks` is non-empty before `_drain_staging_tasks()`.

### 2.2 `checkpointing.py` Idle `rerun_state_machine` Strict Load (`e564873e9b`)

In `megatron/training/checkpointing.py` (blob `7fd3d41591`, lines `2926–2941`), `checkpointing.py` checks `idle_rerun_state = isinstance(saved_rerun_state, dict) and saved_rerun_state.get("state") == RerunState.NOT_RUNNING_YET`. When `restore_rerun_state or idle_rerun_state` holds, it populates `gen_sd_rerun_state` for strict load compatibility while setting `ignore_rerun_state = not restore_rerun_state` (`True` when idle).

### 2.3 veRL `offload_megatron_optimizer` Skipping `HDO` Sub-Optimizers (`472dd2153a`)

In `verl/utils/megatron_utils.py` (blob `e47ec921b8`, lines `678–704`), `offload_megatron_optimizer` still runs `offload_megatron_copy_params(_opt)` to free FP32 copy parameters on GPU, but skips `_opt.optimizer.state` migration when `_opt.optimizer` has `("sub_optimizers", "inner_param_to_orig_param", "state")`.

---

## 3. Deterministic Offline Experiment & Quantitative Findings

Verified via [`experiments/megatron_payload_staging_verl_hdo_offload/run.py`](../../experiments/megatron_payload_staging_verl_hdo_offload/run.py) (`own-offline`):

| Evaluation Dimension | Pre-Fix Baseline | Verified Implementation | Architectural Impact |
|---|---|---|---|
| **Megatron Dynamic Engine Max Step Latency (`12 ms` step + `45 ms` write)** | Sync write stalls decode step to **`102.0 ms`** (`414.0 ms` total) | Async `to_thread` keeps decode step at **`12.0 ms`** (`144.0 ms` total) | **`8.50x`** lower peak decode step latency; **`270.0 ms`** GPU stall saved |
| **GPU Memory Retained by In-Flight Staged Requests (`peak=4`)** | Retains **`146.0 MiB`** of VLM media & prompt tensors on GPU | In-place `.cpu()` in `from_request` retains **`0.0 MiB`** on GPU | Decouples slow storage I/O backpressure from GPU memory capacity |
| **Idle `RerunState.NOT_RUNNING_YET` Strict Checkpoint Load** | Fails (`False`) on `unexpected_keys=['rerun_state_machine']` | Succeeds (`True`) with `ignore_rerun_state=True` | Preserves cached save-plan shards under strict dist-checkpoint load |
| **veRL `HybridDeviceOptimizer` (`fraction=0.5`) Next-Step FusedAdam** | Moves GPU sub-optimizer state to `cpu` -> **Crashes (`True`)** | Offloads `16 GiB` copy params, keeps GPU sub-opt on `cuda` (**`0` crashes**) | Eliminates `#8237` device-mismatch crash after RL rollout offload |

---

## 4. Architect Takeaways & Falsifiable Questions

1. **Three-Stage Pattern for Async Inference Output Offloading**: (a) Synchronously detach and move device tensors (`prompt_tokens`, `media_tensors`) to CPU on the bookkeeping thread, (b) execute blocking storage I/O on a worker thread via `asyncio.to_thread`, and (c) serialize and transmit over ZMQ after returning to the single event-loop thread.
2. **Respect Self-Managed Hybrid Optimizers During RL Offload**: External memory managers in hybrid RL frameworks (`veRL`, `slime`) must not mutate internal `sub_optimizers` device placement inside `HybridDeviceOptimizer`.
