# Megatron-LM Activation & Parameter Memory Lifecycle: `CheckpointWithoutOutput` 4-State Zero-Cycle Storage Sharing & `Megatron-FSDP` Frozen-Module Deferred Cleanup / Microbatch Overlap

> **中文版本**: [megatron-checkpoint-without-output-mfsdp-frozen-grad.md](megatron-checkpoint-without-output-mfsdp-frozen-grad.md) | **Back to Index**: [README.en.md](../../README.en.md) | [README.md](../../README.md)

## 1. Engineering Problem: Hidden Memory-Lifecycle and Stream-Wait Hazards in Large-Scale Training

In `NVIDIA/Megatron-LM`, fine-grained activation recomputation (`CheckpointWithoutOutput`) and Fully Sharded Data Parallelism (`Megatron-FSDP`) exhibited two subtle lifecycle bugs fixed in commits `d2c430ab485c6947701447e5d551d3aafaa39161` and `d37b3a8fdb37af1b3a1d75720636af4c323bd522`:

1. **Autograd Reference Cycles & Redundant Unpacking in `CheckpointWithoutOutput`**:
   - `CheckpointWithoutOutput` runs a region under `torch.no_grad()`, lets downstream ops consume the output, and then calls `discard_output_and_register_recompute` to free the output buffer (`untyped_storage().resize_(0)`) until backward reaches `hook_tensor`.
   - Previously, the `CheckpointWithoutOutput` facade directly held references to `ctx` and `outputs`: if backward was skipped or aborted, `output -> grad_fn -> ctx -> facade -> outputs` formed a **reference cycle across the autograd graph**, leaking GPU memory. Moreover, fine-grained activation offloading (`saved_tensors_hooks`) requires unpacking stashed inputs exactly once, and downstream views (such as `TE GroupedLinear`'s `inp.reshape()` + `torch.split()`) must see rebuilt storage without bumping `TensorImpl` version counters.
2. **Use-After-Free on Frozen `Megatron-FSDP` Units & Premature `ReduceScatter` Stream Waits**:
   - When an FSDP unit wraps only **frozen weights** (`_trainable_parameter_countdown.initial_value == 0`) and receives **no input gradients** (`all(grad is None for grad in grad_input)` while a separately sharded trainable child keeps the backward graph alive), PyTorch's `register_full_backward_hook` fires **before** internal backward finishes. Resizing the unsharded frozen weight storage to `0` immediately caused internal backward operations to read freed GPU memory!
   - Additionally, calling `finish_grad_sync()` (`current_stream().wait_stream(reduce_scatter_stream)`) at the end of every microbatch blocked the compute stream on non-final accumulation steps (`is_last=False`), preventing the last `ReduceScatter` of microbatch $m$ from overlapping the forward pass of microbatch $m+1$.

---

## 2. First-Hand Source Walkthrough: `_RecomputeRegion` & `Megatron-FSDP` Deferred Cleanup

### 2.1 `_RecomputeRegion` 4-State Automaton & `UntypedStorage` Zero-Copy Sharing (`megatron/core/tensor_parallel/random.py`)

Commit `d2c430ab485c6947701447e5d551d3aafaa39161` (`megatron/core/tensor_parallel/random.py` L773–1195) isolates each invocation into a self-contained `_RecomputeRegion` governed by a forward-only 4-state automaton `_RegionState`:

$$\texttt{RECORDED} \xrightarrow{\text{release()}} \texttt{RELEASED} \xrightarrow{\text{rebuild()}} \texttt{REBUILT} \xrightarrow{\text{backpropagate()}} \texttt{DONE}$$

| Mechanism | Source Implementation (`random.py` L838–1175) | Architectural Impact |
|---|---|---|
| **`_InputStash` (`L838–859`)** | Saves raw tensor inputs via `ctx.save_for_backward(*tensors)` on a dedicated dummy `autograd.Function` returning `torch.empty(0)`; read back once via `stash.grad_fn.saved_tensors` during `rebuild()` | Exposes inputs to `saved_tensors_hooks` (activation offloading) and CUDA Graph observers while **unpacking inputs at most once** |
| **`release()` & `_retained_storage_ptrs` (`L926–941`)** | Stores `output.detach()` aliases (breaking the `grad_fn` cycle), calls `storage.resize_(0)` only on storages not in `_retained_storage_ptrs`, and replaces `self._region` on the facade with **`weakref.ref(region)`** | Preserves outputs that alias retained inputs (e.g., TE `MakeExtraOutput`) and guarantees **zero reference-cycle leaks** if backward never runs |
| **`_share_rebuilt_storages()` (`L979–1000`)** | Swaps `StorageImpl` pointers at the `UntypedStorage` layer (below `TensorImpl`) once per unique underlying buffer | All downstream `view` / `reshape` / `split` tensors immediately observe the rebuilt data with **zero autograd version-counter bumps** |

### 2.2 `Megatron-FSDP` Frozen-Module Deferred Cleanup & Non-Final Microbatch Overlap (`d37b3a8fdb37`)

In commit `d37b3a8fdb37af1b3a1d75720636af4c323bd522`:
1. **Deferred Cleanup Stack for Frozen Modules (`fully_shard.py` L200–213 & `context.py` L99–104, L226–231)**:
   - When `_trainable_parameter_countdown.initial_value == 0`, `grad_checking_post_backward_hook` inspects `grad_input`: if `all(grad is None for grad in grad_input)`, it appends `(module, post_backward_hook)` to `context._delayed_post_backward_callbacks` instead of freeing the weights prematurely.
   - When autograd finishes the backward pass, `context.post_backward()` calls `_run_delayed_post_backward_callbacks()`, popping callbacks in **LIFO order (children before parents)** so frozen weights remain intact (`32 MiB`) throughout internal backward and are resized to `0 B` cleanly afterward.
2. **Skipping `finish_grad_sync()` on Non-Final Microbatches (`context.py` L215–224 & `fully_shard.py` L177–190)**:
   - `context.post_backward()` gates the stream wait on `if not self.caller_managed_grad_sync and self.is_last_microbatch:`.
   - Under `with microbatch(context, is_last=False):`, unsharded weights are still freed after every microbatch, but `current_stream().wait_stream(reduce_scatter_stream)` is skipped until the final microbatch (`is_last=True`), reducing stream waits from `4` to **`1`** across 4 microbatches and unlocking **`3` cross-microbatch `ReduceScatter -> Next Forward` overlap windows**.

---

## 3. Deterministic Verification (`experiments/megatron_ckpt_without_output_mfsdp_sim/run.py`)

Run `python experiments/megatron_ckpt_without_output_mfsdp_sim/run.py --check` to verify:
- `CheckpointWithoutOutput`: `64 MiB` main output drops to `0 B` on `release()` while the `16 MiB` input-aliased output is preserved; `unpack_count == 1` across duplicate hooks; downstream view sees `0x3000` (`64 MiB`) with `version == 0`; abandoned regions are garbage-collected (`abandoned_region_gc_collected == True`).
- `Megatron-FSDP`: Frozen weights remain `32 MiB` during internal backward (`frozen_patched_use_after_free == False`) and are freed in `["child_frozen_module", "parent_frozen_module"]` LIFO order; stream waits drop from `4` to `1` across 4 microbatches.

---

## 4. Primary Sources & Evidence Chain

| Source ID | Repository & Commit / File Path | Evidence Class |
|---|---|---|
| `megatron-ckpt-without-output-region-2026` | `NVIDIA/Megatron-LM` commit `d2c430ab485c6947701447e5d551d3aafaa39161` (`megatron/core/tensor_parallel/random.py`, blob `578e2f5244d4`) | `implementation` |
| `megatron-fsdp-frozen-cleanup-microbatch-2026` | `NVIDIA/Megatron-LM` commit `d37b3a8fdb37af1b3a1d75720636af4c323bd522` (`megatron_fsdp/experimental/context.py`, blob `3013d39e3a12`; `fully_shard.py`, blob `527efff8585c`) | `implementation` |
| `megatron-fsdp-schedule-plan-cleanup-2026` | `NVIDIA/Megatron-LM` commit `d37b3a8fdb37af1b3a1d75720636af4c323bd522` (`megatron/core/models/common/model_chunk_schedule_plan.py`, blob `9e78dc145827`) | `implementation` |
