# Megatron-LM LayerShardedMuon Streaming Batched Newton-Schulz & TE 256-Token Alignment, Mooncake Adaptive RDMA Congestion Control / Tenant Quota Policy, and NIXL Immediate `xfer.complete` Tracing

> **中文版本**: [megatron-lsmuon-ns-stream-te-align-mooncake-acc-quota-nixl.md](megatron-lsmuon-ns-stream-te-align-mooncake-acc-quota-nixl.md)  
> **Question ID**: `megatron-lsmuon-ns-stream-te-align-mooncake-acc-quota-nixl` (`priority: 51`, Phase 42)  
> **Offline Experiment**: [`experiments/megatron_lsmuon_stream_mooncake_acc_quota_nixl/run.py`](../../experiments/megatron_lsmuon_stream_mooncake_acc_quota_nixl/run.py) · [`summary.json`](../../experiments/megatron_lsmuon_stream_mooncake_acc_quota_nixl/summary.json)  
> **Evidence Tier**: `implementation` (primary source audit) + `own-offline` (deterministic Python stdlib model)

---

## 1. Architectural Context & Problem Statement

Across distributed MoE training and disaggregated KV cache storage/RDMA transport planes, five upstream commits merged on `2026-10-21` address critical memory, alignment, congestion, multi-tenant isolation, and observability gaps:

1. **Megatron-LM `LayerShardedMuon` Transient Memory Spike & Single-Rank Fallback (`#8379`)**: Under large MoE models, a home rank or single-rank domain (`gtp_remat x tp == 1`) owns hundreds of identically shaped matrices. Materializing all FP32 Nesterov inputs and holding all Newton-Schulz (NS) orthogonalized outputs simultaneously creates multi-GiB transient memory spikes. Moreover, single-rank domains with `set_param_ns_homes()` previously fell back to parent `TensorParallelMuon`'s unbatched per-parameter loop.
2. **Megatron-LM Transformer Engine `GroupedLinear` 256-Token Alignment (`#8184`)**: CuTeDSL/CUTLASS grouped GEMMs require expert token splits aligned to `256` tokens (`TE_GROUPED_GEMM_EXPERT_TOKEN_ALIGNMENT = 256`) passed directly via `GroupedLinear(..., split_alignment=256)`.
3. **Mooncake Transfer Engine Adaptive RDMA Congestion Control Core (`#4197`)**: High-concurrency RDMA KV transfers need lock-free device/route window admission that reacts to drain-time backlog and RNR receiver pressure while fencing stale feedback across link resets.
4. **Mooncake Store `TenantQuotaPolicy` Upsert Accounting (`#4809`)**: When a tenant upserts an existing key whose new memory replica is still `PROCESSING`, dropping the old replaced object's charge before the new write settles allows in-flight upserts to double-spend physical pool memory.
5. **NIXL Missing `nixl::xfer.complete` Markers on Immediate Completions (`#2462`)**: When `postXferReq` completed synchronously (`status == NIXL_SUCCESS`), it returned without emitting `nixl::xfer.complete`, dropping 17 out of 50 completion trace events in `tracing_nsys` UCX runs.

---

## 2. Megatron-LM `LayerShardedMuon` Streaming `_iter_ns` & TE `split_alignment=256`

### 2.1 Chunk-Streamed Newton-Schulz (`layer_sharded_muon.py:526-574`, `764-775`)

In `NVIDIA/Megatron-LM` commit `6c956376e3ef22528413ff35412ad8b938272986` (PR `#8379`), `LayerShardedMuon` factors `_run_ns` into the generator `_iter_ns(self, full_by_k, load=lambda t: t)`:
- **Peak Transient Memory Reduced from $O(N_{\text{params}})$ to $O(\text{ns\_batch\_size})$**: For replicated parameters (`plan.replicated`), `_iter_ns(replicated, load=ns_input)` materializes the FP32 Nesterov input (`p.grad.lerp(buf, beta).float()`) only for the active chunk of `ns_batch_size` matrices, runs batched `newton_schulz(..., use_syrk=self.use_syrk)`, applies `orth.mul_(scale).mul_(self.extra_scale_factor)` in-place (preserving bitwise rounding parity with `TensorParallelMuon`), updates each parameter immediately via `_apply_update`, and deletes `orth`. For 128 `[2048, 2048]` matrices at `ns_batch_size=8`, peak transient memory drops **16.0x** (`4096.0 MiB` $\to$ `256.0 MiB`).
- **Single-Rank Domain Batched NS**: Single-rank `(gtp_remat x tp)` domains with `set_param_ns_homes()` now route through `plan.replicated` and `_iter_ns` rather than falling back to unbatched `super().step()`.
- **Separate `_run_ns` Call for `tp_complete` (`lines 845-858`)**: `GTP_REMAT`-only parameters (`plan.tp_complete`) are orthogonalized in their own `_run_ns` call so TP replicas never land in differently sized batch chunks across TP columns.

### 2.2 Transformer Engine `TE_GROUPED_GEMM_EXPERT_TOKEN_ALIGNMENT = 256` (`#8184`)

Commit `45b68f914cf15eae6eaa40117fe94cd6a3972d4c` (PR `#8184`) defines `TE_GROUPED_GEMM_EXPERT_TOKEN_ALIGNMENT = 256` in `megatron/core/extensions/transformer_engine.py`, passes `split_alignment=256` to `te.pytorch.ops.GroupedLinear` when supported by its signature, and aligns MoE expert token splits to 256 across `experts.py`, `shared_experts.py`, and `token_dispatcher.py`.

---

## 3. Mooncake Adaptive RDMA Congestion Control (`#4197`), `TenantQuotaPolicy` (`#4809`), and NIXL Trace Fix (`#2462`)

### 3.1 Mooncake Adaptive RDMA Congestion Control (`adaptive_congestion_control.cpp`)

In `kvcache-ai/Mooncake` commit `3b08b77168870d265e32b6cfeab661f037426b9c` (PR `#4197`), `mooncake::adaptive_congestion_control` implements:
- **Two-Stage `tryAcquire` with Generation Fencing & Empty-Domain Oversized Slice Rule**: `tryAcquire` reserves bytes on `path.device` then `path.route`, rolling back `path.device` if `path.route` defers, and returning `Decision::kAvoid` (`AdmissionReason::kStaleGeneration`) if either generation changed. `tryReserveBytes` checks `if (current != 0 && (bytes > limit || current > limit - bytes)) return false;`, allowing a single oversized slice (`bytes > window_bytes_`) to proceed when `inflight_bytes_ == 0`.
- **Drain-Time & RNR AIMD (`observeEpoch`)**: With defaults `min_window_bytes = 4 MiB`, `max_window_bytes = 64 MiB`, `target_drain_time_ns = 2 ms`, `high_pressure_epochs = 2`, and `low_pressure_epochs = 3`:
  - `high_pressure` (`receiver_pressure != 0` or `drain_ns > 2 ms`) for 2 consecutive epochs halves `window_bytes_` down to `min_window_bytes` (`PathState::kCongested`). RNR (`kReceiverPressure`) is recorded only on the `route` domain so a single slow receiver does not throttle unrelated routes sharing the local NIC (`lines 176-181`).
  - `low_pressure` (`successes > 0` and `drain_ns < 1 ms`) for 3 consecutive epochs increases `window_bytes_` by `min_window_bytes / 4 = 1 MiB` up to `64 MiB` (`PathState::kHealthy`).

### 3.2 Mooncake Store `TenantQuotaPolicy` (`quota_policy.cpp`)

In `kvcache-ai/Mooncake` commit `44c4d9cf4ace3ce2955312a2467b1305d910dd48` (PR `#4809`), `TenantQuotaPolicy` tracks two per-key words in `owner_state().policy_words`:
- **`Held(ctx)` (`policy_words[0]`)**: Bytes reserved upfront via `OnGrow` (`AccountOf(ctx).TryCharge(bytes)`).
- **`Carried(ctx)` (`policy_words[1]`)**: Charge of a replaced object recorded in `OnReplace`. During `OnWriteRelease`, `Carried(ctx)` is retained as long as `HasProcessingMemoryReplica(metadata)` is true (`owed = MemoryCharge(metadata) + Carried(ctx)`), and is released only after the replacement replica finishes processing.
- **Slab Eviction (`OnMaintenance`)**: When `used_ratio > eviction_high_watermark_ratio`, evicts `target_bytes = (used_ratio - max(0, watermark - eviction_ratio)) * effective_quota_bytes`.

### 3.3 NIXL Immediate `nixl::xfer.complete` Trace Marker (`nixl_agent.cpp`)

In `ai-dynamo/nixl` commit `f8248873bd5b8d3609ba1448fa2f623aefb361cf` (PR `#2462`), `nixlAgent::postXferReq` (both initial post and repost) and `nixlAgent::releaseXferReq` emit `NIXL_TRACE_MARK(data->tracer_.get(), "nixl::xfer.complete", nixl::trace::Kind::Metadata)` whenever `status == NIXL_SUCCESS`, restoring `50 / 50` `xfer.post` $\leftrightarrow$ `xfer.complete` event pairing in `tracing_nsys`.

---

## 4. Deterministic Offline Verification (`own-offline`)

Running `python experiments/megatron_lsmuon_stream_mooncake_acc_quota_nixl/run.py --check` verifies all four invariants (`16.0x` peak transient memory reduction in `LayerShardedMuon`, Mooncake ACC `4 -> 6 -> 4 MiB` window trajectory and empty-domain oversized slice admission, `TenantQuotaPolicy` `900 B` in-flight upsert charge and `220 B` slab eviction target, and `50 / 50` NIXL completion trace pairing).
