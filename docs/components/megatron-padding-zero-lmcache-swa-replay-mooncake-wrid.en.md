# Megatron-LM CUDA-Graph Dynamic Padding Zeroing, TE Compiled FP8 Weight Caching, LMCache SWA Prefix-Replay Alignment & Mooncake/NIXL Fabric Guardrails

> **Phase 66 · Priority 75 · Track C (Distributed Training, Compiled FP8, KV Cache Offloading & RDMA/UCX Transport)**  
> **Upstream Commits Covered**: `NVIDIA/Megatron-LM@f9d9bdcc6e09`, `NVIDIA/Megatron-LM@92641a421f40`, `NVIDIA/TransformerEngine@9ea05cf2e8dc`, `LMCache/LMCache@a0ff4701284d`, `kvcache-ai/Mooncake@dae22d472e31`, `kvcache-ai/Mooncake@b66bb2b5f650`, `ai-dynamo/nixl@f3673f6be3f4`, `ai-dynamo/nixl@70badd6aad36`  
> **Companion Deterministic Experiment**: `experiments/megatron-padding-zero-lmcache-swa-replay-mooncake-wrid/run.py` / `summary.json`

---

## 1. Architectural Context & Core Engineering Problems

Across large-scale MoE training, CUDA-graphed quantized dynamic inference, and cross-node KV cache RDMA/UCX fabrics, mismatches between static graph/buffer assumptions and dynamic runtime state cause severe numerical quantization degradation, out-of-bounds KV writes, or ring-buffer slot corruption:

1. **Megatron-LM Frozen Python Slice Leaving Padding Rows Unzeroed into FP8/FP4 `amax` Under CUDA Graph Replay**: In `DynamicInferenceContext`, CUDA graphs are captured with full dummy token batches (where `padding_slice` is empty). Because a Python slice is frozen at capture time, replaying the graph with fewer real tokens (e.g., `real_token_count = 72 < 128`) leaves rows `[72, 128)` untouched. Their stale activation values feed directly into the per-tensor `amax` reduction of downstream FP8/FP4 GEMMs, crushing quantization dynamic range for real tokens.
2. **TransformerEngine `torch.compile` Incompatibility with FP8 Weight Caching (`is_first_microbatch`)**: Previously, `Linear` disabled `torch.compile` whenever `self.cache_weight and self.fp8` was active because updating the cached FP8 workspace in place on the first microbatch violated the functional custom op contract (`mutates_args=()`).
3. **LMCache Out-of-Bounds Retrieve Under vLLM Sliding-Window Attention (SWA) Bounded Prefix Replay**: When vLLM enables SWA bounded prefix replay (`prefix_replay_tokens > 0`, e.g., DeepSeek V4.1), the vLLM scheduler floors external connector hits to a scheduler block boundary before allocating GPU blocks. On a 12-token full-prompt hit (`chunk_size = 4`), LMCache's classic `ret - 1 = 11` rule caused vLLM to floor `11 -> 8` and allocate only `2` blocks (`8` token slots), while LMCache still scheduled a retrieve for `[0, 12)`—overflowing the allocation by `4` tokens and leaking the lookup lock on `[8, 12)`.
4. **Mooncake TENT RDMA `sendNotification` Pre-Post `wr_id` Increment Overwriting In-Flight Slot 0 & NVLink VMM/PID-Namespace Failures**: `RdmaEndPoint::sendNotification` indexes its `256` notification send slots via `wr_id % 256`. Legacy code executed `wr.wr_id = notify_send_wr_id_++` *before* `ibv_post_send()`. If `255` notifications (`wr_id 0..254`) were in flight and the 256th `ibv_post_send()` failed (`EINVAL`), `notify_send_wr_id_` advanced to `256` while `notify_pending_count_` remained `255`; the next retry then used `wr_id = 256` (`256 % 256 == 0`), overwriting in-flight slot `0`! Meanwhile, `IntraNodeNvlinkTransport` lacked actionable diagnostics when `cudaIpcGetMemHandle` failed on `cuMemCreate` VMM buffers (`expandable_segments:True`) or when `cudaIpcOpenMemHandle` failed with `cudaErrorDeviceUninitialized` due to colliding container PIDs.
5. **NIXL UCX Notification Missing Request Tag (`treq`) & Telemetry Omissions on Repost/Release**: UCX wire notifications carried the 26-byte `TraceContext` (`tctx`) but lacked the initiator's `(request_id, execution_number)` pair, preventing receiver-side span correlation. Additionally, `nixl_agent` failed to record completion telemetry when transfers were first observed completed during `postXferReq` repost or `releaseXferReq`, and prematurely restarted request timers even when a repost was rejected with `NIXL_ERR_REPOST_ACTIVE`.

---

## 2. First-Hand Implementation Paths & Fixes

### 2.1 Megatron-LM Device-Pointer Padding Zeroing Kernel & Global-Batch K3 Quantile Balancing (`Megatron-LM@f9d9bdcc6e09`, `Megatron-LM@92641a421f40`)

- **CUDA-Graph Safe `zero_padding_rows_kernel`**: Added in `megatron/core/inference/contexts/zero_padding_rows_kernel.py` and wired into `DynamicInferenceContext.zero_padding_tokens(tensor, sequence_parallel_rank)`. The Triton kernel `_zero_padding_rows_kernel` reads the `[1]` `int32` GPU tensor `gpu_view.real_token_count` on device (`real_token_count = tl.load(real_token_count_ptr)`) and writes zeros to rows where `rows + first_token_idx >= real_token_count` without reading the activation. In our deterministic scenario (`128` captured rows, `72` real tokens on replay, real `amax = 2.0`, stale padding `amax = 64.0`), device-pointer zeroing restores the FP8 E4M3 scale from `7.0` to `224.0` (**`32.0x` quantization precision recovery**) across all Sequence Parallel ranks (`first_token_idx = sequence_parallel_rank * tensor.shape[0]`).
- **Global-Batch K3 Quantile Balancing**: In `megatron/core/transformer/moe/{router.py,moe_utils.py}` and `finalize_model_grads.py`, routers configured with `qb_estimation_scope == "global_batch"` accumulate token routing counts into a non-persistent `qb_histogram [num_experts, num_bins]` (`int32`) buffer. At the end of each global batch, `_update_router_expert_bias_with_quantile` stacks all router histograms, executes a single coalesced `all_reduce` per reduction group, updates `expert_bias` and `qb_bin_bounds`, and invokes `mark_qb_bin_bounds_validated(bounds)`.

### 2.2 TransformerEngine FP8 Weight Workspace Caching Under `torch.compile` (`TransformerEngine@9ea05cf2e8dc`)

- In `transformer_engine/pytorch/module/linear.py`, `Linear` removes the eager fallback for `self.cache_weight and self.fp8` (only falling back for GPU-controlled `skip_fp8_weight_update`).
- When `use_compiled_op and is_first_microbatch` holds, `forward()` passes `weight_workspace = None` into the functional custom op so a fresh FP8 weight workspace is created and cached in `self._fp8_workspaces["weight"]` without in-place argument mutation; subsequent microbatches (`is_first_microbatch=False`) pass the cached `weight_workspace`. Across `M = 8` microbatches, redundant FP8 weight quantizations drop from `8` to `1` (**`87.50%` reduction**; peak memory across four 4096-wide layers is `452 MiB` eager / `539 MiB` compiled).

### 2.3 LMCache Chunk-Aligned Full-Prompt Hit Under SWA Bounded Prefix Replay (`LMCache@a0ff4701284d`)

- In `lmcache/integration/vllm/lmcache_mp_connector.py`, `get_prefix_replay_tokens(kv_cache_config)` inspects `group.kv_cache_spec.prefix_replay_tokens`.
- On a full-prompt hit (`ret == len(request.all_token_ids)`) when `self._prefix_replay_tokens > 0`, the connector aligns the hit down by a full chunk instead of subtracting 1 token:
  $$\text{aligned} = \max\left(0, \left\lfloor \frac{\text{ret} - 1}{\text{chunk}} \right\rfloor \times \text{chunk}\right)$$
  It appends `(aligned, ret)` to `tracker.pending_free_ranges` (releasing the lookup lock on the unretrieved tail chunk) and sets `tracker.num_lmcache_hit_tokens = aligned`. For a 12-token prompt (`chunk = 4`), `aligned = 8` matches vLLM's 2-block (`8`-token) allocation, reducing buffer overflow from **`4` tokens to `0` tokens**.

### 2.4 Mooncake Post-Send `notify_send_wr_id_` Increment, NVLink Diagnostics & NIXL `treq` Wire Tag (`Mooncake@dae22d472e31`, `Mooncake@b66bb2b5f650`, `nixl@f3673f6be3f4`, `nixl@70badd6aad36`)

- **Mooncake `sendNotification` Slot Safety**: In `mooncake-transfer-engine/tent/src/transport/rdma/endpoint.cpp`, `wr.wr_id = notify_send_wr_id_;` is assigned before `ibv_post_send()`, and `++notify_send_wr_id_;` executes *only after* `ibv_post_send()` succeeds. When `255` WRs (`0..254`) are in flight and post #256 fails, `notify_send_wr_id_` remains `255`, so the retry safely uses slot `255` rather than wrapping around to `256 % 256 == 0` and corrupting in-flight slot `0`.
- **Mooncake NVLink VMM & PID-Namespace Diagnostics**: In `intranode_nvlink_transport.cpp`, `registerLocalMemory` probes `cuMemRetainAllocationHandle` upon `cudaIpcGetMemHandle` failure to explicitly identify `cuMemCreate` VMM buffers (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), and `relocateSharedMemoryAddress` catches `cudaErrorDeviceUninitialized` to diagnose container PID collisions (`--ipc=host --pid=host`).
- **NIXL UCX 16-Byte `treq` Tag & Repost/Release Telemetry**: `src/plugins/ucx/ucx_backend.cpp` appends a 16-byte little-endian `treq` buffer (`requestId`, `execution`) after the 26-byte `tctx` record so `nixl::notif.received` links directly to the sender's transfer execution. `src/core/nixl_agent.cpp` stages `start_time`/`timer` locally in `postXferReq` until repost eligibility is confirmed, and records `NIXL_TELEMETRY_FINISH` when completion is first observed in `postXferReq` or `releaseXferReq`.

---

## 3. Deterministic Verification Matrix

Running `python experiments/megatron-padding-zero-lmcache-swa-replay-mooncake-wrid/run.py --check` validates:

| Dimension | Legacy Baseline | Phase 66 Fix | Quantitative Impact / Architectural Guarantee |
| :--- | :--- | :--- | :--- |
| **Megatron CUDA Graph Padding Zero (`72/128` rows)** | Frozen Python slice (`0` rows zeroed) | Device pointer `real_token_count[0]` zeroes `56` rows | **`32.0x`** FP8 quantization scale precision recovered |
| **TE `torch.compile` FP8 Weight Cache (`M=8`)** | Eager fallback or `8` casts/step | Fresh workspace on microbatch 0, reused on `1..7` | **`87.50%`** fewer FP8 weight casts per step |
| **LMCache SWA Full-Prompt Hit (`12` tok, `chunk=4`)** | Returns `11` -> vLLM allocs `8`, retrieves `12` (`+4` OOB) | Aligns hit to `8` tokens & frees lock `[8, 12)` | **`0` OOB tokens**, zero leaked lookup locks |
| **Mooncake RDMA Notify (`255` in-flight + failed post)** | `wr_id` becomes `256` -> retry overwrites slot `0` | Advances `notify_send_wr_id_` only after `ibv_post_send` | Retry lands on free slot `255` (**zero slot corruption**) |
| **NIXL UCX Notification & Repost Telemetry** | 26B `tctx` only; rejected repost resets timer | 26B `tctx` + 16B `treq`; staged timer + completion stats | Full cross-node request lineage & accurate telemetry |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Troubleshooting Exercise**: Two containers on the same 8-GPU node communicate via Mooncake `IntraNodeNvlinkTransport` with `--ipc=host` but default separate PID namespaces. Why does `cudaIpcOpenMemHandle` fail with `invalid device context` (`cudaErrorDeviceUninitialized`) exclusively between same-index workers (e.g., Worker 0 <-> Worker 0)?  
   - **Answer**: A CUDA IPC handle embeds the exporting process's PID. With isolated container PID namespaces, Worker 0 in Container A and Worker 0 in Container B are assigned the exact same numerical PID inside their respective namespaces; CUDA then rejects the peer's handle as a self-import (`cudaErrorDeviceUninitialized`). Sharing the PID namespace (`--pid=host` or `hostPID: true`) resolves the collision.
2. **Falsifiable Hypothesis**: When `prefix_replay_tokens == 0` (standard attention without SWA bounded replay), does `LMCacheMPConnector` still drop an entire 4-token chunk on a 12-token full-prompt hit?  
   - **Verification**: **No**. When `self._prefix_replay_tokens == 0`, it preserves the `need_to_load - 1 = 11` path so vLLM allocates 3 blocks (`12` slots) and retrieves all `12` tokens.
