# SGLang DSV4 HiSparse Token-Granular Eager Backup, DP-Attention Prefill Bucket & Decode Custom-AR Alignment, Diffusion IPC `/dev/shm` Cleanup, vLLM Speculator Dynamic `max_model_len` & Streaming KV Offload Reset, and TensorRT-LLM `ClientOSError` Retry

> **Document Status**: Phase 77 (`priority: 86`) First-Hand Source Audit + CPU/Offline Reproducible Verification  
> **Target Audience**: Solution Architects operating production LLM/multimodal serving stacks (SGLang, vLLM, TensorRT-LLM) with hierarchical KV caching, DP-Attention CUDA Graphs, and speculative decoding  
> **中文伴侣文档**: [sglang-hisparse-dsv4-dp-attn-bucket-diffusion-ipc-vllm-spec-autofit-kv-reset-trtllm.md](sglang-hisparse-dsv4-dp-attn-bucket-diffusion-ipc-vllm-spec-autofit-kv-reset-trtllm.md)

---

## 1. Production Failure Modes & Architectural Context

Recent first-hand commits across SGLang, vLLM, and TensorRT-LLM resolve six high-severity correctness, deadlock, and resource-leak bugs in production serving:

1. **Silent KV Cache Corruption in SGLang DSV4 HiSparse When Decode Batch Size Divides `slot_page_size` (`#45104`)**: DSV4 C4 pages store `[value0..value63][scale0..scale63]` (`slot_page_size = 64`). During decode, `HiSparseCoordinator._eager_backup_previous_token` backs up one newly generated token per request at scattered page slots. When the active request count was an exact multiple of 64 (e.g., 64 requests scattered across pages 10 and 25), `backup_from_device_all_layer` saw `numel() % slot_page_size == 0` and executed a whole-page copy from the first slot's page instead of token-granular `transfer_cache_dsv4_mla`, corrupting 50% of backed-up tokens.
2. **DP-Attention Prefill CUDA Graph Bucket Mismatch & Decode Custom All-Reduce Divergence (`#46138`)**: In `prepare_mlp_sync_batch`, prefill CUDA Graph under `DpPaddingMode.MAX_LEN` padded ranks to raw `max(global_num_tokens)` (e.g., `17`) rather than rounding up to the captured graph bucket (`24`); meanwhile, `get_default_mode_in_cuda_graph` ignored `_decode_sum_len_fits_custom_all_reduce`, and idle DP ranks (`0` real tokens) ran unnecessary `init_forward_metadata`.
3. **Leaked `/dev/shm` Array Files on Aborted/Timed-Out SGLang Diffusion IPC Requests (`#46154`)**: Local `ipc://` requests spilling arrays above `_MIN_FILE_REF_BYTES` (`32 MiB`) to `/dev/shm` leaked temporary files whenever a reply timed out, was cancelled, or failed deserialization.
4. **Stale `Speculator.max_model_len` After vLLM Auto-Fit KV Cache Downsizing (`#60513`)**: `Speculator` copied `max_model_len` at `__init__` time, remaining stuck at the initial value (`32768`) after auto-fit reduced `model_config.max_model_len` (`8192`).
5. **Duplicate Block Offloading in vLLM Resumed Streaming Sessions (`#60810`) & Out-of-Order DBO Microbatch Error Reporting (`#63476`)**: Resumed `resumable=True` streaming chunks arrived in `scheduled_new_reqs` with their full accumulated block table (`[0, 1, 2, 3, 4]`), but `yield_req_data` set `is_full_table=False`, appending onto `[0, 1]` and re-offloading stale blocks `[0, 1, 2]` instead of `[2, 3, 4]`; in DBO microbatch execution, `min(errors)` reported microbatch 0's secondary recv-hook error instead of microbatch 1's chronological root cause.
6. **Unhandled `aiohttp.ClientOSError(ECONNRESET)` in TensorRT-LLM Benchmark Client (`#20889`)**: Direct `ClientOSError(errno.ECONNRESET)` exceptions without a wrapped `__cause__` bypassed `_TRANSIENT_TCP_BUDGET` (`5` retries).

---

## 2. First-Hand Source Audit & Fixes

### 2.1 `sglang@c00e7e12b64b` (`#45104`): `backup_tokens_from_device_all_layer`

Adds `HiSparseNSATokenToKVPoolHost.backup_tokens_from_device_all_layer` and routes `HiSparseCoordinator._eager_backup_previous_token` through it so scattered single-token decode backups unconditionally invoke `transfer_cache_dsv4_mla` regardless of whether `numel() % slot_page_size == 0`.

### 2.2 `sglang@9b230c1dae7e` (`#46138`): DP-Attention Prefill Bucket Rounding, Decode `SUM_LEN` & Idle Rank Skip

1. Rounds `global_num_tokens` up to `min((size for size in prefill_cg.bs if size >= max_tokens), default=max_tokens)` across all `sync_group_size` ranks when prefill CUDA Graph uses `DpPaddingMode.MAX_LEN`.
2. Aligns `DpPaddingMode.get_default_mode_in_cuda_graph(global_num_tokens_cpu)` with `_decode_sum_len_fits_custom_all_reduce` so decode CUDA Graphs use `SUM_LEN` whenever total token bytes fit custom all-reduce.
3. Skips `attn_backend.init_forward_metadata` on idle DP ranks (`global_num_token_non_padded_cpu == 0`) when `skips_prefill_metadata_on_idle_dp_rank=True`.

### 2.3 `sglang@c05fb1bd04d2` (`#46154`): Per-Request `/dev/shm` Cleanup via `array_ipc_request`

Wraps local `ipc://` RPCs in `with array_ipc_request(endpoint) as identity:`, creating a per-request `/dev/shm/sgldiffusion-request-*` directory and removing it on exit across all 7 outcomes (`success`, `bad_pickle`, `bad_array`, `timeout_before`, `timeout_after`, `cancel_before`, `cancel_after`).

### 2.4 `vllm@636da2476d18` (`#60513`), `921ed7fa11c4` (`#60810`), `66bf93154420` (`#63476`) & `TensorRT-LLM@315f04f2250f` (`#20889`)

- **Dynamic `Speculator.max_model_len` (`vllm#60513`)**: Converts `max_model_len` to a live `@property` reading `self.vllm_config.model_config.max_model_len` and clamps `_build_attn_metadata` with `min(self.draft_max_seq_len, self.max_model_len)`.
- **Streaming KV Offload Reset (`vllm#60810`)**: Emits `is_full_table=True` for `scheduled_new_reqs` in `yield_req_data` and clears `block_ids` before extending so resumed streaming chunk 2 offloads `[2, 3, 4]` instead of `[0, 1, 2]`.
- **Chronological DBO Microbatch Exception (`vllm#63476`)**: Uses `failed = next(iter(errors))` instead of `min(errors)` in `ubatch_utils.py`.
- **TensorRT-LLM `ClientOSError(ECONNRESET)` Retry (`#20889`)**: Treats `isinstance(e, aiohttp.ClientOSError) and e.errno == errno.ECONNRESET` as `is_transient_tcp`, granting `effective_max = max(max_retries, 5)` while keeping `errno.EPIPE` at `max_retries`.

---

## 3. Offline Verification Matrix

Validated by `experiments/sglang_hisparse_dsv4_dp_attn_bucket_diffusion_ipc_vllm_spec_autofit_kv_reset_trtllm/run.py --check`:

| Scenario | Legacy Behavior | Phase 77 Fixed Behavior | Verified Invariant |
|---|---|---|---|
| DSV4 HiSparse 64-req Decode Backup (`#45104`) | Whole-page copy corrupts **32 / 64 (`50%`)** tokens | `backup_tokens_from_device_all_layer` copies per token (**0 corrupted**) | `corrupted_tokens_fixed == 0` |
| DP-Attention Prefill Bucket & Decode AR (`#46138`) | Unbucketed `17`; decode defaults to `MAX_LEN` | Rounded to bucket **`24`**; decode selects **`SUM_LEN`**; idle rank skips init | `fixed_target_bucket == 24` |
| Diffusion IPC `/dev/shm` Cleanup (`#46154`) | Leaks **`256 MiB`** across 4 fault/timeout branches | `array_ipc_request` reclaims directory (**`0 MiB` leaked**) | `fixed_total_leaked_mib == 0` |
| vLLM Speculator Auto-Fit & Streaming KV (`#60513`, `#60810`) | Stale `32768`; chunk 2 stores `[0, 1, 2]` | Live **`8192`**; chunk 2 stores **`[2, 3, 4]`** | `fixed_chunk2_stored_blocks == [2, 3, 4]` |
| DBO Error Order & TRT-LLM Retry (`#63476`, `#20889`) | Reports `ubatch 0`; `ECONNRESET` gets 0 retries | Reports root-cause `ubatch 1`; `ECONNRESET` gets **5 retries** | `fixed_effective_max_retries == 5` |

---

## 4. Architect Exercises & Falsifiable Follow-Ups

1. **Reliability Sizing Exercise**: On a node with an `8 GiB` `/dev/shm` tmpfs serving `64 MiB` diffusion IPC payloads at a `0.5%` client timeout rate, calculate how many total requests trigger `ENOSPC` before `#46154`.
2. **Falsifiable Hypotheses**:
   - **H1**: Splitting prefill HiSparse backups into whole-page DMAs for full 64-token pages plus `transfer_cache_dsv4_mla` for tail tokens outperforms pure token-granular backup whenever prefill chunk length exceeds 256 tokens.
   - **H2**: When DP-Attention rank token counts exhibit extreme skew (`[0, 0, 0, 512]`), rounding all 4 ranks up to prefill CUDA Graph bucket `512` consumes more total GPU cycles than falling back to unpadded eager execution.
