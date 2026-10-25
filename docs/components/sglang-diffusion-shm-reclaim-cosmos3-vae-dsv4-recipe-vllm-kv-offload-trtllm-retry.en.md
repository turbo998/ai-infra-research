# Source Audit of SGLang Diffusion SHM Reclaim, Cosmos3 Multi-GPU Video VAE, DeepSeek-V4 Rust Recipe, vLLM KV Offload & TRT-LLM Retry

> **中文版**: [sglang-diffusion-shm-reclaim-cosmos3-vae-dsv4-recipe-vllm-kv-offload-trtllm-retry.md](sglang-diffusion-shm-reclaim-cosmos3-vae-dsv4-recipe-vllm-kv-offload-trtllm-retry.md) | **Question ID**: `q-sglang-diffusion-shm-reclaim-cosmos3-vae-dsv4-recipe-vllm-kv-offload-trtllm-retry` (`priority: 91`, Phase 82) | **Reproducible Experiment**: `experiments/sglang_diffusion_shm_reclaim_cosmos3_vae_dsv4_recipe_vllm_kv_offload_trtllm_retry/run.py`

---

## 1. Architectural Context & Core Engineering Failure Modes

In production multimodal generation (Diffusion / Video VAE) and distributed LLM serving stacks (SGLang, vLLM, TensorRT-LLM), four subtle resource-leak and state-desynchronization failure modes arise across the control and storage-offload planes:

1. **Permanent `/dev/shm` Leaks When Large (`>= 32 MiB`) IPC Replies Are Cancelled or Timed Out**: SGLang Diffusion's `array_ipc_request` spills reply arrays `>= 32 MiB` into shared-memory files. Sharing a static ZMQ DEALER identity across concurrent RPCs or failing to reclaim spilled SHM segments on cancellation/error leaks tens of megabytes per aborted request.
2. **Multi-GPU Video VAE Layout Optimization vs. Bit-Exact Image-to-Video (I2V) Encoder Seeding**: Converting Cosmos3 3D causal convolutions to `channels_last_3d` accelerates multi-GPU (`num_gpus >= 2`) video decoding, but applying `channels_last_3d` to `vae.encoder` alters cuDNN convolution rounding and Perturbs I2V latent conditioning seeds.
3. **vLLM KV Offload Corruption Across Preemption Reset, Atomic `rename(2)` `EEXIST`, and Lazy `batch_lookup` Generators**: When a preempted request restarts from `num_computed_tokens == 0`, leaving `group_state.next_stored_chunk_idx` at its pre-preemption value causes the coordinator to skip re-offloading evicted prefix chunks.
4. **Uncaught `aiohttp.ClientOSError` (`ECONNRESET`) in TensorRT-LLM Benchmark Clients**: Under high concurrency, transient TCP connection resets surface as `aiohttp.ClientOSError`, bypassing retry loops that only catch `ClientPayloadError` or `ServerDisconnectedError`.

---

## 2. First-Hand Source Code Implementation Audit

### 2.1 SGLang Diffusion Per-RPC `zmq.IDENTITY` & Spilled SHM Reclaim (`c05fb1bd04d2`, `#46154`)

Auditing `python/sglang/multimodal_gen/runtime/utils/array_ipc.py` in `sgl-project/sglang`:

- **Per-RPC Unique DEALER Identity**: `array_ipc_request()` generates `req_identity = f"sgldiffusion-request-{uuid.uuid4().hex}".encode("ascii")` and binds it via `sock.setsockopt(zmq.IDENTITY, req_identity)` before connecting, isolating reply routing per request.
- **Lifecycle-Wide Spilled SHM Reclamation (`_reclaim_spilled_shm_from_frames`)**: Whenever a reply spills to shared memory (`>= 32 MiB`), completion, exception, and cancellation paths inspect the reply frames and unlink the spilled SHM file, eliminating `/dev/shm` exhaustion.

### 2.2 SGLang Cosmos3 Multi-GPU Video VAE Decode (`97220dd67437`, `#46173`) & DeepSeek-V4 Rust Recipe (`1406ffda099f`, `#46243`)

- **Decoder-Only `channels_last_3d` + `use_vae_fast_path` in Cosmos3 (`#46173`)**:
  - Auditing `cosmos3_vae.py` and `cosmos3.py`: converts only `vae.decoder` 3D convolution weights and activations to `torch.channels_last_3d` while keeping `vae.encoder` contiguous so I2V encoder latent rounding remains bit-identical.
  - Enables `channels_last_3d` and `use_vae_fast_path` automatically when `server_args.num_gpus >= 2` and `quality_allows`.
- **Native Rust `deepseek-recipe = 0.2.0` Prompt Formatting (`#46243`)**:
  - Auditing `rust/sglang-processor/Cargo.toml`, `src/processor.rs`, and `sgl-model-gateway`: integrates `deepseek-recipe = "0.2.0"` (`ChatTemplateContentFormat::DeepSeekV4` and `DeepSeekV4_1`), calling `ChatFormatter::encode_request()` in Rust to render messages and tool calls directly into `Vec<u32>` token IDs without Python Jinja2 overhead.

### 2.3 vLLM KV Offload Consistency Fixes (`1757f6720c3c` `#61390`, `6f51433838b8` `#63010`, `503e4e16ad8d` `#63126`) & TRT-LLM Retry (`315f04f2250f` `#20889`)

- **Idempotent Content-Addressed Atomic `rename(2)` on `EEXIST` (`vllm` `#61390`)**: When concurrent workers write the same content-addressed KV block to disk, `os.replace(tmp_path, final_path)` / `rename(2)` can raise `FileExistsError` (`EEXIST`). The filesystem backend now treats `EEXIST` as stored (`0`) and unlinks `tmp_path`.
- **Resetting `group_state.next_stored_chunk_idx = 0` After Preemption (`vllm` `#63010`)**: In `vllm/v1/core/kv_cache_coordinator.py`, when a preempted request restarts computation from `num_computed_tokens == 0`, `group_state.next_stored_chunk_idx` is reset to `0` so all recomputed KV chunks are re-stored.
- **Eager `list(self.batch_lookup(...))` Materialization Inside `try` (`vllm` `#63126`)**: Forces generator evaluation inside the `try...except` block so lazy `OSError` failures during KV lookup are caught rather than escaping to the caller.
- **TensorRT-LLM `openai_client.py` `ClientOSError` Retry (`TensorRT-LLM` `#20889`)**: Adds `aiohttp.ClientOSError` to the 5-attempt exponential-backoff (`2**attempt` seconds: `1, 2, 4, 8s`, `15s` total) retry handler in `benchmarks/cpp/utils/openai_client.py`.

---

## 3. Reproducible Verification Summary (`own-offline`)

Executing `python experiments/sglang_diffusion_shm_reclaim_cosmos3_vae_dsv4_recipe_vllm_kv_offload_trtllm_retry/run.py --check` verifies:

| Mechanism / Failure Scenario | Pre-Fix Behavior | Post-Fix Contract | Quantitative Verification |
|---|---|---|---|
| SGLang Diffusion `64` cancelled `48 MiB` RPC replies (`#46154`) | Leaks `3,072 MiB` in `/dev/shm` | Per-RPC `zmq.IDENTITY` + frame SHM unlink | **`0 MiB` leaked** (`-100%`) |
| SGLang Cosmos3 Multi-GPU (`num_gpus=4`) Video VAE (`#46173`) | Full-VAE layout conversion perturbs I2V seed | Decoder-only `channels_last_3d` + `use_vae_fast_path` | **`0 ULP` encoder seed drift** |
| vLLM preempted request (`4` of `6` chunks stored before eviction, `#63010`) | `next_stored_chunk_idx=4` skips first `4` chunks | Reset cursor to `0` when `num_computed_tokens == 0` | **`6 / 6` chunks stored** |
| vLLM `16` concurrent duplicate KV block writes (`#61390`) + lazy lookup (`#63126`) | `16` false errors + uncaught lazy `OSError` | `EEXIST -> 0` + unlink `tmp_path` + `list(...)` | **`0` false errors, `0` leaked tmp files** |
| TensorRT-LLM `openai_client.py` transient `ECONNRESET` (`#20889`) | Aborts benchmark on `ClientOSError` | 5-attempt exponential backoff (`1, 2, 4, 8s`) | **`15s` cumulative retry window** |

---

## 4. Architect Design Exercises & Falsifiable Questions

1. **Kubernetes `/dev/shm` Sizing Calculation**: If a video generation worker emits `48 MiB` IPC payloads and runs inside a pod with a `2 GiB` `/dev/shm` volume, calculate how many client-cancelled requests are required to exhaust `/dev/shm` without commit `c05fb1bd04d2` ($\lceil 2048 / 48 \rceil = 43$ cancelled RPCs).
2. **Falsifiable Hypothesis**: Because KV cache offload files are content-addressed by cryptographic/block hash, an `EEXIST` result on atomic `rename(2)` guarantees the destination file already contains identical KV bytes, making `unlink(tmp_path)` + return `0` strictly idempotent under concurrent writers.
