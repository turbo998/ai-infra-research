# SGLang Diffusion SHM Reclaim, Cosmos3 Video VAE & DeepSeek-V4 Recipe, vLLM KV Offload & TRT-LLM Retry

Reproducible Python 3.12 standard-library model auditing seven first-hand commits across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`:

1. **SGLang Diffusion Per-RPC `zmq.IDENTITY` & Spilled SHM File Reclaim** (`sgl-project/sglang` @ `c05fb1bd04d2f9953dcef9a798d0984fb62d5253`, `#46154`): Assigns unique `sgldiffusion-request-<id>` DEALER identities per RPC and reclaims spilled shared-memory reply files (`>= 32 MiB`) across completion, error, and cancellation paths.
2. **SGLang Cosmos3 Multi-GPU Video VAE Decode (`channels_last_3d` + `use_vae_fast_path`) & Rust `deepseek-recipe 0.2.0`** (`sgl-project/sglang` @ `97220dd674370d5f30ba89d8c5ee2a7f36c68d40` `#46173`, `1406ffda099f53c819e548ffea242350a0e55281` `#46243`): Converts only `decoder.*` weights/activations to `channels_last_3d` when `num_gpus >= 2` (preserving encoder bit-exact I2V seeding), activates `use_vae_fast_path` when `quality_allows`, and renders DeepSeek-V4/V4.1 chat templates directly to token IDs (`Vec<u32>`) in Rust.
3. **vLLM KV Offload Atomic `rename(2)` `EEXIST` Idempotency, Preempted `next_stored_chunk_idx` Reset & Eager `batch_lookup` Materialization** (`vllm-project/vllm` @ `1757f6720c3c1cc130a834b41f264ac22a782289` `#61390`, `6f51433838b858a9cd0478b2ffee4d2c610a1712` `#63010`, `503e4e16ad8d4c4b7b1290c9f2792af0ca622c47` `#63126`): Treats `FileExistsError` on content-addressed FS KV blocks as success (`0`) and unlinks `tmp_path`, resets `group_state.next_stored_chunk_idx = 0` when a preempted request restarts from `num_computed_tokens == 0`, and materializes `list(self.batch_lookup(keys, req_context))` inside the `try` block.
4. **TensorRT-LLM `openai_client.py` `ClientOSError` (`ECONNRESET`) Retry Coverage** (`NVIDIA/TensorRT-LLM` @ `315f04f2250fc8485e3ed83cddce5ed380921062`, `#20889`): Extends the 5-attempt exponential backoff (`2^attempt` seconds) retry loop to `aiohttp.ClientOSError`.

## Usage

```bash
python experiments/sglang_diffusion_shm_reclaim_cosmos3_vae_dsv4_recipe_vllm_kv_offload_trtllm_retry/run.py --check
```
