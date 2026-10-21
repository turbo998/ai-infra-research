# SGLang Zero-Sync Mamba Planning / KDA Prefill Graph / ReusePort, TensorRT-LLM DeepSeek-V4 Indexer Prewarm & vLLM Cosmos3-Edge / Rust Parity

> **Primary Upstream Commits**:
> - [`sgl-project/sglang@740d5f9`](https://github.com/sgl-project/sglang/commit/740d5f9f9e33aeebc6b5bcb1ba73f25bf6db062b) (`#45472`, 2026-10-21) — *perf: eliminate host-device syncs in hybrid Mamba extend batch planning*
> - [`sgl-project/sglang@f06d1bb`](https://github.com/sgl-project/sglang/commit/f06d1bb1f45df617aa455c103ce5ede2bc6c6a47) (`#45473`, 2026-10-21) — *feat: support FlashInfer KDA prefill CUDA graph*
> - [`sgl-project/sglang@a3d00ef`](https://github.com/sgl-project/sglang/commit/a3d00eff58b6c22ea7539f882cc10c537634200b) (`#45487`, 2026-10-21) — *perf: compute `contains_last_prefill_chunk` on host and preserve in TBO `filter_batch`*
> - [`sgl-project/sglang@7e5256d`](https://github.com/sgl-project/sglang/commit/7e5256d1ba3aaa628bb8b756fafb904a76a1123a) (`#45482`, 2026-10-21) — *perf: add `--enable-http-worker-reuseport` for multi-worker HTTP server*
> - [`NVIDIA/TensorRT-LLM@370fe4c`](https://github.com/NVIDIA/TensorRT-LLM/commit/370fe4c5096cff7ffb457092d36a208da34006d7) (`#20734`, 2026-10-21) — *fix(deepseek_v4): fix `_prewarm_cute_dsl_indexer_q` dead import and cover all fallback tactics*
> - [`vllm-project/vllm@4a3ee9a`](https://github.com/vllm-project/vllm/commit/4a3ee9a0f6a2958f2712ceb3c9f60fe48cd764b3) (`#57393`, 2026-10-21) — *[Model] Support Cosmos3-Edge ModelOpt quantized checkpoints*
> - [`vllm-project/vllm@2916b4b`](https://github.com/vllm-project/vllm/commit/2916b4b4a6b7e9ae552e661b365fec65117134ac) (`#62822`), [`bc4794e`](https://github.com/vllm-project/vllm/commit/bc4794ef9a9fea5f8a003ef4dc5a49bbe8145241) (`#62825`), [`8a21d0a`](https://github.com/vllm-project/vllm/commit/8a21d0a328bb3cbb9c73ca0ae65d83465768cca4) (`#62416`), [`e0ca7d9`](https://github.com/vllm-project/vllm/commit/e0ca7d9cc11da2011389c725e5d31c54c3b47259) (`#62824`, 2026-10-21) — *[Rust Frontend] Render `return_token_ids`, env vars, CLI flags, and `PromptFillsContext` parity*

---

## 1. SGLang Zero-Sync Mamba Planning, KDA Prefill Graph & ReusePort

### 1.1 Eliminating Implicit Host-Device Syncs in Hybrid Mamba Extend (`#45472`)
- **Host-Tracked `mamba_ping_pong_empty: list[bool]`**: Replaces `(ping_pong_mapping[req_indices] == -1).any().item()` with a CPU boolean mirror, gathers track slots on device via `ping_pong_mapping[req_pool_indices, track_positions]`, and clears slots via `index_fill_(1, idx64, 0)`.
- **Replacing `torch.isin` with Sorted `searchsorted` in `_get_multimodal_mask`**: PyTorch's `torch.isin` switches to `unique()` + sort when $P \ge 10 \cdot N^{0.145}$ (e.g., $P \ge 34$ at $N=4096$), forcing a device-to-host sync. `#45472` sorts placeholders once and uses `torch.searchsorted` + `clamp_` + equality with zero host syncs.

### 1.2 KDA Prefill CUDA Graph, `contains_last_prefill_chunk` & ReusePort (`#45473` / `#45487` / `#45482`)
- **`FlashInferKDAPrefillGraph` (`#45473`)**: Wraps `RecurrentKDAPrefillWrapper` (SM100/SM103) inside `KDAPrefillGraphMetadata` for piecewise CUDA Graph capture and replay.
- **Host-Side `contains_last_prefill_chunk` (`#45487`)**: Computes `req.extend_end == len(req.full_untruncated_fill_ids)` on CPU (excluding `decoding_reqs`) and recomputes it across TBO `filter_batch` splits.
- **`_ReusePortMultiprocess` (`#45482`)**: Adds `--enable-http-worker-reuseport` (`uvicorn>=0.51`), giving each worker a dedicated `SO_REUSEPORT` listener while reserving the parent socket so a second process cannot silently join the port group.

---

## 2. TensorRT-LLM DeepSeek-V4 CuTe DSL Indexer Q Prewarm Fix (`#20734`)

`TensorRT-LLM@370fe4c` resolves three defects in `_prewarm_cute_dsl_indexer_q`:
1. Fixes the broken import to `from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4Indexer` (previously swallowed by `try/except`);
2. Filters for local PP-stage layers where `wq_b.weight is not None` and `_is_fused_project_mxfp4_enabled(torch.bfloat16)` is true;
3. Prewarms token counts `(4, 8, 16)` to hit all three `_fallback_tactic()` branches (`m <= 4`, `4 < m <= 8`, `m > 8`).

---

## 3. vLLM Cosmos3-Edge ModelOpt Remap & Rust Frontend Parity

- **Cosmos3-Edge ModelOpt Quantizer Filtering (`#57393`)**: Drops `.input_quantizer.`, `.weight_quantizer.`, and `.output_quantizer.` state-dict keys and remaps `.weight_scale_inv -> .weight_scale` and `.act_quant_scale -> .input_scale`.
- **Rust Frontend Parity (`#62822` / `#62825` / `#62416` / `#62824`)**: Distinguishes `Ordering::Equal` (`PromptFillsContext`, `param="input_tokens"`) from `Ordering::Greater` (`PromptTooLong`), adds `return_token_ids` to `/v1/chat/completions/render`, and wires `VLLM_LOG_STATS_INTERVAL`, `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`, and `--tokenizer` / `--tokenizer-revision` / `--hf-config-path`.

---

## 4. Deterministic Reproduction

```bash
python experiments/sglang-mamba-zero-sync-kda-graph-reuseport-trtllm-indexer-prewarm-vllm-rust/run.py --check
```
