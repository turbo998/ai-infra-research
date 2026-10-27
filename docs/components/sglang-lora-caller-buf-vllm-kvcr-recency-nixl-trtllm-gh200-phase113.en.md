# SGLang Diffusion LoRA Scaling & Breakable CUDA Graph Caller Buffers, vLLM KVCR Prefix Recency & Rust Grammar/NIXL Stats, and TensorRT-LLM GH200 / DeepSeek-V3.2 Analysis (Phase 113 · Priority 122)

> **中文版**: [sglang-lora-caller-buf-vllm-kvcr-recency-nixl-trtllm-gh200-phase113.md](sglang-lora-caller-buf-vllm-kvcr-recency-nixl-trtllm-gh200-phase113.md)  
> **Question ID**: `sglang-lora-caller-buf-vllm-kvcr-recency-nixl-trtllm-gh200-phase113` · **Priority**: `122` · **Track**: Track B (Inference Serving & AI Chips)  
> **Deterministic Experiment**: [`experiments/sglang_lora_caller_buf_vllm_kvcr_recency_nixl_trtllm_phase113/run.py`](../../experiments/sglang_lora_caller_buf_vllm_kvcr_recency_nixl_trtllm_phase113/run.py) (`--check` passed)

---

## 1. First-Hand Source Matrix

| # | Source ID | Repository & Commit | Date | Evidence Tier | Key Code Path & Architectural Change |
|---|---|---|---|---|---|
| 1 | `sglang-diffusion-lora-format-alpha-rank-scaling-2026` | [`sgl-project/sglang@dc41126cde`](https://github.com/sgl-project/sglang/commit/dc41126cdea3fd7f461b5fa41bf5d15fb9ff99a2) | `2026-10-27` | `implementation` | Propagates split-module `alpha` via `move_alpha` in AI-Toolkit Flux LoRA conversion and strips `.weight` + `.lora_A` cleanly in `apply_peft_config` |
| 2 | `sglang-diffusion-caller-allocated-bcg-break-buffers-2026` | [`sgl-project/sglang@a912002ee2`](https://github.com/sgl-project/sglang/commit/a912002ee288f8ee6c65656ca514dd1a4f78898e) | `2026-10-27` | `implementation` | Writes MiniMax-H3 and Qwen-Image 2.1 `@eager_on_graph` break outputs into caller-allocated buffers with strict `shape`/`dtype` assertions |
| 3 | `sglang-multimodal-clip-pad-spans-truncated-prompt-2026` | [`sgl-project/sglang@6724403444`](https://github.com/sgl-project/sglang/commit/6724403444fb6dd54e4198a3813d499bf8a6e23c) | `2026-10-27` | `implementation` | Clips multimodal token padding spans via `end = min(end + 1, len(padded_input_ids))` when auto-truncation cuts inside or before a span |
| 4 | `sglang-amd-monokernels-register-specs-by-model-type-2026` | [`sgl-project/sglang@5044779f0f`](https://github.com/sgl-project/sglang/commit/5044779f0f4efdb361c3923eb22e12cb84e23966) | `2026-10-27` | `implementation` | Keys the AMD mono-kernel registry by `(architecture, model_type)` |
| 5 | `vllm-kv-offload-align-kvcr-recency-request-prefixes-2026` | [`vllm-project/vllm@e3883b7a7e`](https://github.com/vllm-project/vllm/commit/e3883b7a7e676cc37b9d4ab2d94e61b48e878) | `2026-10-27` | `implementation` | Aligns KVCR tiered cache recency with request prefix order via `order_request_keys` and `align_sequence` so leaf blocks evict before root prefix blocks |
| 6 | `vllm-cpu-attn-encoder-only-bidirectional-fix-2026` | [`vllm-project/vllm@a23d7dc3f1`](https://github.com/vllm-project/vllm/commit/a23d7dc3f15b91e06fc888a4bb70ca57a7e1586c) | `2026-10-27` | `implementation` | Runs `AttentionType.ENCODER_ONLY` bidirectionally in `CPU_ATTN` |
| 7 | `vllm-rust-frontend-decode-current-nixl-connector-stats-2026` | [`vllm-project/vllm@c56cdbea15`](https://github.com/vllm-project/vllm/commit/c56cdbea151dc4289f55b93a8496c15f11085fd9) | `2026-10-27` | `implementation` | Removes `#[serde(deny_unknown_fields)]` on `NixlStats` and decodes `num_failed_handshakes` and `num_notifications_after_expiry` in the Rust frontend |
| 8 | `vllm-rust-frontend-detect-lark-grammars-like-python-2026` | [`vllm-project/vllm@405276b8f0`](https://github.com/vllm-project/vllm/commit/405276b8f07fc2e258d7f5551fd8680e79b21899) | `2026-10-27` | `implementation` | Ports `grammar_is_likely_lark` to the Rust frontend, ignoring `::=` inside `#` or `//` comments |
| 9 | `trtllm-restore-default-model-storage-gh200-2026` | [`NVIDIA/TensorRT-LLM@e805aac356`](https://github.com/NVIDIA/TensorRT-LLM/commit/e805aac356643a21dc7644f08c475f50dfcf7e79) | `2026-10-27` | `implementation` | Restores default model storage volume mounts on GH200 pods |
| 10 | `trtllm-deepseek-v32-encoder-restore-tools-preamble-blank-lines-2026` | [`NVIDIA/TensorRT-LLM@c932dd8abb`](https://github.com/NVIDIA/TensorRT-LLM/commit/c932dd8abb3f15b91a90ab9185caef3f8d72fd1c) | `2026-10-27` | `implementation` | Restores 4 structural blank lines in DeepSeek-V3.2 `TOOLS_SYSTEM_TEMPLATE` |

---

## 2. Deterministic Verification Summary (`own-offline`)

| Metric | Baseline / Before | Phase 113 Verified | Delta / Takeaway |
|---|---|---|---|
| SGLang Flux LoRA delta (`rank=16, alpha=8, x=4`) | `4.0` (dropped `alpha` -> `1.0x`) | **`2.0` (`alpha/rank = 0.5x`)** | `move_alpha` + `.lora_A` suffix matching |
| SGLang truncated prompt multimodal pad (`len=12`, span `[8, 15]`) | Out-of-bounds write to `16` | **Clipped to `[8, 12]` (`4` tokens)** | `end = min(end + 1, len)` |
| SGLang breakable CUDA graph dynamic output allocs per break | `2` | **`0`** | Caller-allocated buffer + shape/dtype assert |
| vLLM KVCR contiguous prefix hits after 1-block eviction (`4`-block prefix) | `0` blocks (`B0` evicted first) | **`3` blocks (`B3` evicted first)** | `align_sequence` preserves root prefix |
| vLLM Rust Lark detection with `# ::= ` comment | `False` (misclassified as EBNF) | **`True` (`grammar_is_likely_lark`)** | Comment-aware rule scanning |
| TRT-LLM DeepSeek-V3.2 tools preamble blank lines & GH200 storage | `0` blank lines / scratch override | **`4` blank lines + default GH200 storage** | Exact DSML prompt encoding |
