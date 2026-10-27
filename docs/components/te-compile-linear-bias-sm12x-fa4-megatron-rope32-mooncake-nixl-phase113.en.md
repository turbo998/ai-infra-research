# TransformerEngine `torch.compile` Linear+Bias Fusion & SM120 Deterministic FA4 Skip, Megatron-LM Vision RoPE FP32 & NCCL Determinism, TorchTitan ActorSystem & Mooncake / LMCache / Dynamo / NIXL Analysis (Phase 113 · Priority 122)

> **中文版**: [te-compile-linear-bias-sm12x-fa4-megatron-rope32-mooncake-nixl-phase113.md](te-compile-linear-bias-sm12x-fa4-megatron-rope32-mooncake-nixl-phase113.md)  
> **Question ID**: `te-compile-linear-bias-sm12x-fa4-megatron-rope32-mooncake-nixl-phase113` · **Priority**: `122` · **Track**: Track C (Distributed Training & KV Transfer)  
> **Deterministic Experiment**: [`experiments/te_compile_linear_bias_sm12x_fa4_megatron_rope32_mooncake_phase113/run.py`](../../experiments/te_compile_linear_bias_sm12x_fa4_megatron_rope32_mooncake_phase113/run.py) (`--check` passed)

---

## 1. First-Hand Source Matrix

| # | Source ID | Repository & Commit | Date | Evidence Tier | Key Code Path & Architectural Change |
|---|---|---|---|---|---|
| 1 | `te-pytorch-compile-linear-bias-forward-fusion-2026` | [`NVIDIA/TransformerEngine@45223d764a`](https://github.com/NVIDIA/TransformerEngine/commit/45223d764a02fc4557dc5d1e5715c1028836773f) | `2026-10-08` | `implementation` | Compiles `BasicLinear` + `Bias` with forward fusion in TE, specializing custom ops by TP size and preserving quantized tensors for dequantized backward |
| 2 | `te-skip-fa4-deterministic-backward-sm12x-2026` | [`NVIDIA/TransformerEngine@6d6a500860`](https://github.com/NVIDIA/TransformerEngine/commit/6d6a50086016d6bd648c4ef33b3e571ab0b5b970) | `2026-10-06` | `implementation` | Passes `device` and `requires_backward` into TE attention backend selection, skipping FA4 on SM12x (`sm_120`) when deterministic backward is required |
| 3 | `megatron-vision-rope-frequencies-fp32-bf16-conversion-2026` | [`NVIDIA/Megatron-LM@3ca52d95d6`](https://github.com/NVIDIA/Megatron-LM/commit/3ca52d95d694c5a373db7c1a5fd0936947cd59a0) | `2026-10-27` | `implementation` | Stores `self.freqs` outside `register_buffer` in `Pixtral2DRotaryEmbedding` so `Float16Module` cannot cast vision RoPE frequencies to BF16/FP16 |
| 4 | `megatron-determinism-accept-empty-nccl-algo-ring-nvls-2026` | [`NVIDIA/Megatron-LM@62c063f2b7`](https://github.com/NVIDIA/Megatron-LM/commit/62c063f2b7a5eb4ca01279ef0f249cc54e109032) | `2026-10-18` | `implementation` | Accepts empty `NCCL_ALGO` and `Ring,NVLS` under `--deterministic-mode` on non-Hopper GPUs and verifies `assert_nccl_algo_still_empty` |
| 5 | `torchtitan-rl-select-actor-framework-by-config-monarch-2026` | [`pytorch/torchtitan@f1bb643a71`](https://github.com/pytorch/torchtitan/commit/f1bb643a714492d92cf45ef944a969bb04011906) | `2026-10-18` | `implementation` | Introduces a config-selected `ActorSystem` interface (implemented for Monarch) across the TorchTitan RL controller, rollouter, batcher, and trainer |
| 6 | `mooncake-store-reuse-validated-shard-paths-cached-dfs-2026` | [`kvcache-ai/Mooncake@d782a5998f`](https://github.com/kvcache-ai/Mooncake/commit/d782a5998fd1bf06a7c823ef454033df10a5dd06) | `2026-10-19` | `implementation` | Bypasses path normalization on cached DFS shard reads when the path matches the exact open-time validated string (`64 -> 1` normalizations) |
| 7 | `mooncake-store-add-session-ids-kv-events-dynamo-indexer-2026` | [`kvcache-ai/Mooncake@0e626c572b`](https://github.com/kvcache-ai/Mooncake/commit/0e626c572b85f47664546b0862e11cc6f0b95f56) | `2026-10-19` | `implementation` | Adds per-key `session_ids` (`<= 256 B` UTF-8) to `ReplicateConfig` and KV events for Dynamo's session prefix indexer |
| 8 | `lmcache-include-request-tags-mooncake-lookup-keys-2026` | [`LMCache/LMCache@e45d4d5a85`](https://github.com/LMCache/LMCache/commit/e45d4d5a85b832a853c51f60f2b0485cb4846d3a) | `2026-10-22` | `implementation` | Forwards `request_configs` to `token_database.process_tokens` in `MooncakeLookupClient.lookup` so request tags isolate lookup keys |
| 9 | `dynamo-frontend-skip-per-token-text-decode-no-logprobs-2026` | [`ai-dynamo/dynamo@cffc4fa75c`](https://github.com/ai-dynamo/dynamo/commit/cffc4fa75cc919a342be97fd90ee275f78e59326) | `2026-10-27` | `implementation` | Skips per-token UTF-8 text decode in Dynamo's Rust frontend when `data.log_probs.is_none()` (**`640 -> 128` decode calls `5.0x` reduction for 512 tokens in 4-token chunks**) |
| 10 | `nixl-mooncake-fix-request-array-leak-postxfer-error-2026` | [`ai-dynamo/nixl@7bd1d1386e`](https://github.com/ai-dynamo/nixl/commit/7bd1d1386eb07e1f7efdd262135a8820dd84289f) | `2026-10-12` | `implementation` | Replaces `new transfer_request_t[request_count]` with `std::vector<transfer_request_t>` in `nixlMooncakeEngine::postXfer`, eliminating memory leaks on early error return |

---

## 2. Deterministic Verification Summary (`own-offline`)

| Metric | Baseline / Before | Phase 113 Verified | Delta / Takeaway |
|---|---|---|---|
| TE `BasicLinear` + `Bias` `torch.compile` forward kernels | `2` (separate GEMM + Bias) | **`1` (fused forward + TP custom op)** | Preserves quantized tensors for backward |
| TE SM120 (`sm_120`) deterministic bwd / no-grad infer backend | Misrouted to non-det FA4 bwd | **Skips FA4 on bwd / keeps `FlashAttention4` on infer** | Uses `requires_backward` + `device` |
| Megatron `Pixtral2DRotaryEmbedding` `patch=63` cos error | `> 0.0001` (BF16 buffer cast) | **`0.0` (FP32 attribute)** | Immune to `Float16Module` cast |
| Megatron `--deterministic-mode` `Ring,NVLS` (SM90 vs SM100) | Rejected everywhere | **`SM90: False` / `SM100: True`** | Enables deterministic Blackwell NVLS |
| Mooncake cached DFS path normalizations across 64 reads | `64` | **`1` (`64.0x` reduction)** | Plus LMCache `request_configs` tag isolation |
| Dynamo decode calls (512 tok, chunk 4, no logprobs) & NIXL leak | `640` decodes / `128` leaked | **`128` decodes (`5.0x` reduction) / `0` leaked** | `needs_token_text` gate + `std::vector` RAII |
