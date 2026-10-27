# Phase 112 Track B: SGLang Paged-Experts Single-Sync Residency, vLLM Bounded Prompt-Score Prefix Reuse & DSv4.1 FP8 Sparse Indexer, TensorRT-LLM Rubin SM107 MiniMax-M3 (2026-10-28)

> **中文对照 / Chinese Version**: [`sglang-paged-experts-vllm-ds41-fp8-indexer-trtllm-sm107-phase112.md`](./sglang-paged-experts-vllm-ds41-fp8-indexer-trtllm-sm107-phase112.md)  
> **7-Card 4:5 Visual Benchmark Report**: [`../reports/ai-infra-phase112-benchmark-cards.html`](../reports/ai-infra-phase112-benchmark-cards.html)  
> **Reproducible Experiment Suite**: [`../../experiments/sglang_paged_experts_vllm_ds41_fp8_indexer_trtllm_sm107_phase112/`](../../experiments/sglang_paged_experts_vllm_ds41_fp8_indexer_trtllm_sm107_phase112/)

---

## 1. Evidence Classification & First-Hand Commit Ledger (10 Inference & Runtime Commits)

| # | Repo & Commit | Date | Evidence Tier | File Paths & Core Mechanism |
|---|---|---|---|---|
| 1 | [`sglang@a4a19332ab`](https://github.com/sgl-project/sglang/commit/a4a19332abd9b15bdfc2b983e0032b54f687af04) | 2026-10-27 | `implementation` | `python/sglang/srt/layers/moe/paged_experts/{executor,residency,runners}.py`: Computes routed entries per expert on device via `scatter_add_` and copies `[slot_expert, slot_lastuse, step, counts]` in one `.tolist()` D2H sync per layer (`4 → 1` syncs/layer) |
| 2 | [`sglang@69767e5aa6`](https://github.com/sgl-project/sglang/commit/69767e5aa6f2da1eaf135e28f4a46e45c6f35be2) | 2026-10-27 | `implementation` | `python/sglang/multimodal_gen/runtime/utils/camera_geometry.py`: Promotes camera poses and pixel meshgrids to `torch.float32` in `get_plucker_embeddings` so wide frames (`>256` px) do not collapse neighboring pixels onto identical rays under BF16 |
| 3 | [`sglang@c45ebb507d`](https://github.com/sgl-project/sglang/commit/c45ebb507d6e02daa66bef22aef2daf177a13800) | 2026-10-27 | `implementation` | `python/sglang/multimodal_gen/runtime/{models,pipelines}/hunyuan_image3.py`: Preserves layer input for `shared_mlp` when CUDA `FusedMoE` overwrites its input in place, and sets `use_uniform_sigmas=True` on `FlowMatchEulerDiscreteScheduler` |
| 4 | [`vllm@e1e77a279d`](https://github.com/vllm-project/vllm/commit/e1e77a279d68a9601be062cb267a5f9911c422c7) | 2026-10-27 | `implementation` | `docs/training/prompt_token_id_logprobs.md`, `rust/src/text/src/lower/logprobs.rs`: Bounds local prefix cache reuse by `prompt_logprob_start` when `skip_reading_prefix_cache=False`, while setting `skip_reading_external_kv=True` for external KV connectors |
| 5 | [`vllm@a2ad4b74b9`](https://github.com/vllm-project/vllm/commit/a2ad4b74b9f76e99eae3148b729993ba16d474ff) | 2026-10-27 | `implementation` | `vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py`, `vllm/model_executor/layers/sparse_mqa_indexer.py`: Enables DeepGEMM sparse MQA logits (`fp8_fp4_sparse_mqa_logits` + `DeepSelect`) on the SM100 FP8 indexer cache (4 equal UE8M0 bytes per row) for DeepSeek-V4.1 |
| 6 | [`vllm@2897401370`](https://github.com/vllm-project/vllm/commit/289740137027e1d91490fcf9a48490a8a712e200) | 2026-10-27 | `implementation` | `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`: Vectorizes Mooncake KV transfer descriptor contiguous-run detection via NumPy `_contiguous_runs` and caches runs across regions sharing a KV group |
| 7 | [`vllm@22d1bc8814`](https://github.com/vllm-project/vllm/commit/22d1bc881401041310f1c4808f1da64cdf281571) | 2026-10-27 | `implementation` | `rust/src/server/src/routes/openai/{chat_completions,completions}.rs`: Streams `kv_transfer_params` on the terminal `finish_reason` SSE chunk in the Rust server, matching Python |
| 8 | [`vllm@05a303fdc8`](https://github.com/vllm-project/vllm/commit/05a303fdc8681f0dd586a8ac22103f3b4a8bf4ad) | 2026-10-27 | `implementation` | `rust/src/server/src/routes/checkpoint.rs`, `rust/src/engine-core-client/src/protocol/utility.rs`: Exposes HTTP `/checkpoint/{prepare,status,restore}` lifecycle endpoints backed by the 5-state `CheckpointState` machine |
| 9 | [`TensorRT-LLM@854ccee8a9`](https://github.com/NVIDIA/TensorRT-LLM/commit/854ccee8a9f0dded7235a0642032de105b3b47f7) | 2026-10-23 | `implementation` | `tensorrt_llm/_torch/attention/backends/{fmha/flashinfer_trtllm_gen,sparse/minimax_m3/cache_manager}.py`: Enables MiniMax-M3 on Rubin (`SM107`) with NVFP4 KV exemption for sparse MSA layers, 128-token paged FMHA for Eagle3 draft layers, and zero-initialized KV pools |
| 10 | [`TensorRT-LLM@b0af321b0c`](https://github.com/NVIDIA/TensorRT-LLM/commit/b0af321b0c8771291bf201acf8bc2e68586ee5d9) | 2026-10-23 | `implementation` | `tensorrt_llm/_torch/modules/low_m_gemm.py`: Allocates distinct output tensors per invocation during `torch.cuda.is_current_stream_capturing()` so repeated module calls (e.g., draft logits) do not alias and overwrite earlier outputs |

---

## 2. Architectural Mechanisms & Formal Formulations

### 2.1 SGLang Paged-Experts Single-Copy D2H Sync & BF16 Plücker Ray Collapse (`sglang@a4a19332ab`, `sglang@69767e5aa6`)

1. **Coalesced Residency + Route Count Sync (`a4a19332ab`)**: Instead of calling `torch.unique(topk_ids).tolist()`, `slot_expert.tolist()`, `slot_lastuse.tolist()`, and `step.item()` separately ($4$ D2H syncs/layer), SGLang accumulates routed entries per expert into `counts[E+1]` on the GPU via `scatter_add_` and copies `torch.cat([slot_expert, slot_lastuse, step, counts]).tolist()` in **1 D2H sync per layer** ($4\times$ fewer host syncs, saving $2562\text{ }\mu\text{s}$ per step across 61 layers at $14\text{ }\mu\text{s}$/sync).
2. **FP32 Promotion for Plücker Camera Rays (`69767e5aa6`)**: Because BF16 has only 7 mantissa bits, integer pixel coordinates above $256$ step by $2$ and above $512$ step by $4$. Across a $640$-pixel-wide frame, BF16 represents only $416$ distinct column coordinates, collapsing **$224$ columns ($35\%$)** onto neighboring rays. Promoting `c2ws_mat` and `Ks` to `torch.float32` preserves all $640$ rays exactly.

### 2.2 vLLM Bounded Prompt-Score Prefix Cache Reuse & DSv4.1 FP8 Sparse MQA Indexer (`e1e77a279d`, `a2ad4b74b9`)

1. **Block-Aligned Prefix Cache Reuse Before `prompt_logprob_start` (`e1e77a279d`)**: When `skip_reading_prefix_cache=False` and `prompt_logprob_start` $= s > 0$, local prefix cache blocks up to $\lfloor s / B \rfloor \cdot B$ tokens are reused while ensuring every scored row ($s+1 \dots L-1$) is computed; external KV connectors remain disabled (`skip_reading_external_kv = True`).
2. **SM100 FP8 Indexer Cache UE8M0 Packing (`a2ad4b74b9`)**: On SM100, the FP8 indexer cache stores each token's power-of-two scale $2^{\lceil \log_2(\text{amax}/448) \rceil}$ replicated across 4 equal UE8M0 bytes in an `int32`. This matches DeepGEMM's granularity-32 UE8M0 contract directly, cutting indexer scoring complexity from $O(L)$ to $O(N_{\text{cand}}\cdot B_{\text{cand}})$ (**$16\times$ reduction** at $64\text{K}$ context with $64 \times 64 = 4096$ candidate tokens).

---

## 3. Deterministic Offline Verification Results (`own-offline`)

Validated via `python experiments/sglang_paged_experts_vllm_ds41_fp8_indexer_trtllm_sm107_phase112/run.py --check` ([`summary.json`](../../experiments/sglang_paged_experts_vllm_ds41_fp8_indexer_trtllm_sm107_phase112/summary.json)):

| Dimension | Legacy / Baseline | Fixed / Optimized | Key Quantitative Metric |
|---|---|---|---|
| SGLang Paged-Experts D2H Syncs (61 layers) | `244` D2H syncs/step | `61` D2H syncs/step | **`4.0x`** sync reduction (`2562 µs` saved/step) |
| SGLang Plücker Ray Pixel Grid ($W=640$) | `416` distinct columns in BF16 | `640` distinct columns in FP32 | Eliminates **`224` (`35%`)** collapsed columns |
| vLLM Prompt-Score Prefix Cache ($B=4, s=5$) | `0` tokens reused | `4` local tokens reused (`0` external) | Exact block-aligned reuse before first scored row |
| vLLM DSv4.1 FP8 Sparse MQA Indexer ($64\text{K}$) | `65,536` dense tokens scored | `4,096` candidate tokens (`0x7b7b7b7b`) | **`16.0x`** reduction in indexer logit work |
| TRT-LLM Low-M GEMM Capture (3 draft steps) | Aliased `[33.0, 33.0, 33.0]` | Distinct `[11.0, 22.0, 33.0]` | Zero output aliasing during CUDA graph capture |
