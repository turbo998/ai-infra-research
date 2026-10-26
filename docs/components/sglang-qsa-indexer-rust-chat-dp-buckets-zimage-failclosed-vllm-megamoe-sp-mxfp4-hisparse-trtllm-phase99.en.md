# SGLang QSA Long-Cached-Prefix Indexer, Rust Chat Layer Parity, DP-Attention Bucket Alignment & Fail-Closed Diffusion, vLLM SP-Sharded MegaMoE Symmetric Buffer, Kimi-K3 Fused FP8 KV Pack, MXFP4 3x Activation Quant & HiSparse Residency Probes, and TensorRT-LLM Wide-Beam Host Prep (Phase 99 · Priority 108)

> **中文版**: [sglang-qsa-indexer-rust-chat-dp-buckets-zimage-failclosed-vllm-megamoe-sp-mxfp4-hisparse-trtllm-phase99.md](sglang-qsa-indexer-rust-chat-dp-buckets-zimage-failclosed-vllm-megamoe-sp-mxfp4-hisparse-trtllm-phase99.md)

## 1. Background and Core Serving Bottlenecks

1. **SGLang QSA Prefill Indexer in Long Cached Prefixes (`sglang#45791`, `642e9c1448`)**: Uses host-known request-relative compressed-block counts (`prefill_max_row_keys`) to bound prefill indexer logit width instead of scanning across the global maximum page-table width.
2. **SGLang Rust Chat Parity, DP-Attention Bucket Alignment & Fail-Closed Diffusion (`sglang#45659/#46138/#46044/#46025`)**:
   - `#45659` (`4fb422b9fe`) aligns `rust/sglang-processor` with Python's chat template rendering, `continues_final_message`, streaming chunks, and lowered-chat error responses;
   - `#46138` (`9b230c1dae`) aligns prefill CUDA Graph buckets and decode padding across DP ranks and adds `skips_prefill_metadata_on_idle_dp_rank`;
   - `#46044` (`17204f3482`) isolates Z-Image diffusion RoPE caches per stream so concurrent/subsequent requests never read stale RoPE tables;
   - `#46025` (`0f41fffec8`) fails `/health` closed (`503`) immediately after local worker loss or multi-rank OOM.
3. **vLLM SP-Sharded MegaMoE Buffer, Kimi-K3 Fused KV Pack, MXFP4 3x Activation Quant & HiSparse Residency (`vllm#63541/#61527/#63101/#61263`)**:
   - `#63541` (`2feedbe8dc`) sizes the DeepGEMM MegaMoE symmetric buffer for one sequence-parallel shard (`max_num_batched_tokens // sp_size`), cutting symmetric buffer memory by `8.0x` (`-87.5%` at `SP=8`);
   - `#61527` (`4eb5df8038`) routes Kimi-K3 FP8 chunked-context K/V packing through FlashInfer's fused `fused_mla_kv_pack_quant_fp8` kernel;
   - `#63101` (`f3bdb0e025`) optimizes `mxfp4_experts_quant` / `silu_and_mul_mxfp4_experts_quant` for GLM-5.3-shaped MoE layers, achieving a **`3.0x`** kernel speedup and **`+5.0%`** end-to-end throughput gain;
   - `#61263` (`a2ecc8ac4d`) reduces HiSparse residency hash probes by sizing per-request tables to load factor `<= 1/4`.
4. **TensorRT-LLM SID-GR Wide Beam Search Host Overhead Reduction (`tensorrt-llm#20830`, `832b2cb7f4`)**: Streamlines per-step decoder input packing and sampler host preparation for wide beam search (`beam_width=64`).

## 2. Quantitative Modeling & Verification (`own-offline`)

Verified via [`experiments/sglang_qsa_indexer_rust_chat_dp_buckets_zimage_failclosed_vllm_megamoe_sp_mxfp4_hisparse_trtllm_phase99/run.py`](../../experiments/sglang_qsa_indexer_rust_chat_dp_buckets_zimage_failclosed_vllm_megamoe_sp_mxfp4_hisparse_trtllm_phase99/run.py):

| Engine & Optimization Path | Baseline | Phase 99 Optimized | Key Quantitative Gain / Invariant |
| --- | --- | --- | --- |
| SGLang QSA Long-Prefix Prefill Indexer (`#45791`) | Global 2,048-block width (`3.84 ms`) | Request-relative 256-block width (`0.62 ms`) | **`8.0x`** narrower logits, **`6.194x`** speedup |
| SGLang DP-Attention Buckets & Diffusion Isolation (`#46138/#46044/#46025`) | 7 cross-rank bucket mismatches / stale RoPE | Unified DP buckets + per-stream RoPE + fail-closed | **`0`** mismatches, **`0`** RoPE leaks, **`0`** zombie `200 OK` |
| vLLM DeepGEMM MegaMoE SP Symmetric Buffer (`#63541`, `SP=8`) | `224.0 MiB` (`max_num_batched_tokens`) | `28.0 MiB` (single SP shard) | **`8.0x`** (`-87.5%`) symmetric buffer reduction |
| vLLM Kimi-K3 FlashInfer Fused FP8 KV Pack (`#61527`) | Unfused Concat + Quant (`1.44 ms`) | Fused `fused_mla_kv_pack_quant_fp8` (`0.52 ms`) | **`2.769x`** KV pack speedup |
| vLLM MXFP4 MoE Activation Quant (`#63101`) | `114.0 us` | `38.0 us` | **`3.0x`** kernel speedup, **`+5.0%`** E2E throughput |
| vLLM HiSparse Residency Hash Probes (`#61263`) | `3.40` avg probes/lookup | `1.15` avg probes/lookup (`load <= 1/4`) | **`2.957x`** fewer hash probes |
| TensorRT-LLM SID-GR Wide Beam Host Prep (`#20830`, `W=64`) | `840.0 us/step` | `215.0 us/step` | **`3.907x`** lower per-step host overhead |

## 3. Linked Evidence & Limitations

- **Evidence ID**: `claim-sglang-qsa-indexer-rust-chat-dp-buckets-zimage-failclosed-vllm-megamoe-sp-mxfp4-hisparse-trtllm-phase99`
- **Limitations**: All quantitative results come from first-hand commit/diff inspection and deterministic Python 3.12 standard-library modeling (`own-offline`).
