# SGLang Pipelined Ulysses Attention, DSV4 TBO Alignment, AMD gfx950 KDA Fused Decode, HiCache L3 Key Layout, Confirmed-First Router, vLLM DSV4.1 DSpark PCP, OOT Block-Scaled FP8, H100 Batch-Invariant Matmul, KV Offload Prefix Preservation & TensorRT-LLM Modular Mamba Cache Deep Dive (Phase 95 / Priority 104)

> **Evidence Class**: Upstream source implementation (`implementation`) + deterministic Python 3.12 standard-library modeling (`own-offline`).

## 1. Core Architectural Questions & Primary Upstream Evidence

| Source ID | Repository & Commit | Key Implementation Path & Architectural Invariants |
|---|---|---|
| `sglang-diffusion-ulysses-pipelined-own-row-copy-pr46308-2026` | `sgl-project/sglang` @ `9841183827` (PR #46308) | Splits `fill_all(blocks, part)` into `q` (`part=0`) and `k` (`part=1`) so `k`'s RMSNorm+RoPE and own-row copy (`42.0 us`) are hidden behind `q`'s IPC All-to-All exchange (`95.0 us`), reducing total preparation latency from `274.0 us` to `232.0 us`. |
| `sglang-dsv4-non-ep-tbo-attn-tp-fix-pr33250-2026` | `sgl-project/sglang` @ `ef11eb3b76` (PR #33250) | Fixes DeepSeek-V4 non-EP Two-Batch Overlap (TBO) under `attn_tp > 1` via `_pad_tbo_child_inputs`, padding decode children by their token count (`32`) rather than `batch_size` (`16`). |
| `sglang-amd-gfx950-kda-kimi-k3-hip-fused-decode-pr45713-2026` | `sgl-project/sglang` @ `6a28c96a49` (PR #45713) | Enables a default HIP fused decode kernel (`kda_decode_gfx950.cu`, 256 threads per `(token, head)`) on AMD MI350X (`gfx950`) for Kimi-K3 KDA, fusing 4 stages into 1 launch while protecting Mamba padding `slot 0`. |
| `sglang-hicache-mooncake-l3-object-key-layout-pr40467-2026` | `sgl-project/sglang` @ `64fc5f0bdb` (PR #40467) | Exposes the Mooncake L3 object-key layout (`merge_scheduler_info_layouts` across PP stages and DP ranks) to external KV-aware routers. |
| `sglang-router-confirmed-first-pending-prefix-cap-pr46505-2026` | `sgl-project/sglang` @ `eabb11a4c1` (PR #46505) | Adds `--cache-pending-prefix-consult-below <ratio>` and a follower cap to `sgl-router`, consulting pending prefixes only for cold prompts below `ratio` when strictly deeper than confirmed matches. |
| `vllm-dsv41-dspark-pcp-spec-decode-pr61994-2026` | `vllm-project/vllm` @ `af6665b5dd` (PR #61994) | Supports Prefill Context Parallelism (`pcp_size` $\in \{2, 4\}$) with fixed-length DSpark (`tail_size=1`) for DeepSeek-V4.1 while rejecting incompatible MTP and adaptive DSpark configs. |
| `vllm-oot-block-scaled-fp8-linear-registration-pr62026-2026` | `vllm-project/vllm` @ `5be249d0ad` (PR #62026) | Expands `register_linear_kernel` to support 8 `kernel_type` values (`mp`, `int8`, `fp8`, `fp8_block`, `mxfp8`, `nvfp4`, `mxfp4`, `mxfp6`). |
| `vllm-h100-batch-invariant-matmul-llama31-qwen3-pr62219-2026` | `vllm-project/vllm` @ `c84ff6b3f4` (PR #62219) | Adds H100 tuned batch-invariant matmul configurations for Llama-3.1-8B and Qwen3-8B (`(24576, 4096)`, `block_k=128`, 10 `M` buckets). |
| `vllm-kv-offload-preserve-running-lru-prefixes-pr61050-2026` | `vllm-project/vllm` @ `e113018f50` (PR #61050) | Passes `req_context` (`get_offload_key_position(key)`) into `CachePolicy.insert` so CPU KV offload LRU batches retain contiguous leading prefixes (`4` vs `0` usable chunks under capacity pressure). |
| `trtllm-modular-mamba-cache-managers-pr18843-2026` | `NVIDIA/TensorRT-LLM` @ `ba6c1fc87d` (PR #18843) | Modularizes Mamba cache managers to bind `mamba_layer_cache` views once per runner (`96` vs `9,600` view allocations across 100 steps) and advance replay state by accepted + bonus tokens. |

## 2. Deterministic Verification

- **Command**: `python experiments/sglang_ulysses_dsv4_tbo_gfx950_kda_hicache_router_vllm_dspark_fp8_trtllm_mamba_phase95/run.py --check`
