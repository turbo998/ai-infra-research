# Inference & Disaggregated Serving Stack: vLLM DeepSeek V4/V4.1 Packed Vision & Online FP8/NVFP4 Weight-Cache Restarts, SGLang One-Warp MoE Router, TRT-LLM 8-Token Tiled QKV & GB300 Host-Cache Zeroing, and Dynamo Immediate Decode Terminal Forwarding (Phase 181)

> 🌐 **Language / 语言切换**: **[中文](inference-vllm-dsv4-packed-vision-weight-cache-sglang-warp-moe-router-trtllm-qkv-tiling-gb300-host-cache-dynamo-handoff-phase181.md)** | **[English](inference-vllm-dsv4-packed-vision-weight-cache-sglang-warp-moe-router-trtllm-qkv-tiling-gb300-host-cache-dynamo-handoff-phase181.en.md)**

---

## 1. Core Architectural Problems & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 181 (`priority: 190`) audits **11 upstream mainline commits** across vLLM, SGLang, TensorRT-LLM, and NVIDIA Dynamo:

1. **vLLM DeepSeek V4/V4.1 Vision `forward_packed` Multi-Image Encoding, Fast-Start Online FP8 / CUTLASS NVFP4 Weight-Cache IPC Restarts & ShortConv (`LFM2`) Batch Invariance**:
   - `0eeae6a47333` (#65030) encodes all images in a multi-image call in a single packed pass via `forward_packed` on DeepSeek V4 and V4.1 Vision, achieving **`2.61x`** total encoder speedup on V4 (`962.9 -> 328.4 ms` at 32 images) and **`2.10x`** on V4.1 (`961.2 -> 418.6 ms` across 54 H100 multi-image calls); on 2x B200 with `DeepSeek-V4-Flash-Vision-Exp`, median TTFT drops from `367–378 ms` to **`271 ms`** (`c=1`, 4 images) and from `914–959 ms` to **`548 ms`** (`c=16`, VisionArena, **`1.709x`** speedup) while matching MMMU-Pro accuracy (`862 vs 861/860`).
   - `44965a10974e` (#63672) enables the weight-cache daemon to reuse online FP8/MXFP8 linear weights and ModelOpt NVFP4 CUTLASS MoE weights across engine IPC restarts without re-quantizing, preserving unpadded CUTLASS FP8 output width and preventing activation scales from being multiplied into shared NVFP4 weight scales a second time (`0` double-scale bugs).
   - `9226d6e59cd1` (#62367) enables `VLLM_BATCH_INVARIANT=1` for `SHORT_CONV_ATTN` (`Lfm2ForCausalLM`, `Lfm2MoeForCausalLM`), passing `22/22` tests in `test_batch_invariance.py` on `LiquidAI/LFM2-8B-A1B` and eliminating round-0 prefix-cache output divergence.
2. **SGLang One-Warp-Per-Token CUDA MoE Router (`moe_route_warp`), `HashTopK` Prefill Graph Guard, Ascend 950 ModelSlim MLA FP8 KV Cache & Grammar Timeout De-Poisoning**:
   - `4494d682373e` (#47641) introduces `moe_route_warp` (one warp per token, $E/32$ experts/lane as order-preserving `u32` keys sorted once via a compile-time Batcher network, with `redux` max + min-lane tie-break per top-$k$ round for $128\dots 512$ experts), cutting B200 router latency at `T=16384` on Qwen3.5 (512E, top-10) from `63.1 -> 15.6 us` (**`4.045x`**), Kimi-K2 (384E) from `39.7 -> 13.1 us` (**`3.031x`**), and DeepSeek-V3 (256E) from `24.6 -> 12.3 us` (**`2.000x`**).
   - `1ef3335da114` (#48041) checks `is_hash` before reading `topk_config` in breakable-prefill dual-stream CUDA graph capture and vision top-$k$ paths; `99fd01047d67` (#46921) adds ModelSlim static `Q_FP8_DYNAMIC_KV_FP8` latent KV cache quantization and `MLAProlog` on Ascend 950 NPUs; `86ab345175d0` (#45466) stops `GrammarManager` from caching `InvalidGrammarObject("Grammar preprocessing timed out")` when `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS` is exceeded under load bursts (`0` poisoned schema cache entries).
3. **TensorRT-LLM 8-Token Tiled QKV Preprocessing, GB300 Nemotron-Ultra-V3 Host-Cache Zeroing against NUMA Compaction Stalls & Disaggregated KV Timeout Attribution**:
   - `a6a8e8d67183` (#21198) adds opt-in 8-token tiled fused QK-norm + RoPE + gate extraction (`TRTLLM_ENABLE_QKV_ROPE_TILING=1` for `tokens >= 256`), delivering **`2.41x`** and **`2.67x`** faster preprocessing at `4,096` and `32,737` tokens on B200.
   - `508f8dddefa8` (#21550) eliminates `86.24 GiB/rank` (`79.61 GiB` target + `6.63 GiB` draft) automatic host-tier pinning on 4x GB300 Nemotron-Ultra-V3 FP4 (`TEP4 MTP6`) by setting `host_cache_size: 0`. Previously, pinning fragmented NUMA nodes $>90\%$ at the 512 MiB huge-page order and triggered 13–49 s Linux kernel proactive-compaction stalls (`8.88–9.02 ms` vs `8.10–8.47 ms` decode steps, dropping throughput to `326.28–382.78 tok/s`). With `host_cache_size: 0`, median throughput reaches **`390.54 tok/s`** and runs below `367.24 tok/s` drop from `4/24` to **`0/24`**.
   - `da980cce4754` (#20708) attributes disaggregated KV-transfer timeouts to specific deadline owners and non-blocking TX/RX progress; `c26c166b8e49` (#21199) rejects encoder inputs exceeding single-iteration bounds at request admission.
4. **NVIDIA Dynamo Immediate Decode Terminal Forwarding in Disaggregated Serving & Declared GPU Costs for Mocker Pools**:
   - `2ec0b1112d00` (#17484) forwards a successful decode terminal (`finish_reason` + trailing tokens) immediately in disaggregated SGLang bootstrap handoff instead of holding it for ~1.5 prefill scheduler iterations until the prefill reply returns (which previously accounted for 94% of the slowest 10% requests), boosting agentic multi-turn P90 per-user output token throughput from **`360 tok/s` to `470 tok/s` (`1.306x` / `+30.6%`)**; `c80a17b4631e` (#14055) enables declared GPU costs for zero-GPU mocker pools in Global Planner budget arbitration.

---

## 2. End-to-End Dataflow & Architecture Topology (Pillar 1)

```
+------------------------------------------------------------------------------------------------------+
| Phase 181 Inference & Disaggregated Serving: vLLM, SGLang, TensorRT-LLM & NVIDIA Dynamo              |
+------------------------------------------------------------------------------------------------------+
  [vLLM Multimodal, Fast-Start Weight-Cache IPC & Determinism (#65030, #63672, #62367)]
    |-- DeepSeek V4/V4.1 Vision (#65030)--> forward_packed Multi-Image Pass (2.61x V4, c=16 TTFT 548 ms)
    |-- Fast-Start Weight Cache (#63672)--> Online FP8/MXFP8 & ModelOpt NVFP4 MoE IPC Restart (0 Re-Scale)
    +-- LFM2 ShortConv (#62367)         --> VLLM_BATCH_INVARIANT=1 Support (22/22 Determinism Tests Pass)
                                      |
                                      v
  [SGLang Warp MoE Router, TRT-LLM B200/GB300 & Dynamo Disagg Handoff (#47641, #21198, #21550, #17484)]
    |-- SGLang moe_route_warp (#47641)  --> 1-Warp/Token Batcher Sort + redux Top-K (4.045x Qwen3.5 512E)
    |-- SGLang Reliability (#48041/#45466)-> HashTopK Dual-Stream Guard + No Grammar Poll-Timeout Poisoning
    |-- TRT-LLM B200/GB300 (#21198/#21550)-> 8-Tok QKV Tiling (2.67x) + GB300 host_cache_size:0 (-86.24 GiB)
    +-- Dynamo Prefill-Router (#17484)  --> Immediate Decode Terminal Forward (P90 360 -> 470 tok/s, 1.306x)
```

---

## 3. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/vllm_dsv4_packed_vision_sglang_warp_moe_trtllm_qkv_tiling_gb300_host_cache_dynamo_phase181/run.py`](../../experiments/vllm_dsv4_packed_vision_sglang_warp_moe_trtllm_qkv_tiling_gb300_host_cache_dynamo_phase181/run.py) `--check`:
- `vllm_dsv4_packed_vision_54call_speedup = 2.61`, `vllm_dsv41_packed_vision_54call_speedup = 2.10`, `vllm_dsv4_b200_c16_ttft_speedup = 1.709` (`548.0 ms`)
- `vllm_weight_cache_nvfp4_double_scale_bugs = 0`, `vllm_lfm2_shortconv_batch_invariance_pass_rate = 1.0`
- `sglang_moe_route_warp_qwen35_t16384_speedup = 4.045`, `sglang_moe_route_warp_kimik2_t16384_speedup = 3.031`, `sglang_moe_route_warp_dsv3_t16384_speedup = 2.0`
- `trtllm_b200_qkv_rope_tiling_4k_speedup = 2.41`, `trtllm_b200_qkv_rope_tiling_32k_speedup = 2.67`
- `trtllm_gb300_nemotron_host_pin_avoided_gib_per_rank = 86.24`, `trtllm_gb300_nemotron_median_tok_s = 390.54`, `trtllm_gb300_nemotron_runs_below_367_tok_s = 0`
- `dynamo_disagg_decode_terminal_p90_speedup = 1.306` (`360 -> 470 tok/s`)

---

## 4. Enterprise Business Scenario Playbook (Pillar 4)

| Scenario Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **Multi-Image VLM & High-Expert-Count MoE Serving** | Enable `forward_packed` multi-image passes on vLLM DeepSeek V4/V4.1 Vision (`#65030`, `2.61x` encoder speedup, `548 ms` TTFT at `c=16`) and `moe_route_warp` on SGLang B200/H200 for $128\dots 512$ experts (`#47641`, `4.045x` on Qwen3.5). | 32-image calls on GPUs with `< 2 GiB` free headroom without capping pass size (`forward_packed` adds `1.7–2.0 GB` peak encoder memory at 32 images), or `>512`-expert models (e.g., Kimi-K3 896E). |
| **Grace Blackwell (GB300) Serving & Fast Warm Restarts** | Set `host_cache_size: 0` on GB300 workloads that do not require host-tier KV offloading (`#21550`, avoiding `86.24 GiB/rank` pinned memory and 13–49 s NUMA 512 MiB huge-page proactive compaction stalls), and enable online FP8/NVFP4 weight-cache IPC restarts in vLLM (`#63672`). | Enabling `TRTLLM_ENABLE_QKV_ROPE_TILING=1` (`#21198`) when strict cross-batch token-for-token bit invariance is required across dispatch boundaries. |
| **Disaggregated Multi-Turn Agent Serving** | Enable immediate decode terminal forwarding in Dynamo + SGLang bootstrap handoff (`#17484`, `+30.6%` P90 user tok/s) and deadline-owner KV timeout attribution in TRT-LLM (`#20708`). | Dropping the underlying decode worker stream before reading to EOF (causes `Failed to publish complete final` errors on the worker). |
