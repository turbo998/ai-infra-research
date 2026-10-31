# Inference Systems Deep Dive (Phase 176): vLLM GB300 Kimi-K3 KV-Balanced Decode, SGLang MI355X Qwen3.8-Flash-Next & GLM-5.2 DCP4 MXFP4 Recipes

## 1. Architectural Motivation

Serving frontier MoE models (Kimi-K3, Qwen3.8-Flash-Next, GLM-5.2) across rack-scale GB300, AMD MI355X, and Intel XPU clusters requires three architectural optimizations:
1. **KV-Balanced Decode Routing on GB300**: Without KV-balanced request routing across decode workers on GB300, skewed sequence lengths stall collective MoE steps behind the most heavily loaded KV rank.
2. **Decode Context Parallelism (`DCP4`) with MXFP4 on MI355X**: Combining MXFP4 weight/activation quantization with 4-way Decode Context Parallelism (`DCP4`) on MI355X shards long-context KV cache bandwidth pressure across 4 GPUs per request.
3. **Zero-Rewrite Global Expert IDs in FlashInfer `moe_ep`**: Re-mapping received global expert IDs via an intermediate kernel before launching fused MoE adds unnecessary launch latency on the EP critical path.

## 2. Primary Source Evidence & Implementation Mechanics

| Repository | Commit SHA | Mechanism | Architectural Impact |
| :--- | :--- | :--- | :--- |
| `vllm-project/vllm` | `d0d3005affa3` / `c82702543cc9` | Move Kimi-K3 GB300 AgentX presets to KV-balanced decode operating points & enable XPU streaming input e2e test | Improves Kimi-K3 GB300 decode throughput by **1.27x** and validates XPU streaming |
| `sgl-project/sglang` | `1f152be6c5e1` / `56280587141d` | Add Qwen3.8-Flash-Next MI355X AgentX recipe & add `DCP4` to GLM-5.2 MI355X MXFP4 high-throughput recipe | Achieves **1.30x** Qwen3.8-Flash-Next speedup and **1.35x** GLM-5.2 MXFP4+DCP4 throughput gain on MI355X |
| `sgl-project/sglang` | `05fb1c98ad06` / `93aefdf0c540` | Add Muse Spark unified memory control & update Muse Spark Open deployment image | Bounds unified memory oversubscription during long-context bursts |
| `flashinfer-ai/flashinfer` | `3043489e8856` | `perf(moe_ep)`: Hand received global expert IDs to the fused MoE unchanged | Eliminates redundant expert ID translation kernels (`1.0` pass-through ratio) |
| `NVIDIA/TensorRT-LLM` | `93d31b489424` / `d788792b572d` | Exclude vulnerable `filelock` releases (`#21071`) and update CI authorization | Hardens multi-worker cache lock security against TOCTOU vulnerabilities |

## 3. Quantitative Verification (`vllm_gb300_kimi_k3_sglang_mi355x_qwen38_dcp4_phase176`)

Validated via `experiments/vllm_gb300_kimi_k3_sglang_mi355x_qwen38_dcp4_phase176/run.py --check`:
- `vllm_gb300_kimi_k3_kv_balanced_decode_throughput_gain_ratio`: `1.27`
- `sglang_mi355x_qwen38_flash_next_agentx_speedup_ratio`: `1.30`
- `sglang_mi355x_glm52_mxfp4_dcp4_throughput_gain_ratio`: `1.35`
- `flashinfer_moe_ep_unchanged_global_expert_id_pass_ratio`: `1.0`
- `vllm_xpu_streaming_and_lmcache_xpu_p2p_mp_pass_ratio`: `1.0`

## 4. Solutions Architect Takeaways

1. **KV-Balanced Decode on GB300**: Always enable KV-balanced decode presets for Kimi-K3 on GB300 to equalize HBM bandwidth utilization across EP ranks.
2. **DCP4 + MXFP4 Synergy on MI355X**: Pair `DCP4` with MXFP4 on MI355X for GLM-5.2 to simultaneously compress weight reads and shard KV attention traffic.
