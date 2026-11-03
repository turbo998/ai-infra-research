# Phase 181: Inference Engine & Disaggregated Serving Deterministic Verification Suite

Validates the quantitative inference and serving claims for Phase 181 (`priority: 190`):
- **vLLM `#65030` (`0eeae6a47333`)**: DeepSeek V4/V4.1 Vision `forward_packed` multi-image encoder passes (`2.61x` V4 / `2.10x` V4.1 encoder speedup across 54 calls; 2x B200 `c=16` VisionArena median TTFT reduced from `914–959 ms` to `548 ms`, `1.709x`).
- **vLLM `#63672` (`44965a10974e`) & `#62367` (`9226d6e59cd1`)**: Online FP8 / ModelOpt NVFP4 CUTLASS MoE weight-cache IPC restart fidelity (`0` double-scale bugs) and LFM2 `SHORT_CONV_ATTN` batch invariance (`22/22 = 1.0`).
- **SGLang `#47641` (`4494d682373e`)**: One-warp-per-token CUDA MoE router (`moe_route_warp`, compile-time Batcher network + `redux` top-k) on B200 (`4.045x` on Qwen3.5 512E, `3.031x` on Kimi-K2 384E, `2.0x` on DeepSeek-V3 256E at `T=16384`).
- **SGLang `#48041`, `#46921`, `#45466`**: `HashTopK` breakable-prefill CUDA graph guard, Ascend 950 ModelSlim MLA FP8 KV cache, and grammar poll-timeout cache-poisoning prevention (`0` poisoned entries).
- **TensorRT-LLM `#21198` (`a6a8e8d67183`) & `#21550` (`508f8dddefa8`)**: 8-token tiled QKV preprocessing on B200 (`2.41x` at 4K / `2.67x` at 32K tokens) and GB300 Nemotron-Ultra-V3 `host_cache_size: 0` (`86.24 GiB/rank` unpinned, `390.54 tok/s` median, `0/24` runs below `367.24 tok/s`).
- **Dynamo `#17484` (`2ec0b1112d00`)**: Immediate decode terminal forwarding in SGLang bootstrap handoff (`360 -> 470 tok/s`, `1.306x` P90 per-user throughput).

## Run
```bash
python experiments/vllm_dsv4_packed_vision_sglang_warp_moe_trtllm_qkv_tiling_gb300_host_cache_dynamo_phase181/run.py --check
```
