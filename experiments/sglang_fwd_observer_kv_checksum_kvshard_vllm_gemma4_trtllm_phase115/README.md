# Phase 115 Track B: SGLang Forward Observer, Radix KV Checksum & 0-SM KVShard, vLLM Gemma-4 Padding Skip & GLM-5.3 MLA Scale Skip, and TensorRT-LLM SM107 MXFP8 MoE & KVCM v2 Retention

Deterministic stdlib model verifying:
- `sgl-project/sglang@4a58dc9838`: scheduler forward observer & deferred plugin responses (`6.25 -> 4.88 ms`, `1.2807x`).
- `sgl-project/sglang@ee98bb09b0`: decode-side RadixCache KV checksum verification (`4/4` silent corruptions caught).
- `sgl-project/sglang@c00e7e12b6`: HiSparse DeepSeek-V4 token-by-token eager backup copy (`1152.0 -> 72.0 KiB`, `16.0x` reduction).
- `sgl-project/sglang@3ea7e210d2`: KVShard 0-SM KV cache AllGather (`16 -> 0` SMs reserved, `+12.12%` compute SMs, `1.1324x` step speedup).
- `vllm-project/vllm@89f0c8c668`, `0a95e4d73c`, `56e4e42868`: Gemma-4 MoE CUDA-graph padding row skip (`28.4 -> 9.2 us`, `3.087x`), XPU per-token dynamic FP8 quant (`+12.4 dB` SQNR), and GLM-5.3 `concat_and_cache_mla` scale skip (`2.57%` TPOT reduction).
- `NVIDIA/TensorRT-LLM@39f300f1bd`, `4636b6311f`, `41913522c7`: OpenEngine gRPC binary image inputs (`4.4211x` faster deserialization), Rubin SM107 CUTLASS MXFP8 MoE (`1.7215x` speedup), and KVCM v2 priority retention (`61.0% -> 96.5%` system-prefix hit rate).
