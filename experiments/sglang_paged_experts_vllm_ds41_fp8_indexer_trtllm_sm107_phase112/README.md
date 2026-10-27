# Phase 112 Track B: SGLang Paged-Experts Single-Sync, vLLM Prompt-Score Prefix Reuse & DSv4.1 FP8 Sparse Indexer, TRT-LLM SM107 MiniMax-M3

Deterministic stdlib model covering 10 first-hand commits across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`:
- `sglang@a4a19332ab`: Paged-experts single-copy D2H residency + `scatter_add_` routed expert counts (`4 -> 1` D2H syncs/layer).
- `sglang@69767e5aa6` & `sglang@c45ebb507d`: Diffusion `float32` Plücker ray promotion (`224` collapsed columns prevented at width `640`) & HunyuanImage-3 in-place `FusedMoE` shared MLP input preservation.
- `vllm@e1e77a279d`: Prefix cache reuse bounded by `prompt_logprob_start` (`block-aligned` local reuse before first scored row; external KV connector disabled).
- `vllm@a2ad4b74b9`: DeepSeek-V4.1 `SparseMQAIndexer` on SM100 FP8 indexer cache (`4x` repeated UE8M0 byte scale, `16x` work reduction at `64K`).
- `vllm@2897401370`, `vllm@22d1bc8814`, `vllm@05a303fdc8`: Mooncake vectorized `_contiguous_runs`, Rust final-chunk `kv_transfer_params`, and `/checkpoint/{prepare,status,restore}` lifecycle.
- `TensorRT-LLM@854ccee8a9` & `TensorRT-LLM@b0af321b0c`: Rubin SM107 MiniMax-M3 (`NVFP4` KV exemption + `128`-token FMHA draft route) & Low-M GEMM capture distinct output buffers.
