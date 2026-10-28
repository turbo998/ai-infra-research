# SGLang ROCm 10.1 Profiler Inheritance Fix, Diffusion LoRA & Breakable CUDA Graphs, vLLM Sleep State Telemetry & Harmony Reasoning Tokens, and TensorRT-LLM CuteDSL NVFP4 MoE & KDA Prefill (Phase 122)

Reproducible stdlib model covering:
- `sgl-project/sglang#44292` (`291f38c3`), `#45833` (`67b3c9f9`), & `#46660` (`a912002e`): Pre-`torch` `ROCPROFILER_REGISTER_LIBRARY` cleanup on ROCm 10.1 (`0 -> 29,728` GPU trace events on `gfx1250`), fractional `alpha` + Kohya RSLoRA scaling, and caller-allocated output buffers for 4 breakable CUDA graph sites.
- `vllm-project/vllm#63959` (`9df11cf6`) & `#63960` (`00d4b782`): Rust frontend `vllm:engine_sleep_state` Prometheus gauges (`awake`, `weights_offloaded`, `discard_all`) and `gpt-oss` Harmony reasoning token accounting (`analysis` + tool-targeted `commentary`).
- `NVIDIA/TensorRT-LLM#21101` (`f6c4125c`) & `#20653` (`222676c8`): CuteDSL NVFP4 MoE FP32-chain verification (`0%` mismatch vs TRTLLM-Gen on Mixtral `14336` / Grok `32768`) and FlashInfer KDA prefill with fused CuTe DSL conv + SiLU warp-vote fallback (`293 passed` on B200).
