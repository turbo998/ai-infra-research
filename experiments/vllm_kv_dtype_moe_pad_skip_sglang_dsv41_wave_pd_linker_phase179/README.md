# vLLM Per-Layer KV Dtype & Fused MoE Pad Skip, SGLang DSv4.1 Wave Rounding & PD Linker Dedup, and TRT-LLM GB300 Qwen3.5 (Phase 179)

Deterministic offline verification suite for Phase 179 (`priority: 188`, Track 2: `inference`).

## Audited Upstream Commits
- `vllm-project/vllm` @ `8530a778f78d` (#56155): Resolve per-layer KV dtype in attention runtime (`43.75%` KV memory reduction on hybrid 56-FP8/8-BF16 models)
- `vllm-project/vllm` @ `2a4792ddcc4b` (#64955): Fuse padding skipping in grouped and Muse prepared MoE routing (`1.21x` speedup)
- `vllm-project/vllm` @ `63d3d39c8472` (#59961) & `e0d00e4b8eb0` (#64607): Carry shared DSv3.2 indexer top-k across PP stages & add `Qwen3.5-397B-A17B` B200 NVFP4 TP2 agentic preset (`1.34x` speedup)
- `sgl-project/sglang` @ `f171859219b5` (#47684) & `e232ebf6b0b7` (#47236): Round DeepSeek-V4.1 dense indexer logits tiles to whole SM waves (`1.19x` speedup) & AMD Triton softmax top-k router for Qwen3-Next (`1.27x` speedup)
- `sgl-project/sglang` @ `cb0a7a953fa6` (#46395) & `c757c4bb7c55` (#47732): Write PD decode KV to linker skipping prefill-stored pages (`68.75%` write traffic reduction) & offset `kv_events` IPC endpoints per DP rank
- `NVIDIA/TensorRT-LLM` @ `5d86481f60bd` (#20815) & `7a62522fb820` (#20900): GB300 Qwen3.5 AgentX side-branch optimizations (`1.23x` speedup) & `FORCE_CHUNK` micro-batch scheduler context limit check
