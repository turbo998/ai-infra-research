# SGLang MNNVL NVML Fabric Detection, SM107 KDA Spec-Verify BV=4, DCP Block-FP8 Q-Proj Replication, vLLM FP8 Indexer WK Persistence, cuBLAS Workspace Release & TensorRT-LLM Gemma4 / INT8-WOQ MoE Padding

Deterministic Python 3.12 stdlib simulator verifying:
1. SGLang MNNVL NVML fabric registration detection (`39b54ae0eeca`), SM107 KDA spec-verify `BV=4` tile narrowing (`640e11ebc0df`, `13.66 us -> 6.99 us`, `1.954x`), DCP `128x128` block-FP8 `q_b_proj` replication (`f264183bfe1d`), and DeepSeek-V4 MoE local finalize guard (`ad950cdcd966`).
2. vLLM DeepSeek-V3.2 FP8 `indexer.wk` pair persistence across incremental `load_weights` calls (`2c880f1e6820`), `WeightCacheDaemon` post-warmup cuBLAS workspace release (`b3466364b69d`), and NIXL OBJ tier `capture_telemetry=True` (`4a81a5097631`).
3. TensorRT-LLM Gemma4 multimodal encoder budget scheduling (`abb94ab8fc1c`) and INT8-WOQ MoE intermediate 64-row alignment padding (`64ef54ce1249`).
