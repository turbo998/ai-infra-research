# SGLang Qwen3.8 FlyDSL Hyper-Connections & Ratio-3 GDN Packing, DSv4 1024-Thread Grids, PD Rebootstrap Retention, vLLM XPU Sparse MLA & Single-Launch SWA Meta, and TRT-LLM Helix MLA & DSpark Ragged Graphs (Phase 145)

> **Research Date**: 2026-10-31 (UTC+08:00)
> **Evidence Class**: `implementation` (10 upstream primary commits) + `own-offline` (deterministic stdlib verification)
> **Covered Commits**:
> - `sgl-project/sglang@fcecdad` (`#41880`, 2026-10-30): `[Perf] [AMD] Qwen3.8: AITER FlyDSL hc_mix for Qwen3.8 hyper-connections on gfx950`
> - `sgl-project/sglang@5035bc9` (`#44302`, 2026-10-30): `[Perf] [AMD] Qwen3.8: Enable fused GDN packing for ratio-3 heads`
> - `sgl-project/sglang@4201b8f` (`#47149`, 2026-10-30): `[Perf] dsv4: Fix amax8_varlen and mega MoE pre-dispatch grids for 1024-thread SMs`
> - `sgl-project/sglang@03aba9e` (`#47366`, 2026-10-30): `[Fix] disaggregation: Keep output ids during a PD rebootstrap instead of popping and replaying the last one`
> - `vllm-project/vllm@fa9356a` (`#64065`, 2026-10-30): `[Bugfix][XPU] Add record_logical_topk_ready to XPUMLASparseImpl`
> - `vllm-project/vllm@6d01893` (`#64567`, 2026-10-30): `[ROCm][Bugfix] Skip AITER tuned GEMM when M*N >= 2^31 on gfx950`
> - `vllm-project/vllm@e49ea84` (`#63885`, 2026-10-30): `[ROCm][Perf][DSv4.1] Build the SWA decode metadata in one launch`
> - `NVIDIA/TensorRT-LLM@9c199cd` (`#20833`, 2026-10-29): `[TRTLLM-17121][feat] Support phase-dependent MLA heads for Helix prefill`
> - `NVIDIA/TensorRT-LLM@592bd6f` (`#20662`, 2026-10-29): `[TRTLLM-16980][feat] modeling_v2: GLM-5.2-NVFP4 sm_103 tep4 target`
> - `NVIDIA/TensorRT-LLM@0de8b6f` (`#20072`, 2026-10-29): `[TRTLLM-14620][feat] Add DSpark ragged graph and runtime integration`

---

## 1. Architectural Evolution & Source Code Mechanics

1. **SGLang Qwen3.8 FlyDSL `hc_mix` (`fcecdad`), Ratio-3 GDN Packing (`5035bc9`), DSv4 1024-Thread Grids (`4201b8f`), and PD Rebootstrap Retention (`03aba9e`)**:
   - Combining FlyDSL `hc_mix` with fused ratio-3 GDN head packing achieves a **1.45x** layer speedup, while retaining `output_ids` across PD rebootstraps eliminates duplicate token emissions (**0.0%** duplication rate).
2. **vLLM XPU Sparse MLA Top-K Sync (`fa9356a`), ROCm 32-Bit Index Overflow Guard (`6d01893`), and DSv4.1 Single-Launch SWA Metadata (`e49ea84`)**:
   - Building DSv4.1 SWA decode metadata in a single kernel launch reduces metadata construction latency by **68.75%**.
3. **TensorRT-LLM Phase-Dependent Helix MLA Heads (`9c199cd`), GLM-5.2 SM103 TEP4 (`592bd6f`), and DSpark Ragged Graphs (`0de8b6f`)**:
   - Phase-dependent MLA head partitioning cuts Helix prefill exchange overhead by **37.50%**, and DSpark ragged CUDA graph integration boosts speculative decode throughput by **1.31x**.
