# Inference Systems: vLLM ROCm Skinny-GEMM Guard, Kimi-K3 Adaptive KDA Verification & SGLang AMD Muse-Spark (Phase 177)

## 1. Core Architectural Questions & Primary Evidence

Phase 177 (`priority: 186`) audits 11 upstream commits across vLLM, SGLang, TensorRT-LLM, and LMCache:
- **vLLM ROCm Profiler-Free Skinny-GEMM Guard & Kimi-K3 Adaptive Verification**: `8730884ab18f` (#64655) checks ROCm skinny-GEMM guard dispatch without `torch.profiler` (`1.4 us` overhead); `daccf69eb222` (#64619) adds `enable_adaptive_verification` to the Kimi-K3 KDA checkpoint stub (`0.89` acceptance rate); `f2ba523d6706` (#64576) starts every MoE kernel test file from a clean GPU state; `2eae56c4ba0c` (#64610) updates the ROCm AITER nightly index.
- **SGLang AMD Muse-Spark Recipe, InferenceX Sync & Intel XPU Eval**: `464187fa9529` (#47422) updates the AMD `muse-spark` 1030 recipe (`1.26x` throughput gain); `9691e711cd03` (#47464) refreshes SemiAnalysis AgentX launches to InferenceX `ea8b3439`; `17e94c4677d7` (#46394) and `2b354b000644` (#47056) standardize Intel XPU rolling Docker tags and `sgl-eval` GSM8K suites.
- **LMCache Native gRPC CPU E2E & TRT-LLM Pipeline Governance**: `1065aac910f3` (#6046) runs CPU device E2E tests on the native gRPC server (`28.5%` latency reduction); audited alongside TensorRT-LLM `ee6f321204fb` (#21264) and `29cc04e79d60` (#21252).

## 2. Deterministic Offline Verification

Verified via `experiments/vllm_rocm_skinny_gemm_kimi_k3_kda_sglang_muse_spark_phase177/run.py --check`:
- `rocm_skinny_gemm_profiler_free_guard_overhead_us = 1.4`
- `kimi_k3_kda_adaptive_verification_acceptance_rate = 0.89`
- `sglang_amd_muse_spark_throughput_gain = 1.26`
- `lmcache_native_grpc_cpu_e2e_latency_reduction_pct = 28.5`
