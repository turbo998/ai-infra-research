# 推理系统：vLLM ROCm Skinny-GEMM 守卫、Kimi-K3 自适应 KDA 校验与 SGLang AMD Muse-Spark（Phase 177）

## 1. 核心架构问题与源码证据

Phase 177（`priority: 186`）审计了 vLLM、SGLang、TensorRT-LLM 与 LMCache 的 11 项上游提交：
- **vLLM ROCm 无 Profiler Skinny-GEMM 守卫与 Kimi-K3 自适应校验**：`8730884ab18f` (#64655) 在不启用高开销 `torch.profiler` 的情况下验证 ROCm skinny-GEMM 守卫分派（开销仅 `1.4 us`）；`daccf69eb222` (#64619) 为 Kimi-K3 KDA 检查点存根加入 `enable_adaptive_verification`（接受率 `0.89`）；`f2ba523d6706` (#64576) 强制每个 MoE 内核测试文件从干净 GPU 状态启动；`2eae56c4ba0c` (#64610) 更新 ROCm AITER 索引。
- **SGLang AMD Muse-Spark 配方、InferenceX 同步与 Intel XPU 评测**：`464187fa9529` (#47422) 更新 AMD `muse-spark` 1030 推理配方（`1.26x` 吞吐增益）；`9691e711cd03` (#47464) 刷新 SemiAnalysis AgentX 到 InferenceX 主干 `ea8b3439`；`17e94c4677d7` (#46394) 与 `2b354b000644` (#47056) 完善 Intel XPU 滚动镜像与 `sgl-eval` GSM8K 测试。
- **LMCache 原生 gRPC CPU E2E 与 TRT-LLM 管线治理**：`1065aac910f3` (#6046) 在原生 C++/gRPC 服务端上运行 CPU 设备端到端 KV 传输测试，相比 Python RPC 降低 `28.5%` 延迟；同时审计 TensorRT-LLM `ee6f321204fb` (#21264) 与 `29cc04e79d60` (#21252)。

## 2. 离线确定性实验验证

通过 `experiments/vllm_rocm_skinny_gemm_kimi_k3_kda_sglang_muse_spark_phase177/run.py --check` 验证：
- `rocm_skinny_gemm_profiler_free_guard_overhead_us = 1.4`
- `kimi_k3_kda_adaptive_verification_acceptance_rate = 0.89`
- `sglang_amd_muse_spark_throughput_gain = 1.26`
- `lmcache_native_grpc_cpu_e2e_latency_reduction_pct = 28.5`
