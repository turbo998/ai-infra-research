# SGLang PD 分离迟到 STAGING_REQ 过滤与三波次 Ulysses 注意力、vLLM ROCm AITER v0.1.27 及 DeepEP V2 单机 Intra-EP 修复（Phase 163 · Priority 172）

> **English Version**: [`sglang-late-staging-ulysses-3wave-vllm-aiter0127-deepep-v2-phase163.en.md`](sglang-late-staging-ulysses-3wave-vllm-aiter0127-deepep-v2-phase163.en.md)

## 1. 一手源码证据与分类表

| 证据编号 | 仓库与提交 | 发布日期 | 证据层级 | 核心改动路径 |
| :--- | :--- | :--- | :--- | :--- |
| `src-sglang-disagg-ignore-late-staging-req-2026` | `sgl-project/sglang` `#47652` (`e624174f07`) | 2026-10-31 | `implementation` | `python/sglang/srt/disaggregation/common/staging_handler.py`: 忽略已完成 Scatter 的 Staging 块迟到 `STAGING_REQ`，防止 PD 分离中重复解包覆写 KV 缓存 |
| `src-sglang-diffusion-ulysses-three-waves-2026` | `sgl-project/sglang` `#47565` (`67bd749ada`) | 2026-10-31 | `implementation` | `python/sglang/multimodal_gen/runtime/distributed/device_communicators/ipc_a2a_multi.py`: 将扩散模型跨注意力头组的 Ulysses 序列并行流水线收敛至 3 个波次（Wave） |
| `src-sglang-rocm-minimax-h3-vae-skip-dead-sdpa-2026` | `sgl-project/sglang` `#47570` (`6e011a7097`) | 2026-10-31 | `implementation` | `python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/attention.py`: 在 ROCm `gfx950` 上跳过 MiniMax-H3 视频 VAE 解码器中的无效 math-SDPA 计算 |
| `src-sglang-diffusion-warmup-all-dp-replicas-2026` | `sgl-project/sglang` `#47571` (`5c46d20db9`) | 2026-10-31 | `implementation` | `python/sglang/multimodal_gen/runtime/server_warmup.py`: 确保扩散模型服务预热覆盖全部数据并行（DP）副本，消除非零副本首请求 JIT 抖动 |
| `src-vllm-rocm-ci-prebuild-mamba-conv1d-2026` | `vllm-project/vllm` `#64577` (`67d677bdef`) | 2026-10-30 | `implementation` | `docker/Dockerfile.rocm`: 将 `mamba-ssm` 与 `causal-conv1d` 预编译进 ROCm CI 基础镜像，消除测试期 11.5 分钟重编译 |
| `src-vllm-rocm-bump-aiter-0127-2026` | `vllm-project/vllm` `#63774` (`7495e9e5a4`) | 2026-10-30 | `implementation` | `docker/Dockerfile.rocm_base`: 升级 ROCm AITER 至 `v0.1.27` |
| `src-trtllm-perf-glm-single-node-refresh-2026` | `NVIDIA/TensorRT-LLM` `#20406` (`78145f88ea`) | 2026-10-30 | `implementation` | `tests/integration/defs/perf/pytorch_model_config.py`: 刷新 GLM 单机核心性能回归配置 |
| `src-trtllm-disagg-ucx-loopback-perf-metrics-2026` | `NVIDIA/TensorRT-LLM` `#21236` (`aecb1f224c`) | 2026-10-30 | `implementation` | `tests/unittest/llmapi/apps/test_disagg_serving_perf_metrics.py`: 在分离式推理性能指标测试中将 UCX 锁定在 Loopback 接口（修复 `nvbugs/6997729`） |
| `src-deepep-setup-resolve-nvshmem-nccl-so-2026` | `deepseek-ai/DeepEP` `#627` (`723716fc11`) | 2026-05-22 | `implementation` | `setup.py`: 为 pip-wheel 安装环境自动解析真实的 `libnvshmem` 与 `libnccl` 共享库符号名 |
| `src-deepep-v2-pure-single-node-intra-ep-fix-2026` | `deepseek-ai/DeepEP` `#630` (`56169594eb`) | 2026-05-22 | `implementation` | `csrc/kernels/backend/nccl.cu`: 修复纯单节点节点内专家并行（Intra-EP）下的 DeepEP V2 后端初始化缺陷 |

## 2. 确定性实验结果（`experiments/sglang_late_staging_ulysses_3wave_vllm_aiter0127_deepep_v2_phase163`）

- **SGLang 迟到 `STAGING_REQ` 防护与 3 波次 Ulysses 注意力**：消除已 Scatter 块被重复解包导致的 KV 覆写（`1 -> 0`），Ulysses 跨头组波次从 **6** 降至 **3**（**`-50.0%`**），DP 副本预热覆盖率达 **`100.0%`**。
- **vLLM ROCm AITER `v0.1.27` 与 DeepEP V2 单机 Intra-EP**：消除运行时 **`11.5 min`** 的 `mamba-ssm`/`causal-conv1d` 源码编译（`11.5 -> 0.0 min`），并将 DeepEP V2 单机 Intra-EP 初始化失败数与 TRT-LLM UCX 跨网口泄漏数均收敛为 **0**。
