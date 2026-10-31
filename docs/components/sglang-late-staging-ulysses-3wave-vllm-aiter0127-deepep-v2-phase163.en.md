# SGLang Disagg Late STAGING_REQ Guard & 3-Wave Ulysses Diffusion, vLLM ROCm AITER v0.1.27 & DeepEP V2 Single-Node Intra-EP (Phase 163 · Priority 172)

> **中文版本**: [`sglang-late-staging-ulysses-3wave-vllm-aiter0127-deepep-v2-phase163.md`](sglang-late-staging-ulysses-3wave-vllm-aiter0127-deepep-v2-phase163.md)

## 1. Primary Source Evidence & Classification

| Source ID | Repository & Commit | Release Date | Evidence Tier | Key Code Path & Architectural Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| `src-sglang-disagg-ignore-late-staging-req-2026` | `sgl-project/sglang` `#47652` (`e624174f07`) | 2026-10-31 | `implementation` | `python/sglang/srt/disaggregation/common/staging_handler.py`: Ignores late `STAGING_REQ` messages for already-scattered staging chunks, preventing duplicate scatter corruption |
| `src-sglang-diffusion-ulysses-three-waves-2026` | `sgl-project/sglang` `#47565` (`67bd749ada`) | 2026-10-31 | `implementation` | `python/sglang/multimodal_gen/runtime/distributed/device_communicators/ipc_a2a_multi.py`: Pipelines Ulysses attention over head groups down to 3 waves |
| `src-sglang-rocm-minimax-h3-vae-skip-dead-sdpa-2026` | `sgl-project/sglang` `#47570` (`6e011a7097`) | 2026-10-31 | `implementation` | `python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/attention.py`: Skips dead math-SDPA work in the MiniMax-H3 VAE decoder on ROCm `gfx950` |
| `src-sglang-diffusion-warmup-all-dp-replicas-2026` | `sgl-project/sglang` `#47571` (`5c46d20db9`) | 2026-10-31 | `implementation` | `python/sglang/multimodal_gen/runtime/server_warmup.py`: Broadcasts diffusion server warmup across every data-parallel (DP) replica |
| `src-vllm-rocm-ci-prebuild-mamba-conv1d-2026` | `vllm-project/vllm` `#64577` (`67d677bdef`) | 2026-10-30 | `implementation` | `docker/Dockerfile.rocm`: Prebuilds `mamba-ssm` and `causal-conv1d` into the ROCm CI test image |
| `src-vllm-rocm-bump-aiter-0127-2026` | `vllm-project/vllm` `#63774` (`7495e9e5a4`) | 2026-10-30 | `implementation` | `docker/Dockerfile.rocm_base`: Upgrades ROCm AITER to `v0.1.27` |
| `src-trtllm-perf-glm-single-node-refresh-2026` | `NVIDIA/TensorRT-LLM` `#20406` (`78145f88ea`) | 2026-10-30 | `implementation` | `tests/integration/defs/perf/pytorch_model_config.py`: Refreshes GLM single-node core performance coverage |
| `src-trtllm-disagg-ucx-loopback-perf-metrics-2026` | `NVIDIA/TensorRT-LLM` `#21236` (`aecb1f224c`) | 2026-10-30 | `implementation` | `tests/unittest/llmapi/apps/test_disagg_serving_perf_metrics.py`: Pins UCX to loopback in disaggregated serving perf-metrics tests (`nvbugs/6997729`) |
| `src-deepep-setup-resolve-nvshmem-nccl-so-2026` | `deepseek-ai/DeepEP` `#627` (`723716fc11`) | 2026-05-22 | `implementation` | `setup.py`: Resolves real NVSHMEM and NCCL shared library `.so` names for pip-wheel installations |
| `src-deepep-v2-pure-single-node-intra-ep-fix-2026` | `deepseek-ai/DeepEP` `#630` (`56169594eb`) | 2026-05-22 | `implementation` | `csrc/kernels/backend/nccl.cu`: Fixes V2 initialization for pure single-node intra-node Expert Parallelism (Intra-EP) |

## 2. Deterministic Verification (`experiments/sglang_late_staging_ulysses_3wave_vllm_aiter0127_deepep_v2_phase163`)

- **SGLang Late `STAGING_REQ` Guard & 3-Wave Ulysses**: Reduces duplicate staging scatters from **1** to **0**, cuts Ulysses head-group waves by **`50.0%`** (`6 -> 3`), and reaches **`100.0%`** DP replica warmup coverage.
- **vLLM ROCm AITER `v0.1.27` & DeepEP V2 Intra-EP**: Eliminates **`11.5 min`** of runtime `mamba-ssm`/`causal-conv1d` compilation (`11.5 -> 0.0 min`) and reduces DeepEP V2 single-node Intra-EP init failures and TRT-LLM UCX off-host leaks to **0**.
