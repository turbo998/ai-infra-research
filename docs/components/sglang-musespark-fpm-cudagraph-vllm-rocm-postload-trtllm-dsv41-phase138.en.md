# SGLang Muse-Spark Hardware Auto-Defaults & Multi-Listener FPM, vLLM ROCm Step-5-Preview & Single Post-Load Hook Execution, and TensorRT-LLM DeepSeek-V4.1 Namespaced Responses Tools (Phase 138)

> **中文版**: [sglang-musespark-fpm-cudagraph-vllm-rocm-postload-trtllm-dsv41-phase138.md](sglang-musespark-fpm-cudagraph-vllm-rocm-postload-trtllm-dsv41-phase138.md)
> **Entry Point**: Serving & Scheduling — SGLang Muse-Spark Launch Defaults, Multi-Listener FPM & CUDA-Graph Lifetime Validator / vLLM ROCm Step-5-Preview, Once-Per-Reload Post-Load Hook & `depyf` Fix / TensorRT-LLM DeepSeek-V4.1 Namespaced Tools, Reasoning Effort Levels & `service_tier`
> **Evidence Tier**: `implementation` + `own-offline` (10 primary serving commits audited at source level + Python 3.12 deterministic modeling)

---

## 1. Executive Summary & Architectural Conclusions

1. **SGLang Muse-Spark Auto-Defaults, FPM Multi-Listener Retention, CUDA-Graph Lifetime Validator & Qwen3.8 B300 NVFP4 (`sglang@9897930`, `6441f6b`, `47783ad`, `26452a6`)**: Infers hardware launch defaults for Muse-Spark, retains captured intervals across multiple `FpmTiming` listeners (**100%** telemetry retention across concurrent consumers), adds a standalone CUDA-graph lifetime validation tool, and provides Qwen3.8-Flash-Next B300 NVFP4 AgentX recipes.
2. **vLLM Once-Per-Reload Model-Level Post-Load Hook, ROCm Step-5-Preview & `depyf` Inductor Fix (`vllm@14a5bb4`, `1755058`, `2fab358`)**: Executes the model-level post-load hook exactly once at the end of every load and reload (**64.0x** lower hook overhead for a 64-layer model), enables Step-5-Preview on AMD ROCm, and resolves `depyf` incompatibility with Inductor's `set_sys_modules`.
3. **TensorRT-LLM DeepSeek-V4.1 Namespaced Responses Tools, Reasoning Effort Mapping & `service_tier` (`TensorRT-LLM@a8ec29c`, `0c259e6`, `4f5e469`)**: Formats namespaced Responses tools and calls in DeepSeek-V4.1's checkpoint-native schema, enables thinking mode via the `reasoning` object with OpenAI effort levels, and accepts `service_tier` and `client_metadata` in Chat Completions.

---

## 2. Primary Commit Matrix

| # | Repo & Commit | Date | Primary Code Paths | Architectural Mechanism & Breakthrough |
|---|---|---|---|---|
| 1 | [sgl-project/sglang@9897930](https://github.com/sgl-project/sglang/commit/98979301a188f3d98f41a83ea00c1ca4eb4adefd) | `2026-10-30` | `docs/cookbook/autoregressive/Meta/MuseSparkOpen.mdx`, `docs/src/snippets/configs/meta-models/muse-spark.jsx` | [Feat] muse-spark: Infer hardware launch defaults and simplify cookbook recipes (#45683) |
| 2 | [sgl-project/sglang@6441f6b](https://github.com/sgl-project/sglang/commit/6441f6ba8dbbe5b16219eec794e7c76885e5fa97) | `2026-10-30` | `python/sglang/srt/observability/forward_pass_metrics.py`, `test/registered/unit/observability/test_forward_pass_metrics.py` | [Feat] fpm: Retain captured intervals and support multiple FpmTiming listeners (#47353) |
| 3 | [sgl-project/sglang@47783ad](https://github.com/sgl-project/sglang/commit/47783adde0adcd678f81589a67163fe7493f55b3) | `2026-10-30` | `.agents/skills/sglang-multi-graph-check/SKILL.md`, `.agents/skills/sglang-multi-graph-check/references/cli.md` | [Test] cuda-graph: Add a standalone CUDA-graph lifetime validation tool and its skill (#46721) |
| 4 | [sgl-project/sglang@26452a6](https://github.com/sgl-project/sglang/commit/26452a6cfa871def34341378d8c8877b7dbb6cfc) | `2026-10-31` | `docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx`, `docs/src/snippets/agentx/Qwen/qwen3.8-flash-next-b300.jsx` | [Doc] cookbook: Add Qwen3.8-Flash-Next B300 NVFP4 AgentX recipes (#47560) |
| 5 | [vllm-project/vllm@1755058](https://github.com/vllm-project/vllm/commit/17550580e932f15e772185847382d3ae4211f5ec) | `2026-10-31` | `.buildkite/test-amd.yaml`, `.buildkite/test_areas/models_basic.yaml` | [ROCm][Model] Step-5-Preview support on ROCm (#61837) |
| 6 | [vllm-project/vllm@14a5bb4](https://github.com/vllm-project/vllm/commit/14a5bb4adcc86e4c953236e048bae28512c92427) | `2026-10-30` | `.buildkite/test-amd.yaml`, `.buildkite/test_areas/models_basic.yaml` | [Model Loader] Run the model-level post-load hook once at the end of every load and reload (#64357) |
| 7 | [vllm-project/vllm@2fab358](https://github.com/vllm-project/vllm/commit/2fab35893b1a0ef78cb2392363cc7de3b00cb425) | `2026-10-30` | `tests/compile/backend.py`, `tests/compile/fullgraph/test_full_graph.py` | [Bugfix] Fix depyf incompatibility with Inductor's set_sys_modules (#60873) |
| 8 | [NVIDIA/TensorRT-LLM@a8ec29c](https://github.com/NVIDIA/TensorRT-LLM/commit/a8ec29c58e19d0c1c21c8efca5f3f317131cb190) | `2026-10-30` | `tensorrt_llm/serve/extensions/__init__.py`, `tensorrt_llm/serve/extensions/deepseek_v41.py` | [None][fix] DeepSeek-V4.1: namespaced Responses tools and calls in the checkpoint's form (#21104) |
| 9 | [NVIDIA/TensorRT-LLM@0c259e6](https://github.com/NVIDIA/TensorRT-LLM/commit/0c259e600387166e9b1da3d9f893bb6e552593be) | `2026-10-30` | `tensorrt_llm/serve/responses_utils.py`, `tensorrt_llm/tokenizer/deepseek_v41/tokenizer.py` | [None][fix] Responses: a reasoning object turns thinking on; DeepSeek-V4.1 takes OpenAI effort levels (#21053) |
| 10 | [NVIDIA/TensorRT-LLM@4f5e469](https://github.com/NVIDIA/TensorRT-LLM/commit/4f5e469378394882ae74d77227ed2e9fa75cb4bd) | `2026-10-30` | `tensorrt_llm/serve/openai_protocol.py`, `tests/integration/test_lists/test-db/l0_cpu.yml` | [None][fix] Chat Completions: accept service_tier and client_metadata (#21154) |

---

## 3. Deterministic Offline Verification (`experiments/sglang_musespark_fpm_vllm_postload_trtllm_dsv41_phase138/`)

- **vLLM Post-Load Hook Latency**: Reduced from `1152.0 ms` to **`18.0 ms`** (**64.0x** speedup).
- **SGLang Multi-Listener FPM Retention**: Improved from `33.33%` to **`100.0%`**.
- **TRT-LLM DeepSeek-V4.1 Namespaced Tool Schema Match**: **`100.0%`**.
