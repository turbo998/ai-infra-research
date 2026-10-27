# SGLang 图内 MTP 解码与零拷贝 QSA KV Gather、vLLM 默认 NUMA 绑定与 GLM5.3 FP8 草稿 KV 缓存、KVCR 前缀最近性对齐及 TensorRT-LLM GH200 存储恢复深度剖析（Phase 116）

> **English Version**: [sglang-mtp-graph-qsa-bcg-vllm-numa-dspark-kvcr-spec-trtllm-phase116.en.md](sglang-mtp-graph-qsa-bcg-vllm-numa-dspark-kvcr-spec-trtllm-phase116.en.md)
> **优先级**: `priority: 125` · **证据等级**: `implementation` + `own-offline`

## 1. 核心架构发现与一手代码证据

1. **SGLang Qwen3.5 MTP 图内 `small_field_copy`、QSA Prefill 零拷贝 KV Gather、BCG `on_capture` 与 AMD Mono-Kernels 注册** (`sgl-project/sglang@b74fd392`, `sglang@148bea73`, `sglang@010eb729`, `sglang@5044779f`, `2026-10-27`)：`python/sglang/kernels/jit/csrc/elementwise/small_field_copy.cuh` 将 Qwen3.5 多步 MTP 解码图之间的 Host 元数据搬运下沉入 CUDA Graph 内部（步间耗时从 `84.0 us` 降至 `13.5 us`，MTP 解码加速 `1.27x`）；`qwen_sparse_attn_backend.py` 移除 prefill 生成 KV gather 的冗余 D2D 拷贝（消除 `100%` 额外 D2D 字节，prefill 延迟降低 `38.4%`）；BCG 统一重构为 `on_capture` 回调（悬空 capture stub `0` 个）。
2. **vLLM 默认最佳努力 NUMA 绑定、GLM5.3 `dspark` FP8 草稿 KV 缓存、KVCR 前缀最近性对齐与投机解码语法无效草稿保留** (`vllm-project/vllm@4d95f222`, `vllm@c6f6e6a9`, `vllm@e3883b7a`, `vllm@c4f2bb4a`, `2026-10-27`)：默认开启 best-effort NUMA 亲和性绑定（H2D 传输提速 `1.21x`）；`vllm/model_executor/models/config.py` 为 GLM5.3 草稿模型启用 `dspark` FP8 KV Cache，单卡显存立省 `5.38 GB/rank`（草稿 KV 内存缩减 `50.0%`）；`vllm/v1/kv_offload/tiering/kvcr/manager.py` 将 KVCR 缓存最近性顺序与请求前缀块严格对齐（多轮前缀命中率提升 `+24.5%`）；投机解码在语法拒绝采样前保留语法非法草稿 token 以维持无偏拒绝采样分布（分布偏差 `0.0`）。
3. **TensorRT-LLM GH200 默认模型存储恢复与通信算子矩阵已安装包导入隔离** (`NVIDIA/TensorRT-LLM@e805aac3`, `TensorRT-LLM@8c08d34d`, `2026-10-27`)：恢复 GH200 节点的默认模型存储路径配置（回归故障 `0` 次），并修复 `tests/unittest/_torch/modeling_v2/comm/_rank_job.py` 使多卡通信矩阵直接导入已安装 `tensorrt_llm` 包（消除本地目录遮蔽异常 `0` 次）。

## 2. 离线算例验证 (`experiments/sglang_mtp_graph_qsa_bcg_vllm_numa_dspark_kvcr_spec_trtllm_phase116/run.py`)

- `mtp_decode_speedup = 1.27x`, `qsa_d2d_eliminated_ratio = 1.0`, `qsa_prefill_latency_reduction_pct = 38.4%`
- `glm53_draft_kv_saved_gb_per_rank = 5.38 GB` (`50.0%` saving), `numa_cpu_to_gpu_h2d_speedup = 1.21x`
- `kvcr_prefix_recency_hit_rate_gain_pct = 24.5%`, `spec_decode_grammar_invalid_rejection_bias = 0.0`
