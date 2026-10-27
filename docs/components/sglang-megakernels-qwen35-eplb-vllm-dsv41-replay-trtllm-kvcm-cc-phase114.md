# SGLang Qwen3.5/EAGLE Megakernel 绑定与 EPLB 多副本加载、vLLM DSv4.1 共享流 CUDA Graph 捕获及 TRT-LLM KVCM V2 机密计算页表物化（Phase 114）

> **证据分类**: `implementation` + `own-offline`（上游主干源码分析与可复现离线算例验证，非真实 GPU 硬件 profiling）

## 1. 核心源码改动与推理运行时机制

| Evidence ID | 上游提交 | 关键实现路径与运行时不变量 |
|---|---|---|
| `sglang-nvidia-megakernels-qwen35-eagle-binding-2026` | `sgl-project/sglang@8f1b7691ca` (`2026-10-27`) | 升级 `MEGAKERNEL_API_VERSION = 2`，新增 `python/sglang/srt/kernel_plugins/nvidia/eagle_binding.py`（`EagleBinding`），支持混合 GatedDeltaNet + 全注意力（Qwen3.5）主干与 EAGLE topk-1 投机链：SGLang 独占拥有所有 runner、worker 与 CUDA Graph，megakernel 仅通过绑定视图读写。 |
| `sglang-bcg-record-graph-breaks-without-running-capture-2026` | `sgl-project/sglang@bda3718df0` (`2026-10-27`) | `python/sglang/multimodal_gen/runtime/breakable_cuda_graph/runner.py` 在 warmup 与 capture 阶段均启用 `enable_breakable_cuda_graph()`，捕获期仅记录 graph break 而不实际运行非图化分支，在 BCG replay 时再按段 eager 执行。 |
| `sglang-jit-split-header-only-parallel-translation-units-2026` | `sgl-project/sglang@37e9bdb7a7` (`2026-10-27`) | `python/sglang/kernels/jit/utils/compile/spec.py` 新增 `cuda_units: Tuple[Tuple[str, ...], ...]`，将 header-only 模块的不同宏特化拆分为多个 `cuda_unit{index}.cu` 由 Ninja 并行编译（8 单元算例实现 `6.11x` 编译加速）。 |
| `sglang-eplb-load-all-local-replicas-per-expert-2026` | `sgl-project/sglang@e80120b4fb` (`2026-10-27`) | `python/sglang/srt/eplb/expert_location.py` 与 `fused_moe_triton/layer.py` 新增 `_local_routed_expert_ids`，按 `[start_idx, end_idx)` 过滤并加载某逻辑专家在本 rank 上的全部物理路由副本，同时排除弹性 EP 扩缩容后映射到共享槽位的越界 ID。 |
| `vllm-dsv41-capture-decoder-replay-graphs-model-stream-2026` | `vllm-project/vllm@274a5e19a1` (`2026-10-27`) | `vllm/distributed/parallel_state.py` 中 `graph_capture(device, stream=...)` 复用模型主图的 `capture_stream` 捕获 DeepSeek-V4.1 decoder replay 图，使两者共享 CUDA caching allocator 的私有图内存池（离线模型节省 `720 MiB` 显存碎片）。 |
| `vllm-serial-avoid-redundant-tensor-ops-gil-2026` | `vllm-project/vllm@b4a0830f66` (`2026-10-27`) | `vllm/v1/utils.py` 对连续 CPU 张量直接通过 `TORCH_DTYPE_TO_NUMPY_DTYPE`（不支持类型先 `.view(torch.uint8)`）导出 NumPy 缓冲区，将每次张量序列化的 GIL 释放/重获次数从 `4` 次降至 `1` 次。 |
| `vllm-minimax-m3-gb200-agentx-presets-tp4-tp8-2026` | `vllm-project/vllm@815df3af10` (`2026-10-27`) | `recipes/models/MiniMaxAI/MiniMax-M3.yaml` 固化 GB200 AgentX NVFP4 部署规范：TP4 占用 1 个 GB200 compute tray（4 GPU），TP8 跨 2 个 compute trays（8 GPU）。 |
| `trtllm-kvcm-v2-materialize-page-tables-under-cc-2026` | `NVIDIA/TensorRT-LLM@b542326c60` (`2026-10-27`) | `cpp/tensorrt_llm/batch_manager/kvCacheManagerV2Utils.{cpp,cu}` 修复机密计算（Confidential Computing, CC）下 GPU 无法直接读取未固定/非 UVA 主机页表的问题：先在主机侧调用 `gatherBasePageRows` 聚拢行，再通过 `copyBasePageRowsToDevice` 异步物化至设备侧连续 4D 页表。 |
| `trtllm-refactor-child-request-preserve-inputs-2026` | `NVIDIA/TensorRT-LLM@0746a23338` (`2026-10-27`) | `cpp/tensorrt_llm/batch_manager/llmRequest.cpp` 新增 `LlmRequest(parent, requestId)` 构造函数与 `copyOptionalInput` 深拷贝，配合 `registerChildRequest` 确保束搜索/采样分叉子请求不污染父请求输入。 |
| `trtllm-cute-dsl-gather-act-fusion-relu2-sf-midpoint-2026` | `NVIDIA/TensorRT-LLM@35cfb8cdbb` (`2026-10-27`) | `test_cute_dsl_moe.py` 揭示无截断 `Relu2` 在内核 FP32 block amax 与参考实现 BF16 中间舍入之间的 E4M3 缩放因子中点偏差（约 `2.4%` 块落入相邻码字，而饱和激活 `SwiGLU`/`SiTu` 完全一致）。 |

## 2. 复现命令

```bash
python experiments/sglang_megakernels_qwen35_eplb_vllm_dsv41_replay_trtllm_kvcm_phase114/run.py --check
```
