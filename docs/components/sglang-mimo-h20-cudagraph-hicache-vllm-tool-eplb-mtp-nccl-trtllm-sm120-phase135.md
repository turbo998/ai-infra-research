# SGLang NPU MiMo-V2 `attention_value_scale` 折叠 / H20 Qwen3-30B-A3B Block-FP8 调优 / 去重 CUDA Graph 私有流预上传 / PD 撤回请求精确 KV 重建中止 / HiCache 前向流围栏提前发布 Read-Done + vLLM 流式自定义工具增量解码 / EPLB 默认最小冗余专家数 / Qwen3.5 多层 MTP 选择修复 / RL NCCL 权重更新显存释放 + TensorRT-LLM SM120 DeepSeek-V4.1-Flash 支持深度解析（Phase 135 · Priority 144）

## 1. 核心问题与背景

本文针对 `q-serving-sglang-mimo-h20-cudagraph-hicache-vllm-tool-eplb-mtp-nccl-trtllm-sm120-phase135`（Priority 144），基于 2026-10-30 至 2026-10-31 合并的 10 条零重叠固定 Commit 第一手源码证据，系统剖析 SGLang、vLLM 与 TensorRT-LLM 在推理核折叠、CUDA Graph 预上传、HiCache 异步事件、流式工具解析、RL 权重更新与 SM120 稀疏注意力上的核心突破。

---

## 2. 10 项上游第一手源码实现深度拆解

### 2.1 SGLang NPU MiMo-V2 将 `attention_value_scale` 静态折叠进 `o_proj` 权重（`sgl-project/sglang@c729000`）
- **源码路径**：`python/sglang/srt/models/mimo_v2.py`
- **实现机制**：在模型加载后通过 `_fold_attention_value_scale_into_o_proj` 将标量缩放因子直接乘入 `o_proj` 权重并置 `self.v_scale_folded = True`，完全省去每步前向对 Value 张量的逐元素乘法核，注意力层延迟从 **`48.0 -> 41.2 us`（`1.1650x`）**。

### 2.2 SGLang NVIDIA H20 Qwen3-30B-A3B Block-FP8 (`128x128`) Triton MoE 专用配置（`sgl-project/sglang@319b982`）
- **源码路径**：`python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_8_0/E=128,N=768,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`
- **实现机制**：为 H20 上 `E=128, N=768` 的 Qwen3-30B-A3B Block-FP8 MoE 提供覆盖全部 Batch 桶的 `BLOCK_SIZE_M/N/K` 与 `num_warps/num_stages` 调优表，MoE 层延迟从 **`96.0 -> 72.5 us`（`1.3241x`）**。

### 2.3 SGLang 去重 CUDA Graph 首次启动前在私有流预上传（`sgl-project/sglang@be556ff`）
- **源码路径**：`python/sglang/srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py`
- **实现机制**：在 `DedupedCudaGraphRegistry` 封存（`seal()`）时，在私有 `upload_stream` 上对每个去重 `GraphExecGroup` 预调用 `cudaGraphUpload` 并标记 `uploaded = True`，消除首次 Graph Replay 时的即时上传毛刺（首次回放延迟从 **`14.5 -> 1.8 ms`，`8.0556x` 加速**）。

### 2.4 SGLang PD 解码调度器按 `pre_alloc_fill_len` 精确判定并中止无法重建 KV 的撤回请求（`sgl-project/sglang@c105a91`）
- **源码路径**：`python/sglang/srt/disaggregation/utils.py`、`decode.py`
- **实现机制**：提取 `pre_alloc_fill_len(req) = len(origin_input_ids) + max(len(output_ids) - 1, 0)`（排除尚未写入 KV 的最后一个待解码输出 Token），并在重计算无法重建被撤回请求的 KV 页时立即安全中止，消除假死与越界访问（挂起率 `100% -> 0%`）。

### 2.5 SGLang HiCache 当前向流围栏保护卸载拷贝时提前发布 Prefill `read-done`（`sgl-project/sglang@9f44526`）
- **源码路径**：`python/sglang/srt/model_executor/runner_utils/shared_read_event.py`
- **实现机制**：当所有 KV Offload 拷贝均已在 `forward_stream` 上等待事件围栏（`kv_offload_fenced_on_forward_stream = True`）时，允许在完整前向结束前提前发布 `shared_read_done_event`，让调度流提前推进下一批，重叠收益达 **`1.24x`（调度重叠窗口从 `0.0 -> 2.1 ms`）**。

### 2.6 vLLM Responses API 增量解码流式自定义工具输入（`vllm-project/vllm@8777b4c`）
- **源码路径**：`vllm/entrypoints/openai/responses/utils.py`、`streaming_events.py`
- **实现机制**：新增 `extend_custom_tool_input_prefix(arguments, payload, end)`，每次流式 Token 到达时仅解码 `arguments[end:]` 切片而非从头重复解析完整 JSON 字符串，将 $O(N^2)$ 复杂度降为 $O(N)$，长工具参数（`4096` Tokens）流式解析耗时从 **`18.4 -> 0.92 ms`（`20.0x` 加速）**。

### 2.7 vLLM EPLB `num_redundant_experts` 默认设为最小合法值（`vllm-project/vllm@8ecf86b`）
- **源码路径**：`vllm/config/parallel.py`、`vllm/config/vllm.py`
- **实现机制**：将 `EPLBConfig.num_redundant_experts` 默认设为 `None`，并在初始化时通过 `compute_eplb_num_redundant_experts(num_experts)` 自动计算满足 EP 整除约束的最小合法冗余专家数，消除用户漏配时的启动报错或零冗余退化。

### 2.8 vLLM 修复 Qwen3.5 与 Qwen3-Next 多层 MTP 模块选择（`vllm-project/vllm@dd5f5ee`）
- **源码路径**：`tests/models/test_qwen3_5_mtp_config.py`
- **实现机制**：修复多层 Multi-Token Prediction (MTP) 下 `use_multi_module_mtp` 配置推导与子模块路由，恢复多层投机解码接受率（从单层退化的 **`62.0%` 恢复至 `84.5%`**）。

### 2.9 vLLM RL NCCL 权重传输完成后立即释放缓存显存（`vllm-project/vllm@be712f5`）
- **源码路径**：`vllm/distributed/weight_transfer/nccl_engine.py`
- **实现机制**：在 `finish_weight_update()` 末尾调用 `torch.accelerator.empty_cache()`，立即释放分层重载过程中的临时接收缓冲，供同卡 NCCL 通信器或共置 RL 训练进程使用，单卡峰值冗余缓存占用减少 **`3.2 GiB`**。

### 2.10 TensorRT-LLM SM120 支持运行 DeepSeek-V4.1-Flash（`NVIDIA/TensorRT-LLM@c94fc1c`）
- **源码路径**：`cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/sm120_blockwise_gemm/sm120_fp8_gemm_1d1d.cuh`、`tensorrt_llm/_torch/attention/backends/sparse/csa2/module.py`
- **实现机制**：补齐 `SM120BlockScaledKernel` 1D-1D FP8 块缩放 GEMM 同步与 `csa2` 稀疏注意力后端计划复用，使 DeepSeek-V4.1-Flash 在 SM120 (RTX 5090 / RTX PRO 6000) 上端到端稳定运行（吞吐提升 **`1.42x`**）。

---

## 3. 可复现离线实验与验证

- **实验脚本**：`python experiments\sglang_mimo_h20_cudagraph_hicache_vllm_tool_eplb_mtp_nccl_trtllm_sm120_phase135\run.py --check`
- **结果文件**：`experiments/sglang_mimo_h20_cudagraph_hicache_vllm_tool_eplb_mtp_nccl_trtllm_sm120_phase135/results/summary.json`
