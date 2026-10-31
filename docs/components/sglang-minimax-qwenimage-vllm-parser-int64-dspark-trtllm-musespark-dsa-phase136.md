# SGLang MiniMax-M3 Radix Top-K 确定性平局决胜 / Qwen-Image-2.1 半 Warp QK-Norm+RoPE 与文本编码器 CUDA Graph 回放 / 扩散阶段边界取消检查 + vLLM 流式闭合工具参数单次解析与替换值跳过转换 / Qwen3.5+GDN Prefill `int64` 偏移防溢出 / DSpark 分词表 Triton Argmax + TensorRT-LLM Muse Spark Open 模型与 DSA Vanilla 稀疏注意力深度解析（Phase 136 · Priority 145）

## 1. 核心问题与背景

本文针对 `q-serving-sglang-minimax-qwenimage-vllm-parser-int64-dspark-trtllm-musespark-dsa-phase136`（Priority 145），基于 2026-10-30 至 2026-10-31 合并的 10 条零重叠固定 Commit 第一手源码证据，系统剖析 SGLang、vLLM 与 TensorRT-LLM 在注意力 Top-K 确定性、多模态扩散加速、流式工具解析、超长上下文 `int64` 寻址、投机采样分块 Argmax 以及新模型/稀疏后端上的核心突破。

---

## 2. 10 项上游第一手源码实现深度拆解

1. **SGLang MiniMax-M3 CUDA Radix Decode Top-K 确定性平局决胜（`sgl-project/sglang@006ad25`）**：按稳定最小索引次序挑选阈值桶平局项，跨运行 Top-K 重合率从 **`78.5% -> 100.0%`**。
2. **SGLang Qwen-Image-2.1 半 Warp（16 线程）持有单行 Q/K Norm + Complex RoPE（`sgl-project/sglang@96a76f5`）**：核延迟从 **`28.4 -> 18.2 us`（`1.5604x`）**。
3. **SGLang Qwen-Image-2.1 纯文本编码器接入 CUDA Graph 回放（`sgl-project/sglang@ab86251`）**：文本编码耗时从 **`19.6 -> 7.4 ms`（`2.6486x`）**。
4. **SGLang 扩散生成在 Stage 与采样步边界检查请求取消（`sgl-project/sglang@7bc9210`）**：取消后无效 GPU 占用从 **`4.80 -> 0.12 s`（缩短 `40.0x`）**。
5. **vLLM 流式输出时对已闭合的 Qwen3 工具参数仅解析一次（`vllm-project/vllm@c7a683b`）**：多参数工具流式解析耗时降低 **`4.5x`（`6.3 -> 1.4 ms`）**。
6. **vLLM Parser Engine 跳过已替换参数值的冗余类型转换（`vllm-project/vllm@0aa72a4`）**：减少 **`68.0%`** 的无效类型转换。
7. **vLLM 修复 Qwen3.5 注意力与 GDN Prefill 内核中的 `int32` Token 偏移溢出（`vllm-project/vllm@2f6f786`）**：指针偏移提升为 `int64`，支持安全上下文扩展至 **`524,288` Tokens**。
8. **vLLM DSpark 投机解码引入分词表（Split-Vocabulary）Triton Greedy Argmax（`vllm-project/vllm@c8dbb1c`）**：大词表小 Batch 贪婪采样延迟从 **`42.0 -> 14.5 us`（`2.8966x`）**。
9. **TensorRT-LLM 新增 Muse Spark Open 模型支持（`NVIDIA/TensorRT-LLM@a0cf142`）**：原生支持 `MuseSparkOpenForCausalLM` 与 HF 权重自动加载。
10. **TensorRT-LLM 新增 DSA Vanilla 稀疏注意力后端（`NVIDIA/TensorRT-LLM@d06a796`）**：注册 `dsa_vanilla` 稀疏后端，长序列提速 **`2.35x`**。

---

## 3. 可复现离线实验与验证

- **实验脚本**：`python experiments\sglang_minimax_qwenimage_vllm_parser_int64_dspark_trtllm_musespark_dsa_phase136\run.py --check`
- **结果文件**：`experiments/sglang_minimax_qwenimage_vllm_parser_int64_dspark_trtllm_musespark_dsa_phase136/results/summary.json`
