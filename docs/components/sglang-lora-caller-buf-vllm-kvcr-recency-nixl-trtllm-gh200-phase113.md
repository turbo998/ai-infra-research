# SGLang 扩散模型 LoRA 缩放与可中断 CUDA Graph 调用方缓冲、vLLM KVCR 前缀近期性对齐与 Rust 语法/NIXL 统计解码及 TensorRT-LLM GH200 / DeepSeek-V3.2 分析 (Phase 113 · Priority 122)

> **English Edition**: [sglang-lora-caller-buf-vllm-kvcr-recency-nixl-trtllm-gh200-phase113.en.md](sglang-lora-caller-buf-vllm-kvcr-recency-nixl-trtllm-gh200-phase113.en.md)  
> **问题 ID**: `sglang-lora-caller-buf-vllm-kvcr-recency-nixl-trtllm-gh200-phase113` · **优先级**: `122` · **赛道**: Track B (推理服务与新型芯片)  
> **确定性实验**: [`experiments/sglang_lora_caller_buf_vllm_kvcr_recency_nixl_trtllm_phase113/run.py`](../../experiments/sglang_lora_caller_buf_vllm_kvcr_recency_nixl_trtllm_phase113/run.py) (`--check` 通过)

---

## 1. 第一手源码证据矩阵（First-Hand Source Matrix）

| 序号 | 来源 ID | 仓库与提交 | 发布日期 | 证据层级 | 核心变更与代码路径 |
|---|---|---|---|---|---|
| 1 | `sglang-diffusion-lora-format-alpha-rank-scaling-2026` | [`sgl-project/sglang@dc41126cde`](https://github.com/sgl-project/sglang/commit/dc41126cdea3fd7f461b5fa41bf5d15fb9ff99a2) | `2026-10-27` | `implementation` | `lora/format_adapter.py`, `peft_adapter.py`：在 AI-Toolkit Flux LoRA 转换中新增 `move_alpha` 将 `linear1/2.alpha` 同步迁移到拆分后的 `to_qkv_mlp_proj`/`to_out`，并使用 `removesuffix(".weight").removesuffix(".lora_A")` 避免漏匹配无 `.weight` 后缀的权重键 |
| 2 | `sglang-diffusion-caller-allocated-bcg-break-buffers-2026` | [`sgl-project/sglang@a912002ee2`](https://github.com/sgl-project/sglang/commit/a912002ee288f8ee6c65656ca514dd1a4f78898e) | `2026-10-27` | `implementation` | `models/dits/minimax_h3.py`, `minimax_h3_vdn_attention.py`：重构 MiniMax-H3 与 Qwen-Image 2.1 的 `@eager_on_graph` 中断点，将输出直接写入调用方预分配缓冲区并强制校验 `out.shape == result.shape and out.dtype == result.dtype` |
| 3 | `sglang-multimodal-clip-pad-spans-truncated-prompt-2026` | [`sgl-project/sglang@6724403444`](https://github.com/sgl-project/sglang/commit/6724403444fb6dd54e4198a3813d499bf8a6e23c) | `2026-10-27` | `implementation` | `srt/managers/mm_utils.py`：在 `pad_mm_tokens` 中以 `end = min(end + 1, len(padded_input_ids))` 裁剪多模态填充区间，防止提示词自动截断切穿图像 span 时触发越界写入 |
| 4 | `sglang-amd-monokernels-register-specs-by-model-type-2026` | [`sgl-project/sglang@5044779f0f`](https://github.com/sgl-project/sglang/commit/5044779f0f4efdb361c3923eb22e12cb84e23966) | `2026-10-27` | `implementation` | `kernels/plugin_kernels/amd/spec.py`：将 AMD 单体内核注册表键升级为二元组 `(architecture, model_type)`，区分共享同一 HF 架构类的不同模型变体 |
| 5 | `vllm-kv-offload-align-kvcr-recency-request-prefixes-2026` | [`vllm-project/vllm@e3883b7a7e`](https://github.com/vllm-project/vllm/commit/e3883b7a7e676cc37b9d4ab2d94e61f61b48e878) | `2026-10-27` | `implementation` | `vllm/v1/kv_offload/tiering/kvcr/manager.py`：通过 `order_request_keys` 与 `align_sequence` 将 KVCR 分层缓存的 LRU 近期性顺序与请求前缀顺序对齐，防止异步乱序加载导致根前缀块被优先淘汰 |
| 6 | `vllm-cpu-attn-encoder-only-bidirectional-fix-2026` | [`vllm-project/vllm@a23d7dc3f1`](https://github.com/vllm-project/vllm/commit/a23d7dc3f15b91e06fc888a4bb70ca57a7e1586c) | `2026-10-27` | `implementation` | `vllm/v1/attention/backends/cpu_attn.py`：修复 `CPU_ATTN` 后端将 `AttentionType.ENCODER_ONLY` 误施加因果掩码的问题，恢复双向全注意力 |
| 7 | `vllm-rust-frontend-decode-current-nixl-connector-stats-2026` | [`vllm-project/vllm@c56cdbea15`](https://github.com/vllm-project/vllm/commit/c56cdbea151dc4289f55b93a8496c15f11085fd9) | `2026-10-27` | `implementation` | `rust/src/engine-core-client/src/protocol/stats.rs`, `metrics.rs`：移除 `NixlStats` 的 `#[serde(deny_unknown_fields)]` 并新增 `num_failed_handshakes` 与 `num_notifications_after_expiry` 指标解码 |
| 8 | `vllm-rust-frontend-detect-lark-grammars-like-python-2026` | [`vllm-project/vllm@405276b8f0`](https://github.com/vllm-project/vllm/commit/405276b8f07fc2e258d7f5551fd8680e79b21899) | `2026-10-27` | `implementation` | `rust/src/chat/src/output/structural_tag.rs`：移植 Python 的 `grammar_is_likely_lark`，忽略 `#` 与 `//` 注释中的 `::=`，修复带注释 Lark 语法被误判为 EBNF 的问题 |
| 9 | `trtllm-restore-default-model-storage-gh200-2026` | [`NVIDIA/TensorRT-LLM@e805aac356`](https://github.com/NVIDIA/TensorRT-LLM/commit/e805aac356643a21dc7644f08c475f50dfcf7e79) | `2026-10-27` | `implementation` | `jenkins/L0_Test.groovy`：移除 GH200 上的 Austin scratch NFS 覆盖分支，恢复 GH200 默认模型存储卷挂载 |
| 10 | `trtllm-deepseek-v32-encoder-restore-tools-preamble-blank-lines-2026` | [`NVIDIA/TensorRT-LLM@c932dd8abb`](https://github.com/NVIDIA/TensorRT-LLM/commit/c932dd8abb3f15b91a90ab9185caef3f8d72fd1c) | `2026-10-27` | `implementation` | `tensorrt_llm/tokenizer/deepseek_v32/encoding.py`：恢复 `TOOLS_SYSTEM_TEMPLATE` 中 `<function_calls>` 示例前后的 4 处结构性空行，保证工具调用 token 编码精确对齐 |

---

## 2. 核心实现机理与架构量化分析（Implementation Mechanics & Quantitative Analysis）

### 2.1 vLLM KVCR 分层缓存前缀近期性对齐（`e3883b7a7e`）
- **故障机理**：当长上下文请求从远端/主机层异步回填前缀块 `[B0, B1, B2, B3]` 时，底层 I/O 完成顺序可能与逻辑前缀位置相反（例如 `B0` 最先完成写入，随后 `B3, B2, B1` 完成）。若按物理完成时间维护 LRU 链表，根前缀块 `B0` 会成为最旧条目并在容量淘汰时第一个被驱逐。由于前缀缓存命中要求从位置 0 开始连续命中，一旦 `B0` 丢失，后续保留的 `B1..B3` 将全部失效（连续前缀命中块数降至 **`0`**）。
- **修复**：引入 `align_sequence(order_request_keys(keys))`，按请求前缀拓扑顺序校准 LRU 近期性，使叶节点 `B3` 优先被淘汰，根前缀 `B0..B2` 保持 **`3` 个连续命中块**。

### 2.2 SGLang LoRA 缩放传播与可中断 CUDA Graph 零分配写回（`dc41126cde` & `a912002ee2`）
- **LoRA `alpha/rank` 缩放**：当 `rank=16, alpha=8.0` 时，若 `linear1.alpha` 在拆分到 `to_qkv_mlp_proj` 时丢失或因键名不含 `.weight` 未被 `apply_peft_config` 识别，缩放系数会退化为 `1.0` 而非 `8/16 = 0.5`，导致增量输出翻倍（`4.0` vs 正确值 **`2.0`**）。
- **可中断 CUDA Graph (`@eager_on_graph`)**：将 `_minimax_h3_attention_core_bcg` 改为接收调用方预分配的 `out` 张量并校验精确 `shape/dtype`，使每次图中断点的动态张量分配数从 `2` 降至 **`0`**。

---

## 3. 确定性实验结果验证（Deterministic Experiment Results · `own-offline`）

运行 `python experiments/sglang_lora_caller_buf_vllm_kvcr_recency_nixl_trtllm_phase113/run.py --check` 的密封快照：

| 指标 | 修复前 / 基线 | 修复后 / Phase 113 | 变化率 / 结论 |
|---|---|---|---|
| SGLang Flux LoRA 增量输出 (`rank=16, alpha=8, x=4`) | `4.0` (丢失 `alpha` 退化为 `1.0x`) | **`2.0` (`alpha/rank = 0.5x`)** | `move_alpha` + `.lora_A` 后缀归一 |
| SGLang 截断提示词多模态填充 (`len=12`, span `[8, 15]`) | 写至 `16` 越界崩溃 | **裁剪至 `[8, 12]`（填充 `4` tokens）** | `end = min(end + 1, len)` |
| SGLang 可中断 CUDA Graph 每次中断动态张量分配 | `2` | **`0`** | 写入调用方缓冲区并严查 `shape/dtype` |
| vLLM KVCR 淘汰 1 块后下一轮连续前缀命中数 (`4` 块前缀) | `0` 块 (`B0` 根块被误删) | **`3` 块 (`B3` 叶块优先淘汰)** | `align_sequence` 对齐前缀近期性 |
| vLLM Rust 带 `# ::= ` 注释的 Lark 语法判定 | `False` (误判为 EBNF) | **`True` (`grammar_is_likely_lark`)** | 剥离 `#` 与 `//` 注释后再搜 `::=` |
| TRT-LLM DeepSeek-V3.2 工具提示模板结构空行 | 缺失 4 行导致 token 偏移 | **恢复 `4` 处结构空行 + GH200 默认存储** | 保障工具调用渲染与 GH200 挂载一致 |
