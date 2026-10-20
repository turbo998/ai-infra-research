# TorchTitan Muse Spark 局部编译图预算 (`muse_spark_rmsnorm` / Head-Gate) 与 Megatron-LM `processed_logprobs` 提示词原分布保持

> **English Version**: [torchtitan-muse-spark-rmsnorm-head-gate-compile-megatron-processed-logprobs.en.md](torchtitan-muse-spark-rmsnorm-head-gate-compile-megatron-processed-logprobs.en.md)

## 1. 一手源码证据与核心定位

| 系统 / 仓库 | 精确 Commit / 日期 | 核心源码路径 | 证据等级 |
|---|---|---|---|
| **TorchTitan** (`pytorch/torchtitan`) | `e923bfc7fa6f78da7f1bbce48133457a9eb77b92` (`2026-10-20`, PR `#5434`) | `torchtitan/models/muse_spark/model.py`, `tests/unit_tests/gpu/models/muse_spark/test_attention_compile.py` | `implementation` + `author-benchmark` |
| **TorchTitan** (`pytorch/torchtitan`) | `08b4c0ebdfc9d2f733020abdb829c76c9d8e3cb9` (`2026-10-20`, PR `#5458`) | `torchtitan/models/muse_spark/model.py`, `tests/unit_tests/gpu/models/muse_spark/test_attention_compile.py` | `implementation` + `author-benchmark` |
| **Megatron-LM** (`NVIDIA/Megatron-LM`) | `4f86e2c1ab17b29a2fd54d64819a0c5fe437a5a3` (`2026-10-19`, PR `#8162`) | `megatron/core/inference/contexts/dynamic_context.py`, `tests/unit_tests/inference/contexts/test_dynamic_context.py` | `implementation` |

---

## 2. TorchTitan Muse Spark：Attention 逐头归一化、Sigmoid 门控与残差流 RMSNorm 局部编译 (`#5434`, `#5458`)

### 2.1 64-Wide 窄行 ATen `rms_norm` 的带宽利用率瓶颈 (`#5434`)
在 `MuseSparkAttention`（Muse Spark Open 816B-A42B）中，每个 Attention 层需要对查询和键张量执行逐头归一化（`q_TNH` 形状 `[T, 128, 64]`、`k_TKH` 形状 `[T, 16, 64]`）并乘上标量 `scale_query_by`，随后在注意力输出 `output_TNH` 上执行 `contiguous -> output_norm -> reshape([T, 8192]) -> * sigmoid(output_gate_TO)`。

在 GB300 上的实测表明（$T = 4096$）：
- Eager 模式下 ATen `rms_norm` 在宽度仅 64 的窄行上带宽利用率极低：单层前向 `q + k` 归一化耗时 **$403\ \mu\text{s}$**，而理想带宽耗时仅 **$\sim 21\ \mu\text{s}$**（仅达理想性能的 **$5.21\%$**）；
- 加上独立的缩放、展平与门控逐元素核，单层 $T=4096$ 正反向总胶水耗时高达 **q/k glue $794\ \mu\text{s}$**（理想 $42\ \mu\text{s}$）+ **output glue $814\ \mu\text{s}$**（理想 $66\ \mu\text{s}$），合计 **$1,608\ \mu\text{s}/\text{layer}$**（理想 $108\ \mu\text{s}$）。

`e923bfc7` 引入两个 `@local_compile(..., batch_invariant=False)` 编译域：
1. `muse_spark_attention_qk_norm`（融合 `q_norm(q) * scale_query_by` 与 `k_norm(k)`）；
2. `muse_spark_attention_output`（融合 `contiguous + output_norm + reshape + * torch.sigmoid(gate)`，将含 TP 规约的 `self.wo` 留在编译域外）。
不仅单算子相对 FP64 的 BF16 最大相对误差由于 FMA 收缩反而比 Eager 更低（输出门控反向从 $4.3\times 10^{-3}$ 降至 $2.3\times 10^{-3}$），而且使 2 层基准在训练上提速 **$-6.7\%\sim -7.2\%$**、预填充提速 **$-10.8\%\sim -11.5\%$**、Decode-256 提速 **$-7.3\%$**。

### 2.2 残差流 `muse_spark_rmsnorm` 的 12-Graph 重编译预算设计 (`#5458`)
`08b4c0eb` 在 `MuseSparkRMSNorm.forward` 上添加 `@local_compile("muse_spark_rmsnorm", batch_invariant=False, recompile_limit=16)`：
- **为什么必须将 `recompile_limit` 从默认 8 提升至 16？** 因为 `attention_norm`、`ffn_norm`、`post_attention_norm`、`pre_expert_norm` 与最终层归一化共享同一个 `MuseSparkRMSNorm.forward` 区域，共覆盖 **4 种调用形态**（带权重的 FP32 残差流、无权重的 BF16 流、4096 宽度的专家归一化、最终层归一化）$\times$ **梯度模式**（训练 `grad` / 推理 `no_grad`）$\times$ **动态 $T$ 形状特化** = **12 张编译图**。若沿用默认上限 8，将在第 9 张图触发重编译回退！
- 叠加 PR 1–4 后，GB300 上 2 层 Muse Spark 816B-A42B 的端到端耗时实现全面下降：4k 训练 CUDA Graphs 从 **$42.13\text{ ms} \to 33.17\text{ ms}$ ($-21.27\%$)**，16k 预填充从 **$59.31\text{ ms} \to 45.17\text{ ms}$ ($-23.84\%$)**，Decode-256 从 **$1.55\text{ ms} \to 1.20\text{ ms}$ ($-22.58\%$)**。

---

## 3. Megatron-LM：`processed_logprobs` 模式下保持 Prompt Log-Probs 原始分布 (`#8162`)

### 3.1 缺陷机理：提示词行被错误施加 Top-K / Top-P 掩码
在 Megatron-Core 动态推理上下文 (`DynamicInferenceContext.calculate_log_probs_tensors`) 中，当开启 `logprobs_mode='processed_logprobs'` 时，旧代码通过 `row_to_request = torch.arange(n_active).repeat_interleave(active_query_lengths)` 将每个请求的采样参数（`temperature`、`top_k`、`top_p`）应用到了**包括提示词（Prompt）在内的所有 Token 行**。
然而，提示词 Token 是用户输入而非引擎从过滤后分布采样生成的：对其施加 Top-K/Top-P 会将不在前 $K$ 名的真实提示词 Token 或 `top_n_logprobs` 变成 $-\infty$。

### 3.2 采样行精确切片与 Chunked Prefill 尾部豁免
`4f86e2c1` 重构了混合预填充/解码路径：
1. 首先对全部活动 Token 行计算原始对数概率：`log_probs = F.log_softmax(logits_squeezed, dim=-1)`；
2. 每个请求唯一由引擎采样的行位于累积长度末尾：`new_token_idx = active_query_lengths_gpu.cumsum(0) - 1`；
3. 若最后一个活动请求是**未完成的分块预填充 (partial chunked prefill)**（`chunked_idx == self.total_request_count - 1`），其最后一行评分的是**下一个提示词 Token** 而非采样 Token，因此将 `num_sampled = n_active - 1`，仅对 `sampled_idx = new_token_idx[:num_sampled]` 覆写 `_processed_log_probs`；
4. 若被采样 Token 落在 Top-K/Top-P 过滤支撑集外（值为 $-\infty$），仅在 `sampled_idx` 上回退至仅经过温度缩放的对数概率 `tempered = gathered - scaled.logsumexp(dim=1)`。

---

## 4. 可复现离线验证

```bash
python experiments/torchtitan_muse_spark_compile_megatron_logprobs/run.py --check
```
