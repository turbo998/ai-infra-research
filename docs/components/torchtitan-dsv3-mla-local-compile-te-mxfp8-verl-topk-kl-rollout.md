# TorchTitan DeepSeek-V3 MLA 局部区域编译、TransformerEngine Blackwell Grouped GEMM FP8BS (MXFP8 广播仿真) 与 veRL 尾部分布感知 Top-K KL 蒸馏

> **English Version**: [torchtitan-dsv3-mla-local-compile-te-mxfp8-verl-topk-kl-rollout.en.md](torchtitan-dsv3-mla-local-compile-te-mxfp8-verl-topk-kl-rollout.en.md)  
> **证据等级**: `implementation`（上游一手 commit 源码） + `own-offline`（`experiments/torchtitan_mla_compile_te_mxfp8_verl_tail_kl/run.py --check`）

## 1. 一手源码定位与核心问题

1. **TorchTitan DeepSeek-V3 MLA `@local_compile("mla_qk")` 禁用 Inductor Pointwise-Cat 掩码加载退化**（`pytorch/torchtitan` commit [`21db340cba613c00702ea97730cc3cb79b6d6f50`](https://github.com/pytorch/torchtitan/commit/21db340cba613c00702ea97730cc3cb79b6d6f50)，`2026-10-20`）：
   - **代码路径**：`torchtitan/models/deepseek_v3/attention.py`（`_apply_rope_to_q_and_k_pe`）与 `torchtitan/models/deepseek_v3/model.py`。
   - **问题机理**：DeepSeek-V3 MLA 将查询向量切分为 `q_nope`（128 维）与 `q_pe`（64 维），对 `q_pe` 施加实数域 RoPE 后执行 `torch.cat([q_nope, q_pe], dim=-1)`（合成为 192 维）。在 TorchInductor 默认编译选项下，Pointwise-Cat 融合会将 `q_nope` 与 `RoPE(q_pe)` 塞进同一个带条件分支（`idx < 128`）的逐元素内核，生成大量非对齐掩码加载（masked loads），性能甚至劣于独立拼接拷贝。
   - **修复机制**：将切分、RoPE 与拼接封装进 `_apply_rope_to_q_and_k_pe`，并标注 `@local_compile("mla_qk", batch_invariant=True, options={"max_pointwise_cat_inputs": 0, "max_complex_pointwise_cat_inputs": 0})`，强制 Inductor 将 RoPE 内联融合在 64 维张量上，而对 192 维拼接发射纯粹的无分支 DMA/cat 内核。

2. **TransformerEngine Blackwell Grouped GEMM 基于 MXFP8 广播的 FP8 Block Scaling (`FP8BS`) 仿真**（`NVIDIA/TransformerEngine` commit [`f4a69dc81eabb5002a662e56a36e8454c9840a92`](https://github.com/NVIDIA/TransformerEngine/commit/f4a69dc81eabb5002a662e56a36e8454c9840a92)，`2026-10-19`）：
   - **代码路径**：`transformer_engine/pytorch/module/grouped_linear.py`、`transformer_engine/pytorch/ops/basic/grouped_linear.py` 及 `tests/cpp/operator/test_cast_float8blockwise_grouped.cu`。
   - **问题机理**：cuBLASLt 原生 FP8 Block Scaling（128 元素块缩放）Grouped GEMM 仅在 Hopper（`SM90`，cuBLASLt 13.6+）提供硬件内核；在 Blackwell（`SM100+`）上此前直接报错拒绝。
   - **修复机制**：在 Blackwell（`device_capability >= (10, 0)`）框架绑定层通过 MXFP8 Grouped GEMM（32 元素微块 + `E8M0` 2 的幂次缩放因子）无损仿真 FP8BS：强制量化器开启 `force_pow_2_scales=True`（并在 C++ 算子层拦截非 2 的幂次缩放因子，`GroupedFP8BlockwiseBlackwell.RejectsNonPow2Scales`），将每个 128 元素 FP8BS 块的 2 的幂次缩放因子精确广播至 4 个连续的 32 元素 MXFP8 微块，同时校验 `split_alignment % 128 == 0`。

3. **veRL 在线策略蒸馏（On-Policy Distillation）尾部分布感知 Top-K KL (`forward_kl_topk_tail`)**（`volcengine/verl` commit [`5a8dd519ab32bfd36d2509a137f830ba79e89c54`](https://github.com/volcengine/verl/commit/5a8dd519ab32bfd36d2509a137f830ba79e89c54)，`2026-10-20`）：
   - **代码路径**：`verl/trainer/distillation/fsdp/losses.py`（`tail_aware_kl_divergence`）、`verl/trainer/distillation/megatron/losses.py` 与 `docs/algo/opd.md`。
   - **问题机理**：传统截断式 `forward_kl_topk` 仅计算教师模型 Top-$K$ 词表子集上的 $\sum_{i \in \text{Top}K} p_i (\log p_i - \log q_i)$。当学生模型将过多概率质量挤压到 Top-$K$ 词元（使得 $q_i > p_i$）而掏空长尾词元概率时，截断式 Top-$K$ KL 会变为**负数**，错误地奖励学生模型的分布坍缩！
   - **修复机制**：新增 `forward_kl_topk_tail`，将所有非 Top-$K$ 词元折叠为第 $K+1$ 个聚合尾部桶（Aggregate Tail Bucket）：
     $$\mathcal{L}_{\text{tail}} = \left(1 - \sum_{i \in \text{Top}K} p_i\right) \log \frac{1 - \sum_{i \in \text{Top}K} p_i}{\max\left(1 - \sum_{i \in \text{Top}K} q_i, \epsilon_{\text{tail}}\right)}, \quad \epsilon_{\text{tail}} = 10^{-6}$$
     同时强制要求 `log_prob_min_clamp=None`，防止逐词元下界截断导致 $\sum q_i > 1$ 破坏 $K+1$ 维概率单纯形。

## 2. 确定性实验验证（`experiments/torchtitan_mla_compile_te_mxfp8_verl_tail_kl`）

运行 `python experiments\torchtitan_mla_compile_te_mxfp8_verl_tail_kl\run.py --check` 的量化结果：

| 子方向 | 关键指标 | 基线（Legacy / Default） | 修复后（Patched） | 架构收益 |
|---|---|---:|---:|---|
| TorchTitan `@local_compile("mla_qk")` (`21db340c`) | 每步掩码加载分支数 / 区域延迟 | `100,663,296` / `198.0 us` | `0` / `112.0 us` | 相比默认编译 **`1.7679x` 加速**（相比 Eager **`2.5446x` 加速**） |
| TE Blackwell FP8BS via MXFP8 (`f4a69dc8`) | $128 \to 4 \times 32$ `E8M0` 广播误差 / 非 2 幂次拦截 | SM100 不支持 | `0.0` / `4/4` 拦截 | 在满足 `split % 128 == 0` 下零误差复用 Blackwell MXFP8 硬件张量核 |
| veRL `forward_kl_topk_tail` (`5a8dd519`) | 学生尾部坍缩时（$\sum q_{\text{topk}}=0.99$ vs $\sum p_{\text{topk}}=0.80$）的 KL 值 | `-0.170396`（病态负值） | `+0.428726`（精确惩罚） | 恢复严格非负的 $(K+1)$ 类粗粒度 KL 散度并拦截非法 clamp |
