# veRL 低精度 QAT Rollout 原地权重热更新 (`vllm_fp4_utils.py` 零拷贝存储别名) 与尾桶感知粗粒度 Top-K 在策蒸馏 (`forward_kl_topk_tail`) 机制

> **English Version**: [verl-qat-nvfp4-mxfp8-rollout-tail-kl-distillation.en.md](verl-qat-nvfp4-mxfp8-rollout-tail-kl-distillation.en.md) | **关联实验**: [`experiments/verl_fp4_refit_tail_kl_sim/README.md`](../../experiments/verl_fp4_refit_tail_kl_sim/README.md)

### 一句话结论

veRL 在强化学习与在策蒸馏（On-Policy Distillation, OPD）主线中合入了两大关键机制：（1）在低精度 QAT Rollout（commit `8f016846e3`）中，通过 **`_mxfp4_staging_data` / `_nvfp4_staging_data` 零拷贝存储重解释（`data.flatten().view(dtype).reshape(shape)`）+ `_replace_parameter_in_place` 原地回写**，使 61 层 MoE（每卡 32 专家，`40,992 MiB` 4-bit 权重）在每步 RL Refit 权重同步时**无需分配第二份专家权重副本**，仅需为块缩放因子分配 **`1/16`（MXFP4，`2,562 MiB`）或 `1/8`（NVFP4，`5,124 MiB`）** 的临时缓冲即可保持 vLLM CUDA Graph 捕获地址绝对不变；（2）在 Top-K 在策蒸馏（commit `5a8dd519ab`）中，针对传统截断 `forward_kl_topk` 在学生 Top-$k$ 概率质量 $Q_k > P_k$ 时出现负损失并因 `clamp(min=0)` 丢失全部梯度的问题，引入 **`forward_kl_topk_tail`（`compute_tail_bucket_kl` 与 `compute_tail_aware_logit_gradient`）**，在不增加 Teacher RPC 传输字节的前提下，通过单聚合尾桶（Ghost Token）恢复精确非负粗粒度 KL 下界（如 $P_k=0.70, Q_k=0.92$ 时将截断负值 `-0.19125` 修正为严格正值 **`+0.205276`**）与词表分片解析梯度。

### 组件责任与上游主线位置

| 模块与文件 | 上游仓库与 Commit / Blob SHA | 核心工程职责 |
| :--- | :--- | :--- |
| `verl/utils/qat/compressed_tensors.py` | `volcengine/verl` commit `8f016846e3010e225072a947270dc0c1b98c1cd0` / blob `3408132742c3291a760e4a484d9090dae61545d4` | 定义 `SUPPORTED_SCHEMES`（`FP8_BLOCK`, `MXFP8`, `MXFP4`, `NVFP4`）、`DEFAULT_IGNORE` 正则白名单（保持 `lm_head`, `embed_tokens`, `mlp.gate` 高精度）及 Trainer 侧 `CompressedTensorsWeightQuantizer`。 |
| `verl/utils/vllm/vllm_fp4_utils.py` | `volcengine/verl` commit `8f016846e3010e225072a947270dc0c1b98c1cd0` / blob `184e96191aef2955b201eda3ff88905abded8fcf` | 实现 `install_mxfp4_mxfp8_compressed_tensors_patch`（默认路由至验证正确的 `flashinfer_cutlass`）、零拷贝权重别名 `_mxfp4_staging_data` 及 CUDA Graph 地址保持钩子 `_replace_parameter_in_place`。 |
| `verl/trainer/distillation/tail_kl.py` | `volcengine/verl` commit `5a8dd519ab32bfd36d2509a137f830ba79e89c54` / blob `12be58de3657be1748c0f62750691ea31a5f3343` | 实现单侧 $\epsilon$ 保护的尾桶 KL `compute_tail_bucket_kl` 与支持词表张量并行（Vocab Shard）的解析 Logit 梯度 `compute_tail_aware_logit_gradient`。 |
| `docs/algo/opd.md` | `volcengine/verl` commit `5a8dd519ab32bfd36d2509a137f830ba79e89c54` / blob `a844cc30588eb5182fda8c080914c576b401c40a` | 推导 GKD OPD、PG OPD（`k1` 逆向 KL 估计器）与 `forward_kl_topk_tail` 的数学等价性（LFM2 Decoupled Top-K KD 与 Ghost Token 下界性质）。 |

### 核心实现路径、数据布局与同步契约

1. **为什么 vLLM MoE 4-bit 权重热更新（Refit）容易破坏 CUDA Graph 或使显存翻倍（`vllm_fp4_utils.py:16-30, 238-319`）**：
   - 在 vLLM 启动加载 MXFP4 / NVFP4 专家权重后，`process_weights_after_loading` 会将 Checkpoint 布局的 `w13_weight`（`uint8 [E, 2I, H/2]`）和 `w2_weight`（`uint8 [E, H, I/2]`）重排（Tile Permute / Repack）为特定后端（如 `FLASHINFER_CUTLASS_MXFP4_MXFP8`、`DEEPGEMM_MXFP4` 或 `MARLIN` `int32` Tile）的内核布局，随后 CUDA Graph 捕获这些参数的**物理显存首地址（`data_ptr()`）**。
   - 在每步 RL 训练结束向 vLLM 推送新权重（Refit）时，流式传入的仍是 Checkpoint 布局张量。若新建一套 Checkpoint 缓冲接收权重再调用 `replace_parameter`，不仅会令 MoE 权重峰值显存瞬间翻倍（`+40.99 GiB/GPU`），而且 `replace_parameter` 替换 `nn.Parameter` 会改变 `data_ptr()`，导致已捕获的 CUDA Graph 读到悬空旧权重！
   - `vllm_fp4_utils.py` 的精妙解法：
     - **权重零拷贝字节视图别名（Zero-Copy Storage Reinterpretation）**：无论是 DeepGEMM、FlashInfer CUTLASS 还是 Marlin 的 4-bit 重排，本质上都是对相同数量 4-bit 元素的置换（`data.numel() * data.element_size() == shape.numel() * dtype.itemsize`）。因此 `_mxfp4_staging_data` 直接返回 `data.flatten().view(dtype).reshape(shape)`——让 `load_weights` **零额外分配地直接写入原 `param.data` 的物理存储**！
     - **仅对膨胀/变形的 Block Scales 分配临时缓冲**：MXFP4 Scale（`_MXFP4_SF_BLOCK = 32`，`E8M0`）仅占打包权重的 `1/16`（`6.25%`），NVFP4 Scale（`_NVFP4_SF_BLOCK = 16`，`E4M3`）仅占打包权重的 `1/8`（`12.5%`）。
     - **`_replace_parameter_in_place` 拦截回写**：在重放 `process_weights_after_loading` 内部调用 `replace_parameter` 的瞬间，将其拦截并执行 `param.data.copy_(new_data)` 原地拷回原 `param`，从而确保 `_setup_kernel` 重建 `moe_kernel` 以及 CUDA Graph 重放时看到的始终是同一个物理地址！
2. **修复 vLLM 0.29 MXFP4×MXFP8 激活降级与 TRTLLM 数值缺陷（`vllm_fp4_utils.py:48-144`）**：
   - vLLM 0.29 在加载 `compressed-tensors` MXFP4 权重时忽略了 `input_activations: mxfp8` 配置，在 Blackwell 上无条件构造了 W4A4 方法；同时 TRTLLM 的 `MXFP4xMXFP8` 内核在 Qwen3.5 上存在数值错误。`install_mxfp4_mxfp8_compressed_tensors_patch` 动态注入 `CompressedTensorsW4A8Mxfp4MoEMethod`，将 `activation_key=kMxfp8Dynamic` 与默认 `moe_backend="flashinfer_cutlass"` 绑定。
3. **尾桶感知粗粒度 Top-K 在策蒸馏 `forward_kl_topk_tail`（`tail_kl.py:18-70`, `opd.md:430-502`）**：
   - 在分布式 OPD 中，Teacher 只向 Student 发送 Top-$k$（如 $k=128$）Token 的概率 $p_v = \nu(v\mid s_t)$。记 $P_k = \sum_{v \in T_k} p_v \le 1$，$Q_k = \sum_{v \in T_k} q_v \le 1$（其中 $q_v = \pi_\theta(v\mid s_t)$）。
   - 当学生模型在 Top-$k$ 集合上的累计概率大于教师（$Q_k > P_k$）时，截断和 $\sum_{v \in T_k} p_v \log(p_v / q_v)$ 可以为**负数**！旧版 `forward_kl_topk` 将负损失直接 `clamp(min=0)`，导致所有 $Q_k > P_k$ 的位置梯度彻底归零，学生无法学到“压低过度自信的 Top-$k$ 质量并向尾部释放概率”。
   - `tail_kl.py` 将全部非 Top-$k$ 词表合并为一个聚合尾桶（Ghost Token），定义粗粒度正向 KL：
     $$\mathcal{L}_{\mathrm{tail}}^{(k)} = \sum_{v \in T_k} p_v \log\frac{p_v}{q_v} + (1 - P_k)\log\frac{\max(1 - P_k, \text{tiny})}{\max(1 - Q_k, \epsilon_{\text{tail}})}$$
   - **单侧 $\epsilon$ 保护设计**：$\epsilon_{\text{tail}} = 10^{-6}$ **仅作用于分母中的学生尾桶 $1 - Q_k$**（防止 $Q_k \to 1$ 时 $\log(1 - Q_k) \to -\infty$ 导致梯度爆炸）；而教师尾桶 $1 - P_k$ 作为乘法权重保持不截断（仅对 $\log$ 输入取 `finfo.tiny` 避免 `0 * (-inf) = NaN`），利用 $\lim_{x \to 0^+} x \log x = 0$ 保证当 $P_k = 1$ 时尾桶项精确归零。

### 关键公式、单位与工程假设

1. **支持词表分片（Vocab-Sharded）的粗粒度 Top-K + 尾桶对 Logit $z_j$ 的解析梯度（`tail_kl.py:42-70`）**：
   记 $P_{\text{tail}} = \operatorname{clamp}(1 - P_k, 0, 1)$，$Q_{\text{tail}} = \operatorname{clamp}(1 - Q_k, \epsilon_{\text{tail}}, 1)$，激活指示符 $\mathbb{I}_{\text{active}} = (1 - Q_k > \epsilon_{\text{tail}})$。定义尾桶外缩放系数：
   $$s_{\text{out}} = \begin{cases} P_k - P_{\text{tail}} \cdot \dfrac{Q_k}{Q_{\text{tail}}}, & \text{if } 1 - Q_k > \epsilon_{\text{tail}} \\ P_k, & \text{otherwise} \end{cases}, \qquad s_{\text{topk}} = P_k + \mathbb{I}_{\text{active}} \cdot P_{\text{tail}}$$
   则对任意词表 Token $j \in V$，损失关于学生 Logit $z_j$ 的解析梯度为：
   $$\frac{\partial \mathcal{L}_{\mathrm{tail}}^{(k)}}{\partial z_j} = \begin{cases} s_{\text{topk}} \cdot q_j - p_j = (P_k + \mathbb{I}_{\text{active}} P_{\text{tail}}) q_j - p_j, & j \in T_k \\ s_{\text{out}} \cdot q_j = \left(P_k - \mathbb{I}_{\text{active}} P_{\text{tail}} \dfrac{Q_k}{Q_{\text{tail}}}\right) q_j, & j \notin T_k \end{cases}$$
   当 $1 - Q_k > \epsilon_{\text{tail}}$ 时，$s_{\text{topk}} = P_k + (1 - P_k) = 1$，因此 Top-$k$ 内梯度退化为极简的 $q_j - p_j$，而全体非 Top-$k$ Token 的梯度之和为 $\sum_{j \notin T_k} s_{\text{out}} q_j = (1 - Q_k) \left(P_k - (1 - P_k)\frac{Q_k}{1 - Q_k}\right) = P_k - Q_k$，故全词表梯度总和恒等于 $\sum_{j \in T_k}(q_j - p_j) + (P_k - Q_k) = Q_k - P_k + P_k - Q_k = 0$！

### 合成实验与定量验证（`experiments/verl_fp4_refit_tail_kl_sim`）

运行 `python experiments/verl_fp4_refit_tail_kl_sim/run.py --check` 对三类 Top-K 蒸馏分布及 61 层 MoE（每卡 32 专家，`H=7168, I=2048`）Refit 显存开销进行确定性验证（数据源自 `experiments/verl_fp4_refit_tail_kl_sim/results/summary.json`）：

| 验证维度 / 方案 | 关键指标 1 | 关键指标 2 | 关键指标 3 | 关键指标 4 |
| :--- | :---: | :---: | :---: | :---: |
| **Case A：学生高估 Top-$k$ 质量** ($P_k=0.70, Q_k=0.92$) | 原始截断 KL：`-0.19125`（旧版 Clamp 后为 `0.0`，梯度丢失） | 尾桶 KL 修正项：`+0.396527` | **粗粒度总 KL：`+0.205276`（恢复非负下界）** | 尾部 L1 梯度范数：`0.2200`（解析/差分误差 `0.0`） |
| **Case B：学生低估 Top-$k$ 质量** ($P_k=0.88, Q_k=0.55$) | 原始截断 KL：`+0.413848` | 尾桶 KL 修正项：`-0.158611` | **粗粒度总 KL：`+0.255237`** | 尾部 L1 梯度范数：`0.3300`（$\sum \nabla_z = 0$） |
| **Refit：`verl_MXFP4xMXFP8_zero_copy_alias`** | 常驻 4-bit 专家权重：`40,992.0 MiB/GPU` | **Refit 峰值临时分配：`2,562.0 MiB/GPU`** | **临时/常驻比：`0.0625` (`1/16`)** | CUDA Graph 地址保持：`true`（对比朴素克隆省 `94.4%`） |
| **Refit：`verl_NVFP4_W4A4_zero_copy_alias`** | 常驻 4-bit 专家权重：`40,992.0 MiB/GPU` | **Refit 峰值临时分配：`5,124.0 MiB/GPU`** | **临时/常驻比：`0.1250` (`1/8`)** | CUDA Graph 地址保持：`true`（对比朴素克隆省 `88.9%`） |

### 架构师设计练习与可证伪问题

1. **为什么 `forward_kl_topk` 和 `forward_kl_topk_tail` 严禁与 `use_policy_gradient=true`（PG OPD）同时开启（`opd.md:490-502`）？**
   - **解答**：Top-$k$ 正向 KL 度量的是完整 Top-$k$ 分布（外加尾桶）之间的散度，但在策略梯度更新 $\nabla_\theta \mathcal{L}_{\text{PG}} \propto -A_t \nabla_\theta \log \pi_\theta(y_t \mid s_t)$ 中，梯度仅通过当前采样的单个 Token $y_t$ 反向传播。如果学生在已采样的 Token $y_t$ 上概率已经与教师完全一致（$\pi_\theta(y_t) = \nu(y_t)$），但在其他未采样的 Top-$k$ Token 上偏离了教师，此时正向 KL 仍然严格大于 0；若将其作为标量奖励惩罚 $y_t$，会错误地推开原本已经对齐的采样 Token $y_t$。
2. **可证伪假设**：在 vLLM 的所有 MXFP4 MoE 后端中，`Triton` 后端能否像 `FLASHINFER_CUTLASS_MXFP4_MXFP8`、`DEEPGEMM_MXFP4` 和 `MARLIN` 一样支持 `_replace_parameter_in_place` 零拷贝热更新？
   - **验证方式**：查阅 `vllm_fp4_utils.py:196-198`——不能，`_refittable_mxfp4_backends()` 显式排除了 Triton，因为 Triton 后端将权重包装为非 `nn.Parameter` 对象并将 Scale 停放在 `quant_method` 属性上，绕过了 `replace_parameter` 钩子。
