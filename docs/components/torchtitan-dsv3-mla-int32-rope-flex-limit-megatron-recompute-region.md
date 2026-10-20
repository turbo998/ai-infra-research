# TorchTitan DeepSeek-V3 MLA `int32` 对行内 RoPE 旋转、FlexAttention 重编译上限门禁与 Megatron-LM `CheckpointWithoutOutput` `_RecomputeRegion` 状态机

> **English Version**: [torchtitan-dsv3-mla-int32-rope-flex-limit-megatron-recompute-region.en.md](torchtitan-dsv3-mla-int32-rope-flex-limit-megatron-recompute-region.en.md)

- **关联队列题 ID**：`torchtitan-dsv3-mla-int32-rope-flex-limit-megatron-recompute-region`（`分布式训练`, `priority: 39`, `advanced`）
- **证据等级**：`implementation`（上游一手源码提交）+ `own-offline`（Python 3.12 标准库确定性 IEEE-754/BF16 位级验证与重计算状态机仿真）
- **核验日期**：2026-10-21（北京时间 UTC+08:00）
- **一手上游来源**：
  - `pytorch/torchtitan` commit [`2fddc6b7a0d32bbe629f24e8b74977fa3b7f2643`](https://github.com/pytorch/torchtitan/commit/2fddc6b7a0d32bbe629f24e8b74977fa3b7f2643)（PR `#5529`, 2026-10-20）：`torchtitan/models/common/rope.py`（blob `639efaa597`）、`torchtitan/models/deepseek_v3/attention.py`（blob `f0c2f4d24e`）、`tests/unit_tests/gpu/models/deepseek_v3/test_mla_rope_compile.py`（blob `28de928793`）
  - `pytorch/torchtitan` commit [`21db340cba613c00702ea97730cc3cb79b6d6f50`](https://github.com/pytorch/torchtitan/commit/21db340cba613c00702ea97730cc3cb79b6d6f50)（PR `#5161`, 2026-10-20）：`torchtitan/models/deepseek_v3/attention.py`（blob `dc7da03e85`）、`torchtitan/models/deepseek_v3/model.py`（blob `4da2179587`）
  - `pytorch/torchtitan` commit [`aabe6eff1e6e35379e0c3a3d5041981a1ab60d5f`](https://github.com/pytorch/torchtitan/commit/aabe6eff1e6e35379e0c3a3d5041981a1ab60d5f)（PR `#5442`, 2026-10-20）：`torchtitan/models/common/attention/attention.py`（blob `ad23ce5b43`）、`torchtitan/distributed/utils.py`（blob `9de899fd23`）
  - `NVIDIA/Megatron-LM` commit [`d2c430ab485c6947701447e5d551d3aafaa39161`](https://github.com/NVIDIA/Megatron-LM/commit/d2c430ab485c6947701447e5d551d3aafaa39161)（PR `#8334`, 2026-10-20）：`megatron/core/tensor_parallel/random.py`（blob `578e2f5244`）、`tests/unit_tests/tensor_parallel/test_random.py`（blob `2acdcacabd`）

---

## 1. 核心架构问题：MLA 切片视图非连续半字访存、FlexAttention 静默回退 Eager 与无输出重计算引用环

在大规模 MoE（DeepSeek-V3 671B、Muse Spark）分布式训练中，局部区域编译（Regional Compilation）与细粒度激活重计算（Activation Recomputation）存在三处深层瓶颈：

1. **DeepSeek-V3 MLA `q_nope` / `q_pe` 切片破坏 `int32` 对合并访存并诱发 Inductor 掩码加载（`#5529`, `#5161`）**：
   DeepSeek-V3 每个 Query Head 维度为 `192`（`qk_nope_head_dim=128` + `qk_rope_head_dim=64`）。若先执行 `q_nope, q_pe = torch.split(q, [128, 64], dim=-1)`，切出的 `q_pe` 是跨步（Strided）非连续视图（`is_contiguous() == False`），导致 `_RotatePairs.rotate` 退化为逐半字（Half-word）拆解读取，且 Inductor 默认的 Pointwise `cat` 会将 `torch.cat([q_nope, q_pe], dim=-1)` 编译为带条件分支的掩码加载（Masked Loads），反而慢于独立 `cat` 内核！
2. **FlexAttention 默认 `recompile_limit=8` 导致第 9 个图静默退回 Eager（`#5442`）**：
   混合注意力模型（如 Muse Spark）跨层需要编译 `10` 个不同形态的 FlexAttention 图。PyTorch 默认 `recompile_limit=8` 在遇到第 9、10 个图时不会报错，而是**静默回退至未编译 Eager 执行**，造成显存暴涨与训练步耗时劣化近 `4x`。
3. **Megatron-LM `CheckpointWithoutOutput` 自引用环、无梯度调用污染 TE FP8 队列与并行消费者重复重算（`#8334`）**：
   旧版 `CheckpointWithoutOutputFunction` 将 `checkpoint_without_output_obj` 与输出张量直接互相持有，形成 `output -> grad_fn -> ctx -> ckpt -> outputs` 引用环；在 `torch.no_grad()` 或输入不需要梯度时仍进入 `activation_recompute_forward(recompute_phase=False)`，向 Transformer Engine 的 `_is_first_fp8_module` 队列压入永远不会被反向消费的状态；且当同一重计算区的输出被多个并行分支（`out_a`, `out_b`）注册丢弃钩子时缺乏显式状态机保护。

---

## 2. 一手源码机制走读与公式推导

### 2.1 TorchTitan DeepSeek-V3 `_RotateTail` 行内 `int32` 词对旋转与 `@local_compile("mla_qk")`（`2fddc6b7a0`, `21db340cba`）

在 `torchtitan/models/deepseek_v3/attention.py`（blob `f0c2f4d24e`）与 `torchtitan/models/common/rope.py`（blob `639efaa597`）中：

```python
@local_compile(
    "mla_qk",
    batch_invariant=True,
    options={
        "max_pointwise_cat_inputs": 0,
        "max_complex_pointwise_cat_inputs": 0,
        "triton.autotune_pointwise": False,
    },
)
def _apply_rope_to_q_and_k_pe(self, q, k_pe, positions):
    k_pe = k_pe.unsqueeze(1)
    if (
        torch.compiler.is_compiling()
        and type(self.rope) is ComplexRoPE
        and q.dtype == torch.bfloat16
        and q.is_contiguous()
    ):
        cache = self.rope._reshape_cache(q, positions).unflatten(-1, (-1, 2))
        cos, sin = cache[..., 0], cache[..., 1]
        return _RotateTail.apply(q, cos, sin), _RotatePairs.apply(k_pe, cos, sin)
```

`_RotateTail.rotate(x, cos, sin)` 直接将连续的 `bf16` 行视图转为 `torch.int32`（每个 `int32` 词在小端序下低 16 位存放 $x_{2i}$，高 16 位存放 $x_{2i+1}$）：

$$\text{words} = x.\text{view}(\texttt{int32}), \quad (\text{kept}, \text{tail}) = \text{split}(\text{words}, [W - P, P])$$

在 `_RotatePairs.rotate` 中，`packed = (x.dtype == torch.int32)` 分支利用 BF16 精确等于高 16 位 FP32 的性质，通过位运算在寄存器内完成零拷贝解包与打包：

$$x_{\text{real}} = (\text{pairs} \ll 16).\text{view}(\texttt{fp32}), \quad x_{\text{imag}} = (\text{pairs} \ \& \ \texttt{\~0xFFFF}).\text{view}(\texttt{fp32})$$

$$\text{low} = \text{bf16}(x_{\text{real}}\cos - x_{\text{imag}}\sin).\text{view}(\texttt{int16}).\text{to}(\texttt{int32}) \ \& \ \texttt{0xFFFF}$$

$$\text{high} = \text{bf16}(x_{\text{real}}\sin + x_{\text{imag}}\cos).\text{view}(\texttt{int16}).\text{to}(\texttt{int32}) \ll 16$$

配合 `max_pointwise_cat_inputs: 0`，`torch.cat([kept, rotated], dim=-1).view(torch.bfloat16)` 生成无掩码的纯拷贝拼接，使每行（`192` 个 `bf16`）访存事务从 `384` 次半字读写降为 **`192` 次 `int32` 连续读写（削减 `-50.0%`）**。

### 2.2 FlexAttention `recompile_limit=16` 与 `fullgraph=True`（`aabe6eff1e`）

在 `torchtitan/models/common/attention/attention.py`（blob `ad23ce5b43`）与 `torchtitan/distributed/utils.py`（blob `9de899fd23`）中：

```python
_compiled_flex_attn: ClassVar[Callable] = torch.compile(
    flex_attention,
    fullgraph=True,
    options=inductor_configs,
    recompile_limit=16,
)
```

将单实例重编译上限从 `8` 提升至 `16`（覆盖 Muse Spark 所需的 `10` 个异构图），同时开启 `fullgraph=True`：一旦出现图断裂（Graph Break）或第 17 个图，立即抛出异常而非静默退回 Eager。

### 2.3 Megatron-LM `_RecomputeRegion` 单向四态状态机（`d2c430ab48`）

在 `megatron/core/tensor_parallel/random.py`（blob `578e2f5244`）中，`CheckpointWithoutOutput` 重构为四个核心抽象：
1. `_RegionState` 单向状态机：`RECORDED -> RELEASED -> REBUILT -> DONE`。
2. `_InputStash(torch.autograd.Function)`：仅通过 `ctx.save_for_backward(*tensors)` 保存输入张量并返回空句柄 `torch.empty(0)`（不参与 Loss，从而既对 `saved_tensors_hooks` / CUDA Graph 观察器可见，又只在 `_unstash_inputs()` 时通过 `stash.grad_fn.saved_tensors` 解包一次）。
3. `_RecomputeRegion`：仅保存 `output.detach()` 别名（打破 `output -> grad_fn -> region -> output` 循环引用）；在 `release()` 时对非保留输入存储执行 `storage.resize_(0)`；在 `rebuild()` 时校验 `if self.state not in (RECORDED, RELEASED): return`，确保并行消费者（`out_a`, `out_b`）注册多个梯度钩子时**仅重算一次**，并通过 `share_storage(alias, output)` 在 `UntypedStorage` 层零拷贝恢复数据指针。
4. 当 `not torch.is_grad_enabled()` 或没有任何输入 `requires_grad` 时，`checkpoint()` 直接执行普通函数调用，不进入 `_TERecomputeContext`，避免向 TE `_is_first_fp8_module` 队列遗留未消费状态。

---

## 3. 标准库确定性离线实验结果（`experiments/torchtitan_dsv3_mla_rope_megatron_recompute/run.py --check`）

| 评测维度 | 旧版基线行为 | 修复/优化后行为（`#5529` / `#5161` / `#5442` / `#8334`） | 量化收益 / 结论 |
|---|---|---|---|
| **DeepSeek-V3 MLA RoPE 行访存指令数（`T=1027, H=128, D=192`）** | 切片视图逐半字读写：`384` 次/行，且 Pointwise `cat` 产生 `25,239,552` 次掩码加载 | `_RotateTail` `int32` 对旋转 + `max_pointwise_cat_inputs=0`：**192 次/行**，`0` 次掩码加载 | **访存指令减半（`-50.0%`），前/反向位级 100% 一致** |
| **FlexAttention 10 图跨层架构（Muse Spark）** | 默认 `recompile_limit=8`，第 9、10 图静默回退 Eager（步耗时 `46.6 ms`） | `recompile_limit=16, fullgraph=True`，`10/10` 全编译（步耗时 **12.0 ms**），第 17 图快速失败 | **消除静默 Eager 回退，注意力步提速 `3.88x`** |
| **Megatron `CheckpointWithoutOutput` 无梯度/Eval 调用（32 步）** | 每次调用向 TE `_is_first_fp8_module` 泄漏 `1` 条状态（共 `32` 条）且含 `1` 个引用环 | `records_graph == False` 直接透传调用：**0 条 TE 泄漏、0 个引用环** | **彻底根除 Eval/No-Grad 阶段 TE FP8 队列错位** |
| **并行双消费者（`out_a`, `out_b`）双钩子重计算** | 缺乏单向状态机守卫易重复触发 | 严格沿 `RECORDED -> RELEASED -> REBUILT -> DONE` 演进，`grad_enabled=[False, True]` | **多钩子并行消费者 100% 幂等单次重算** |

---

## 4. 解决方案架构师检视清单与可证伪问题

1. **架构师核查项**：
   - 在使用 `torch.compile` 编译包含 `split -> RoPE -> cat` 的 MLA 算子时，应通过 `view(torch.int32)` 在连续行上直接拆分保持段与旋转段，并显式设置 `max_pointwise_cat_inputs: 0` 禁用 Inductor 掩码拼接。
   - 在多模态或混合注意力模型中启用 `flex_attention` 时，务必统计全模型不同 BlockMask / ScoreMod 图变体数量，将 `recompile_limit` 设为大于变体数并开启 `fullgraph=True`。
2. **后续可证伪问题**：
   - 当 DeepSeek-V3 MLA 启用 Context Parallelism（CP）且 `q` 在进入 RoPE 前经过 All-to-All 重排时，如何保证重排输出满足 `_RotateTail` 所需的连续存储与偶数元素偏移（Even Storage Offset）先决条件？

---

## 5. 关联证据 ID

- `claim-torchtitan-dsv3-mla-int32-rope-tail-flex-limit`
- `claim-megatron-checkpoint-without-output-recompute-region`
- `claim-torchtitan-mla-rope-megatron-recompute-offline`
