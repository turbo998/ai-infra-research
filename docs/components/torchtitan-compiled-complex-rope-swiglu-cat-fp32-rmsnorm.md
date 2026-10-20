# TorchTitan 编译态实数算术 `ComplexRoPE`（`int32` 双 `bf16` 打包）、SwiGLU 反向无掩码半区存储与 Muse Spark `fp32` 残差流 RMSNorm 融合核验

**语言 / Language**: [中文](torchtitan-compiled-complex-rope-swiglu-cat-fp32-rmsnorm.md) | [English](torchtitan-compiled-complex-rope-swiglu-cat-fp32-rmsnorm.en.md)

> **证据级别 / Evidence Tier**: `implementation`（一手上游提交源码）+ `author-benchmark`（TorchTitan 作者在 NVIDIA GB300 / A10G / MI350X 上的实测数据）+ `own-offline`（Python 3.12 标准库确定性验证 `experiments/torchtitan_compiled_rope_swiglu_fp32_rmsnorm/run.py --check`）。

---

## 1. 核心架构问题与结论速览

在 PyTorch `torch.compile`（TorchInductor）局部区域编译（`@local_compile`）中，复数算术回退、拼接（`cat`/`stack`）掩码条件选择以及混合精度张量 dtype 不一致是导致算子退化为多内核启动或带宽翻倍的三大根源。2026-10-20 的三项 `pytorch/torchtitan` 提交给出了极具代表性的编译器协同优化范式：

1. **编译态实数算术 `ComplexRoPE` 与 `int32` 双 `bf16` 连续读写（[`pytorch/torchtitan@b95271ed07`](https://github.com/pytorch/torchtitan/commit/b95271ed0709d63a60d185b88994cba10ea748a1)）**：
   - **问题根因**：Inductor 无法为 `view_as_complex` 与复数乘法生成融合内核，导致编译态下的 `ComplexRoPE` 回退为多个独立 eager 内核（在 GB300 上，Llama 3 8B 64k 预填充耗时 `1126.5 us`，DeepSeek-V3 671B 64k 预填充耗时 `2589.9 us`）。
   - **实数旋转与浮点 Cache Gather 融合**：在 `torch.compiler.is_compiling()` 分支下，`_reshape_cache` 直接对 `torch.view_as_real(self.cache).flatten(-2)`（形状 `(T, 1, dim)`）做实数行 gather，使其与 `_RotatePairs` 的实数旋转融合进同一个内核。
   - **单 `int32` 打包相邻 `bf16` 对**：若按两个半区分别写出 `bf16`，stride-2 存储在动态长度 `T` 的行尾无法向量化。`_RotatePairs.rotate` 对连续 `bf16` 输入将每对 `(x[2i], x[2i+1])` 视为一个 `int32` 读取：
     $$x_{\text{real}} = (\text{pairs} \ll 16).\text{view}(\text{float32}), \quad x_{\text{imag}} = (\text{pairs} \mathbin{\&} \sim\text{0xFFFF}).\text{view}(\text{float32})$$
     旋转后再将两个 `bf16` 拼回单个 `int32`（`(low | high).view(torch.bfloat16)`）连续写出。在 GB300 上，64k 预填充延迟在 Llama 3 8B 上从 `1126.5 -> 224.3 us`（**`5.02x` 加速**，优于手写 Helion RoPE 的 `231.5 us`），在 DeepSeek-V3 671B 上从 `2589.9 -> 426.5 us`（**`6.07x` 加速**）。
   - **关闭 Pointwise Autotune 保证多卡确定性**：启用 `triton.autotune_pointwise` 时，Inductor 按计时在不同 FMA 收缩（contraction）配置间摇摆，导致 `deepseek_v3_fsdp+cp+pp+ep` 在 13 次运行中出现 `9` 个不同的 step-1 loss；显式配置 `"triton.autotune_pointwise": False` 后，12 次运行产生 **`1` 个完全一致的确定性 loss**。

2. **SwiGLU 反向传播去除掩码 `tl.where`：每半区独立 Store（[`pytorch/torchtitan@eb48bab350`](https://github.com/pytorch/torchtitan/commit/eb48bab35099a79dfcd7d9c5d26b5af5e41f06d8)）**：
   - **问题根因**：`fused_binary_activation` 在前向解包 `gate_up [..., 2, F]`，反向传播需将 `grad_gate` 与 `grad_up` 重新堆叠为 `[..., 2, F]`。Inductor 默认将 2 输入拼接编译为单个跨两个半区的 pointwise 内核：每个元素都在掩码后加载输入并用 `tl.where` 选择半区，导致 GB300 上 `T=16k, F=2048` 的反向内核耗时 `788 us`（是其 `377 us` 带宽下限的 `2.09x`）。
   - **双参数归零缺一不可**：仅设置 `max_pointwise_cat_inputs=0` 不足以关闭该路径，因为 `max_complex_pointwise_cat_inputs`（默认值 8）仍会将 2 输入拼接送入掩码分支。同时设置 `options={"max_pointwise_cat_inputs": 0, "max_complex_pointwise_cat_inputs": 0}` 后，Inductor 为每个半区发出独立的无掩码 store，使 GB300 `T=16k, F=2048` SwiGLU `fwd+bwd` 从 `1440.0 -> 1047.8 us`（**`-27.24%`**，比手写 Triton override 的 `1060.5 us` 还快 `1.20%`），反向内核估算耗时降至 `~395.8 us`（逼近带宽下限的 `1.05x`）。

3. **Muse Spark `fp32` 残差流 RMSNorm 权重预转型恢复单内核融合（[`pytorch/torchtitan@b78c4aed96`](https://github.com/pytorch/torchtitan/commit/b78c4aed9640b5fb034b9be1c05f24fe30f1d093)）**：
   - Muse Spark 将残差流保持为 `fp32`（`residual_dtype`），而 `effective_weight_D`（`weight + gain_center`）在 FSDP 混合精度或推理生成器下为 `bf16`。PyTorch `F.rms_norm` 仅在 `input.dtype == weight.dtype` 时调用融合内核；dtype 不匹配时会回退为 `pow`、`mean`、`rsqrt`、`mul` 等 **`~20` 个小算子内核**。
   - 由于 `bf16 -> fp32` 拓宽转换在二进制上完全精确（7 位尾数补零扩展为 23 位尾数，`0.0` 数值误差），在 `MuseSparkRMSNorm.forward` 中增加 `effective_weight_D = effective_weight_D.to(input_dtype)` 即可将前向收敛为 **`1` 个融合内核**（2 层端到端训练耗时 `-7%`、预填充 `-5%`、解码 `-2%..-4%`）。

---

## 2. GB300 作者实测与确定性离线模型对比 (`author-benchmark` + `own-offline`)

通过 `python experiments/torchtitan_compiled_rope_swiglu_fp32_rmsnorm/run.py --check` 验证了位级打包与性能模型（详见 `experiments/torchtitan_compiled_rope_swiglu_fp32_rmsnorm/results/summary.json`）：

| 优化模块 | 工作负载与规模 (`GB300`, `author-benchmark`) | 优化前 (`main` / Eager) | 优化后 (`this PR`) | 手写内核对照 (`Helion` / `Triton`) | 关键收益与确定性指标 (`own-offline`) |
|---|---|---:|---:|---:|---|
| **`ComplexRoPE` (`rope` region)** | Llama 3 8B (`T=64k` prefill) | `1126.5 us` | **`224.3 us`** | `231.5 us` (Helion) | **`5.02x`** 加速，`int32` 对打包 `0` ULP 误差 |
| **`ComplexRoPE` (`rope` region)** | DeepSeek-V3 671B (`T=64k` prefill) | `2589.9 us` | **`426.5 us`** | `384.9 us` (Helion, 128k 溢出失败) | **`6.07x`** 加速，`autotune_pointwise=False` 消除 `9/13` 步间漂移 |
| **SwiGLU (`fused_binary_activation`)** | `[8T, 2, 2048]` (`T=16k` train fwd+bwd) | `1440.0 us` | **`1047.8 us`** | `1060.5 us` (Triton override) | **`-27.24%`**，比手写 Triton 快 `1.20%`，反向逼近带宽下限 `1.05x` |
| **Muse Spark `RMSNorm`** | `fp32` 残差流 + `bf16` 权重 | `~20` 个回退内核 | **`1` 个前向融合内核** | ATen fused `rms_norm` | 2 层训练 `-7.0%`、预填充 `-5.0%`，`fp32` 零精度损失 (`0.0`) |

---

## 3. 面向解决方案架构师的设计练习与可证伪问题

1. **位运算练习**：为何 `_RotatePairs.rotate` 对连续 `bf16` 对 `(x[2i], x[2i+1])` 打包为小端 `int32` `pairs` 后，`x_real = (pairs << 16).view(torch.float32)` 与 `x_imag = (pairs & ~0xFFFF).view(torch.float32)` 能精确还原两个 `fp32` 浮点数？
   - **解答**：小端序下 `x[2i]` 位于 `pairs` 的低 16 位 `[15:0]`，`x[2i+1]` 位于高 16 位 `[31:16]`。IEEE 754 `fp32` 与 `bf16` 共享 1 位符号位与 8 位指数位，`bf16` 恰为 `fp32` 的高 16 位截断；因此低 16 位左移 16 位（低位自动补 0）、高 16 位通过 `& ~0xFFFF` 清零低 16 位，即得到精确对应的 `fp32` 位模式，无需调用浮点类型转换指令。
2. **后续可证伪问题**：在 `CosSinRoPE`（非相邻维度旋转，即前半维度与后半维度配对）中，若同样设置 `max_pointwise_cat_inputs=0` 和 `max_complex_pointwise_cat_inputs=0`，为何在 Qwen3 形状下反而比默认配置慢 `1.3x-2.6x`？
