# FlashInfer 大通道 Causal Conv1D 反向算子选择、TensorRT-LLM SM107 DSpark RMSNorm/RoPE 门控解耦与 vLLM PaddleOCR-VL 视觉编码器编译

> **English Version**: [flashinfer-causal-conv1d-big-channel-bwd-trtllm-sm107-dspark-vllm-mm-compile.en.md](flashinfer-causal-conv1d-big-channel-bwd-trtllm-sm107-dspark-vllm-mm-compile.en.md)  
> **证据等级**: `implementation` + `author-benchmark`（源码审计与作者实测）+ `own-offline`（确定性离线建模）  
> **对应实验**: [`experiments/flashinfer_causal_conv1d_bwd_trtllm_sm107_vllm_compile/run.py`](../../experiments/flashinfer_causal_conv1d_bwd_trtllm_sm107_vllm_compile/run.py)

---

## 1. 工程背景与核心问题

在混合注意力/SSM 与多模态大模型推理及训练中，算子库必须在保持 CUDA Graph 可捕获性与确定性逐位复现（Bitwise Reproducibility）的前提下，针对不同 GPU 微架构（`sm_100a` B200、`sm_103a` GB300、`sm_107a` R200）和通道规模动态切换最优内核：
1. **FlashInfer `cake_causal_conv1d_train` 大通道反向带宽与占用率瓶颈（`flashinfer-ai/flashinfer@5f8f8aad`，`#6874`）**：宽度 4 的深度可分离因果一维卷积（融合 SiLU 激活，BF16 输入/输出 + FP32 累加与确定性 `dweight` 归约）在 `C >= 512`（Blackwell B200/GB300）或 `C >= 1024`（R200）时，单 CTA/SM 默认反向内核受限于 token 流水级数与 SM 驻留度。
2. **TensorRT-LLM SM107 DSpark 融合 RMSNorm+RoPE 门控耦合问题（`NVIDIA/TensorRT-LLM@083447cd`，`#20366`）**：`dspark_rmsnorm_rope_custom_op.py` 原本复用 `_get_dspark_arch_str()`（仅允许 `sm_100` 与 `sm_103` 的 DSV4 DSpark 注意力内核），导致纯 SIMT CuTe 实现的融合 RMSNorm/RoPE 算子在 `SM107` 上被误拒。
3. **vLLM `PaddleOCR-VL` 视觉编码器逐层 `torch.compile` 覆盖缺口（`vllm-project/vllm@0ca531ba`，`#61838`）**：`PaddleOCR-VL` 的 Ernie-4.5 语言主干仅注册 `2` 个编译子模块，而 `27` 层 `SiglipEncoderLayer` 缺少变长多模态动态维度编译装饰器。

---

## 2. 一手源码机制剖析与公式推导

### 2.1 FlashInfer 按通道 `channels_at_least` 反向程序选择与容量所有者工作区
在 `flashinfer/cake_causal_conv1d_train_backend/cake_jit.py` 中，新增大通道反向程序 `cake_causal_conv1d_train_90351ecca30604957251`（构建参数：`vec=8, copy="tma2d", stage_tokens=8, min_blocks=2`，`dynamic_smem_bytes=82944`）：
- **确定性主机端路由规则**：`select_rule = {"kind": "channels_at_least", "channels": {"sm_100a": 512, "sm_103a": 512, "sm_107a": 1024}}`。`selected_modules(arch, "bwd", C=C)` 仅依赖架构与通道数 `C`（与序列长度 `S` 及输入数据完全无关），确保固定 `(S, C, device)` 的 `dweight` 逐位确定性。
- **跨程序公共工作区容量上界（Capacity Owner）**：协作式反向将每 CTA 的 FP32 `dweight` 局部和写入 `[rows, C, 4]` 工作区，其中：
  $$n_{\text{ch\_tiles}} = \left\lceil \frac{C}{\text{threads} \times \text{vec}} \right\rceil = \left\lceil \frac{C}{128 \times 8} \right\rceil, \quad \text{rows}(\text{prog}) = \left\lceil \frac{\text{ctas\_per\_sm}(\text{prog}) \cdot \text{SM\_count}}{n_{\text{ch\_tiles}}} \right\rceil$$
  公共接口 `cake_causal_conv1d_train_backward_workspace_bytes` 取架构下所有反向程序 `ctas_per_sm` 的最大值（`max(1, 2) = 2`），例如在 `sm_103a`（`160` SMs）、`C=256` 时，回退程序仅写入前 `160` 行（`655,360 B`），而公共工作区按容量所有者分配 `320` 行（`1,310,720 B`），无需在切换程序或重放 CUDA Graph 时重新分配或清零。
- **逻辑字节流量公式**：
  $$\text{Bytes}_{\text{fwd}} = 4SC + 16C, \quad \text{Bytes}_{\text{bwd}} = 6SC + 32C + 32C \cdot \text{rows}(\text{prog})$$
- **作者实测加速比（`author-benchmark`，CUDA 13.4）**：
  - **GB300 (`sm_103a`)**：全部 14 组形状几何平均加速 `1.0569x`，命中的 12 组（`C >= 512`）几何平均加速 `1.0667x`，在 `S=250,340, C=8192` 下从 `2264.3 us` 降至 `2060.1 us`（`1.0991x`）。
  - **B200 (`sm_100a`)**：全部 14 组几何平均加速 `1.0765x`，命中的 12 组几何平均加速 `1.0898x`，在 `C=1024` 峰值达 `1.1226x`（`330.6 -> 294.5 us`）。
  - **R200 (`sm_107a`)**：全部 14 组几何平均加速 `1.0511x`，命中的 4 组（`C >= 1024`）几何平均加速 `1.1908x`，在 `C=8192` 下从 `1652.2 us` 降至 `1313.3 us`（`1.2581x`）。

### 2.2 TensorRT-LLM SM107 DSpark 算子门控解耦与 vLLM 多模态编译
- **TRT-LLM SM107 门控解耦**：新增 `_DSPARK_RMSNORM_ROPE_SM_VERSIONS = frozenset((100, 103, 107))` 与 `_is_dspark_rmsnorm_rope_sm_supported()`，在 `SM107` 上启用 SIMT CuTe 融合 RMSNorm+RoPE，同时保持 `_get_dspark_arch_str(107) is None` 以隔离依赖特定张量核心指令的 DSV4 DSpark 注意力算子。
- **vLLM `SiglipEncoderLayer` 动态图编译**：通过 `@support_torch_compile(dynamic_arg_dims={"hidden_states": 1, "cu_seqlens": 0, "rotary_pos_emb": 0}, enable_if=should_torch_compile_mm_encoder, is_encoder=True)`，开启 `compile_mm_encoder=True` 后编译子模块数从 `2` 提升至 `29`（`+27` 层视觉编码器）。

---

## 3. 架构师设计练习与可证伪问题

1. **计算题**：在 `sm_103a`（160 SMs）上运行 `S=250,340, C=8192` 的 `cake_causal_conv1d_train_backward`，求 `n_ch_tiles`、`rows` 以及反向逻辑字节流量 `Bytes_bwd`。
   - **解答**：`n_ch_tiles = 8192 / 1024 = 8`；`C=8192 >= 512` 命中 `ctas_per_sm=2` 的大通道程序，故 `rows = ceil(320 / 8) = 40`；`Bytes_bwd = 6 * 250340 * 8192 + 32 * 8192 + 32 * 8192 * 40 = 12,305,711,104 + 262,144 + 10,485,760 = 12,316,459,008 B`（约 `12.316 GB`）。
2. **可证伪假设**：如果在捕获 CUDA Graph 前仅编译当前输入 `C=256` 对应的回退库而未调用 `specs_for_arch(arch)` 预构建全部路由程序，当同一进程随后在另一张图捕获 `C=1024` 时会发生什么？（会在图捕获流内触发 JIT 编译并导致非法 CUDA Stream Capture 错误）。
