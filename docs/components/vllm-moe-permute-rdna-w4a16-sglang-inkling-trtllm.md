# vLLM `moe_permute` 无 `expert_map` 填充路由重映射与动态量化行上界、ROCm RDNA (`gfx1151`) W4A16 `v_and_or_b32` 打包反量化、SGLang Inkling BF16-In/FP32-Out LM-Head 保真、Quark Block-FP8 加载与 TensorRT-LLM CBTS Dataclass 静态上界深度解析

> **语言 / Language**: [中文](vllm-moe-permute-rdna-w4a16-sglang-inkling-trtllm.md) | [English](vllm-moe-permute-rdna-w4a16-sglang-inkling-trtllm.en.md)  
> **Phase**: 70 (`priority: 79`, Track B — LLM Serving, Quantization & MoE Execution)  
> **核心证据等级**: `implementation` (固定 commit 源码审计) + `own-offline` (本项目确定性标准库仿真 `experiments/vllm-moe-permute-rdna-w4a16-sglang-inkling-trtllm/run.py`)

---

## 1. 架构背景与核心工程问题

在 vLLM、SGLang 与 TensorRT-LLM 的最新推理与量化路径演进中，五个跨硬件平台（CUDA / ROCm CDNA3 `gfx942` / RDNA3.5 `gfx1151`）的精度、路由与测试覆盖工程问题被集中修复：

1. **vLLM `moe_permute` 在无 `expert_map`（单卡/非 EP）下的非法路由 (`topk_ids == -1`) 排序错位与动态量化工作区污染 (`vllm-project/vllm@e4db46c79bca`, `#59717`)**：当 CUDA Graph 填充 token 或掩码路由将 `topk_ids` 设为 `-1` 且未传入 `expert_map` 时，为什么 `cub::DeviceRadixSort::SortPairs` 会把 `-1` 排在 Expert 0 **之前**导致全部有效专家区间整体右移？为什么即便修复了排序，未清零的 `mm1_out` 尾部行仍会污染 CUTLASS FP8 与 FusedHummingMoE 的 per-tensor 动态激活缩放因子？
2. **vLLM ROCm RDNA (`gfx1151`) W4A16 Prefill 单指令打包 FP16 反量化 (`vllm-project/vllm@96bbaa300fe4`, `#60644`)**：在 RDNA3.5 (`gfx1151`) 上执行 Triton W4A16 prefill GEMM 时，如何用一条内联 AMDGCN 指令 `v_and_or_b32 $0, $1, $2, 0x64006400` 将两个 INT4 半字节同时解包为两个 `1024.0 + nibble` 的打包 FP16 通道，并将偏置 `1024.0` 折叠进零点减数以直接发射 `v_pk_fma_f16`？
3. **vLLM ROCm `gfx942` Kimi-K3 MXFP4 专家加载期 INT4 重量化校验 (`vllm-project/vllm@2c58be22b68d`, `#63131`) 与投机解码 Drafter LoRA 校验豁免 (`@58301f6cb9b1`, `#63186`)**：为什么 `AiterMxfp4Experts` 的 fail-closed 检查在 MI300X (`gfx942`) 上拒绝了 Kimi-K3？为什么投机解码（Speculative Decoding）在二次校验草稿模型（如 `DeepSeekMTPModel`）配置时会错误拦截主模型的 LoRA 请求？
4. **SGLang Inkling `enable_lm_head_output_fp32=True` 保持 BF16 运算数与 FP32 输出防贪心翻转 (`sgl-project/sglang@855c092deb93`, `#45335`)**：当两个候选 token 的真实内积分别为 $1 + 2^{-10}$ 与 $1 + 2^{-9}$ 时，为何两者的输入激活与 LM-Head 权重在 BF16 下均**完全精确**，但默认的 BF16 输出投影却将两者同时舍入为 `1.0` 并令 `argmax` 从 `1` 翻转为 `0`？如何在不把 LM-Head 权重提升为 FP32（零额外显存占用）的前提下消除该翻转？
5. **SGLang AMD Quark Block-FP8 DeepSeek 权重缩放名称映射与层级 `weight_block_size` 优先级 (`sgl-project/sglang@9d7ad308bdbb`, `#40776`) 及 TensorRT-LLM CBTS Dataclass 静态上界 (`NVIDIA/TensorRT-LLM@dbc875d517fb` `#20411`, `@461183a90a04` `#20883`)**：如何让 DeepSeek 权重加载器将 Quark 的 `weight_scale` 安全映射到 `weight_scale_inv` 而不误伤 MXFP4 专家的 `w13_weight_scale`？

---

## 2. 核心实现路径与源码级机制拆解

### 2.1 vLLM `moe_permute` 非法 `topk_ids` 哨兵重映射与动态量化行上界 (`#59717`)

在 `csrc/moe/moe_permute_unpermute_op.cu`、`csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl`、`cutlass_moe.py` 与 `fused_humming_moe.py` 中，vLLM 修复了三个紧密耦合的填充路由 Bug：

1. **无 `expert_map` 时的有符号基数排序错位**：
   - 旧代码仅在 `expert_map.has_value()`（开启 EP）时启动 `remap_expert_ids` 将非本地/无效路由映射为哨兵值 `n_expert`。当 `expert_map` 为空时，原始有符号 `int32` 的 `topk_ids` 直接送入 `cub::DeviceRadixSort::SortPairs`。
   - 由于有符号整数 `-1 < 0`，所有填充路由（`topk_ids == -1`）被排到了 Expert 0 **之前**（占据排序后缓冲区的最前端行 `0 .. n_pad_routes - 1`），而直方图 `compute_and_sum_expert_offsets` 忽略 `< 0` 的键，导致 `expert_offsets[0] = 0` 实际指向了 `-1` 填充行而非 Expert 0 的真实 token！
   - **修复**：无论是否传入 `expert_map`，统一调用 `remap_expert_ids<<<...>>>(topk_ids_ptr, remapped_topk_ids.data_ptr<int32_t>(), expert_map_ptr, n_tokens * topk, n_expert)`；当 `expert_map == nullptr` 时，对满足 `uint32_t(id) < uint32_t(n_expert)` 的合法路由保留 `id`，其余一律重写为正哨兵 `n_expert`，保证所有无效路由被排在有效行末尾 `expert_offsets[n_expert]` 之后。
2. **CUTLASS FP8 激活工作区污染 (`cutlass_moe.py`)**：
   - 在 `run_cutlass_moe_fp8` 中，当 `per_act_token=False`（使用 per-tensor 动态激活缩放）时，第二步量化会归约 `mm1_out` 的**全部** $M \times \text{topk}$ 行以计算全局 `amax`。旧代码仅在 `expert_map is not None or use_batched_format` 时执行 `mm1_out.fill_(0)`；当无 `expert_map` 但含有 `topk_ids == -1` 填充行时，grouped GEMM 不会写入排在末尾的无效行，未初始化工作区中的陈旧脏数据直接污染 per-tensor 缩放因子（仿真显示若脏数据含 `64.0` 而有效激活为 `1.25`，缩放因子被放大 **`51.2x`**，导致有效输出下溢）。修复后只要 `not per_act_token` 即无条件执行 `mm1_out.fill_(0)`。
3. **FusedHummingMoE 动态输入量化上界 (`fused_humming_moe.py`)**：
   - 将 `num_valid_tokens = expert_offsets[-1:] if expert_map is not None else None` 简化为无条件 `num_valid_tokens = expert_offsets[-1:]`，使动态输入量化始终只处理前 `expert_offsets[-1]` 条有效展开行。

### 2.2 vLLM ROCm RDNA (`gfx1151`) W4A16 打包 FP16 反量化 (`#60644`)

在 `vllm/model_executor/layers/quantization/kernels/mixed_precision/rdna_hybrid_w4a16.py` 中，针对 RDNA3.5 (`gfx1151`) 的 Triton W4A16 prefill 内核引入了 `_int4_pair_to_fp16x2`：

- **IEEE-754 FP16 指数偏置代数技巧 (`0x64006400`)**：
  - FP16 的 `1024.0` 编码为 `0x6400`（符号 `0`，偏置指数 $25 = 10 + 15$，尾数 `0`），在此区间内最低位 1 ULP 恰好等于 $2^{10 - 10} = 1.0$。
  - 因此，对于任意 4-bit 无符号半字节 $q \in [0, 15]$，位模式 `0x6400 | q` 的精确 FP16 数值即为：
    $$
    \operatorname{fp16\_bits}(0\text{x}6400 \mid q) = 1024.0 + q
    $$
  - 内联汇编 `tl.inline_asm_elementwise("v_and_or_b32 $0, $1, $2, 0x64006400", "=v,v,s", [pair_word, 0x000F000F], ...)` 在**单条 32 位 VALU 指令**内同时提取低 16 位与高 16 位中的两个 INT4 半字节并拼入 `0x6400` 指数头，生成两个打包的 FP16 值 $(1024 + q_{\text{lo}}, 1024 + q_{\text{hi}})$。
- **零点与偏置融合为单条 `v_pk_fma_f16`**：
  - 将 `1024.0` 折叠进每组的零点减数 $\text{sub}$：
    $$
    w = (b_{\text{raw}} - (1024 + z)) \cdot s = b_{\text{raw}} \cdot s - \underbrace{(1024 + z) \cdot s}_{\text{sub}}
    $$
    其中对称量化 $z = 8 \implies \text{sub} = 1032.0 \cdot s$；非对称量化 $\text{sub} = (1024.0 + \text{zp}) \cdot s$。相邻通道配对后由编译器直接合成为双通道 `v_pk_fma_f16`，并在 `gfx1151` 上配置 `BLOCK_N=64, BLOCK_K=64, GROUP_SIZE_M=4` 与 `BLOCK_M=16/32/64` 分档。

### 2.3 ROCm `gfx942` Kimi-K3 INT4 专家重量化校验 (`#63131`) 与 Drafter LoRA 豁免 (`#63186`)

1. **`gfx942` Kimi-K3 MXFP4 专家加载期 INT4 重量化 (`#63131`)**：
   - Kimi-K3 的原生权重为 MXFP4，但在不支持原生 MXFP4 指令的 MI300X (`gfx942`) 上，模型加载器在加载期将其重量化为静态 INT4（量化键 `(kInt4Static32, None)`）。
   - `AiterMxfp4Experts._supports_quant_scheme` 与 `make_mxfp4_moe_quant_config` 新增对 `(kInt4Static32, None)` 在 `on_gfx942()` 上的显式放行（同时保持原生 `(kMxfp4Static, *)` 仅限 `on_gfx950() and not on_gfx1250()`）。
2. **投机解码 Drafter LoRA 校验豁免 (`#63186`)**：
   - `LoRAConfig.verify_with_model_config` 在主模型通过校验后，会被草稿模型配置（如 `DeepSeekMTPModel`，其 `runner_type == "draft"` 且 `is_lora_supported == False`）再次调用。由于草稿模型在投机解码中不挂载 LoRA 适配器，将守卫条件更新为 `if model_config.runner_type != "draft" and not model_config.is_lora_supported: raise ValueError(...)`。

### 2.4 SGLang Inkling `enable_lm_head_output_fp32=True` 防贪心翻转 (`#45335`)

在 `python/sglang/srt/configs/inkling.py` 与 `python/sglang/srt/layers/logits_processor.py` 中：

- **BF16 输出舍入导致的 Greedy Argmax 翻转数学证明**：
  - 设隐藏状态 $h = [1.0, 1.0]$（BF16 精确表示），LM-Head 两行权重分别为 $w_0 = [1.0, 2^{-10}]$ 与 $w_1 = [1.0, 2^{-9}]$（$2^{-10} = 1/1024$ 与 $2^{-9} = 1/512$ 均为 2 的整数次幂，在 BF16 中**100% 精确无损**）。
  - Tensor Core 在内部使用 FP32 累加器计算出精确内积：
    $$
    z_0 = 1.0 + 2^{-10} = 1.0009765625, \quad z_1 = 1.0 + 2^{-9} = 1.001953125 \implies z_1 > z_0 \ (\text{真實 argmax} = 1)
    $$
  - 然而在 $[1.0, 2.0)$ 区间内，BF16 的尾数仅有 7 位，1 ULP $= 2^{-7} = 1/128 = 0.0078125$。若将 `torch.mm` 的输出张量存为 BF16，$z_0$ 与 $z_1$ 距离 `1.0` 的差值均小于 $0.5 \text{ ULP} = 2^{-8} = 1/256$，因此两者在写回显存时**双双被舍入为 `1.0`**，平局决胜（tie-breaking）选出索引 `0`，导致贪心解码结果彻底翻转！
- **零权重膨胀的修复方案**：
  - 旧标志 `enable_fp32_lm_head` 会把整个 `lm_head.weight` 与 `hidden_states` 提升为 FP32（令大词表 LM-Head 权重显存翻倍）。
  - 新增 `enable_lm_head_output_fp32: bool = True`（映射为 `LogitsProcessor.use_fp32_lm_head_output`）：保持 `hidden_states` 与 `lm_head.weight` 为 **BF16**，仅在 `use_mm_out_dtype`（CUDA 可用）时调用 `torch.mm(hidden_states, lm_head.weight.T, out_dtype=torch.float32)`，直接将 cuBLAS FP32 累加器结果写出为 FP32 logits，在零权重显存增加下完整保留 $z_1 > z_0$。

### 2.5 SGLang Quark Block-FP8 DeepSeek 加载器 (`#40776`) 与 TensorRT-LLM CBTS Dataclass 上界 (`#20411`)

1. **SGLang AMD Quark Block-FP8 加载修复 (`#40776`)**：
   - Quark 导出的 DeepSeek block-FP8 检查点使用 `*.weight_scale` 命名，而 SGLang 内部 block-FP8 参数注册名为 `*.weight_scale_inv`；同时混合量化模型中 MXFP4 专家使用的是合法注册的 `*.w13_weight_scale`。
   - `_resolve_quark_fp8_block_scale_name(name, params_dict)` 仅当 `name.endswith("weight_scale") and name not in params_dict` 且替换后缀后的 `weight_scale_inv` 存在于 `params_dict` 时才执行重命名，并补齐了 `_load_fused_indexer_wk` 与 `fused_qkv_a_proj_with_mqa` 的缩放加载路径。
   - `_resolve_layer_weight_block_size(layer, quant_config)` 优先读取 `layer.quant_method.weight_block_size`（例如层级的 `[64, 128]`），避免被顶层配置摘要 `linear_fp8_config.weight_block_size = [128, 128]` 覆盖。
2. **TensorRT-LLM CBTS Top-Level Dataclass 变更边界推导 (`#20411`) 与 Executor 存活测试 (`#20883`)**：
   - 在 `jenkins/scripts/cbts/cbts/coverage/selection/` 中，CBTS 静态 AST 变更选择器支持对无基类、无 metaclass、使用内建类型注解与不可变字面量默认值的顶层 `@dataclass` / `@dataclass(frozen=...)` 通过消费该类绑定的局部函数闭包来界定受影响测试集，避免回退至全量重跑；`#20883` 为 `test_forward_error_survival.py` 补齐 `kv_cache_retention_config=None` 与 `_is_kv_manager_v2=False`。

---

## 3. 本项目确定性离线实验与验证 (`own-offline`)

实验脚本：`experiments/vllm-moe-permute-rdna-w4a16-sglang-inkling-trtllm/run.py`（结果固化于 `summary.json`）：

| 验证子项 | 关键输入 / 参数 | 实测 / 仿真输出 | 架构结论 |
|---|---|---|---|
| vLLM `moe_permute` 无 `expert_map` 填充路由 | 3 有效 token + 2 填充 token (`topk=4, E=8`) | 旧版首键 `-1`（偏移错位），新版首键 `0`、尾部 8 个哨兵 `8`；未清零工作区缩放膨胀 `51.2x` | 必须无条件重映射非法 `topk_ids` 并清零/截断动态量化行 |
| ROCm RDNA (`gfx1151`) W4A16 `v_and_or_b32` | 对称 `(3, 14), s=0.0625`；非对称 `(5, 11), s=0.125, zp=6` | 打包字 `0x640E6403` / `0x640B6405`，反量化误差 `0.0` | 单指令 `0x64006400` 掩码或运算精确等价于 `(q - zp) * s` |
| SGLang Inkling `enable_lm_head_output_fp32` | `w=[[1, 1/1024], [1, 1/512]]`, `h=[1, 1]` | BF16 输出 `[1.0, 1.0]` (`argmax=0`) vs FP32 输出 `[1.0009766, 1.0019531]` (`argmax=1`) | `out_dtype=torch.float32` 在不增加权重显存下根除 BF16 尾数截断平局翻转 |
| SGLang Quark Block-FP8 加载器 | `qkv_a.weight_scale` + `experts.w13_weight_scale`, 层级 `[64, 128]` | QKV 映射至 `weight_scale_inv`，MXFP4 保持 `w13_weight_scale`，块大小取 `[64, 128]` | 兼顾 Quark FP8 与 MXFP4 混合量化检查点兼容性 |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **位级计算练习**：在 RDNA W4A16 `_int4_pair_to_fp16x2` 中，为什么常量必须选取 `0x6400` ($1024.0$) 而不是 `0x3C00` ($1.0$)？
   - *解答*：FP16 有 10 位尾数。在 `0x3C00` ($1.0$) 处，最低位 bit 0 代表 $2^{-10} = 1/1024$，拼入 4-bit 整数 $q$ 得到的是 $1.0 + q / 1024$；若要让最低位 bit 0 的步长恰好为 $1.0$，偏置指数必须再加 $10$（即指数域从 `15` 变为 $15 + 10 = 25 = 0\text{x}19$，左移 10 位后正是 `0x6400` = $1024.0$）。
2. **可证伪假设**：在单卡部署（`EP=1`, `expert_map=None`）下，FusedMoE 的 `topk_ids` 是否永远不会出现负数？
   - *证伪条件*：当启用 CUDA Graph 批次填充（padding tokens）、Layer-wise Expert 跳过（skip connection routing）或 Top-K 概率阈值截断时，单卡模式同样会产生 `topk_ids = -1`，若跳过 `remap_expert_ids` 将立即导致 Expert 0 及其后所有专家的 token 区间错位。
