# TensorRT-LLM PrimsTS SageAttention 块稀疏 VSA/SOL、INT8-woq MoE 64 行对齐填充、vLLM LoRA MoE 权重热重载 `_lora_context` 重绑与 SGLang Rust PD Bootstrap 端口路由深度解析

> **语言 / Language**: [中文](trtllm-primsts-sage-vsa-sol-woq-align64-vllm-lora-sglang.md) | [English](trtllm-primsts-sage-vsa-sol-woq-align64-vllm-lora-sglang.en.md)  
> **Phase**: 71 (`priority: 80`, Track B — LLM Serving, Quantization & Disaggregated PD)  
> **核心证据等级**: `implementation` (固定 commit 源码审计) + `author-benchmark` (作者 B200 SageAttention INT8/FP8 精度评测) + `own-offline` (本项目确定性标准库仿真 `experiments/trtllm-primsts-sage-vsa-sol-woq-align64-vllm-lora-sglang/run.py`)

---

## 1. 架构背景与核心工程问题

在 TensorRT-LLM、vLLM 与 SGLang 的最新生产服务栈演进中，视频生成/长上下文块稀疏量化注意力、MoE 张量并行分片对齐、LoRA 权重热重载与 PD 分离控制面暴露出四个关键工程问题：

1. **TensorRT-LLM PrimsTS SageAttention 块稀疏注意力（VSA / SOL）的缩放因子布局、Proxy 路由量化与 Blackwell TMA 越界修复 (`NVIDIA/TensorRT-LLM@791a402fc8d5`, `#20224`)**：在 VisualGen 视频稀疏注意力（VSA）与在线稀疏注意力（SOL）中，如何将 `smooth_k` / `smooth_v` 与 sub-tile `k_block_size` $\in \{1, 4, 16\}$ 的 INT8/FP8 QK 量化融合进 Blackwell (`SM100`/`SM103`) PrimsTS 块稀疏内核？为什么当连续 Q 前缀不足 `128 KiB` 时 Blackwell TMA 描述符会触发越界错误？
2. **TensorRT-LLM INT8 Weight-Only (`W8A16`) FusedMoE 在张量并行（TP）分片下的 64 行 Tile 对齐崩溃 (`NVIDIA/TensorRT-LLM@64ef54ce1249`, `#16198`) 与 Model Express (`MX`) `Phi3ForCausalLM` 变换后 ABI 资格校验 (`@b78033a95183`, `#19997`)**：当 `moe_intermediate_size` 经 TP 切分后（如 `1856 / 2 = 928` 或 `200`）不是混合精度 GEMM 预处理器 Tile 行数 `64` 的整数倍时，为什么 `preprocess_weights_for_mixed_gemm` 会直接崩溃？为什么 `fc1`/`fc2` 形状检查必须放在 `makeActivationParams` 之后？
3. **vLLM `FusedMoEModularMethod` 在权重热重载 (`reload_weights`) 时丢失 `_lora_context` (`vllm-project/vllm@7dc44a00161a`, `#63319`) 及 ROCm MI355 AITER `W4A16` MXFP4 `SWIGLUOAI_UNINTERLEAVE` 激活验证 (`@be03d19750cc` `#62907`, `@f869d6478838` `#63283`)**：当 `LoRAFusedMoEModularMethod` 包装量化 MoE 层（如 `Qwen3-30B-A3B-quantized.w4a16`）并在 RL/多租户场景调用 `reload_weights` 时，底层 `old_quant_method` 会重建 `moe_kernel`，为什么这会导致 LoRA 适配器静默失效且继续读取旧权重？
4. **SGLang Rust HTTP Server 在 PD 分离下的 Prefill Bootstrap 端口误合并 (`sgl-project/sglang@0d592c475bca`, `#46066`) 与 `/v1/responses` Muse 命名空间工具解析 (`@6a319e7b3446`, `#44685`)**：当启用 `SGLANG_RUST_SERVER=1` 时，若传输后端（如自定义 RDMA/Mooncake/NIXL 引导器）自带独立的 `BaseKVBootstrapServer` 监听 `8998` 端口，无条件把 `disaggregation_bootstrap_port` 覆写为 HTTP `--port (31000)` 为什么会直接打断 PD 握手？

---

## 2. 核心实现路径与源码级机制拆解

### 2.1 TensorRT-LLM PrimsTS SageAttention 块稀疏 VSA / SOL 与 Blackwell TMA/LDTM 适配 (`#20224`)

在 `tensorrt_llm/_torch/visual_gen/jit_kernels/flash_attention/cute/`（`sage.py`, `sage_quant.py`, `block_sparse_attention.py`）与 `cpp/tensorrt_llm/kernels/cutlass_kernels/primsts/` 中，打通了 Blackwell (`SM100`/`SM103`) 上的 SageAttention 块稀疏 VSA 与 SOL 流水线：

- **`sage_quantize` 扁平化带哨兵缩放因子布局**：
  - 对于总 Token 数为 $T$、批次大小为 $B$、量化块大小为 $b_{\text{blk}} \in \{1, 4, 16\}$ 的压缩变长序列，`q_scale` 与 `k_scale` 采用形状为 $[H, \lceil T / b_{\text{blk}} \rceil + B - 1]$ 的一维扁平布局（相邻序列之间插入 1 个对齐填充槽位），序列 $b$ 的起始槽位精确定义为：
    $$
    \text{slot}_b = \lfloor \text{cu\_seqlens}[b] / b_{\text{blk}} \rfloor + b
    $$
  - `v_scale`（及 `smooth_v=True` 时的 `v_mean`、`smooth_k=True` 时的 `k_mean`）具有形状 $[B, H_{KV}, D]$；为防止长序列下 FP32 累加精度漂移，新增 `int64_t` 工作区切片 `sageKMeanSum` 用于精确累加每个序列的 K 均值。
- **Proxy Routes 摘要复用与 `_PrepareSageScales` + PDL 重叠**：
  - 当 `use_proxy_routes=True`（VSA/SOL 两阶段路由）时，`q_summary` 与 `k_summary`（每 KV 块 1 个摘要 Token，共 $B \times N_{\text{blocks}}$ 个）使用 $b_{\text{blk}}=1$ 独立量化为 `q_summary_scale` / `k_summary_scale`，但直接复用主注意力的 `v_scale` 与 `k_mean`。
  - `_PrepareSageScales` 在主注意力前将 `q_scale`/`k_scale` 预打包为 `float4`（`_SAGE_ATOM_TOKENS = 64`, `_SAGE_ATOM_MIN_WORDS = 2`），随后路由准备内核通过 `use_pdl=True`（Programmatic Dependent Launch）与缩放因子准备内核流水重叠。
- **Blackwell `SM103`/`SM107` `LDTM.STAT` 与 TMA 描述符 Bit-85 越界规避**：
  - 在 `SM103`/`SM107` 上启用硬件归约加载指令 `tcgen05.ld.red.sync.aligned.32x32b.x{num}.f32.max` (`LDTM.STAT`) 计算在线行最大值；
  - 当连续 Q 前缀字节数小于 `128 KiB`（在 BF16 $H=16, D=128$ 下即 `< 32` 行）时，Blackwell TMA 硬件存在越界读取缺陷，内核显式清除 TMA 描述符第 1 字的第 21 位（即全局 Bit 85：`desc.w1 &= ~(1u << 21)`）。
- **B200 实测有损精度上界 (`author-benchmark`, 对比未量化稀疏 BF16 参考)**：
  - **INT8 QK (`k_block_size in {1, 4, 16}`)**: 最差 `relative_l2 = 0.03618`, `cosine = 0.999403`, `max_abs = 0.140625`；
  - **FP8 E4M3 QK (`k_block_size in {1, 4, 16}`)**: 最差 `relative_l2 = 0.04979`, `cosine = 0.998761`, `max_abs = 0.167969`。

### 2.2 TensorRT-LLM INT8-woq MoE 64 行 Tile 填充 (`#16198`) 与 Model Express Phi-3 资格校验 (`#19997`)

1. **INT8 Weight-Only (`W8A16`) MoE 分片 64 行对齐填充 (`NVIDIA/TensorRT-LLM@64ef54ce1249`, `#16198`)**：
   - 在 `tensorrt_llm/_torch/moe/fused_moe/quantization.py` 中，CUTLASS mixed-GEMM 权重预处理器要求 K/N 维度的每个分片行数为 `ALIGN = 128 * 8 // BITS_PER_ELT_A (16) = 64` 的整数倍。当模型 `moe_intermediate_size = 1856` 在 `tp_size = 2` 下切分为 `inter_per_partition = 928`（`928 % 64 = 32 != 0`），或单卡 `moe_intermediate_size = 200`（`200 % 64 = 8 != 0`）时，旧代码直接触发预处理器断言崩溃。
   - **修复方案**：定义 `inter_padded = ((inter_size + 63) // 64) * 64`（如 `928 -> 960`，填充 `32` 行；`200 -> 256`，填充 `56` 行）。在拼接 `w3` 与 `w1` 之前先对各自尾维做零填充（保证门控激活下 `w3` 与 `w1` 分界点对齐在 `inter_padded` 处），并对 `w2` 的输入维度做相同零填充（因对应输入恒由 `fc1` 零权重行产生，零填充对 `fc2` 输出数学无损）。
   - **后置 `fc1`/`fc2` 形状校验**：在 `fused_moe_cute_dsl.py` 中，将 `fc1_weight_experts.size(2) == fc2_weight_experts.size(1) * expand_ratio` 的 `TORCH_CHECK` 移至 `makeActivationParams` 之后——因为 `makeActivationParams` 会将带偏置门控激活提升为 `SwigluBias`（`expand_ratio: 1 -> 2`），前置检查会误判形状不匹配。
2. **Model Express (`MX`) `Phi3ForCausalLM` 变换后 ABI 校验 (`@b78033a95183`, `#19997`)**：
   - 新增 `phi3-for-causal-lm-bf16-target-v1`（ABI 标识 `trtllm-phi3-dense-target-layout-v1`），在 `TP=1, 2` 下验证预融合 `qkv_proj` 与 `gate_up_proj` 权重的跨实例直接传输，并在指纹校验中拦截 LongRoPE `rope_scaling`、`tie_word_embeddings` 或不一致的 `max_seq_len`。

### 2.3 vLLM LoRA MoE 权重热重载 `_lora_context` 重绑 (`#63319`) 与 ROCm MI355 MXFP4/FP8 验证 (`#62907`, `#63283`)

1. **`LoRAFusedMoEModularMethod.process_weights_after_loading` 修复 (`vllm-project/vllm@7dc44a00161a`, `#63319`)**：
   - 在 `vllm/lora/layers/fused_moe.py` 中，`LoRAFusedMoEModularMethod` 继承自 `FusedMoEModularMethod` 并持有被包装的量化方法 `self.old_quant_method`（如 Marlin/GPTQ `W4A16`）。
   - 当调用 `reload_weights` 热更新基础模型权重时，`FusedMoEModularMethod.process_weights_after_loading` 原本未委托给 `self.old_quant_method.process_weights_after_loading(layer)`。更隐蔽的是，`old_quant_method.process_weights_after_loading` 执行完毕后会构建一个全新的 `new_kernel = self.old_quant_method.moe_kernel`，导致旧的 `self.moe_kernel` 仍指向重载前的权重且新内核丢失了 `_lora_context`。
   - **四步修复闭环**：(1) 调用 `self.old_quant_method.process_weights_after_loading(layer)`；(2) 提取旧 `fused_experts._lora_context` 并通过 `set_lora_context(lora_ctx)` 注入 `new_kernel.fused_experts` 与 `new_kernel.prepare_finalize`；(3) 更新 `self.moe_kernel.impl = new_kernel.impl`；(4) 重新绑定 `self.old_quant_method.moe_kernel = self.moe_kernel`。
2. **ROCm MI355 AITER `W4A16` MXFP4 `SWIGLUOAI_UNINTERLEAVE` (`@be03d19750cc`, `#62907`) 与在线 FP8 Batch-Invariance (`@f869d6478838`, `#63283`)**：
   - 验证了 MiniMax-M3 在 MI355 AITER W4A16 MXFP4 下的非交错门控激活 `SWIGLUOAI_UNINTERLEAVE`（AITER Triton 内核强制 `interleave=True` 步进 `2*i` / `2*i+1`，パラメータ `limit=7.0, alpha=1.702, beta=1.0`）与 `SITU`（`beta=4.0, linear_beta=25.0`），并启用 MI300/MI355 在线 FP8 MoE 的 `bs=1` vs `bs=N` Bitwise Batch-Invariance 校验。

### 2.4 SGLang Rust PD Bootstrap 端口保留 (`#46066`) 与 `/v1/responses` Muse 命名空间工具 (`#44685`)

1. **Rust HTTP Server PD 分离 Bootstrap 端口判定 (`sgl-project/sglang@0d592c475bca`, `#46066`)**：
   - 在 `python/sglang/srt/server_args.py` 与 `disaggregation/launch_lb.py` 中，新增 `rust_server_serves_kv_bootstrap(transfer_backend)`（检查 `get_kv_bootstrap_server_class(transfer_backend)` 是否为 `CommonKVBootstrapServer` 的子类）。
   - **修复前**：只要开启 `SGLANG_RUST_SERVER`，Prefill 节点就会无条件把默认 `disaggregation_bootstrap_port=8998` 别名为 HTTP `--port (31000)`，或对自定义 `--disaggregation-bootstrap-port 33000` 抛出冲突异常——导致自带独立 `BaseKVBootstrapServer` 的传输后端无法在预期端口握手。
   - **修复后**：仅当后端确实使用内置 HTTP 注册表（`CommonKVBootstrapServer`）时才合并至 `api_port (31000)`；若后端运行自己的 `BaseKVBootstrapServer`，则原样保留配置的 `8998` 或 `33000` 端口。
2. **Muse `/v1/responses` 命名空间工具与 `defer_loading` (`@6a319e7b3446`, `#44685`)**：
   - 在 `entrypoints/openai/serving_responses.py` 中，将 `type="namespace"` 的工具树同时保存为兩份视图：保留完整嵌套结构与 `defer_loading: True` 标记存入 `req._template_tools` 供 Jinja 聊天模板渲染，同时将子工具扁平化为 `<namespace>.<function>`（如 `browser.search`）供 `FunctionCallParser` 匹配解析。

---

## 3. 本项目确定性离线实验与验证 (`own-offline`)

实验脚本：`experiments/trtllm-primsts-sage-vsa-sol-woq-align64-vllm-lora-sglang/run.py`（结果固化于 `summary.json`，由 `tests/test_phase71_experiments.py` 校验）：

| 验证子项 | 关键输入 / 参数 | 实测 / 仿真输出 | 架构结论 |
|---|---|---|---|
| TRT-LLM PrimsTS SageAttention VSA/SOL | $B=2, S_Q=S_{KV}=1024, H_Q=16, H_{KV}=2, b_K=16$ | `q_scale=[16, 2049]`, `k_scale=[2, 129]`, `k_summary_scale=[2, 33]`, Seq1 K 槽位 `65` | 精确对齐序列哨兵间隔与 Proxy 摘要复用，B200 INT8 余弦相似度 `0.999403` |
| TRT-LLM INT8-woq MoE 64 行填充 | `inter=1856, TP=2` (`928`/rank, gated) | `inter_padded=960` (`+32` 行), `w3_w1=[8, 2048, 1920]`, `w2=[8, 960, 2048]` | 消除非 64 倍数 TP 分片崩溃且后置 `fc1/fc2` 校验兼容 `SwigluBias` |
| vLLM LoRA MoE 热重载 & MI355 MXFP4 | `LoRAFusedMoEModularMethod.reload_weights` | 委托 `old_quant_method` 并重绑 `_lora_context` 与 `moe_kernel.impl` | 修复量化 MoE 层热重载后 LoRA 失效与旧权重残留 |
| SGLang Rust PD Bootstrap 端口路由 | `api_port=31000`, `bootstrap_port=8998 / 33000` | HTTP 注册表：`8998->31000`, `33000->ValueError`；独立后端：保留 `8998 / 33000` | 防止 Rust HTTP Server 误劫持独立 KV 传输后端的 Bootstrap 端口 |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **计算练习**：在 TensorRT-LLM PrimsTS SageAttention 中，设批次 $B=3$，各序列 KV 长度为 `cu_seqlens = [0, 512, 1536, 2048]`，`num_kv_heads = 4`，`k_block_size = 16`。求扁平 `k_scale` 张量的形状以及第 2 个序列（0-indexed `b=2`）在 `k_scale` 中的起始槽位索引。
   - *解答*：总 Token 数 $T = 2048$，扁平长度 $= \lceil 2048 / 16 \rceil + 3 - 1 = 128 + 2 = 130$，故 `k_scale` 形状为 `[4, 130]`。序列 `b=2` 的起始槽位为 $\lfloor \text{cu\_seqlens}[2] / 16 \rfloor + 2 = \lfloor 1536 / 16 \rfloor + 2 = 96 + 2 = 98$。
2. **可证伪假设**：在 INT8 Weight-Only (`W8A16`) 门控 MoE 层中，对拼接后的 `w3_w1` 仅在末尾一次性补零至 `2 * inter_padded` 是否与分别对 `w3` 和 `w1` 补零至 `inter_padded` 再拼接等价？
   - *证伪条件*：不等价！门控激活（SwiGLU / GeGLU）将 `fc1` 输出沿最后一维对半切分为 `[0 : inter_padded]` (`w3`) 与 `[inter_padded : 2 * inter_padded]` (`w1`)；若仅在拼接后的末尾补零，`w1` 的前部会混入 `w3` 的尾部权重，导致激活乘积完全错位。
