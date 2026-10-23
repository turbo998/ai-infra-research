# TensorRT-LLM KDA 融合 Prefill 短卷积与 GVR v2 Top-K、SGLang GLM-5.3 Flash Padded CP BCG 与 MegaGate 掩码复用、vLLM ROCm gfx942 32 位偏移修复深度解析

> **语言 / Language**: [中文](sglang-glm53-cp-bcg-trtllm-kda-conv-vllm-gfx942.md) | [English](sglang-glm53-cp-bcg-trtllm-kda-conv-vllm-gfx942.en.md)
> **对应问题**: `sglang-glm53-cp-bcg-trtllm-kda-conv-vllm-gfx942` (`priority: 76`, Phase 67 Track B)
> **证据等级**: `implementation` + `own-offline`

---

## 1. 核心架构问题与背景

在混合线性注意力（KDA / Gated DeltaNet）、多模态 MoE 门控与 AMD Instinct (`gfx942` / `gfx950`) 解码算子演进中，推理引擎面临三大核心问题：

1. **TensorRT-LLM KDA Prefill 短卷积的快速 SiLU 灾难性抵消与 GVR Top-K 采样分派**：
   - 在 CuTe-DSL 编写的 KDA Prefill 融合短卷积 (`fused_prefill_conv.py`) 中，利用硬件 `tanh.approx.f32` 计算快速 SiLU $\text{silu}(x) = h + h \cdot \tanh(h)$ ($h = x/2$) 在 $h \ge -2$ 时相对误差 `< 1.3%`；但当 $h \to -\infty$ 时，$\tanh(h) \to -1$，$h + h \cdot \tanh(h)$ 发生灾难性有效数字抵消（在 $h = -6$ 时最坏相对误差超 `3900%`）。
   - 同时，GVR Decode Top-K 自采样算子 (`gvr_topk_decode_self_sampling.py`) 在宽词表/长序列下的 SMEM 字节计算使用了未按实际阶梯覆盖的 `NBSEL`，且确定性路径 (`det=True`) 在 `VPT > 8` 时存在 32 位打包位掩码溢出风险。
2. **SGLang GLM-5.3 Flash Padded CP BCG 的静态/动态 Token 混淆与 DSV4.1 MegaGate 重复掩码构造**：
   - 在 Context Parallel (`CP`) + Batch-level CUDA Graph (`BCG`) 模式下，当捕获桶大小 (`static_num_tokens`) 大于真实有效 Token 数时，若对 KDA 重排、Attention Q 切片与 KV Cache 写入使用同一个 Token 计数，会导致 KDA 序列顺序错位、填充尾行污染归约或 KV Cache 索引截断。
   - 在 DeepSeek-V4.1 的 60 层 MoE 前向传播中，`MegaGate` 每层都重复执行 `torch.arange(num_tokens) < bound` 与 `image_input_ids == image_token_id` 构造相同的二维有效/图像行掩码（每步 120 次小 Kernel 启动）。
3. **vLLM ROCm `gfx942` (MI300X) Triton Paged Attention 寄存器溢出误编译与 `gfx950` FlyDSL 代码分叉**：
   - 在 `gfx942` 上，将 `kernel_paged_attention_2d` 的 `output_stride_1`（等于 `head_size`）标注为 `tl.int64`，触发了 LLVM AMDGPU 后端在高寄存器压力下的寄存器溢出 Bug：64 位乘加中的 `output_stride_1` 被无关的溢出 SGPR 覆盖，导致输出部分 Head 未被写入而残留 `NaN`。

---

## 2. 上游源码实现与关键机制分析

### 2.1 TensorRT-LLM：KDA 融合 Prefill 短卷积 (`#20653`) 与 GVR v2 Top-K (`#20106`)

1. **KDA 融合 Prefill 短卷积 + Warp-Vote 守护的混合 SiLU (`222676c8d7bd`)**：
   - `fused_prefill_conv.py` 将 4-tap 深度可分离短卷积、SiLU 激活与按 `state_indices[seq_idx]` 寻址的循环状态池更新（支持 `slot == -1` 跳过写回与 `-pool <= slot < -1` 负索引回绕）融合在单个 CuTe-DSL Kernel 中，并删除了 `7,214` 行旧版 KDA Kernel（`akk_inverse.py`、`fused_k123.py`、`fused_k1234.py`、`k4_persistent.py`），统一迁移至 vendored FlashInfer `kda_chunked_bt16`。
   - **混合精度 SiLU 守护**：对每 8 个 Token 的寄存器组先计算 `h + h * tanh.approx(h)`，再通过 **1 次 Warp Vote** 检测该组是否存在 `min(h) < -2.0` (`x < -4.0`)；仅对触发守护的分支回退执行保次正规数的精确公式 `x * e / (1 + e)`（其中 $e = \exp(-|x|)$）。
2. **GVR v2 Top-K Decode 自采样向量化与阶梯扩展 (`4ccdca1e8492`)**：
   - `scan_cross_w` 使用 `cute.autovec_copy` 按 `num_bits_per_copy = 32 * BPT`（`BPT=2` 为 64-bit，`BPT=4` 为 128-bit）向量化搬运 SMEM 直方图。
   - 将寄存器路径跨桶阈值 `QC` 从 `96` 放宽至 `160`，新增 `short_wide` (`k == 2048, 512 < n4 <= 1024`) 与 `long_register` (`n4 > 2048`) 阶梯，修正 `IMGOFF = NBH` 与 `smem = (NBH + 2 * CMP) * 4`（在 `NBH=1024, NBSEL=2048` 时每个 CTA 节省 `4,096 B` SMEM），并在 `det=True and VPT > 8` 时自动执行 `BLK *= 2, VPT = (VPT + 1) // 2` 防止 32 位位掩码溢出。

### 2.2 SGLang：GLM-5.3 Flash Padded CP BCG (`#43774`)、DSV4.1 MegaGate 缓存 (`#46071`) 与 Granian Keep-Alive (`#45760`)

1. **GLM-5.3 Flash `InterleaveBcgAdapter` 三分离 Token 计数 (`ea226cf78267`)**：
   - `metadata.graph_num_tokens = static_num_tokens`：供 CUDA Graph 捕获的 KDA 序列重排使用固定形状。
   - `self.moe_local_token_count`（标量 GPU 张量，存 `per_rank_logical_token[cp_rank]`）：供 `attention_query_tokens` 将本地 Q 切片至真实有效行数，并由 `attention_output_tokens` 将本地 `[raw:]` 填充尾行清零，同时 `cache_tokens` 保持全局 `total_seq_lens` 以写入正确的 KV Cache 槽位。
2. **DeepSeek-V4.1 `mega_gate_forward_scope` 掩码缓存 (`5f7542d7627f`)**：
   - 在模型前向入口通过 `with mega_gate_forward_scope(forward_batch):` 将 `(valid, image)` 掩码缓存在 `forward_batch._mega_gate_row_masks` 中（按 `(num_tokens, id(bound), id(image_input_ids))` 校验），将 60 层 MoE 的掩码构造算子调用从 `120` 次降至 **`2` 次（减少 98.33%，60.0x）**。
3. **Granian HTTP/1.1 Keep-Alive 透传 (`15dc838f9bf5`)**：
   - 新增 `_granian_http1_settings()`，将 `SGLANG_TIMEOUT_KEEP_ALIVE`（秒）校验为正数后映射为 `HTTP1Settings(header_read_timeout=timeout_s * 1000)` 毫秒传入 Granian Server。

### 2.3 vLLM：ROCm `gfx942` 32 位 Head 偏移 (`#63268`) 与 `gfx950` FlyDSL 统一 (`#61649`)

- **ROCm `gfx942` Triton Paged Attention 修复 (`18df4cd9a98b`)**：
  - 移除 `output_stride_1: tl.int64` 标注，先用 64 位指针算术执行 `output_ptr += cur_batch_in_all_start_index * output_stride_0`，再用 32 位 `int32` 计算 `output_offset = query_head_idx * output_stride_1`（对于 64 heads × 128 head_dim，最大偏移仅 `8,064` 元素，远小于 `2^31 - 1`），彻底消除了 64 位向量-标量乘法引发的 SGPR 溢出误编译与随机 `NaN`。
- **ROCm `gfx950` FlyDSL TurboQuant/UltraQuant 统一 (`29cf023770a0`)**：
  - 将 `tq_decode.py` 迁移至公开 `buffer_view` API 与硬件 `ds_read_tr16_b64` 寄存器内转置，统一支持 `GQA-{6, 8, 16}` 并删除了 `875` 行重复的 `tq_decode_gqa6.py`；同时引入 `base_byte_offset` 将首块偏移合入基址指针，确保 32 位 `voffset` 永不超过 AMDGPU V# 描述符的 `4 GiB` 上限。

---

## 3. 确定性实验验证 (`own-offline`)

运行 `python experiments/sglang-glm53-cp-bcg-trtllm-kda-conv-vllm-gfx942/run.py --check`：

| 验证维度 | 修复/优化前 | 修复/优化后 | 确定性结论 |
|---|---|---|---|
| TRT-LLM KDA SiLU 相对误差 ($h=-6$) | `> 3953%` (灾难性抵消) | Warp-vote 精确分支 (`0%` 抵消) | 阈值 $h=-2$ 处误差严格 `< 1.35%`，删除 `7,214` 行旧代码 |
| TRT-LLM GVR v2 Top-K (`NBH=1024, CMP=2560`) | `28,672 B` SMEM, `QC=96` | `24,576 B` SMEM, `QC=160` | 节省 **`4,096 B` SMEM/CTA**，`VPT=10` 确定性重映射为 `(1024, 5)` |
| SGLang GLM-5.3 CP=4 (`static=1024, live=770`) | 静态/动态 Token 混用 | 各 Rank 有效 `[193, 193, 192, 192]` | 填充尾行 `[63, 63, 64, 64]` 精确清零，DSV4.1 掩码构造 **`60.0x` 缩减** |
| vLLM `gfx942` Head 偏移 & `gfx950` V# 偏移 | `int64` SGPR 溢出 `NaN` / `4.91 GiB` 越界 | `int32` (`8,064` elems) / `base_byte_offset` | 消除 `gfx942` `NaN` 误编译，删除 `875` 行 `gqa6` 重复实现 |
