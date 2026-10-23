# SGLang Marlin WNA16 MoE 哨兵路由与分离式心跳重置、TensorRT-LLM SM100 SwiGLU Epilogue 融合与 KDA 集成及 vLLM Mamba/Spec-Decode 护栏解析

> **双语导航 / Bilingual Navigation**: [中文版](sglang-marlin-moe-trtllm-kda-swiglu-vllm.md) | [English Version](sglang-marlin-moe-trtllm-kda-swiglu-vllm.en.md)
> **证据锚点 / Evidence IDs**: `src-sglang-marlin-moe-trtllm-kda-swiglu-vllm-2026` · `ev-sglang-marlin-moe-trtllm-kda-swiglu-vllm-2026` (`priority: 73`, Phase 64 Track B)
> **可复现实验 / Reproducible Experiment**: `python experiments/sglang-marlin-moe-trtllm-kda-swiglu-vllm/run.py --check`

---

## 1. 架构师问题定义与工程背景

在大规模 MoE 推理、Prefill-Decode (PD) 分离服务、Blackwell SM100 稠密/混合架构以及 Mamba/投机解码（Speculative Decoding）中，推理引擎层存在四个关键系统瓶颈与正确性隐患：

1. **Marlin WNA16 MoE 路由哨兵块（Inactive Prefix Blocks）与非法 Expert ID 越界**：当启用虚拟专家（Virtual Experts）或 LoRA/EP 掩码时，未激活路由被标记为 `-1`（映射至 +1 偏移的 Bucket `0`）。若 Marlin 核在每个 Warp 内串行扫描所有块或对未激活块执行冗余计算，将浪费大量 CTA 周期；同时若有符号负值 ID（如 `-99`）直接写入 `cumsum_buffer[expert_id]`，会引发设备端越界原子写。
2. **SGLang PD 分离架构中恢复节点的陈旧心跳计数误杀**：当某个 Prefill 节点因连续达到 `max_failures` 次心跳超时被 `_handle_node_failure` 摘除后，若其 `heartbeat_failures[addr]` 计数器未被清空，一旦后续新请求重新连接该恢复节点，仅出现**单次**瞬态繁忙超时（1 missed heartbeat）就会使累计计数越界并再次误杀节点。
3. **TensorRT-LLM SM100 GatedMLP 中间激活 `[M, 2I]` 的 HBM 往返流量与双重 BF16 舍入**：未融合路径先将 Gate/Up 投影写入 `[M, 2I]` BF16 缓冲区，再启动独立 SwiGLU 核读回并写出 `[M, I]` BF16，在大 Token 批（如 MiniMax-H3 $M=4096, I=14336$）下产生数百 MiB 的冗余 HBM 读写。
4. **vLLM Mamba 共享前缀兄弟请求串行推迟与 V2 Model Runner 投机解码 RoPE 越界**：Mamba 混合缓存管理器在 `_cache_partial_tail_block` 中对已提交的块重复计算缓存哈希，导致同一 Step 到达的兄弟请求（Siblings）被误判为“本步新缓存”而逐 Step 串行推迟；同时 V2 Model Runner 不支持在长序列上跳过投机，若独立 Draft 模型或 YaRN/Dynamic RoPE 的位置上限小于 Target，会导致旋转位置编码缓存越界。

---

## 2. 核心源码变更与数学建模

### 2.1 SGLang Marlin WNA16 MoE `int2` 计数发布与无符号哨兵映射 (`#45462`, `8b9b992`) 及 PD 心跳重置 (`#45922`, `16aa816`)

1. **无符号偏移哨兵映射 `sentinel_expert_bucket`**：在 `python/sglang/kernels/jit/csrc/moe/moe_align_kernel.cu` 中：
   ```cpp
   template <typename scalar_t>
   __device__ __forceinline__ int32_t sentinel_expert_bucket(scalar_t expert_id, int32_t num_buckets) {
     using unsigned_t = std::make_unsigned_t<scalar_t>;
     const unsigned_t shifted = static_cast<unsigned_t>(expert_id) + unsigned_t{1};
     return shifted < static_cast<unsigned_t>(num_buckets) ? static_cast<int32_t>(shifted) : 0;
   }
   ```
   利用无符号模加将 `expert_id = -1` 精确映射为 `shifted = 0`，将有效专家 $i \in [0, E-1]$ 映射为 $i+1 < E+1$，并将所有其他非法负数（如 `-99`）或超界正数（$\ge E$）统一钳制到哨兵桶 `0`，仅需单次无符号比较。
2. **`int2` 原子对发布与未激活前缀块跳过**：`align_single_token_kernel` 与 `moe_align_block_size` 在 `kWriteInactivePrefixCount=true` 时通过单次对齐 `int2` 写同时发布 `(num_tokens_past_padded, inactive_prefix_blocks)`。Marlin 主核（`marlin_template.h`）使用 `const int2 counts = __ldg(reinterpret_cast<const int2*>(num_tokens_past_padded_ptr))` 一次加载两个计数，直接跳过头部 `inactive_prefix_blocks` 个哨兵块（在 16 个填充块含 6 个未激活前缀块时减少 **37.5%** 的无效 CTA 块调度），并通过 `zero_inactive_output_rows` 安全清零未激活路由输出。
3. **PD 分离节点恢复心跳计数清零 (`#45922`, `16aa816`)**：`python/sglang/srt/disaggregation/common/conn.py` 在 `_handle_node_failure(failed_bootstrap_addr)` 中新增 `self.heartbeat_failures.pop(failed_bootstrap_addr, None)`。在序列 `["miss", "miss", "miss", 200, "miss", 200]`（`max_failures=2`）下，未修复代码在重连后的单次 `"miss"` 即触发第 2 次误杀摘除，修复后节点在重连后面对单次瞬态超时保持存活（仅记录 1 次初始故障摘除）。

### 2.2 TensorRT-LLM SM100 QuACK BF16 Gate/Up + SwiGLU Epilogue 融合 (`#19773`, `90d4dcb`) 与 FlashInfer KDA 预填充集成 (`#20653`, `222676c`)

1. **SM100/SM103 QuACK `gemm_act` Epilogue 融合**：`tensorrt_llm/_torch/modules/gate_up_swiglu_quack.py` 利用 QuACK 的 `gemm_act(..., activation="swiglu", concat_layout=("B",), tuned=False)`，在不复制 TRT-LLM `[gate; up]` 权重布局的前提下，直接在 SM100 Tensor Core 的 FP32 累加器 Epilogue 中完成 $\text{SiLU}(\text{gate}) \odot \text{up}$ 并一次性舍入写出 `[M, I]` BF16：
   - **未融合路径激活 HBM 流量**：
     $$\text{Bytes}_{\text{unfused}} = \underbrace{M \cdot 2I \cdot 2}_{\text{GEMM Write}} + \underbrace{M \cdot 2I \cdot 2}_{\text{SwiGLU Read}} + \underbrace{M \cdot I \cdot 2}_{\text{SwiGLU Write}} = 10 \cdot M \cdot I \text{ bytes}$$
   - **融合 Epilogue 路径激活 HBM 流量**：
     $$\text{Bytes}_{\text{fused}} = \underbrace{M \cdot I \cdot 2}_{\text{Epilogue Write}} = 2 \cdot M \cdot I \text{ bytes}$$
   - 在 MiniMax-H3 典型形状（$M=4096, K=5376, I=14336$）下，单层激活 HBM 流量从 **560.0 MiB** 降至 **112.0 MiB**，每层节省 **448.0 MiB（减少 80.0%）**，并将 BF16 舍入次数从 2 次降为 1 次。若运行期 QuACK JIT 编译失败，`_disable_kernel` 自动将 `_kernel_state["ok"]` 置为 `False` 并平滑回退至未融合路径。
2. **FlashInfer KDA `bt16` 融合卷积预填充集成 (`#20653`, `222676c`)**：通过 `3rdparty/vendor_patches/flashinfer-kda-bt16.patch`（207 行补丁，增加 `mStateIndices` 状态池槽位间接寻址与填充步长支持）替换了原先 3 个独立手写 Blackwell KDA 文件（`akk_inverse.py`, `fused_k123.py`, `fused_k1234.py`，共移除 **5,706 行**冗余维护代码）。

### 2.3 vLLM Mamba 前缀缓存尾部重标记守卫 (`#62841`, `d561af5`) 与 V2 投机解码 RoPE 边界控制 (`#63251`, `37cf712`)

1. **Mamba 兄弟请求同 Step 准入 (`#62841`)**：`vllm/v1/core/single_type_kv_cache_manager.py` 的 `_cache_partial_tail_block` 新增守卫：
   ```python
   if source_block.block_hash_num_tokens == num_tokens:
       return None
   ```
   当某块已在先前 Step 被提交并缓存到 `num_tokens` 时，不再重复调用 `cache_partial_block` 将其标记为“当前 Step 新缓存”，从而使共享 32-token Mamba 前缀的 3 个兄弟请求在 **1 个引擎 Step** 内全部并发准入（修复前需串行分 **3 个 Step** 准入，准入吞吐提升 **`3.0x`**）。
2. **V2 Model Runner 投机解码 RoPE Cache 与 `max_model_len` 协调 (`#63251`)**：
   - 在 `vllm/config/speculative.py` 中定义 `_CACHE_ONLY_ROPE_TYPES = frozenset({"default", "linear", "llama3", "ntk", "proportional"})`。对于非独立 `draft_model`（如 EAGLE3, MTP, DFlash, DSpark）且 RoPE 类型属于 `_CACHE_ONLY_ROPE_TYPES` 的起草器，自动将其 `max_position_embeddings` 提升至 Target 的 `max_model_len`（如 `EAGLE3-LLaMA3.1-8B` 从 `2048` 提升至 `131072`）；
   - 对于独立 `draft_model` 或频率依赖长度的 RoPE（如 `yarn`, `dynamic`, `longrope`），在 V2 Model Runner 下通过 `VllmConfig._limit_max_model_len_to_draft()` 自动将引擎 `max_model_len` 下调至 Draft 上限，彻底消除 `cos_sin_cache` 越界读取。

---

## 3. 确定性实验与验证结果 (`own-offline`)

运行 `python experiments/sglang-marlin-moe-trtllm-kda-swiglu-vllm/run.py --check` 输出如下验证矩阵：

| 框架与模块 | 核心机制 | 定量验证结果 | 工程收益 |
|---|---|---|---|
| **SGLang Marlin MoE** (`#45462`) | `sentinel_expert_bucket` + `int2` 前缀计数 | `-1/-99/256 -> bucket 0`, `0 -> 1`；跳过 `6/16` 未激活块 | 消除非法专家 ID 越界写，跳过 **37.5%** 空闲路由块 |
| **SGLang PD Disagg** (`#45922`) | `heartbeat_failures.pop(failed_addr, None)` | 重连后单次超时保持存活（误杀次数 `2 -> 1`） | 防止繁忙 Prefill 节点重连后被瞬态抖动误摘除 |
| **TRT-LLM SM100 SwiGLU** (`#19773`) | QuACK `gemm_act` FP32 Epilogue 融合 | MiniMax-H3 单层激活 HBM `560 MiB -> 112 MiB` | 每层节省 **448.0 MiB (-80.0%)**，BF16 舍入 `2 -> 1` |
| **TRT-LLM KDA Prefill** (`#20653`) | FlashInfer `bt16` + `mStateIndices` 补丁 | 移除 **5,706 行**手写核，仅保留 207 行槽位补丁 | 统一状态池间接寻址与融合卷积预填充 |
| **vLLM Mamba + Spec** (`#62841`, `#63251`) | `block_hash_num_tokens` 守卫 + RoPE 分类 | 3 兄弟准入步数 `3 -> 1` (`3.0x`)；EAGLE3 提升至 `131072` | 消除 Mamba 共享前缀饥饿与 V2 Spec RoPE OOB |
