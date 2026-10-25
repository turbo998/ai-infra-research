# 推理引擎 PD 分离 KV Adler-32 校验、QSA 预填充聚合、ROCm 32-bit 缓冲寻址、DBO 微批偏移与 MiniMax-M3 VL 深度解析（Phase 85）

> **关联实验**：`experiments/serving_pd_checksum_qsa_rocm_dbo_minimax_phase85/`  
> **关联问题**：`q-serving-pd-checksum-qsa-rocm-dbo-minimax-phase85` (`priority: 94`)  
> **English Version**: [`serving-pd-checksum-qsa-rocm-dbo-minimax-phase85.en.md`](./serving-pd-checksum-qsa-rocm-dbo-minimax-phase85.en.md)

---

## 1. 背景与问题域

在大规模 PD（Prefill-Decode）分离推理、稀疏注意力（QSA）、双微批重叠（DBO）以及多模态服务中，SGLang、vLLM 与 TensorRT-LLM 近期修复并优化了 8 项关键生产级正确性与性能瓶颈：

1. **SGLang PD 分离 KV Adler-32 校验与元数据缓冲区对齐保护（PR #39729，`5d11819f`）**。
2. **SGLang LTX-2 扩散模型 Q/K RMSNorm + Split-RoPE 16-Byte 向量化内核（PR #46111，`a887ac8f`）**。
3. **SGLang QSA 预填充压缩 Key 单次 Triton 聚合内核（PR #45752，`dba22e9b`）**。
4. **SGLang EAGLE 投机解码在 PD HiCache 恢复中的 Bigram 边界 Token 补偿（PR #45076，`0ff67ee7`）**。
5. **vLLM ROCm `TRITON_ATTN` KV Cache 视图 32-bit 缓冲区寻址恢复（PR #60084，`a55937f2`）**。
6. **vLLM MRV2 DBO（Dual-Batch Overlap）路由专家行偏移与首错微批上报（PR #63372 `2c19f679` & PR #63476 `66bf9315`）**。
7. **vLLM `SimpleCPUOffloadConnector` 分块局部注意力（Chunked-Local Attention）对齐门控（PR #63266，`65bc6983`）**。
8. **TensorRT-LLM MiniMax-M3 VL 预渲染模板防重与 LoRA `ckpt_source` 驱逐重载保持（PR #20626 `a0cee56d` & PR #20774 `9666d742`）**。

---

## 2. 源码级机制、公式与定量分析

### 2.1 SGLang PD 分离 KV Adler-32 校验与 Aux Buffer 对齐 (`5d11819f`)
- **前缀命中与整请求摘要冲突**：`--disaggregation-enable-kv-checksum` 计算整个请求所有 KV 页的 Adler-32 摘要；但当 Decode 端开启 `--disaggregation-decode-enable-radix-cache` 并命中前缀时，Prefill 仅传输后缀 KV 页，导致 Decode 端重算的整请求摘要必然与 Prefill 摘要不匹配。SGLang 在检测到二者同时开启时自动抑制该虚假校验失败，并在異构 KV 布局（TP 宽度、PP 阶段、页大小不同）时降级为带单次告警放行。
- **Aux 缓冲区位置配对保护**：Mooncake (`mooncake/conn.py`) 与 MORI (`mori/conn.py`) 按位置索引传输辅助元数据缓冲区（KV Checksum、Sampling Mask、DSA Top-K）。新增 `check_peer_aux_buffer_count` 并将传输循环上界钳制为：
  $$N_{\text{xfer}} = \min\left(|\text{prefill\_aux\_ptrs}|,\; |\text{dst\_aux\_ptrs}|\right)$$
  防止单侧开启可选元数据缓冲区时发生越界 RDMA 写或静默错位。

### 2.2 SGLang LTX-2 向量化 Q/K RMSNorm + Split-RoPE (`a887ac8f`) 与 QSA 预填充聚合 (`dba22e9b`)
- **LTX-2 `kVec=true` 向量化 (`a887ac8f`)**：在 `ltx2_qknorm_split_rope_kernel` 中，当指针与步长满足 16-Byte 对齐时，每个线程单次迭代处理 **`kWidth = 8` 对复数（16 个 BF16 = 32 B）**，以两次 16B 向量加载分别读取前半维度 `idx0` 与后半维度 `idx1 = idx0 + half_dim`，并保持与标量路径完全一致的 FMA 舍入语义（`y0 = fmaf(-sin, x1, cos*x0)`, `y1 = fmaf(sin, x0, cos*x1)`），将全局内存事务数降低 **`8x`**。
- **QSA 单次 Triton 聚合 (`dba22e9b`)**：由于 `QSATokenToKVPool` 采用页对齐分配，每个压缩块的物理槽位等于其首个原始 Token 槽位整除压缩比：
  $$\text{compressed\_slot}(s, b) = \left\lfloor \frac{\text{token\_slot\_table}[s,\; b \times \text{RATIO}]}{\text{RATIO}} \right\rfloor$$
  `_gather_qsa_compressed_keys_kernel` 以单次 2D Triton Launch 替代了此前每个请求的 Python 循环 `index_select` + `torch.cat`（在 5 请求批次上将 Launch 从 `5` 次降至 **`1` 次**，且在已知 `prefill_sequence_lengths_cpu` 时实现 **0 次 Host-Device 同步**）。

### 2.3 SGLang EAGLE HiCache Bigram 边界补偿 (`0ff67ee7`)
EAGLE 投机树以 **$N + 1$ 个 Token ID 作为键索引 $N$ 个 Bigram**：
$$\text{rematch\_token\_len} = \text{pm.decode\_prefix\_len} + \mathbb{I}(\text{tree\_cache.tree\_core.is\_eagle})$$
修复前 `max_prefix_len = pm.decode_prefix_len` 少算了 1 个边界 Token，导致最后一个 Bigram 在 PD Decode HiCache L2 恢复重匹配时丢失。

### 2.4 vLLM ROCm `TRITON_ATTN` 32-bit 缓冲区寻址 (`a55937f2`)
AMD CDNA GPU 的 `buffer_load_dword` / `buffer_store_dword` 指令仅在指针访问范围 $< 2^{31}\text{ B}$（`2 GiB`）时启用 32-bit 偏移寻址。Triton AMD 后端优先检查张量的 `ptr_range()`，若未设置则回退检查整个底层 `UntypedStorage` 的字节数。当 32 层共享一个 `8 GiB` 统一 KV 分配时，单层视图（仅 `256 MiB`）因无 `ptr_range()` 而退化为 64-bit 标量+向量地址运算。`set_triton_ptr_range` 显式注入：
$$\text{view.ptr\_range} = \text{region\_end} - \text{view.data\_ptr}() < 2\text{ GiB}$$
使 `key_cache`（`256 MiB`）与 `value_cache`（`128 MiB`）全面恢复 32-bit 硬件缓冲指令。

### 2.5 vLLM MRV2 DBO 微批隔离 (`2c19f679`, `66bf9315`) 与分块局部注意力 CPU 卸载 (`65bc6983`)
- **DBO 路由专家偏移 (`2c19f679`)**：`RoutedExpertsCapturer.capture()` 改用 `offset = dbo_current_ubatch_token_offset()` 写入 `device_buffer[offset : offset + len(local_topk_ids), layer_id]`，消除微批 1 覆盖微批 0（`[0:64]`）的静默数据损坏。
- **DBO 首错上报 (`66bf9315`)**：`ubatch_utils.py` 将 `failed = min(errors)` 改为 `failed = next(iter(errors))`（按 Python `dict` 插入顺序），避免微批 1 先抛错退出后、微批 0 在注册 `recv_hook` 时触发衍生异常并因索引 `0 < 1` 掩盖真实根因。
- **Chunked-Local CPU Offload 对齐门控 (`65bc6983`)**：在 Llama-4 等分块局部注意力（如 `attention_chunk_size = 64`）中，若 GPU 局部前缀未对齐分块边界（如 `num_computed_tokens = 16`，`16 % 64 != 0`），直接拼接 CPU 缓存后缀会导致后缀内部的分块相对位置与绝对位置错位；`SimpleCPUOffloadConnector` 现要求 `num_computed_tokens % spec.attention_chunk_size == 0` 方可复用 CPU KV。

### 2.6 TensorRT-LLM MiniMax-M3 VL 与 LoRA `ckpt_source` (`a0cee56d`, `9666d742`)
- **MiniMax-M3 VL (`a0cee56d`)**：将注册占位符修正为模板原生裸标记 `]<]image[>[` / `]<]video[>[`，新增 `TextPrompt.chat_template_applied = True` 防止 `trtllm-serve` 预渲染 Prompt 被二次套模板；并在分块预填充/KV 复用中跳过全部视觉占位符已缓存或超出当前分块的 Vision Tower 前向。
- **LoRA `ckpt_source` 保持 (`9666d742`)**：在 `LlmRequest` 与 `ExecutorRequest` 中持久化 `py_lora_ckpt_source`，确保适配器被驱逐出 `PeftCacheManager` 后重新加载时使用请求准入时的格式（如 `"nemo"`），而非回退至服务器默认格式（如 `"hf"`）。

---

## 3. 架构师设计练习与可证伪问题

1. **计算练习（ROCm 32-bit 寻址阈值）**：若单卡分配 40 层 KV Cache，每层 `192 MiB`（总 `7.5 GiB`），写出 `set_triton_ptr_range` 为第 $i$ 层的 `key_cache`（前半）和 `value_cache`（后半）计算出的 `ptr_range()` 字节数（分别为 `192 MiB` 和 `96 MiB`，均 $< 2\text{ GiB}$）。
2. **可证伪假设**：在 `attention_chunk_size = 64` 的分块局部注意力模型中，当 GPU 命中 `16` 个局部 Token 且 CPU 缓存命中后续 `48` 个 Token 时，若不拒绝该 CPU 后缀命中，位置 `64..79` 的 Token 将错误地与位置 `16..63` 形成跨块注意力或丢失塊内上下文。
