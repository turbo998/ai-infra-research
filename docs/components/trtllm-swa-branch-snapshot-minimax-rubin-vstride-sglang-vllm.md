# TensorRT-LLM SWA 前缀分叉快照、MiniMax-M3 Rubin (SM107) P128 混合路由、Context FMHA 独立 V 步长、SGLang LTX-2.3 SP RoPE / FLUX.3 融合 SwiGLU-FP8 与 vLLM 投机解码护栏深度解析

> **关联研究问题**：`q-trtllm-swa-branch-snapshot-minimax-rubin-vstride-sglang-vllm`（Priority 69，Phase 60 Track B）  
> **覆盖上游源码提交**：
> - `NVIDIA/TensorRT-LLM@ae1aa43964e12afb5157ca5d551993a89db2224e` (`#19877`, 2026-10-23)
> - `NVIDIA/TensorRT-LLM@854ccee8a9f0dded7235a0642032de105b3b47f7` (`#20167`, 2026-10-23)
> - `NVIDIA/TensorRT-LLM@d7ec5162ac8d935b21bb5eac719e5eac99664cec` (`#20242`, 2026-10-23)
> - `NVIDIA/TensorRT-LLM@0f3e1d093dceb454f01a02a578cde1aed99d0495` (`#20125`, 2026-10-23)
> - `sgl-project/sglang@2fbd094a8cb641deb099b5b2ac0d362b7eaa6aec` (`#45995`, 2026-10-23)
> - `sgl-project/sglang@c80285ca9adc23946447128926c22e8e7dfb2b80` (`#43809`, 2026-10-22)
> - `sgl-project/sglang@698fa949ecd3f5662a87edcb4aefcb44884a2776` (`#42758`, 2026-10-22)
> - `vllm-project/vllm@37cf712552509d5bf2e67128b80ab435079434fc` (`#63251`, 2026-10-23)
> - `vllm-project/vllm@f3871c3cbf8a608f7ee833ec4c2c5277f77bee41` (`#63339`, 2026-10-23)
> - `vllm-project/vllm@404c2b510e1d73df151d78e5576e1ee7ca0bcaa3` (`#62171`, 2026-10-23)  
> **配套确定性实验**：`experiments/trtllm-swa-branch-snapshot-minimax-rubin-vstride-sglang-vllm/run.py`

---

## 1. 架构背景与核心问题

在混合滑动窗口注意力（SWA）、Rubin（SM107）新架构使能、多模态视频/图像生成（LTX-2.3 / FLUX.3）以及 GDN/Eagle3 投机解码的生产推理栈中，近期暴露出五个极具代表性的系统级问题：

1. **SWA 滑动窗口在多轮分支对话中的前缀缓存失效**：在 TensorRT-LLM `KVCacheManagerV2` 中，滑动窗口层（Sliding Window Attention）仅保留请求末尾的窗口块。当请求 1（如 4096 tokens）与历史树在第 1500 个 token 处发生分叉时，请求 1 跑完 Prefill 后其 SWA 窗口已滑至末尾（`[4096 - W, 4096]`），分叉点 1500 处的 SWA KV 状态已被驱逐。随后到达的兄弟请求 2（同样共享前 1472 tokens）无法复用该分叉点的 SWA 状态，被迫从头重算全部 1472 tokens。
2. **MiniMax-M3 在 Rubin (SM107) 上的混合精度 KV 与 P128 页面路由不兼容**：MiniMax-M3 采用“稠密层 + 稀疏 MSA 层 + 共享 Eagle3 草稿层（页面大小 128 tokens）”的异构架构。在 Rubin `SM107` 上，`FlashInferTrtllmGenFmha` 仅支持 128-token 页面而不支持稠密目标层的 64-token 页面；且未初始化的 KV 池尾块会在 CSR 稀疏 Prefill V 反量化时混入 NaN/脏字节。
3. **Context FMHA 将 K 步长硬编码复用于独立 V 张量**：当模型（如 `qwen4_exp`）将 `[q | k | v | gate | index | pad]` 融合为单行宽张量（55 个 256 维头，物理行步长 28,160 B）并切片出 `v` 传入非 FP8 Context MLA 时，`fmhaRunner.cpp` 假设 $V$ 紧邻 $K_{\text{nope}}$ 打包，错误使用 `numKvHeads * (headSizeQkNope + headSizeV)` 作为 $V$ 的行步长，读取了错误的列。
4. **SGLang LTX-2.3 视频序列并行（SP）RoPE 起始帧归零与 ROCm TP>2 确定性失真**：LTX-2.3 在 Batched Classifier-Free Guidance（BCG）缺失预计算坐标时，未向 `prepare_coords` 传入 `video_sp_start`，导致所有 SP 秩均从第 0 帧开始计算 3D RoPE；同时在 AMD ROCm 上，当 `tp_size > 2` 且禁用自定义 1-stage all-reduce 时，回退至 RCCL 会因树/环归约顺序随 batch 变化而破坏确定性推理。
5. **vLLM `cooperative_topk` 误用缓冲区 `stride` 代替实际行长 `sl`**：在宽词表或预分配大步长缓冲区（如 `stride = 1,048,576`）上对短有效行（如 `sl = 4096`）执行 `cooperative_topk` 时，内核按 `(stride + CS - 1) / CS = 65536 > kFusedMax (512)` 判定单 CTA 负载，导致本可在单趟（single-pass）完成的短行退化为高开销的两趟（two-pass）全局归约。

---

## 2. 核心实现机制与源码级剖析

```
+--------------------------------------------------------------------------------------------------+
| Phase 60 Track B: Serving Runtime, KV Cache & Speculative Decoding Architecture                  |
+--------------------------------------------------------------------------------------------------+
| 1. TensorRT-LLM SWA Branch Snapshot (#19877)                                                     |
|    - Fork point = (divergence // tokens_per_block) * tokens_per_block (when reused < point < L)  |
|    - Splits prefill with FORCE_CHUNK at fork point + commits SWA window -> 71.88% sibling reuse  |
|                                                                                                  |
| 2. TensorRT-LLM MiniMax-M3 on Rubin SM107 (#20167) & Separate V Stride (#20242)                  |
|    - Sparse MSA layers keep NVFP4 KV on SM100/103/107; dense + Eagle3 draft stay FP8             |
|    - EXTRA_PAGE_SIZES_ONLY_SMS={107} routes 128-token draft pages to FlashInferTrtllmGenFmha     |
|    - Explicit v_stride_in_bytes (16B TMA aligned) prevents corrupt column reads on sliced V      |
|                                                                                                  |
| 3. SGLang Diffusion & ROCm Determinism (#45995, #43809, #42758)                                  |
|    - LTX-2.3 BCG RoPE passes start_frame = video_sp_start; ROCm refuses RCCL AR when TP > 2     |
|    - FLUX.3 fused_packed_swiglu_fp8_rowwise cuts HBM traffic from 9 B/elem to 5 B/elem (-44.44%) |
|                                                                                                  |
| 4. vLLM Spec-Decode & Top-K Guards (#63251, #63339, #62171)                                      |
|    - cooperative_topk sizes per_block = ceil(sl / CS) instead of stride -> 256x less staging     |
|    - CPU GDN conv-state matches GPU seq_lens[0] == state_len - (width - 2); draft RoPE cap       |
+--------------------------------------------------------------------------------------------------+
```

### 2.1 TensorRT-LLM SWA 前缀分叉快照 (`TensorRT-LLM@ae1aa43964`)

在 `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2/_core/_kv_cache.py` 中，当 `enable_branch_snapshot=True` 时，`_swa_branch_snapshot_point` 在基数树（radix tree）查找阶段探测首个未命中位置 `divergence`（即 `len(matched)`）：
$$\text{point} = \left\lfloor \frac{\text{divergence}}{\text{tokens\_per\_block}} \right\rfloor \times \text{tokens\_per\_block}$$
只要满足 $0 < \text{reused} < \text{point} < \text{prompt\_len}$ 且 $\text{divergence} < \text{num\_lookup\_tokens}$，`LlmRequest` 就会调用 `split_chunk_at(point)` 并标记 `ChunkingPolicy.FORCE_CHUNK`。同时，`RequestScheduler._first_chunk_waits_for_budget` 确保当剩余 token 预算不足以容纳该 `FORCE_CHUNK` 首块时，推迟该请求至下一轮迭代，避免破坏分叉块边界。首块 Prefill 完成后，`KVCacheManagerV2` 立即提交（commit）该分叉点处的完整 SWA 窗口块，使后续所有共享该前缀的兄弟请求可直接从 `point` 处恢复 SWA 状态。

### 2.2 MiniMax-M3 Rubin (SM107) 使能 (`TensorRT-LLM@854ccee8a9`) 与 Context FMHA 独立 V 步长 (`TensorRT-LLM@d7ec5162ac`)

1. **MiniMax-M3 on Rubin (`SM107`)**：
   - **层间差异化 KV 量化**：在 `_exempt_minimax_m3_dense_layers_from_nvfp4_kv` 中，仅保留稀疏 MSA 层的 `NVFP4` KV Cache（其 Triton/MSA 内核原生支持 `(10, 0), (10, 3), (10, 7)`），而将稠密目标层与共享 Eagle3 草稿层回退至 `FP8` KV Cache。
   - **按页面大小混合注意力后端路由**：在 `attention_Selector.py` 中设置 `EXTRA_PAGE_SIZES_ONLY_SMS = {107}`。在 SM107 上，64-token 页面的稠密层路由至 `TrtllmAttention`（SDPA），而 128-token 页面的共享 Eagle3 草稿层路由至 `FlashInferTrtllmGenFmha`（`P128` cubin）。
   - **KV 池零初始化**：在 `_zero_kv_cache_manager_v2_pools` 中对主 KV 池与草稿 KV 池执行 `.zero_()`，防止 CSR 稀疏 Prefill 反量化读取未写入的页尾随机字节。
2. **Context FMHA 独立 V 行步长（`#20242`）**：
   - 在 `fmhaRunner.cpp` 与 `fp8Quantize.cu`（`quantizeCopyInputToFp8Kernel`）中新增显式 `v_stride_in_bytes`（并在 Python 侧校验 `v.stride(0) * v.element_size() % 16 == 0` 的 16B TMA 对齐）。当 `qwen4_exp` 从 55 头融合投影张量中切片出 `v` 时，`v_token_stride_elements = 14080`（28,160 B）被准确传递给 TMA 描述符与 FP8 量化内核。

### 2.3 SGLang LTX-2.3 SP RoPE 偏移、ROCm 确定性护栏与 FLUX.3 融合 SwiGLU-FP8 (`sglang@2fbd094a8c`, `c80285ca9a`, `698fa949ec`)

- **LTX-2.3 BCG 视频 RoPE SP 起始帧对齐（`#45995`）**：在 `_prepare_ltx2_rope_coords_for_bcg` 中透传 `video_start_frame = video_sp_start or 0` 至 `prepare_coords(..., start_frame=video_start_frame)`，确保 4 卡 SP（每卡 16 帧）各自生成 `[0..15], [16..31], [32..47], [48..63]` 的连续帧位置编码。
- **ROCm 确定性推理拒绝 TP>2 的 RCCL All-Reduce（`#43809`）**：在 `_deterministic_rocm_rccl_check` 中，当 `is_hip and enable_deterministic_inference and tp_size > 2 and disable_custom_all_reduce` 同时成立时立即抛出 `ValueError`，强制使用固定顺序归约的 1-stage 自定义 all-reduce。
- **FLUX.3 融合 Packed SwiGLU + 逐行 FP8 量化（`#42758`）**：`_swiglu_fp8_rowwise_kernel` 从打包的 BF16 `[M, 2*K]` 输入中读取 `gate` 和 `up`，通过两次 `round_bf16_to_fp32` 精确复现 `F.silu(gate) * up` 的两次 BF16 舍入边界，随后在寄存器内直接计算行缩放因子 $\text{scale} = \max(\max_j |v_j| / 448.0, 10^{-12})$ 并写出 `float8_e4m3fn`。每列元素 HBM 流量从 $4\text{B (read)} + 2\text{B (write)} + 2\text{B (read)} + 1\text{B (write)} = 9\text{ B}$ 降至 $4\text{B (read)} + 1\text{B (write)} = 5\text{ B}$，减少 **44.44%**。

### 2.4 vLLM `cooperative_topk` 行长分块、CPU GDN 卷积状态与草稿 RoPE 上限 (`vllm@404c2b510e`, `f3871c3cbf`, `37cf712552`)

- **`cooperative_topk` 按实际行长 `sl` 计算 `per_block`（`#62171`）**：将 `const int64_t per_block = (stride + CS - 1) / CS` 修正为 `const int64_t per_block = (sl + CS - 1) / CS`。对于 `stride = 1,048,576`、`sl = 4096`、`CS = 16` 的场景，`per_block` 从 `65536`（被迫走两趟）降至 `256 <= kFusedMax (512)`（恢复单趟寄存器/SMEM 归约），单 CTA 扫描量缩减 **256 倍**。
- **CPU GDN 投机解码 `conv_state` 布局对齐（`#63339`）**：将 CPU `causal_conv1d_update` 的满长投机行判定条件修正为 `seq_lens[0] == state_len - (width - 2)`，与 GPU 端 `width - 1` 历史加上 `1 + num_spec` 查询 token 的滑动步进完全一致。
- **V2 Model Runner 投机解码草稿 RoPE 缓存上限（`#63251`）**：新增 `_limit_max_model_len_to_draft`，在构建输入缓冲区前将主模型的有效 `max_model_len` 截断至草稿模型的 `max_position_embeddings`（如 EAGLE 草稿模型仅训练至 2048 位置时，防止长序列触发草稿 `rotary_emb.cos_sin_cache` 越界索引）。

---

## 3. 量化实验与关键指标对比

基于 `experiments/trtllm-swa-branch-snapshot-minimax-rubin-vstride-sglang-vllm/run.py` 的确定性验证结果如下：

| 子系统 / 优化项 | 修复前行为 / 开销 | 修复后行为 / 开销 | 量化收益 / 核心不变量 |
| :--- | :--- | :--- | :--- |
| **TRT-LLM SWA 分叉快照 (`#19877`)** | 兄弟请求重算全部 `2048` tokens (`reused=0`) | 在分叉块 `1472` 提交 SWA 窗口，仅算 `576` tokens | 兄弟请求 Prefill 计算节省 **71.88%** |
| **TRT-LLM MiniMax-M3 Rubin (`#20167`)** | SM107 稠密层无 P64 cubin 报错 / 稀疏页尾脏数据 | 稀疏层 NVFP4 + 稠密/草稿 FP8 + 128-token P128 路由 | KV 池 100% 零初始化，消除页尾 NaN |
| **TRT-LLM 独立 V 步长 (`#20242`)** | 误用打包 `k_nope+v` 步长导致切片 V 读错列 | 显式传入 `28,160 B` (`14,080` elems, 16B TMA 对齐) | 消除 `qwen4_exp` 融合投影切片精度崩塌 |
| **SGLang FLUX.3 SwiGLU-FP8 (`#42758`)** | 落盘 BF16 中间激活再读回量化（`9 B/elem`） | 寄存器内双次 BF16 舍入直出 E4M3（`5 B/elem`） | 激活量化阶段 HBM 流量降低 **44.44%** |
| **vLLM `cooperative_topk` (`#62171`)** | 按 `stride=1M` 算得 `per_block=65536`（退化两趟） | 按实际行长 `sl=4096` 算得 `per_block=256`（单趟） | 恢复单趟 Top-K，单 CTA 分片缩减 **256x** |

---

## 4. 解决方案架构师设计指南与反证问题

1. **Agentic 多分支树搜索（Tree-of-Thoughts / Best-of-N）下的 SWA 缓存策略**：当推理服务承载带工具调用分支或并行采样的混合 SWA 模型时，必须开启 `enable_branch_snapshot=True`，否则所有晚于首条长分支完成的同前缀兄弟分支都将失去 SWA 复用。
2. **融合投影切片张量的 TMA 步长契约**：在将宽融合投影（如 `[Q|K|V|Gate|Index]`）通过视图切片传入底层 FMHA 或量化内核时，不能假设 $K$ 与 $V$ 具有相同的行步长或紧邻布局，必须显式传递 `k_stride_in_bytes` 与 `v_stride_in_bytes` 并校验 16B 对齐。
3. **后续可证伪问题**：在 SWA 分叉快照机制下，若一条长请求在基数树中连续触发多个历史分叉点（multi-fork chain），频繁插入 `FORCE_CHUNK` 分块会对单请求 TTFT 产生多大的尾延迟开销？是否应设置最小复用块数阈值（如 $\ge 4$ blocks）才触发快照？
