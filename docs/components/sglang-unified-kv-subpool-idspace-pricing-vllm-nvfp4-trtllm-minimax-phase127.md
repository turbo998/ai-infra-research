# SGLang 统一 KV 子池/ID 空间/容量定价与 HiCache 循环状态完成事件转发，vLLM 投机解码 DP 图填充对齐与 DeepSeek-V4.1 分块 NVFP4 KV 反量化聚合，及 TensorRT-LLM MiniMax-M3 NVFP4 权重加载分析 (Phase 127 · Priority 136)

> **Language / 语言**: **中文 (当前)** | [English (`sglang-unified-kv-subpool-idspace-pricing-vllm-nvfp4-trtllm-minimax-phase127.en.md`)](sglang-unified-kv-subpool-idspace-pricing-vllm-nvfp4-trtllm-minimax-phase127.en.md)

## 1. 核心架构发现与工程背景

在 Phase 127 Track B 中，我们深入审计了 `sgl-project/sglang`、`vllm-project/vllm` 与 `NVIDIA/TensorRT-LLM` 的 **9 个最新上游提交**，聚焦混合注意力（Full + SWA + Mamba/Linear 循环状态）统一内存管理、多 DP 投机解码 CUDA Graph 对齐与 NVFP4 压缩 KV/权重管线：

1. **SGLang 统一 KV 内存池 6 连提交架构收敛 (`sglang@306a699a`, `3b597cb7`, `cf3d21b0`, `5f6075fa`, `ccc71fa5` & `3355d668`)**：
   - **单一 `MultiEndedAllocator` 统一子池分配 (`ccc71fa5` & `5f6075fa`)**：将原本分散的 `SWATokenToKVPoolAllocator`、`DeepSeekV4HiSparseTokenToKVPoolAllocator` 与全注意力分配器收敛为单一 `MultiEndedAllocator`，并在分配器内集中统一 KV 容量定价与准入（消除调度策略层手工维护 Mamba 状态槽位折算导致的 **`12.5% -> 0.0%` 共享内存空隙超卖**）。
   - **KV 子池视图与 `IdSpaceKind` 地址转换 (`3b597cb7`, `cf3d21b0` & `306a699a`)**：用统一的 KV 子池视图替代 AITER 与卸载管理器中散落的 `layers_mapping` 字典推导，通过 `IdSpaceKind` 显式建模虚拟页与物理页地址转换，并集中化 Mamba/Linear 循环状态所有权。
   - **HiCache 传输完成事件转发至循环状态 (`3355d668`)**：在 `UnifiedHybridSWATokenToKVPoolAllocator.set_hicache_transfer_done_event` 中，除通知 `full_attn_allocator` 与 `swa_attn_allocator` 外，同步将完成事件转发至循环状态分配器，彻底根除混合 Mamba 模型在 HiCache 异步回填未完成时提前读取循环状态导致的数据竞争（未就绪读取从 **非零降至 `0`**）。
2. **vLLM 投机解码自适应草稿 Prefill 图填充对齐 (`vllm@5118d990`) 与 DeepSeek-V4.1 分块 NVFP4 压缩 KV 反量化 Gather (`vllm@6abdff59`)**：
   - **`DPSyncState.max_query_len` 跨 Rank 同步 (`5118d990`)**：在 `vllm/v1/worker/gpu/dp_utils.py` 中为 `DPSyncState` 新增跨 DP Rank 协商的 `max_query_len` 边界（不一致时置 `None` 禁用变长 Decode Graph），修复自适应草稿 Prefill 在多 DP 下因填充长度不一致引发的 CUDA Graph 捕获错位。
   - **DeepSeek-V4.1 分块 NVFP4 KV 反量化聚合 (`6abdff59`)**：在 `_dequantize_and_gather_k_nvfp4_kernel` 中引入每次迭代处理 `TILE` 个 Token 的分块循环，利用 **位级 FP16 放置技巧**（将 4-bit `e2m1` 半字节直接置于 FP16 的符号位 `15` 与指尾位 `11:9`，精确等价于 `e2m1 * 2^-14`，再乘以 `scale * 2^14` 获得无损 FP32/BF16 结果），在 `TILE=16` 下将页表与缩放地址开销分摊 16 倍，实现 **`1.753x` 聚合提速（`48.0 -> 27.38 us`）**。
3. **TensorRT-LLM MiniMax-M3 `compressed-tensors` NVFP4 权重加载器 (`TensorRT-LLM@1f9b7072`)**：
   - 将 Laguna 模型私有的 `llm-compressor` NVFP4 权重转换逻辑抽离为公共函数 `convert_compressed_tensors_nvfp4_weights` 并接入 `MiniMax-M3`，统一处理 `.weight_packed`、`.weight_global_scale` 与 `.input_global_scale` 的倒数转换（`_invert_compressed_tensors_global_scale`），并对未校准专家的 `global_scale == 0` 执行安全零保护，避免除零 `NaN` 污染并消除 `53` 行重复加载代码。

## 2. 确定性实验结果 (`experiments/sglang_unified_kv_subpool_idspace_pricing_vllm_nvfp4_trtllm_minimax_phase127`)

| 子系统 / 机制 | 修复前状态 | Phase 127 架构升级后 | 量化验证结果 |
|---|---|---|---|
| SGLang 统一 KV 子池分配器与容量定价 | 3 套独立分配器 + 调度层手工折算 Mamba 槽位 | 单一 `MultiEndedAllocator` + 集中准入与 HiCache 循环状态事件转发 | **超卖率 `12.5% -> 0.0%`，未就绪 Mamba 读取 `0`** |
| vLLM 投机解码多 DP 草稿图填充 (`5118d990`) | 仅同步 `uniform_token_count`，缺 `max_query_len` | `DPSyncState` 同步 `max_query_len` 门控变长图 | **跨 DP CUDA Graph 形状失配 `0`** |
| vLLM DSv4.1 分块 NVFP4 KV Gather (`6abdff59`) | 逐 Token (`TILE=1`) 算术解码 (`48.0 us`) | `TILE=16` + FP16 位级 `2^-14 * 2^14` 精确解码 (`27.38 us`) | **`1.753x` 提速** |
| TRT-LLM MiniMax-M3 NVFP4 加载 (`1f9b7072`) | 缺失 `compressed-tensors` 映射且未校准专家除零 | 共享 `convert_compressed_tensors_nvfp4_weights` + 零缩放保护 | **消除 `53` 行冗余与除零 `NaN`** |
