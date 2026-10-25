# SGLang Push-Plane TP All-Reduce + Residual + RMSNorm 融合、DSA CP KV Gather 与 Indexer 重叠、QSA 单次启动 Gather、EAGLE HiCache 双元边界及 vLLM 批分片自适应采样验证与 TRT-LLM MiniMax-M3 VL

## 1. 工程问题定义与核心架构矛盾

在 DeepSeek-V3/V4、Qwen3-MoE 与 MiniMax-M3 VL 等大模型生产推理中，推理引擎层面的尾延迟与正确性取决于跨层边界融合、稀疏注意力（DSA/QSA）索引准备以及投机解码/KV 卸载的边界契约：

1. **层边界通信与归一化多次读写 HBM**（`sgl-project/sglang@d558ac2f3e7a`，`#46137`）：在 Attention 输出进入 FFN、FFN/MoE 输出进入下一层 Attention 以及最终 `final_norm` 处，若分别启动 MoE Finalize、CustomAllReduce、残差相加与 RMSNorm，会产生 3–4 次全局内存往返。
2. **DSA 交错上下文并行（Interleave CP）中 KV All-Gather 与 Indexer 串行阻塞**（`sgl-project/sglang@5971f47bc8ea`，`#46179`）：在长序列 Prefill/Extend 阶段，CP KV 聚合通信与 DSA Indexer Top-K 计算原本串行执行，浪费了双流重叠窗口。
3. **QSA Prefill 索引器按序列循环调用 `index_select` + `torch.cat` 的启动风暴**（`sgl-project/sglang@dba22e9b68dd`，`#45752`）：对批次内 $N$ 条序列逐一执行 `compressed_buffer.index_select(0, compressed_locs)` 再拼接，产生 $N+1$ 次小算子启动。
4. **PD 分离解码端 HiCache 恢复时漏计 EAGLE Bigram 边界 Token**（`sgl-project/sglang@0ff67ee72aaf`，`#45076`）：EAGLE 树以 $N+1$ 个 Token ID 索引 $N$ 个 Bigram，若 `max_prefix_len` 仅传 `decode_prefix_len` 会少匹配 1 个边界 Token。
5. **vLLM 批分片采样（Batch-Sharded Sampling）与 GPU 自适应草稿验证（Adaptive Verification）冲突及 MRV2/KV Offload 边界缺陷**（`vllm-project/vllm@7050ef881ff0` `#58466`、`fc2b6ab45ed3` `#62698`、`2c19f6793e98` `#63372`、`65bc698356b4` `#63266`、`ceb01b6ddf6e` `#61976`）与 **TRT-LLM MiniMax-M3 VL 多模态模板重复渲染缺陷**（`NVIDIA/TensorRT-LLM@a0cee56d3a6e`，`#20626`）。

---

## 2. 第一手源码实现路径与机制审计

### 2.1 SGLang Push-Plane TP All-Reduce + Residual + RMSNorm 融合 (`#46137`)

审计 `python/sglang/kernels/jit/csrc/distributed/all_reduce_fusion.cuh`、`python/sglang/kernels/ops/communication/all_reduce_fusion.py` 与 `custom_all_reduce_fusion.py`：

- **RowClusterTrait 几何映射**：每行宽度 `kHiddenDim` 按 16B 向量（8 个 BF16 元素）划分为 `kRowVecs = kHiddenDim / 8`。每个线程处理 `kVecs` 个 16B 向量，块内线程数由 `kBlockSize = kRowVecs / (kClusterSize * kVecs)` 决定。
- **One-Shot 与 Two-Shot 切换策略（`H = 6144`，B300 TP4 / H200 TP8）**：
  - 阈值：`TWO_SHOT_MIN_TOKENS = {4: 48, 8: 8}`。
  - 当 `tokens < TWO_SHOT_MIN_TOKENS[ws]` 时采用 One-Shot（`vecs_per_thread = 1, cluster_size = 4`）。
  - 当 `tokens >= TWO_SHOT_MIN_TOKENS[ws]` 时切换至 Two-Shot，并根据是否包含 MoE Finalize 选择寄存器驻留友好的 `(max_rows, vecs_per_thread, cluster_size)`：
    - 无 Finalize (`plain`)：`tokens ≤ 272` 用 `(1, 4)`（`192` 线程/CTA），`tokens > 272` 用 `(2, 3)`（`128` 线程/CTA）。
    - 含 Finalize (`MoeDeferredFinalize`)：`tokens ≤ 208` 用 `(1, 4)`，`208 < tokens ≤ 232` 用 `(1, 3)`（`256` 线程/CTA），`tokens > 232` 用 `(2, 3)`（`128` 线程/CTA）。
  - 网格 `dim3(num_tokens + 1, kClusterSize)` 中的最后一个额外 Cluster（`blockIdx.x == num_tokens`）作为 **Bumper**，专门翻转剩余未使用的 Push 计数器 `[num_tokens, num_push_counters)`，保证 CUDA Graph 重放安全。

### 2.2 SGLang DSA CP KV Gather 重叠 (`#46179`)、QSA 单次 Gather (`#45752`) 与 HiCache Bigram 边界 (`#45076`)

- **DSA CP Interleave 重叠 (`python/sglang/srt/layers/cp/interleave.py`, `forward_mla.py`)**：
  - 新增 `DSAInterleaveCPIndexerOverlap`，在 `forward_absorb_prepare` 中当 `self.use_dsa and is_interleave_extend(forward_batch)` 时，在通信流启动 CP KV All-Gather 的同时并发执行 `self.indexer(...)`，将串行耗时 $T_{\text{kv}} + T_{\text{idx}}$ 压缩至 $\max(T_{\text{kv}}, T_{\text{idx}}) + \epsilon_{\text{sync}}$（离线模型在 8K–128K 下获得 **`1.743x`** 加速）。
- **QSA 单次 Triton 字节拷贝 Gather (`python/sglang/srt/layers/attention/qsa/kernel.py`)**：
  - `_gather_qsa_compressed_keys_kernel` 以 2D 网格 `(ceil(max_blocks / BLOCK_B), num_sequences)` 一次性讀取 `token_slot_table` 中每个压缩块首 Token 槽位 `first // RATIO`，按原始字节行宽 `ROW = heads * dim * element_size()` 直接拷贝至紧凑输出缓冲区，兼容 BF16 与 FP8，将 $N+1$ 次启动降为 **1 次**（128 序列下提速 **`17.56x`**）。
- **HiCache EAGLE Bigram 边界修复 (`decode_hicache_mixin.py`)**：
  - 新增 `_bigram_boundary_tokens(tree_cache) = int(tree_cache.tree_core.is_eagle)`，在 `_try_hicache_queue_load_back` 中设置 `rematch_token_len = pm.decode_prefix_len + _bigram_boundary_tokens(self.tree_cache)`，消除 PD 解码恢复时的 1-token 截断。

### 2.3 vLLM V1 调度/采样/卸载与 TRT-LLM MiniMax-M3 VL 审计

- **vLLM 批分片采样支持自适应草稿验证 (`#58466`, `7050ef881ff0`)**：移除 `_validate_batch_sharded_sampling` 中的拦截限制；`adaptive_verification.reallocate_drafts()` 返回 `num_admitted_draft_tokens`（`AdmittedDraftCounts`），异步拷贝至 Host 供 `BatchSharder` 构建精确匹配 GPU 实际准入草稿数的分片计划（TP=8、32 请求下采样工作量比上界缩减 **`16.0x`**）。
- **vLLM MRV2 2D 分块并行写 (`#62698`, `fc2b6ab45ed3`)**：`UvaBufferPool.apply_write` 将 `_apply_write_kernel` 网格从 `(n,)` 升级为 `(n, ceil(_staged_write_max_len / 1024))`，在 32 请求 × 64K 元素写更新下实现 **`16.0x`** SM 波次加速。
- **vLLM DBO 微批路由专家偏移 (`#63372`, `2c19f6793e98`)**：`RoutedExpertsCapturer.capture()` 使用 `offset = dbo_current_ubatch_token_offset()` 写入 `self.device_buffer[offset : offset + len(local_topk_ids), layer_id]`，修复第二个微批覆盖第一个微批专家 ID 的 Bug。
- **vLLM Chunked-Local KV Offload 后缀对齐检查 (`#63266`, `65bc698356b4`)**：在 `get_num_new_matched_tokens` 中当 `isinstance(spec, ChunkedLocalAttentionSpec) and num_computed_tokens % spec.attention_chunk_size != 0` 时拒绝 CPU 后缀命中，防止相对块边界与绝对位置错位。
- **vLLM 优先级自抢占后继续调度 (`#61976`, `ceb01b6ddf6e`)**：当最低优先级运行请求自抢占释放 KV 块后（`request.status == RequestStatus.PREEMPTED`），使用 `continue` 而非 `break`，允许后续更高优先级运行请求继续分配刚释放的块。
- **TensorRT-LLM MiniMax-M3 VL (`#20626`, `a0cee56d3a6e`)**：新增 `chat_template_applied: bool` 与 `supports_token_id_mm_expansion = True`，并在 KV 复用/Chunked Prefill 下通过 `find_input_mm_embeds` 仅切片当前 Chunk 内的多模态视觉嵌入行。

---

## 3. 可复现 CPU/Offline 实验与量化验证

```powershell
python experiments\sglang_push_allreduce_norm_dsa_cp_overlap_qsa_gather_vllm_batch_shard_spec_trtllm\run.py --check
```

| 引擎 / 特性 | 场景配置 | 修复/优化前 | 修复/优化后 | 关键指标 / 收益 | 证据分级 |
|---|---|---:|---:|---:|---|
| SGLang Push-Plane AR+Norm (`#46137`) | `H=6144, WS=8, T=224`, 含 Finalize | 4 次独立算子启动 | 单次 `two_shot` `(1, 3)` (`256` thr) | `225` Clusters (含 1 Bumper) | `implementation` + `own-offline` |
| SGLang DSA CP 重叠 (`#46179`) | `seq_len=131072, CP=4` | `2686.98 us` (串行) | `1541.49 us` (双流重叠) | **`1.7431x`** 延迟降低 | `implementation` + `own-offline` |
| SGLang QSA Gather (`#45752`) | `128` 序列 Prefill | `129` 次启动 (`492.9 us`) | **1 次** Triton 启动 (`28.24 us`) | **`17.454x`** 索引聚拢加速 | `implementation` + `own-offline` |
| vLLM Staged Writes (`#62698`) | `32` 请求 × `65536` 元素 | 1D Grid `(32,)` (`64` 波) | 2D Grid `(32, 64)` (`4` 波) | **`16.0x`** 波次缩减 | `implementation` + `own-offline` |
| vLLM Adaptive Batch-Shard (`#58466`) | `TP=8, B=32`, 上界 5 草稿 vs 准入 2 草稿 | `192` Logits (拒绝分片) | 每 Rank **`12` Logits** | **`16.0x`** 单卡采样缩减 | `implementation` + `own-offline` |

---

## 4. 架构师设计练习与可证伪问题

1. **几何推导题**：在 SGLang `#46137` 中，当 `kHiddenDim = 6144`、`world_size = 8`、`tokens = 220` 且带有 `MoeDeferredFinalize` 时，求选中的 `(vecs_per_thread, cluster_size)` 以及每个 CTA 的线程数 `kBlockSize` 和 Warp 数。
   - **解答**：因 $220 \ge \text{TWO\_SHOT\_MIN\_TOKENS}[8]=8$，查 `_TWO_SHOT_GEOMETRY[True]` 表得 $208 < 220 \le 232$ 命中 `(232, 1, 3)`，故 `vecs_per_thread = 1, cluster_size = 3`。每行 16B 向量数 `kRowVecs = 6144 / 8 = 768`，因此 `kBlockSize = 768 / (3 * 1) = 256` 线程（即 **8 个 Warp**）。
2. **可证伪问题**：对于包含 `ChunkedLocalAttentionSpec(attention_chunk_size=512)` 的混合模型，当 GPU 已缓存前 `256` 个 Token 时，能否直接从 CPU Offload 缓冲区加载后续后缀块？
   - **验证**：不能。`256 % 512 != 0`，后缀块在 CPU 缓存中是以块起点相对偏移计算的，直接拼接在 `256` 之后会破坏局部注意力窗口的绝对 512 对齐边界，`get_num_new_matched_tokens` 必须返回 `(0, False)`。
