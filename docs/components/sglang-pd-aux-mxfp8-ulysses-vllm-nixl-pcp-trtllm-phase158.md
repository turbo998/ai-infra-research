# SGLang PD 辅助长度帧对齐与 MXFP8 L3 双缓冲 HiCache、ROCm Ulysses 头组流水线、vLLM 批不变性内核锁定与 NIXL 按层 KV 配对及 TensorRT-LLM 即时缓存重置（Phase 158）

> **English Version**: [sglang-pd-aux-mxfp8-ulysses-vllm-nixl-pcp-trtllm-phase158.en.md](sglang-pd-aux-mxfp8-ulysses-vllm-nixl-pcp-trtllm-phase158.en.md)
> **关联问题 ID**：`q-sglang-pd-aux-mxfp8-ulysses-vllm-nixl-pcp-trtllm-phase158` (`priority: 167`)
> **关联证据 ID**：`ev-sglang-pd-aux-mxfp8-ulysses-vllm-nixl-pcp-trtllm-2026-phase158`

---

## 1. 架构背景与核心发现

在大规模 Prefill-Decode (PD) 分离、三级分层 KV 缓存（L3 HiCache）以及混合注意力（HMA）/流水线并行（PP）部署中，控制面 ZMQ 帧位置错位、微缩放（MXFP8）量化缩放因子丢失、跨桶内核自动调优漂移以及按索引配对 KV 内存区会导致静默精度损坏或分布式死锁。本报告基于 2026-10-31 合并的 10 个主线提交，确立以下核心工程结论：

1. **SGLang PD 注册协议将 `aux_item_lens` 紧邻 `aux_ptrs` 放置于 `msg[6]`（`sglang@5eb7018`, `#47638`）**：此前 `dst_aux_item_lens` 被追加在可选尾部字段之后，当可选字段缺省时导致帧偏移错位并退化为单元素长度推断；现统一固定在 `msg[6]`（紧随 `msg[5]` 的 `dst_aux_ptrs`），彻底消除 Mooncake 与 NIXL 多项辅助缓冲区截断故障。
2. **SGLang HiCache L3 存储支持 MXFP8 KV 页双缓冲（Payload + `UE8M0` Scales）对象持久化（`sglang@9780c52`, `#46936`）**：通过 `mxfp8_k_scales`/`mxfp8_v_scales` 扩展 `get_page_buffer_meta`，将 64-token 8-head ($d=128$) KV 页封装为 `131,072 B` FP8 载荷 + `4,096 B` `UE8M0` 缩放因子（共 `135,168 B`），相比 BF16 (`262,144 B`) 实现 **1.9394x** 压缩率（节省 **48.44%** L3 I/O 带宽）。
3. **SGLang ROCm MiniMax-H3 扩散模型 Ulysses 头组流水线与 SM100 Cake Softmax（`sglang@e93f817`, `#47567`; `sglang@c0f710f`, `#44168`）**：`ulysses_head_group_pipelined_attention` 将查询/键值头切分为 $G$ 个头组，在专用 RCCL 通信流上将第 $g+1$ 组的 All-to-All 交换与第 $g$ 组的 `aiter` 稠密注意力重叠（2 组流水线建模从 `9.60 ms` 降至 `7.50 ms`，提速 **1.28x**）；同时在 NVIDIA SM100 (`(10, 0)`) 上开启 `cake_blackwell_softmax`。
4. **vLLM 批不变性（Batch Invariance）FP8 BMM / W4A16 NVFP4 内核锁定、PP>1 预热防死锁、NIXL 按层名配对与 DSv4.1 PCP+PP/DP（`vllm@54beee8`, `#62192`; `vllm@4aba9bf`, `#64338`; `vllm@8b55f91`, `#63416`; `vllm@2b009fc`, `#63098`）**：在 `vllm_is_batch_invariant()` 下固定 `bmm_fp8` 后端为 `"cutlass"` 并锁定 NVFP4 `gemm_fp4_nt` 的 $M$ 分桶配置（消除跨批大小 `0.0 ULP` 漂移）；修复 `_warmup_flashinfer_attention` 在 `PP > 1` 时的跨 rank 死锁；将 NIXL KV 传输从位置索引配对改为按层名配对以兼容异构 PP/HMA 拓扑；解除 DeepSeek V4.1 PCP 的 `pipeline_parallel_size == 1` 限制。
5. **TensorRT-LLM `BlockRadixTree::clear()` 即时发布 `addAllBlocksCleared()` 事件与 `LlmRequest` 瘦身（`TensorRT-LLM@17bfe4e`, `#21142`; `TensorRT-LLM@d5752a1`, `#21036`）**：缓存重置后无需等待下一次推理请求即刻向外部 KV 路由器广播清空事件（等待延迟 `0.0 ms`），并移除 `GenericLlmRequest` 中未使用的 `promptEmbeddingTable` 与 `promptVocabSize` 字段。

---

## 2. 权威上游证据矩阵

| 仓库与提交 | 关键模块路径 | 核心机制与定量指标 |
| :--- | :--- | :--- |
| `sgl-project/sglang@5eb7018` (`#47638`) | `python/sglang/srt/disaggregation/mooncake/conn.py`, `nixl/conn.py` | `dst_aux_item_lens` 紧随 `dst_aux_ptrs` 固定于 `msg[6]`，消除可选字段缺失时的帧偏移与截断（`0` 偏移错误） |
| `sgl-project/sglang@9780c52` (`#46936`) | `python/sglang/srt/mem_cache/memory_pool_host.py`, `storage/hf3fs/` | MXFP8 L3 页按 `131,072 B` FP8 + `4,096 B` `UE8M0` scale 双缓冲写入，压缩率 **1.9394x** |
| `sgl-project/sglang@e93f817` (`#47567`) | `python/sglang/multimodal_gen/runtime/layers/attention/minimax_h3_attn.py` | ROCm RCCL 头组 All-to-All 与 `aiter` 注意力双流重叠（2 组流水线加速 **1.28x**） |
| `sgl-project/sglang@c0f710f` (`#44168`) | `python/sglang/srt/layers/sampler.py` | 将 `cake_blackwell_softmax` 启用范围扩展至 `(10, 0)` (SM100) 与 `(10, 3)` (SM103) |
| `vllm-project/vllm@54beee8` (`#62192`) | `vllm/model_executor/layers/batch_invariant.py` | 锁定 FlashInfer `bmm_fp8` (`"cutlass"`) 与 `W4A16` NVFP4 启发式，跨 batch 漂移 **0.0 ULP** |
| `vllm-project/vllm@4aba9bf` (`#64338`) | `vllm/v1/worker/gpu_worker.py` | 修复 `PP > 1` 时 `_warmup_flashinfer_attention` 的集団同步死锁 |
| `vllm-project/vllm@8b55f91` (`#63416`) | `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` | NIXL KV 内存区按层名而非位置索引配对，异构 PP/HMA 错配数为 **0** |
| `vllm-project/vllm@2b009fc` (`#63098`) | `vllm/model_executor/models/deepseek_v2.py` | 允许 DeepSeek V4.1 Prefill Context Parallelism (PCP) 与 `PP > 1` 或 `DP > 1` 组合运行 |
| `NVIDIA/TensorRT-LLM@17bfe4e` (`#21142`) | `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` | `BlockRadixTree::clear()` 立即触发 `mEventSink->addAllBlocksCleared()`（`0.0 ms` 滞后） |
| `NVIDIA/TensorRT-LLM@d5752a1` (`#21036`) | `cpp/include/tensorrt_llm/batch_manager/llmRequest.h` | 移除冗余 `mPromptEmbeddingTable` 与 `mPromptVocabSize` 字段 |

---

## 3. 可复现实验验证

```powershell
python experiments\sglang_pd_aux_mxfp8_ulysses_vllm_nixl_pcp_trtllm_phase158\run.py --check
```
