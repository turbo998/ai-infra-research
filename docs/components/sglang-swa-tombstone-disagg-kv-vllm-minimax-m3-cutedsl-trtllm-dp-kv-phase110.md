# SGLang SWA 墓碑前缀钳制与非阻塞分离式 KV 恢复、vLLM Hopper MiniMax-M3 稀疏预填充与 CuteDSL BF16x3 路由 GEMM、TensorRT-LLM Attention-DP KV 连接器与 Logprobs 显存优化（Phase 110）

> English version: [sglang-swa-tombstone-disagg-kv-vllm-minimax-m3-cutedsl-trtllm-dp-kv-phase110.en.md](sglang-swa-tombstone-disagg-kv-vllm-minimax-m3-cutedsl-trtllm-dp-kv-phase110.en.md) | 关联实验：`experiments/sglang_swa_tombstone_disagg_kv_vllm_minimax_trtllm_phase110/run.py`

## 1. 一手代码证据（2026-10-27）

| 仓库 | Commit | 日期 | 核心机制变更 | 证据等级 |
|---|---|---|---|---|
| `sglang` | `e7c55f6959` | 2026-10-27 | `UnifiedRadixCache` SWA 墓碑恢复无条件执行 `window_start = max(evicted_seqlen, prev_prefix_len)`，防止非独立组件释放请求受保护前缀内的有效页面 | `implementation` |
| `sglang` | `b28f4fc065` | 2026-10-27 | 分离式解码通过 `start_restore_kv_cache`、独立 `index_stream` 与 `restore_done_event` 异步恢复主机暂存 KV，消除调度器同步阻塞并在拷贝完成后归还主机槽位 | `implementation` |
| `sglang` | `b7ac4c0e72` | 2026-10-27 | Rust `sglang-radix-tree` 使用 `pyo3::sync::MutexExt::lock_py_attached` 执行 `has_ongoing_insert`/大小/摘要等 `O(1)` 读取，无争用时保持 GIL 零切换 | `implementation` |
| `vllm` | `8e26dec5f5` | 2026-10-27 | MiniMax-M3 在 Hopper (`SM90`) 新增查询分块 Triton 稀疏预填充核（`_PREFILL_TILE_Q = 16`，`SPARSE_BLOCK_SIZE = 128`），实现每 `16` 个查询共享一次选中 KV 页加载 | `implementation` |
| `vllm` | `0076619a94` | 2026-10-27 | Hopper (`SM90`) 新增 CuteDSL `BF16x3` 路由 GEMM，在寄存器中将 FP32 路由权重拆分为 `3` 个 BF16 项并通过 `wgmma` 与分布式共享内存保序 Split-K 归约保持 FP32 精度 | `implementation` |
| `vllm` | `5c1eb11cb3` | 2026-10-27 | Qwen4Exp PLE 与 MTP drafter 支持流水线并行 `PP > 1`（非首 stage 使用 `PPMissingLayer()`，MTP 按 `intermediate_tensors is None` 分支，KV 规划跳过非本 stage 的 uniform 组） | `implementation` |
| `vllm` | `8977b13a1c` | 2026-10-27 | DeepSeek-V4.1 稀疏 MLA 与 Compressor 元数据构建器从 `AttentionCGSupport.ALWAYS` 调整为 `UNIFORM_BATCH`（`max_decode_query_len`），将非图安全稀疏预填充排除在 FULL CUDA Graph 之外 | `implementation` |
| `vllm` | `cebcc5d8c7` | 2026-10-27 | Model Runner V2 (`MRV2`) 在权重池外初始化 `self.model_state`，防止 `sleep_mode` 卸载权重池时清空 Mamba2/EPLB 运行时设备状态张量 | `implementation` |
| `TensorRT-LLM` | `0a03c3ed05` | 2026-10-27 | V2 KV Cache Connector 支持 Attention DP（每个 ADP owner rank 运行独立 `KvCacheConnectorScheduler`，携带 `attention_dp_rank` 与取消哨兵检查，并在 `_failed_kv_connector_executors` 隔离异常 DMA 生命周期） | `implementation` |
| `TensorRT-LLM` | `8354c4c398` | 2026-10-27 | Sampler logprobs 仅对 `need_processed_logprobs` 请求分配全词表 `[rows, V]` 临时张量；`raw_logprobs` 通过 Triton `selected_raw_logprobs` 直接从 logits 采样（`B=128, V=151936` 省去 `74.19 MiB` 临时显存） | `implementation` |

## 2. 关键实现与架构师结论

1. **SWA 前缀安全与非阻塞 KV 恢复**：当 `evicted_seqlen = 256 < prev_prefix_len = 512` 且 `total_prefix_len = 384` 时，未钳制的墓碑恢复会错误释放 `128` 个受保护前缀 token 页面；钳制后损坏页数降为 `0`。同时分离式解码通过 `index_stream` 独立翻译索引并用 `restore_done_event` 门控入批，将 8 请求批次的调度器阻塞从 `1440 us` 降至 `0 us`。
2. **Hopper 算子分块与采样器显存复杂度降维**：MiniMax-M3 查询分块稀疏预填充以 `_PREFILL_TILE_Q = 16` 复用每个 `128-token` 稀疏 KV 页，降低 `16x` 重复 HBM 读取；TensorRT-LLM 将仅需 `raw_logprobs` 的临时显存从 `O(B \cdot V)`（`74.19 MiB`）降为 `0.0 MiB` 全词表分配。
