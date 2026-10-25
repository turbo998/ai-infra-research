# SGLang DSV4 HiSparse 逐 Token Eager 备份、DP-Attention Prefill 分桶与 Decode Custom-AR 对齐、Diffusion IPC `/dev/shm` 清理、vLLM Speculator 动态 `max_model_len` 与流式 KV Offload 重置及 TensorRT-LLM `ClientOSError` 重试

> **文档状态**：Phase 77（`priority: 86`）第一手源码审计 + CPU/offline 合成验证  
> **面向读者**：负责大规模在线推理服务（SGLang、vLLM、TensorRT-LLM）稳定性、分层 KV Cache 卸载、DP-Attention CUDA Graph 与推测解码的解决方案架构师  
> **English Companion**: [sglang-hisparse-dsv4-dp-attn-bucket-diffusion-ipc-vllm-spec-autofit-kv-reset-trtllm.en.md](sglang-hisparse-dsv4-dp-attn-bucket-diffusion-ipc-vllm-spec-autofit-kv-reset-trtllm.en.md)

---

## 1. 背景与生产故障根因

在 DeepSeek-V4（DSV4）稀疏注意力（HiSparse）、多卡 Data-Parallel Attention（DP-Attention）、多模态/扩散模型本地 IPC 调度以及流式会话 KV 卸载生产环境中，近期暴露了六类隐蔽的数据损坏、死锁与资源泄漏隐患：

1. **SGLang DSV4 HiSparse 解码批次恰好整除 `slot_page_size` 时的静默 KV 损坏（`#45104`）**：DSV4 C4 量化 KV 页采用页内分块布局 `[value0..value63][scale0..scale63]`（`slot_page_size = 64`）。当 `HiSparseCoordinator._eager_backup_previous_token` 在 decode 阶段备份每个请求新增的 1 个离散 token 时，若并发请求数恰好是 64 的整数倍（例如 64 个请求分散在页 10 与页 25），旧逻辑调用 `backup_from_device_all_layer` 并因 `numel() % slot_page_size == 0` 误判为“整页连续备份”，仅按首元素拷贝了页 10 的整页数据，导致其余离散槽位（50% tokens）的 value 与 scale 完全错位。
2. **SGLang DP-Attention Prefill CUDA Graph 分桶失配与 Decode Custom All-Reduce 模式分叉（`#46138`）**：在 `prepare_mlp_sync_batch` 中，当启用 Prefill CUDA Graph 且处于 `DpPaddingMode.MAX_LEN` 时，旧代码仅将各 rank token 数设为 `max(global_num_tokens)`（如 `17`），未向上对齐到已捕获的 CUDA Graph bucket（如 `24`）；同时在 Decode 阶段，`get_default_mode_in_cuda_graph` 硬编码返回 `MAX_LEN`，未与 `get_dp_padding_mode` 的 `_decode_sum_len_fits_custom_all_reduce` 判定对齐；此外空闲 DP rank（`0` real tokens）仍执行了冗余的 `init_forward_metadata`。
3. **SGLang Diffusion 本地 `ipc://` 调度器在异常/超时路径下的 `/dev/shm` 共享内存泄漏（`#46154`）**：大于 `_MIN_FILE_REF_BYTES`（`32 MiB`）的 NumPy/Torch 张量通过 `/dev/shm` 临时文件引用传递。若响应在消费前发生超时、取消或反序列化损坏，临时文件将永久残留在 `/dev/shm` 中。
4. **vLLM Speculator 在 `max_model_len` 自动缩减（Auto-Fit）后的长度越界（`#60513`）**：`EagleProposer` / `Speculator` 在构造时将 `vllm_config.model_config.max_model_len` 拷贝为静态实例属性；当后续 KV 显存探测将 `model_config.max_model_len` 自动下调（如 `32768 -> 8192`）时，Speculator 仍持有旧值 `32768`。
5. **vLLM 流式会话（`resumable=True`）跨 Chunk 恢复时的 KV Offload 块表重复追加（`#60810`）与 DBO 微批次乱序报错（`#63476`）**：流式会话第 2 个 chunk 作为 `scheduled_new_reqs` 恢复时携带完整累积 `block_ids`（如 `[0, 1, 2, 3, 4]`），但旧代码 `yield_req_data` 将 `is_full_table` 设为 `False`，导致处理器将其追加到第 1 个 chunk 已有的 `[0, 1]` 之后变成 `[0, 1, 0, 1, 2, 3, 4]`，进而将旧块 `[0, 1, 2]` 当作新块重复卸载；而在双微批次（DBO）执行器中，`min(errors)` 总是报告编号最小的 microbatch 0（被动死锁在 recv hook），掩盖了最先崩溃的 microbatch 1 根因。
6. **TensorRT-LLM OpenAI 基准客户端漏判 `aiohttp.ClientOSError(ECONNRESET)`（`#20889`）**：高并发压测下服务端主动 reset 连接常表现为直接抛出 `aiohttp.ClientOSError(errno.ECONNRESET)`（无 `__cause__`），旧逻辑仅检查 `isinstance(e.__cause__, ConnectionResetError)`，导致未能启用 `_TRANSIENT_TCP_BUDGET = 5` 重试预算。

---

## 2. 第一手源码审计与修复机制

### 2.1 `sglang@c00e7e12b64b`（`#45104`）：`backup_tokens_from_device_all_layer` 逐 Token 备份契约

在 `python/sglang/srt/mem_cache/storage/hisparse/hisparse_storage.py` 与 `hisparse_coordinator.py` 中：

- 新增 `HiSparseNSATokenToKVPoolHost.backup_tokens_from_device_all_layer(device_pool, host_locs, device_locs)`，无条件调用逐 token 内核 `transfer_cache_dsv4_mla`（按每个 token 槽位独立解析其所在页内的 value 偏移与 scale 偏移），不再执行 `device_locs.numel() % self.slot_page_size == 0` 的整页分支判定。
- `HiSparseCoordinator._eager_backup_previous_token` 改为调用 `backup_tokens_from_device_all_layer`，彻底消除 decode 批次大小恰好为 `64, 128, ...` 时的 C4 页布局错位。

### 2.2 `sglang@9b230c1dae7e`（`#46138`）：DP-Attention Prefill Bucket 对齐、Decode `SUM_LEN` 与空闲 Rank 跳过

在 `python/sglang/srt/Batch_info.py`、`dp_padding.py` 与 `model_runner.py` 中：

1. **Prefill CUDA Graph 向上分桶**：在 `prepare_mlp_sync_batch` 中，当 `DpPaddingMode.MAX_LEN` 与 Prefill CUDA Graph 同时启用时，计算 `target_tokens = min((size for size in prefill_cg.bs if size >= max_tokens), default=max_tokens)`，确保所有 `sync_group_size` ranks 填充到同一个已捕获的 CUDA Graph bucket（如 `17 -> 24`）。
2. **Decode Custom All-Reduce 一致性**：`DpPaddingMode.get_default_mode_in_cuda_graph(global_num_tokens_cpu)` 与 `get_dp_padding_mode` 统一检查 `_decode_sum_len_fits_custom_all_reduce(global_num_tokens)`，只要总 token 字节数不超过 Custom All-Reduce 阈值（512 KiB），一致返回 `SUM_LEN`。
3. **空闲 DP Rank 跳过注意力元数据初始化**：当 `global_num_token_non_padded_cpu == 0` 且后端声明 `skips_prefill_metadata_on_idle_dp_rank = True` 时，直接跳过 `attn_backend.init_forward_metadata(forward_batch)`。

### 2.3 `sglang@c05fb1bd04d2`（`#46154`）：`array_ipc_request` 作用域化 `/dev/shm` 回收

在 `sglang/multimodal_gen/runtime/entrypoints/ipc_utils.py` 中新增上下文管理器 `with array_ipc_request(endpoint) as identity:`：

- 为每次本地 `ipc://` RPC 在 `/dev/shm` 下创建独立目录 `sgldiffusion-request-*`，并通过线程局部上下文与 `zmq.IDENTITY = identity` 将服务端产生的 `NumpyArrayFileRef` / `TorchTensorFileRef`（`> 32 MiB`）定向写入该请求目录。
- 在 `finally` 块中执行 `shutil.rmtree(req_dir, ignore_errors=True)`，经单元测试验证在 `success`、`bad_pickle`、`bad_array`、`timeout_before`、`timeout_after`、`cancel_before`、`cancel_after` 全部 **7 种结果分支**下均实现 **0 字节残留**。

### 2.4 `vllm@636da2476d18`（`#60513`）、`921ed7fa11c4`（`#60810`）、`66bf93154420`（`#63476`）与 `TensorRT-LLM@315f04f2250f`（`#20889`）

- **Speculator 动态 `@property max_model_len`（`vllm#60513`）**：移除构造期静态拷贝，改为 `@property def max_model_len(self) -> int: return self.vllm_config.model_config.max_model_len`，并在 `_build_attn_metadata` 中传入 `max_seq_len=min(self.draft_max_seq_len, self.max_model_len)`。
- **流式会话 KV Offload `is_full_table=True` 重置（`vllm#60810`）**：`yield_req_data` 对 `scheduled_new_reqs` 与 preempted cached reqs 统一标记 `is_full_table=True`；`OffloadingHandler` 在 `is_full_table` 为真时先执行 `.clear()` 再 `.extend(...)`，确保流式 chunk 2 准确卸载新增块 `(2, 3, 4)` 而非重复卸载 `(0, 1, 2)`。
- **DBO 微批次按时间顺序抛出首个异常（`vllm#63476`）**：`ubatch_utils.py` 将 `failed = min(errors)` 改为 `failed = next(iter(errors))`，利用 Python `dict` 插入顺序精确上报首个崩溃的微批次。
- **TensorRT-LLM `ClientOSError(ECONNRESET)` 纳入瞬态 TCP 预算（`#20889`）**：`tensorrt_llm/serve/openai_client.py` 将 `(isinstance(e, aiohttp.ClientOSError) and e.errno == errno.ECONNRESET)` 纳入 `is_transient_tcp`，赋予 `effective_max = max(max_retries, 5)`，而对 `errno.EPIPE` 保持 `0` 重试快速失败。

---

## 3. 合成实验验证结果

通过运行 `experiments/sglang_hisparse_dsv4_dp_attn_bucket_diffusion_ipc_vllm_spec_autofit_kv_reset_trtllm/run.py --check` 得到如下验证矩阵：

| 验证场景 | 修复前行为（Legacy） | 修复后行为（Phase 77 Audited） | 核心不变量校验 |
|---|---|---|---|
| DSV4 HiSparse 64-req Decode 备份 (`#45104`) | 误走整页拷贝，损坏 **32 / 64 (`50%`)** tokens | `backup_tokens_from_device_all_layer` 逐 token 拷贝，**0 损坏** | `corrupted_tokens_fixed == 0` |
| DP-Attention Prefill Bucket & Decode AR (`#46138`) | `max_len=17` 未对齐 bucket；Decode 误选 `MAX_LEN` | 向上对齐至 bucket **`24`**；Decode 选 **`SUM_LEN`**；空闲 rank 跳过 init | `fixed_target_bucket == 24` |
| Diffusion IPC `/dev/shm` 7 分支清理 (`#46154`) | 4 个异常/超时后分支共泄漏 **`256 MiB`** | `array_ipc_request` 作用域回收，**`0 MiB` 泄漏** | `fixed_total_leaked_mib == 0` |
| vLLM Speculator Auto-Fit & 流式 KV (`#60513`, `#60810`) | Speculator 滞留 `32768`；Chunk 2 误存 `[0, 1, 2]` | 动态同步 **`8192`**；Chunk 2 准确卸载 **`[2, 3, 4]`** | `fixed_chunk2_stored_blocks == [2, 3, 4]` |
| DBO 微批次报错 & TRT-LLM 重试 (`#63476`, `#20889`) | 误报 `ubatch 0`；`ECONNRESET` 0 次重试即失败 | 报首个崩溃 `ubatch 1`；`ECONNRESET` 获 **5 次**重试预算 | `fixed_effective_max_retries == 5` |

---

## 4. 架构师设计练习与后续可证伪问题

1. **容量与可靠性练习**：在并发度为 128 的 SGLang Diffusion 服务节点上（`/dev/shm` 容量 `8 GiB`），若每张生成图触发 `64 MiB` 的 `TorchTensorFileRef` 落盘且客户端超时率为 `0.5%`，计算在未合入 `#46154` 前需要多少次请求就会耗尽 `/dev/shm` 并引发全节点 `ENOSPC` 崩溃。
2. **可证伪问题**：
   - **H1**：在 DSV4 HiSparse 预填充阶段，当请求长度同时包含整页（64 tokens 的倍数）与尾部非整页余数时，分离整页 DMA 与尾部 `transfer_cache_dsv4_mla` 相比全量逐 token 拷贝能降低多少 PCIe 事务开销？
   - **H2**：在 8 卡 DP-Attention 下，当各 rank 预填充 token 数方差极大（如 `[0, 0, 0, 512]`）时，`DpPaddingMode.MAX_LEN` 向上对齐至 bucket `512` 的冗余计算成本在何种阈值下劣于退回 Eager 非填充执行？
