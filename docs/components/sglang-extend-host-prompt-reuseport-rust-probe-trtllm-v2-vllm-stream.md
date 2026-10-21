# SGLang 宿主端 Extend 完成标记与 TBO 继承、`SO_REUSEPORT` 多进程分词监听、Rust 响应环健康探针应答、TensorRT-LLM KV V2 预热清理与多模态 Responses API 及 vLLM 流处理器共享

> **Phase 55 (`priority: 64`) · 一手源码与 Commit 级深挖**
> - `sgl-project/sglang@a3d00eff58b6c22ea7539f882cc10c537634200b` (`#45487`)
> - `sgl-project/sglang@7e5256d1ba3aaa628bb8b756fafb904a76a1123a` (`#45482`)
> - `sgl-project/sglang@ca72dad953171ea4f580d89f70f999fc89501067` (`#45475`)
> - `sgl-project/sglang@68623cae9256ad8ee00269c8c1d52e4c2915daa5` (`#45670`)
> - `NVIDIA/TensorRT-LLM@1c7fadcfbe50911d784658e902d1bd003d89fdd7` (`#20253`) & `271ed9f22ab85e5ce2b45e326f5b41cbb249bd93` (`#20231`)
> - `vllm-project/vllm@6ae1051bd9120c9e2974a8ad502b4c9716db6b9f` (`#62741`) & `5ea2c7ff1e64b36b45bf6bd502367c85633b2d8c` (`#62706`)

---

## 1. SGLang 宿主端 `contains_last_prefill_chunk` 与 Mamba 乒乓轨道收集 (`#45487`, `#45670`)

- **宿主端记录 Extend 批次是否完成 Prompt (`#45487`)**：在 `ForwardBatch.init_new` 中，当 `batch.forward_mode.is_extend_without_speculative()` 为真时，排除已在解码的 `decoding_reqs`，并在宿主端计算是否存在请求满足 `req.extend_end == len(req.full_untruncated_fill_ids)`，记录为 `contains_last_prefill_chunk`；同时在双批次重叠（TBO `filter_batch`）切分子批次时继承该标志，避免仅含中间分块（Chunked Prefill 未到末尾）的批次触发不必要尾处理。
- **Mamba 乒乓缓冲区 `track_position` 索引收集 (`#45670`)**：在混合批次 Mamba 状态追踪中，通过 `sb._gather_mamba_track_indices(ping_pong_buffers, req_pool_indices, track_positions)` 按每个请求的 `track_position ∈ {0, 1}` 从双槽乒乓缓冲区解析真实物理轨道索引。

---

## 2. SGLang `SO_REUSEPORT` 监听器与 Rust 响应环健康探针应答 (`#45482`, `#45475`)

- **每 Worker 独立 `SO_REUSEPORT` 监听 (`#45482`)**：升级 `uvicorn>=0.51` 并引入 `--enable-http-worker-reuseport`，让每个 Uvicorn HTTP/Tokenizer Worker 拥有独立的内核级 `SO_REUSEPORT` 监听套接字，消除单共享套接字在多 Worker 下的惊群与负载倾斜。
- **Rust Server 繁忙调度器健康探针应答 (`#45475`)**：在长上下文预填充期间，当调度器在 `maybe_send_health_check_signal()` 中检测到 `self.rust_server is not None` 时，不再向未被消费的 ZMQ Tokenizer 套接字发送 `HealthCheckOutput`，而是清空 `return_health_check_ipcs` 并调用 `self.rust_server.push_health_ack()` 向 Rust 响应环推入单帧 `frame_health_ack()`，一次性应答所有被繁忙步骤跳过的健康探针。

---

## 3. TensorRT-LLM KV V2 预热清理 / 多模态 Responses API 与 vLLM 前后端流收敛 (`#20253`, `#20231`, `#62741`, `#62706`)

- **TRT-LLM 废弃预热路径清理与多模态 Responses API (`#20253`, `#20231`)**：`#20253` 移除动态草稿长度下额外捕获最大投机 CUDA Graph 的历史变通方案（`original_max_draft_len`）及 `Rocket` 稀疏缓存的 `prepare_resource` 测试开关；`#20231` 修复 Responses API 中 `_create_input_tokens` 误返回 `(mm_data, mm_embeddings)` 二元组的问题，支持 `input_image` 多模态输入。
- **vLLM Rust `max_tool_calls` 与共享 `ChatStreamProcessor` (`#62741`, `#62706`)**：`#62741` 新增 `--exclude-tools-when-tool-choice-none`，并在 Rust `StructuredEventState` 中以 `max_tool_calls: Option<usize>`（`parallel_tool_calls=false` 时为 `Some(1)`）精确拦截后续多余工具调用；`#62706` 提取 `ChatStreamProcessor`，统一 OpenAI 流式聊天与在线 Derenderer 的逐 Choice 增量过滤与 `finish_reason` 判定。

---

## 4. 确定性复现实验验证

运行 `python experiments/sglang-extend-host-prompt-reuseport-rust-probe-trtllm-v2-vllm-stream/run.py --check` 全部不变量通过。
