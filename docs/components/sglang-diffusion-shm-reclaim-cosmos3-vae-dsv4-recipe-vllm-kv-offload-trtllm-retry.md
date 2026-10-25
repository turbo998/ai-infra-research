# SGLang Diffusion 共享内存回收、Cosmos3 多卡视频 VAE 与 DeepSeek-V4 Rust 渲染，以及 vLLM KV Offload 与 TRT-LLM 容错审计

> **English Edition**: [sglang-diffusion-shm-reclaim-cosmos3-vae-dsv4-recipe-vllm-kv-offload-trtllm-retry.en.md](sglang-diffusion-shm-reclaim-cosmos3-vae-dsv4-recipe-vllm-kv-offload-trtllm-retry.en.md) | **关联问题**: `q-sglang-diffusion-shm-reclaim-cosmos3-vae-dsv4-recipe-vllm-kv-offload-trtllm-retry` (`priority: 91`, Phase 82) | **可复现实验**: `experiments/sglang_diffusion_shm_reclaim_cosmos3_vae_dsv4_recipe_vllm_kv_offload_trtllm_retry/run.py`

---

## 1. 架构背景与核心工程矛盾

在多模态生成（Diffusion / Video VAE）与大规模 LLM 分布式推理（SGLang / vLLM / TensorRT-LLM）的生产服务栈中，控制面与存储卸载层存在四类高隐蔽性的资源泄漏与状态失步故障：

1. **大张量 IPC 溢出共享内存 (`/dev/shm`) 在请求取消/超时后的永久泄漏**：SGLang Diffusion 的 `array_ipc_request` 在回复张量超过 `32 MiB` 时会落盘到共享内存文件；若多个并发 RPC 复用固定 ZMQ DEALER identity 或在客户端取消/异常中断时未排空延迟到达的回复包，溢出的 SHM 文件将永远无法 unlink。
2. **多卡视频 VAE 解码加速与图生视频 (I2V) 编码种子逐比特一致性的冲突**：Cosmos3 的 3D 因果卷积 VAE 在多卡 (`num_gpus >= 2`) 下可通过 `channels_last_3d` 与 `use_vae_fast_path` 显著加速视频帧解码，但如果对 `encoder.*` 同样施加 `channels_last_3d`，cuDNN 卷积舍入差异会改变 I2V 潜变量噪声种子。
3. **vLLM 磁盘/外部 KV Offload 在抢占重算 (Preemption)、并发原子重命名 (`EEXIST`) 与惰性迭代器异常下的状态损坏**：当请求发生 preemption（`num_computed_tokens` 归零）后重新计算时，若 `group_state.next_stored_chunk_idx` 仍停留在被抢占前已落盘的 chunk 游标，前序被驱逐的 KV 块将被永久跳过不再写出。
4. **TensorRT-LLM 异步基准客户端对 `aiohttp.ClientOSError` (`ECONNRESET`) 的漏捕**：高并发长连接压测下服务端主动复位 TCP 连接会抛出 `aiohttp.ClientOSError`，此前未被纳入 5 次指数退避重试捕获列表。

---

## 2. 第一手源码实现路径审计

### 2.1 SGLang Diffusion 每请求独立 `zmq.IDENTITY` 与溢出 SHM 全路径回收 (`c05fb1bd04d2`, `#46154`)

审计 `sgl-project/sglang` 的 `python/sglang/multimodal_gen/runtime/utils/array_ipc.py`：

- **每请求独立 DEALER Identity**：在 `array_ipc_request()` 中为每次 RPC 分配唯一标识 `req_identity = f"sgldiffusion-request-{uuid.uuid4().hex}".encode("ascii")` 并设置 `sock.setsockopt(zmq.IDENTITY, req_identity)`，彻底隔离不同 RPC 的 ROUTER 回包路由。
- **全生命周期 SHM 回收 (`_reclaim_spilled_shm_from_frames`)**：当回复负载 `>= 32 MiB` 触发共享内存溢出文件时，无论 RPC 正常返回、抛出异常还是被客户端协程取消 (`asyncio.CancelledError`)，均在清理路径解析回复帧元数据并回收对应的 SHM 段，消除 `/dev/shm` 累积耗尽导致的宿主机 OOM。

### 2.2 SGLang Cosmos3 多卡视频 VAE 解码优化 (`97220dd67437`, `#46173`) 与 DeepSeek-V4 Rust 模板渲染 (`1406ffda099f`, `#46243`)

- **Cosmos3 仅解码器 `channels_last_3d` + `use_vae_fast_path` (`#46173`)**：
  - 审计 `cosmos3_vae.py` 与 `cosmos3.py`：仅将 `vae.decoder` 子树的 3D 卷积权重与激活转换为 `torch.channels_last_3d`（显式跳过 `vae.encoder`，保证 I2V 首帧编码潜变量与随机种子在单卡/多卡间逐比特一致）。
  - 在 `Cosmos3Pipeline` 中，当 `server_args.num_gpus >= 2` 且 `quality_allows` 时自动启用 `channels_last_3d` 与 `use_vae_fast_path`。
- **Rust 原生 `deepseek-recipe = 0.2.0` 渲染 (`#46243`)**：
  - 审计 `rust/sglang-processor/Cargo.toml`、`src/processor.rs` 与 `sgl-model-gateway`：引入 `deepseek-recipe = "0.2.0"`，新增 `ChatTemplateContentFormat::DeepSeekV4` 与 `DeepSeekV4_1`，通过 `ChatFormatter::encode_request()` 直接在 Rust 层将多轮对话与工具调用渲染并分词为 `Vec<u32>` token IDs，绕过 Python Jinja2 瓶颈。

### 2.3 vLLM KV Offload 三项存储一致性修复 (`1757f6720c3c` `#61390`, `6f51433838b8` `#63010`, `503e4e16ad8d` `#63126`) 与 TRT-LLM 重试 (`315f04f2250f` `#20889`)

- **内容寻址 KV 块原子 `rename(2)` `EEXIST` 幂等处理 (`vllm` `#61390`)**：在文件系统 KV Offload 后端中，两个 worker 并发写出同一内容哈希的 KV 块时，后完成者在 `os.replace(tmp_path, final_path)` / `rename(2)` 上可能收到 `EEXIST` (`FileExistsError`)。修复后将其视为落盘成功（返回 `0`）并安全 `unlink(tmp_path)`，消除伪写失败告警与临时文件残留。
- **抢占重算后重置 `group_state.next_stored_chunk_idx = 0` (`vllm` `#63010`)**：在 `vllm/v1/core/kv_cache_coordinator.py` 中，当请求因显存压力被 preempt 并从 `num_computed_tokens == 0` 重新调度时，若 `group_state.next_stored_chunk_idx > 0`，立即将其重置为 `0`，确保被驱逐的前序 KV chunk 在重算后被完整重新卸载存储。
- **`batch_lookup` 惰性生成器在 `try` 块内显式物化 (`vllm` `#63126`)**：在远程/磁盘 KV connector 中将 `return self.batch_lookup(keys, req_context)` 改为 `return list(self.batch_lookup(keys, req_context))`，确保惰性迭代器在执行期间抛出的 `OSError` / 超时异常被当前 `try...except` 块正确捕获并降级处理。
- **TensorRT-LLM `openai_client.py` 扩展 `ClientOSError` 重试 (`TensorRT-LLM` `#20889`)**：在 `benchmarks/cpp/utils/openai_client.py` 的 5 次重试循环 (`max_retries = 5`, 指数退避 `2**attempt` 秒，累计退避 `1 + 2 + 4 + 8 = 15s`) 中将 `aiohttp.ClientOSError` 纳入捕获元组，妥善恢复 `Connection reset by peer` (`ECONNRESET`)。

---

## 3. 可复现实验与验证 (`own-offline`)

运行 `python experiments/sglang_diffusion_shm_reclaim_cosmos3_vae_dsv4_recipe_vllm_kv_offload_trtllm_retry/run.py --check` 的验证结果如下：

| 机制 / 故障场景 | 修复前行为 | 修复后契约 | 关键定量结果 |
|---|---|---|---|
| SGLang Diffusion `64` 次取消的 `48 MiB` RPC 回复 (`#46154`) | 泄漏 `3,072 MiB` `/dev/shm` 文件 | 独立 `zmq.IDENTITY` + 全路径 unlink | 泄漏 **`0 MiB`** (`-100%`) |
| SGLang Cosmos3 多卡 (`num_gpus=4`) Video VAE (`#46173`) | 全图转换破坏 I2V 编码种子或未启用加速 | 仅 `decoder.*` `channels_last_3d` + `use_vae_fast_path` | 编码器种子误差 **`0 ULP`** |
| vLLM 抢占重算 (`6` chunks 中已存 `4` chunks 后被驱逐, `#63010`) | `next_stored_chunk_idx=4`，跳过前 `4` 个 chunk | `num_computed_tokens==0` 时重置为 `0` | 完整写出 **`6 / 6` chunks** |
| vLLM `16` 路并发重复 KV 块原子 `rename(2)` (`#61390`) + 惰性 `batch_lookup` (`#63126`) | `16` 次 `FileExistsError` 异常 + 惰性 `OSError` 逃逸 | `EEXIST -> 0` + 清理 `tmp_path` + `list(...)` 物化 | 伪错误 **`0`**，临时文件残留 **`0`** |
| TensorRT-LLM `openai_client.py` 瞬态 `ECONNRESET` (`#20889`) | `ClientOSError` 直接中断基准压测 | 5 次尝试 (`1, 2, 4, 8s` 指数退避) | 容忍最长 **`15s`** 瞬态连接抖动 |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **Kubernetes Pod `/dev/shm` 容量规划计算**：Kubernetes 默认 `emptyDir { medium: Memory }` 或容器默认 `/dev/shm` 常为 `64 MiB`（或按节点内存比例配置）。若单次高分辨率视频生成 RPC 回复为 `48 MiB`，请计算在未应用 `#46154` 补丁时，多少次客户端超时取消即可打满 `2 GiB` 的 `/dev/shm` 挂载卷（答案：$\lceil 2048 / 48 \rceil = 43$ 次）。
2. **可证伪假设**：在内容寻址（SHA-256 / 块哈希命名）的文件系统 KV 缓存中，同一路径下的目标文件字节内容由键值唯一决定，因此 `os.replace(tmp_path, dst_path)` 遇到 `EEXIST` 时无需比对文件内容即可幂等视为写入成功，只需删除本地未命名的 `tmp_path`。
