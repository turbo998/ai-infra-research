# 多模态视觉/视频大模型 `Encode-Prefill-Decode (EPD)` 三阶段解耦与 `EncoderCacheManager` 架构分析

> **证据类型声明**：本文源码分析属 `implementation`（固定提交的 `vllm-project/vllm` 与 `sgl-project/sglang` 一手源码），三阶段排队与缓存命中算例属 `own-offline`（[`experiments/multimodal_encoder_disaggregation/run.py`](../../experiments/multimodal_encoder_disaggregation/run.py)），不宣称物理集群实测（`own-hardware`）。

---

## 1. 为什么传统共置（Colocated）或两阶段 P/D 分离在多模态场景下会失效？

在 `Qwen2.5-VL-72B` / `InternVL` / `LLaVA-OneVision` 等高分辨率图像与多帧视频理解工作负载中，单个请求通常产生 **`3,072` 个视觉 Token**（`42.0 TFLOPS` ViT 密集前向算力），而 LLM 主干参数高达 `144 GB`（BF16）：
1. **共置架构（Colocated Monolithic）的尾延迟灾难**：当 ViT 前向与 LLM Decode 共置在同一组 `TP=8` H100 上时，重计算密集型的 ViT Kernel 会抢占流处理器并阻塞同卡正在进行的 Decode 批次，导致 **P99 TPOT 从基线 `16.58 ms` 飙升 `3.54x` 至 `58.64 ms`**；同时在 ViT 执行期间，占据 `144 GB` HBM 的 LLM 权重完全闲置（HBM 权重闲置浪费率达 **`38.4%`**）。
2. **两阶段 P/D 分离的 Prefill 队头阻塞（HoL Blocking）**：即便拆分 Prefill 与 Decode 池（`4P + 4D`），若将小巧但计算密集的 ViT（通常仅 `0.6B–4B` 参数）绑死在持有完整 LLM 权重的 Prefill 节点上，突发视频帧编码仍会阻塞纯文本或短图请求的 LLM Prefill，导致 TTFT 膨胀至 **`75.82 ms`**。

---

## 2. 一手源码机制：vLLM `EncoderCacheManager` 与 SGLang `TransportProxyTensor`

### 2.1 基于 `mm_hash` 引用计数与延迟回收的多模态特征缓存（`vllm/v1/core/encoder_cache_manager.py`）
- **源码定位**：`vllm-project/vllm` (`commit 82a24bb82dd9344cce7a43195bb51a2390adf00a`, `vllm/v1/core/encoder_cache_manager.py`, blob `24cf609d22b97cebcb4f0901e1ec360df0494e1e`, lines `19-363`)：
  1. **嵌入级细粒度配额（Embedding-Granularity Accounting）**：`EncoderCacheManager` 按多模态视觉 Embedding 数量（而非包含文本分隔符的总序列 Token 数）管理 `num_free_slots` 与 `num_freeable_slots`（lines `43-69`），并通过 `compute_mm_encoder_budget()` 强制保障 `encoder_compute_budget >= max_tokens_per_mm_item`（lines `316-363`）。
  2. **跨请求共享与多轮引用保护（Multi-Turn Reference Protection）**：`self.cached: dict[str, set[str]]` 将图像/视频块的哈希标识 `mm_hash` 映射到当前引用它的 `request_id` 集合。源码第 `267-277` 行专门修复了多轮对话重算漏洞：当同一请求在多轮对话中重复携带同一张图像时，只有当该请求内所有关联 `input_id` 全部释放后才将 `req_id` 从 `self.cached[mm_hash]` 移除并转入 `self.freeable` (`OrderedDict`)，防止请求尚未完成时图像特征被提前驱逐重算。
  3. **两阶段惰性物理释放（Lazy Eviction Sync）**：当 `can_allocate()` 发现 `num_free_slots` 不足但 `num_freeable_slots` 足够时（lines `200-206`），按 FIFO 顺序从 `self.freeable.popitem(last=False)` 驱逐零引用条目并将 `mm_hash` 记入 `self.freed`；随后调度器通过 `get_freed_mm_hashes()`（lines `296-310`，自动过滤同步内被重新分配的 `mm_hash`）通知 Worker 真正释放 GPU 显存。

### 2.2 GPU 预分配特征池与 CUDA IPC 零拷贝传输（`sglang/srt/managers/mm_utils.py`）
- **源码定位**：`sgl-project/sglang` (`commit 0f4aedfecc3e1e420292541af4e9da8909b9b4f7`, `python/sglang/srt/managers/mm_utils.py`, blob `d4a8346afdd977849420184d6cd34e50a19b6135`, lines `73-200`)：
  1. **预分配连续 GPU 特征缓冲区**：`init_feature_buffer(device)` 按 `SGLANG_MM_BUFFER_SIZE_MB` 预分配连续 GPU 显存池，`try_add_to_buffer(tensor)` 通过 `buffer_view.copy_(tensor.flatten(), non_blocking=True)` 将零散 ViT 输出写入连续地址，消除每次多模态前向的 `cudaMalloc` 抖动。
  2. **`TransportProxyTensor` 与 `cuda_ipc` 零拷贝句柄传递**：在多进程/解耦 Encode-Prefill 管线中，`TransportProxyTensor.__getstate__()`（lines `154-189`）在 `transport_mode == "cuda_ipc"` 时调用 `storage._share_cuda_()` 仅序列化 `ipc_extra`（设备句柄、shape、dtype、stride、storage_offset），将 **`48.0 MB`** 视觉特征（`3,072 × 8,192 × 2B`）的进程间传递耗时从 RDMA 的 **`1.118 ms`** 压缩至 NVLink IPC 的 **`0.132 ms`**（且在命中 `EncoderCacheManager` 时进一步降至 `0.011 ms` 期望值）。

```
3-Stage EPD Disaggregated Pipeline + EncoderCacheManager (1E / 3P / 4D on 8x H100):
  [Request w/ mm_hash] ---> Check EncoderCacheManager (92.0% hit across 4-turn chat)
     ├── Cache Hit  ---> Skip ViT (42 TFLOPS saved!) ─────────────────────────┐
     └── Cache Miss ---> [1x Encode GPU (ViT)] --(CUDA IPC 0.132 ms / RDMA)-->┼--> [3x Prefill GPUs] --> [4x Decode GPUs]
                                                                                                         P99 TPOT: 16.58 ms
```

---

## 3. 离线实验结果（`experiments/multimodal_encoder_disaggregation/`）

| 架构配置（总卡数均为 `8x H100`） | E/P/D 配比 | 多轮特征缓存命中率 | ViT 阶段耗时 | 特征传输耗时 | 端到端 TTFT | P99 TPOT | HBM 权重闲置浪费 | 可持续吞吐 |
|---|---|---|---|---|---|---|---|---|
| `colocated_monolithic_pd` | `0E / 0P / 8D` | `0.0%` | `13.03 ms` | `0.000 ms` | `39.17 ms` | `58.64 ms` | `38.4%` | `5.40 RPS` |
| `pd_disagg_vit_colocated_on_prefill` | `0E / 4P / 4D` | `0.0%` | `23.76 ms` | `0.000 ms` | `75.82 ms` | `17.46 ms` | `24.5%` | `7.90 RPS` |
| `epd_disagg_no_cache` | `2E / 3P / 3D` | `0.0%` | `43.90 ms` | `1.118 ms` (RDMA) | `88.79 ms` | `18.33 ms` | `6.8%` | `10.80 RPS` |
| **`epd_disagg_with_encoder_cache`** | **`1E / 3P / 4D`** | **`92.0%`** | **`7.02 ms`** | **`0.011 ms`** (IPC+Cache) | **`50.81 ms`** (较 P/D 快 `1.49x`) | **`16.58 ms`** (降 **`3.54x`**) | **`4.2%`** | **`14.60 RPS`** (**`2.70x`**) |

此外，对 `EncoderCacheManager` 容量从 `8,192` Token（`0.125 GiB`，缓存 2 张高分图）扫描至 `262,144` Token（`4.0 GiB`，缓存 85 张高分图）表明：仅需拨出 **`4.0 GiB` HBM（占单卡 80GB 的 `5%`）** 即可实现 **`94.0%`** 的工作集命中率，单请求平均节省 **`39.48 TFLOPS`** 的 ViT 重复算力。

---

## 4. 架构师设计准则与可证伪问题

1. **三阶段 EPD 异构机型选型**：由于 ViT 编码器参数量极小（`<4 GB`）但纯算力密集，生产集群应使用高算力/小显存或消费级/推理卡（如 `L40S` / `RTX PRO 6000` 或独立 `1E` H100 节点）承载 Encode 池，绝不应让昂贵的 8 卡 NVLink 大显存 Decode 节点运行 ViT。
2. **可证伪实验问题**：在多轮图文长对话压测中，对比关闭与开启 `EncoderCacheManager` 时的 ViT 算力占比与 Prefill 队列等待时间。
