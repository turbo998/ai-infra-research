# Megatron-LM 最小显存 DSA 注意力与异步路由重放、Mooncake 悬空副本自愈、Dynamo 全局路由与 NIXL UCX SGL 分批审计（Phase 152）

> **证据分类**：源码实现级审计 (`implementation`) + 本项目标准库确定性离线算例 (`own-offline`)。

## 一、 10 项核心公开提交审计矩阵（Zero-Overlap）

| 编号 | 仓库 | Commit (`SHA`) | 日期 | 核心工程突破与代码路径 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `NVIDIA/Megatron-LM` | [`f87f092`](https://github.com/NVIDIA/Megatron-LM/commit/f87f0920c5c755779209f10b7b7d72b13f810771) | 2026-10-14 | `Support routing replay with asynchronous scheduling` (`#7310`)：在动态推理异步调度下支持确定性 MoE 路由重放 (`dynamic_context.py`)。 |
| 2 | `NVIDIA/Megatron-LM` | [`80781ab`](https://github.com/NVIDIA/Megatron-LM/commit/80781ab3d959a7203be3f8ef743d2df69c993c33) | 2026-10-14 | `fix(mtp): preserve metric reduction groups across graph replays` (`#7768`)：在 CudaGraph 重放期间保留 Multi-Token Prediction (MTP) 指标归约通信组。 |
| 3 | `NVIDIA/Megatron-LM` | [`1e3c0c6`](https://github.com/NVIDIA/Megatron-LM/commit/1e3c0c640431880cb5ee049e348573ec7d6336bc) | 2026-10-13 | `Add the min-memory DSA attention kernels` (`#7152`)：新增 `dsa_min_memory_triton.py` 最小显存 Dynamic Sparse Attention (DSA) 内核。 |
| 4 | `pytorch/pytorch` | [`bc7e151`](https://github.com/pytorch/pytorch/commit/bc7e151be297e99850b09dc3be15c95c165fa536) | 2026-10-30 | `dynamo: Add extra guards on dict length for correctness` (`#202374`)：在 `symbolic_convert.py` 中补齐字典长度 Guard，防止字典增删后误复用旧图。 |
| 5 | `pytorch/pytorch` | [`1dbd304`](https://github.com/pytorch/pytorch/commit/1dbd304877e1297dc2d3bb578cb17fb3d03bf094) | 2026-10-30 | `[precompile] A Dynamo artifact imports the modules that register its custom ops` (`#202258`)：预编译产物反序列化时自动导入注册自定义算子的模块。 |
| 6 | `kvcache-ai/Mooncake` | [`09a570d`](https://github.com/kvcache-ai/Mooncake/commit/09a570da6c9fd865b73943c3568fc6b2e5e2b546) | 2026-10-14 | `[Store] Self-heal dangling LOCAL_DISK replicas on the read path` (`#3889`)：读取路径检测到本地磁盘悬空副本时自动触发主控自愈清理。 |
| 7 | `kvcache-ai/Mooncake` | [`dae4828`](https://github.com/kvcache-ai/Mooncake/commit/dae4828071f2f0db49c4678791847df710663d48) | 2026-10-14 | `[Bugfix][Store] Skip replicas on unloaded segments during standby restore` (`#4553`)：备库恢复时跳过所在内存段尚未挂载的副本。 |
| 8 | `LMCache/LMCache` | [`2ef13a3`](https://github.com/LMCache/LMCache/commit/2ef13a3dc118fc4b7777d56a3dc77694fc541f46) | 2026-10-19 | `[Feat] Emit a per-request prompt trace span from the MP server` (`#5817`)：在多进程缓存服务端输出单请求粒度 OpenTelemetry Trace Span。 |
| 9 | `ai-dynamo/dynamo` | [`9f88a47`](https://github.com/ai-dynamo/dynamo/commit/9f88a474f0e1a4a9573be5fb11f5dd1744019806) | 2026-10-29 | `feat(global-router): Global Frontend with load- and session-aware pool selection` (`#17524`)：新增负载与会话亲和感知的 Global Frontend 跨池路由器。 |
| 10 | `ai-dynamo/nixl` | [`7077d02`](https://github.com/ai-dynamo/nixl/commit/7077d0230a1e6c993db81d251500c1f1a33eb602) | 2026-10-02 | `PLUGINS/UCX: Set split_batch_size=4096 in threadpool when SGL is enabled` (`#2329`)：开启 Scatter-Gather List (SGL) 时将 UCX 线程池切分批次设为 `4096`。 |

## 二、 确定性离线算例验证 (`own-offline`)

运行 `python experiments/megatron_dsa_routing_mooncake_heal_dynamo_ucx_phase152/run.py --check`：
- **Megatron 最小显存 DSA 激活显存节省**：`1536.0 MB -> 640.0 MB`（节省 **`58.33%`**）
- **Mooncake 悬空本地磁盘副本自愈**：`12/12` 自动清理恢复；备库恢复未挂载段崩溃：`3 -> 0`
- **Dynamo Global Frontend 会话/负载感知路由 TTFT 加速**：`340.0 ms -> 215.0 ms` (**`1.5814x`**)
- **NIXL UCX SGL `split_batch_size=4096` 加速比**：`285.0 us -> 192.0 us` (**`1.4844x`**)
