# Mooncake RDMA 数据面 Auto-GID 纪元恢复、LMCache 进程内 GPU Connector 隔离校验与 NIXL UCX 闭合词汇表阶段追踪

> **English Version**: [mooncake-auto-gid-recovery-reshard-inflight-lmcache-nixl-ucx-spans.en.md](mooncake-auto-gid-recovery-reshard-inflight-lmcache-nixl-ucx-spans.en.md)  
> **证据等级**: `implementation`（上游一手 commit 源码） + `own-offline`（`experiments/mooncake_gid_recovery_lmcache_inproc_nixl_ucx/run.py --check`）

## 1. 一手源码定位与核心问题

1. **Mooncake TransferEngine 在 `IBV_WC_RETRY_EXC_ERR` 下基于纪元令牌的 Auto-GID 自动恢复**（`kvcache-ai/Mooncake` commit [`5ea6cf0d95d3a69b4ba196d28bf1f96e28ec5e10`](https://github.com/kvcache-ai/Mooncake/commit/5ea6cf0d95d3a69b4ba196d28bf1f96e28ec5e10)，`2026-10-20`）：
   - **代码路径**：`mooncake-transfer-engine/include/transport/rdma_transport/rdma_gid_probe.h`、`worker_pool.cpp`（`performPostSend`、`processCompletions`、`recordAutoGidDataPathFailure`）及 `transport.h`（`RdmaCompletionToken`）。
   - **问题机理**：在多宿主容器或双栈 RoCEv2 网络中，建链阶段选中的初始 GID（如过期 IPv4 别名或不可达子网 GID）在数据面发送切片（Slice）时才会暴露 `IBV_WC_RETRY_EXC_ERR`。旧版 `WorkerPool::processCompletions` 一旦遇到该错误便直接调用 `markRailFailed(slice->peerNicPath(), true)` 拉黑整条物理网卡 Rail；而且若同一批次有 16 个并发 Slice 同时失败，无纪元去重的 GID 轮转会瞬间耗尽所有候选 GID。
   - **修复机制**：为每个投递的 Slice 绑定 `RdmaCompletionToken{endpoint_id, selection_epoch, valid, observe_success}`。当首个携带 `selection_epoch == current_epoch` 的失败完成事件到达时，`recordAutoGidDataPathFailure` 返回 `AutoGidDataPathAction::kAdvanceGid` 将端点推进到下一个 `AutoGidCandidateClass` 候选 GID（`epoch + 1`），并将同批次其余 15 个携带旧 `selection_epoch` 的失败事件标记为 `kIgnoreDuplicate` 忽略，从而在不误杀健康网卡 Rail 的前提下原地恢复数据面连接。

2. **LMCache 进程内（In-Process）GPU Connector 物理隔离与前置设备特性矩阵校验**（`LMCache/LMCache` commit [`df19cd68cd959510b36de88c72b526232359505d`](https://github.com/LMCache/LMCache/commit/df19cd68cd959510b36de88c72b526232359505d)，`2026-10-20`）：
   - **代码路径**：`lmcache/inprocess/v1/gpu_connector/__init__.py`（`_validate_vllm_device_features`）、`lmcache/v1/gpu_connector/__init__.py` 与 `lmcache/integration/vllm/vllm_service_factory.py`。
   - **问题机理**：此前 `lmcache/v1/gpu_connector` 混杂了供独立进程（Standalone/MP）使用的布局工具与强耦合 vLLM 进程内指针的 `VLLMPagedMemGPUConnectorV2/V3`、MUSA/XPU/HPU 连接器。在非 CUDA/XPU 设备（如 MUSA、HPU）上误开启 `enable_blending`、`use_gpu_connector_v3` 或在 HPU 上开启 `use_layerwise=True` 时，工厂函数会落入默认的 CUDA 分支，触发深层 `torch.cuda.Stream()` 崩溃。
   - **修复机制**：将所有进程内连接器迁移至独立的 `lmcache/inprocess/v1/gpu_connector/` 包，解除 `lmcache/v1/gpu_connector` 对进程内运行时的依赖；并在 `CreateGPUConnector` 入口通过 `_validate_vllm_device_features` 提前拦截不支持的加速器与特性组合，抛出清晰的 `ValueError`。

3. **NIXL UCX 后端闭合词汇表阶段追踪（`submit` / `wire.submitted` / `notif.sent`）与轮询偏置隔离**（`ai-dynamo/nixl` commit [`8d5660c118cf2d5249edd83d9afc311cdda2e43d`](https://github.com/ai-dynamo/nixl/commit/8d5660c118cf2d5249edd83d9afc311cdda2e43d)，`2026-10-20`）：
   - **代码路径**：`src/plugins/ucx/ucx_backend.cpp`（`sampledTraceContext`、`recordPhase`、`postXfer`）、`src/plugins/ucx/ucx_backend_req.h` 与 `docs/tracing.md`。
   - **问题机理**：在异步 RDMA 传输中，应用层仅在调用 `checkXfer` 轮询状态时才会感知传输完成。若在 `checkXfer` 发现完成时打点 `nixl::wire.completed`，会把上层推理循环的轮询间隔（如 `170+ us`）算进底层网络传输耗时，造成跨后端（UCX vs Libfabric）性能对比严重失真。
   - **修复机制**：UCX 仅在采样请求（`opt_args->traceContext->sampled()`）上发射发送端三阶段事件：`nixl::submit`（`postXfer` 接收请求）、`nixl::wire.submitted`（所有 `put`/`get` 与 `flush` 成功投递至 UCX 队列；若对端断连则仅记录 `submit`）以及单次 `nixl::notif.sent`，配合接收端 `nixl::notif.received`，明确不记录 `nixl::wire.completed` 以消除轮询延迟污染。

## 2. 确定性实验验证（`experiments/mooncake_gid_recovery_lmcache_inproc_nixl_ucx`）

运行 `python experiments\mooncake_gid_recovery_lmcache_inproc_nixl_ucx\run.py --check` 的量化结果：

| 子方向 | 关键指标 | 旧行为（Legacy） | 修复后（Patched） | 架构收益 |
|---|---|---:|---:|---|
| Mooncake Auto-GID 纪元恢复 (`5ea6cf0d`) | 16 并发 Slice 遇 `IBV_WC_RETRY_EXC_ERR` 后的恢复率 | `0 / 16`（整条 Rail 被拉黑） | `16 / 16`（`epoch 1->2`，忽略 `15` 个旧纪元重复报错） | 避免容器/双栈 RoCEv2 假死 GID 误杀健康网卡通道 |
| LMCache In-Process 隔离 (`df19cd68`) | 非法加速器特性组合前置拦截 / 深层 Stream 崩溃 | `0 / 3` 拦截 / `3` 次崩溃 | `3 / 3` `ValueError` 前置拦截 / `0` 崩溃 | CPU-only / Standalone 导入彻底解耦，多芯片配置错误秒级定位 |
| NIXL UCX 阶段追踪 (`8d5660c1`) | 轮询延迟对真实线速耗时的放大倍数 | `195.5 us` vs `23.5 us`（**`8.3191x` 失真**） | 仅记录 `submit`/`wire.submitted`/`notif.sent` | 彻底剥离应用轮询间隔与 RDMA 硬件投递耗时 |
