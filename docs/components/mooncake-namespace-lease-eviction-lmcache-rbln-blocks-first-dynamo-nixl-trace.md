# Mooncake Store 命名空间与免读锁租约驱逐索引、LMCache vLLM-RBLN 6-D 块优先 KV 布局及 Dynamo 文本工作节点 KV 路由门禁 / NIXL 显式时间相位事件

**English Companion**: [mooncake-namespace-lease-eviction-lmcache-rbln-blocks-first-dynamo-nixl-trace.en.md](mooncake-namespace-lease-eviction-lmcache-rbln-blocks-first-dynamo-nixl-trace.en.md)

## 1. 背景与架构问题

在多租户分布式 KV 存储（Mooncake Store）、异构 NPU 卸载（LMCache + Rebellions RBLN）与分离式推理路由/追踪（NVIDIA Dynamo & NIXL）中，近期上游提交解决了四项核心架构问题：
1. **Mooncake Store 多租户对象路由与读路径驱逐索引锁争用**：如果每次读请求延长租约（Lease）都去更新全局驱逐索引，高并发读取会在索引锁上产生严重缓存行争用；同时单租户默认模式下每次查询若经过 `shared_mutex` 哈希表并拷贝 `shared_ptr` 句柄，会污染共享缓存行。
2. **LMCache vLLM-RBLN 默认注意力内核的 6-D 块优先 KV 布局与 DMA 次数翻倍**：vLLM-RBLN 默认内核（`rbln_custom_ops`）将每层 KV 缓存分配为 6-D 块优先 HND 张量 `[num_blocks, 2, num_kv_heads, 1, block_size, head_size]`（轴 3 恒为单维 `1`），而此前 LMCache 仅注册了 `rbln_triton_ops` 的 K/V 优先格式 `NL_X_TWO_NB_NH_ONE_BS_HS = 15` (`[2, NB, NH, 1, BS, HS]`)。此外，K/V 优先布局中 `layer[:, block]` 跨首轴不连续，每个 `(block, layer)` 必须拆成 K、V 两次 `memcpy_v2v`。
3. **Dynamo 文本输入工作节点误启用 KV 感知路由**：当 vLLM 工作节点开启 `--use-vllm-tokenizer`（`ModelInput::Text`）时，请求绕过了 Dynamo 前端分词器，若此时路由器处于 KV 模式（`RouterMode::KV`），前端因缺少 Token ID 序列无法计算前缀哈希。
4. **NIXL 后端追踪相位丢失插件时间戳与通信方向语义**：此前 `TracerPhaseSink::recordPhase` 将所有异步传输相位（含 `NOTIF_SENT`、`NOTIF_RECEIVED`）一律记为调用时刻的 `Kind::Metadata` Span，并将插件的真实单调时钟仅作为普通属性附加，导致导出后端无法按真实发生时刻排序或配对收发通知。

## 2. 一手源码实现路径（Implementation Audit）

### 2.1 `kvcache-ai/Mooncake` (`d855a6aed9c8d1343af008b2cd9b9f4257d62a9b`, `#4721`, `2026-10-20`)
- **`mooncake-store/include/metadata/{namespace.h,namespace_table.h,lease_eviction_index.h,eviction_index.h,namespace_policy.h,object_route.h}`**：
  - `NamespaceTable` 用永不销毁的 `Namespace` 替代引用计数句柄，并将 `default_namespace_` 缓存在 `std::atomic<Namespace*>` 中，使单租户默认解析实现 **0 锁获取、0 共享缓存行写入**；
  - `LeaseEvictionIndex`（桶宽 `kBucketWidth = 100ms`）在构造时声明 `observes_reads = false`：延长租约的读操作完全不触发索引回调（`OnRead` 跳过）；某键在索引中的归档桶号永远不晚于其真实到期时间。当驱逐游标（`Scan(now)`）扫描到桶 `<= now_bucket` 时，在键写锁下调用 `StillAt` 核对真实租约：若已被读操作延期，则在释放写锁时自动重归档（Refile）至新桶，仅驱逐真实到期的键。

### 2.2 `LMCache/LMCache` (`462006abb998d4e4cb4209b14eb3d2db74a931b7`, `#5748`, `2026-10-20`)
- **`csrc/engine_kv_format.h`, `lmcache/v1/gpu_connector/kv_format/detectors/vllm.py`, `lmcache/v1/platform/devices/rbln/{kv_layout.py,kv_ops.py}`**：
  - 新增一级枚举 `EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS = 19`（`is_layer_list = true, is_hnd = true`），纯靠 6-D 形状签名（`ndim == 6 and shape[1] == 2 and shape[3] == 1`）完成设备无关探测，并在 `shape[0] == 2 and shape[1] == 2`（`num_blocks == 2` 歧义）时发出告警；
  - 在 `kv_ops.py` 的 `_hnd_block_pairs` 中，因块优先布局下 `layer[block]`（形状 `[2, H, BS, D]`）将 K 与 V 连续存放，每个 `(block, layer)` 仅需 **1 次连续 `memcpy_v2v` 视图配对**（$B \times L$ 次，相比 K/V 优先 `_hnd_kv_first_block_pairs` 的 $2 \times B \times L$ 次直接减半 **`-50.0%`**）。

### 2.3 `ai-dynamo/dynamo` (`1b6fee46c88b0f7339ac19423150d0613c82fa38`, `#16486`) & `ai-dynamo/nixl` (`705e0fc4f76993e9bbc5d39ed372db23c0c79e39`, `#2460`)
- **Dynamo `lib/llm/src/discovery/watcher.rs`**：在 `ModelWatcher` 服务发现阶段执行 `anyhow::ensure!(card.model_input != ModelInput::Text || !router_mode.is_kv_routing(), ...)`，无论 `KV` 路由来自前端全局配置还是模型卡覆盖，均在发现期立即拒绝 `ModelInput::Text` 工作节点；
- **NIXL `src/core/tracing/{trace.h,trace_sink.cpp}`**：新增 `TraceBackend::emitSpan` 与 `emitEvent`，`TracerPhaseSink::recordPhase` 不再打开 `CorrelationScope`，而是以插件 `steady_clock` 时间戳 `timestamp_us` 直接发射点事件，并将 `NOTIF_SENT` 映射为 `Kind::CommSend`、`NOTIF_RECEIVED` 映射为 `Kind::CommRecv`。

## 3. 确定性实验复算（`own-offline`）

运行 `python experiments\mooncake_namespace_eviction_lmcache_rbln_dynamo_nixl\run.py --check`：

| 子系统 | 关键核验指标 | 修复前 / 基线 | 修复后 (`Phase 37`) | 证据分级 |
|---|---|---|---|---|
| Mooncake `LeaseEvictionIndex` | 10,000 次读延期索引写锁 / 260ms 游标扫描 | 每次读更新索引锁 | **`0` 次读索引锁；驱逐 2 键、延迟重归档 `k_read_extended -> bucket 6`** | `implementation` + `own-offline` |
| LMCache vLLM-RBLN 6-D HND | 格式枚举 / `B=16, L=32` `memcpy_v2v` 对数 | 仅格式 `15` (`1,024` 次拷贝) | **新增块优先格式 `19`，拷贝对数降至 `512` (`-50.0%`)** | `implementation` + `own-offline` |
| Dynamo `ModelWatcher` | `ModelInput::Text` + `RouterMode::KV` 发现门禁 | 运行时路由静默失效 | **发现期立即拒绝 (`False`)，支持卡片级 `RoundRobin` 覆盖 (`True`)** | `implementation` + `own-offline` |
| NIXL `TracerPhaseSink` | 异步相位时间戳与 `NOTIF_SENT/RECEIVED` 类型 | 调用时刻 `Metadata` Span | **插件 `steady_clock` `at_us` + `CommSend` / `CommRecv` 事件** | `implementation` + `own-offline` |
