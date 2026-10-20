# Mooncake LocalHotCache 并发扫描防竞态、NVIDIA Dynamo 就地 Token 块 KV 哈希与 Prefill 激活期 Fail-Closed 路由深度解析

> **English Version**: [mooncake-hotcache-scan-race-dynamo-inplace-kv-hash-prefill-failclosed.en.md](mooncake-hotcache-scan-race-dynamo-inplace-kv-hash-prefill-failclosed.en.md) | **返回索引**: [README.md](../../README.md) | **7-Card 评测报告**: [ai-infra-phase33-benchmark-cards.html](../reports/ai-infra-phase33-benchmark-cards.html)

## 1. 一手源码变更矩阵（2026-10-20）

| 组件 | 仓库与 Commit SHA | 发布日期 | 核心文件与函数 | 证据级别 |
| :--- | :--- | :--- | :--- | :--- |
| **Mooncake Store Detached Hot-Cache Fill Key-Scan Race Fix** | `kvcache-ai/Mooncake` [`ac4d29bff2e8bb98c36799aaf19e6c73529f0088`](https://github.com/kvcache-ai/Mooncake/commit/ac4d29bff2e8bb98c36799aaf19e6c73529f0088) (`#4334`) | `2026-10-20` | `mooncake-store/src/local_hot_cache.cpp::LocalHotCache::ReleaseHotKey`, `hasActiveBlockForKeyLocked` | `implementation` + `own-offline` |
| **NVIDIA Dynamo In-Place Token-Block KV Hashing (`hash_complete_blocks`)** | `ai-dynamo/dynamo` [`1246a4bbd153de6d927cb074881afcc20d40ac15`](https://github.com/ai-dynamo/dynamo/commit/1246a4bbd153de6d927cb074881afcc20d40ac15) (`#15701`) | `2026-10-20` | `lib/tokens/src/lib.rs::hash_complete_blocks`, `lib/kv-hashing/src/compute.rs::Request::block_lineages` | `implementation` + `own-offline` |
| **NVIDIA Dynamo PrefillRouter Fail-Closed Activation & `WorkerUnavailable` (HTTP 503)** | `ai-dynamo/dynamo` [`7aaf17b62691a5dc1141f71da1d5bca72ce6667d`](https://github.com/ai-dynamo/dynamo/commit/7aaf17b62691a5dc1141f71da1d5bca72ce6667d) (`#16736`) | `2026-10-20` | `lib/llm/src/kv_router/prefill_router/activation.rs::ActivationSnapshot`, `lib/bindings/python/rust/errors.rs::routed_error_to_pyerr` | `implementation` + `own-offline` |

---

## 2. Mooncake `LocalHotCache` 分离填充块（Detached Fill）与 Key 扫描的数据竞态修复

### 2.1 `GetFreeBlock()` 锁外写 `key_` 与 `blocks_` 全表扫描的并发冲突
在 Mooncake Store 的本地热缓存（`LocalHotCache`）中，当 `SubmitPutTask` 异步填充一个热块时，工作线程首先调用 `GetFreeBlock()` 从尾部摘取一个空闲 `HotMemBlock`（此时 `block->ref_count == 0`）。为了避免大块数据拷贝与哈希填充阻塞全局 `mutex_`，填充线程在**不持有 `mutex_` 的情况下直接修改 `filling->key_`**，直到填充完成后再调用 `PutHotKey` 加锁发布并将 `ref_count` 置为正数。

然而，`ReleaseHotKey(const std::string& key)` 与 `hasActiveBlockForKeyLocked(const std::string& key)` 在持有 `mutex_` 遍历 `blocks_` 时，旧代码的短路判断顺序写成了：

```cpp
// 旧代码（存在 C++ std::string 读写数据竞态 / UB）：
if (block && block->key_ == key && block->ref_count > 0)
```

由于 `GetFreeBlock()` 从空闲链表尾部取块，正在锁外并发重写 `filling->key_ = std::string(...)` 的分离填充块仍然位于 `blocks_` 向量中！先执行 `block->key_ == key` 会直接读取一个正在被另一线程重新分配缓冲区的 `std::string`，触发未定义行为（UB / 段错误）。

### 2.2 短路谓词重排（`ref_count > 0` 优先）
Commit `ac4d29bff2e8`（`#4334`）将两处条件重排为：

```cpp
// 修复后：在 mutex_ 保护下先检查 ref_count > 0
if (block && block->ref_count > 0 && block->key_ == key)
```

因为所有分离填充中的块均满足 `ref_count == 0`，短路求值保证扫描线程永远不会触碰调用方独占拥有的 `block->key_`。在本仓库确定性并发扫描模型中（64 个总块含 16 个分离填充块、1,000 次扫描），未加锁 `key_` 竞态读取次数从 **`16,000` 次归零至 `0` 次**。

---

## 3. NVIDIA Dynamo 就地 Token 块 KV 哈希（`hash_complete_blocks`）

### 3.1 `TokenBlockSequence::new_with_mm` 的双重 Token 堆克隆开销
在 KV 路由器的每个请求入口，`dynamo-kv-hashing` 需要将请求 token 流切分为固定 `block_size` 的完整窗口，计算每块的 `BlockHash`、父链式 `SequenceHash` 与 `PositionalLineageHash (PLH)`。

在 Commit `1246a4bbd153`（`#15701`）之前，`Request::into_blocks` / `block_hashes` / `sequence_hashes` 均调用 `TokenBlockSequence::new_with_mm(Tokens::from(self.tokens.clone()), ...)`：
1. 首先完整克隆一次 `self.tokens`（$4 \times N_{\text{tokens}}$ 字节）；
2. 随后将每个完整窗口拆分为独立的 `TokenBlock` 结构体，每个结构体内部再次分配一个拥有所有权的 `Vec<Token>`（合计又分配 $N_{\text{blocks}}$ 次堆内存、再拷贝 $4 \times N_{\text{blocks}} \times \text{block\_size}$ 字节），而最终返回给路由器的 `UniversalBlock` 根本不需要保存 token 本身！

### 3.2 `hash_complete_blocks`：纯文本零拷贝 + 多模态滑动游标按需分配
Commit `1246a4bbd153` 在 `lib/tokens/src/lib.rs` 中新增了 `hash_complete_blocks`，通过单调递增的游标 `run_cursor` 扫描已排序且互不重叠的多模态区间 `mm_runs`：
- **纯文本窗口（无多模态重叠）**：直接在借用的切片 `&tokens[block_offset..block_end]` 上调用 `compute_block_hash_for_tokens` 就地计算 XXH3 哈希，**Token 堆拷贝与分配字节数为 `0`**。
- **多模态重叠窗口（`mm_runs[run_cursor].offset < block_end`）**：仅对该窗口调用 `compute_block_bytes_with_mm`，按每槽位 **13 字节**（1B tag + 4B token + 8B `mm_hash`，共 $13 \times \text{block\_size}$ 字节）分配带标签帧缓冲并哈希。

| 基准场景（`benches/request_hash.rs`） | Token 数 / 块大小 | 完整块数 / MM 重叠块数 | 旧版 Token 堆分配字节（分配次数） | 新版就地哈希堆分配字节（分配次数） | 节省比例 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `text_128k_block_128` | $131{,}072$ / $128$ | $1{,}024$ / $0$ | **$1{,}048{,}576\text{ B}$**（$1{,}025$ 次） | **$0\text{ B}$**（$0$ 次） | **$100.0\%$** |
| `text_8k_block_16` | $8{,}192$ / $16$ | $512$ / $0$ | **$65{,}536\text{ B}$**（$513$ 次） | **$0\text{ B}$**（$0$ 次） | **$100.0\%$** |
| `mm_mid_8k_block_16`（offset=4000, len=32） | $8{,}192$ / $16$ | $512$ / $2$ | **$65{,}952\text{ B}$**（$515$ 次） | **$416\text{ B}$**（$2$ 次） | **$99.37\%$** |

---

## 4. NVIDIA Dynamo PrefillRouter 激活窗口期 Fail-Closed 与 HTTP 503 保真

### 4.1 为什么分离式推理启动/滚动升级期会出现“请求误入 Decode”？
在 PD 分离部署中，Decode Worker 往往比 Frontend 完成 Prefill Router 构建早几百毫秒上线。旧版 `PrefillRouter` 使用无锁分散读取：只要 Prefill Router 尚未激活（`Pending`），就将请求按“无 Prefill 服务”直接透传（passthrough）给下游 Decode Worker。然而，纯 Decode 模式的 Worker 拒绝接收未经 Prefill 的原始请求，导致滚动发布期间出现不可恢复的请求报错。

### 4.2 `ActivationSnapshot` 原子快照与 `ErrorClass::WorkerUnavailable` 全链路穿透
Commit `7aaf17b62691`（`#16736`）进行了三层加固：
1. **锁下原子三元组快照 `activation_snapshot()`**：在 `self.target.lock()` 保护下一次性读取 `(target, lifecycle, binding)`，映射为三态枚举：
   - `Active(Arc<PrefillBinding>)`：正常路由至 Prefill（热路径在已发布 `binding` 时无需加锁）；
   - `Untargeted`：确实未注册任何 Prefill 目标（`target.is_none()`），安全透传至聚合 Decode；
   - `Pending`：`target.is_some()` 但 `binding` 仍在构建中 $\to$ **Fail Closed**，立即返回 `prefill_activation_pending_error`（`ErrorClass::WorkerUnavailable`）。
2. **防止 `ErrorClass::Unavailable` 归一化吞没 `WorkerUnavailable`**：`DynamoError::class()` 会将 `WorkerUnavailable` 归一化为 `Unavailable`，若直接依赖 `class()` 会丢失“可重试单路由未就绪”语义；`is_worker_unavailable()` 通过遍历错误因果链检查原始 `error.error_type() == ErrorClass::WorkerUnavailable`。
3. **跨 PyO3 / Python Processor 保真 HTTP 503**：`RoutedEngine.generate` 使用 `routed_error_to_pyerr` 抛出 `dynamo._core.WorkerUnavailable`，`vllm_processor.py` 与 `sglang_processor.py` 显式透传该异常，再由 Rust `worker_unavailable_to_dynamo` 还原为 HTTP **503 Service Unavailable**（而非被错误信封包装成 HTTP 500）。

---

## 5. 可复现实验

```bash
python experiments/mooncake_hotcache_race_dynamo_kvhash_prefill_gate/run.py --check
```
