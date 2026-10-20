# LMCache Device-DAX/CXL L1 偏移分区与 APC 命中归因、以及 Dynamo Cuckoo DC 快照删除重放与迁移 Token 账本重算

> **English Version**: [lmcache-devdax-offset-apc-attribution-dynamo-relay.en.md](lmcache-devdax-offset-apc-attribution-dynamo-relay.en.md)  
> **证据等级 (Evidence Class)**: `implementation`（一手源码核验） + `own-offline`（确定性标准库仿真）  
> **上游仓库与提交**: `LMCache/LMCache` commits [`e8dc1a1daeb918aa5a1797d2ecc2c6b2cdeab1a3`](https://github.com/LMCache/LMCache/commit/e8dc1a1daeb918aa5a1797d2ecc2c6b2cdeab1a3) & [`168130df9c8fafa376b650814598179c212d3e66`](https://github.com/LMCache/LMCache/commit/168130df9c8fafa376b650814598179c212d3e66) (2026-10-20)；`ai-dynamo/dynamo` commits [`909ebee0c4b63988bea6ab474a46913073e4a2dc`](https://github.com/ai-dynamo/dynamo/commit/909ebee0c4b63988bea6ab474a46913073e4a2dc) & [`cc02d8b1c2b60a9cbe291b5264456897c64d7e7c`](https://github.com/ai-dynamo/dynamo/commit/cc02d8b1c2b60a9cbe291b5264456897c64d7e7c) (2026-10-20)  
> **关联实验**: [`experiments/lmcache_devdax_apc_dynamo_relay_sim/run.py`](../../experiments/lmcache_devdax_apc_dynamo_relay_sim/run.py) (`--check` 通过)

---

## 1. 架构背景与核心工程矛盾

在多实例共享 CXL/Device-DAX 内存池与跨数据中心 KV 路由的生产级推理集群中，存在四类隐蔽的系统级陷阱：
1. **Linux Device-DAX `fstat` 盲区与延迟 `SIGBUS`**：多个 LMCache 实例共享同一块 512 GiB CXL Device-DAX（`/dev/dax0.0`）时，字符设备的 `os.fstat(fd).st_size` 恒返回 `0`，且内核在 `mmap` 调用时不检查容量边界，越界或未按设备对齐（通常 `2 MiB`）的映射会在首次缺页访问时直接触发致命 `SIGBUS`。
2. **引擎本地 APC 前缀污染 LMCache L1 命中率**：当推理引擎自身的自动前缀缓存（APC）已命中前 `start_chunk` 个块时，LMCache 仅查询 `[start_chunk, end)`；但旧版 `_consume_prefetch_result` 执行了 `l1_found_count += job.start_chunk`，把引擎 APC 已持有的前缀算进 LMCache 的 L1 命中分子与分母中，导致即使 LMCache 完全未命中（`0%`）也会上报虚假的 L1 命中率。
3. **Dynamo 跨机房 Cuckoo 过滤器 (`DcCkfState`) 恢复快照尾部删除丢失**：当 `LocalKvIndexer` 返回缓存的 `TreeDump` 快照及其缓冲事件尾部时，尾部包含 `KvCacheEventData::Removed` 或 `Cleared` 事件；旧版 `DcCkfRankReplacement::push_event` 仅接受 `Stored` 事件，导致恢复失败或残留零持有者的幽灵块。
4. **流式迁移重试 (`RetryManager`) 将已交付 Token 重复计入 Prompt**：当 Worker 在输出 $k$ 个 Token 后断连迁移时，`RetryManager` 将这 $k$ 个已交付 Token 拼接到重发请求的 Prompt 中；新 Worker 将其计入 `prompt_tokens`，导致客户端账单的 `prompt_tokens` 虚增 $k$ 且 `cached_tokens` 失真。

---

## 2. 一手源码机制拆解 (`LMCache/LMCache` & `ai-dynamo/dynamo`)

### 2.1 LMCache Device-DAX sysfs 几何预检与多 Arena 区间不交叠不变量 (`utils.py`, `devdax_memory_allocator.py` @ `e8dc1a1dae`)

- **`get_dax_device_geometry(fd)` 与 `check_dax_range(device_path, fd, offset, size)`**：
  - 对字符设备（`stat.S_ISCHR`），通过 `major(st_rdev):minor(st_rdev)` 读取 `/sys/dev/char/{major}:{minor}/size` 与 `align`（绕过任意符号链接别名与 `fstat == 0` 限制）。
  - 在 `mmap` 前强制校验三项不变量：
    $$\text{offset} \bmod G_{\text{mmap}} = 0, \quad \text{offset} + \text{size} \le \text{capacity}, \quad (\text{offset} \bmod \text{align} = 0) \land (\text{size} \bmod \text{align} = 0)$$
- **`_DevDaxArena.overlaps(offset, size)`**：在同一进程热插拔多个同设备区间时，通过 `(is_chr, major, minor)` 识别物理设备，并拒绝满足 `offset < arena.offset + arena.size and arena.offset < offset + size` 的重叠区间。

### 2.2 LMCache APC 覆盖前缀剥离与子块重叠去重 (`lookup.py` @ `168130df9c`)

- **LMCache 纯净命中率分母与分子**：
  $$\text{looked\_up\_tokens} = \text{requested\_tokens} - \text{start\_chunk} \times \text{chunk\_size}$$
  在 `_consume_prefetch_result` 中，先在纯查询子区间上归因 `l1_chunks = min(l1_found_count, found_count)` 与 `l2_chunks = found_count - l1_chunks`，随后再将 `found_count += job.start_chunk` 还原为绝对块号供锁释放使用，从而保证 `apc_covered_tokens = start_chunk * chunk_size` 绝不混入 `l1_hit_tokens`。
- **请求级总命中去重 (`request_hit_tokens`)**：
  $$\text{request\_hit\_tokens} = \max\big(\text{engine\_hit\_tokens},\; \text{covered\_tokens} + \text{hit\_tokens}\big)$$
  由于引擎上报的 `engine_hit_tokens` 包含未满一整块的子块尾部（例如 `3` 块 `768` Token + `100` 子块 Token = `868` Token），而 `covered_tokens + hit_tokens` 均从位置 `0` 起算，取 `max` 而非直接相加可精确消除 `100` Token 的子块重叠计费。

### 2.3 Dynamo `DcCkfRankReplacement` 删除重放与 `RetryManager::rebase_usage` (`dc.rs` @ `909ebee0c4`, `migration.rs` @ `cc02d8b1c2`)

1. **快照尾部 `Removed` / `Cleared` 重放 (`lib/kv-router/src/indexer/cuckoo/dc.rs`)**：
   - `DcCkfRankReplacement::push_event` 对 `KvCacheEventData::Removed(remove)` 执行 `source_lineage.remove(&hash)`，对 `Cleared` 清空 `source_lineage` 与 `canonical_candidates`。
   - 在 `DcCkfState::install_replacement` 中，统计存活 `source_lineage` 的 `owner_increments` 后，调用 `replacement.canonical_candidates.retain(|canonical| owner_increments.contains_key(canonical))`，将快照尾部已被删除的候选块彻底剔除。
2. **迁移后 Token 用量重算 (`lib/llm/src/migration.rs:rebase_usage`)**：
   设客户端原始 Prompt 长度为 $P$，流式迁移重放了 $R = \text{token\_ids.len()} - P$ 个已交付 Token，重试 Worker 上报 $(P_{\text{worker}}, C_{\text{worker}}, H_{\text{worker}})$，则 `rebase_usage` 原位修正：
   $$P' = P_{\text{worker}} - R, \quad C' = C_{\text{worker}} + R, \quad H' = \min(H_{\text{worker}}, P'), \quad \text{completion\_tokens\_details} = \text{None}$$

---

## 3. 确定性离线实验验证 (`lmcache_devdax_apc_dynamo_relay_sim`)

运行 `python experiments/lmcache_devdax_apc_dynamo_relay_sim/run.py --check`，核心验证指标如下：

| 子系统与验证项 | 异常输入 / 旧版行为 | 修复后确定性结果 |
|---|---|---|
| CXL Device-DAX (`512 GiB, align=2 MiB`) | 1) `[256 GiB, 512 GiB)` 不交叠分区；2) `[128 GiB, 384 GiB)` 重叠；3) `+1 MiB` 未对齐；4) 越界 | 1) `admitted = True`；2/3/4) 在 `mmap` 前精准拦截（避免运行时 `SIGBUS`） |
| LMCache APC 前缀剥离 (`2,100` Token) | `start_chunk = 3` (`768` Token)，LMCache 命中 `0` 块时旧版上报 **`37.5%`** 虚假 L1 命中率 | 修复后 `looked_up_tokens = 1,280`，未命中时 `l1_hit_rate = 0.0%`；命中 `2 L1 + 1 L2` 时 `request_hit_tokens = 1,536`（消除 `100` Token 子块重复计数） |
| Dynamo `DcCkfRankReplacement` | `Stored([11,12,13]) -> Removed([12,13,99]) -> Stored([14])` | 过滤 `2` 个已被删除的幽灵候选 `[1012, 1013]`，仅保留存活候选 **`[1011, 1014]`** |
| Dynamo `RetryManager::rebase_usage` | 原始 Prompt `3` Token，2 次迁移重放 `4` Token，Worker 上报 `(P=7, C=2, cached=7)` | 重算还原为 **`(P'=3, C'=6, total=9, cached'=3)`**，清空局部 `completion_tokens_details` |
