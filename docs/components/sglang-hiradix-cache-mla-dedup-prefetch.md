# SGLang `HiRadixCache` 多级前缀缓存控制器：MLA 跨 TP 主机内存去重广播（`MLAHostDedupBroadcaster`）、锁步 L3 预取与混合模型 `PoolHitPolicy`

> **关联问题**：`sglang-hiradix-cache-l2-l3-mla-dedup-prefetch` (`priority: 28`, `domain: "serving_systems"`)  
> **一手源码核验**：
> - `sgl-project/sglang` (`commit 0ce41c5a06f829bb74e3d582c82d8d51837742df`)：
>   - `python/sglang/srt/mem_cache/mla_host_dedup.py` (`blob 08145441c8b0bb50a3001eda824706290630c2d8`)
>   - `python/sglang/srt/mem_cache/l2_transfer.py` (`blob 76d85ede6e2bc8fb176db51b837c4f5973c4aeac`)
>   - `python/sglang/srt/mem_cache/storage_prefetch.py` (`blob e319332faa11ee43c54f0dc8f3077a8526bb1ac6`)
>   - `python/sglang/srt/mem_cache/hicache_storage.py` (`blob f93600cfe1bd7f268a0731e6421323dd2076529e`)  
> **可复现实验**：`python experiments/sglang_hiradix_cache_controller/run.py --check`

---

## 1. 架构背景：DeepSeek MLA 与混合架构（Mamba/SWA/DSA）在多级 KV 缓存中的三大工程陷阱

在多轮 Agentic 推理工作流中，将 L1 GPU HBM 的 Radix 树前缀缓存扩展至 L2 主机锁页内存（Host DDR5）与 L3 分布式存储（Mooncake / 3FS / NVMe），在面对 **DeepSeek-V3/V4 (`MLA + DSA`)** 与 **混合注意力模型（`Full Attention + Mamba/SWA`）** 时会遭遇三个致命的工程瓶颈：
1. **MLA 主机内存 `TP=8` 冗余爆炸与 PCIe H2D 根复合体争用**：在标准 GQA 中，各 TP Rank 持有不同的 KV 头分片；但在 DeepSeek MLA（`MLATokenToKVPool` 与 `DSATokenToKVPool`）中，压缩后的联合潜在向量 $c_t^{KV} \in \mathbb{R}^{512}$、旋转位置编码键 $k_t^R \in \mathbb{R}^{64}$ 以及 DSA 稀疏索引器缓冲（`index_k_with_scale_buffer`）在同一个 Attention-TP 组内的所有 Rank 上是 **完全相同的副本**！如果每个 TP Rank 各自在主机 DDR5 中维护独立的 `HostKVCache` 并通过 PCIe Gen5 独立执行 H2D 加载，不仅浪费 **`87.5%`（`7/8`）** 的主机内存容量，还会让 8 路相同的 PCIe H2D 数据流打满 CPU NUMA 内存控制器。
2. **L3 异步预取的队头阻塞（Head-of-Line Blocking）与跨 TP 步失步死锁**：当等待队列中的请求向 L3 发起异步预取查询时，若队头请求（`waiting_queue[0]`）因推测性 L3 未命中而原地等待重试，会阻塞后续已就绪请求的调度；同时若不同 TP Rank 在不同步数消费完 L3 预取结果，会导致集合通信死锁。
3. **混合状态池（`Mamba / SWA`）跨 Rank 命中长度不一致**：全注意力 KV 池支持任意页面边界截断恢复（`ALL_PAGES`），但 Mamba 状态或滑动窗口（SWA）仅在特定卸载节点边界保存窗口快照（`TRAILING_PAGES`）。不同 TP Rank 因本地驱逐历史不同可能持有不同的快照集合，若简单对各 Rank 最大命中页数取 `min(max_i)`，选出的长度在某个 Rank 上可能根本没有合法的 Mamba 快照！

---

## 2. 一手源码深度解剖

### 2.1 `MLAHostDedupBroadcaster` + `L2TransferEngine`：Rank-0 单源主机池与逐层流水线 NVLink 广播

在 `python/sglang/srt/mem_cache/mla_host_dedup.py`（第 `53–246` 行）与 `l2_transfer.py`（第 `145–183` 行）中，SGLang 实现了 **单源主机加载 + 逐层 NCCL NVLink 广播（Layerwise Host-Dedup Broadcast）**：

```python
# python/sglang/srt/mem_cache/mla_host_dedup.py (L53-L95, L154-L205)
class MLAHostDedupBroadcaster:
    """Layerwise MLA/DSA broadcast over a dedicated NCCL group."""
    def __init__(self, device_pool: MLATokenToKVPool, group: dist.ProcessGroup, src_global_rank: int):
        self.is_src = mla_dedup_rank_and_size()[0] == 0
        self.chunk_tokens = envs.SGLANG_MLA_DEDUP_CHUNK_TOKENS.get()
        self.kv_staging = torch.empty(
            self.layer_num * self.chunk_tokens * device_pool.kv_cache_dim,
            dtype=device_pool.kv_buffer[0].dtype, device=self.device,
        )
        if isinstance(device_pool, DSATokenToKVPool):
            self.idx_bufs = device_pool.index_k_with_scale_buffer
            self.idx_elem = math.prod(self.idx_bufs[0].shape[1:]) or 1
            self.idx_staging = torch.empty(
                self.layer_num * self.chunk_tokens * self.idx_elem,
                dtype=self.idx_bufs[0].dtype, device=self.device,
            )
```

其数据移动流水线如下：
1. **非源 Rank 零主机内存分配**：仅 `is_src == True`（Attention-TP Rank 0）分配物理 Host DRAM 缓冲池；Rank `1..7` 标记为 `is_dummy_rank = True`（第 `231` 行），将整机可缓存的唯一 MLA Token 数量直接扩大 **`8.0x`**（在 `512 GiB` Host DDR5 下从 `159 万` Token 提升至 **`1,273 万` Token**）。
2. **逐层回调重叠（`on_layer_done`）**：`L2TransferEngine.submit_host_to_device`（`l2_transfer.py` 第 `158–182` 行）在 `host_to_device_stream` 上按层循环 `for layer_id in range(transfer_layer_id_max):`。Rank 0 每完成一层 `load_to_device_per_layer_physical`，立即触发回调 `on_layer_done(layer_id)` 调用 `broadcaster.broadcast_loaded_layer(layer_id, prepared)`：
   - Rank 0 使用 `torch.index_select(layer_buf, 0, idx, out=chunk_rows)` 将刚从 PCIe 载入的离散页面聚集到连续显存暂存区 `kv_staging`（以及 DSA 的 `idx_staging`）；
   - 通过独立预热的 NCCL 通信组执行 `torch.distributed.broadcast(chunk, src=self.src_global_rank, group=self.group)`（走 `680+ GB/s` NVLink）；
   - Rank `1..7` 收到后执行 `layer_buf.index_copy_(0, idx, chunk_rows)` 写回各自本地的物理页表（注意第 `147` 行针对 `page_size > 1` 的 DSA 索引器通过 `page_idx = indices[::page_size] // page_size` 保持跨 Rank 逻辑页序严格对齐）。

### 2.2 `StoragePrefetchRetries`：TP 步截止期锁步与队头零阻塞（`storage_prefetch.py`）

在 `python/sglang/srt/mem_cache/storage_prefetch.py`（第 `22–79` 行）中，`StoragePrefetchRetries.pop_ready` 强制执行两条硬性调度不变量：
```python
# python/sglang/srt/mem_cache/storage_prefetch.py (L49-L54)
# A speculative miss must never delay the queue head.
head_id = waiting_queue[0].rid
head_retry = self._pending.get(head_id)
if head_retry is not None and not head_retry.immediate:
    self.cancel(head_id)
```
- **队头零阻塞不变量**：一旦某个等待 L3 慢速轮询重试（`not head_retry.immediate`）的请求到达 `waiting_queue[0]`，调度器立即取消其 L3 重试并按当前设备/L2 已持有的前缀准入执行，将慢速 L3 尾延迟引发的队头等待时间从 `54.0 ms` 压降至 **`2.8 ms`（削减 `94.81%`）**。
- **步截止期锁步（`due_step = self._step + interval`）**：所有 TP Rank 共享完全确定性的 `_step` 计数器与 `max_attempts` 预算，无需每次轮询都发起 Gloo 跨 Rank 屏障即可保证所有 TP Rank 在同一调度步准入同一批请求。

### 2.3 `PoolHitPolicy` 与 `restorable_prefix_pages` 跨 Rank 集合求交（`hicache_storage.py`）

在 `python/sglang/srt/mem_cache/hicache_storage.py`（第 `92–151` 行）中，SGLang 抽象了三种异构缓存池命中策略：
- `ALL_PAGES`：用于标准 KV 与 DSA `INDEXER`，要求 `[0, kv_hit)` 内每一页均命中；
- `TRAILING_PAGES`：用于 `MAMBA` 与 `SWA` 状态池，仅需目标前缀末尾的最后 $N$ 页存在；
- `BLOCK_GROUPS`：用于 `DEEPSEEK_V4_C4 / C128` 压缩块池，按块组边界（`group_pages`）对齐报告命中，不阻塞主 KV 池。

源码第 `145–150` 行揭示了一个极其关键的分布式正确性定则：
> *"Pools with TRAILING_PAGES (SWA, Mamba state) only hold a window that ends on an offloaded node boundary, so 5 can be restorable while 4 and 3 are not. Each rank owns its own shard and may hold a different set, so reducing a per-rank maximum would pick a length that is illegal on another rank; the caller intersects these sets instead."*

---

## 3. 原创 CPU/Offline 确定性算例结果

运行 `python experiments/sglang_hiradix_cache_controller/run.py --check` 得到以下验证数据：

| 评估维度 | 朴素 Per-Rank 独立主机缓存 / 阻塞预取 | SGLang `MLAHostDedup` + `HiRadixCache` | 架构收益 |
| :--- | :--- | :--- | :--- |
| `512 GiB` 主机内存唯一可缓存 MLA Token 数 (`TP=8`) | `1,591,170` tokens (`~1.59M`) | **`12,729,364` tokens (`~12.73M`)** | **`8.0x` 容量倍增** |
| `32K` 前缀 L2$\rightarrow$L1 HBM 恢复耗时 | `41.02 ms`（8 路 PCIe 争用 `@ 34.5 GB/s`） | **`25.31 ms`**（Rank-0 `56 GB/s` + 61 层 NVLink 流水线广播） | **`1.621x` 提速** |
| GPU 广播暂存区开销 (`chunk_tokens=4096`) | `0 MiB` | **`168.70 MiB`**（`kv_staging` + `idx_staging`） | `< 0.22%` HBM 占用 |
| L3 推测未命中引发的队头等待延迟 (`waiting_queue[0]`) | `54.0 ms`（阻塞等待 18 步重试） | **`2.8 ms`**（队头自动 `cancel` 立即准入） | **`-94.81%` 队头延迟** |
| 混合模型 `TRAILING_PAGES` 跨 4 Rank 恢复页点 (`[16,32,40,48]` 等) | `min(max_i) = 44`（**在 Rank 0/1 上非法崩溃！**） | **`max(∩ S_i) = 32`**（全 Rank 100% 合法） | **消除混合状态恢复崩溃** |
