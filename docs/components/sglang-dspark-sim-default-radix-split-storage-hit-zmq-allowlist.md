# SGLang DSpark 模拟接受默认关闭、RadixCache `UpdatePendingBackupOnNodeSplit` 统一拆分动作、存储层命中预填充负载扣除与 ZMQ ZAP 白名单鉴权

> **语言 / Language**: **中文 (当前)** | [English (`sglang-dspark-sim-default-radix-split-storage-hit-zmq-allowlist.en.md`)](sglang-dspark-sim-default-radix-split-storage-hit-zmq-allowlist.en.md)
> **证据等级**: `implementation`（上游源码提交核验）+ `own-offline`（本项目标准库确定性复现实验）
> **上游版本锚点**:
> - `sgl-project/sglang@f0b65907227480afc376a66db62931b7e9c46c38` (`2026-10-20`, PR `#45433`)
> - `sgl-project/sglang@cc7ae372ac2fba9605460308c8f347c80702c828` (`2026-10-20`, PR `#45242`)
> - `sgl-project/sglang@0d7638b03509967e749c201146bfb04bbdfe380c` (`2026-10-20`, PR `#45158`)
> - `sgl-project/sglang@194876bea07214539306526f1bd937a96e02bff9` (`2026-10-20`, PR `#45250`, `CVE-2026-93034`)
> - `sgl-project/sglang@468e26f253b2f1d42bf3545742b7c363d0c6592e` (`2026-10-20`, PR `#45280`)

---

## 1. 核心架构发现与问题定位

### 1.1 DSpark 投机解码模拟接受捷径改为显式 Opt-In (`f0b6590`)

- 在 Phase 38 (`43ed385`) 引入 CUDA 图内模拟接受折叠后，`python/sglang/srt/environ.py` 中 `SGLANG_SIMULATE_ACC_GREEDY` 与 `SGLANG_DSPARK_SIMULATE_FOLD` 的默认值曾被设为 `EnvBool(True)`。
- **架构隐患**：当用户仅设置 `SGLANG_SIMULATE_ACC_LEN` 进行基准测试但未显式指定采样旁路策略时，默认开启这两个开关会导致温度采样请求意外退化为贪心草稿接受路径，或跳过 Eager 接受提交验证。
- **修复 (`f0b6590`)**：将 `SGLANG_SIMULATE_ACC_GREEDY` 与 `SGLANG_DSPARK_SIMULATE_FOLD` 默认值统一改为 `EnvBool(False)`，要求基准测试显式 Opt-In。

---

### 1.2 UnifiedRadixCache `UpdatePendingBackupOnNodeSplit` 统一组件动作 (`cc7ae37`)

在 `UnifiedRadixCache` 中，当树节点发生前缀拆分（`parent -> old_node` 拆分为 `parent -> new_node -> new_child`）时，挂起的增量备份（Pending Backup）需要同步重映射：
- **修复前缺陷**：旧版 `ReplaceWriteThroughOnNodeSplit` 是缓存级（Cache-owned）动作，仅在 `child.write_through_pending_id is not None`（即存在树级 `ack_id`）时生成。但在 **Host Memory Buffer-Only 模式**（`is_host_memory_buffer_only=True`）或 SWA 窗口叶节点锁上限拆分（`SWAComponent._maybe_split_leaf_for_swa_lock`）中，节点拆分可能没有树级 `ack_id`（`ack_id is None`），导致 `FullComponent` 自有的缓冲区写依赖跟踪（Buffer-write dependencies）丢失拆分子节点映射。
- **修复方案 (`cc7ae37`)**：
  1. 将 `ReplaceWriteThroughOnNodeSplit` 重构为继承自 `ComponentAction`（`component_type = ComponentType.FULL`）的 `UpdatePendingBackupOnNodeSplit(ack_id: int | None, old_node_id, new_node_id, new_child_node_id)`。
  2. 在 `UnifiedTreeCore._split_node` 中，只要 `ack_id is not None or self.is_host_memory_buffer_only` 即发射该动作；`SWAComponent._maybe_split_leaf_for_swa_lock` 将返回的 `action` 收集进 `cache_actions`。
  3. 在 `UnifiedRadixCache._apply_cache_action` 中，若 `action.ack_id is not None`，先在缓存层执行 `_replace_pending_write_through_node` 与 `linker.replace_pending_offload_node`，随后将同一动作路由至 `FullComponent.apply_component_action`，消除 Buffer-Only 模式下的孤儿写依赖（`1 -> 0`）。

---

### 1.3 调度器等待队列预填充负载排除已接受的二级存储命中 (`0d7638b`)

在分层 KV 缓存（Device HBM + Host RAM + Storage Tier L3）架构下，`SchedulerLoadInquirer.get_num_waiting_uncached_tokens()` 用于向负载均衡器（如 Dynamo / SGLang Router）报告当前实例等待预填充的未命中 Token 总量：
- **修复前缺陷**：等待队列仅扣除了 `req.num_matched_prefix_tokens`（即已驻留 Device/Host 内存的前缀命中长度），而将**已被调度器接受、正在从二级存储预取回填（Prefetch）的存储层命中区间**仍计入“需计算 Prefill 的未缓存 Token”，导致 Prefill 队列负载被严重高估。
- **修复方案 (`0d7638b`)**：
  1. 在 `UnifiedRadixCache` 中实现 `waiting_storage_span(request)`：仅当预取请求**已被调度器正式接受**（`ongoing_prefetch` 中 `info.host_indices is not None` 返回 `(op.storage_start, op.storage_start + op.storage_hit_count)`，或位于 `buffer_pipeline.staged_prefetches` 中返回 `(staged.matched_len, staged.matched_len + staged.num_tokens)`）时，才返回逻辑 Token 区间；尚未接受的纯查询结果（`host_indices is None`）返回 `None`。
  2. 在 `SchedulerLoadInquirer.get_num_waiting_uncached_tokens()` 中计算：
     $$\text{storage\_tokens} = \max\!\left(0,\; \min(\text{span}_1, \text{seqlen}) - \max(\text{span}_0, \text{matched})\right), \quad \text{uncached} \mathrel{-}= \text{storage\_tokens}$$
  3. **定量收益 (`own-offline`)**：在典型 3 请求混合负载（已接受 Ongoing Prefetch `4096` tokens + Staged Pipeline `8192` tokens + 未接受 Query `2048` tokens）下，等待队列报告的未缓存 Token 从 `12,544` 降至真实需计算的 `3,328`（消除 **73.47%** 虚假 Prefill 负载膨胀），且不受 Worker 异步行拷贝进度的竞态干扰。

---

### 1.4 TCP 绑定 ZMQ IPC 套接字 ZAP 对端 IP 白名单鉴权 (`194876b`, `CVE-2026-93034`) 与分词器批量回包开关 (`468e26f`)

- **ZMQ ZAP 白名单鉴权 (`194876b`)**：针对 `CVE-2026-93034`（跨节点 DP Attention 控制器与 Snapshot 加载器在 TCP 绑定的 ZMQ IPC 套接字上默认接受任意可达 IP 连接并反序列化帧），新增环境变量 `SGLANG_ZMQ_ALLOWED_PEERS` 与 `restrict_zmq_peers(context, socket)`：
  - 由于 `libzmq` 每个 `zmq.Context` 仅允许注册一个 ZAP 处理线程，使用以 `context.underlying` 为键的全局字典 `_zmq_authenticators` 复用 `ThreadAuthenticator`。
  - 通过 `ipaddress.ip_address(p).compressed` 将 IPv4/IPv6 地址规范化为与 `libzmq` ZAP 完全一致的小写压缩字符串，并在 `socket.bind()` 之前设置 `socket.zap_domain = b"sglang"` 激活 NULL 机制连接校验。
- **多Tokenizer返回路径开关 (`468e26f`)**：新增 `SGLANG_BATCH_TOKENIZER_OUTPUTS`（默认 `True`），允许在低并发流式场景下关闭 `_split_batch_by_key` 分组聚合，切换为逐请求即时回包。

---

## 2. 本项目确定性复现实验 (`own-offline`)

- **实验脚本**: [`experiments/sglang_dspark_default_radix_split_storage_load_zmq/run.py`](../../experiments/sglang_dspark_default_radix_split_storage_load_zmq/run.py)
- **验证命令**: `python experiments/sglang_dspark_default_radix_split_storage_load_zmq/run.py --check`
- **不变量验证结果**:
  - `dspark_simulated_shortcuts_default_off = true`
  - `radix_split_routes_both_wt_and_buffer_only = true`（孤儿写依赖 `1 -> 0`）
  - `storage_tier_waiting_load_deducts_only_accepted = true`（`12,544 -> 3,328` tokens，虚假负载降低 `73.47%`）
  - `zmq_zap_blocks_unauthorized_peers = true`（精确放行压缩 IPv4/IPv6 白名单并拦截未授权 IP）
