# vLLM V1 UMBP 共享 KV 连接器 (`UMBPStoreConnector`) 拓扑/分片规划契约与 THUDM/slime Straw 原地检查点回滚 (`rewind.py`) 机制

> **English Version**: [vllm-v1-umbp-kv-connector-slime-straw-rewind.en.md](vllm-v1-umbp-kv-connector-slime-straw-rewind.en.md) | **关联实验**: [`experiments/vllm_umbp_kv_connector_slime_rewind/README.md`](../../experiments/vllm_umbp_kv_connector_slime_rewind/README.md)

### 一句话结论

vLLM V1 在 `vllm/distributed/kv_transfer/kv_connector/v1/umbp/`（`#61790`）中建立了**与底层传输客户端解耦的统一内存缓冲池（UMBP）共享 KV 连接器契约**：通过 `[PP, PCP, TP]` 排布且由 `DCP` 切分 `TP`（`dcp_rank = tp_rank % dcp_size`，避免第 4 维笛卡尔积膨胀）的 `RankTopology`、由 Worker 派生并通过握手向 Scheduler 汇报的 32 字符 SHA-256 `UMBPNamespace`（含 `layout_version: 2`、逐层 `KVCacheSpec` 指纹与 EAGLE 投机草稿模型隔离）、支持大页跨步多段拆分的 `KVLayoutPlanner`、强制保留至少 1 个末尾 Token（`max_external_token = num_tokens - 1`）并要求全 Rank 命中的 `UMBPStoreConnectorScheduler`，以及全键成功才原子发布（`TransferJobStatus.COMPLETED`）、抢占前同步排空（`handle_preemptions`）且全 Worker 完成后逆序释放 GPU 块池的 `UMBPStoreConnectorWorker`；与此同时，THUDM/slime（`#2482`）在 `slime/data/{rewind.py,checkpoint.py}` 中实现了 **Straw 队列原地检查点回滚（In-Place Checkpoint Rewind）**：通过逆深度目录 `fsync` 写入不可变联合提交边界 `rollout/committed_{step}.json`，在恢复时若模型 Tracker 超前于联合提交边界则自动钳位至 $\min(\text{tracker}, \max(\text{commits}))$，并按四级回退层次（`_queue_original_input` 摘要去重引用 $\to$ 未触碰 `PENDING` 样本 $\to$ `RolloutDataSource` 跨 Epoch 精确游标反算 $\text{divmod}(\text{epoch}\cdot N + \text{offset} - \Delta, N)$ $\to$ `fresh_samples` 状态剥离）重建在途 Prompt 组、释放超前步骤（`step_id > step`）存储所有者并原子替换空白 WAL 日志。

### 组件责任与上游主线位置

| 模块与文件 | 上游仓库与 Commit / Blob SHA | 核心工程职责 |
| :--- | :--- | :--- |
| `vllm/distributed/kv_transfer/kv_connector/v1/umbp/data.py` | `vllm-project/vllm` commit `82b7dad5e315040219ec657398d6197010121d15` / blob `9374c9ced8020e85b006e0ebd8ea154bddbfc658` | 定义 `RankTopology`、`UMBPNamespace`、`BlockIdentityCodec`、`KVRegion`/`KVLayoutPlanner`、`RequestTracker`、`TransferJobState` 与 `UMBPConnectorWorkerMetadata`。 |
| `vllm/distributed/kv_transfer/kv_connector/v1/umbp/{scheduler.py,worker.py,connector.py}` | `vllm-project/vllm` commit `82b7dad5e315040219ec657398d6197010121d15` / blobs `dd7ecadb16`, `34101f294c`, `e828ac4ee0` | 实现 Scheduler 侧全 Rank 命中校验、EAGLE 尾块丢弃、异步加载转发、`_StoreEvent` 引用计数释放，以及 Worker 侧本地化键重写、原子发布与抢占屏障。 |
| `slime/data/rewind.py` | `THUDM/slime` commit `4f204660abcd531882dd4918ee301f107a952eb4` / blob `e1e053461f8e18672797b93c4602fb4a478f62c2` | 实现 `restore_dataset_cursor`、`fresh_samples`、四级 `regeneration_groups` 跨 Epoch 游标回退、`reset_journal` 原子 WAL 重置与 `discard_future` 未来步骤清理。 |
| `slime/data/checkpoint.py` | `THUDM/slime` commit `4f204660abcd531882dd4918ee301f107a952eb4` / blob `03b1eede3e7c61cce7a6841bda812fdc77e4fde0` | 实现 `save_checkpoint`、`commit_checkpoint`（文件与逆深度目录 `fsync` + SHA-256 校验）、`_select_checkpoint` 安全边界选择与 `resolve_checkpoint` 恢复计划生成。 |

### 核心实现路径、数据布局与同步契约

1. **vLLM V1 UMBP 拓扑映射、Worker 握手命名空间与多段物理块规划（`data.py:52-415`）**：
   - **`RankTopology` 中 `DCP` 对 `TP` 的内嵌切分**：vLLM 引擎 Worker 按 `[PP, PCP, TP]` 排布（`TP` 为最内层维度：`tp_rank = rank % tp_size`，`pcp_rank = (rank // tp_size) % pcp_size`，`pp_rank = (rank // (tp_size * pcp_size)) % pp_size`）。`DCP`（Decode Context Parallelism）**不会创建第 4 个笛卡尔积 Worker 维度**，而是切分现有的 `TP` Worker：`dcp_rank = tp_rank % dcp_size`（当 `dcp_size > 1` 时强制要求 `tp_size % dcp_size == 0`）。因此 `all_namespaces()` 生成的唯一 Rank 命名空间总数为 $\text{PP} \times \text{PCP} \times \text{TP}$，生成的键格式为 `umbp:vllm:v1:{namespace}:tp{tp}:pcp{pcp}:dcp{dcp}:pp{pp}:g{group_id}:{block_hash.hex()}`。
   - **为什么 `UMBPNamespace` 必须由 Worker 派生并通过握手传给 Scheduler**：Scheduler 侧的 `KVCacheConfig` 会将 `UniformTypeKVCacheSpecs` 组展平，丢失逐层差异；因此只有 Worker 能遍历 `_layer_specs(group)` 计算完整的 SHA-256 前 32 位十六进制指纹（包含 `model`、`revision`、`block_size`、`cache_layout`、逐层 `_spec_fingerprint`、`tp/pp/pcp/dcp`、`layout_version: 2` 以及可选的 `speculative` 草稿模型配置），并通过 `get_handshake_metadata()` / `set_xfer_handshake_metadata()` 强制校验所有 Worker 汇报唯一相同的命名空间后才激活 Scheduler 查找。
   - **`KVLayoutPlanner` 的跨步多段拆分**：当某层的逻辑块字节数 `region.block_bytes` 大于底层 Tensor 的 `physical_stride`（且满足整除约束 `region.block_bytes % physical_stride == 0`）时，`_ranges_for_block` 将单个逻辑 `block_id` 拆分为 `physical_parts = region.block_bytes // physical_stride` 个连续的 `KVRange`（物理块号为 `block_id * physical_parts + part`）。
2. **Scheduler 末尾 Token 保护、StoreEvent 引用计数与 Worker 原子发布/抢占屏障（`scheduler.py:118-395`, `worker.py:110-220`）**：
   - **末尾 Token 保护与全 Rank 命中**：在 `get_num_new_matched_tokens` 中，Scheduler 强制设置 `max_external_token = request.num_tokens - 1`，确保外部 KV 命中永远不会覆盖整个 Prompt，从而留出至少最后 1 个 Prompt Token 供模型前向计算 logits；若开启 EAGLE（`use_eagle`），则额外将 `matched_blocks = max(0, matched_blocks - 1)` 以便 Drafter 重算最后一块。每个逻辑块必须在其全部 `per_rank = topology.rank_count` 个 Rank 键上同时命中（`all(group_hits[index : index + per_rank])`），并在首个 Miss 处立即截断前缀匹配。
   - **`_StoreEvent` GPU 块池生命周期**：当 `build_connector_meta` 生成 `store_plans` 时，Scheduler 分配递增的 `store_event` ID，并调用 `gpu_block_pool.touch(...)` 锁住待导出的 GPU 物理块；只有当 `update_connector_output` 聚合到的 `event.result.completed_workers >= self._num_workers`（全部 Worker 完成写出）时，才按逆序 `reversed(block_ids)` 调用 `pool.free_blocks(...)` 归还块池。
   - **Worker 原子发布与抢占排空**：`UMBPStoreConnectorWorker._finish_job` 仅在整批 `TransferJobState.status == TransferJobStatus.COMPLETED` 时才调用 `self.runtime.publish(result)`（任一键失败均不发布任何键，并将失败块记入 `failed_block_ids`）；在 `handle_preemptions` 中，Worker 对所有被抢占请求的在途 Load/Store 作业同步调用 `self.runtime.wait(job)`，防止 vLLM 在 DMA 传输尚未结束时复用该 GPU 物理块。
3. **THUDM/slime Straw 联合提交边界与四级原地回滚恢复（`checkpoint.py:42-265`, `rewind.py:14-145`）**：
   - **两阶段同步提交与逆深度目录 `fsync`**：`save_checkpoint` 在 Straw 模式下强制 `actor_model` 和 `critic_model` 以 `force_sync=True` 同步保存，再调用 `rollout_manager.save.remote(rollout_id)`，最后由 `commit_checkpoint` 对所有模型文件调用 `os.fsync`、计算 `.json`/`.metadata` 及 `queue_state_{id}.json`/`builder_state_{id}.json` 的 SHA-256，并按路径深度从深到浅对所有父目录执行 `os.fsync(fd)` 后才写出不可变的 `rollout/committed_{id}.json`，最后通知 `rollout_manager.checkpoint_committed.remote(rollout_id)`。
   - **四级 Prompt 重建（`regeneration_groups`）与跨 Epoch 游标反算**：原地回滚时，队列中未消费完的样本组按四级策略恢复原始 Prompt：
     1. 若样本携带 `_queue_original_input` 引用，按 `ref.manifest.digest` 去重加载原始输入；
     2. 若组内全部样本处于 `Sample.Status.PENDING` 且 `response_length == 0`，直接深拷贝保留；
     3. 若生产者为带数据集的 `RolloutDataSource`（兼容旧版无引用检查点），根据组号距离 $\Delta = \text{cursor}[\text{"sample\_group\_index"}] - \text{group\_id}$ 反算全局绝对偏移 $\text{abs} = \text{cursor}[\text{"epoch\_id"}] \times |\mathcal{D}| + \text{cursor}[\text{"sample\_offset"}] - \Delta$，通过 `divmod(abs, |D|)` 精确还原 `(epoch_id, sample_offset)`、重洗牌该 Epoch 并调用 `producer.get_samples(1)` 重新读取未被生成钩子污染的原始样本；
     4. 对无数据集自定义生产者，调用 `fresh_samples(group)` 仅保留 `prompt`/`label`/`metadata` 等 10 个输入属性，剥离生成响应与训练状态。

### 关键公式、单位与工程假设

1. **UMBP `RankTopology` 映射与命名空间基数**：
   $$\text{tp\_rank} = r \bmod T, \quad \text{pcp\_rank} = \left\lfloor \frac{r}{T} \right\rfloor \bmod P_{\text{pcp}}, \quad \text{pp\_rank} = \left\lfloor \frac{r}{T \cdot P_{\text{pcp}}} \right\rfloor \bmod P_{\text{pp}}, \quad \text{dcp\_rank} = \text{tp\_rank} \bmod D_{\text{dcp}}$$
   当 $D_{\text{dcp}} > 1$ 时要求 $T \bmod D_{\text{dcp}} = 0$，且总命名空间数恒为 $|\mathcal{N}| = P_{\text{pp}} \times P_{\text{pcp}} \times T$。
2. **Straw 数据集游标跨 Epoch 逆向恢复公式（`rewind.py:56-68`）**：
   $$\Delta = G_{\text{cursor}} - G_{\text{group}}, \quad A = E_{\text{cursor}} \cdot |\mathcal{D}| + O_{\text{cursor}} - \Delta$$
   $$(E_{\text{recovered}}, O_{\text{recovered}}) = \left(\left\lfloor \frac{A}{|\mathcal{D}|} \right\rfloor, \; A \bmod |\mathcal{D}|\right), \quad I_{\text{recovered}} = I_{\text{cursor}} - \Delta \cdot K_{\text{samples\_per\_prompt}}$$

### 合成实验与验证结果（`experiments/vllm_umbp_kv_connector_slime_rewind`）

运行 `python experiments/vllm_umbp_kv_connector_slime_rewind/run.py --check` 的确定性验证结果如下：

| 子系统与验证项 | 测试配置 / 输入条件 | 确定性输出 / 验证结果 | 核心架构意义 |
| :--- | :--- | :--- | :--- |
| UMBP `RankTopology` (`rank=5`) | `TP=4, PP=1, PCP=2, DCP=2` | `local_namespace = [1, 1, 1, 0]`, `total_namespaces = 8` | `DCP` 切分 `TP` 而不产生 16 路笛卡尔积膨胀 |
| UMBP 前缀匹配上限 (`256` tokens) | `block_size = 64`, `local_tokens = 0` | 标准加载上限 `192` tokens (`3` 块)；EAGLE 上限 `128` tokens (`2` 块) | 严守 `max_external_token = 255`，保证尾块前向执行 |
| `KVLayoutPlanner` 多段拆分 | Layer 0 `64 KiB` + Layer 1 `128 KiB` (`stride=64 KiB`) | 生成 `3` 个 `KVRange`，Layer 1 偏移 `[65536, 131072]` | 支持混合注意力层不同页面大小同块打包 |
| `_StoreEvent` 块池引用计数释放 | `8` Workers 异步导出块 `[10, 12, 14]` | 仅当 `completed_workers == 8` 后按 `[14, 12, 10]` 逆序释放 | 杜绝慢 Worker 导出期间 GPU 块被提前回收覆盖 |
| Slime Straw 跨 Epoch 游标回退 | $|\mathcal{D}|=100, E=2, O=15, G_{\text{cur}}=215 \to G=195$ | `recovered_epoch = 1`, `offset = 95`, `sample_index = 780` | 跨 Epoch 边界精确重读原始未污染 Prompt |
| Slime `_select_checkpoint` + `discard_future` | `tracker = 12`, `committed = {8, 10}`, 磁盘存在 `{8,10,11,12}` | 安全钳位至 `step = 10`，清理 `{11, 12}` 并释放其 `storage_owner` | 模型保存完成但 Rollout 崩溃时不加载半成品状态 |

### 架构师的设计练习与可证伪问题

1. **计算练习**：对于一个恰好是块大小整数倍的 Prompt（如 `num_tokens = 256, block_size = 64`），如果本地 GPU 缓存已命中前 `192` 个 Token（`num_computed_tokens = 192`），UMBP Scheduler 的 `get_num_new_matched_tokens` 会发起几次外部 `runtime.lookup`？
   - *解答*：`0` 次。因为 `max_external_token = 256 - 1 = 255`，而 `_lookup_objects` 的循环范围为 `range(local_tokens + 64, 256, 64) = range(256, 256, 64)`（为空），因此不会发起无效的外部查找，最后 64 个 Token 直接在本地执行 Prefill。
2. **可证伪假设**：在 `THUDM/slime` 异步 RL 训练中，能否直接依据 Megatron/Torch 的 `latest_checkpointed_iteration.txt` 决定恢复步数？
   - *验证方式*：不能（`checkpoint.py:139-174` 明确禁止）。因为 `actor_model.save_model` 先于 `rollout_manager.save` 和 `commit_checkpoint` 执行，若进程在模型写完 `latest_checkpointed_iteration.txt = 12` 后、写出 `committed_12.json` 前崩溃，直接加载 Step 12 会导致 Rollout 队列状态缺失或错位；必须以 `min(tracker, max(commits))` 对应的 `committed_{step}.json` 为准。
