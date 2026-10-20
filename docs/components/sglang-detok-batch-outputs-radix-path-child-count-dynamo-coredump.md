# SGLang Detokenizer 批量回传分派、Radix 树路径子节点计数与 Dynamo CUDA Coredump 等待门控剖析

> **English Version**: [sglang-detok-batch-outputs-radix-path-child-count-dynamo-coredump.en.md](sglang-detok-batch-outputs-radix-path-child-count-dynamo-coredump.en.md)

## 1. 一手源码证据与核心结论

| 仓库与提交 | 证据等级 | 源码路径 | 核心机制与定量结论 |
|---|---|---|---|
| `sgl-project/sglang` `#45280` (`468e26f253`, 2026-10-20) | `implementation` | `python/sglang/srt/environ.py`, `python/sglang/srt/managers/multi_tokenizer_mixin.py` | 新增 `SGLANG_BATCH_TOKENIZER_OUTPUTS`（默认 `True`），在 `_send_output_to_tokenizer_workers` 中按目标 tokenizer worker 聚合非连续批次输出并保持首次出现顺序，同时支持置 `False` 切换逐请求立即流式回传。 |
| `sgl-project/sglang` `#45241` (`dc62c3effd`, 2026-10-20) | `implementation` | `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py`, `rust/sglang-radix-tree/src/{unified_tree_core.rs,python_bindings.rs}` | 将 Python 与 Rust `TreePathNodeInfo` 扩展为 6 元组 `(node_id, parent_id, path_start, path_end, full_value_len, num_children)`，在 `collect_path_info` 快照中直接暴露每个路径节点的直接子节点数。 |
| `ai-dynamo/dynamo` `#16500` (`b169bcc08a`, 2026-10-20) | `implementation` | `components/src/dynamo/sglang/args.py`, `components/src/dynamo/sglang/tests/test_sglang_unit.py` | 新增 `_default_cuda_coredump_wait(server_args)`：当未显式启用用户触发 CUDA coredump 或 `--crash-dump-folder` 时，自动设置 `SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS=0`，消除 `SIGQUIT` 时无意义的 60 秒休眠。 |

## 2. 关键实现路径、数据结构与状态机

### 2.1 多 Tokenizer Worker 批量回传分派与保序不变式（`#45280`）
在多网关或多 Tokenizer Worker 部署（`tokenizer_worker_num = W`）下，一个推理批次包含 $B$ 个交错归属不同 IPC 通道的请求：
- 当 `SGLANG_BATCH_TOKENIZER_OUTPUTS=True`（默认）时，`_split_batch_by_key(output, ipcs)` 将同一 `ipc_name` 的索引聚合成单次 ZMQ IPC 发送，ZMQ 发送次数从 $B$ 降至 $\le W$（在 $B=12, W=3$ 算例中从 `12` 次降至 `3` 次，减少 `75%`），且严格保持每个 owner 首次出现的顺序（例如 `["worker-b", "worker-a", "worker-c"]`）。
- 当 `SGLANG_BATCH_TOKENIZER_OUTPUTS=False` 时，回退至逐请求 `_handle_output_by_index(output, index)` 发送，适用于超低并发下追求首包零聚合延迟的场景。

### 2.2 Rust/Python 统一 Radix 树 `TreePathNodeInfo` 6 元组契约（`#45241`）
`UnifiedTreeCore.collect_path_info` 从叶子向根回溯构造不可变路径快照，新增第 6 字段 `num_children = len(node.children)`（Rust 端 `node.children.len()`）：
$$\text{TreePathNodeInfo} = (\text{node\_id},\; \text{parent\_id},\; \text{path\_start},\; \text{path\_end},\; \text{full\_value\_len},\; \text{num\_children})$$
- **拓扑语义**：`num_children == 0` 标识路径叶节点，`num_children == 1` 标识单链中间节点，`num_children >= 2` 标识被多条前缀共享的分叉节点（如插入分支后头部节点 `num_children` 从 `1` 变为 `2`），而且历史快照保持不可变（`frozen=True`）。

### 2.3 Dynamo SGLang 崩溃前 CUDA Coredump 等待门控（`#16500`）
当 detokenizer 异常退出触发 `SIGQUIT` 时，SGLang `TokenizerManager` 默认尝试触发 CUDA user-triggered coredump 并无条件 `sleep(60)`。Dynamo 在 `parse_args` 中调用 `_default_cuda_coredump_wait(server_args)`：
$$\text{wait\_secs} = \begin{cases} \text{env}[K] & \text{if } K \in \text{env} \\ 60 & \text{if } \text{CUDA\_ENABLE\_USER\_TRIGGERED\_COREDUMP}=\text{"1"} \lor \text{crash\_dump\_folder} \neq \text{None} \\ 0 & \text{otherwise} \end{cases}$$
其中 $K = \text{SGLANG\_CUDA\_COREDUMP\_BEFORE\_CRASH\_WAIT\_SECS}$，默认情况下将 Kubernetes Pod 故障重启 MTTR 缩短整整 **60 秒**。

## 3. 确定性实验验证与架构师练习

- **验证命令**：`python experiments\sglang_detok_batch_radix_child_count_dynamo_coredump\run.py --check`
- **可证伪问题**：在高吞吐流式生成（$B=128, W=4$）下，开启 `SGLANG_BATCH_TOKENIZER_OUTPUTS=True` 节省的 96.9% ZMQ IPC 系统调用开销是否始终大于批次内切分子集 `_handle_output_by_indices` 的 Python 切片开销？
