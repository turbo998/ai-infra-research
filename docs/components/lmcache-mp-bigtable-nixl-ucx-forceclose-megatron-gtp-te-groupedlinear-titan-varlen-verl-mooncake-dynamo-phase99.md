# LMCache 共享 MP 存储提取与 Bigtable L2 超限失败保护、NIXL UCX 强制关闭端点与条件错误回调掩码、Megatron-LM GTP 异步 Gather 签发即记录事件、TransformerEngine GroupedLinear 前反向重构、TorchTitan 图安全变长 CP All-Gather、verl 采样支撑对齐分数中心化、Mooncake TENT 插件拷贝恢复调用方 CUDA 上下文及 Dynamo 零迭代器 Token 提取（Phase 99 · Priority 108）

> **English Edition**: [lmcache-mp-bigtable-nixl-ucx-forceclose-megatron-gtp-te-groupedlinear-titan-varlen-verl-mooncake-dynamo-phase99.en.md](lmcache-mp-bigtable-nixl-ucx-forceclose-megatron-gtp-te-groupedlinear-titan-varlen-verl-mooncake-dynamo-phase99.en.md)

## 1. 背景与核心问题

在多级 KV 存储（LMCache/Mooncake）、RDMA/UCX 传输控制面（NIXL/Dynamo）以及大规模分布式训练与 RL 后训练（Megatron-LM / TransformerEngine / TorchTitan / verl）中，Phase 99 聚焦八项关键可靠性与性能演进：

1. **LMCache 共享 MP 存储组件提取与 Bigtable L2 超限写入失败报告（`LMCache#6010`, `398e14377c`; `LMCache#5471`, `a3b1ebd9cd`）**：
   - `#6010` 将 `raw_block` 等存储组件统一提取到与多进程（MP）模式共享的 `l2_adapters` 层；
   - `#5471` 在混合批次写入 Cloud Bigtable L2 时，对超出单列或单请求大小上限的条目显式报告任务级失败，从而在 buffer-only 策略下保留 L1 副本不丢失，同时正常提交同批次合法写入并准确记账。
2. **NIXL UCX 端点强制关闭与条件错误回调掩码（`nixl#2365`, `41ff71bb6d`; `nixl#2423`, `c8178f563a`）**：
   - `#2365` 在启用错误处理模式时通过 `UCP_EP_CLOSE_FLAG_FORCE` 强制关闭 UCP Endpoint，解决 UCP 永久保留已关闭非故障 EP 导致的资源泄漏；
   - `#2423` 修复 `#2365` 后在 `UCP_ERR_HANDLING_MODE_NONE` 下仍置位 `UCP_EP_PARAM_FIELD_ERR_HANDLER` 导致对端退出时调用未初始化野指针触发 Segfault 的致命缺陷（仅在提供 handler 时置位 `field_mask`）。
3. **Megatron-LM GTP 签发即记录异步 Gather 完成事件与 TE GroupedLinear 重构（`Megatron-LM#8428`, `e3f528380c`; `TransformerEngine#3606`, `e6afa9ddfa`）**：
   - `#8428` 在 Generalized Tensor Parallelism（GTP）签发异步 All-Gather 时立即记录所有完成事件，消除延迟记录导致的流同步竞态窗口；
   - `#3606` 重构 `GroupedLinear` Eager 前向与反向，从已验证的前向张量推导反向特征维度并保持 saved-tensor 布局，使分布式权重分片不会污染 GEMM 形状且可被 `torch.compile` 复用。
4. **TorchTitan 图安全变长 CP All-Gather 与 verl 采样支撑对齐分数中心化（`torchtitan#5071`, `54682c86fb`; `verl@7e98752499`）**：
   - `#5071` 为 `AllgatherKV` 变长注意力（`VarlenAttention`）固定查询分块数（配合 `HeadTailCPLoadBalancer`），使文档边界在迭代间变化时无需重捕 CUDA Graph；
   - `verl@7e98752499` 将 PPO/GRPO 分数中心化（Score Centering）与经核验的采样器支撑集（top-k/top-p support coverage）自动对齐，在支撑集不完整时保留 recorded-head trainer-tail 近似，将尾部偏差降低 **`61.33x`**。
5. **Mooncake TENT 恢复调用方 CUDA 上下文与 Dynamo PyO3 零迭代器 Token 提取（`Mooncake#4839`, `b2c7d0a7b5`; `dynamo#17337`, `e3b7b76225`）**：
   - `#4839` 在 TENT 设备插件拷贝（含正常完成、拷贝报错及无上下文线程）后精确恢复调用方 CUDA Context，且严禁静默退回主机 `memcpy`；
   - `#17337` 用类型化 PyO3 `extract_token_ids` 替代 `pythonize` 逐元素 serde 迭代器，在 8,192-token 提示词下提速 **`8.0x`**（`1120.0 us -> 140.0 us`）。

## 2. 定量建模与架构验证（`own-offline`）

通过确定性标准库模型 [`experiments/lmcache_mp_bigtable_nixl_ucx_forceclose_megatron_gtp_te_groupedlinear_titan_varlen_verl_mooncake_dynamo_phase99/run.py`](../../experiments/lmcache_mp_bigtable_nixl_ucx_forceclose_megatron_gtp_te_groupedlinear_titan_varlen_verl_mooncake_dynamo_phase99/run.py) 验证：

| 子系统与优化路径 | 基线表现 | Phase 99 优化后表现 | 关键量化收益 / 架构不变量 |
| --- | --- | --- | --- |
| LMCache MP Bigtable L2 超限写入保护 (`#5471/#6010`) | 超限静默丢弃或整批失败丢 L1 | 60/64 合法提交 + 4 个超限保留 L1 | **`0`** L1 缓存数据丢失 |
| NIXL UCX EP 强制关闭与条件 `ERR_HANDLER` 掩码 (`#2365/#2423`) | 512 个泄漏 EP / 野指针 Segfault | `UCP_EP_CLOSE_FLAG_FORCE` + 按需置位掩码 | **`0`** 泄漏 EP、**`0`** 断连 Segfault |
| Megatron-LM GTP 签发即记录完成事件 + TE GroupedLinear (`#8428/#3606`) | 32 次延迟记录竞态窗口 / 分片形状污染 | 签发即记录 + 前向张量推导反向维度 | **`0`** 异步 Gather 竞态、**`0`** 反向形状失配 |
| TorchTitan 固定分块图安全变长 CP All-Gather (`#5071`) | 100 步重捕 100 次 CUDA Graph | 100 步仅首次捕获 **1 次** | 重捕降 **`100x`**，CP 步速提升 **`1.462x`** |
| verl 采样支撑对齐分数中心化 (`7e98752499`) | 未对齐尾部偏差 `0.184` | Recorded-head trainer-tail 偏差 `0.003` | 策略梯度中心化偏差降低 **`61.33x`** |
| Mooncake TENT 插件拷贝恢复调用方 CUDA 上下文 (`#4839`) | 跨卡插件拷贝污染调用线程 Context | 全路径精确保存/恢复调用方 Context | **`0`** CUDA Context 污染或隐式 Host 回退 |
| Dynamo Rust `SelectionService` Token 列表提取 (`#17337`) | `pythonize` 逐元素迭代 `1120.0 us` | 类型化 PyO3 `extract_token_ids` `140.0 us` | **`8.0x`** 绑定层提取加速 |

## 3. 关联证据与后续可证伪问题

- **证据 ID**：`claim-lmcache-mp-bigtable-nixl-ucx-forceclose-megatron-gtp-te-groupedlinear-titan-varlen-verl-mooncake-dynamo-phase99`
- **一手信源**：
  - `lmcache-refactor-extract-storage-components-shared-wi-6010-398e143-2026`
  - `lmcache-bugfix-mp-report-failure-for-rejected-bigtabl-5471-a3b1ebd-2026`
  - `nixl-7-n-plugins-ucx-force-close-eps-2365-41ff71b-2026`
  - `nixl-plugins-ucx-set-err-handler-field-only-when-a-2423-c8178f5-2026`
  - `megatron-lm-gtp-record-all-asynchronous-gather-completion-8428-e3f5283-2026`
  - `transformerengine-pytorch-refactor-groupedlinear-forward-and-ba-3606-e6afa9d-2026`
  - `torchtitan-cp-add-graph-safe-varlen-k-v-all-gather-5071-54682c8-2026`
  - `verl-fix-automatically-align-score-centering-with-7e98752-2026`
  - `mooncake-bugfix-tent-restore-caller-cuda-context-after-4839-b2c7d0a-2026`
  - `dynamo-perf-bindings-read-selectionservice-token-ids-17337-e3b7b76-2026`
- **确切限制**：全部数据基于上游源码补丁分析与本地 Python 3.12 标准库确定性建模（`own-offline`）。
