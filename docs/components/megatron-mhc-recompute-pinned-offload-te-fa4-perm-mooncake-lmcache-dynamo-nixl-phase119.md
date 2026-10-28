# Megatron-LM mHC 精度控制与固定锁页卸载池、TE FA3/FA4 THD 掩码与置换降显存、Mooncake 逻辑盒重分片、LMCache MP 边界、Dynamo BLAKE3 校验与 NIXL V3 遥测（Phase 119 / Priority 128）

> **证据分类**：Upstream `implementation` + `author-benchmark`（10 个上游公开提交，2026-10-01 至 2026-10-28）+ 本仓库 `own-offline` 确定性算术/拓扑验证（`experiments/megatron_mhc_recompute_pinned_offload_te_fa4_perm_mooncake_lmcache_dynamo_nixl_phase119/run.py`）。本文不包含未标注的真实 GPU 硬件重测声明。

## 1. 核心架构问题与背景

在超大规模混合架构训练（mHC / MoE）、FP8/FA4 上下文并行、跨集群权重重分片（Mooncake）以及多进程 KV 缓存与 NIXL 传输引擎中，近期解决了五类关键内存膨胀与正确性问题：

1. **Megatron-LM mHC 激活重计算 Autocast 漂移与 MoE 变长卸载触发的锁页内存（Pinned Memory）无界增长**：
   - `CheckpointWithoutOutputFunction` 在反向重放前向时未恢复原始前向的 `torch.autocast` 状态，且 MoE `combine_postprocess` 未及时释放 `self.probs` 导致重计算图被钉死到下一迭代；
   - 细粒度激活卸载在处理变长 MoE 组（`expert_fc1`、`moe_act`、`fused_group_mlp`）时直接向 PyTorch Pinned Host Allocator 申请内存，该分配器按 **2 的幂次（Power-of-Two）向上取整**并缓存已释放块，随路由波动不断钉死新的宿主机物理内存。
2. **TransformerEngine FA3/FA4 `pad_between_seqs=True` 脏填充行与 MoE 置换参考实现 30 GB 显存峰值**：FA3/FA4 内核在物理填充行与缓冲区尾部可能留下未写回的垃圾值，在 CP All-to-All 重排前若不清零会污染跨卡梯度；同时 JAX MoE 置换参考实现按 `[num_tokens * num_experts, hidden_size]` 物化中间张量，单测峰值高达 **30 GB**。
3. **Mooncake 已存储权重分片与训练 Rank 语义耦合及昇腾容器内 DSMI 亲和性 ID 错位**：已发布至对象存储的权重快照（`StoredWeightManifest`）不应依赖源训练拓扑或运行时 Rank；且昇腾容器在设置 `ASCEND_RT_VISIBLE_DEVICES` 后，ACL 用户设备 ID（User ID）与驱动逻辑 ID（Logic ID）不再一致，误调 `aclrtGetPhyDevIdByLogicDevId` 会导致 DSMI URMA 网卡亲和性查询失败。
4. **LMCache MP 有符号指针截断与混合/DCP 部分命中下的少存（Under-Storing）Bug**：`lmcache_objects_ptrs` 使用 `int64_t` 存在高地址符号位隐患；且在混合注意力或 DCP 下，`vllm_hit` 按块向下取整（如 `112` 实际已算 Token 被取整为 `64`），仅靠 `num_scheduled_tokens + max(vllm_hit, lmcache_hit)` 会把上限算成 `252 -> 192`，漏存 `[192, 256)` 区间。
5. **Dynamo GMS 检查点缺乏流式哈希校验与 NIXL 遥测析构顺序/丢包黑洞**：NIXL GDS/POSIX 权重落盘与恢复缺少并发受控的流式 BLAKE3 完整性校验；NIXL 传输引擎在环形缓冲区满丢弃 Trace Phase 时无计数指标，且 `telemetry_` 成员声明在 `backendEngines_` 之后导致引擎进度线程退出时访问已析构的遥测实例。

---

## 2. 上游一手实现剖析（10 个零重叠 Commit）

| # | 仓库与提交 | 日期 | 核心实现路径与微架构机制 |
|---|---|---|---|
| 1 | `NVIDIA/Megatron-LM@8e134118ba` (`#7521`) | 2026-10-27 | `megatron/core/transformer/hyper_connection.py` & `recompute.py`：引入 mHC 精度控制（`_sinkhorn_iterations(input_logits.detach())`、禁用 autocast 的 `native_h_post_bda` `torch.bmm`）、可选均值输出收缩（`mhc_learned_output_contract=False`）、`CheckpointWithoutOutputFunction` 保存/恢复 CUDA autocast 上下文、在 `combine_postprocess` 末尾置 `self.probs = None` 释放重计算图，并在 `HybridOptimizer.update_fp32_param_by_new_param` 中同步刷新 `param_to_inner_param` CPU 副本。 |
| 2 | `NVIDIA/Megatron-LM@57018e45e4` (`#7961`) | 2026-10-27 | `megatron/core/pipeline_parallel/pinned_activation_buffer.py`：新增 `--fine-grained-offloading-buffer-size-gib`（`PinnedActivationBuffer`），通过 `cudaHostAlloc` 一次性分配 256 字节对齐的固定大小锁页缓冲区（不按 2 的幂次取整），所有微批次/变长 MoE 激活顺序追加并在全部重载完成后整体复用；同时为 CUDA Graph 捕获预留独立地址区间 `[0, graph_end)`，Eager 轮次仅使用尾部剩余空间。 |
| 3 | `NVIDIA/TransformerEngine@f38a1fce7f` (`#3205`) | 2026-10-02 | `transformer_engine/pytorch/attention/dot_product_attention/`：扩展 `get_thd_padding_mask(num_tokens, cu_seqlens, cu_seqlens_padded)` 同时覆盖序列间空隙与缓冲区尾部 `[cu_seqlens_padded[-1], num_tokens)`；非 CP 路径通过自定义 `_MaskTHDPaddingGrad` 在反向清零填充梯度，CP 路径在逆向 All-to-All 改变序列顺序前调用 `_zero_thd_padding` 清零 `out` 与 `dq/dk/dv`。 |
| 4 | `NVIDIA/TransformerEngine@0838ecbe4f` (`#3609`) | 2026-10-01 | `tests/jax/test_permutation.py`：重构 `_reference_permute_impl`，不再物化 `[num_tokens * num_experts, hidden_size]` 的 `gathered_inp`，而是先将一维源 Token 索引 scatter 到长度为 `num_out_tokens` 的 `src_tokens`（未命中位置填越界值 `num_tokens`），再执行单次 `inp.at[src_tokens].get(mode="fill", fill_value=0)`，将显存峰值从 **30 GB 降至 1.75 GB**（降低 **17.14x**）。 |
| 5 | `kvcache-ai/Mooncake@c21a541827` (`#4827`) | 2026-10-28 | `mooncake-reshard/`：验证 `StoredWeightManifest` 与 `plan_stored_transfer_to_target_placement` 的纯逻辑盒（`global_offset` + `local_shape` -> `object_key`/`object_offset`/`nbytes`）字节区间求交机制，在无源 Rank/训练网格语义下精确重建非均匀 `4->8` 与 `8->4` 跨维切片、乱序分片及带填充偏移的张量字节。 |
| 6 | `kvcache-ai/Mooncake@a8acb4da20` (`#4971`) | 2026-10-28 | `mooncake-transfer-engine/src/transport/ascend_transport/ascend_direct_transport/utils.cpp`：将 `CpusForLogicDevice` 替换为 `CpusForUserDevice`，先调用 `aclrtGetLogicDevIdByUserDevId(user_id, &logic_id)` 将 ACL 用户设备 ID 转换为驱动逻辑 ID，再查询 DSMI `kDsmiUbSubUrmaName`，修复 `ASCEND_RT_VISIBLE_DEVICES` 容器隔离下的 NUMA 亲和性绑定。 |
| 7 | `LMCache/LMCache@b540ae9ea0` (`#5672`) | 2026-10-28 | `csrc/cuda/mp_mem_kernels.cu` & `csrc/kv_transfer_plan_types.h`：将 `KernelGroupSpec` 与 `multi_layer_block_kv_transfer` 中的 `lmcache_objects_ptrs` 从有符号 `std::vector<int64_t>` 全面统一为无符号 `std::vector<uintptr_t>`。 |
| 8 | `LMCache/LMCache@4f7d283a4d` (`#5911`) | 2026-10-20 | `lmcache/integration/vllm/lmcache_mp_metadata.py`：在 `LMCacheMPRequestTracker` 中新增 `num_engine_computed_tokens = num_computed_tokens + num_new_tokens`，将存储上界更新为 `max(num_scheduled_tokens + max(vllm_hit, lmcache_hit), num_engine_computed_tokens)`，修复混合/DCP 部分命中下（如 112 已算 Token 被取整为 64 块命中 + 188 新调度 Token）将 `(64, 256)` 误截断为 `(64, 192)` 的少存缺陷。 |
| 9 | `ai-dynamo/dynamo@17bffaf9c7` (`#16146`) | 2026-10-28 | `components/src/dynamo/common/Storage/`：为 GMS NIXL 检查点落盘与恢复引入流式及恢复前 BLAKE3 完整性校验（`FileIntegrity`），并按世界大小设置每分片哈希线程上限 `blake3_threads_per_shard(w) = max(1, min(4, 16 // w))`（8 卡并发总线程数受控于 16）。 |
| 10 | `ai-dynamo/nixl@349514f94e` (`#2323`) | 2026-10-28 | `src/core/telemetry/` & `src/api/cpp/nixl.h`：新增原子计数器 `AGENT_TRACE_PHASES_DROPPED = 20`（`AGENT_TELEMETRY_EVENTS_DROPPED` 移至 `21`），将 `TELEMETRY_VERSION` 升至 `5`、MP Store Schema 升至 `2`、插件 API 升至 `V3`，并将 `nixlAgentData::telemetry_` 成员声明移至 `tracer_` 与 `backendEngines_` 之前，确保后端进度线程退出时 `telemetry_` 仍然存活。 |

---

## 3. 关键公式、数据移动与离线验证结果

### 3.1 锁页激活池 256B 对齐 vs 2 的幂次缓存膨胀 & TE 索引散射显存模型
1. **锁页卸载内存节省率**：设变长 MoE 组在连续步中请求字节集合为 $\{s_1, \dots, s_m\}$，PyTorch Pinned Allocator 独立缓存各 2 的幂次分桶时的驻留内存为 $M_{\text{pow2}} = \sum_{i=1}^m 2^{\lceil \log_2 s_i \rceil}$，而 `PinnedActivationBuffer` 单缓冲区顺序打包容量为 $M_{\text{fixed}} = \sum_{i=1}^m 256 \lceil s_i / 256 \rceil$。在典型请求序列 `[130, 190, 260, 310, 400] MiB` 下，$M_{\text{pow2}} = 2048\text{ MiB}$，$M_{\text{fixed}} = 1290\text{ MiB}$，节省 **37.01%** 宿主机锁页内存且彻底消除跨步分桶膨胀。
2. **TE 置换参考显存比**：直接 gather 输入行需物化 $[T \cdot E, H]$ 张量，而先 scatter 4 字节索引再按 `[T_{\text{out}}, H]` gather 将峰值显存从 `30.0 GB` 压缩至 `1.75 GB`（**17.14x** 缩减）。

### 3.2 离线模型验证汇总（`run.py --check`）

- `pinned_alignment_bytes` / `pinned_host_memory_savings_ratio`: `256` / `0.3701`（256B 粒度锁页池相比 2 的幂次分配器节省 `37.01%` 锁页内存）
- `thd_padding_mask_exact` / `thd_nonzero_padded_rows_after_cp_cleanup`: `True` / `0`（精确识别序列间空隙与尾部填充 `[12, 16)` 并保证 CP 逆向 A2A 前零非零残留）
- `te_perm_memory_reduction_factor`: `17.14`（置换参考实现峰值显存 `30.0 GB -> 1.75 GB`）
- `mooncake_4to8_intersections` / `mooncake_exact_elements_reconstructed`: `8` / `True`（`19x23` 二维张量 `4->8` 逻辑盒求交精确恢复全部 `437` 个元素）
- `dsmi_resolved_logic_ids`: `[2, 3]`（容器内 ACL `user_id` `(0, 1)` 正确映射为驱动 `logic_id` `[2, 3]`）
- `buggy_stored_end` / `fixed_stored_end`: `192` / `256`（修复 LMCache MP 在混合/DCP 部分命中下漏存 `[192, 256)` 块的 Bug）
- `blake3_threads_w8` / `blake3_total_threads_w8`: `2` / `16`（8 分片并发下每分片 2 线程、总计 16 线程上限）
- `nixl_plugin_api_version` / `nixl_telemetry_version`: `3` / `5`（NIXL V3 遥测插件 API 与 `AGENT_TRACE_PHASES_DROPPED = 20` 验证通过）

---

## 4. 面向云解决方案架构师的练习与可证伪问题

1. **计算练习**：在 LMCache MP 与 vLLM 集成中，设块大小 `chunk_size = 64` tokens。某请求总长 300 tokens，在混合注意力下 vLLM 已拥有 `num_computed_tokens = 112` tokens，但由于前缀对齐限制，LMCache 记录的 `num_vllm_hit_tokens` 被向下取整为 `64`（`num_lmcache_hit_tokens = 0`）。本步 vLLM 调度了剩余 `num_scheduled_tokens = 188` tokens。求修复前与修复后 `GetStoreMetadata` 可存储的完整 64-token 块区间。（答案：修复前 `computed_tokens = 188 + max(64, 0) = 252`，按 64 向下取整仅存储 `[64, 192)` 共 2 个块；修复后 `num_engine_computed_tokens = 112 + 188 = 300`，`computed_tokens = max(252, 300) = 300`，按 64 向下取整存储 `[64, 256)` 共 3 个块，多保护了 `[192, 256)` 的 64 个已算 Token）。
2. **可证伪假设**：在 C++ 类 `nixlAgentData` 中，为何将成员 `std::unique_ptr<nixlTelemetry> telemetry_` 声明在 `backendEngines_` 之前就能修复析构时的 Use-After-Free？因为 C++ 标准严格规定类非静态数据成员按**声明顺序的逆序**析构；将 `telemetry_` 声明在 `backendEngines_` 之前，意味着在析构阶段 `backendEngines_` 会先于 `telemetry_` 被销毁，从而保证后端引擎进度线程在 join/销毁过程中发出的最后一批遥测事件仍能安全写入存活的 `telemetry_`。
