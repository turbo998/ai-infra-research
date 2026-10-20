# Megatron-LM MFSDP v2 分片动量 Muon、离线 KD v2 单体归档与 CP Zigzag 重组、以及 NIXL 多进程 Prometheus 导出器剖析

> **English Version**: [megatron-mfsdp-v2-sharded-muon-offline-kd-nixl-prometheus-mp.en.md](megatron-mfsdp-v2-sharded-muon-offline-kd-nixl-prometheus-mp.en.md)

## 1. 一手源码证据与核心结论

| 仓库与提交 | 证据等级 | 源码路径 | 核心机制与定量结论 |
|---|---|---|---|
| `NVIDIA/Megatron-LM` `#8313` (`91d774de79`, 2026-10-20) | `implementation` | `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/muon.py`, `tests/unit_tests/distributed/mfsdp_v2/test_muon.py` | 实现 MFSDP v2 `FsdpMuon`：动量保持为分片 `DTensor`（`RowAtomic`），通过 `assign_owner_work` 将各 2D 矩阵 `gather` 到均衡分配的 owner rank 执行 Newton-Schulz 正交化，再 `scatter` 回非 owner rank，与未分片 Muon 保持逐位等价（`atol=0, rtol=0`）。 |
| `NVIDIA/Megatron-LM` `#5482` (`fce1e2051f`, 2026-10-20) | `implementation` | `megatron/training/distillation/{logits_saver.py,utils.py}`, `megatron/training/checkpointing.py` | 升级离线 Logits KD 至 `LOGPROBS_FORMAT_VERSION = 2`：按全局样本区间命名 `dp{d}__{start}-{end}.tar`，`v2_pack_indices` 用 `np.packbits` 将第 17 位压缩至 `1 bit/elem`（节省 `87.5%` 第 17 位存储），支持按文档边界 `cu_seqlens_padded` 的 CP Zigzag 切分与重组，并将进度文件写入绑定至末级 PP 的 `tp_rank==0 and cp_rank==0 and dp_rank==0`。 |
| `ai-dynamo/nixl` `#2405` (`60dd8b7cd0`, 2026-10-20) | `implementation` | `src/core/nixl_plugin_manager.cpp`, `docs/telemetry.md` | 在 `nixlPluginManager::loadTelemetryPlugin` 中将 `prometheus` 别名映射至多进程导出器 `prometheus_mp`（`libtelemetry_exporter_prometheus_mp.so`）并移除旧单进程导出器，解决 TensorRT-LLM 多 TP 进程共用端口仅导出单 rank 的缺陷（NVBug 6575930）。 |

## 2. 关键实现路径、数学公式与分布式协议

### 2.1 MFSDP v2 `FsdpMuon` 分片动量与 Owner-Compute 正交化（`#8313`）
`FsdpMuon` 继承自 Emerging-Optimizers 的 `Muon`，仅处理 2D 矩阵参数（非 2D 参数自动跳过交由 SGD/Adam 处理）：
1. **分片权重衰减与动量更新（`_compute_momentum_inputs`）**：每个 DP rank 在本地分片 $W_{\text{local}}, M_{\text{local}}, G_{\text{local}}$ 上先执行四种权重衰减之一（`decoupled`: $-\lambda \eta W$，`independent`: $-\lambda W$，`l2`: $G \leftarrow G + \lambda W$，`palm`: $-\lambda \eta^2 W$），随后就地线性插值更新动量：
   $$M_{\text{local}} \leftarrow \beta M_{\text{local}} + (1 - \beta) G_{\text{eff}},\quad U_{\text{in}} = \begin{cases} G_{\text{eff}} + \beta (M_{\text{local}} - G_{\text{eff}}) & \text{if nesterov} \\ M_{\text{local}} & \text{otherwise} \end{cases}$$
2. **Owner-Compute Gather / Orthogonalize / Scatter（`_step_group`）**：初始化时 `assign_owner_work` 将各参数张量均衡指派给某个 DP owner rank。每步将 `momentum_inputs` `gather` 到对应 owner 的 `full_updates[i]`，owner 在 `fp32_matmul_precision` 下调用 `self.orthogonalize(param, update.float())`，随后 `scatter` 到非 owner 的 `local_updates`；owner 自身通过 `full_updates[i].view(-1).narrow(0, offset, weight.numel())` 截取本地分片，最后调用 `fsdp_group.sync_model_weight_from_main_weight()`。

### 2.2 离线 Logits KD v2：17-Bit 词表压缩与按文档 CP Zigzag 重组（`#5482`）
1. **17-Bit 词表索引打包（`v2_pack_indices` / `v2_unpack_indices`）**：针对最高 $2^{17}=131,072$ 词表，低 16 位存为 `uint16`，第 17 位在单次迭代拼接后的单体张量 `[seq, samples_per_dp_per_iter, K]` 上调用 `np.packbits((indices >> 16).to(uint8).reshape(-1))`，将第 17 位从 v1 `torch.bool` 的 `1 byte/elem` 压缩至 `1 bit/elem`（第 17 位体积缩减 **87.5%**，总索引体积从 `3.0 B/elem` 降至 `2.125 B/elem`，节省 **29.17%**）。
2. **按文档边界的 CP Zigzag 切分（`_document_zigzag_index`）**：在 SFT / 打包序列运行中，每个文档 $[s_i, e_i)$（满足 $(e_i - s_i) \bmod (2 \cdot \text{CP}) = 0$）独立均分为 $2 \cdot \text{CP}$ 个块，CP rank $r$ 获取每个文档的第 $r$ 块与第 $2\cdot\text{CP}-r-1$ 块；尾部右填充产生的零长度区间（$e_i \le s_i$）自动跳过。
3. **后台写入故障检测与 Checkpoint Tracker 归属**：新增 `check_logits_saver_failure()` 每步检查后台异步写进程的 `_failure_event`；且在 `--skip-weight-ckpt` 下将 `latest_checkpointed_iteration.txt` 更新绑定给末级 PP 上的 `tp_rank == 0 and cp_rank == 0 and dp_rank == 0`（持有非空 `take_pending_data()` 的真实写入者），修复 `is_last_rank()`（`tp_rank = tp_size - 1`）提前写入进度导致恢复漏窗的缺陷。

### 2.3 NIXL `prometheus` -> `prometheus_mp` 统一多进程导出（`#2405`）
`loadTelemetryPlugin("prometheus")` 在构造 `.so` 路径前映射为 `prometheus_mp` 并在 `loaded_telemetry_plugins_` 中同时缓存两个键，使同一主机同一端口的所有 TP rank 共享 `/dev/shm/nixl-telemetry-<euid>-<port>`，附带 `local_rank` 标签与 `30s` `NIXL_TELEMETRY_MP_STALE_TTL`。

## 3. 确定性实验验证与架构师练习

- **验证命令**：`python experiments\megatron_mfsdp_v2_muon_offline_kd_nixl_prometheus\run.py --check`
- **可证伪问题**：当模型层数不能被 DP 大小整除时，`assign_owner_work` 在 MFSDP v2 `FsdpMuon` 中如何平衡各 owner rank 的 Newton-Schulz 矩阵正交化 FLOPs？
