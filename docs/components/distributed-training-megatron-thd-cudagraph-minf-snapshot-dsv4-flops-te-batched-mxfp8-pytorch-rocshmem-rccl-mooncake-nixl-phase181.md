# 分布式训练与通信传输栈：Megatron-LM THD 序列打包注意力 CUDA Graph 与 MInf 实时 Token 快照、TE 批量 MXFP8 GEMM、PyTorch Triton `rocSHMEM` 设备端集合通信、RCCL Zen CPUID 拓扑修复及 Mooncake/LMCache/NIXL 传输加固（Phase 181）

> 🌐 **Language / 语言切换**: **[中文](distributed-training-megatron-thd-cudagraph-minf-snapshot-dsv4-flops-te-batched-mxfp8-pytorch-rocshmem-rccl-mooncake-nixl-phase181.md)** | **[English](distributed-training-megatron-thd-cudagraph-minf-snapshot-dsv4-flops-te-batched-mxfp8-pytorch-rocshmem-rccl-mooncake-nixl-phase181.en.md)**

---

## 1. 核心架构问题与一手源码证据（Pillar 0 & Pillar 1）

Phase 181（`priority: 190`）审计了 Megatron-LM、TransformerEngine、PyTorch SymmetricMemory、ROCm RCCL/ROCr、Mooncake、LMCache 与 NIXL 的 **9 组（11 个）上游主干提交**：

1. **Megatron-LM `dp_balanced` THD 序列打包 TE 注意力 CUDA Graph、MInf RL 实时请求 Token 级快照、DSv4 混合模型注意力 FLOPs 计量与流水线跨流存储保护**：
   - `631e146322af` (#8361) 为主干 `dp_balanced` 序列打包调度器生成的 THD 批次启用了 Transformer Engine 注意力 CUDA Graph 捕获与重放：固定 token 与元数据缓冲区形状，同时保留动态变化的文档边界、4 组逻辑/物理 Q/KV 偏移张量以及动态打包微批次数（`4 -> 2 -> 4`，填充尾贡献 0 真实 token），在 4x GB200（`PP2/CP2` 与 `PP2/VPP2/CP2`，含真实融合 CSA ratio-4 稀疏索引损失与 HCA ratio-128）上 **100% 通过全部 `288/288` 项**输出与梯度一致性校验。
   - `01fa4164c8b5` (#8137) 在 Megatron Inference (`MInf`) 中新增 `DynamicInferenceEngine.snapshot_live_requests()`（返回 `LiveRequestSnapshot`）与 `ResumedGeneration` 恢复机制，使 RL 后训练在抢占或中断推理引擎时能够以 Token 级粒度保存并恢复在途请求状态（`0` 丢失在途生成 token）。
   - `e1afbb93db5c` (#7972) 补齐了 `HybridModel` 训练指标中遗漏的 DeepSeek-V4 C/H/W 注意力层（含 Q/KV/O 投影、窗口注意力、压缩、ratio-4 索引器与稀疏注意力、ratio-128 注意力及 MTP 深度）FLOPs 计算，通过全部 `48/48` 项 CPU 算例校验；`a0ef64615cc3` (#8823) 修复 `ScheduleNode` 仅在 `free_input=True` 时在消费流上记录输入存储的漏洞，对所有输入张量无条件调用 `record_stream`（消除异步反向传播期间生产者流覆写仍在使用的输入显存竞态）。
2. **TransformerEngine `BatchedLinear` BF16/MXFP8 跨步批量 GEMM 与 `AdamCapturableFunctor` 访存重排**：
   - `b021980343e3` (#3160) 新增支持 BF16 与 MXFP8 的跨步批量 GEMM（`strided_batched_gemm`）与 `ops.BatchedLinear`，自动从张量批次维度推导布局并对非 MXFP8 量化回退至高精度 GEMM。
   - `5c3f067c28b6` (#3782) 优化 CUDA Graph 可捕获的 `AdamCapturableFunctor` 内核：仅读取一次 `lr` 与 `inv_scale`，在发出全部全局内存加载指令后再写出反缩放梯度，并在 `inv_scale == 1` 时直接跳过该写出（`0` 冗余反缩放写操作），消除加载指令间的串行等待。
3. **PyTorch Triton `rocSHMEM` 设备端 `alltoall`/`broadcast` 与 ROCm RCCL Zen 2/3/4 CPUID 拓扑恢复 / ROCr SVM 指针识别**：
   - `fa6060d66f9d` (#202454) 在 PyTorch SymmetricMemory 的 `rocSHMEM` Triton 后端中通过 LLVM IR bitcode 包装层（在包装层内注入 `rocshmem_ctx_t` 与 `rocshmem_team_t` 并向 Triton 暴露纯整型参数接口）启用了设备端 `alltoall` 与 `broadcast`（`2` 项设备端集合通信原语）。
   - `b1a305f97a6a` (#14146) 修复了 NCCL 2.31.2 同步后将拓扑 XML 中的 CPU family 从旧编码 `143/175/191` 改为原始 CPUID `0x17/0x19/0x1A`（`23/25/26`）导致所有真实 AMD Zen 2/3/4（Rome/Milan/Genoa）主机无法匹配 Rome 类的严重回退：新增 `rcclTopoCpuIsRome()`，同时恢复了 `parseRome4P2H`、`parse1H16P`、`parse4H4P`、用户环 `RCCL_TOPO_4P2H_ROME` 标记以及同 CPU Root Complex 下 GPU+NIC 的 **`GDR PHB`** 放宽策略（共 `5` 个消费端）；`34442d5ccd80` (#14694) 使 ROCr `hsa_amd_pointer_info` 在调用 `hsa_amd_svm_attributes_set` 后准确返回 `HSA_EXT_POINTER_TYPE_SVM (7)` 而非 `RESERVED_ADDR`。
4. **Mooncake XPU IPC 导出进程 `pidfd` 退出回收与 DFS LRU 键去重、LMCache 非阻塞 ZMQ 及 NIXL 范围溢出/断连修复**：
   - Mooncake `e5ef0e427a49` (#5026) 在 `XpuTransport` 中通过专用轮询线程监听每个 Level Zero IPC 导出进程的 `pidfd`，在导出进程（如崩溃重启的预填充/解码对端）退出时立即关闭导入映射（在途拷贝完成后释放），消除对端异常退出导致的 VRAM 永久泄漏（`0` 泄漏字节）；`e8dda21b8fab` (#4787) 将 `ShardAllocator` DFS LRU 索引键改为指向 `std::list` 节点的 `std::string_view`，将每对象键拷贝从 `2` 份减至 `1` 份（节省 **50%** 键内存并消除二次堆分配）。
   - LMCache `719f1d643558` (#6256) 将多进程 ZMQ 客户端发送改为 `NOBLOCK` + `POLLOUT` 事件轮询，防止单个阻塞/宕机服务端（如堆积 1,100 个请求）冻结共享事件循环中的其他健康客户端（`0` 队头阻塞冻结），`236f3eec32a2` (#6208) 抽象 `PeerL2AdapterBase` 并支持可重试卸载；NIXL `a91561a83f81` (#2142) 修复 `nixlBasicDesc::covers()` 在 `SIZE_MAX` 无界文件注册与非零偏移下的整数加法溢出（`unbounded_covers_finite=1, finite_covers_wrapping=0`），`c656976fcd6d` (#2560) 在 `loadRemoteSections` 失败时主动断开 `remoteBackends_` 中的 UCX 连接，消除重连时的 `NIXL_ERR_INVALID_PARAM` 死锁。

---

## 2. 端到端数据通路与架构拓扑图（Pillar 1）

```
+------------------------------------------------------------------------------------------------------+
| Phase 181 Distributed Training, Collectives & KV Fabrics: Megatron, TE, PyTorch, RCCL, Mooncake, NIXL|
+------------------------------------------------------------------------------------------------------+
  [Megatron-LM & TransformerEngine Training / RL Rollout Stack (#8361, #8137, #7972, #8823, #3160, #3782)]
    |-- Megatron THD CUDA Graph (#8361) --> dp_balanced Packed TE Attention Graph (288/288 GB200 Checks)
    |-- Megatron MInf Snapshot (#8137)  --> Token-Level LiveRequestSnapshot + ResumedGeneration for RL
    |-- Megatron DSv4 & PP (#7972/#8823)--> C/H/W+MTP FLOPs (48/48) + Unconditional Consumer Stream Record
    +-- TE Batched GEMM & Adam (#3160/#3782)-> BF16/MXFP8 BatchedLinear + Capturable Adam Load-Before-Store
                                      |
                                      v
  [PyTorch rocSHMEM, ROCm RCCL/ROCr, Mooncake, LMCache & NIXL (#202454, #14146, #5026, #4787, #6256, #2142)]
    |-- PyTorch SymmetricMemory (#202454)--> LLVM IR Bitcode Wrapper -> Triton rocSHMEM alltoall/broadcast
    |-- ROCm RCCL & ROCr (#14146/#14694)--> Zen 2/3/4 CPUID 23/25 Rome Restore (5 Consumers + GDR PHB)
    |-- Mooncake TENT & Store (#5026/#4787)-> XPU IPC Exporter pidfd Watch (0 Leak) + string_view LRU (-50%)
    +-- LMCache & NIXL (#6256/#2142/#2560)-> ZMQ NOBLOCK+POLLOUT + Overflow-Safe covers() + UCX Disconnect
```

---

## 3. 确定性离线算例验证（Pillar 2 & Pillar 3）

通过 [`experiments/megatron_thd_cudagraph_minf_snapshot_dsv4_flops_te_mxfp8_rocshmem_rccl_mooncake_nixl_phase181/run.py`](../../experiments/megatron_thd_cudagraph_minf_snapshot_dsv4_flops_te_mxfp8_rocshmem_rccl_mooncake_nixl_phase181/run.py) `--check` 100% 验证通过：
- `megatron_thd_cudagraph_gb200_checks_passed = 288`、`megatron_thd_cudagraph_pass_rate = 1.0`
- `megatron_minf_rl_interrupted_tokens_lost = 0`、`megatron_dsv4_hybrid_flops_cases_passed = 48`、`megatron_pp_retained_input_stream_races = 0`
- `te_batched_linear_mxfp8_supported = True`、`te_capturable_adam_redundant_unscaled_writes_at_unit_scale = 0`
- `pytorch_rocshmem_triton_device_collectives = 2`、`rccl_zen_rome_restored_consumers = 5`、`rocr_svm_pointer_type_id = 7`
- `mooncake_xpu_ipc_exporter_exit_leaked_vram_bytes = 0`、`mooncake_shard_allocator_key_copy_reduction_ratio = 0.50`
- `lmcache_zmq_blocked_client_head_of_line_stalls = 0`、`nixl_unbounded_file_covers_finite = 1`、`nixl_finite_covers_wrapping_range = 0`、`nixl_section_error_orphaned_ucx_connections = 0`

---

## 4. 企业级业务场景落地指南（Pillar 4）

| 业务场景维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **GB200 长序列打包预训练与 RL 异步 Rollout** | 在 GB200 `PP/VPP/CP` 训练中启用 `dp_balanced` THD 序列打包 TE 注意力 CUDA Graph（`#8361`），并在 RL 推理引擎重平衡/抢占前调用 `MInf` `snapshot_live_requests()` 保存 Token 级在途状态（`#8137`）。 | 在微批次数超出预捕获上界时不经检查直接重用旧图（`#8361` 会在微批溢出前显式拒绝重用）。 |
| **AMD Zen 2/3/4 + MI300/MI350 集群通信与 Triton 内核内通信** | 升级包含 `#14146` 的 RCCL 以恢复 Milan/Genoa 主机上的 `1H16P`/`4H4P` 拓扑预设与同 Root Complex GPU-NIC `GDR PHB` 零拷贝路径，并在 Triton 内核中使用 PyTorch `rocSHMEM` 设备端 `alltoall`/`broadcast`（`#202454`）。 | 在未打 `#14146` 补丁的 NCCL 2.31+ 衍生 RCCL 上运行无 `noCpuCheck` 的 Rome/Milan/Genoa 拓扑预设（会因 CPUID `23/25` 不匹配 `143/175` 而丢失 GDR PHB）。 |
| **跨进程 KV 缓存共享（Mooncake / LMCache / NIXL）** | 部署 Mooncake `#5026` `pidfd` 监控以防分离式节点重启泄漏 IPC 显存、LMCache `#6256` `NOBLOCK+POLLOUT` 防单慢节点拖垮全局 ZMQ 循环，以及 NIXL `#2142/#2560` 安全文件范围校验与元数据异常自动断连。 | 在 `loadRemoteSections` 失败后仅擦除本地 map 而不调用后端 `disconnect`（会导致 UCX `remoteConnMap` 残留旧连接并永久拒绝重连）。 |
