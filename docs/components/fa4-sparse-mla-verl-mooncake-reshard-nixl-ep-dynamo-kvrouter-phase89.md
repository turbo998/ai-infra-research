# FlashAttention-4 SM90 内部无掩码块快路径与 SM100 64-Head 稀疏 MLA 重计算 P、verl vLLM >= 0.28 多机 Hybrid-LB Leader 门控、Mooncake Tiled COO 部分读取 / HA SIGTERM 租约释放 / SSE4.2 CRC-32C、NIXL EP Barrier Put 完成等待 / EFA 非请求写接收禁用与 Dynamo KV-Router 自定义反序列化 / OnceLock CRTC 叶子存储深度解析（Phase 89）

> **English Version**: [fa4-sparse-mla-verl-mooncake-reshard-nixl-ep-dynamo-kvrouter-phase89.en.md](fa4-sparse-mla-verl-mooncake-reshard-nixl-ep-dynamo-kvrouter-phase89.en.md)  
> **证据等级**: `implementation`（固定 commit 源码审阅）+ `author-benchmark`（上游 PR 实测数据，非本项目硬件重测）+ `own-offline`（`experiments/fa4_sparse_mla_verl_mooncake_reshard_nixl_ep_dynamo_kvrouter_phase89/run.py` 纯标准库建模）

---

## 1. 背景与架构师问题定义

在百卡至千卡规模的后训练（RLHF/GRPO）与分离式推理（Disaggregated Serving）集群中，端到端吞吐与高可用性同时受制于 **长序列反向激活显存峰值、多节点 Rollout 负载均衡协议兼容、在线稀疏权重重分片（Resharding）拓扑耦合、元数据主备切换（HA Failover）租约排空、GPU 发起的单边 RDMA/NVLink Put 屏障竞态、以及全局 KV 路由器的 JSON 解析与前缀树内存回收开销**。本篇基于 `Dao-AILab/flash-attention`、`volcengine/verl`、`kvcache-ai/Mooncake`、`ai-dynamo/nixl` 与 `ai-dynamo/dynamo` 的 10 个非重叠一手 commit 进行系统拆解。

---

## 2. 核心实现路径与分布式不变量剖析

### 2.1 FlashAttention-4：SM90 内部无掩码块跳过 `FSEL` 与 SM100 64-Head 稀疏 MLA 重计算 $P$（`flash-attention@eed1971f51` #2904, `flash-attention@6a3ecb9be2`）
1. **SM90 前向内部无掩码 KV 块跳过因果/局部掩码（`flash-attention@eed1971f51`, PR #2904）**：
   - 在 `flash_attn/cute/flash_fwd_sm90.py` 中，前向主循环虽已将 KV 迭代切分为“对角线/窗口边缘需掩码块”与“内部完全可见块”两段，但第二段循环仍传入了绑定了 `mask_causal / mask_local` 的 `mask_fn`，导致每个内部块迭代额外执行 **64 条 `FSEL`（浮点条件选择）指令**。
   - 对齐 SM100 逻辑，在内部块循环仅当用户定义了自定义 `mask_mod` 时才传入掩码函数，输出与 LSE 逐位不变（Bitwise Unchanged）。
   - **作者实测（`author-benchmark`，H200 BF16 Forward）**：`hdim128 causal 16k` 从 `613 TFLOPS` 提升至 **`660 TFLOPS`**（**`+7.67%`**）；`hdim64 causal 16k` 从 `463 TFLOPS` 提升至 **`531 TFLOPS`**（**`+14.69%`**，超越 FA3 的 `514 TFLOPS`）。
2. **SM100 稀疏 MLA 反向重计算 $P$ 支持 64 Q Heads（`flash-attention@6a3ecb9be2`）**：
   - 稀疏 MLA 反向传播通过重计算 $P$（`gather_bwd_recompute_p=True`，复用 `tiled_mma_VdO` 计算 $S^T$）消除保存整张 $P$ 矩阵的显存开销。此前该路径被硬编码限制在 `num_head == 128`；由于反向 Tile 高度本身支持 `64` 或 `128` 行（`num_head == 64` 时 `tile_m = 64, pad_qheads = False`，无需像 24/96 头那样对 `lse_log2` 补 `+inf`），现放开至 `num_head in (64, 128)`。
   - **作者实测（`author-benchmark`，GB200 64-Head 64k Tokens 训练步，`B=1, gather_width=2048, token_chunk=4096`）**：相比 `load-P` 路径（`144.3 ms`，峰值显存 `41.2 GiB`，保存激活 `20.3 GiB`），`recompute-P` 路径耗时 `150.8 ms`（仅慢 `4.5%`），但 **峰值显存骤降至 `13.7 GiB`（降低 `66.75%`），保存激活降至 `8.0 GiB`（节省 `12.3 GiB`）**。

### 2.2 verl：修复 vLLM >= 0.28 多机 Rollout DP Leader `data_parallel_start_rank` 拒连故障（`verl@3dd97a0e1b`, PR #8234）
- **故障机理**：在多节点 vLLM Rollout 副本（`data_parallel_size > data_parallel_size_local`）中，`vLLMHttpServer` 在 `node_rank == 0` 启动非 Headless API Server，在 `node_rank > 0` 启动 `--headless` Worker。自 vLLM 0.28 起，`EngineArgs.create_engine_config` 只要检测到 `data_parallel_start_rank is not None and not headless`，就会将 Leader 强制推入 `hybrid-LB` 模式，并立即拒绝所有 `--headless` 远端引擎（报错 `Remote engine 15 must not use --headless in external or hybrid dp lb mode`）。
- **修复与实测（`author-benchmark`）**：在 `verl/workers/rollout/vllm_rollout/vllm_async_server.py` 中仅当 `self.node_rank > 0` 时传递 `data_parallel_start_rank`。在 **128 节点 GB200 集群（16 副本 × 32 GPUs，DeepSeek-V4-Pro）** 上验证通过，GRPO 端到端稳定运行（Rollout/Actor 对数概率 Pearson 相关系数达 **`0.995`**）。

### 2.3 Mooncake：目标无关 Tiled COO 稀疏部分读取、HA SIGTERM 排空后主动释放租约与 SSE4.2 CRC-32C（PR #3751, #4814, #4821）
1. **目标无关 Tiled COO 稀疏权重重分片规划器（`Mooncake@880bcf6763`, PR #3751）**：
   - 在 `mooncake-reshard/python/mooncake/reshard/weight/tiled_coo.py` 中新增 `TiledCOOLayout` 与 `plan_tiled_coo_read`。生产者按“源记录 $\to$ 行主序全局对齐 Tile”排列稀疏坐标（Coordinates）与数值（Values），并为每条记录维护紧凑的连续 Tile 目录，**无需按特定目标推理拓扑预先重分区**；支持训练端与推理端采用不同并行维度切分时的 2D 盒求交（`box_intersection`，上限 `_MAX_HIGH_DIMENSIONAL_PAIRWISE_COMPARISONS = 1,000,000`）。
2. **Mooncake-Store HA SIGTERM 优雅排空后主动释放 Leader 租约（`Mooncake@ac6b494e5b`, PR #4814）**：
   - 此前 HA Master 在收到 `SIGTERM` 排空 RPC 与 OpLog 后直接退出，遗留 etcd Leader 租约等待 TTL 超时才能完成主备切换。
   - 修复后在 `MasterServiceSupervisor::FinishShutdown` 中（15 秒 `kHaShutdownDeadline` 看门狗保护下）：先停止 `StopHttpMetadataCleanup()` 与 `LeadershipRenewalGuard` 续租轮询线程（防止续租与撤销竞态），再调用一次 `ReleaseLeadership`；若释放未获确认则以退出码 `kHaShutdownReleaseFailedStatus = 2` 告警退出，使计划内滚动重启实现毫秒级零 TTL 等待切换。
3. **Mooncake-Store 运行时检测 SSE4.2 `crc32` 指令加速 CRC-32C（`Mooncake@93bc4ff541`, PR #4821）**：
   - Offset Allocator 与 HA Batch OpLog 快照默认对每条记录计算多项式为 `0x82F63B78` 的 CRC-32C 校验和以剔除撕裂写（Torn Writes）。新实现在 x86-64 运行时通过 CPUID 检测 SSE4.2 并调用硬件 `_mm_crc32_u64 / _mm_crc32_u8`（即使二进制 Wheel 编译时未加 `-msse4.2` 也能生效），其他架构则从逐字节查表升级为 **Slice-by-8（8 张 256 项编译期常量表 `kTables`）**，且校验值与旧版完全兼容。

### 2.4 NIXL & Dynamo：EP 低延迟屏障等待本端 Put 完成、Libfabric EFA 强制禁用非请求写接收与 KV-Router 零 Flatten 解析 / OnceLock CRTC 叶子（PR #2442, #2494, #17238, #17259）
1. **NIXL EP 低延迟 Barrier 等待本端 `nixlPut` 完成（`nixl@48d531f9e6`, PR #2442）**：
   - **竞态根因**：`examples/device/ep/csrc/kernels/nixl_ep_ll.cu` 中的 `barrier()` 向对端发送 `sync_count[dst]` 后，原本只等待对端计数到达便立即返回，而未等待本端异步 `nixlPut` 读取完本地源缓冲区。在 GB200 NVLink + CPU Proxy 模式下，下一次 Barrier 会提前将同一本地内存字递减为 `-2`，导致对端收到 `-2`（期望 `-1`）并在 5/5 次运行中全部超时误杀健康 Rank。
   - **修复**：向 `nixlPut` 传入 `nixlGpuXferStatusH xfer_status`，在对端值到达后继续轮询 `nixlGpuGetXferStatus` 直至本端发送完成，GB200 CPU Proxy 弹性扩缩容测试从 **0/5 通过提升至 5/5 全部通过**（8x H100 36/36 通过）。
2. **NIXL Libfabric 跨头文件版本强制禁用 EFA Unsolicited Write Recv（`nixl@410b8dc22e`, PR #2494）**：
   - AWS EFA 在 Libfabric $\ge 2.0$ 默认开启 `unsolicited write recv`，高负载下会静默丢失远端 RDMA Write 完成通知。由于 NIXL PyPI Wheel 基于旧版 Libfabric 1.21 头文件编译（缺少 2.5 才引入的宏定义），导致运行在系统级 Libfabric $\ge 2.5$ 时未调用 `fi_setopt()`。
   - 修复后在 `libfabric_rail.cpp` 中硬编码 ABI 稳定常量 `NIXL_OPT_EFA_USE_UNSOLICITED_WRITE_RECV = -FI_PROV_SPECIFIC_EFA + 8`（并在有头文件时通过 `static_assert` 校验），确保任何编译产物在运行时均关闭该隐患。
3. **Dynamo KV-Router：移除 `serde(flatten)` 与 CRTC 叶子节点 `OnceLock<ArcSwap>` 惰性分配（`dynamo@2d17836993` #17238, `dynamo@d1d61fc619` #17259）**：
   - 在 `lib/kv-router/src/services/selection/types.rs` 中用手写 `Visitor` 取代 `#[serde(flatten)]` 和 `serde_path_to_error`，消除每次路由请求将整个 JSON 反序列化到中间 `serde_json::Value` 映射的分配开销，并严格拒绝 `"is_eagle": "false"` 等字符串伪布尔值；
   - 在压缩并发基数树（CRTC，`children.rs`）中，将 `NodeChildren.state` 从直接持有 `ArcSwap<ChildrenState>` 改为 `OnceLock<ArcSwap<ChildrenState>>`（无子节点时指向静态 `NO_CHILDREN`）。因为每次析构或交换 `ArcSwap` 都必须遍历所有线程的 `arc_swap` Debt 列表，而 KV 缓存淘汰释放的绝大多数节点都是从未有过子节点的叶子（Leaf），惰性初始化彻底消除了叶子节点回收时的跨线程 Debt 扫描开销。

---

## 3. 创新架构横向对比（Fireworks / Together / Cerebras / Groq）

| 维度 | FA4 + verl + Mooncake + NIXL + Dynamo（Phase 89） | Fireworks AI / Together AI | Cerebras WSE-3 / Groq LPU |
| :--- | :--- | :--- | :--- |
| **超长上下文训练显存墙** | FA4 SM100 64-Head Sparse MLA `recompute-P` 以 `4.5%` 时间代价削减 **66.75% 峰值显存** | 长上下文微调服务依赖激活重计算与 FP8 KV 压缩降低单租户 GPU 占用 | 片上 SRAM 容量固定，超长序列必须依赖主机或跨晶圆激活流式卸载 |
| **RLHF 权重同步与 KV 路由** | verl 128 节点 GB200 Hybrid-LB 门控 + Mooncake Tiled COO 重分片 + NIXL EP 双端屏障确认 | 异步 RL 与分离式 P/D 集群高度依赖无损 RDMA 完成语义与低开销前缀树路由 | 静态编译图切换权重的开销较大，更适合固定权重的确定性在线推理 |

---

## 4. 架构师练习与可证伪问题

1. **单边通信屏障不变量练习**：为什么在 GPU Direct 模式下 `nixlPut` 未等待本地发送完成极少触发故障，而在开启 Kineto Profiling 的 GB200 CPU Proxy 模式下 100% 触发 `expected -1, actual -2`？请画出本地源字 `sync_count[dst]` 在相邻两次 `barrier()` 之间的读写时序图。
2. **可证伪后续问题**：在 Dynamo KV-Router 的 CRTC 基数树中，当节点扇出（Fanout）超过 4 个子节点时会从紧凑快照升级为 `DashMap`；在具有高淘汰率的 `block_size = 16` 多租户工作负载中，叶子节点占比通常超过 75%，启用 `OnceLock<ArcSwap>` 后后台淘汰线程的 CPU 占用率下降幅度是否与叶子节点释放速率呈线性正相关？
