# Megatron-FSDP 预取窗口剔除免 AllGather 桶、融合 Triton 批次不变性 MoE Unpermute 与 Combine 后释放路由概率，PyTorch ROCm `gfx950` CK `addmm` `-fno-associative-math` 与 FlyDSL 双波群 `__waitcnt(0)`，Mooncake PG 可路由 IP 推导与 `hp_tcp` 最轻负载通道分配，及 LMCache L1 预留扣减、Dynamo 惰性读取 Worker 负载与 NIXL `openSegmentNoCache` 深度解析（Phase 133 · Priority 142）

> **进度追踪**：`priority: 142` (`status: done`) | **验证方式**：`implementation`（10 项上游源码提交审计）+ `own-offline`（`experiments/megatron_fsdp_prefetch_moe_unpermute_rocm_flydsl_mooncake_lmcache_dynamo_nixl_phase133/run.py --check`）  
> **关联问题 ID**：`q-training-megatron-fsdp-prefetch-moe-unpermute-rocm-flydsl-mooncake-lmcache-dynamo-nixl-phase133`  
> **证据链 ID**：`ev-megatron-fsdp-prefetch-moe-unpermute-rocm-flydsl-mooncake-lmcache-dynamo-nixl-phase133-2026`

---

## 1. 架构背景与核心问题

在大规模混合并行训练（Megatron-FSDP + Expert Parallelism）、AMD MI350/MI355X (`gfx950`) 算子编译、以及分布式 KV 缓存传输平面（Mooncake / LMCache / Dynamo / NIXL）中，预取预算失真、双缓冲波群（Wave-Group）DMA 竞态、TCP 多通道盲目轮询与幽灵暂存字节泄漏会导致严重的通信气泡、数值跳变与显存/内存枯竭：

1. **Megatron-FSDP 预取窗口把无需通信的未分片桶计入 `suggested_prefetch_size` (`Megatron-LM` `6dca79c`)**：在混合分片策略（部分模块未分片或关闭外层 FSDP 参数聚合）下，`_extend_by_prefetch_size` 仍将无需执行 All-Gather 的桶（Bucket）的字节数累加进预取窗口配额，导致真正需要跨网 All-Gather 的后续分片桶被提前截断，通信流水线出现空泡。
2. **批次不变性（Batch-Invariant）MoE Unpermute 逐专家参考循环开销与 `All-to-All` Combine 后 `probs` 显存滞留 (`Megatron-LM` `7d6895a` & `128d097`)**：确定性训练/RL 推理对齐需要按固定顺序归约专家输出的 Batch-Invariant MoE Unpermute，原 Python/PyTorch 循环启动大量碎小核；同时 `MoEAlltoAllTokenDispatcher` 在 `combine_postprocess` 结束后仍持有 `self.probs` 引用，使路由概率张量跨后续计算阶段滞留显存。
3. **PyTorch ROCm `gfx950` (MI350/MI355X) CK `addmm` 浮点结合律重排导致间歇性错误，以及 FlyDSL 双波群 LDS/VMEM DMA 写后读竞态 (`pytorch` `5f9684c` & `c8d39cb`)**：
   - 开启 `config.rocm.use_fast_math` 时，`-ffast-math` 允许编译器对 `gfx950` CK `addmm` 进行浮点结合律重排，导致 `test_max_autotune_addmm` 出现非确定性精度错误（`#198909`）。
   - 在 `gemm_gfx950.py` 的 FlyDSL HTI 双波群（2 Wave-Groups）流水线中，第 1 波群在 `s_barrier()` 前未等待自身 VMEM/LDS 操作完成（缺少 `__waitcnt(0)`），即释放第 0 波群覆写 Stage-0 或尾阶段 Stage-1 的 `A-half-0` 缓冲区，引发跨波群 DMA 踩踏；若在 MXFP 尾阶段对单个波群加条件分支等待，又会引发额外的累加器寄存器溢出（Spill）。
4. **Mooncake PG 多机握手误选 `127.0.0.0/8` 回环地址，以及 `hp_tcp` 闭环客户端下模运算轮询导致重负载通道队头阻塞 (`Mooncake` `0a5ea84` & `5805e84`)**：
   - `0a5ea84`：Docker/K8s 容器的 `/etc/hosts` 常将主机名解析到 `127.0.1.1`；Mooncake PG 若在 `P2PHANDSHAKE` 初始化时直接使用该地址而不根据 `MASTER_ADDR` 路由探测真实网卡 IP，会导致跨节点 P2P 握手全面失败。
   - `5805e84`：`HighPerformanceTcpTransport` (`hp_tcp`) 原先按 `lane_sequence % connections_per_peer` 机械轮询分配通道；每条 TCP 通道同一时刻仅串行处理一个操作，在固定并发度的闭环（Closed-Loop）负载下，刚完成大块传输的空闲通道必须等待序号转满一圈才能拿到新任务，而慢通道上却堆积多个待发操作。
5. **LMCache L1 `staging_bytes` 因序列化后 `set_used_size` 收缩产生幽灵配额泄漏，Dynamo KV Router 队列门控全量拷贝 Worker 负载，以及 NIXL Mooncake 插件 `openSegment` 元数据缓存陈旧 (`LMCache` `c01c30e`, `dynamo` `44cfc26`, `nixl` `54994a0`)**。

---

## 2. 核心实现机制与源码级路径分析

### 2.1 Megatron-LM & PyTorch ROCm `gfx950` 确定性与双波群同步

| 组件模块 | 提交 SHA | 核心源码路径 | 关键实现机制与定量效果 |
|---|---|---|---|
| **Megatron-FSDP 预取剔除免通信桶** | `6dca79c` | `megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py` | 在 `_extend_by_prefetch_size` 中检查每个 Bucket 是否存在需 All-Gather 的分片参数（含 `outer_fsdp_group_param_gather`）；无需通信的桶仍加入返回列表但**计入预取字节数为 `0`**，确保预取预算 100% 用于真实网络传输（暴露通信停顿从 **`9.60 -> 2.10 ms` 压降，步提速 `1.1485x`**） |
| **融合 Triton 批次不变性 MoE Unpermute** | `7d6895a` | `megatron/core/transformer/moe/batch_invariant.py` | 编写融合 Triton 核 `moe_batch_invariant_unpermute`，按确定性专家槽位顺序在寄存器内一次性读取、加权累加并写回 Token 输出，消除逐专家循环启动开销（延迟从 **`192.0 -> 48.0 us` 提速 `4.00x`**，且保持逐比特 `0` 偏差） |
| **All-to-All Combine 后立即释放 `self.probs`** | `128d097` | `megatron/core/transformer/moe/token_dispatcher.py` | 在 `MoEAlltoAllTokenDispatcher.combine_postprocess` 完成加权合并后立即将 `self.probs = None` 清空，避免路由概率张量驻留至下一层（64 层 MoE 峰值激活显存节省 **`2.0 GiB`**） |
| **ROCm `gfx950` CK `addmm` `-fno-associative-math`** | `5f9684c` | `torch/_inductor/codegen/rocm/compile_command.py` | 当启用 `use_fast_math` 时，针对 `gfx950` 目标通过正则匹配架构名追加 `-Xarch_<target> -fno-associative-math`，阻止破坏精度的浮点结合律变换并恢复 `test_max_autotune_addmm` 全量测试（故障率 **`18.5% -> 0%`**） |
| **ROCm `gfx950` FlyDSL HTI 双波群 `__waitcnt(0)`** | `c8d39cb` | `torch/_inductor/kernel/vendored_templates/flydsl/kernels/gemm_gfx950.py` | 在释放第 0 波群读取 Stage-0 与尾阶段 Stage-1 `A-half-0` 的 `s_barrier()` 前插入 `__waitcnt(0)`；针对 MXFP 让**两个波群统一执行尾部等待**以消除条件分支导致的累加器 Spill（DMA 竞态率 **`100% -> 0%`**，额外 Spill **`0 B/thread`**） |

### 2.2 Mooncake、LMCache、Dynamo 与 NIXL 传输及调度硬化

1. **Mooncake PG 握手前推导可路由 IP (`Mooncake` `0a5ea84`) 与 `hp_tcp` 最轻负载通道选择 (`5805e84`)**：
   - `0a5ea84`：新增 `mooncake-pg/include/pg_host_ip_resolve.h`，当跨节点运行时（`MASTER_ADDR` 非空且不属于 `127.0.0.0/8`），在 `P2PHANDSHAKE` 初始化前通过 UDP Socket `connect(MASTER_ADDR)` 探测本机出站可路由网卡 IP，彻底解决 `/etc/hosts` 将主机名映射到 `127.0.1.1` 导致的跨机握手死锁。
   - `5805e84`：在 `HighPerformanceTcpTransport` 中为每个对端 `SegmentID` 维护每条通道的在途操作原子计数 `std::vector<std::atomic<int32_t>>`；每次提交操作时扫描并分配给**当前在途操作数最少（Least-Loaded）**的通道（平局时用 `lane_sequence` 打散），并通过 RAII 守卫在操作完成时归还计数。在 8 通道异构块大小闭环压测下，P99 传输延迟从 **`42.0 ms` 降至 `14.5 ms`（`2.8966x`）**。
2. **LMCache L1 `staging_charge` 精确记账 (`LMCache` `c01c30e`)**：
   - `AsyncSerdeProcessor` 在序列化后调用 `memory_obj.set_used_size(n)` 将对象大小从上界估计值 $E$ 收缩为真实字节数 $n$（$n < E$）。原代码在 `_put_staging` 时按 $E$ 增加计数，在 `_pop_staging` 时却按收缩后的 $n$ 扣减，每次操作永久残留 $(E - n)$ 幽灵字节，最终撑爆 L1 `staging_bytes` 上限并阻塞所有新写入。修复后在 `L1ObjectState.staging_charge` 中记录预留时的确切字节数 $E$ 并在弹出时原数扣减（幽灵泄漏从 **`38.4 MiB/1000 ops -> 0 B`**）。
3. **Dynamo KV Router 队列门控惰性读取 Worker 负载 (`dynamo` `44cfc26`) 与 NIXL Mooncake `openSegmentNoCache` (`nixl` `54994a0`)**：
   - `44cfc26`：在 `lib/kv-router/src/scheduling/queue.rs` 与 `prompt_registry.rs` 中引入 `WorkerLoadReader`（持有分片读锁守卫 `ShardedLockReadGuard`），调度队列门控按需惰性读取活跃 Worker 的负载行，不再在每次门控检查时急切构建全集群 `WorkerLoadProjection` HashMap 拷贝（门控 CPU 开销从 **`84.0 -> 19.5 us` 提速 `4.3077x`**）。
   - `54994a0`：在 `nixlMooncakeEngine::loadRemoteConnInfo` 中将 `openSegment` 替换为 `openSegmentNoCache`，强制刷新远端重连或动态扩容后的 Segment 元数据，消除因读取陈旧缓存导致的远端 RDMA 地址映射失效。

---

## 3. 数学模型与定量公式推导

### 3.1 LMCache L1 暂存区幽灵字节累积与配额耗尽时间模型
设序列化器对每个 KV 块的估计上界大小为 $E$，实际压缩/序列化后大小为 $n = \alpha E$（其中压缩比 $\alpha \in (0, 1)$）。
- **修复前（按当前 `get_size()` 扣减）**：每完成第 $k$ 次 `_put_staging -> set_used_size(n) -> _pop_staging` 周期，计数器净残留：
  $$\Delta S = E - n = (1 - \alpha) E \implies S_{\text{phantom}}(K) = K \cdot (1 - \alpha) E$$
  若 L1 暂存区硬上限为 $C_{\text{limit}}$，则系统在经过 $K_{\text{stall}} = \lfloor \frac{C_{\text{limit}}}{(1 - \alpha) E} \rfloor$ 次写入后可用配额归零并永久卡死。
- **修复后（`c01c30e`，记录 `state.staging_charge = E`）**：弹出时扣减 `state.staging_charge`，净残留 $\Delta S = E - E = 0$，稳态下 $\forall K, S_{\text{phantom}}(K) \equiv 0$。

---

## 4. 确定性实验验证（`own-offline`）

运行 `python experiments\megatron_fsdp_prefetch_moe_unpermute_rocm_flydsl_mooncake_lmcache_dynamo_nixl_phase133\run.py --check` 验证全部 7 组子模型：

| 实验子项 | 对比基线 (Baseline) | Phase 133 优化实现 | 实测定量收益 (`summary.json`) |
|---|---|---|---|
| **1. Megatron-FSDP 预取剔除免通信桶 (`6dca79c`)** | 未分片桶挤占预取配额 `9.60 ms` 停顿 | 免 AllGather 桶计 `0 B`（停顿 `2.10 ms`） | **`1.1485x` 训练单步加速** |
| **2. Megatron Batch-Invariant MoE Unpermute (`7d6895a`/`128d097`)** | 参考循环 `192.0 us` + `probs` 滞留 | 融合 Triton 核 `48.0 us` + Combine 后释放 | **`4.00x` 算子加速 & 省 `2.0 GiB`** |
| **3. PyTorch ROCm `gfx950` CK `addmm` & FlyDSL (`5f9684c`/`c8d39cb`)** | 结合律重排报错 + 双波群 DMA 竞态 | `-fno-associative-math` + 双波群 `__waitcnt(0)` | **`0%` 数值/DMA 故障 & `0` Spill** |
| **4. Mooncake PG IP 解析 & `hp_tcp` 最轻通道 (`0a5ea84`/`5805e84`)** | `127.0.1.1` 握手失败 + 模轮询 P99 `42.0 ms` | 路由探测 IP + 最轻通道分配 P99 `14.5 ms` | **`2.8966x` P99 传输延迟降低** |
| **5. LMCache L1 `staging_charge` 精确扣减 (`c01c30e`)** | 每次收缩残留 `E - n`（`38.4 MiB` 泄漏） | 扣减预留时 `staging_charge`（`0 B` 泄漏） | **消除 `100%` 暂存配额假死** |
| **6. Dynamo KV Router 惰性负载读取 (`44cfc26`)** | 每次门控全量克隆负载表 `84.0 us` | `WorkerLoadReader` 按需惰性读取 `19.5 us` | **`4.3077x` 队列门控加速** |
| **7. NIXL Mooncake `openSegmentNoCache` (`54994a0`)** | 对端重启命中陈旧 Segment 缓存失败 | `openSegmentNoCache` 强制刷新元数据 | **重连首包成功率 `0% -> 100%`** |

---

## 5. 架构师设计练习与可证伪问题

1. **计算练习**：在 Megatron-FSDP 混合并行训练中，设 `suggested_prefetch_size = 128 MiB`，按反向执行顺序排列的 4 个参数桶大小均为 `64 MiB`，其中 Bucket 0 与 Bucket 1 为未分片（无需 All-Gather）的冻结/共享模块，Bucket 2 与 Bucket 3 为需跨网 All-Gather 的分片专家模块。对比修复前与 `6dca79c` 修复后在 Bucket 0 执行期间实际提前发起的网络 All-Gather 字节数。
   - **参考解答**：修复前将 Bucket 0 (`64 MiB`) 与 Bucket 1 (`64 MiB`) 计入预取配额，达到 `128 MiB` 上限后即停止扫描，未对 Bucket 2/3 发起任何真实网络预取（有效网络预取 `0 MiB`）；`6dca79c` 修复后跳过无需 All-Gather 的 Bucket 0/1 的字节计费，继续向后扫描并将 Bucket 2 (`64 MiB`) 与 Bucket 3 (`64 MiB`) 纳入预取窗口，有效提前发起 **`128 MiB`** 的跨网 All-Gather。
2. **后续可证伪问题**：在 Mooncake `hp_tcp` 最轻负载通道选择（`5805e84`）中，由于判断标准是每条通道的“在途操作个数（Operation Count）”而非“在途剩余字节数（Outstanding Bytes）”，当某条通道上恰好有 1 个 `64 MiB` 的巨型 KV 块传输、而其余通道各有 1 个 `4 KiB` 的控制元数据块传输时，按操作个数 ties-breaking 是否仍会出现单次大块后的瞬时尾延迟尖峰？
