# 架构师练习与评分依据

先独立作答，再读提示。以下是原创合成问题，不是设备 benchmark。

## 1. 显存 payload

一个传统 GQA decoder：24 层、4 KV heads、head dimension 128、context 8192、batch 2、KV BF16。只算 K/V payload，不算权重/allocator。结果多少？

答案：`2 × 24 × 4 × 128 × 8192 × 2 × 2 = 805306368 bytes = 0.75 GiB`。如果权重变 INT4 而 KV 不变，KV 结果仍为 0.75 GiB；batch1/BF16 或 batch2/FP8 才为 0.375 GiB。满分需要区分 logical payload、token block 取整和整体进程峰值；不能称模型一定能部署。

## 2. Roofline

合成 kernel 做 `1e12 FLOPs`，从指定内存层移动 `1e10 bytes`；同精度计算峰值 `1e14 FLOP/s`，该层带宽 `1e12 bytes/s`。求 arithmetic intensity、ridge point、性能上界和时间下界。

答案：`100 FLOP/byte`、`100 FLOP/byte`、`1e14 FLOP/s`、`0.01 s`。刚好在 ridge point；0.01 s 是理想下界，不是预计或测得耗时。加入同步/不规则访存通常不会缩短这个理想下界。

## 3. Ring AllReduce

8 ranks，每 rank 输入 1 GiB，理想 Ring 的每 rank 发送 payload 多少？是否能用标称“双向聚合 200 GB/s”直接估算时间？

答案：`2 × (8-1)/8 × 1 GiB = 1.75 GiB`。不能直接拿双向聚合做单向模型分母；还要考虑 step latency、实际方向、并发渠道、协议/拓扑/拥塞和计算重叠。此处是解析通信量，不是 NCCL trace 测量。

## 4. Local memory 与 occupancy

为了提高 occupancy 强行减少寄存器，性能可能下降吗？

答案：可能。寄存器压力改变可能引入 spill 到 device/global backing；增加访存未必被更多 active warps 抵消。要用寄存器、spill、memory throughput、stall 与端到端计时共同验证，不能单看 occupancy。

## 5. 架构选型作业

需求：低并发、长输出的交互式推理；质量不可下降、严格 TPOT。另一个需求是高并发长输入的批量 summarization。为何不一定使用同一 batch/并行布局？

评分看工作负载分解、prefill/decode、权重/KV/activation 账本、排队、通信、尾延迟和质量。不能凭“decode 永远带宽瓶颈”作答；实际模型、复用和并行开销可以改变瓶颈。

## 6. 多节点训练方案

为一个合成训练任务设计 TP/DP/PP/FSDP 组合及 checkpoint。提交三张图：张量/状态流、物理/逻辑拓扑、失败/恢复。给出通信与显存预算、基线、验证命令的计划和退出条件。

评分不以 GPU 数量大小决定。没有固定 workload、同步路径、存储/恢复分母和预算的方案不通过；实测另需环境授权。

## 7. 质疑“最高 2x”

阅读 [UEC 1.1 条目](../trends/uec-1-1.md)。网络内归约减少 payload 是否必然使应用训练快两倍？

答案：不必然。作者通信路径示例不代表整体训练时间；收敛到相同质量、可用 switch/NIC、数据类型、collective、通信占比和故障恢复都影响收益。满分需要一个可证伪的对照实验，而不是另引用一个宣传数字。

## 8. Transport 四层与完成四阶段

同机 P2P 可用就必定走 P2P 吗？GDR 为真就没有 host buffer 吗？看到 TCP
连接就意味着 IB 数据面没有工作吗？为什么 NET request 完成不能代替应用
CUDA stream 完成？

评分看 [固定实现](../components/nccl-topology-transport.md) 的条件：
同机 NET preference、LL host-buffer 例外、IB metadata 控制连接，
以及 request 完成、GPU 可见、GPU 消费、应用 stream 完成的区别。
只背 transport 名称、未检查 canConnect/setup/runtime 错误者不通过。

## 9. Rank 顺序与共享 NIC

8 ranks，A/B 节点各 4 ranks，每 rank 1 GiB；共享单方向 NIC payload
上限 50 GB/s。分别复算节点分组和节点交替 Ring 的每条 edge、每节点
发送量与时间下界。分母能换成双向聚合带宽吗？全部 ranks 同节点时如何解释？

答案见 [原创 Ring cut](../../experiments/ring_cut/README.md)：每 edge 1.75 GiB，
两例每节点 1.75/7 GiB，下界 0.03758096384/0.15032385536 s。
4 倍只是该共享资源下界比，不能称实际加速比；全机内网络下界 0 不表示
collective 时间 0。满分须保留 closed-loop、单方向、GB/GiB、独立 NIC/
多 channel/其他算法的反例及质量/SLO 缺口。用户掌握仍未评估。

## 10. 多 Channel、Multi-Rail 与机内 `PATH_PHB` 反证

8 ranks（A/B 各 4 GPUs），`S = 1 GiB`，`C = 4` Channels，单张 NIC 单向
`50 GB/s`。为什么 4 张 Rail NIC 对齐（`crossNic = 0`）能把 NIC 下界从
`0.03758096384 s` 降到 `0.00939524096 s`，但一旦机内 `ringIntra` 某条
边跨 `PATH_PHB`（`32 GB/s`）且被 4 个 Channel 叠加穿过，联合下界反而变成
`0.058720256 s`？

答案见 [NCCL 图搜索与 Channel 映射](../components/nccl-graph-channel-mapping.md)
与 [原创 Rail channel cut](../../experiments/rail_channel_cut/README.md)。
满分需指出：(1) 每 Channel 每条有向边发送量为 `0.4375 GiB`；(2) `crossNic=1`
偏斜会让最忙单张 NIC 重回 `0.875 GiB`（`18.79 ms`）；(3) 机内瓶颈边叠加 4 个
Channel 后的 `1.75 GiB / 32 GB/s = 58.72 ms` 解释了 `ncclTopoCompute`
优先按 `speedArray` 搜索高 `bwIntra` 的机理。

## 11. GPU HBM 推理栈 vs 片上 SRAM / 晶圆级芯片（Fireworks / Together / Groq / Cerebras）

阅读 [AI Infra 创新公司深挖](../components/ai-infra-innovators.md)：
(1) 按 `groqflow` 的 `calculate_num_chips`（每芯片 `50M` 参数并向上取 2 的幂次），
8B 参数模型需要多少张 TSP 芯片串联？为什么单芯片 `220 MiB` SRAM 带来极低确定性尾延迟，
却在超长上下文或多模型混部时面临容量墙？
(2) 对比 Fireworks SM100 M3 `KV-outer` 块稀疏 Attention 与 Together Sequoia
硬件感知推测解码树，说明二者分别在什么序列长度或 Batch 区间会失去收益？

评分看是否明确区分 `fio1x` 与 `fbwAct`、Sequoia 验证延迟的 Roofline 拐点、
Groq PCIe replay 唯一外部抖动源，以及 Cerebras WSE 单 PE `48 KB` SRAM 与
主机边缘 `H2D/D2H`（`<= 16` 通道）的边界。用户掌握状态仍为 `not-assessed`。

## 12. NCCL IB Port-Failover：选择性重传与接收端有符号事件守恒

阅读 [NCCL IB Port-Failover 与异步端口恢复不变量](../components/nccl-port-recovery-invariants.md)
与 [IB failover invariants 算例](../../experiments/ib_failover_invariants/README.md)：
设双端口通信域 `ndevs = 2, nqps = 4, max_requests = 64, S = 1 MiB`（每 QP 负责 `256 KiB`），
接收端初始登记 `events = [2, 2]`。当 Dev 0 报 `IBV_WC_RETRY_EXC_ERR` 时，
发送端经存活 Dev 1 的 `probingQp` 发起 `128 B` `IBV_WR_RDMA_READ` 读取对端
`cmplsRecords[slot].completions`，发现 `qpIndex = 0` 已送达而 `qpIndex = 2` 未送达。
求：(1) 启用 Resiliency 后发送/接收 CQ 与接收 RQ 的容量；(2) 选择性重传相比盲目重传节省的字节数；
(3) 接收端收到 Dev 1 替代 QP 重传的 `qpIndex = 2` 分片后，`events[]` 数组的值是多少？为什么朴素无符号完成判据会死锁？

答案：(1) Send CQ = `64 × 4 × 2 = 512`，Recv CQ = `64 × 2 × 4 × 2 = 1024`，Recv RQ = `64 × 4 = 256`；
(2) 仅重传 `262144 B`（`256 KiB`），相比盲目重传 `1 MiB` 节省 `786432 B`（`75%`）；
(3) 接收端 Dev 0 只收到 1 个完成、Dev 1 收到 3 个完成，故 `events = [1, -1]`。朴素判据要求所有元素为 `0` 会永久等待 Dev 0，而 NCCL 有符号判据 `|negativeEventsSum| == remainingEventsSum`（`|-1| == 1`）正确完成。还要指出 `nFailedDevices > 1` 或重传期间二次故障属于不可恢复边界。用户掌握状态仍为 `not-assessed`。

## 13. 张量 Layout/Dtype 到 CuTe AOT C-ABI 与 SM100 `tcgen05` 的分派账本

阅读 [张量 Layout/Dtype 到编译器与 SM100 Kernel 的分派链路](../components/kernel-dispatch-layout.md)
与 [Innovator arch tradeoffs 算例](../../experiments/innovator_arch_tradeoffs/README.md)：
(1) 为什么在把 `[num_pages, Hkv, page_size, 128]` 的 Paged KV Cache 喂给 CuTe AOT Kernel 前执行 `.permute(0, 2, 1, 3)` 既不需要 HBM 拷贝，又能保持 `leading_dim = 3`（步长为 1）？
(2) 在 `Tq = 4096, Hq = 16, Hkv = 1, topK = 16, D = 128, BF16` 且全部命中 `16` 个唯一 KV 块（`M = 4096`）时，`fio1x`、`fioAct` 与无 KV 复用 `ioAlt` 分别是多少 GB？为什么 `512` 线程 CTA 要把寄存器按 `176 / 176 / 96 / 64` 非对称切分？

答案：(1) `.permute(0, 2, 1, 3)` 仅重排 `sizes/strides` 元数据，最后一维 `128` 仍保持连续 `stride = 1`，满足 TMA 16B 向量化对齐与 CuTe AOT ABI 省略末维步长的契约；
(2) `fio1x = 0.295 GB`，`fioAct = 0.546 GB`（因 `Q` 按 tile 重流式读取 `0.268 GB`），`ioAlt = 4.588 GB`（节省 `8.40x`）。SM100 将 $S/P$ 与 $O$ 累加器放入片上 `TMEM`，使 `mma` 与 `load` warps 仅需 `64` 个寄存器，从而把 `176` 个寄存器让给算术密集的 `softmax0/1` warpgroups，且总和 $128\times(176+176+96+64) = 65536$ 恰好占满单 SM 寄存器池。用户掌握状态仍为 `not-assessed`。

## 14. PyTorch DCP 跨 Rank 去重均衡、`BlockingAsyncStager` 与 `_OverlappingCpuLoader` 退回反例

阅读 [数据加载与分布式 Checkpoint/恢复契约](../components/storage-checkpoint.md)
与 [Checkpoint recovery 算例](../../experiments/checkpoint_recovery/README.md)：
64 卡（`DP = 8, TP = 4, PP = 2`）训练任务中，每个 `(tp, pp)` 副本组（共 8 组）包含 `35 GiB` 仅跨 TP/PP 分片但跨 DP 完全重复的张量，以及 `87.5 GiB/组` 额外按 `DP = 8` 切分的 ZeRO-1 优化器状态。
求：(1) 未去重落盘总字节、去重后唯一落盘总字节，以及 `save_to_lowest_rank = True` 与 `False`（均衡指派）下最慢 Rank 的写出字节；
(2) 为什么为了加速加载而预先把 `state_dict` 放在 Pinned CPU 内存中，在 PyTorch v2.6.0 `FileSystemReader(use_async_cpu_loader=True)` 下反而会比普通 Pageable CPU `state_dict` 更慢？

答案：(1) 未去重 `5,184 GiB`；去重后唯一字节 `8 × (35 + 87.5) = 980 GiB`（`5.29x` 缩减）。若 `save_to_lowest_rank = True`，每组 `dp_rank == 0` 独自承担全部 `35 GiB` 重复分片加自身 `10.9375 GiB` 唯一分片（单 Rank `45.9375 GiB`；若集中在全局 Rank 0 则高达 `297 GiB`），而均衡指派（`save_to_lowest_rank = False`）将每 Rank 写出严格均衡到 `15.3125 GiB`；
(2) `filesystem.py:259-265` 检查 `any(tensor.is_pinned() for tensor in state_dict.values())`，若为真则静默放弃 `_OverlappingCpuLoader`（磁盘读取与 `non_blocking` H2D 流水线重叠）并退回 `_SerialCpuLoader`（先全量串行读盘再拷贝）。用户掌握状态仍为 `not-assessed`。

## 15. vLLM V1 统一 `token_budget` 调度、100% Prefix Cache 末块重算与 Chunked Prefill / P/D 分离

阅读 [推理调度、KV Cache 生命周期与 SLO/Goodput 边界](../components/inference-slo.md)
与 [Inference SLO scheduler 算例](../../experiments/inference_slo_scheduler/README.md)：
(1) 当新到达的 `2,048-token` Prompt 在 `KVCacheManager.get_computed_blocks` 中 100% 命中已缓存的 `128` 个物理块（`block_size = 16`）时，为什么 `vllm/v1/core/sched/scheduler.py:265-275` 必须将 `num_computed_tokens` 减 1 并对第 128 个块执行 `unTouch` + `pop`？
(2) 在 `32` 条活跃 Decode 流（每流 `2,048` tokens，每步耗费 `32` 个 `token_budget`）运行的实例上，若到达一条未命中缓存的 `8,192-token` 长 Prompt，对比无分块内联 Prefill、`max_num_batched_tokens = 2,048` 的 Chunked Prefill 与 P/D 分离（跨实例 `400 Gbps` RDMA 传输 `2.50 GiB` KV Cache），说明三者对在途 Decode TPOT 与新请求 TTFT 的权衡。

答案：(1) Prefix Cache 中只保存历史块的 Key/Value 张量，不保存末位置的采样 logits；回退 1 个 token（`2047`）使 `num_new_tokens = 1`，并且因为 `2047` 使末块不再填满，必须将其 `ref_cnt` 减 1 归还空闲池并重新分配可写块以完成最后 1 个 token 的前向传播与槽位写回；
(2) 无分块内联 Prefill 令单步耗时飙升至 `441.91 ms`（严重违反 `50 ms` TPOT SLO）；Chunked Prefill 将单步 Prefill 截断为 `2,016` tokens，使同批次 Decode 最大 TPOT 压回 `45.01 ms`（分 5 步完成 Prefill，TTFT `183.75 ms`）；P/D 分离彻底消除 Prefill 对 Decode 实例的算力挤占（Decode 最大 TPOT `15.11 ms`），但需承担 `2.50 GiB / 50 GB/s = 50.00 ms` 的跨实例 KV 传输延迟（总 TTFT `221.66 ms`）。用户掌握状态仍为 `not-assessed`。

## 16. 6D 分布式训练：FSDP2 `Shard(dim != 0)` 瞬态缓冲、CP 因果首尾配对与 DeepEP v2 去重

阅读 [6D 分布式训练状态与通信联合账本](../components/training-parallelism.md)
与 [Training parallelism 算例](../../experiments/training_parallelism/README.md)：
(1) 在 `64` 卡（8 节点 × 8 GPU）训练 `70B` 级模型（`L = 80`，单层参数 `0.875 GiB` BF16，选择性重计算单层激活 `0.35 GiB` 在 `S = 8192`）时，为什么纯 `FSDP (DP = 64)` 的单卡峰值显存（`49.00 GiB`）和单步时间（`166.96 ms`）全面劣于 `FSDP (DP = 8) + TP = 8 + SP`（`35.44 GiB`，`55.97 ms`）？
(2) 若某参数张量第 0 维已被 `ColwiseParallel`（`Shard(0)`）切分，导致 FSDP2 在该参数上采用 `Shard(dim = 1)`，根据 `_fsdp_collectives.py:306-346` 的 `foreach_all_gather_copy_out`，其前向双缓冲瞬态峰值相比 `Shard(dim = 0)` 会增加多少？
(3) 在 `CP = 8` 因果 Ring Attention 与 `E = 64, EP = 8, top_k = 8` 的 MoE DeepEP v2 训练态 Dispatch 中，`_RoundRobinLoadBalancer` 与 `do_expand = False` 分别消除了多少比例的算力倾斜或跨节点流量？

答案：(1) 纯 `FSDP=64` 虽然把稳态模型状态压至 `17.50 GiB`，但未切分激活（`80 × 0.35 = 28.00 GiB`），且 FSDP AllGather/ReduceScatter 跨越全部 64 卡慢速跨机链路（暴露 `130.91 ms`）；`FSDP=8 + TP=8 + SP` 将激活按 `TP=8` 切分至 `3.50 GiB`，并把 FSDP 通信缩减至 8 个同列 Rail IB 节点、TP+SP 通信锁在机内 `400 GB/s` NVLink；
(2) `Shard(0)` 下 `all_gather_copy_in` 直接使用 `param_all_gather_outputs` 的底层存储作为接收缓冲（前向 2 层双缓冲 + 反向 2 层预取 + 1 层规约共 `4 × 0.875 = 3.50 GiB`）；而 `Shard(dim != 0)` 无法将分片直接拼入目标布局，必须在每层额外分配 `1` 份完整参数大小的 `all_gather_outputs` 临时缓冲再执行 `chunk + cat` 拷贝，使瞬态缓冲膨胀 `+50%` 至 `5.25 GiB`；
(3) 未开启首尾配对时，因果掩码令 `Rank 7` 承担 $15/64$ 的全局有效计算（均值的 `1.875x`，首尾比 `15:1`），`_RoundRobinLoadBalancer` 将第 $i$ 与 $2\text{CP}-1-i$ 块配对后各 Rank 严格均衡为 `1.000x`；DeepEP v2 训练态 Compact 布局按唯一目标 EP Rank 去重发送（期望唯一目标 Rank 数 $8\times(1-(7/8)^8) = 5.251$，精确无放回 `4.9998`），将跨节点 All-to-All 载荷从 `512.0 MiB` 压至 `319.99 MiB`（节省 `37.50%`）。用户掌握状态仍为 `not-assessed`。

## 17. Kubernetes Topology Manager、Slurm `topology/block` 反碎片与大规模弹性训练 Goodput

阅读 [AI 集群拓扑感知调度、设备隔离与弹性恢复契约](../components/cluster-orchestration.md)
与 [Cluster orchestration 算例](../../experiments/cluster_orchestration/README.md)：
(1) 在双 Socket 双 NUMA（NUMA 0 挂载 GPU 0..3、NUMA 1 挂载 GPU 4..7）的 8-GPU 节点上，为什么申请 `8` 张 GPU 的整机单容器训练 Pod 在 Kubelet `--topology-manager-policy=single-numa-node` 下会被拒绝（`TopologyAffinityError`），而在 `restricted` 下能正常准入？若申请 1 张 GPU (0b01) + 1 张跨 Socket RDMA NIC (0b10)，`restricted` 会放行吗？
(2) 在 `bblock_node_cnt = 18`、仅开放 `block_levels = [0, 2]`（即 18 节点与 72 节点两档）的 Slurm `topology/block` 集群中，提交 `20` 节点作业会锁定多少个节点的拓扑块？当提交 `16` 节点作业且 4 个基础块空闲节点数为 `[12, 16, 18, 10]` 时，`_choose_best_bblock` 为什么选 `bblock[1]`（16 空闲）而不选 `bblock[2]`（18 空闲）？
(3) `256` 节点（单节点 `MTBF = 360 h`）同步训练任务，Checkpoint 写出 `30 s`、读取 `45 s`，对比冷节点替换（`600 s`）+ `30 min` 间隔与 `torchelastic` 4 热备节点（`45 s` 重排）+ `15 min` 间隔的集群运行 Goodput 与单配置节点净产出。

答案：(1) 8-GPU 单容器的最佳对齐掩码为 `0b11`（`Count() = 2` 且 `Preferred = true`）；`single-numa-node` 在 `filterSingleNumaHints` 中强制要求 `NUMANodeAffinity.Count() == 1`，因此把 `0b11` 过滤掉并回退到 `defaultAffinity (Preferred = false)` 后拒绝 Pod；而 `restricted` 接受任意 `Preferred == true` 的最小掩码故放行 `0b11`。若 GPU (`0b01`) 与 NIC (`0b10`) 跨 NUMA，按位与结果为空集，落回 `Preferred = false`，在 `single-numa-node` 与 `restricted` 下均被拒绝；
(2) `20` 节点需 `ceil(20/18) = 2` 个基础块（`raw_level = 1`），因未启用 Level 1 被迫向上吸附至 Level 2（4 个基础块 = `72` 节点）；`16` 节点可装入单个基础块（`required_bblocks = 1`），`_choose_best_bblock` 采用 Best-Fit 选择最小满足块 `bblock[1]`（16 节点），将完整空闲的 `bblock[2]`（18 节点）留给后续满块作业；
(3) 256 节点聚合 `MTBF = 360 / 256 = 1.40625 h`（`5062.5 s`）。冷替换 + `30 min` 间隔下每次故障损失 `900 + 600 + 45 = 1545 s`，运行 Goodput 仅 `65.89%`；切换至 `4` 热备节点（`45 s` 恢复）+ `15 min` 间隔（接近 Young-Daly 最优 `551.14 s`）后，运行 Goodput 升至 `84.58%`，即便除以 `260` 台总配置节点分母，单配置节点净产出仍达 `83.28%`（净增 `+9.81 pts`）。用户掌握状态仍为 `not-assessed`。

## 18. 训练/推理容量经济学、SLO Goodput 悬崖与四家创新公司 I/O 成本模型

阅读 [容量、成本与吞吐经济学](../components/capacity-economics.md)
与 [Capacity economics 算例](../../experiments/capacity_economics/README.md)：
(1) 为什么在 `torchtitan/models/utils.py` 与 `components/metrics.py` 中，`671.1B` 总参数 MoE（路由专家 `664B`、`256` 选 `8`，稠密骨架 `4.8B`、非绑定词表 `0.9B`、路由器 `0.2B`、共享专家 `2.1B`）的单 Token 训练算力比 `70B` 稠密模型低一半以上？为什么开启 FP8 量化训练后 `MetricsProcessor.log` 将 `mfu` 强制置为 `None`？
(2) 在 `vllm/benchmarks/benchmark_serving.py` 的 Gamma 突发到达（`burstiness = 0.25`，即 $\text{CV}_a^2 = 4.0$）且单次未命中 Prefill 服务耗时 `291.67 ms`、`SLO_TTFT = 600 ms` 场景下，为什么盲目按 92% 利用率配置 Prefill 池会导致 `request_goodput` 跌至 `7.27%`？改用 Kingman 安全水位与 Prefill/Decode 分离（对比混部）以及将 Prefix Cache 命中率从 `12.5%` 提升至 `87.5%` 分别如何改变总 GPU 数与每百万达标输出 Token 成本？
(3) 结合 Fireworks `prefill_load_test.py`、Together/Sequoia `offload_engine.py`、Groq `groqmodel.py` 与 Cerebras `bandwidth-test/run.py`，分别说明：为什么 Fireworks 预热必须发 `cached_tokens + 1` 个 Token？Sequoia 投机树如何把 PCIe 双缓冲卸载吞吐放大 `5.4x–6.8x`？Groq 为什么在流水线模式下将编译器单级周期乘以 `num_chips_used`？Cerebras 为何必须在晶圆东西边缘划分 `46 KB/PE` 的硬件 FIFO 缓冲列并对参考时钟扣除 `px + py`？

答案：(1) MoE 每 Token 实际参与矩阵乘的参数仅为 `(4.8B - 0.9B) + 0.2B + 2.1B + 664B × 8 / 256 = 26.95B`（单 Token 训练 FLOPs 为 `204.68 GFLOPs`），低于 `70B` 稠密模型的 `68.95B`（`478.12 GFLOPs`），因此每百万训练 Token 成本从 `$0.9456` 降至 `$0.5355`；`gpu_peak_flops` 固定为 BF16 稠密峰值，若在 FP8 下继续用该分母会产生失真虚高，故 TorchTitan 置 `mfu = None`；
(2) `burstiness = 0.25` 使到达方差放大 `4` 倍，92% 利用率下期望排队等待高达 `4,083.33 ms`（期望 TTFT `4,375.0 ms`），仅 `7.27%` 的请求能满足 `600 ms` SLO；按安全水位 `34.58%` 与 P/D 分离定容（`31` Prefill + `5` Decode 实例 = `288` 卡）比混部（`392` 卡）节省 `104` 张 GPU，进一步将 Prefix Cache 命中率提至 `87.5%`（配 FP8）可把总卡数压至 `40` 张、每百万达标输出 Token 成本从 `$12.3698` 降至 `$1.7180`（下降 `86.1%`）；
(3) Fireworks 服务端把最后一个 Prompt Token 留给首个 Decode 槽位（仅缓存 `prompt[:-1]`），故需发 `cached_tokens + 1`；Sequoia 的 `OffloadEngine` 受限于 PCIe 权重传输时间，投机树验证在不增加 PCIe 权重搬运字节数的前提下每步接受 `5.4–6.8` 个 Token，等比例放大吞吐；Groq `stats_file` 中的 `total_cycles` 是单张芯片（1 个流水线级）周期，单请求走完 $N$ 张芯片须乘以 `num_chips_used`；Cerebras 为防止计算核心繁忙反压阻塞 100GbE 主机通道，需按 $\lceil W \cdot L_{\text{pe}} \cdot 4 / 46000 \rceil$ 配置 `46 KB/PE` FIFO 缓冲列（外加东西各 3 列 `memcpy` 系统列），并扣除全晶圆广播从 `PE(0,0)` 到 `(px, py)` 的 Manhattan 传播延迟 `px + py`。用户掌握状态仍为 `not-assessed`。

## 19. 高密液冷机柜热力学、Slurm 令牌桶浪涌控制与 DCGM MNNVL/IMEX 故障域

阅读 [机架功率密度、液冷/风冷边界与机柜级故障域](../components/rack-power-cooling.md)、[端到端 AI Infra 综合架构案例](../architectures/end-to-end-ai-infra-case.md) 与 [Rack power & cooling 算例](../../experiments/rack_power_cooling/README.md)：
(1) 单台 72x 1200W GPU + 36x 300W CPU + 18x 400W NVSwitch + `14.4 kW` 辅助负载的高密机柜（$\eta_{\text{PSU}} = 0.94$，$f_{\text{liq}} = 0.88$），在纯水进液 $T_{\text{in}} = 32^\circ\text{C}, \Delta T = 12^\circ\text{C}$ 与切换 PG25 防冻液且一次侧退化至 $T_{\text{in}} = 45^\circ\text{C}, \Delta T = 10^\circ\text{C}$ 两种工况下，所需冷却液流量（LPM）、GPU 结温 $T_j$（$R_{\theta} = 0.028\,\text{K/W}$）与 DCGM 时钟状态分别是什么？
(2) 在 Slurm `power_save.c` 中，唤醒 `64` 台 `10.2 kW` 节点时，设置 `ResumeRate = 12`、`power_save_interval = 15s` 如何通过令牌桶限制初始突发功率与爬坡斜率？なぜ `acct_gather_energy_rapl.c` 对 Xeon 服务器 DRAM 必须使用 $0.5^{16}\,\text{J}$ 单位，且丢失 `/dev/cpu/#/msr` 权限会发生什么？
(3) 在 NVIDIA DCGM `v4.6.1` `DcgmHealthWatch.cpp` 中，为什么 Blackwell 架构上 `DCGM_FI_DEV_NVLINK_RECOVERY_SUCCESSFUL_TOTAL > 0` 仅判为 `WARN`，而 `DCGM_FI_IMEX_DOMAIN_STATUS = DEGRADED` 会直接判为 `FAIL` 并影响整个机柜 MNNVL Clique？

答案：(1) 机柜 IT 负载 `118.80 kW`、墙插功率 `126.383 kW`（液冷 `111.217 kW` + 风冷 `15.166 kW / 1,798.6 CFM`）。纯水 $32^\circ\text{C}/\Delta T=12^\circ\text{C}$ 下需 `133.31 LPM`，GPU $T_j = 44.0 + 1200 \times 0.028 = 77.6^\circ\text{C}$（距 `83°C` `MAX_OP` 裕度 `+5.4°C`，`NOMINAL_NO_THERMAL_THROTTLE`）；切换 PG25 且 $T_{\text{in}}=45^\circ\text{C}/\Delta T=10^\circ\text{C}$ 后，因比热容下降与温差收窄，流量升至 `165.62 LPM`（`+24.2%`），且 $T_j = 55.0 + 33.6 = 88.6^\circ\text{C} > 87.0^\circ\text{C}$，触发 `DCGM_CLOCKS_EVENT_REASON_HW_THERMAL` 硬件频率腰斩；
(2) `power_save_rl_setup` 计算初始令牌数 `max_tokens = max(1, (12 × 15) // 60) = 3`、`refill_period = 60000 // 12 = 5000 ms`，使初始瞬态功率阶跃从不限流的 `652.8 kW` 压至 `30.6 kW`，后续斜率严格限制在 `122.4 kW/min`（总唤醒耗时 `305.0 s`）；Xeon 服务器 DRAM 固定采用 $0.5^{16} \approx 15.2588\,\mu\text{J}$ 而不跟随 `MSR_RAPL_POWER_UNIT` 的 Package 单位，且 `_get_MSR_fd` 打开 `/dev/cpu/#/msr` 失败会直接调用 `slurm_drain_node()` 将节点 Drain 下线；
(3) Blackwell 支持硬件链路层透明自愈，`RECOVERY_SUCCESSFUL > 0` 不破坏上层通信语义故仅报 `WARN`（避免误杀）；而 `IMEX` 守护进程与 Domain 维护机柜级多节点 NVLink 跨节点虚拟地址导入导出，一旦 `DEGRADED/DOWN` 会导致跨节点 P2P 与集合通信失效，必须隔离或重切分整个 MNNVL Clique。用户掌握状态仍为 `not-assessed`。

## 20. PD 分离多轨 RDMA KV 传输、DeepEP 双层去重/EPLB 与 SGLang 自适应 EAGLE-3 / NVFP4 Swizzle

阅读 [UEC 1.1 与 RoCEv2 多轨拥塞与喷洒路由对比](../components/uec-transport-congestion.md)、[Prefill/Decode 分离与分层 KV 缓存路由契约](../components/disaggregated-kv-routing.md)、[大规模 MoE 专家并行与 DeepEP 双层去重/EPLB](../components/moe-expert-parallel-alltoall.md) 及 [AI Infra 创新者架构深挖 v2](../components/ai-infra-innovators-v2.md)：
(1) 在 `S = 32,768`、单请求 `4.0 GiB` KV Cache 的 PD 分离系统中，为什么将 `page_size` 设为 `1` Token（`65,536` 个散落 IOV 描述符）会导致 `11.796 ms` 的 MMIO 门铃开销？当 4 条 `400GbE` RoCE 轨中有 1 条拥塞降速至 `25%`（`10.0 GB/s`）时，为什么 Mooncake `RailMonitor` 主动摘除该慢轨（仅用 3 条健康轨）反而将有效传输吞吐从 `32.0 GB/s` 提升至 `127.5 GB/s`（快 `3.98x`）？
(2) 在 DeepSeek-V3 规模（`64 GPU / 8 节点, 256 专家, topk = 8, H = 7168, B = 4096`）下，根据 DeepEP `token.cuh` 与 `hybrid_dispatch.cuh`：单 Token 的 FP8 (`with_sf=True, with_metadata=True, kWithMBarrier=True`) 精确缓冲区字节数是多少？为什么 Hybrid 2-Tier `ptx::deduplicate` 能将单卡跨节点 RDMA 发送量从 `80.51 MiB` (BF16 朴素全发) 压至 `21.84 MiB`（节省 `72.87%`）？
(3) 在 SGLang `adaptive_spec_params.py`、`eagle_disaggregation.py`、`fp4_utils.py` 与 `nvfp4_kv_cache.py` 中：为什么 `DEFAULT_ADAPTIVE_CONFIG` 在 `BS = 64` 时强制 `candidate_steps = [0]`？`pd_first_draft_proposal` 如何用 One-Hot $q(X)=1.0$ 实现零词表分布传输的无损拒绝采样？为什么微批次 `M = 8, K = 7168` 误用 `128x4` NVFP4 缩放布局会产生 `+1500%` Padding 膨胀，而 NVFP4 KV Cache 必须要求 `page_size % 4 == 0`？

答案：(1) 每个非连续 IOV 切片触发额外 WQE/MMIO 门铃开销（`65,536` 切片累计 `11.796 ms`，占传输时长 `62%`），聚合成 `64-Token` Block 可消除 `98.4%` 门铃开销；静态 4 轨轮询分片受最慢第 4 轨（`10 GB/s`）木桶效应同步阻塞（$4 \times 10 \times 0.8 = 32.0\text{ GB/s}$），摘除慢轨后 3 条健康轨满速并发达 $3 \times 42.5 = 127.5\text{ GB/s}$；
(2) `hidden = 7168 B`（32B 对齐），`num_scales = 56 -> 14 个 sf_pack_t = 56 B -> 32B 对齐为 64 B`，`metadata = 8×8 + 9×4 = 100 B -> 32B 对齐为 128 B`，三者之和 `7360 B` 已对齐 64B? 注意：`7168 + 64 + 64 (当 with_metadata=False) = 7296 B + 32B mbarrier = 7328 B`（含 `with_metadata=True` 时为 `7392 B`）；Hybrid 2-Tier 先按目标节点位掩码去重，使每个源 Token 对同一远端节点最多发 1 份 RDMA 拷贝（`4.699` 唯一远端节点而非 `7.0` 次重复发送），再叠加以 FP8 替代 BF16，将跨节点 RDMA 字节减少 `72.87%`；
(3) `BS = 64` 下验证 `64 × (K + 1)` 个 Token 已完全进入 Tensor Core Compute-bound 区，固定 `steps = 5` 在低接受率（`0.25`）下使 TPOT 恶化 `3.96x`（`47.626 ms` vs `12.017 ms`）；Decode 端令 $q(X)=1.0$ 时接受概率 $\min(1, p(X)/1) = p(X)$，拒绝时从 $\max(0, p-q)$ 重采样，数学上严格还原目标分布 $p$；`128x4` 布局将 `sf_rows = round_up(8, 128) = 128` 放大 `16` 倍（`+1500%`），导致等效位宽恶化至 `12.0 bits/elem`（改用 `8x4` 恢复 `4.5 bits/elem`）；`nvfp4_v_scale_swizzle_indices` 以连续 4 个 Token 为一个交织置换 Tile，故强制 `page_size % 4 == 0`。用户掌握状态仍为 `not-assessed`。

## 21. 超长上下文混合注意力（FlashMLA / Ring vs Ulysses CP / Mamba 检查点）、RL 权重重分片与机密 GPU (PPCIE)

阅读 [超长上下文混合注意力](../components/long-context-hybrid-attention.md)、[大规模推理强化学习引擎（RLVR / GRPO）](../components/rl-post-training-engine.md) 与 [机密计算 GPU（CC-On / PPCIE）](../components/confidential-gpu-tee.md)：
(1) 在 `deepseek-ai/FlashMLA` 中，为什么 Hopper SM90 单 CTA 处理 64 个 Query Heads 时，反量化 `512` 个 FP8 元素需要 `50.0` cycles，高于 Tensor Core MMA 的 `34.0` cycles？2-CTA Cluster DSM Crossover 如何利用 `st.async` 将反量化压至 `25.0` cycles？为什么在 `1M` Token、`CP = 8` 下，MLA (`h_kv = 1`) 使用纯 DeepSpeed-Ulysses 会产生每层 `7,616 MiB` 的 All-to-All 通信，而 Ring Attention 仅需 `1,008 MiB`？
(2) 在 `volcengine/verl` 与 `vllm/device_allocator/cumem.py` 中：为什么 `DeltaShardedCheckpointEngine._GatherQueue` 必须按条目数（`len(entries) >= batch_k`）而非本地累积字节数触发 Flush？为什么共置 RL 训练应当优先选择 `CuMemAllocator` 的 `Sleep Level 2`（`0.038 s`）而非 `Sleep Level 1`（`0.9118 s`）？
(3) 在 `NVIDIA/nvtrust` 中：当 Driver RIM 与 VBIOS RIM 在同一 `index` 均标记 `active` 时会发生什么？为什么 `OPAQUE_FIELD_ID_NVDEC0_STATUS == DISABLED` 时要豁免 `MSR 35`？为什么多卡 TP 推理在单卡 `CC-On` 模式下 TPOT 恶化 `2.366x`（`27.6029 ms`），而在 8-GPU + 4-NVSwitch `PPCIE` 模式下恢复至 `11.6811 ms`（`1.001x`）？

答案：(1) Hopper SM90 缺乏原生 FP8 到 BF16 直接指令，CUDA Core 串联 `e4m3->half->fp32->bf16` 与 scale 乘法需 `(1/64 + 1/64 + 1/16 + 1/256) × 512 = 50.0 cycles`；2-CTA Cluster 让两个各管 64 个 Q 头的 CTA 分别加载并反量化 `256` 个元素（`25.0 cycles`），再经 `st.async` 写入对方 Distributed Shared Memory，使算子回到 `34.0 cycles` Tensor Core 主导态；MLA 的 `h_kv = 1 < CP = 8`，纯 Ulysses 按头切分被迫在 8 个 Rank 上全量复制 KV 头（4 次 All-to-All 共 `7,616 MiB`），而 Ring CP 直接点对点流转未复制的 `576` 维潜变量（`1,008 MiB`）且与计算完全重叠；
(2) 在 FSDP/EP 混合并行中，各 Rank 参数变化的稀疏字节数不同，若按本地字节阈值触发集合 Gather 会导致不同 Rank 在不同步长发起 NCCL 通信而死锁，而参数条目数在所有 Rank 上严格一致；共置 RL 唤醒 vLLM 后紧接着就会用 Trainer 的新权重覆写 vLLM 权重，因此 `Sleep Level 1` 备份旧权重到 Pinned CPU 再拷回 GPU 的 `39.06 GiB` PCIe 流量纯属冗余，`Sleep Level 2` 直接 `unmap` 丢弃并在唤醒后通过 `0.0835 s` CUDA IPC 原地重填；
(3) 同一槽位在两份 RIM 中同时活跃会抛出 `InvalidMeasurementIndexError` 拒绝通过；`NVDEC0` 禁用时硬件不填充第 35 号度量槽，豁免 `MSR 35` 避免误杀合法 Hopper 配置；单卡 `CC-On` 禁用直接 NVLink P2P，迫使每步 `64 MiB` TP AllReduce 绕行主机 Bounce Buffer 做 CPU AES-GCM 加解密（有效带宽 `4.2 GB/s`），而 `PPCIE` 完成 8 GPU $\leftrightarrow$ 4 NVSwitch 双向 PDI 证明后开启硬件加密 NVLink（`372 GB/s`），将每步 AllReduce 从 `16.10 ms` 压回 `0.181 ms`。用户掌握状态仍为 `not-assessed`。

## 22. 共封装光学（CPO / ELSFP / OCS）、DeepSeek 3FS `hf3fs_usrbio` 与 Agentic RL 万级沙箱协同调度

阅读 [共封装光学（CPO）、线性直驱（LPO）与光路交换（OCS）](../components/optical-interconnect-cpo-ocs.md)、[DeepSeek 3FS (`hf3fs_usrbio`) 与 GPUDirect Storage 多级 KV/Checkpoint 管线](../components/storage-gds-3fs-kv-tiering.md) 与 [Agentic RL 万级 MicroVM 沙箱与 GPU KV Cache 协同调度](../components/agentic-sandbox-coscheduling.md)：
(1) 在十万卡（100,000-GPU）800G 3-Tier Clos 集群中，将传统 PAM4 DSP 可插拔光模块替换为 CPO + ELSFP 或 OCS + Leaf CPO 分别节省多少兆瓦（MW）功耗并将同步训练 Goodput 提升至多少？根据 `sonic-xcvrd/xcvrd/cpo/cpo_manager_task.py`，为什么严禁将同一 GPU 节点的多张网卡顺序插在同一颗 `OE` 或同一只 8 通道 `ELSFP` 的兄弟端口上？
(2) DeepSeek 3FS 的 `hf3fs_usrbio` 如何通过 `/dev/shm` 与 `<mount>/3fs-virt/iovs/` 符号链接绕过 Linux FUSE 的 `400K` IOPS 内核自旋锁墙？为什么 `LMCache` 当前的 `Hf3fsConnector`（`connector.cpp`）仍存在 `75%` 的 PCIe Root Complex 负载，而结合 GPUDirect RDMA / BAR1 P2P（需关闭 PCIe Bridge `PCI_EXT_CAP_ID_ACS` 重定向）能将 `100 GiB` KV 恢复 P99 从 POSIX 的 `5.536 s` 压降至 `1.265 s`？
(3) 在 `32,768` Token 上下文（`4.0 GiB` KV Cache）的 Agentic RL (`verl` `ToolAgentLoop` + Firecracker UFFD) 中，GDS 零拷贝卸载与 Prefix 重算相对 HBM 驻留的盈亏平衡工具耗时阈值 $t_{\text{BE,gds}}$ 和 $t_{\text{BE,recomp}}$ 分别是多少秒？为什么在 `500 VMs/s` UFFD 缺页风暴下，未限流的 GDS 卸载（`136.92 GiB·s`）反而比直接驻留 HBM（`120.0 GiB·s`）更慢？当 `uffd.copy()` 返回 `-EAGAIN` 时，Handler 必须做什么才能避免 Guest vCPU 永久死锁？

答案：(1) 十万卡 3-Tier Clos 共需 `450,000` 等效 800G 光学端口，传统 DSP 可插拔功耗 `7.92 MW`（`22.0 pJ/bit`，日均 `32.0` 次闪断，Goodput `70.31%`），CPO + ELSFP 降至 `2.16 MW`（省 `5.76 MW`，Goodput `91.32%`），OCS 去 Spine + Leaf CPO 降至 `0.72 MW`（省 `7.20 MW`，日均 `2.4` 次闪断，Goodput `96.93%`）；因为 SONiC `deinit_oe_sibling_pports()` 在单端口触发 `ModuleLowPwr` 时会同步关停同颗 `OE`（`1.6 Tbps`）或同只 8 端口 `ELSFP`（`6.4 Tbps`）下的全部兄弟端口，同节点多网卡若共用同一 `ELSFP` 会在单次激光器故障中同时断网，必须按 8-Rail 正交打散；
(2) `hf3fs_iovcreate` 在 `/dev/shm` 创建共享内存并在 `<mount>/3fs-virt/iovs/` 创建编码块大小与队列深度的符号链接，由本地 `fuse.hf3fs` 守护进程代管 IB `ibv_reg_mr`，后续 `hf3fs_submit_ios` 完全在用户态无锁共享环投递；`LMCache` 的 `connector.cpp` 在调用者 Tensor 与 `/dev/shm` `iov.base` 之间执行了一次 `std::memcpy`（消耗 `24%` DDR 带宽与 `75%` RC 负载，吞吐 `48.0 GB/s`），直接将 GPU BAR1 映射进 `hf3fs_iov` 并关闭 PCIe Switch ACS 重定向后，吞吐达 `90.0 GB/s`（RC 负载仅 `6.0%`，`100 GiB` KV 恢复 `1.265 s`）；
(3) 无争用 GDS 往返 Swap `0.160 s`，故 $t_{\text{BE,gds}} = (16.0 \times 0.160) / 4.0 = 2.56\text{ s}$；Prefill 重算耗时 `0.500 s`，故 $t_{\text{BE,recomp}} = (16.0 \times 0.500) / 4.0 = 8.00\text{ s}$；在 `500 VMs/s` UFFD 唤醒下，按需缺页抢走 `50.33 GB/s` 带宽，使未限流 GDS 可用带宽跌至 `4.07 GB/s`、Swap 时间拉长 `13.4x` 至 `2.139 s`（烧掉 `136.92 GiB·s`），而 35% 宿主机 I/O 令牌桶限流将 Swap 稳在 `0.246 s`（`15.75 GiB·s`，活跃 Rollout 提升 `13.23x–25.58x`）；`uffd.copy()` 返回 `-EAGAIN` 是因为并发气球驱动 `MADV_DONTNEED` 产生了 `UFFD_EVENT_REMOVE`，Handler 必须先从 UFFD 队列排空并处理所有挂起的 `Remove` 事件后再重试。用户掌握状态仍为 `not-assessed`。

## 23. FP8/NVFP4 预训练数值稳定性（`1x128` 分块 / `16x16 RHT` / 随机舍入）、万卡 NVRx 容错/ABFT 与 600kW 机柜功率阶跃缓冲

阅读 [FP8 细粒度分块缩放与 Blackwell NVFP4 预训练数值稳定性](../components/fp8-nvfp4-pretraining-numerics.md)、[万卡同步训练容错与静默数据损坏（SDC）防御](../components/fault-tolerant-training-resilience.md) 与 [132kW–600kW 液冷机柜毫秒级功率阶跃与 OCP ORv3 超级电容削峰](../components/rack-power-transient-smoothing.md)：
(1) 在 `NVIDIA/TransformerEngine`（`nvfp4.rst`、`core_nvfp4.cuh`）与 `deepseek-ai/DeepGEMM`（`sm90_fp8_gemm_1d2d.cuh`）中：为什么 `100x` 激活离群通道会使 Tensor-wise FP8 Delayed Scaling 的非离群通道 SQNR 跌至 `21.49 dB`，而 DeepSeek `1x128` 分块缩放恢复至 `31.94 dB`？为什么 NVFP4（`1x16 E2M1`）在 Wgrad 上必须配合 `16x16` 随机哈达玛变换（`RHT`）与硬件随机舍入指令 `cvt.rs.satfinite.e2m1x4`（对比 `RTN` 的 `38.89%` 偏差）？
(2) 在单卡 `MTBF = 12,000 h` 的 `65,536` 卡同步训练集群中，集群联合 `MTBF` 是多少分钟？为什么传统 Slurm 整作业重启 + PFS Checkpoint 的有效训练时间比（`ETTR`）暴跌至 `18.58%`，而 `NVRx` `inprocess.Wrapper`（`19.8 s` 热恢复）+ `CliqueReplicationStrategy`（`40.0 s` 周期）能维持 `90.30%`？为什么有限 `BF16` 权重的 `16` 种单比特翻转对 `NaN/Inf` 看门狗的即时检出率为 `0.0%`，而正交行列 ABFT 校验和能以 `0.0732%` 算力开销实现 `100%` 单步拦截？
(3) 在 `16,384` 卡 `GB200 NVL72`（`132 kW/rack`，单步摆幅 `66.24 kW` 于 `5.0 ms` 内完成）中，未缓冲的全集群同步阶跃斜率 `dP/dt` 高达多少 `MW/s`？若用 OCP ORv3 超级电容缓冲架（`CBU`，工作在 `48V -> 36V`）将单柜斜率限制在 `0.05 kW/ms`，单柜需多少 `kJ` 可用能量与多少 `Farads` 电容量？为什么通信等待期 `Dummy Burn Kernel` 不应作为首选？

答案：(1) Tensor-wise 缩放因子被单个 `100x` 离群通道撑大 `100` 倍，导致其余正常通道 `37.5%` 的激活值下溢归零（`21.49 dB`），而 `1x128` 分块将离群点隔离在单个 Tile 内，其余通道保持 `31.94 dB`（`+10.45 dB`）；Wgrad 沿 Token 维规约时离群点落入同一 `1x16` 微块（`18.19 dB`），`16x16 RHT` 将峰值因子从 `3.048` 压至 `1.410`、恢复 `35.06 dB`（`+16.87 dB`），且硬件随机舍入（`SR`）满足 $\mathbb{E}[Q(x)]=x$，将亚网格小梯度的范数偏差从 `RTN` 的 `38.89%` 压至 `1.25%`（降低 `31.1x`）；
(2) `65,536` 卡联合 `MTBF = (12000 × 60) / 65536 = 10.99 min`（`659.2 s`）；Slurm + PFS 单次重启高达 `960.0 s`（接近整个集群 MTBF）且落盘阻塞 `95 s`，故 `ETTR` 跌至 `18.58%`；`NVRx inprocess.Wrapper` 通过 `AbortTorchDistributed` 与 `ShiftRanks` 激活 `reserve_fn` 热备节点（`18.0 s`）并从对等节点 RAM 拉取分片（`1.8 s`），将 Young-Daly 最优周期压至 `40.0 s`、维持 `90.30%` `ETTR`（`+71.72 pts`）；有限 `BF16`（如 `0.375`, `exp=125`）翻转任意 1 个比特后指数仍 `< 255`（全为有限数），故 `isnan/isinf` 检出率为 `0.0%`，而正交 ABFT 利用 $Ce = A(Be)$ 在 `1` 步内 `100%` 捕获；
(3) `228` 台 `GB200 NVL72` 机柜同步阶跃斜率高达 `3,014.7 MW/s`（`600 kW` Rubin Ultra 机柜更达 `13,926.4 MW/s`）；将单柜 `66.24 kW` 阶跃展宽至 `1,324.8 ms`（`0.05 kW/ms`）需可用能量 $E = 0.5 \times 66.24 \times (1.3248 - 0.005) / 0.997 = 43.88\text{ kJ}$，对应电容量 $C = 2 \times 43,880 / (48^2 - 36^2) = 87.1\text{ Farads}$（结合 `15 ms` DCGM 固件平滑将集群斜率压至 `11.38 MW/s`，削减 `264.91x`）；而 `Dummy Burn` 白白浪费 `+5.88%` 至 `+7.40%` 全集群总电耗且斜率仍达 `1,191.2 MW/s`。用户掌握状态仍为 `not-assessed`。

## 24. SM90/SM100 Warp-Specialized Ping-Pong & TMEM、Zero-Bubble / DualPipe 双向流水线与千级 Multi-LoRA SGMV 服务

阅读 [SM90 Hopper 与 SM100 Blackwell Warp-Specialized Ping-Pong 算子与 TMEM 工程](../components/warp-specialized-kernel-engineering.md)、[Zero-Bubble (ZB-H1/ZB-H2/ZB-V) 与 DeepSeek-V3 双向 DualPipe/DualPipeV 流水线编排](../components/zero-bubble-dualpipe-scheduling.md) 与 [千级多租户 Serverless Multi-LoRA 服务架构](../components/multi-lora-serving-architecture.md)：
(1) 在 `NVIDIA/cutlass` `sm90_gemm_tma_warpspecialized_pingpong.hpp` 与 `sm100_gemm_tma_warpspecialized.hpp` 中：为什么 `128x256x64` Tile 在均匀分配（`168 regs/thread`）下会产生 `87 regs/thread` 溢出，而 `HeavyRegisterPressure >= 208` 触发 `setmaxnreg`（Producer `24` / Consumer `240 regs`）能把总寄存器控制在 `64,512 <= 65,536` 并提速 `2.303x`？Blackwell SM100 的 `256 KB/SM` `TMEM` + 单 Warp（`32` 线程）`tcgen05` UMMA + `2-CTA Cluster MMA` 如何进一步将每级流水线 L2 流量降低 `50%`？
(2) 在 `PP = 16, M = 64`（`F=10ms, B_dgrad=10ms, W=9ms, Comm=9.5ms`）下，对比标准 `1F1B`、`ZB-V` 与 `DualPipe`/`DualPipeV` 的气泡率、峰值激活显存（chunks）与通信暴露占比。为什么 `sail-sg/zero-bubble` 的 `rollback_adamw`（`zbpp_optimizer_helper.py`）能在异步 `AllReduce` 发现 `Inf/NaN` 后以 `6.94e-18` 的精度无损原地回滚参数与动量？
(3) 在单台 `8x H100` 服务 `1,000` 个动态秩 LoRA（并发活跃 `64` 个，`r=16, BS=256`）时，为什么按适配器拆批会导致吞吐崩塌至 `415.6 tok/s`，而 `vLLM` Punica 共享基座 GEMM + Triton `SGMV`（`_BI_SPLIT_K = 8`）可达 `15,171.3 tok/s`（`36.50x`）？`S-LoRA` 统一分页池如何把外部碎片率从 `31.4%` 压至 `1.6%`（KV 容量 `+154.7%`），且 SGMV 相对合并权重实例池的秩盈亏拐点 $r^*$ 是多少？

答案：(1) `128x256` FP32 累加器需 `256 regs/thread`（含流水线共 `255 regs`），均匀 `168 regs` 导致每线程 `87 regs` 溢出到本地内存（SOL `38.30%`）；`setmaxnreg` 将无需累加器的 Producer 压至 `24 regs`（`3,072 regs`），释放给 2 个 Consumer 各 `240 regs`（`61,440 regs`，总计 `64,512 <= 65,536`），把溢出压至 `15 regs`、SOL 提升至 `88.20%`（`2.303x`）；SM100 将累加器移入每 SM `256 KB` 专用 `TMEM` 并由单 Warp（`32` 线程）发射 `tcgen05` UMMA，`2-CTA Cluster MMA` 让相邻两个 CTA 各通过 TMA Multicast 加载半份右矩阵 `B`（每级 L2 读取从 `48 KB` 降至 `24 KB`，减少 `50.0%`，4K Prefill 达 `94.20%` SOL）；
(2) 标准 `1F1B` 气泡率 `14.97%`、暴露通信 `21.17%`（步耗时 `2,906.41 ms`，`16` chunks）；`ZB-V` 按 V-Shape 折返并拆分 `B_dgrad`/`W_wgrad`，在保持 `1x` 参数副本与 `16` chunks（`13.60 GiB`）的同时将纯计算气泡压至 `1.26%`（`2,145.68 ms`）；`DualPipe` 将异向微批次结对为 `_forward_backward_chunk`，用一侧计算掩盖另一侧 `EP All-to-All` 通信，把暴露通信压至 `0.45%`（`2,008.71 ms`，提速 `1.447x`），`DualPipeV` 在 V-Half 上免除 `2x` 参数副本并保持 `0.53%` 暴露通信（`2,010.23 ms`）；`rollback_adamw` 先重构步长向量 $\Delta p = -\text{lr}_{\text{eff}} \cdot \hat{m}/(\sqrt{\hat{v}}+\epsilon)$，通过 $p \leftarrow (p + \Delta p)/(1 - \text{lr}\cdot\lambda)$、$m \leftarrow (m - (1-\beta_1)g)/\beta_1$ 与 $v \leftarrow (v - (1-\beta_2)g^2)/\beta_2$ 逆向还原（最大误差 `6.94e-18`），在 `PP=16` 下消除同步 Barrier 带来 `+7.39%` 吞吐增益；
(3) 按适配器拆批将 1 次 `M=256` 基座 GEMM 拆成 64 次 `M=4` 极瘦 GEMM，强制每层重复从 HBM 读取 64 次基座权重（`615.94 ms`）；Punica 保持 1 次共享基座 GEMM，旁路通过 `LoRAKernelMeta` 稳定排序聚簇与 `_BI_SPLIT_K=8` Triton `SGMV` 仅耗时 `16.87 ms`（`36.50x`）；`S-LoRA` 将 KV Cache 与动态秩 LoRA 权重统一映射为粒度 `[page_size, H]` 的物理页，消除静态 `max_rank=128` Slab 的 `31.4%` 内部留白（碎片率 `1.6%`，可用 KV Token 从 `51,712` 增至 `131,712`）；当 $r \le 64$ 时 SGMV（`16,357.8 tok/s`）优于合并权重多实例池（`16,250.9 tok/s`），而 $r \ge 96$ 时 SGMV 开销达 `+46.2%`，故秩盈亏拐点为 $r^* = 64$。用户掌握状态仍为 `not-assessed`。

## 25. PyTorch FSDP2 (`DTensor`) & `SymmetricMemory` `Async-TP`、NCCL 零拷贝注册 (`CUMEM` NVLS) 与 DCGM HBM3e Row Remapping 预测性排空

阅读 [PyTorch FSDP2 (`fully_shard` 逐参数 `DTensor`) 与 `SymmetricMemory` `Async-TP`](../components/fsdp2-dtensor-async-tp-composability.md)、[NCCL 零拷贝用户缓冲区注册 (`ncclCommWindowRegister`) 与 `CUMEM` NVLS](../components/nccl-zero-copy-user-buffer-registration.md) 与 [GPU HBM3e ECC 行重映射 (`Row Remapping`) 与预测性无损排空](../components/gpu-ecc-row-remapping-predictive-failure.md)：
(1) 在 `pytorch/pytorch`（`torch/distributed/fsdp/_fully_shard/_fully_shard.py` 与 `torch/distributed/_symmetric_memory/__init__.py`）中：为什么 `FSDP1` 单一 `FlatParameter` 在 `70B VLM + 冻结编码器 + LoRA` 和 `405B Interleaved 1F1B + FP8` 场景下会产生 `18.6%–21.8%` 的 CUDA Caching Allocator 碎片率，而 `FSDP2` 逐参数 `DTensor` + 显式双缓冲预取（`set_modules_to_forward_prefetch`）将其压至 `1.15%–1.42%`（每卡节省 `6.72–9.65 GiB`）？`_pipelined_multi_all_gather_and_consume` 如何利用双流交错隐藏 `91.27%–97.09%` 的节点内 `TP=8` 通信气泡？
(2) 在 `NVIDIA/nccl`（`src/register/register.cc`、`coll_reg.cc`、`allocator.cc`、`nvls.cc`）中：为什么在 `132-SM` H100 上并发运行 `1,024-CTA` GEMM（理想 `8` 波，`1.720 ms`）时，未注册缓冲区的 NCCL 占用 `16` 或 `32` 个通信 CTA 会触发 Wave Quantization 尾波膨胀（导致 GEMM 劣化 `+16.46%` 至 `+47.18%`），而 `ncclCommWindowRegister` + `CUMEM` Fabric Handles 零拷贝仅需 `2` 个 CTA（`0.00%` GEMM 劣化）并每卡节省多少通信缓冲池显存？
(3) 在 `NVIDIA/DCGM`（`dcgmlib/dcgm_fields.h`、`dcgm_structs.h`、`modules/health/DcgmHealthWatch.cpp`）中：区分 `Xid 63`、`Xid 64`、`Xid 94` 与 `Xid 95` 的故障级别。为什么不能仅按 `SBE > 2/hr` 盲目驱逐节点（会产生多少次误报），而应结合 `DCGM_FI_DEV_BANK_REMAP_AVAIL_LOW/NONE`（Field `388/389`）+ `SBE` 突增速率（$\ge 15/\text{hr}$）+ `ROW_REMAP_PENDING`（Field `396`）触发步边界 `2.4 s` 微检查点热备排空？

答案：(1) `FSDP1` 将层内混合 `requires_grad`/精度的参数强行拼接进单一 1D `FlatParameter`，迫使可训练与冻结参数分离重打包，且跨流 `recordStream` 延迟释放生命周期各异的大块临时缓冲（碎片率 `18.6%–21.8%`）；`FSDP2` 对每个 `nn.Parameter` 独立构建 `DTensor` 并复用固定大小的双缓冲 Unshard 池，将碎片率压至 `1.15%–1.42%`（节省 `6.72–9.65 GiB/GPU`，重打包提速 `16.0x–16.77x`）；`_pipelined_multi_all_gather_and_consume` 先在本地同步完成 `copy_shard` 与第 `0` 块 `shard_consumer` 子 GEMM，随后在 `backend_stream`（通过 NVLink P2P + `symm_mem.barrier(channel=step%2)` 拉取下一块）与 `current_stream`（计算当前块 GEMM）之间实现乒乓重叠，在 `8,192` Token 下将暴露通信从 `2.012 ms` 压至 `0.078 ms`（隐藏 `96.13%`），使 1,024 卡 5D 单步提速 `1.322x`（`4,820 -> 3,645 ms`）；
(2) `1,024` 个 GEMM CTA 在 `132` 个 SM 上需 $\lceil 1024/132 \rceil = 8$ 波；未注册 NCCL 占用 `16` 或 `32` 个 CTA 搬运内部 `buffSizes` FIFO 时，可用 SM 降至 `116` 或 `100` 个，使 GEMM 波数跳变为 $\lceil 1024/116 \rceil = 9$ 波（`+16.46%`）与 $\lceil 1024/100 \rceil = 11$ 波（`+47.18%`）；通过 `ncclMemAlloc`（`CU_MEM_HANDLE_TYPE_FABRIC`）+ `ncclCommRegister` 将用户缓冲区直接绑定至 NVSwitch 多播组（`NVLS_REG_COMPLETE`）与 IB MR（`NET_REG_COMPLETE`），通信 CTA 压至 `2` 个（剩余 `130` SM 仍保持 $\lceil 1024/130 \rceil = 8$ 波，`0.00%` 尾波膨胀），在 `2–5` 个通信域下每卡节省 `576.0 MiB`（`0.562 GiB`）至 `2,360.0 MiB`（`2.305 GiB`）内部 FIFO 显存，并将 `0.25–256 MiB` 集合通信 P99 延迟降低 `2.07x–3.74x`；
(3) `Xid 63` 为成功登记页退役/行重映射（`WARN`），`Xid 64` 为行重映射硬件失败（`FAIL`），`Xid 94` 为受控未修复 ECC 错误（`WARN`，仅杀死当前进程），`Xid 95` 为非受控致命 ECC 错误（`FAIL`，污染全局上下文需整机排空）；单纯 `SBE > 2/hr` 在 10,000 卡 30 天内会触发 `362` 次误报驱逐（准确率仅 `21.65%`），而联合 `BANK_REMAP_AVAIL_LOW/NONE`（72h 致命故障概率 `68.40%/91.20%`）+ SBE 突增速率触发 `2.4 s` 微检查点热替换，可将非计划中断从 `146` 次削减 `89.04%` 至 `16` 次（准确率 `87.84%`），将 30 天集群 ETTR 从 `95.10%` 提升至 `99.93%`。用户掌握状态仍为 `not-assessed`。






