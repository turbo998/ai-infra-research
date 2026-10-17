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





