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

