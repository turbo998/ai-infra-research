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
