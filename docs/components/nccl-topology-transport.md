# NCCL topology/transport：谁选择路径，谁推进完成？

状态：2026-10-13，固定 `NVIDIA/nccl@3e46c089f4ae4732722a419eb6096aead7e47a30`
的有界静态研究。先修：[互连分层](interconnect-network.md)、
[RDMA 接口生命周期](rdma-lifecycle.md)。没有运行 NCCL/GPU/RDMA，没有
device trace、生产调优、故障演练或作者 benchmark 重测。

证据 ID：`nccl-selection-path`、`nccl-connector-path`、`nccl-net-progress`、
`nccl-ib-lifecycle`、`nccl-stream-contract`、`ring-cut-offline`。
每个来源的实际行范围、blob SHA 和限制在 [sources](../../research/sources.json)；
主张与证据类型见 [evidence](../../research/evidence.json)。

## 1. 不要拿旧版本路径解释新提交

这个 commit 的旧路径 `src/enqueue.cc`、`src/graph/tuning.cc`、
`src/transport/net_ib.cc` 返回 404；实现已拆到 `src/enqueue/enqueue.cc`、
`src/tuning/*` 和 `src/transport/net_ib/*`。下面只解释读到的固定实现，
不将它当全部 NCCL release 或部署版本的保证。发布日期未知，登记为 null；
核验日期和 commit 可读不代表新 release、硬件交付或测得性能。

| 决策层 | 问题 | 固定实现入口 |
|---|---|---|
| Topology/eligibility | 哪些 GPU/peer/NIC 路径及算法组合可用？ | `graph/paths.cc`、`tuning/cost_model.cc` |
| Algorithm | collective 用什么通信计划？ | `tuning/ring.cc`、`tuning/tree.cc` 等 |
| Protocol | LL/LL128/SIMPLE 怎样分块、传 flag、同步？ | `include/device.h`、`device/prims_simple.h` |
| Connector transport | 每条 send/recv 连接怎样访问 peer？ | `transport.cc`，P2P/SHM/NET/CollNet |
| NET provider | NET API 由哪个实现提供？ | `plugin/net.cc`，外部插件/内置 IB/Socket |

初始化先选 NET provider，再构建 topology/graphs、初始化 tuning，并可建立
连接；消息 enqueue 再使用候选集与成本选择 algorithm/protocol。不是每个
消息都重新选择 provider、重建所有连接。

## 2. 选择是 eligibility + model + override，而非一个消息阈值

[算法入口](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/enqueue/enqueue.cc#L2159-L2225)
`ncclGetAlgoInfo` 形成 function、datatype、计算字节数、注册状态、支持条件
与候选 mask。全局强制环境配置优先于 per-call `algSelection`；
per-call soft selection 无有效候选时可以重新自动选择，forced selection
则报错。本文以读取到的 legacy enqueue 路径为主，另一路 raw-task enqueue
不作完整分析。

`ncclTuningCompute` 过滤候选并计算成本，tuner plugin 可以修改形成的时间表，
再按正的 `selectionTimeUs` 或 `timeUs` 最小值选择，并调整 channel/thread
及 CTA efficiency。Tree 模型只支持 AllReduce；没有实现的组合不可因为
成本“便宜”而生效。LL128 的默认平台门控又可能被显式 protocol 配置改变，
所以强制变量不是安全、质量或性能保证。

一般时间模型（`tuning_general.cc:52–64`）：

`T_us = (L_us × k + N_bytes / (1000 × B_GB_per_s)) × q`

`1 GB/s = 10^9 bytes/s = 1000 bytes/μs`；`N` 取自
`ncclFuncMaxSendRecvCount × elementSize`，不能对所有 collective 都直接等同
用户输入 buffer 大小。`k` 对应 pipeline/batch 假设，`q` 是模型修正。
模型不是逐次计时，也不包含真实拥塞下 p99。

Ring AllReduce 取 `nSteps = 2(R−1)`，把 channel 聚合后的 bus bandwidth
乘 `R/nSteps` 转为 algorithm bandwidth，再应用 protocol 效率与上限。
Tree 使用不同的带宽 cap、转换、深度延迟及消息大小修正。因此
“小消息总是 Tree，大消息总是 Ring”只是经验概括，不是完整实现规则。
此提交还对 FP8、Ring、ranks > 8 的组合将成本乘 1024：这是实现中的
数值质量偏好，不是误差实测，不证明某质量门槛已通过。

## 3. Connector 与 NET provider：初始化 fallback 的确切边界

[selectTransport](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/transport.cc#L15-L43)
按 P2P → SHM → NET → CollNet 调用 `canConnect`，选第一个接受者；
选中后的 `setup` 错误向上传播，**该函数没有 setup 失败后尝试下一个
transport 的循环**。

P2P 要检查 host/container、topology、NVML read/write、CUDA peer access
和需要的 IPC 能力。SHM 要检查 hostHash/shmDev。`ncclTopoCheckNet` 若发现
两端到近端 NIC 的路径带宽都高于 GPU-GPU 路径，P2P/SHM 可主动拒绝，
让**同机连接也走 NET**。仅凭“同机”“GPU 可 P2P”不能断言实际 transport。
一些 NVLink 邻接/P2P 禁用分支返回错误，不是无条件静默降级。

Linux NET provider 初始化按外部插件、内置 IB、Socket 注册/尝试；
部分 init 失败会禁用候选继续尝试。但 `netName` 过滤及 load/setup 错误
意味着不能承诺任何失败都落到 Socket。**初始化候选选择、初始化错误、
运行期异步错误是三类事件，不能统一称为 fallback。**

## 4. 数据移动：GPU buffer 与 user buffer 不是同一个概念

| 路径 | 读到的数据/控制布局 | 反例或限定 |
|---|---|---|
| P2P write/read | SIMPLE write 指向接收方缓冲；read 在发送方保留 SIMPLE 数据供 peer 读；head/tail 控制进度 | 此 commit 默认 read 条件限定同 compute capability 且为 80 的 NVLink pair；不是全部后续 GPU |
| SHM | 协议缓冲连续布置在共享 host-memory 控制结构后，GPU 可访问 | transport 表无 proxy-progress callback；不是“所有 SHM 都由 CPU memcpy worker 搬运” |
| NET + host bank | GPU 与 host 协议缓冲/控制状态协作，host proxy 调 provider | host 内存、CPU 进度和调度仍可能成为约束 |
| NET + GDR bank | NIC 访问 GPU 中 NCCL 协议 buffer，经注册与完成契约推进 | dedicated send buffer 的 LL 仍在 host；`useGdr` 不保证每种协议都直访 GPU |
| 注册用户 buffer | 额外注册条件满足时，SIMPLE 分支可用 user/ring registered address | GDR 协议 buffer 不等于 user-buffer zero-copy，更不等于零 HBM/CPU 开销 |

P2P 同 PID 可使用直接指针，跨 PID 导入 shareable buffer；cuMem 同进程
映射的引用计数还用于防止 peer abort/free 后失效访问。SHM 既可使用 cuMem
host shareable allocation，也可传统共享内存；排障不能只看 `/dev/shm`。

GDR 检查 GPU/NIC 能力、方向、GDR read 策略和 topology 距离。PXN 可以借
同节点 NVLink peer GPU 接近 NIC；这里的路径改写是 GPU→NIC 发送方向，
接收端不支持相同 remote PXN proxy。RDMA NIC、GDR 支持和该条连接实际
生效必须分别举证，不能从 `directSend` 函数名推断。

## 5. SIMPLE/NET 的两个同步环，加上应用的完成契约

`ncclConnInfo` 有每协议 buffer/MR handle、head/tail、flags、FIFO、
stepSize 与持续 step。SIMPLE 有 8 个 slot，通常 `stepSize=buffSize/8`；
sender 等 head credit，receiver 等 tail；发布前的 system fence 和 abort
检查不能省略。

```text
send: GPU 填协议 buffer/发布状态
   -> proxy 检查就绪 -> isend -> test 完成 -> 清 FIFO -> 返还 head credit
recv: proxy irecv -> test 完成 -> 按需 flush
   -> 发布 tail -> GPU 消费/更新 head -> proxy 回收接收资源
```

已读 `net.cc` 的 `sendProxyProgress` 与 `recvProxyProgress`；
recv 的 posted/received/transmitted/done 分开推进。GDR 接收 flush 可是
GDRCopy PCIe read 或 provider `iflush`，完成后才发布 tail。
`wc_store_fence()` 中 wc 指 write-combining，不是 verbs Work Completion。
device handle 的 `needsProxyProgress` 又可取消 host progress callback；
上图是所读标准 host NET 路径，不代表所有外部/device provider。

**NET request 完成 ≠ GPU 可见 ≠ GPU 已消费 ≠ 应用 CUDA stream 完成。**
固定 `docs/userguide/source/usage/streams.rst:6–18` 说明 NCCL 返回通常表示
已 enqueue 到 CUDA stream，不是结果已经完成；group 内多 stream 会加入
跨 stream 同步依赖。应用应以相应 CUDA stream/event 契约检查消费者何时
可用数据，本轮没有独立验证全部 CUDA/硬件内存序规范。

## 6. 内置 IB 将 verbs 生命周期接到实际 transport

| 阶段 | 固定路径/读取到的操作 | 必须保留的边界 |
|---|---|---|
| 发现 | `net_ib/init.cc` open/query device，选择 Active IB/Ethernet link-layer 端口，应用 HCA filter | 端口能力不是端到端可通或 RoCE 配置正确的证明 |
| 建连接 | `connect.cc` 引用计数 PD、CQ、RC QP；INIT 后交换 metadata 再 RTR/RTS | IB 数据面仍用 IP socket 建控制连接；看到 TCP 不证明 Socket payload |
| 注册 | `reg.cc` 按页覆盖范围、cache/refcount；普通 MR、DMA-BUF、relaxed ordering 等分支 | CUDA/HOST bank 决定注册类型；不等于 provider/kernel 全部支持已核验 |
| 接收发布 | `common.h/p2p.cc` 的 CTS FIFO 含 address/size/rkey/tag/idx；irecv 先建请求与接收 WR，isend 等匹配发布 | 远端地址/key 和 credit 都在生命周期里，不是随时任意远端写 |
| 数据提交 | local SGE address/lkey，remote address/rkey；RDMA_WRITE，最后 WR 用 immediate + signaled | 本提交多 QP 按设备权重分块，不是无条件均分 |
| 完成 | `ncclIbTest` poll WC、请求/设备 event 计数、request slot 回收 | 调用成功不代替 WC status；失败不能算成功吞吐 |
| 清理/错误 | MR dereg/refcount、buffer、provider close、QP/CQ/PD；proxy 错误写 asyncResult | 局部 teardown 函数不是任意 abort 已安全 drain 的证明 |

`IB_RESILIENCY_PORT_FAILOVER` 默认 0；disabled 时数据 CQ error 返回
`ncclRemoteError`，enabled 才进入 recovery handler。本轮只读门控和
调用点，没有读完或运行 recovery/retransmission state machine。
“完全没有恢复”与“自动无损恢复”都超出了现有证据。

## 7. 合成 cut 预算：相同 per-rank payload，不同共享 NIC 压力

[原创程序、输入和结果](../../experiments/ring_cut/README.md) 实际在 CPU
运行，标为 `own-offline`。固定 8 ranks、每 rank 1 GiB、两节点各 4 ranks、
每节点共享单方向 `50 GB/s` payload 上限；一个理想等分 Ring：

`V_edge = 2(p−1)S/p = 1.75 GiB`，
`V_node = outgoing_cross_edges × V_edge`，
`T_floor = max(V_node / B_node)`。

节点分组 Ring 每节点一条跨节点 outgoing edge，发送 1.75 GiB，
NIC payload 下界 `0.03758096384 s`；节点交替每节点四条，发送 7 GiB，
下界 `0.15032385536 s`。**只有该共享 NIC 资源下界相差 4 倍**，
不是预计 benchmark、NCCL channel trace 或整体应用加速比。
收发不重复加总，GB/GiB 和单/双向口径显式固定；没有把网络 payload
算成 GPU/HBM 流量。多 rail/独立 NIC、其他算法和慢节点内链路是反例。

## 8. 架构师解释、计算与设计练习

1. 解释：画出 algorithm/protocol/transport/provider 四层选择，并给出
   “同机仍 NET”“GDR 仍 host LL”“IB 控制面仍 TCP”三个反例的代码证据。
2. 计算：不看结果独立复算两个 Ring 的 closed-loop cut、节点发送量和
   单方向时间下界；全部 ranks 同节点时网络下界 0 为什么不表示耗时 0？
3. 设计：为低延迟 TP 与大块 DP 各列 topology/eligibility 表和 SLO 证据
   清单；固定 count/dtype/quality、rank mapping、协议、channel、
   driver/NIC/container，分别记录 CPU proxy、host buffer、MR/QP/CQ
   和共享链路约束。不能只交峰值链路速率表。
4. 故障：分别处理 canConnect 拒绝、setup 错误、runtime CQ error。
   说明何时应中止、保留失败分母及请求关联；不要承诺透明重试整个作业。

增加 channel/CTA、GDR、注册 cache 和 recovery 都有资源及运维代价，
不能仅用 GPU utilization 或 algbw 排名。模型未提供 TTFT/TPOT、p99、
质量验证、美元或功率数据；成本应以后续有效工作/SLO 与失败恢复分母计算。
这些练习没有用户学习反馈，mastery 仍 `not-assessed`。

## 9. 尚未核验与后续可证伪问题

未深入 graph search/channel 排列、LL/LL128 完整 device primitives、
NVLS/CollNet/PAT、GIN/GDAKI/RMA、外部 NET/tuner、Socket payload、
DMA-BUF/peermem 内核、完整 abort/drain 与 port recovery。静态读取的
默认路径不代表具体部署的有效路径。

下一研究：不运行硬件，先追 graph 如何将可行边变成 channel/rank/NIC
映射，找出 cut 预算不能代表它的条件；另读 port-failover 状态机，检查
重传、完成计数和 cleanup 的不变量。以后获批硬件时，固定 workload/
拓扑/版本，对“路径改变是否改善 SLO-goodput”作有失败分母的对照实验；
若路径没变、CPU proxy 更紧或质量不达标，就反驳仅凭峰值/GDR 名称的收益。
