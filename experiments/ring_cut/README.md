# Ring cut：rank 顺序怎样改变共享 NIC 的 payload 下界

原创 Python 标准库、CPU/offline 算例；不运行 NCCL、GPU、RDMA 或模型。
固定的 NCCL Ring 静态证据见 [组件正文](../../docs/components/nccl-topology-transport.md)；
**本程序不模拟 NCCL graph search、algorithm selection 或实际 channel 布局。**

## 输入、公式与方向

8 ranks，各输入 `S = 1 GiB = 1073741824 bytes`；两个合成节点 A/B，
各 4 ranks。一个理想等分 Ring AllReduce，无 padding、协议开销或重传。
每条有向 Ring edge 的发送量：

`V_edge = 2 × (p - 1) × (S / p) = 1879048192 bytes = 1.75 GiB`。

节点 `n` 的共享 NIC 发送量为 `V_n = k_n × V_edge`，
`k_n` 是该节点到另一节点的有向 Ring edge 数，包含最后 rank 回到第一个
rank 的闭环。跨节点总发送量只累计每条跨节点 edge **一次**，不重复加接收。
它不是所有 GPU 的 HBM 流量。

假设每节点一个共享发送 payload 上限 `B = 50 × 10^9 bytes/s`，即十进制
`50 GB/s = 400 Gb/s`；接收为独立全双工方向，不是“收发合计 50 GB/s”。
这是合成有效 payload 上限，不是某 NIC 标称规格或实测值。

`T_floor = max_n(V_n / B)`：仅该共享发送资源的乐观时间下界。

## 可重算结果

| Ring 顺序 | 跨节点有向 edges | A/B 每节点发送 | NIC payload 时间下界 |
|---|---:|---:|---:|
| `0,1,2,3,4,5,6,7`，节点分组 | 2 | 1.75 GiB | 0.03758096384 s |
| `0,4,1,5,2,6,3,7`，节点交替 | 8 | 7 GiB | 0.15032385536 s |

两例固定 p、S、NIC 上限与节点归属，只改变 Ring 顺序；后者共享 NIC
payload 下界是前者的 4 倍。**这不证明 NCCL 实际采用其中任一 Ring，
也不证明应用耗时或吞吐差 4 倍。**多 rail、多个 channel、树或网络内归约
需要各自模型。节点内链路更慢、同步或归约计算都可能成为更紧的约束。

反例：全部 ranks 在一个节点时本模型网络下界是 0，但 collective 时间
不是 0。若每 rank 有独立 NIC 而非共享 NIC，不能沿用这里的节点合计分母。
质量、TTFT/TPOT、p99、美元成本与功率均未建模。

## 运行

```powershell
python experiments\ring_cut\run.py
python experiments\ring_cut\run.py --check
python -m unittest tests.test_ring_cut
```

[输入](input.json) 与 [结果](results.json) 都是合成数据。测试独立断言每条
edge、每节点发送、闭环、全机内反例，以及非法输入和过期结果。
