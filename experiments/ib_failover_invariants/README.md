# IB failover invariants：NCCL 端口故障转移的选择性重传、有符号完成计数与容量放大

原创 Python 标准库、CPU/offline 算例；不运行 NCCL、IB HCA、RDMA 交换机或 GPU。
固定的 `NVIDIA/nccl@3e46c089f4ae4732722a419eb6096aead7e47a30` 源码证据见
[NCCL IB Port-Failover 与异步恢复不变量正文](../../docs/components/nccl-port-recovery-invariants.md)。

## 为什么需要这个算例

在启用 `NCCL_IB_RESILIENCY_PORT_FAILOVER=1` 时，`src/transport/net_ib/p2p_resiliency.cc` 与 `p2p.cc`
并不是简单地“把失败的 `ibv_post_send` 再调一次”，而是依赖四条相互咬合的正确性不变量：

1. **单设备容错硬上限（`ncclIbResiliencyCheckErrorNotFatal`）**：
   仅当 WC 状态属于 `{IBV_WC_WR_FLUSH_ERR, IBV_WC_RETRY_EXC_ERR}` 且失败设备数 `nFailedDevices <= 1` 且 `< ndevs` 时才允许故障转移。单 NIC 通信域（`ndevs = 1`）或双 NIC 同时故障（`nFailedDevices = 2`）会立即返回 `ncclRemoteError`。
2. **CQ/RQ 预分配容量放大（`ncclIbResiliencyDataCqSizeGet` / `DataRqSizeGet`）**：
   当故障设备上的 `activeQps[]` 全部迁移到存活设备后，存活设备的 CQ 必须容纳原本分布在 `ndevs` 个设备上的所有 CQE，因此发送端 CQ 放大为 `NET_IB_MAX_REQUESTS × nqps × ndevs`，接收端 CQ 放大为 `NET_IB_MAX_REQUESTS × 2 × nqps × ndevs`。
3. **基于 `IBV_WR_RDMA_READ` 探测的选择性重传（`ncclIbResiliencyHandleProbeCompleted`）**：
   发送端在 `10 ms` 延迟后通过存活设备上的 `probingQp` 单边读取接收端 `cmplsRecords[slot].completions[0..127]`（`128 B`）：
   - 若所有 `qpIndex` 均已在接收端落盘（仅硬件 ACK 丢失），发送端零重传直接完成；
   - 若部分 `qpIndex` 未到达，仅重传 `completions[qpIndex] == false` 的分片，且保持原 `origDevIndex` 的 `128 B` 对齐分块偏移不变。
4. **跨设备接管的有符号事件守恒律（`ncclIbResiliencyRequestIsComplete`）**：
   接收端初始在 Dev 0 和 Dev 1 各登记 `+2` 个期望事件；若 Dev 0 的某条分片通过 Dev 1 的替代 QP 送达，Dev 0 剩余 `+1` 而 Dev 1 变为 `-1`。朴素无符号判定 `all(events[d] == 0)` 会永久死锁，而 NCCL 的有符号抵消判据 `|negativeEventsSum| == remainingEventsSum`（`|-1| == 1`）能够准确识别请求完成。

## 可重算结果摘要（`S = 1 MiB, nqps = 4, max_requests = 64`）

| Case ID | `ndevs` / 故障设备 | 允许 Failover | 发送/接收 CQ 容量（基线 → Resiliency） | 盲目重传 vs 选择性重传 | 接收端 `events[]` 终态 | 有符号完成 vs 无符号完成 |
|---|---:|---|---:|---:|---|---:|
| `lost_ack_all_delivered_zero_retransmit` | `2` / `[0]` | `True` | `256→512` / `512→1024` | `1048576 B` vs `0 B`（省 `100%`） | `[0, 0]` | `True` vs `True` |
| `partial_loss_selective_retransmit_and_signed_events` | `2` / `[0]` | `True` | `256→512` / `512→1024` | `1048576 B` vs `262144 B`（省 `75%`） | `[1, -1]` | `True` vs `False`（防挂起） |
| `full_dev0_loss_redirected_to_dev1` | `2` / `[0]` | `True` | `256→512` / `512→1024` | `1048576 B` vs `524288 B`（省 `50%`） | `[2, -2]` | `True` vs `False`（防挂起） |
| `dual_device_failure_fatal_ceiling` | `2` / `[0, 1]` | `False` | `256→512` / `512→1024` | 拒绝恢复（`ncclRemoteError`） | `[2, 2]` | `False` vs `False` |
| `single_nic_comm_cannot_failover` | `1` / `[0]` | `False` | `128→128` / `256→256` | 拒绝恢复（`ndevs == 1`） | `[2]` | `False` vs `False` |

## 运行与校验

```powershell
python experiments\ib_failover_invariants\run.py
python experiments\ib_failover_invariants\run.py --check
python -m unittest tests.test_ib_failover_invariants
```
