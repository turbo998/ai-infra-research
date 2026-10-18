# UEC 1.1：规范发布与官方参考实现（`uet-ref-prov` / `spcl/pcm`）深挖

证据状态：**announcement + implementation + own-offline**。
- UEC 官方发布与规范下载索引（2026-10-12，[UEC 官方发布](https://ultraethernet.org/ultra-ethernet-consortium-releases-specification-v1-1-advancing-ai-and-hpc-networking-at-scale/)，完整规范 [UE-Specification-1.1.pdf](https://ultraethernet.org/wp-content/uploads/sites/20/2026/10/UE-Specification-1.1.pdf)，来源 ID `uec-1-1`）。
- 本轮已进一步核验并逐行精读 UEC 官方开源 libfabric 参考实现 [`ultraethernet/uet-ref-prov`](https://github.com/ultraethernet/uet-ref-prov)（commit `ae19b9776c42cd78f57118b9b7993ae913625e46`，来源 ID `uet-ref-prov-pkt-hdr`、`uet-ref-prov-cc-st`、`uet-ref-prov-pds-rudi`）与可编程拥塞管理框架 [`spcl/pcm`](https://github.com/spcl/pcm)（commit `356f67d890c16200f1bf8c31b50a05e76a2e16c7` 及论文 [arXiv:2610.13130](https://arxiv.org/abs/2610.13130)，来源 ID `spcl-pcm-nscc-reps`、`spcl-pcm-paper-2026`）。
- 完整源码级协议剖析与离线复现实验见组件深挖文档 **[`docs/components/uec-transport-congestion.md`](../components/uec-transport-congestion.md)** 与实验 **[`experiments/uec_fabric_congestion/run.py`](../../experiments/uec_fabric_congestion/run.py)**。

## 从公告线索到一手源码核验的闭环对照

| 机制 | 公告信息 (`announcement`) | 源码与实验已核验结论 (`implementation` + `own-offline`) |
|---|---|---|
| Scale-up / UFH / 小包优化 | UFH 相比 IP 节省 20B，IPG 最多节省 11B | `uet_pkt_hdr.h` 定义了 `uet_pds_rudi_req`（8B）与 `uet_ses_req_onm`（`UET_HDR_REQ_SMALL`，20B）；叠加 UFH 后 64B 小包线开销从 RoCEv2 的 98B 降至 63B（节省 35B），800G 有效吞吐提升 **+27.56%**（`316.05 -> 403.15 Gbps`）。 |
| UET PDS / SES / TSS 分层 | 支持乱序传输、同步组与零信任加密 | `uet_pds_ack_cc`（28B）内置 64-bit `sack_bitmap` 吸收多路径乱序；`uet_ses_atomic_ext` 原生支持 `BF16 (0x10)` / `FP16 (0x11)` 原子操作；`uet_pds_rudi.c` 明确 RUDI 无 PDC 状态下 TSS 无防重放保护，目标 MR 必须满足 `IDEMPOTENT_SAFE`。 |
| In-Network Collectives (`INC`) | 将部分归约放到网络，减少特定路径最多 2x 流量 | `uet_pkt_hdr.h:106` 定义 `UET_PDS_REQ_PDC_INFO_USE_RSV_PDC = 0x01` 标识 INC 预留 PDC；但端到端训练加速仍受 Amdahl 定律与计算/存储占比约束。 |
| NSCC 四象限拥塞控制与 Packet Trimming | 结合延迟、ECN 与数据包裁剪快速收敛 | `uet_cc_st.c` 与 `spcl/pcm/algorithms/nscc.c` 实现四象限控制：当 `ECN && q_delay < target_qdelay` 时保持 `cwnd` 不变并仅切换路径熵 `EV`（过滤幻影 ECN）；收到 `UET_NACK_TRIMMED` 时 `quick_adapt` 在 1 个 RTT 内将 16:1 Incast 窗口收敛至公平份额（`37,500 B`）。 |
| 多路径喷洒 (`Selective Spray` vs `REPS`) | 利用多路径负载均衡绕开拥塞与故障链路 | `uet_cc_st.c` 采用 2-bit 状态表（`GOOD/SKIP/BAD`），而 `spcl/pcm/algorithms/reps.c` 通过环形缓存 `VAR_EVC[16]` 复用非 ECN 路径：在 16 路径中 4 条热点拥塞（10G vs 50G）场景下，`REPS` 有效吞吐达 **`602.35 Gbps`**，是盲目喷洒（`160.0 Gbps`）的 **3.76 倍**。 |

## 为什么“通信最高 2x”不等于应用 2x

网络内归约可改变通信路径的数据移动数量，但总体训练/推理还包含计算、其他 collectives、同步、存储与恢复。就算某个通信环节改善，Amdahl 限制和 workload 分布仍影响端到端收益。

应核验基线是否同算法、拓扑、精度/归约顺序和质量；忽略 CPU/GPU、NIC/switch 的边界会得到错误分母。本项目不采用发布正文中的加速倍率作为实测或容量规划参数。

## 必须通过的采用门槛

1. **已核验**：固定 `ultraethernet/uet-ref-prov` 与 `spcl/pcm` 版本/commit，明确 PDS（`ROD/RUD/RUDI/UUD`）、SES（`STD/MEDIUM/SMALL`）、NSCC 与 REPS 的异常与幂等边界。
2. **待硬件验证**：核验具体商用 800G NIC/switch ASIC（如 Tomahawk 5/6 Packet Trimming 水位线与 CSIG 硬件开销），而非仅厂商加入联盟或宣布兼容。
3. 在相同模型/数据/精度/拓扑和质量门槛下比较有效吞吐、尾延迟、功率、失败/恢复。
4. 保留不支持/回退路径、拥塞/多租户条件与负面结果。

