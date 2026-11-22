# 第 485 篇里程碑：企业智算集群从交付验收到生产上线（`Day-0 to Day-2 AI Cluster Acceptance, Burn-In & Go-Live`）五阶段终极验收手册 (#485)

> **核心观点（Executive Summary）**：
> 在大型智算中心工程交付中，最昂贵的失误是 **“看到 `nvidia-smi` 8 张卡都在、`ibstat` 端口显示 Active，就签字付尾款并直接放业务流量上线”**！大规模硬件统计表明，新出厂上架的服务器与光模块存在约 **`2.5%–3.5%` 的早期失效率（Infant Mortality / “浴盆曲线”左端）**——包括散热器导热垫虚贴导致满载 15 分钟即降频 35% 的 **慢速劣质卡（Lemon Straggler GPU）**、高温高压下算错浮点数却不报错的 **静默数据损坏卡（Silent Data Corruption, SDC）**，以及满载 2 小时才开始闪断的脏光纤端口。只要千卡集群里混入 **1 张慢速劣质卡**，木桶效应就会把整座集群 1,024 张卡的速度拖慢 35%，使首月生产任务异常中断率高达 **`38.5%`**！本篇作为 **#485 里程碑（Milestone #485）**，制定了架构师签字收货的 **“Day-0 至 Day-2 五阶段全栈验收门禁（72 小时热烤机 + SDC 校验 + 两两配对二分 RDMA 压测 + GDS 存储压测 + AIPerf 推理长稳与拔纤混沌演练）”**：在签收前揪出 **`99.6%` 的暗病节点由厂商免费 RMA 更换**，将首月生产中断率压至 **`0.3%`**，集群长期有效算力（Sustained Goodput）达 **`98.4%`**！
>
> **⚡ 配套交互式 H5 精读沙盘**：[**打开智算集群 Day-0 至 Day-2 五阶段交付验收门禁与暗病筛查模拟器 (`docs/h5/milestone-485-day0-day2-cluster-acceptance-golive-lab.html`)**](../h5/milestone-485-day0-day2-cluster-acceptance-golive-lab.html)

![Milestone #485 Day-0 to Day-2 AI Cluster Acceptance Summary](../../assets/svg/enterprise-day0-day2-ai-cluster-acceptance-burnin-chaos-golive-selection-faq-summary.svg)

---

## 一、 灵魂拷问：怎样在签字付尾款前，把 3.2% 的“暗病劣质卡”和“脏光模块”全部逼出来？

### Q1：什么是“静默数据损坏（SDC）”与“慢速木桶卡（Straggler GPU）”？怎么用命令 100% 测出来？
1. **慢速木桶卡（Thermal / Power Straggler）筛查法**：
   - 正常 `H200/B200` 在跑满 `700W/1000W` 持续 15 分钟后，结温应稳定在 $\le 72^\circ\text{C}$（液冷 $\le 58^\circ\text{C}$），实测 `BF16/FP8 GEMM` 算力在全集群均值 $\pm 2\%$ 以内；
   - 而散热冷板贴合不良或供电模块瑕疵的“暗病卡”，前 1 分钟跑分正常，第 10 分钟温度冲上 $84^\circ\text{C}$ 触发硬件 `SW_THERMAL_SLOWDOWN`，算力暴跌 `30%–40%`！
   - **必跑门禁**：连续运行 **`dcgmi diag -r 4`（Level-4 深度压力测试）+ 72 小时满载热浸泡**，自动剔除任何单卡算力偏离集群中位数 **$> 2.5\%$** 或触发任何 `THERMAL / POWER_BRAKE` 降频告警的节点！
2. **静默数据损坏（Silent Data Corruption, SDC）筛查法**：
   - 比死机更可怕的是：某张卡的某个 Tensor Core 在高频低压下把矩阵乘法结果算错了最后几位，但没有触发 ECC 报错（直接导致大模型训练第 5 天 Loss 莫名其妙变成 `NaN`，或在线推理胡言乱语）；
   - **必跑门禁**：在 `700W` 高温满载下循环运行确定性随机种子矩阵乘法校验（固定输入 $A \times B$ 与黃金 SHA256 校验和比对），**要求 72 小时内 Bitwise 校验和 100% 零偏差**！

### Q2：为什么跑一次“全集群大锅饭 `AllReduce`”抓不出坏光模块？必须怎么跑网络验收？
- 如果直接跑 128 台机器全量 `all_reduce_perf`，一旦总带宽只有 `32 GB/s`，你根本不知道是哪两台机器之间的哪根光纤出了问题；
- **标准三步网络验收法（Three-Step RDMA Sweep）**：
  1. **Step A（单机内部 NVLink 环网验收）**：128 台节点并发各自跑单机 8 卡 `all_reduce_perf`，要求每台机器内部 `busbw` 偏差 `< 1%`；
  2. **Step B（相邻与跨柜两两配对扫描 Pairwise Sweep）**：将节点按 `(Node 0, Node 1), (Node 2, Node 3)...` 以及跨 SU 对角线配对并发跑 2 机 16 卡 `all_reduce_perf` 与 `alltoall_perf`——哪一对的 `busbw < 45.0 GB/s`（400G 线速的 90%），**3 秒钟直接锁定故障机器与故障 Rail 端口**！
  3. **Step C（全集群二分带宽满载浸泡 Bisection Soak）**：64 台发送端与 64 台接收端跨 Spine 满载对打 12 小时，要求 **零光模块闪断（Link Flap = 0）、零 PFC 死锁、全部端口 `Pre-FEC BER < 10^{-7}`**！

---

## 二、 智算集群签字收货“五阶段全栈验收门禁表”（5-Stage Acceptance Gate）

```mermaid
flowchart LR
    G1["Gate 1: 固件/BIOS/拓扑基线<br/>ACS=Off | Topo=PIX | memlock=-1"] --> G2["Gate 2: 72h 满载热烤机与 SDC 筛查<br/>DCGM Level-4 | 算力偏差<2% | 0 降频"]
    G2 --> G3["Gate 3: 两两配对与二分 RDMA 压测<br/>busbw >= 90% 线速 | 0 闪断 | BER<1e-7"]
    G3 --> G4["Gate 4: 3FS/Lustre GDS 存储压测<br/>单机 GDS >= 28 GB/s | 671B 冷启 <= 3s"]
    G4 --> G5["Gate 5: AIPerf 推理洪峰与拔纤混沌演练<br/>3x 洪峰 Goodput >= 99% | 故障自愈 <= 90s 🏆"]
```

| 交付验收严格度档位 | 上线前揪出暗病卡比例 (%) | 首月生产任务异常中断率 (%) | 集群长期有效算力 `Goodput` (%) | 静默数据损坏 (`SDC`) 拦截 |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 零烤机直接签收（仅看 `nvidia-smi` 与 `ibstat` UP）** | `12.0%` ✕ | `38.5%` ✕ | `54.0%` ✕ | 不支持 ✕ |
| **Tier 2: 单机 1 小时冒烟测试（无跨柜配对压测与 72h 热浸泡）** | `48.0%` | `18.2%` ✕ | `74.5%` | 不支持 ✕ |
| **Tier 3: 24 小时 `DCGM Level-4` + 全群单次 `AllReduce`** | `88.5%` ✓ | `4.1%` | `91.2%` ✓ | 支持 ✓ |
| **Tier 4: Day-0 至 Day-2 五阶段全栈门禁（72h + 二分网络 + 混沌演练 🏆）** | **`99.6%` 🏆** | **`0.3%` (-99.2% 🏆)** | **`98.4%` 🏆** | **支持（Bitwise 校验 🏆）** |

---

## 三、 生产交付签字前 10 项一票否决核对单（SA Sign-Off Checklist）

- [ ] **Gate 1（拓扑与固件）**：全部节点 BIOS `PCIe ACS = Disabled`、`nvidia-smi topo -m` 对应网卡全部为 `PIX`、同批次网卡与交换机固件版本 100% 统一。
- [ ] **Gate 2（72h 烤机与 SDC）**：72 小时 `DCGM Level-4 + SDC` 满载测试通过，全集群单卡算力极差 `< 2.5%`，`DBE (双比特 ECC 错误) = 0`，`Xid 31/48/79/119 = 0`。
- [ ] **Gate 3（网络与光模块）**：两两配对与全群 `nccl-tests`（8MB–1GB）实测 `busbw >= 90%` 硬件线速，所有光模块收发光功率裕度 `> 3 dB`、`Pre-FEC BER < 10^{-7}`。
- [ ] **Gate 4 & Gate 5（存储、推理与混沌自愈）**：`gdsio` 零拷贝吞吐达标；模拟拔掉任意 1 根光纤或注入单卡 `Xid 79` 故障，验证 K8s `Node Problem Detector` 在 30 秒内自动给故障节点打污点（`Cordon/Drain`）并在 90 秒内由热备节点完成作业恢复。

---

## 四、 延伸阅读与配套工具

- **交互式 H5 精学沙盘**：[docs/h5/milestone-485-day0-day2-cluster-acceptance-golive-lab.html](../h5/milestone-485-day0-day2-cluster-acceptance-golive-lab.html)
- **7-Card 4:5 视觉评测卡片**：[docs/reports/enterprise-day0-day2-ai-cluster-acceptance-burnin-chaos-golive-selection-cards.html](../reports/enterprise-day0-day2-ai-cluster-acceptance-burnin-chaos-golive-selection-cards.html)
- **交互式 Sizer 测算器**：[docs/reports/enterprise-day0-day2-ai-cluster-acceptance-burnin-chaos-golive-selection-sizer.html](../reports/enterprise-day0-day2-ai-cluster-acceptance-burnin-chaos-golive-selection-sizer.html)
- **可复现实验模型**：[`experiments/enterprise_day0_day2_ai_cluster_acceptance_burnin_chaos_golive_selection/run.py`](../../experiments/enterprise_day0_day2_ai_cluster_acceptance_burnin_chaos_golive_selection/run.py)
