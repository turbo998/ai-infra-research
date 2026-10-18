# HBM3e ECC 行重映射（Row Remapping）、`Xid 63/64/94/95` 故障演化模型与万卡集群预测性无损排空

- **覆盖问题 ID**：`gpu-hbm-ecc-row-remapping-predictive-drain`（`priority: 18`）
- **关联证据 ID**：`ev-dcgm-row-remap-fields-2026`、`ev-dcgm-healthwatch-xid-2026`、`ev-gpu-ecc-predictive-drain-sim-2026`
- **关联离线实验**：`experiments/gpu_ecc_predictive_drain/run.py`（`results.json`）
- **上游源码核验基线**：`NVIDIA/DCGM` (`commit 64df9f894541e426e416131a9820cae97aa4dd81`)
  - `dcgmlib/dcgm_fields.h` (`blob aa65ab23eb0d0ca3f60a062e61e6a62148ba1b21`)
  - `dcgmlib/dcgm_structs.h` (`blob cc49146e2d21d9ccd0304966d89ecaf4fcdf750a`)
  - `modules/health/DcgmHealthWatch.cpp` (`blob e3c7a5f811eebba2b0c8ae4190eea3d6f90d5b36`)

---

## 1. 架构背景：从单机 ECC 到万卡 HBM3e 集群的可靠性范式转移

在 10,000 张 H100/H200/B200 GPU 组成的同步大模型预训练集群中，共包含 **`60,000–80,000` 颗 HBM3e 高带宽显存堆栈**。在如此庞大的硅片面积与高热流密度下，单比特翻转（`SBE`, Single-Bit Error）与弱存储单元退化不再是偶发异常，而是每小时持续发生的背景物理过程。

自 Ampere (`GA100`)、Hopper (`GH100`) 至 Blackwell (`GB100`)，NVIDIA 用硬件级 **Row Remapping（行重映射）** 取代了早期 Pascal/Volta 的软件动态页退役（Dynamic Page Retirement）：
- 每个 HBM Bank 在制造时预留了有限数量的硬件冗余行（Spare Rows）；
- 当 InfoROM 记录到重复的 `SBE` 或不可纠正的双比特错误（`DBE`, Double-Bit Error）时，GPU 在下次复位（Reset）或重映射周期将该损坏物理行永久重映射到同一 Bank 的冗余行，保持物理地址空间连续且无需操作系统拆页。
- **核心运维陷阱**：一旦某个 HBM Bank 的冗余行耗尽（`BANK_REMAP_AVAIL_NONE`），或重映射挂起未及时复位，后续发生在同一行/列的电荷泄漏就会直接演变为运行期致命的 **`Xid 64`（Row Remap Failure）**、**`Xid 94`（Contained ECC Error）** 或 **`Xid 95`（Uncontained Fatal ECC Error）**，导致万卡同步训练作业瞬间崩溃中断。

---

## 2. 一手源码走读：`NVIDIA/DCGM` 遥测字段与 `DcgmHealthWatch` 判定逻辑

### 2.1 HBM 行重映射与可用冗余桶字段（`dcgmlib/dcgm_fields.h` & `dcgm_structs.h`）

在 `dcgmlib/dcgm_fields.h` (`blob aa65ab23eb0d0ca3f60a062e61e6a62148ba1b21`) 与 `dcgmlib/dcgm_structs.h` (`blob cc49146e2d21d9ccd0304966d89ecaf4fcdf750a`) 中，DCGM 暴露了完整的 HBM 健康状态面：

1. **Volatile 与 Aggregate ECC 计数器**（`dcgm_fields.h` L620–645）：
   - `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL` (`310`) / `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL` (`311`)：自上次驱动重置以来的运行期单/双比特错误计数；
   - `DCGM_FI_DEV_ECC_SBE_AGG_TOTAL` (`312`) / `DCGM_FI_DEV_ECC_DBE_AGG_TOTAL` (`313`)：持久化在 InfoROM 中的生命周期累计计数。
2. **HBM Bank 冗余行可用性五级分桶**（`dcgm_fields.h` L775–795）：
   - `DCGM_FI_DEV_BANK_REMAP_AVAIL_MAX` (`385`)：冗余行完全充足的 Bank 数量；
   - `DCGM_FI_DEV_BANK_REMAP_AVAIL_HIGH` (`386`)、`DCGM_FI_DEV_BANK_REMAP_AVAIL_PARTIAL` (`387`)；
   - `DCGM_FI_DEV_BANK_REMAP_AVAIL_LOW` (`388`)：冗余行濒临枯竭的 Bank 数量（**关键预警信号**）；
   - `DCGM_FI_DEV_BANK_REMAP_AVAIL_NONE` (`389`)：冗余行已彻底耗尽（0 剩余）的 Bank 数量。
3. **行重映射状态与硬阈值**（`dcgm_fields.h` L805–820，`dcgm_structs.h` L161）：
   - `DCGM_FI_DEV_ROW_REMAP_UNCORRECTABLE_TOTAL` (`393`)：因 `DBE` 触发的行重映射总数，硬上限常量为 `DCGM_LIMIT_MAX_ROW_REMAP_UNCORRECTABLE = 512`（`dcgm_structs.h` L161）；
   - `DCGM_FI_DEV_ROW_REMAP_CORRECTABLE_TOTAL` (`394`)：因频繁 `SBE` 触发的预防性行重映射总数；
   - `DCGM_FI_DEV_ROW_REMAP_FAILED` (`395`)：行重映射失败标志（非零即致命故障）；
   - `DCGM_FI_DEV_ROW_REMAP_PENDING` (`396`)：行重映射挂起（等待 GPU Reset 生效）。

### 2.2 `DcgmHealthWatch` 内存子系统与毁灭性 Xid 源码剖析（`DcgmHealthWatch.cpp`）

在 `modules/health/DcgmHealthWatch.cpp` (`blob e3c7a5f811eebba2b0c8ae4190eea3d6f90d5b36`) 中，健康巡检引擎通过 `MonitorMem`（L2200–2251）与 `BuildXidMappings`（L3355–3440）执行分级判定：

1. **六大内存监控探针（`MonitorMem`，L1701–2016）**：
   - `MonitorMemVolatileDbes`（L1701）：只要最近时间窗口内 `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL > 0`，立即置 `DCGM_HEALTH_RESULT_FAIL`（`DCGM_FR_VOLATILE_DBE_DETECTED`）；
   - `MonitorMemRetiredPending`（L1748）：当 `PAGE_RETIRED_PENDING != 0` 时返回 `DCGM_HEALTH_RESULT_WARN`；
   - `MonitorMemSbeDbeRetiredPages`（L1782）：不仅检查总退役页是否超过 `DCGM_LIMIT_MAX_RETIRED_PAGES` (`64`)，还专门计算过去 `1` 周（`oneWeekInUsec = 604800000000` 微秒）内新增的 `dbePagesRetiredThisWeek`，若总 DBE 退役页 `> DCGM_LIMIT_MAX_RETIRED_PAGES_SOFT_LIMIT` (`15`) 且本周新增 `> 1`，立即升级为 `FAIL`（`DCGM_FR_RETIRED_PAGES_DBE_LIMIT`）；
   - `MonitorMemRowRemapFailures`（L1893）：若 `DCGM_FI_DEV_ROW_REMAP_FAILED > 0`，立即判定 `DCGM_HEALTH_RESULT_FAIL`（`DCGM_FR_ROW_REMAP_FAILURE`）；
   - `MonitorMemRowRemapUncorrectable`（L1939）：在 Ampere 及更新架构上，当 `ROW_REMAP_UNCORRECTABLE_TOTAL >= 512` 时发出 `DCGM_HEALTH_RESULT_WARN`（`DCGM_FR_UNCORRECTABLE_ROW_REMAP_LIMIT`）；
   - `MonitorMemUnrepairableFlag`（L1977）：若 `DCGM_FI_DEV_MEMORY_UNREPAIRABLE != 0`，立即判定 `DCGM_HEALTH_RESULT_FAIL`（`DCGM_FR_FAULTY_MEMORY`）。
2. **`m_devastatingXids` 与 `m_subsystemXids[DCGM_HEALTH_WATCH_MEM]`（L3360–3390）**：
   - **全局毁灭性 Xid（无论开启哪个子系统均强制监控）**：
     - `Xid 48`（Double Bit ECC Error -> `FAIL`）
     - `Xid 74`（NVLink Critical Error -> `FAIL`）
     - `Xid 79`（Fallen Off the Bus -> `FAIL`）
     - `Xid 94`（Contained ECC Error -> `WARN`, `DCGM_FR_CONTAINED_ERROR`：硬件成功将未纠正 ECC 隔离在当前进程上下文内，GPU 未死锁但当前训练进程已终止）
     - `Xid 95`（Uncontained ECC Error -> `FAIL`, `DCGM_FR_UNCONTAINED_ERROR`：错误波及全局硬件状态，必须重置或隔离整卡）
     - `Xid 119 / 120`（GSP RPC Timeout / GSP Error -> `FAIL`）
     - `Xid 140`（ECC Unrecovered Error -> `FAIL`）
   - **内存子系统专属 Xid**：
     - `Xid 63`（Pending Page Retirements / Row Remap Recorded -> `WARN`）
     - `Xid 64`（Row Remap Failure -> `FAIL`, `DCGM_FR_ROW_REMAP_FAILURE`）

---

## 3. HBM3e 马尔可夫退化模型与万卡预测性排空实验（`experiments/gpu_ecc_predictive_drain`）

为什么仅靠 DCGM 默认的 `FAIL` 规则（被动等待 `Xid 64/94/95` 或 `ECC_DBE_VOL > 0`）不够？因为当 `FAIL` 触发时，运行在该 GPU 上的训练进程已经收到 `SIGBUS` / `cudaErrorECCUncorrectable`，导致 10,000 卡同步训练当前 Step 崩溃！

我们在 `experiments/gpu_ecc_predictive_drain/run.py` 中建立了 **5 态马尔可夫退化模型**，利用 **`S2_BANK_AVAIL_LOW_BURST`** 与 **`S3_BANK_EXHAUSTED_AVAIL_NONE`** 两个前置可观测状态，在 `Xid 94/95/64` 爆发前触发 **微检查点热备无损切换（Predictive Live Drain）**：

### 3.1 HBM3e 5 态马尔可夫退化与 72 小时致命 Xid 风险概率

| 状态 ID | DCGM 遥测复合判据 | DCGM 默认级别 | SBE 速率 (`/hr`) | 72h 内爆发 `Xid 64/94/95` 概率 | 生产调度动作契约 |
|---|---|---|---:|---:|---|
| **`S0_HEALTHY_AVAIL_MAX`** | `BANK_REMAP_AVAIL_MAX=all, SBE_VOL=0` | `PASS` | 0.02 | **0.04%** | 正常训练 |
| **`S1_SBE_CORRECTABLE_REMAP`** | `ROW_REMAP_CORRECTABLE>=1, AVAIL_HIGH/PARTIAL, Xid 63` | `WARN` | 1.80 | **3.50%** | 在下次计划内 Checkpoint 窗口异步执行 `nvidia-smi -r` 使重映射生效 |
| **`S2_BANK_AVAIL_LOW_BURST`** | `BANK_REMAP_AVAIL_LOW>=1, SBE_RATE>=15/hr, PENDING=1` | `WARN` | 24.50 | **68.40%** | **触发预测性无损排空**：在下一个 Step 边界执行 `1.8s` 对等内存 Checkpoint + NVRx 热备替换 |
| **`S3_BANK_EXHAUSTED_AVAIL_NONE`** | `BANK_REMAP_AVAIL_NONE>=1` 或 `UNCORRECTABLE>=512` | `WARN_CRITICAL` | 110.00 | **91.20%** | **立即预测性排空并 Cordon 节点送修 RMA** |
| **`S4_FATAL_XID_64_94_95`** | `ROW_REMAP_FAILED>0 (Xid 64)` 或 `DBE_VOL>0 (Xid 94/95)` | `FAIL` | 500.00 | **100.00%** | 硬隔离与 RMA（若未提前排空则当前训练 Step 已中断） |

### 3.2 10,000 卡集群 30 天（`720h`）四种排空策略对比

| 运维排空策略 | 30 天非计划训练中断次数 | 预测性无损排空次数 | 误报排空次数 | 非计划中断削减率 | 30 天累计损失集群小时 (`h`) | 有效训练时间比 (`ETTR`) |
|---|---:|---:|---:|---:|---:|---:|
| **`reactive_slurm_restart`（被动等崩溃 + Slurm 重启）** | 146 | 0 | 0 | 0.00% | 35.28 h | `95.10%` |
| **`reactive_nvrx_inprocess`（被动等崩溃 + NVRx 回滚重启）** | 146 | 0 | 0 | 0.00% | 4.01 h | `99.44%` |
| **`naive_any_sbe_cordon`（单凭 `SBE > 2/hr` 盲目驱逐）** | 28 | 490 | 362 | 80.82% | 3.76 h | `99.48%` |
| **`dcgm_markov_predictive_drain`（Bank 冗余分桶 + SBE 斜率预测排空）** | **16** | **148** | **18** | **89.04%** | **0.54 h** | **`99.93%`** |

---

## 4. 解决方案架构师运维守则与可证伪问题

1. **绝不单凭 `ECC_SBE_VOL_TOTAL > 0` 隔离节点，也绝不忽视 `BANK_REMAP_AVAIL_LOW/NONE`**：
   - 单纯 `SBE > 2/hr` 阈值会在 30 天内造成 **`362` 次误报驱逐**（False-Positive Drain），耗尽备用节点池；
   - 正确的复合触发器是：`(DCGM_FI_DEV_BANK_REMAP_AVAIL_LOW > 0 && d(SBE_VOL)/dt >= 15/hr) || DCGM_FI_DEV_BANK_REMAP_AVAIL_NONE > 0 || (DCGM_FI_DEV_ROW_REMAP_PENDING > 0 && d(SBE_VOL)/dt >= 10/hr)`。
2. **区分 `Xid 94`（Contained）与 `Xid 95`（Uncontained）的处置路径**：
   - `Xid 94` 表明硬件已将毒化数据隔离在故障进程内，节点 Kubelet / Slurm Epilog 只需重置该 GPU（`nvidia-smi -r`）使挂起的 Row Remap 生效，并校验 `DCGM_FI_DEV_ROW_REMAP_FAILED == 0` 即可重新纳管；
   - `Xid 95` 或 `Xid 64` 必须立即 Cordon 节点并转入硬件诊断/RMA 队列。
3. **后续可证伪实验问题**：
   - HBM3e 堆栈在冷却液进口温度从 `25°C` 升至 `40°C`（温水液冷节电模式）时，`S1 -> S2` 的单比特电荷泄漏斜率 $d(\text{SBE})/dt$ 是否遵循阿伦尼乌斯（Arrhenius）指数加速定律，从而改变机柜冷却 PUE 与集群 ETTR 的联合最优温度点？
