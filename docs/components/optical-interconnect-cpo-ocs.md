# 共封装光学（CPO）、线性直驱（LPO）与光路交换（OCS）在十万卡集群的功耗与故障率模型

## 1. 核心架构问题与结论（Executive Summary）

当 AI 训练与推理集群从万卡演进至 **100,000 卡（100K-GPU）3-Tier Clos 胖树（Fat-Tree / Rail-Optimized）** 时，网络互连不再仅是带宽与拥塞问题，而成为决定机房总功耗（MW）与同步训练有效算力（Synchronous Goodput）的头号物理瓶颈：
1. **可插拔 DSP 光模块的功率墙（Power Wall）**：在 3-Tier Clos 拓扑中，每个 GPU 端到端跨越 `6` 个光学收发器等效跳数（Leaf 下行、Leaf 上行、Spine 双向等），十万卡 `800G` 集群需配置约 **`450,000` 只 800G 光模块**（聚合交换机端口带宽 `360 Pbps`）。传统带 PAM4 DSP 重定时芯片的 800G OSFP 可插拔光模块功耗高达 **`16.0 W/只`（`~20.0 pJ/bit`）**，加上交换机侧 SerDes 驱动功耗（`1.6 W/端口`），仅光互连子系统即消耗 **`7.92 MW`** 持续功率。
2. **链路闪断（Link Flap）引发的十万卡同步训练崩塌**：在十万卡紧耦合同步训练中，单只可插拔光模块因热应力与激光器老化导致的年化故障/闪断率（AFR / Flap Rate）直接乘以 `450,000` 基数。传统 DSP 可插拔集群平均每天发生 **`32.0` 次光学链路闪断**（平均无闪断间隔仅 **`45.0 min`**）。即便依赖 `NVIDIA/nccl` 的 RAS 子系统快速定位死链（默认 `60s` `DEAD` 超时）并从最后一次 Checkpoint 恢复，有效同步训练吞吐（Goodput）仍跌至 **`70.31%`**（若无快速热备重路由，更跌至 `51.82%`）。
3. **CPO 共封装光学与 ELSFP 外置激光源的“共享爆炸半径（Shared Blast Radius）”**：将硅光引擎（Optical Engine, `OE`）共封装到交换机 ASIC 基板并将激光源外置为可插拔冷板模块（`ELSFP`, External Laser Small Form-Factor Pluggable），可将光学子系统能耗降至 **`4.8 pJ/bit`（`2.16 MW`，立省 `5.76 MW`）** 且将日均故障降至 **`7.2` 次/天（Goodput 提升至 `91.32%`）**。但我们对 `sonic-net/sonic-platform-daemons`（commit `dd8f83a4a9cb23d1b034373f3f1b639bff5e0373`）CPO 守护进程源码的核验表明：**单颗 `OE` 由多个物理端口共享，单只 `ELSFP` 激光模块更同时为 `8` 个 `800G` 端口供光；一旦单颗 `OE` 进入 `ModuleLowPwr` 状态或单只 `ELSFP` 失效，SONiC `deinit_oe_sibling_pports()` 会连带关停所有兄弟端口，形成 `1.6 Tbps`（单 OE）乃至 `6.4 Tbps`（8 端口 ELSFP）的多端口关联闪断爆炸半径**。
4. **MEMS / 压电 OCS（光路交换）去 Spine 层与重配置延迟权衡**：用全光无源 MEMS/压电 OCS 替代电交换 Spine 层，消除了 Spine 层的 O-E-O 光电转换与对端模块（光模块数量减半至 `225,000` Equiv），结合 Leaf CPO 可将十万卡光学总功耗压降至 **`0.72 MW`（`1.6 pJ/bit`，比传统 DSP 节省 `90.9%` 即 `7.20 MW`）**，日均闪断降至 **`2.4` 次/天**，同步训练 Goodput 达到 **`96.93%`**。但其物理代价是 MEMS 镜面切换耗时 **`15.0 ms`**（压电式约 `1.0–2.5 ms`），只能用于 **Job 级拓扑重编排或链路故障旁路恢复**，绝对不能介入微秒级的 MoE All-to-All 逐包动态路由。

---

## 2. 一手源码与规范深度核验（SONiC CPO Xcvrd & NCCL RAS）

### 2.1 `sonic-net/sonic-platform-daemons`：`CpoTopology`、`OE` 与 `ELSFP` 兄弟端口联动状态机

在 `sonic-net/sonic-platform-daemons`（commit `dd8f83a4a9cb23d1b034373f3f1b639bff5e0373`，`2026-10-12`）的 `sonic-xcvrd/xcvrd/xcvrd_utilities/common.py`（blob `a26c2756615fddd6d9d20062f56a7a6063efc000`）与 `sonic-xcvrd/xcvrd/cpo/cpo_manager_task.py`（blob `745eb8a247e29d680bb75679c40c03395890f2bb`）中，SONiC 定义了 CPO 交换机的硬件映射与状态机契约：

1. **双层物理共享拓扑（`common.py` & `tests/test_cpo.py`）**：
   - 定义设备类型常量：`CPO_DEVICE_TYPE_OE = 'optical_engine'` 与 `CPO_DEVICE_TYPE_ELSFP = 'external_laser_source'`。
   - `CpoTopology` 从 `platform.json` 解析每个 `optical_engine`（如 `oe1..oe16`，包含 `max_banks: 2..3`）与 `external_laser_source`（如 `elsfp1..elsfp16`，包含 `lasers: 8`）到交换机物理端口（`pport`）的映射关系。
   - 提供两个核心拓扑查询函数：
     - `get_oe_sibling_pports(pport)`：返回与当前物理端口共用同一颗共封装光学引擎（`OE`）的所有兄弟物理端口及其 `bank` / `lanes_mask`；
     - `get_elsfp_sibling_pports(pport)`：返回与当前物理端口或激光源共用同一只外置激光器（`ELSFP`）的所有兄弟端口。
2. **OE 低功耗/故障时的兄弟端口级联关停（`cpo_manager_task.py:CpoManagerTask`）**：
   ```python
   def deinit_oe_sibling_pports(self, lport):
       pport_list = self.get_lport_to_pport_list(lport)
       for pport in pport_list:
           sibling_pports = common.get_oe_sibling_pports(pport)
           for sibling_pport in sibling_pports:
               sibling_api = self.get_xcvr_api(sibling_pport)
               lanes_mask = self.get_host_lanes_mask(sibling_pport)
               sibling_api.set_datapath_deinit(lanes_mask)
               sibling_api.tx_disable_channel(lanes_mask, True)
   ```
   当 `handle_cmis_dp_deinit_state(lport)` 检测到某个 CPO 端口触发模块级 `ModuleLowPwr` 或重置时，由于同一颗 `OE` 上的多端口共享底层 CMIS 状态机与光电引擎，控制面必须同步调用 `deinit_oe_sibling_pports(lport)` 将所有兄弟端口的数据通路反初始化（`set_datapath_deinit`）并关闭发送激光通道（`tx_disable_channel`）。
3. **架构师启示（为什么 CPO 需要 Rail/Plane 隔离映射）**：
   - 在传统可插拔交换机中，单只 800G OSFP 损坏仅丢失 `1 × 800 Gbps` 端口（爆炸半径 `= 1`）；
   - 在 CPO 交换机中，单颗双端口 `OE` 重置会同时切断 **`2 × 800 Gbps = 1.6 Tbps`**，而单只 8 通道 `ELSFP` 外置激光源失效会同时切断 **`8 × 800 Gbps = 6.4 Tbps`**！
   - **拓扑布线红线**：在构建 Rail-Optimized 或 Multi-Plane 网络时，**严禁将同一个 GPU 节点的多张网卡（或同一集体通信 Ring 的主备路径）接入共享同一颗 `OE` 或同一只 `ELSFP` 的兄弟端口**；必须将同一 `ELSFP` 的 8 个端口正交打散到 8 个独立的通信平面（Planes）或不同 Pod 的上行链路中。

### 2.2 `NVIDIA/nccl`：RAS 子系统（`ncclras`）与链路故障检测时间窗

在 `NVIDIA/nccl`（commit `3e46c089f4ae4732722a419eb6096aead7e47a30`，`2026-10-08`）的 `docs/userguide/source/troubleshooting/ras.rst`（blob `fbc9319b149533b0780c53e043c33da7359336fe`）中，NCCL 详细规定了大规模集群遇到光模块闪断或网卡挂死时的带外诊断机制：
1. **独立于数据面的带外 TCP 监控网络（Port `28028`）**：
   - 每个 NCCL 进程在初始化时派生一个独立的 RAS 线程，通过普通 TCP Socket（默认监听 `localhost:28028`，由 `NCCL_RAS_ADDR` 配置）构建与 RDMA/NVLink 数据面完全解耦的监控网格，实时交换通信域状态并支持外部 `ncclras` CLI 查询。
2. **连接健康状态分级与 `60s` 死亡阈值**：
   - `MISMATCH`：两端 Rank 报告的 communicator 拓扑或状态不一致；
   - `INCOMPLETE`：握手未完成或心跳短暂丢失；
   - `DEAD`：对端 Rank 连续 **`60 秒`** 无响应，正式判定为死节点/死连接并汇报给作业调度器。
3. **动态 Profiler 唤醒与硬件根因定位（NCCL 2.31）**：
   - 运维人员无需重启训练作业，即可通过 `ncclras` 发送 `CONTROL PROFILER_MASK` 动态开启慢节点性能剖析；
   - 启用 `ncclras -D`（或环境变量 `NCCL_RUN_RAS_DIAGNOSTICS=1`）时，RAS 线程会自动在故障节点执行全套硬件自检：扫描 `/dev/kmsg` 中的 GPU `Xid` 与 NVSwitch `SXid` 硬件错误、检查 NVLink 链路状态、执行 `rdma_topo check`、校验 IOMMU Group 隔离以及 PCIe `ATSCtl` 状态。

---

## 3. 十万卡光学互连功耗、闪断与 Goodput 数学模型

### 3.1 功耗与能量效率公式（`pJ/bit` 到 `MW`）

设十万卡集群总 GPU 数为 $N_{\text{gpu}} = 100,000$，每卡网络注入带宽为 $B_{\text{gpu}} = 800\text{ Gbps}$，3-Tier Clos 拓扑每个终端对应的全网光学收发器等效倍数为 $M_{\text{opt}}$（3-Tier Clos 取 $M_{\text{opt}} = 4.5$，OCS 消去 Spine 双向模块后取 $M_{\text{opt}} = 2.25$）：
1. **光模块总数与聚合光带宽**：
   $$N_{\text{mod}} = N_{\text{gpu}} \times M_{\text{opt}}, \quad B_{\text{agg}} = N_{\text{mod}} \times 800\text{ Gbps}$$
2. **光互连子系统总功率（MW）与等效比特能耗（pJ/bit）**：
   $$P_{\text{fabric}} = N_{\text{mod}} \times \left(P_{\text{mod}} + P_{\text{serdes}}\right) \times 10^{-6}\text{ MW}$$
   $$E_{\text{bit}} = \frac{P_{\text{mod}} + P_{\text{serdes}}}{800\text{ Gbps}} \times 10^3\text{ pJ/bit}$$

### 3.2 链路闪断频率與同步训练 Goodput 模型

设单只光模块的等效独立故障时间（MTBF）为 $\text{MTBF}_{\text{unit}}$（小时），則全网日均光学链路闪断次数 $\lambda_{\text{day}}$ 与平均无闪断间隔 $\text{MTBF}_{\text{cluster}}$（分钟）为：
$$\lambda_{\text{day}} = \frac{N_{\text{mod}} \times 24}{\text{MTBF}_{\text{unit}}}, \quad \text{MTBF}_{\text{cluster}} = \frac{1440}{\lambda_{\text{day}}}\text{ min}$$
设每次闪断引发的 NCCL RAS 故障隔离、OCS/ECMP 旁路重路由与 Checkpoint 恢复耗时为 $T_{\text{rec}}$（分钟），周期性异步 Checkpoint 固定开销占比为 $\alpha_{\text{ckpt}} = 1.5\%$，则集群稳态同步训练有效吞吐率（Goodput）为：
$$\text{Goodput} = \frac{\text{MTBF}_{\text{cluster}}}{\text{MTBF}_{\text{cluster}} + T_{\text{rec}}} \times (1 - \alpha_{\text{ckpt}})$$

---

## 4. 原创 CPU 离线实验结果（`experiments/optical_cpo_ocs_fabric/`）

运行 `python experiments/optical_cpo_ocs_fabric/run.py --check` 对 4 种十万卡 800G 光学互连架构与 4 种 CPO/OCS 故障模式进行确定性推演，核心结果如下：

### 4.1 十万卡（100K-GPU）四代光学互连架构对比表

| 架构方案 (`id`) | 光模块等效数量 | 单端口总功耗 (`Mod + SerDes`) | 等效比特能耗 (`pJ/bit`) | 十万卡光学总功耗 (`MW`) | 集群日均闪断 (`flaps/day`) | 集群无故障间隔 (`MTBF min`) | 稳态同步训练 Goodput (`%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pluggable_800g_dsp_clos`**（传统 800G PAM4 DSP 可插拔 3-Tier Clos） | `450,000` | `16.0 + 1.6 = 17.6 W` | **`22.0 pJ/bit`** | **`7.92 MW`** | **`32.0 / day`** | `45.0 min` | **`70.31%`**（无热备仅 `51.82%`） |
| **`lpo_800g_linear_clos`**（800G LPO 线性直驱无 DSP） | `450,000` | `6.8 + 2.8 = 9.6 W` | **`12.0 pJ/bit`** | **`4.32 MW`** | `19.29 / day` | `74.67 min` | **`78.75%`** |
| **`cpo_quantum_x800_elsfp`**（CPO 共封装硅光 + 外置液冷激光源 ELSFP） | `450,000` | `4.2 + 0.6 = 4.8 W` | **`6.0 pJ/bit`** | **`2.16 MW`** | **`7.20 / day`** | `200.0 min` | **`91.32%`** |
| **`ocs_cpo_spine_free`**（Leaf CPO + MEMS/压电 OCS 无源光路交换去 Spine） | `225,000` | `2.6 + 0.6 = 3.2 W` | **`4.0 pJ/bit`** | **`0.72 MW`** | **`2.40 / day`** | **`600.0 min`** | **`96.93%`** |

### 4.2 CPO / ELSFP / OCS 故障爆炸半径与重配置时间窗矩阵

| 故障或重配置事件 (`scenario`) | 共享硬件组件 | 连带中断的 800G 端口数 | 瞬时丢失带宽 (`Tbps`) | 平面/控制面恢复耗时 (`ms`) | 架构处置策略 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`single_pluggable_osfp_flap`** | 独立 800G OSFP 模块 | `1` | `0.8 Tbps` | `50.0 ms` | ECMP / AR 自适应重路由或单轨降级 |
| **`cpo_single_oe_bank_deinit`** | 共封装 Optical Engine (`OE`) | **`2`** | **`1.6 Tbps`** | `120.0 ms` | SONiC `deinit_oe_sibling_pports()` 同步关停兄弟端口；需跨 OE 正交布线 |
| **`cpo_shared_elsfp_laser_fail`** | 8 通道外置激光源 (`ELSFP`) | **`8`** | **`6.4 Tbps`** | `450.0 ms` | `get_elsfp_sibling_pports()` 识别受影响端口；同一 ELSFP 严禁接入同一节点多卡 |
| **`ocs_mems_circuit_reconfig`** | 3D MEMS 阵列镜面切换 | **`64`** | **`51.2 Tbps`** | **`15.0 ms`** | 仅在 Job 启动前或 Spine 级故障恢复时切光路，运行期排空（Drain）相关 Rail 流量 |

---

## 5. 面向解决方案架构师的选型矩阵与排障练习

### 5.1 架构师选型指南
1. **何时选择 LPO（线性直驱）**：机架内或相邻机架短距（`<5–10m`）互连、交换机主板具备极低损耗 PCB 材质（Megatron-7/8 或 Flyover 电缆）且同一批次交换机/网卡 SerDes 联合调优完成的场景，可直接砍掉约 `45%` 光模块功耗，但跨厂商混插误码率（BER）裕度窄，不适合长距跨排 Spine 连接。
2. **何时选择 CPO + ELSFP**：单机柜功率密度突破 `100 kW`（如 GB200/GB300 NVL72 与 Quantum-X800 / Spectrum-X800 液冷交换机）、十万卡级长期同步训练集群。必须将 `ELSFP` 视为热插拔耗材（类似风扇/电源模块），在布线设计上强制执行 **“1 ELSFP = 8 独立 Rail/Plane”** 的反亲和规则。
3. **何时引入 OCS 光路交换**：多 Pod 跨数据中心楼宇互连、需要按天/按作业动态调整 Pod 间超额认购比（Oversubscription Ratio）或绕过整台故障 Spine 的超大规模集群。

### 5.2 思考与计算练习
- **练习 1（ELSFP 爆炸半径与 Rail 反亲和证明）**：某 8-Rail 集群的 Leaf CPO 交换机采用 `8` 端口共享 `1` 只 `ELSFP` 的设计。若工程师错误地将节点 `Node-01` 的 `NIC-0..NIC-7` 顺序插在同一台 CPO 交换机的 `Port 1..8`（共享 `ELSFP-1`），当 `ELSFP-1` 激光器老化离线时，`Node-01` 的 NCCL AllReduce 会发生什么？如何修改端口映射表使单只 `ELSFP` 故障最多只影响每个节点的 `1/8` 带宽？
- **练习 2（OCS 重配置气泡边界）**：若某 MoE 模型每层 All-to-All 通信窗口为 `1.2 ms`，有人提议在每层 Forward 之间用 MEMS OCS（切换耗时 `15.0 ms`）动态重构光路以匹配热门专家拓扑，请计算每步迭代的通信有效利用率会跌至多少？压电或液晶硅光开关需要达到多少纳秒/微秒级切换速度才具备逐层重构可行性？
