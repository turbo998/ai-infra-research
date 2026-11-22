# 企业智算中心功耗封顶（`Power Capping & DVFS 甜点`）与电源瞬态尖峰消峰（`Power Whip Smoothing`）选型指南 (#492)

> **核心观点（Executive Summary）**：
> 在现代智算中心里，真正限制你集群规模的往往不是机柜空间，而是 **园区变电站批下来的市电总功率上限（如 `1.5 MW` IT 配额）**！如果让所有 `H100 / H200` 全部按默认 **`700W TDP` 满血硬跑**，`1.5 MW` 最多只能上架 `1,024 张 GPU`，且最后 7% 的极限加速频率烧掉了额外 28% 的电力与热量。作为 **Cloud & AI Solution Architect**，利用 CMOS 芯片三次方功耗定律（$P \propto C V^2 f \propto f^3$）与 **“PD 分角色差异化功耗封顶（Prefill 池锁 `650W` 拼算力、Decode 池锁 `500W` 保 HBM 带宽）+ 同步瞬态功率斜坡平滑（`Power Whip Smoothing`）”**：Decode 阶段因为瓶颈在恒定频率的 HBM3e 显存带宽而非 SM 主频，**功耗直降 28.6%（`700W -> 500W`），吐字速度却保留了惊人的 `98.8%`（仅慢 1.2%！）**；在同一个 `1.5 MW` 机房电力配额下，**可上架 GPU 数量从 `1,024 张` 直接增加 25% 至 `1,280 张`**，整座机房总 Token 产出净增 **`+23.5%`**！
>
> **⚡ 配套交互式 H5 精读沙盘**：[**打开 GPU 功耗封顶 (Power Capping) 三次方曲线与机房多装卡计算器 (`docs/h5/gpu-power-capping-dvfs-transient-tco-lab.html`)**](../h5/gpu-power-capping-dvfs-transient-tco-lab.html)

![Enterprise GPU Power Capping DVFS & Transient Smoothing Summary](../../assets/svg/enterprise-gpu-power-capping-dvfs-sweetspot-transient-smoothing-tco-selection-faq-summary.svg)

---

## 一、 灵魂拷问：为什么把 Decode 节点限电到 500W，速度几乎一毫秒都不慢？

### Q1：从芯片物理公式看，为什么 `nvidia-smi -pl 550`（或 Decode 锁 `500W`）是性价比之王？
1. **CMOS 动态功耗的三次方墙（Cubic Voltage-Frequency Wall）**：
   - GPU SM 核心的动态功耗遵循：
     $$P_{\text{dynamic}} = C \cdot V^2 \cdot f_{\text{SM}}$$
   - 当芯片要从甜点频率（如 `1,620 MHz`）强行冲刺到最高加速频率（`1,980 MHz`，仅提升 `+22%`）时，为了防止高频晶体管时序违例，必须同步大幅拉高核心电压 $V$，导致 **功耗呈接近 $f^3$ 的三次方暴涨（从 `500W` 冲上 `700W`，暴增 `+40%`）**！
2. **Decode 阶段的“HBM 频率解耦红利”**：
   - 在 `H100 / H200` 上，当你通过 `nvidia-smi -pl 500` 限制 TDP 时，降频的 **只有 SM 计算核心时钟（$f_{\text{SM}}$）**，而决定 Decode 吐字速度的 **HBM3e 显存时钟（$f_{\text{HBM}}$，`4.8 TB/s`）保持 100% 满速不降频**！
   - 因为 Decode 阶段本来就有 65% 的时间在等 HBM 传数据，SM 稍微降频 15% 完全被访存等待时间掩盖，**实测 Decode 吞吐保留率高达 `98.8%`**！

### Q2：什么是让供电局和机房 UPS 闻风丧胆的“千卡功率鞭效应（Power Whip / Synchronous $dP/dt$）”？怎么防御？
- **故障现场**：
  - 在 1,024 卡同步训练或大规模批处理中，当所有 GPU 正在全速算矩阵乘法时，整群功耗高达 `1.6 MW`；
  - 突然遇到一个跨机 AllReduce 慢节点等待或同步 Checkpoint 停顿，**1,024 张 GPU 在不到 `2 毫秒` 内同时从 `700W` 跌落到 `120W` 空闲功耗（瞬间甩负荷 `1.0 MW`）**；等通信一结束，又在 `2 毫秒` 内同时冲回 `1.6 MW`！
  - 这种高达 **`500 kW/ms` 的方波阶跃震荡** 会直接击穿机房 UPS 逆变器动态响应极限，甚至引发变压器谐波共振跳闸！
- **架构师双层防御法**：
  1. **软件层托底与斜坡（Burn-Floor & Ramp-Up Smoothing）**：在训练/推理框架通信等待超过 `30ms` 时，在低优先级 CUDA Stream 自动启动轻量级占位算子（Dummy GEMM）将单卡谷底功耗托住在 `300W–350W`，并在冷启动时执行 `500ms` 线性功率爬坡；
  2. **硬件层机架级超级电容缓冲（Rack-Level Supercapacitor BBU）**：在 `50kW–132kW` 机柜的 `48V` 直流母排上配置高倍率超级电容架，毫秒级削峰填谷，向前端市电呈现平滑电流！

---

## 二、 四种 GPU 功耗封顶策略选型决策矩阵（`1.5 MW 机房电力配额上限`）

```mermaid
flowchart LR
    Cap["机房固定 IT 电力配额: 1.5 MW"] --> P{"功耗封顶策略"}
    P -->|"Tier 1: 700W 不设限硬跑"| T1["仅能装 1,024 张卡<br/>高温降频 + 故障率高"]
    P -->|"Tier 4: PD 角色感知封顶 (推荐 🏆)<br/>Prefill 锁 650W | Decode 锁 500W"| T4["可装 1,280 张卡 (+25% 容量!)<br/>Decode 速度保留 98.8% | 机房总产出 +23.5% 🏆"]
```

| 智算中心功耗封顶档位（`1.5 MW` 电力上限） | 单卡设定功耗上限 | Prefill 算力保留率 (%) | Decode 吞吐保留率 (%) | 单瓦特能效比 (`Tok/J`) | `1.5 MW` 配额下可装卡数 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 全集群 `700W` 满血硬跑（无封顶）** | `700 W` | `100.0%` | `100.0%` | `1.00x` | `1,024 卡` |
| **Tier 2: 一刀切激进限电 `400W`（拖慢 Prefill）** | `400 W` ✕ | `74.5%` ✕ | `92.0%` | `1.48x` | `1,536 卡` |
| **Tier 3: 全集群统一 `550W` 三次方甜点封顶** | `550 W` ✓ | `92.8%` ✓ | `97.2%` ✓ | `1.24x` | `1,248 卡 (+21.9%)` |
| **Tier 4: PD 角色感知封顶（P池 `650W` / D池 `500W` + 斜坡平滑 🏆）** | **`650W / 500W` 🏆** | **`98.4%` 🏆** | **`98.8%` 🏆** | **`1.31x` 🏆** | **`1,280 卡 (+25.0% 🏆)`** |

---

## 三、 生产落地核对清单（SA Production Checklist）

- [ ] **对 PD 分离集群实施差异化 `nvidia-smi -pl` 封顶**：通过 K8s DaemonSet 对 `role=prefill` 节点设置 `-pl 650`，对访存受限的 `role=decode` 节点设置 `-pl 500`（立省 28.6% 电费与制冷负荷，吐字速度仅差 1.2%）。
- [ ] **机柜 PDU 超配上架前必须绑定硬件级 BMC 功率封顶**：当利用功耗甜点在单机柜多上架 20% 节点时，必须在服务器 BMC 层写入硬功率封顶（Hardware Power Cap），防止极端故障时单柜跳闸。
- [ ] **千卡集群必测同步阶跃负载（Power Step Test）**：在 Day-0 验收阶段模拟全集群每 5 秒启停一次满载 GEMM，验证机房 UPS 与机柜超级电容在 `1 MW` 瞬态阶跃下的母线电压跌落 `< 3%`。

---

## 四、 延伸阅读与配套工具

- **交互式 H5 精学沙盘**：[docs/h5/gpu-power-capping-dvfs-transient-tco-lab.html](../h5/gpu-power-capping-dvfs-transient-tco-lab.html)
- **7-Card 4:5 视觉评测卡片**：[docs/reports/enterprise-gpu-power-capping-dvfs-sweetspot-transient-smoothing-tco-selection-cards.html](../reports/enterprise-gpu-power-capping-dvfs-sweetspot-transient-smoothing-tco-selection-cards.html)
- **交互式 Sizer 测算器**：[docs/reports/enterprise-gpu-power-capping-dvfs-sweetspot-transient-smoothing-tco-selection-sizer.html](../reports/enterprise-gpu-power-capping-dvfs-sweetspot-transient-smoothing-tco-selection-sizer.html)
- **可复现实验模型**：[`experiments/enterprise_gpu_power_capping_dvfs_sweetspot_transient_smoothing_tco_selection/run.py`](../../experiments/enterprise_gpu_power_capping_dvfs_sweetspot_transient_smoothing_tco_selection/run.py)
