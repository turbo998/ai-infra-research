# 10 万卡 AI Factory 全栈 CapEx / OpEx / ETTR / Token 经济学帕累托模型 (`full-stack-ai-factory-tco-simulator`)

> **证据类型声明**：本文综合本仓库 Phases 1–6 已核验的 27 项离线实验与 150+ 项一手规范/源码证据，通过原创可复现实验 `experiments/ai_factory_tco_simulator/run.py`（`own-offline`）构建十万卡（`100,000` GPUs）AI Factory 三年全生命周期总拥有成本（3-Year TCO）、每百万美元 TCO 产出的有效训练算力天（`Effective PFLOPS-days / $M TCO`）与每百万 Token 推理生产成本（`$/1M Tokens`）的跨层联立模型。

---

## 1. 为什么单点优化无法代表 AI Factory 的真实 ROI？

在设计 `100,000` 卡（`120–165 MW` 园区供电级别）的 AI 基础设施时，硬件采购（CapEx）与电力冷却（OpEx）只是分母；真正决定投资回报率（ROI）的分子是 **全栈乘性有效产出**：

$$\text{Effective PFLOPS-days} = N_{\text{GPU}} \times \text{Peak Dense PFLOPS} \times \alpha_{\text{precision}} \times \text{MFU}_{\text{compute+comm}} \times \text{ETTR}_{\text{resilience}} \times D_{\text{days}}$$

其中任何一个环节的短板都会将百亿级投资拖入低效泥潭：
1. **物理层与光互连（Phase 4 `optical-interconnect-cpo-ocs`）**：传统 800G DSP 可插拔光模块不仅消耗 `~145 W/GPU` 的额外光模块与散热功率，其周期性 Link Flap 还会拉低同步训练可用率；演进至 **CPO（`<5 pJ/bit`）+ OCS 光路交换** 可将单卡光网络附加功耗降至 `42 W/GPU`，配合液冷将 PUE 从 `1.28` 压至 `1.09`。
2. **网络传输与网内计算（Phase 2 `uec-specification` + Phase 6 `multi-tier-collective-sharp-ar`）**：从传统 RoCEv2 DCQCN + Ring AllReduce 升级至 **UEC 逐包喷洒 + NVLS + IB/UEC SHARP**，可将万卡跨节点通信步数从 $2(N-1)$ 降至对数级，将大规模模型训练 MFU 从 `0.36` 提升至 `0.56`。
3. **低精度数值管线（Phase 5 `fp8-nvfp4-pretraining-numerics`）**：通过 DeepSeek `1x128` 细粒度 FP8 分块缩放（Hopper 提速 `1.72x`）与 Blackwell `NVFP4`（`1x16` 微块 + 随机哈达玛变换 `RHT` + 随机舍入 `SR`，有效吞吐提速 `1.85x`），在保持与 BF16 无损收敛的同时成倍放大单卡有效算力。
4. **万卡容错与高速存储（Phase 4 `storage-gds-3fs-kv-tiering` + Phase 5 `megascale-fault-tolerant-training-stragglers`）**：传统 Slurm 整作业重启（`15 min`）+ POSIX PFS 写盘在十万卡规模下会将有效训练时间比（ETTR）拖垮至 **`71.2%`**；通过 **`NVRx` 进程内热备替换（`18 s`）+ 对等节点内存级/3FS USRBIO Checkpoint（`1.8 s`）**，可将万卡 ETTR 锁定在 **`94.2%–96.8%`**。
5. **机柜功率瞬态平滑（Phase 5 `rack-power-transient-grid-smoothing`）**：用 **OCP ORv3 超级电容（Supercap CBU）+ 预测性微批次错峰** 替代粗糙的集合通信等待期 `Dummy Burn` 空烧内核，单卡可节省 `93–100 W` 的无效发热。
6. **分离式推理与超长上下文混合注意力（Phase 2 `disaggregated-kv-routing` + Phase 3 `long-context-hybrid-attention-kv` + Phase 6 `heterogeneous-gpu-cluster-scheduling`）**：通过 Prefill/Decode 分离、FlashMLA/DSA 硬件对齐稀疏注意力、Hybrid Mamba-2 线性层混合以及跨代 `SM100 NVFP4 -> SM90 FP8` 紧凑 KV 传输，单卡有效推理吞吐从传统单体部署的 `48 tok/s/GPU` 跃升至 **`540 tok/s/GPU`**。

---

## 2. 四代 10 万卡 AI Factory 全栈基准配置与封印结果 (`experiments/ai_factory_tco_simulator/`)

运行 `python experiments/ai_factory_tco_simulator/run.py`（`100,000` GPUs，`3.0` 年折旧，工业电价 `$0.065/kWh`），封印结果如下（`results.json`）：

| 架构代际与全栈配置 (`profile_id`) | 园区总功率 (`MW`) | 3 年总 CapEx (`$M`) | 3 年总电费 OpEx (`$M`) | 3 年总 TCO (`$M`) | 折算单卡小时成本 (`$/GPU-hr`) | 单卡有效算力 (`Eff PFLOPS`) | **每百万美元 TCO 产出算力天 (`Eff PFLOPS-days / $M`)** | **每百万 Token 生产成本 (`$/1M Tokens`)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gen1_legacy_roce_bf16_slurm`**<br>*(Pluggable 800G + RoCE DCQCN + BF16 + Slurm/PFS + Dummy Burn + Monolithic)* | `161.28 MW` | `$4,700.0M` | `$275.5M` | `$4,975.5M` | `$1.893` | `0.254 PFLOPS` | `5,579.0` (`1.00x`) | **`$18.1038`** (`1.00x`) |
| **`gen2_hopper_fp8_nvrx_disagg`**<br>*(LPO + UEC Spray + FP8 1x128 + NVRx/3FS + CBU + P/D Disagg)* | `127.65 MW` | `$4,510.0M` | `$218.1M` | `$4,728.1M` | `$1.799` | `0.785 PFLOPS` | **`18,184.59` (`3.26x`)** | **`$3.7151` (`-79.5%`)** |
| **`gen3_heterogeneous_h100_b200_hybrid`**<br>*(50K H100 + 50K B200 + Asymmetric VPP + SM100->SM90 FP4 KV Transcode)* | `132.50 MW` | `$5,145.0M` | `$226.3M` | `$5,371.3M` | `$2.044` | `1.436 PFLOPS` | **`29,284.39` (`5.25x`)** | **`$1.8742` (`-89.6%`)** |
| **`gen3_blackwell_cpo_sharp_nvfp4_full`**<br>*(GB200 + CPO/OCS + NVLS/SHARP + NVFP4 + NVRx/3FS + Supercap + Hybrid MLA/Mamba)* | `137.01 MW` | `$5,780.0M` | `$234.0M` | `$6,014.0M` | `$2.288` | **`2.256 PFLOPS`** | **`41,083.27` (`7.36x`)** | **`$1.4307` (`12.65x` 降本)** |

---

## 3. 核心架构洞察

1. **同一代 Hopper 硬件仅靠全栈软件与网络/供电改造即可实现 `3.26x` 训练性价比提升与 `4.87x` Token 降本**：
   对比 `gen1_legacy_roce_bf16_slurm` 与 `gen2_hopper_fp8_nvrx_disagg`，两者均采用 `$36,000` 级别的 Hopper GPU，但通过启用 `1x128 FP8`、UEC 喷洒路由、`NVRx` 进程内热修复、3FS 零拷贝 Checkpoint、ORv3 电容削峰与 P/D 分离，园区总功耗反而从 `161.28 MW` 降至 `127.65 MW`（节省 `33.63 MW` 电力容量），同时每百万美元 TCO 产出的有效算力天从 `5,579` 飙升至 `18,184.59`。
2. **Blackwell 全栈 AI Factory 尽管 CapEx 上升 `23%`，但单位有效训练成本降低 `86.4%`（`7.36x` 效率），单位 Token 成本降低 `92.1%`（`12.65x`）**：
   这证明了评估新一代 AI 机柜绝不能只看单卡采购单价（`$48,000` vs `$36,000`）或单卡小时租金（`$2.288/hr` vs `$1.893/hr`），而必须以 **`$/Effective PFLOPS-day`** 和 **`$/1M Tokens`** 作为北极星指标。
3. **跨代异构车队（`50K H100 + 50K B200`）是存量数据中心平滑演进的帕累托最优点**：
   通过非对称流水线编排与跨代紧凑 KV 转码，异构集群在无需淘汰存量 Hopper 的前提下实现了 `$1.8742 / 1M Tokens` 的极具竞争力成本，仅比纯新建 GB200 集群高 `31%`，但节省了 `$635M` 的首期替换 CapEx。
