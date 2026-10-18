# AI Infra Research

**从实现机制到架构决策，而不是 GPU 规格表或每日新闻汇总。**

面向已有云/解决方案架构背景、希望深入 AI Infrastructure 的学习者。中文分析保留 English terminology，研究覆盖 GPU、HBM、互连/网络、存储、分布式训练、推理平台、调度运维以及容量/成本。优先深入 GPU/HBM/Network，再把组件连接到端到端工作负载。

## 阅读入口

| 入口 | 内容 |
|---|---|
| [研究框架](docs/framework.md) | 分层方法、组件深挖契约、来源和结论边界 |
| [全栈地图](docs/stack-map.md) | 11 个层面、技术栈、架构问题与研究进度 |
| [架构师学习路线](docs/learning/roadmap.md) | 云架构 → 单 GPU → 多 GPU → 多节点 → 平台架构 |
| [能力矩阵](docs/learning/competency-matrix.md) · [练习](docs/learning/exercises.md) | 可检查的知识/设计能力；不以文章数量推断掌握 |
| [GPU/HBM 深入研究](docs/components/gpu-hbm.md) | 执行模型、内存层级、容量账本、Roofline 与工作负载 |
| [互连与网络分层](docs/components/interconnect-network.md) | NVLink/NVSwitch、IB/RoCE/RDMA、NCCL 及固定源码分析 |
| [RDMA 接口生命周期](docs/components/rdma-lifecycle.md) | PD/MR/QP/CQ、WR/WC、缓冲重用、部分提交与错误边界 |
| [NCCL topology/transport](docs/components/nccl-topology-transport.md) | 固定选型、P2P/SHM/NET、GDR/proxy、IB 完成与错误路径 |
| [NCCL 图搜索与 Channel 映射](docs/components/nccl-graph-channel-mapping.md) | `ncclTopoCompute` 回溯搜索、`ringIntra/inter`、Rail Plane、PXN 与多通道反证 |
| [NCCL IB Port-Failover 与异步恢复不变量](docs/components/nccl-port-recovery-invariants.md) | `RDMA_READ` 完成位图探测、`activeQps[]` 选择性重传、有符号完成计数与 `IBV_QPT_UC` 端口恢复 |
| [张量 Layout/Dtype 与 SM100 Kernel 分派链路](docs/components/kernel-dispatch-layout.md) | ATen 步长置换、CuTe AOT C-ABI、TMA Swizzle、Tile-Ordered Partials 与 SM100 `tcgen05`/`TMEM` 寄存器重分配 |
| [数据加载与分布式 Checkpoint/恢复契约](docs/components/storage-checkpoint.md) | NVIDIA GDS (`cuFile`) DMA 路径、PyTorch DCP 两阶段协调、跨 Rank 去重、`BlockingAsyncStager` 与 Young-Daly 预算 |
| [推理调度、KV Cache 生命周期与 SLO/Goodput 边界](docs/components/inference-slo.md) | vLLM V1 统一 `token_budget` 调度、100% Prefix Cache 末块重算、`FreeKVCacheBlockQueue` 与 Chunked Prefill / P/D 分离 |
| [6D 分布式训练状态与通信联合账本](docs/components/training-parallelism.md) | TorchTitan `ParallelDims`、PyTorch FSDP2 双缓冲预取、`Shard(dim!=0)` 惩罚、TP+SP、CP Ring Attention 与 DeepEP v2 |
| [AI 集群拓扑感知调度、设备隔离与弹性恢复契约](docs/components/cluster-orchestration.md) | Kubelet Topology Manager、Kueue TAS、Slurm `topology/block & tree`、NVIDIA Device Plugin XID/MIG 与 `torchelastic` |
| [容量、成本与吞吐经济学](docs/components/capacity-economics.md) | TorchTitan 稠密/MoE 参数与 MFU 护栏、vLLM Gamma 突发排队与 P/D 分离 Goodput 定容、Slurm 梯形能耗积分與创新公司 I/O 边界 |
| [机架功率、液冷热力学与机柜级故障域](docs/components/rack-power-cooling.md) | 126kW 冷板液冷 vs 40kW 风冷 CDU/CFM 边界、Slurm `power_save.c` 令牌桶浪涌控制、RAPL/XCC 能耗积分与 DCGM `v4.6.1` MNNVL/IMEX 故障域 |
| [端到端 AI Infra 综合架构案例](docs/architectures/end-to-end-ai-infra-case.md) | 含工作负载数据流、Rail-Optimized 多轨与 126kW 液冷机柜拓扑、跨层故障自愈状态机三视图及全栈预算/反例/验收清单的综合架构方案 |
| [UEC 1.1 与 RoCEv2 多轨拥塞控制/喷洒路由对比](docs/components/uec-transport-congestion.md) | Linux `bnxt_re` 硬件门铃退避、NCCL 多 QP 喷洒、ECMP+DCQCN PFC 风暴反证与 UEC 风格选择性重传/信用控制模型 |
| [Prefill/Decode 分离与分层 KV 缓存路由契约](docs/components/disaggregated-kv-routing.md) | Mooncake 多轨 RDMA Write-Imm 切片与 `DeadlineMlu`、NVIDIA NIXL 非连续描述符、SGLang `DecodePrefixMatch` 与 HiCache 门控 |
| [大规模 MoE 专家并行与 DeepEP 双层去重/EPLB](docs/components/moe-expert-parallel-alltoall.md) | DeepEP `TokenLayout` 64B 对齐、`ptx::deduplicate` NVLink+RDMA 二级去重、`warpgroup_reg_realloc` 与在线 EPLB TMA 权重搬运 |
| [超长上下文混合注意力：FlashMLA、Ring/Ulysses CP 与 Mamba 检查点池](docs/components/long-context-hybrid-attention.md) | DeepSeek FlashMLA V3.2 (`656B`) / V4.1 (`528B` FP8, `288B` FP4)、Hopper 2-CTA DSM Crossover、Ring vs Ulysses CP 与 SGLang `Int8CheckpointStore` |
| [大规模推理强化学习引擎（RLVR / GRPO）：verl 权重重分片与 CuMem 休眠](docs/components/rl-post-training-engine.md) | `verl` `BucketedWeightSender` (`512 MiB` IPC/SHM)、`NCCLCheckpointEngine` `multi_sender`、Count-Only `DeltaShardedCheckpointEngine` 与 vLLM `CuMemAllocator` |
| [机密计算 GPU（CC-On / PPCIE）：SPDM/RIM 远程证明与 Bounce Buffer 边界](docs/components/confidential-gpu-tee.md) | `NVIDIA/nvtrust` Driver/VBIOS RIM 索引互斥与 `MSR 35` 豁免、8-GPU+4-NVSwitch 双向 PDI 拓扑门禁与 Bounce Buffer vs PPCIE 吞吐 |
| [共封装光学（CPO）、线性直驱（LPO）与光路交换（OCS）十万卡功耗/故障模型](docs/components/optical-interconnect-cpo-ocs.md) | SONiC `CpoManagerTask` (`OE`/`ELSFP` 兄弟端口级联关停)、NCCL RAS 28028 带外监控与十万卡 800G DSP/LPO/CPO/OCS 功耗与 Goodput |
| [DeepSeek 3FS (`hf3fs_usrbio`)、GPUDirect Storage (`cuFile`) 与多级 KV/Checkpoint 管线](docs/components/storage-gds-3fs-kv-tiering.md) | 3FS `/dev/shm` + `3fs-virt/iovs` 符号链接注册、CRAQ+BIBD 链式复制、LMCache `Hf3fsConnector` `memcpy` 瓶颈与 `nvidia-fs` BAR1 P2P / PCIe ACS 惩罚 |
| [Agentic RL 万级 MicroVM 沙箱（Firecracker UFFD）与 GPU KV Cache 协同调度](docs/components/agentic-sandbox-coscheduling.md) | Firecracker UFFD `-EAGAIN` 排空 `UFFD_EVENT_REMOVE`、`verl` `ToolAgentLoop` Token-in-Token-out、`2.56s`/`8.00s` Break-Even 与 35% 宿主机 I/O 令牌桶 |
| [FP8 细粒度分块缩放与 Blackwell NVFP4 预训练数值稳定性机制](docs/components/fp8-nvfp4-pretraining-numerics.md) | DeepSeek `1x128`/`128x128` + WGMMA 128 步晋升、`TransformerEngine` NVFP4 (`1x16 E2M1` + `16x16 RHT` + `cvt.rs` 随机舍入) 与离群通道 SQNR 账本 |
| [万卡同步训练容错与静默数据损坏（SDC）防御：NVRx 进程内热重启与 ABFT](docs/components/fault-tolerant-training-resilience.md) | `NVIDIA/nvidia-resiliency-ext` `inprocess.Wrapper` 热备切换 (`19.8s`)、`CliqueReplicationStrategy` 内存副本、`64K` 卡 `90.30%` ETTR 与正交 ABFT 单步拦截 |
| [132kW–600kW 液冷机柜毫秒级功率阶跃（`dP/dt`）与 OCP ORv3 超级电容削峰](docs/components/rack-power-transient-smoothing.md) | `DCGM` Field 1425-1438 固件功率平滑、`3,014.7–13,926.4 MW/s` 同步冲击、ORv3 `87.1 F` (`43.9 kJ`) 超级电容选型与 `264.91x` 斜率压缩 |
| [网内计算 SHARP 集合通信、NVLS 多播硬件归约与位级漂移边界](docs/components/in-network-collectives-sharp.md) | `NVIDIA/nccl` (`3e46c089`) `nvls.cc`/`collnet.cc` 代价模型、`16K` 卡 `NVLS + SHARP` `27.91x` 加速、`64` 组硬件树溢出回退与 `FP32` 累加消除 `BF16` 非结合律位级漂移 |
| [Hopper 与 Blackwell 跨代异构集群非对称流水线编排与跨代 KV 转码](docs/components/heterogeneous-cluster-scheduling.md) | `NVIDIA/Megatron-LM` (`b607c1b7`) `PipelineParallelLayerLayout` DSL、异构 `H100+B200` `1:3` 非对称 VPP 切分（气泡从 `43.87%` 压至 `0.28%`）与 `SM100 FP4 -> SM90 FP8` 紧凑 KV RDMA 转码 (`3.86x`) |
| [10 万卡 AI Factory 全栈 CapEx / OpEx / ETTR / Token 经济学帕累托模型](docs/components/ai-factory-tco-pareto.md) | 联立光互连、SHARP 集合通信、FP8/NVFP4、NVRx/3FS 与 ORv3 超级电容的十万卡 3 年 TCO 帕累托模型（有效算力产出提升 `7.36x`、每百万 Token 成本降低 `12.65x` 至 `$1.4307`） |
| [交互式 AI Infra 全栈架构师工作台 (H5)](docs/reports/ai-infra-architect-workbench.html) · [Phase 3](docs/reports/ai-infra-phase3-benchmark-cards.html) · [Phase 4](docs/reports/ai-infra-phase4-benchmark-cards.html) · [Phase 5](docs/reports/ai-infra-phase5-benchmark-cards.html) · [Phase 6 7-Card 报告](docs/reports/ai-infra-phase6-benchmark-cards.html) | 零外部依赖交互式 TCO/KV/SHARP 计算器与自包含 `Content-Security-Policy: default-src 'none'` 4:5 视觉卡片组，汇总全仓 33 个研究问题与 27 个可复现实验坐标 |
| [AI Infra 创新公司深挖](docs/components/ai-infra-innovators.md) · [v2 帕累托深挖](docs/components/ai-infra-innovators-v2.md) | 推理加速（Fireworks / Together / SGLang 自适应 EAGLE-3 / NVFP4 Swizzle）与新型 AI 芯片（Groq TSP / Cerebras WSE）帕累托前沿 |
| [技术趋势](docs/trends/index.md) | 证据支持的变化、成熟度与采用条件 |
| [每日研究](docs/daily/2026-10-19.md) | 最新有界研究与限制；按北京时间日期增量归档 |
| [原创离线实验](experiments/capacity_roofline/README.md) | 可重算的显存 payload 与 Roofline 理论界限 |
| [Ring cut 离线算例](experiments/ring_cut/README.md) | rank 顺序、跨节点 payload 与共享 NIC 理论下界 |
| [Rail channel cut 算例](experiments/rail_channel_cut/README.md) | 多 Channel、Multi-Rail 对齐、跨 Rail 偏斜与机内 `PATH_PHB` 瓶颈反证 |
| [IB failover invariants 算例](experiments/ib_failover_invariants/README.md) | NCCL 端口故障转移的 CQ/RQ 容量放大、选择性重传与有符号事件守恒验证 |
| [Innovator arch tradeoffs 算例](experiments/innovator_arch_tradeoffs/README.md) | Fireworks KV-Outer 字节账本、Together Sequoia 投机树 Roofline 与 Groq/Cerebras SRAM 边界 |
| [Checkpoint recovery 算例](experiments/checkpoint_recovery/README.md) | PyTorch DCP 跨 Rank 去重均衡、GDS vs POSIX 路径、异步 Stager 与 Young-Daly Goodput 仿真 |
| [Inference SLO scheduler 算例](experiments/inference_slo_scheduler/README.md) | vLLM V1 `token_budget` 调度、Prefix Cache 末块重算、Chunked Prefill 与 P/D 分离 TTFT/TPOT 权衡 |
| [Training parallelism 算例](experiments/training_parallelism/README.md) | 6D 并行显存/通信账本、FSDP2 `Shard(dim!=0)` 瞬态峰值、CP 因果倾斜与 DeepEP v2 去重 |
| [Cluster orchestration 算例](experiments/cluster_orchestration/README.md) | Kubelet NUMA/PCIe/RDMA 掩码准入、Slurm `topology/block` 反碎片与 Young-Daly/热备弹性训练 Goodput |
| [Capacity economics 算例](experiments/capacity_economics/README.md) | TorchTitan 稠密/MoE 训练成本、vLLM Gamma 突发排队与 P/D 分离 Goodput、Slurm 焦耳积分及四家创新公司吞吐/I/O 模型 |
| [Rack power & cooling 算例](experiments/rack_power_cooling/README.md) | 126kW NVL72 液冷 vs 40kW 风冷 vs PG25 退化热力学、Slurm 令牌桶浪涌限流与 RAPL 回绕、DCGM MNNVL/IMEX 故障状态机及 Groq/Cerebras 机柜画像 |
| [UEC fabric congestion 算例](experiments/uec_fabric_congestion/README.md) | ECMP+DCQCN PFC 队头阻塞、4-QP Packet Spraying 乱序窗口、RTT 梯度与 UEC 风格选择性重传尾延迟对比 |
| [Disaggregated KV routing 算例](experiments/disaggregated_kv_routing/README.md) | 多轨 RDMA Write-Imm KV 切片、小页门铃开销、慢轨动态摘除与 DRAM/NVMe HiCache 晋升排队模型 |
| [MoE EP All-to-All 算例](experiments/moe_ep_alltoall/README.md) | DeepEP `TokenLayout` 字节对齐、NVLink+RDMA 二级去重、Zipf 专家倾斜/冗余副本与 EPLB 在线权重搬运重叠 |
| [Innovator Pareto v2 算例](experiments/innovator_pareto_v2/README.md) | SGLang 自适应 EAGLE-3 步长控制、NVFP4 `128x4` vs `8x4` 缩放对齐/V-Scale 重排与 2026 四大推理流派帕累托前沿 |
| [Long-context hybrid attention 算例](experiments/long_context_attention/README.md) | `1M` 上下文 FlashMLA V4.1 (`528B`/`288B`) + DSA、Hopper 2-CTA DSM 反量化、Ring vs Ulysses CP 与 SGLang `Int8CheckpointStore` |
| [RL rollout resharding 算例](experiments/rl_rollout_resharding/README.md) | `verl` 5 种权重重分片通道（CUDA IPC / SHM / 单发与多发 NCCL / 稀疏 Delta）、vLLM `CuMemAllocator` 休眠级别与 Long-CoT 气泡消除 |
| [Confidential GPU overhead 算例](experiments/confidential_gpu_overhead/README.md) | `NVIDIA/nvtrust` RIM 索引互斥/`MSR 35` 豁免、PPCIE 8-GPU+4-NVSwitch 双向 PDI 门禁与 Bounce Buffer vs PPCIE 延迟模型 |
| [Optical CPO/OCS fabric 算例](experiments/optical_cpo_ocs_fabric/README.md) | 十万卡 800G 集群 DSP/LPO/CPO/OCS 功耗（`7.92 MW` vs `0.72 MW`）、`OE`/`ELSFP` 共享爆炸半径与同步训练 Goodput 模型 |
| [Storage GDS & 3FS KV/Checkpoint 算例](experiments/storage_gds_kv_checkpoint/README.md) | POSIX Page Cache vs `O_DIRECT` vs GDS BAR1 P2P vs 3FS `USRBIO` 在 `100 GiB` KV 恢复与 `1 TiB` Checkpoint 下的吞吐与 PCIe RC 争用 |
| [Agentic sandbox co-scheduler 算例](experiments/agentic_sandbox_coscheduler/README.md) | 万级 Firecracker UFFD 缺页风暴与 `verl` `ToolAgentLoop` GPU KV 驻留/重算/GDS 卸载（`2.56s`/`8.00s` 阈值与 35% I/O 令牌桶）协同调度 |
| [FP8 & NVFP4 numerics 算例](experiments/fp8_nvfp4_numerics/README.md) | `100x` 离群通道下 Delayed Scaling vs DeepSeek `1x128` vs NVFP4 `1x16 + RHT + SR` 的 SQNR、小梯度归零偏差与 WGMMA 晋升误差 |
| [Megascale fault tolerance 算例](experiments/megascale_fault_tolerance/README.md) | `4K–64K` GPU 规模下 Slurm+PFS vs `NVRx` 进程内热备 + 对等内存恢复的 ETTR（`18.58%` vs `90.30%`）与 BF16 比特翻转 ABFT 校验 |
| [Rack power transients 算例](experiments/rack_power_transients/README.md) | `132kW` NVL72 至 `600kW` Rubin Ultra 机柜毫秒级功率阶跃 `dP/dt`、ORv3 超级电容（`87.1 F` / `2675.3 F`）与固件平滑协同算例 |
| [SHARP in-network collectives 算例](experiments/sharp_in_network_collectives/README.md) | `64–16,384` 卡 `Ring` / `Tree` / `NVLS_TREE` / `NVLS + SHARP` AllReduce 带宽（`27.91x` 加速）、`64` 组硬件树饱和回退与 `BF16`/`FP32` 位级漂移 |
| [Heterogeneous cluster scheduler 算例](experiments/heterogeneous_cluster_scheduler/README.md) | `H100 + B200` 跨代异构 `PP=16, VPP=2` 非对称切分（`1.78x` 加速、`0.28%` 气泡）与 `128K` 跨代 `SM100 FP4 -> SM90 FP8` KV RDMA 转码 (`3.86x`) |
| [AI Factory TCO simulator 算例](experiments/ai_factory_tco_simulator/README.md) | 10 万卡 AI Factory 四代全栈配置（Gen-1 Legacy / Gen-2 Hopper / Gen-3 Het / Gen-3 Full Blackwell）3 年 TCO、有效算力天与 `$/1M Tokens` 帕累托仿真 |
| [合成容量架构练习](docs/architectures/synthetic-inference-budget.md) | 将显存账本、数据流和失败边界连接到方案设计 |
| [研究执行契约](docs/research-operations.md) | 研究 queue、跨运行续接、分支/PR、防重叠与失败恢复 |

## 证据状态

`announcement`、`specification`、`implementation`、`author-benchmark`、`own-offline`、`own-hardware`、`unverified` 分开登记。原文读过不等于完整规范审阅，源码静态分析不等于运行，离线计算不等于 GPU benchmark。

来源、主张、研究问题和进度分别保存在 [sources](research/sources.json)、[evidence](research/evidence.json)、[queue](research/queue.json)、[progress](research/progress.json)。用户学习掌握单独评估，当前未评估。

## 运行

Python 3.12 标准库；不需要 GPU 或第三方 Python 包。在仓库根目录运行：

```powershell
python -m unittest discover -s tests
python scripts\validate_research.py
python experiments\capacity_roofline\run.py --check
python experiments\ring_cut\run.py --check
python experiments\rail_channel_cut\run.py --check
python experiments\ib_failover_invariants\run.py --check
python experiments\innovator_arch_tradeoffs\run.py --check
python experiments\checkpoint_recovery\run.py --check
python experiments\inference_slo_scheduler\run.py --check
python experiments\training_parallelism\run.py --check
python experiments\cluster_orchestration\run.py --check
python experiments\capacity_economics\run.py --check
python experiments\rack_power_cooling\run.py --check
python experiments\uec_fabric_congestion\run.py --check
python experiments\disaggregated_kv_routing\run.py --check
python experiments\moe_ep_alltoall\run.py --check
python experiments\innovator_pareto_v2\run.py --check
python experiments\long_context_attention\run.py --check
python experiments\rl_rollout_resharding\run.py --check
python experiments\confidential_gpu_overhead\run.py --check
python experiments\optical_cpo_ocs_fabric\run.py --check
python experiments\storage_gds_kv_checkpoint\run.py --check
python experiments\agentic_sandbox_coscheduler\run.py --check
python experiments\fp8_nvfp4_numerics\run.py --check
python experiments\megascale_fault_tolerance\run.py --check
python experiments\rack_power_transients\run.py --check
python experiments\sharp_in_network_collectives\run.py --check
python experiments\heterogeneous_cluster_scheduler\run.py --check
python experiments\ai_factory_tco_simulator\run.py --check
python scripts\source_monitor.py
```

来源监控联网，只取得 allowlist 中的公开页面并保留哈希/状态，不归档第三方全文。失败返回非零；网页哈希变化不是新技术发布。结构验证也不能代替人工事实、版权与隐私审阅。

## 公开与实验边界

这是干净历史的公开研究仓库，只保存原创分析、公开来源链接和合成实验。研究通过分支/PR 更新，**不自动合并**；PR 在合并前也已经公开。首轮实质成果位于研究 PR，不应把初始化的 `main` 当成全部内容。

默认只做静态阅读和 CPU/offline 实验。GPU、付费云、模型/API 调用需单独批准；不自动安装驱动、不占用现有集群、不执行上游安装器/训练脚本。

每日研究沿用北京时间 22:30 的 Local Copilot 自动化。运行需要本机、App 与网络可用；不能保证关机、休眠或退出后继续。每次从远端有效研究分支恢复，不每天重建基线。

第三方材料保留原作者和原有权利，见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。未声明覆盖全仓库的开源许可证；公开可读不等于取得第三方再分发权。
Evidence-driven AI infrastructure research: GPU, HBM, interconnects, distributed systems, and solution architecture
