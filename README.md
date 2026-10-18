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
| [Phase 2 & 3 7-Card 4:5 视觉基准报告](docs/reports/ai-infra-phase3-benchmark-cards.html) | 自包含 `Content-Security-Policy: default-src 'none'` 4:5 视觉卡片组，汇总 7 大前沿子系统实测/离线坐标与架构取舍 |
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
python scripts\source_monitor.py
```

来源监控联网，只取得 allowlist 中的公开页面并保留哈希/状态，不归档第三方全文。失败返回非零；网页哈希变化不是新技术发布。结构验证也不能代替人工事实、版权与隐私审阅。

## 公开与实验边界

这是干净历史的公开研究仓库，只保存原创分析、公开来源链接和合成实验。研究通过分支/PR 更新，**不自动合并**；PR 在合并前也已经公开。首轮实质成果位于研究 PR，不应把初始化的 `main` 当成全部内容。

默认只做静态阅读和 CPU/offline 实验。GPU、付费云、模型/API 调用需单独批准；不自动安装驱动、不占用现有集群、不执行上游安装器/训练脚本。

每日研究沿用北京时间 22:30 的 Local Copilot 自动化。运行需要本机、App 与网络可用；不能保证关机、休眠或退出后继续。每次从远端有效研究分支恢复，不每天重建基线。

第三方材料保留原作者和原有权利，见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。未声明覆盖全仓库的开源许可证；公开可读不等于取得第三方再分发权。
Evidence-driven AI infrastructure research: GPU, HBM, interconnects, distributed systems, and solution architecture
