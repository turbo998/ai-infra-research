# 云架构师 → AI Infra：按先修进阶

起点是熟悉云与方案架构，但 GPU/CUDA/RDMA/分布式训练需要系统补课。没有日历承诺；一个阶段以解释、计算和设计证据结束，不以读完页数结束。

```mermaid
flowchart LR
    A[张量/attention/训练与推理] --> B[GPU 执行与内存]
    B --> C[容量/KV/Roofline]
    C --> D[多 GPU/P2P/collectives]
    D --> E[RDMA/IB/RoCE/拓扑]
    D --> F[分片/并行与推理调度]
    E --> G[多节点性能与恢复]
    F --> G
    G --> H[平台架构/SLO/TCO]
    H --> I[论文复核与前沿采用决策]
```

| 阶段 | 先修与学习 | 验收 |
|---|---|---|
| A. 工作负载语言 | 矩阵形状、attention、KV、前向/反向、精度与质量 | 解释 prefill/decode 为什么是不同资源问题，画张量/状态路径 |
| B. 单 GPU | SM/warp、kernel、寄存器、shared memory、cache/HBM、coalescing、spill、AOT/TMA/`tcgen05` 分派、Warp-Specialized Ping-Pong & TMEM | 用 [GPU/HBM 报告](../components/gpu-hbm.md)、[Kernel 分派链路](../components/kernel-dispatch-layout.md) 与 [Warp-Specialized 算子与 TMEM 工程](../components/warp-specialized-kernel-engineering.md) 找出“local memory 是片上内存”等误解并解释 64K 寄存器重分配（`24/240 regs`）与 SM100 `256 KB` TMEM |
| C. 容量与带宽 | 权重/激活/KV/优化器/工作区、GB/GiB、Roofline | 独立复算 [离线实验](../../experiments/capacity_roofline/README.md) 与 [Innovator 权衡算例](../../experiments/innovator_arch_tradeoffs/README.md)，列出它不能证明什么 |
| D. 多 GPU | P2P/NVLink/PCIe、AllReduce/AllGather/ReduceScatter、TP/DP/PP/EP | 从 [分层报告](../components/interconnect-network.md) 推导 Ring 通信量并说明适用条件 |
| E. 多节点 | MR/PD/QP/CQ、verbs、IB/RoCE、GPUDirect、拓扑/拥塞、Port-Failover、UEC/喷洒路由 | 阅读 [NCCL 固定路径](../components/nccl-topology-transport.md)、[图搜索/Channel 映射](../components/nccl-graph-channel-mapping.md)、[IB Port-Failover 不变量](../components/nccl-port-recovery-invariants.md)与 [UEC 1.1 / RoCEv2 多轨拥塞与喷洒路由](../components/uec-transport-congestion.md)，独立复算 [Ring cut](../../experiments/ring_cut/README.md)、[Rail channel cut](../../experiments/rail_channel_cut/README.md)、[IB failover invariants](../../experiments/ib_failover_invariants/README.md) 与 [UEC fabric congestion](../../experiments/uec_fabric_congestion/README.md)，设计区分回退/拥塞/负载不均的诊断证据 |
| F. 训练与推理平台 | FSDP2 (`DTensor`)/`SymmetricMemory` `Async-TP`、PyTorch `CUDASymmetricMemory` 无锁信号垫 & NVLS `multimem` (`8 CTAs`, `acc::f32`)、2D `USP = Ulysses × ZigZag Ring`、算子级 `SAC` + NUMA PCIe Gen5 卸载、NCCL 零拷贝注册 (`CUMEM` NVLS)、NCCL 并行聚合二项式树 (`PAT`: `PatRSAlgorithm` / `PatAGAlgorithm` + Multi-RPN `NVLS`)、GPU 直发 RDMA (`IBGDA` / `NCCL_GIN_TYPE_GDAKI` / `DeepEP V2.5` CTA 独占 QP 门铃聚合)、UEC 1.0 数据包裁剪 (`Packet Trimming`) 与 SONiC 动态 Alpha 缓冲池、Zero-Bubble (`ZB-V`)/`DualPipe`、SHARP 网内归约、异构 VPP、FP8/NVFP4 数值稳定性、SM100 块缩放微格式 (`Sm1xxBlockScaledBasicChunk` 512B TMEM 原子 & `LinCombBlockScaleFactor` 融合 Epilogue)、CUTLASS SM100 Ptr-Array / Grouped GEMM 异步 `TensorMapUpdater` & `CLC` 动态窃取、批大小不变确定性推理算子 (`Batch-Invariant Kernels`)、64K NVRx 容错/ABFT、16K 卡灰故障 (`Grey Failure`) 双端遥测检测、PyTorch `FlightRecorder` + NCCL `RAS` 静默死锁 `< 650 ms` 零号病人定位、DCGM HBM3e Row Remapping 预测性排空、Vera Rubin `VR200 NVL144` / `HBM4` / `NVLink-6` 屋脊线、MoE/DeepEP、Megatron-Core MoE 免辅助损失路由 (`expert_bias` 整数符号更新 & `qb_dual_update` vs 容量因子丢弃)、Wide-EP 解耦 `Attention-DP × Expert-EP` (`SUM_LEN` FP8 `AllGatherv` & `TBO`)、RLVR/verl、有界陈旧度异步 RL (`THUDM/slime` `AsyncRolloutWorker` + Partial Rollout 逐 Token `TIS`)、Agentic 沙箱协同、1M Hybrid Attention、FlashInfer 多级级联分页注意力 (`MultiLevelCascadeAttentionWrapper` Base-2 LSE & PDL)、vLLM V1 KV Cache 量化 (`FP8_PER_TOKEN_HEAD`、`INT4` + Walsh-Hadamard `single_rht` + FP32 尾数零点隐写术、`KIVI`、`NVFP4`)、动态 Token 预算 `Chunked Prefill`、DiT 视频生成 (`xDiT` + `TeaCache`)、`torch.compile` (`TorchInductor` / `FxGraphCache` / `MEGA_AOT`) 冷启动消除、Piecewise CUDA Graph、`XGrammar` 结构化受限解码位掩码、多模态 EPD、EAGLE-3 树投机、PD 解耦 EAGLE-3/MTP 零停顿交接、晶圆级/片上 SRAM vs GPU HBM3e 异构投机级联 (`SuffixDecoding` / `DFlash` / `SRAM-Draft + HBM-Verifier`)、四级 KV 缓存层次 (`HBM3e->DDR5->CXL/Mooncake->NVMe GDS`)、Multi-LoRA (`S-LoRA`/Punica `SGMV`)、batching/KV/PD分离、3FS/GDS 存储、CPO/OCS 光学互连、调度/机密计算 TEE、机架液冷/毫秒级功率平滑与 100K-GPU TCO | 结合全部 66 个研究课题文档及 [Phase 3](../reports/ai-infra-phase3-benchmark-cards.html) / [Phase 4](../reports/ai-infra-phase4-benchmark-cards.html) / [Phase 5](../reports/ai-infra-phase5-benchmark-cards.html) / [Phase 6](../reports/ai-infra-phase6-benchmark-cards.html) / [Phase 7](../reports/ai-infra-phase7-benchmark-cards.html) / [Phase 8](../reports/ai-infra-phase8-benchmark-cards.html) / [Phase 9](../reports/ai-infra-phase9-benchmark-cards.html) / [Phase 10](../reports/ai-infra-phase10-benchmark-cards.html) / [Phase 11](../reports/ai-infra-phase11-benchmark-cards.html) / [Phase 12](../reports/ai-infra-phase12-benchmark-cards.html) / [Phase 13](../reports/ai-infra-phase13-benchmark-cards.html) / [Phase 14](../reports/ai-infra-phase14-benchmark-cards.html) / [Phase 15](../reports/ai-infra-phase15-benchmark-cards.html) / [Phase 16](../reports/ai-infra-phase16-benchmark-cards.html) / [Phase 17 7-Card 4:5 视觉基准报告](../reports/ai-infra-phase17-benchmark-cards.html) 与 [交互式全栈架构师工作台](../reports/ai-infra-architect-workbench.html)，独立复算全部 61 个离线算例，设计有 SLO/质量门槛、存储层次取舍、通信域、液冷/功率边界、机密证明门禁、故障域和恢复步骤的系统 |
| G. 架构专家能力 | 真实 profiling/故障演练、成本/功率、论文质疑 | 带证据评审方案，能提出反例和最小可证伪实验 |

D/E 需要 C；训练与推理可以在基础完成后作为分支深化，不能因为熟悉 Kubernetes 就跳过数据流/通信。

## 学习循环

先独立回答 [练习](exercises.md)，再核对答案；把解释、错误原因、假设和下一个问题记录在个人学习反馈中。代理默认只更新研究进度，不替用户填“已掌握”。

没有 GPU 时允许完成算术、模型和静态代码路径；真实 GPU profiling、RDMA 调优、多节点故障恢复保持未验证。后续实践需另外确认环境与预算。

## 云架构经验怎样迁移，而不是从零重学

| 已有云架构能力 | 可以复用 | AI Infra 必须补充的约束 |
|---|---|---|
| 容量与弹性 | 需求预测、余量、伸缩、成本模型 | 权重/KV/训练状态驻留、分片、模型加载、GPU 碎片；不是增加 stateless 副本即可 |
| 网络设计 | 故障域、隔离、路由、带宽预算 | Collective/All-to-all、rank/rail、跨卡/节点拓扑、RDMA 完成与拥塞 |
| Kubernetes/平台 | 声明式运维、配额、升级、控制面 | GPU/driver/runtime 兼容、拓扑感知、gang/联合调度、设备分区与通信隔离 |
| 可用性设计 | SLO、冗余、恢复、演练 | 一次参与 rank 失败可能影响整个作业；checkpoint、一致恢复与重调度成本 |
| 服务性能 | p95/p99、排队、负载测试 | TTFT/TPOT、输入/输出分布、质量门槛、prefill/decode/KV 与 SLO-goodput |
| 成本优化 | TCO、利用率、保留容量 | 有效 token/训练进展、故障/恢复分母、功率/冷却；GPU 忙不等于业务效率 |
| 技术评审 | 需求、约束、取舍、风险 | 固定源码/规范、性能模型、实验反证；不能由厂商峰值替代 workload 证据 |

先用熟悉的架构方法定义目标，再向下追实现；不把全部时间花在厂商 SKU 记忆，也不把算法理论和系统运维割裂。

## 阅读来源顺序

工作负载基础 → CUDA 官方执行/内存契约 → ROCm 的对应模型/兼容说明 → NCCL collectives 与固定代码 → rdma-core verbs/生命周期 → 指定训练/推理 runtime 实现 → 系统论文及作者制品。

来源入口与已读范围见 [台账](../../research/sources.json)，不要把可变 `latest` 页面当不可变规范，也不要把 GPU 代际营销数字当设计参数。

## 怎样积累前沿研究能力

将 MLSys、SC、SIGCOMM、ISCA/ASPLOS、OSDI/SOSP、NeurIPS 等论文入口作为发现渠道，按实际工作负载选择文章，不以会议名代替证据。arXiv 版本、正式论文和作者制品分别锁定，不自动视为同行评审或可运行。

一篇论文读完要留下：它解决哪个瓶颈、旧方案/基线是什么、机制在哪里实现、实验分母与质量/拓扑怎样固定、哪些结果不可外推、什么反例会推翻结论。读一篇后形成自己的问题或小实验，而不是只有翻译摘要。

每当几个相互关联的问题完成，重做一份架构方案并比较“哪些选择因新增证据改变”。能解释改变与不改变的原因，才是知识向架构判断的积累。
