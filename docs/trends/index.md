# 技术趋势：证据与采用门槛

| 方向 | 为什么值得追踪 | 当前证据与下一步 |
|---|---|---|
| 开放 AI Ethernet 与 scale-up | 互连域、传输语义、兼容和拥塞可能影响 GPU 系统选择 | [UEC 1.1 发布分析](uec-1-1.md)；完整规范/产品/实测未验证 |
| HBM、封装与功率约束 | 容量、带宽、控制器和热设计需要联合看 | [GPU/HBM 基础](../components/gpu-hbm.md)；HBM 新代际标准正文缺口 |
| 低精度与 kernel 协同 | 减少 payload 不必然提供同质量/延迟收益 | [张量 Layout/Dtype 到 CuTe AOT 与 SM100 `tcgen05` 分派链路](../components/kernel-dispatch-layout.md) 与 [Innovator arch tradeoffs 算例](../../experiments/innovator_arch_tradeoffs/README.md) |
| MoE 通信与专家布局 | all-to-all、拥塞、负载不均可改变扩展性 | 先完成 NCCL/RDMA 实现与并行模型；不发布无证据排行榜 |
| Topology-aware transport 与注册路径 | 同机/跨机标签不足以解释数据移动，GDR 与 user-buffer 注册是不同条件 | [NCCL 固定实现](../components/nccl-topology-transport.md)、[图搜索与 Channel 映射](../components/nccl-graph-channel-mapping.md)、[IB Port-Failover 与异步恢复不变量](../components/nccl-port-recovery-invariants.md) 与 [IB failover invariants 算例](../../experiments/ib_failover_invariants/README.md) |
| 推理加速平台（Fireworks / Together） | 块稀疏 Attention、Hopper/Blackwell 异步流水线、硬件感知推测解码树与常状态模型直接改变长序列与低延迟成本结构 | [AI Infra 创新公司深挖](../components/ai-infra-innovators.md)（含 M3 反向索引自适应副本/零 D2H 二分调度/延迟归一化合并源码）、[SM100 分派链路](../components/kernel-dispatch-layout.md) 与 [Innovator arch tradeoffs 算例](../../experiments/innovator_arch_tradeoffs/README.md) |
| 新型全 SRAM / 晶圆级芯片（Groq / Cerebras） | 抛弃片外 HBM 换取 `20–55 TiB/s` 片上 SRAM 流带宽或晶圆级单色 2D Mesh AllReduce，带来确定性低尾延迟与容量悬崖的硬取舍 | [AI Infra 创新公司深挖](../components/ai-infra-innovators.md)；已核验 TSP 论文、`groqflow` 多芯片切分与 `Cerebras/sdk-examples`（AllReduce 与 2D Mesh GEMV 集体原语）CSL 源码 |
| KV 管理与分离式推理 | 容量、cache、KV 传输和 SLO 需联合设计 | [vLLM V1 统一调度、KV Cache 生命周期与 SLO/Goodput 边界](../components/inference-slo.md) 与 [Inference SLO 离线算例](../../experiments/inference_slo_scheduler/README.md) 已完成；线上集群压测待授权 |
| 资源效率与可靠性 | 有效工作成本、checkpoint/恢复、机架功率决定系统效率 | [GDS `cuFile` 与 PyTorch DCP 分布式 Checkpoint/恢复契约](../components/storage-checkpoint.md) 及 [Checkpoint 恢复离线算例](../../experiments/checkpoint_recovery/README.md) 已完成；继续推进 `training-parallelism` 与平台/经济性/设施题 |

这是问题驱动的技术观察表，不是“已掌握所有最新成果”。正文、规范、实现和测量成熟度分别登记。网页 hash 变化只产生监控事件，须人工/代理阅读相关正文才能更新趋势。

2026-10-14：完成 `nccl-graph-channel-mapping`、`nccl-port-recovery-invariants` 与
`kernel-dispatch-layout` 三项静态代码深挖，新增 `rail_channel_cut`、`ib_failover_invariants` 与
`innovator_arch_tradeoffs` 三套标准库离线算例，并完成 AI Infra 创新公司专题（Fireworks AI、
Together AI、Groq TSP/LPU、Cerebras WSE2/WSE3）及其 B200/B300 分析字节账本复现。JEDEC 目录复查仍仅返回搜索壳，
`hbm-standard-comparison` 保持 blocked。

2026-10-16：完成 `storage-checkpoint`（NVIDIA GDS Release 13.2 `cuFile` DMA 契约与 PyTorch v2.6.0
Distributed Checkpoint 两阶段保存、跨 Rank 去重均衡、`BlockingAsyncStager`、`_OverlappingCpuLoader`
退回边界）与 `inference-slo`（vLLM V1 v0.8.2 统一 `token_budget` 调度、100% Prefix Cache 末块重算、
`num_preallocate_tokens=64` 与 Chunked Prefill / P/D 分离权衡），新增 `checkpoint_recovery` 与
`inference_slo_scheduler` 两套原创标准库离线算例，并深化 Fireworks M3 辅助 CuTe-DSL 管线与 Cerebras 2D
Mesh GEMV 集体原语源码分析。`hopper-architecture`、`uec-1-1`、`nvlink-overview` 出现动态页面哈希变化
（仅记录监控信号，不视为新规范发布）；`jedec-directory` 返回 `http_403`（覆盖不完整，保留旧基线与 blocked 状态）。
