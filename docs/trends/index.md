# 技术趋势：证据与采用门槛

| 方向 | 为什么值得追踪 | 当前证据与下一步 |
|---|---|---|
| 开放 AI Ethernet 与 scale-up | 互连域、传输语义、兼容和拥塞可能影响 GPU 系统选择 | [UEC 1.1 发布分析](uec-1-1.md)；完整规范/产品/实测未验证 |
| HBM、封装与功率约束 | 容量、带宽、控制器和热设计需要联合看 | [GPU/HBM 基础](../components/gpu-hbm.md)；HBM 新代际标准正文缺口 |
| 低精度与 kernel 协同 | 减少 payload 不必然提供同质量/延迟收益 | 后续固定格式、kernel 支持、质量与真实容量账本 |
| MoE 通信与专家布局 | all-to-all、拥塞、负载不均可改变扩展性 | 先完成 NCCL/RDMA 实现与并行模型；不发布无证据排行榜 |
| Topology-aware transport 与注册路径 | 同机/跨机标签不足以解释数据移动，GDR 与 user-buffer 注册是不同条件 | [NCCL 固定实现](../components/nccl-topology-transport.md)、[图搜索与 Channel 映射](../components/nccl-graph-channel-mapping.md) 与 [Rail channel cut 算例](../../experiments/rail_channel_cut/README.md) |
| 推理加速平台（Fireworks / Together） | 块稀疏 Attention、Hopper 异步流水线、硬件感知推测解码树与常状态模型直接改变长序列与低延迟成本结构 | [AI Infra 创新公司深挖](../components/ai-infra-innovators.md)；已核验源码与论文，商业 API 与实机未重测 |
| 新型全 SRAM / 晶圆级芯片（Groq / Cerebras） | 抛弃片外 HBM 换取 `20–55 TiB/s` 片上 SRAM 流带宽或晶圆级单色 2D Mesh AllReduce，带来确定性低尾延迟与容量悬崖的硬取舍 | [AI Infra 创新公司深挖](../components/ai-infra-innovators.md)；已核验 TSP 论文、`groqflow` 多芯片切分与 `Cerebras/sdk-examples` CSL 源码 |
| KV 管理与分离式推理 | 容量、cache、KV 传输和 SLO 需联合设计 | 离线容量已建模；runtime/网络/硬件复现未开始 |
| 资源效率与可靠性 | 有效工作成本、checkpoint/恢复、机架功率决定系统效率 | 纳入 queue 的平台/经济性/设施题，不以 GPU utilization 代替 |

这是问题驱动的技术观察表，不是“已掌握所有最新成果”。正文、规范、实现和测量成熟度分别登记。网页 hash 变化只产生监控事件，须人工/代理阅读相关正文才能更新趋势。

2026-10-14：完成 `nccl-graph-channel-mapping` 静态代码深挖与 `rail_channel_cut`
多通道/多 Rail/机内瓶颈反证算例，并新增 AI Infra 创新公司专题（Fireworks AI、
Together AI、Groq TSP/LPU、Cerebras WSE2/WSE3）。JEDEC 目录复查仍仅返回搜索壳，
`hbm-standard-comparison` 保持 blocked。
