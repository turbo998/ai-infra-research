# 技术趋势：证据与采用门槛

| 方向 | 为什么值得追踪 | 当前证据与下一步 |
|---|---|---|
| 开放 AI Ethernet 与 scale-up | 互连域、传输语义、兼容和拥塞可能影响 GPU 系统选择 | [UEC 1.1 发布分析](uec-1-1.md)；完整规范/产品/实测未验证 |
| HBM、封装与功率约束 | 容量、带宽、控制器和热设计需要联合看 | [GPU/HBM 基础](../components/gpu-hbm.md)；HBM 新代际标准正文缺口 |
| 低精度与 kernel 协同 | 减少 payload 不必然提供同质量/延迟收益 | 后续固定格式、kernel 支持、质量与真实容量账本 |
| MoE 通信与专家布局 | all-to-all、拥塞、负载不均可改变扩展性 | 先完成 NCCL/RDMA 实现与并行模型；不发布无证据排行榜 |
| Topology-aware transport 与注册路径 | 同机/跨机标签不足以解释数据移动，GDR 与 user-buffer 注册是不同条件 | [NCCL 固定实现](../components/nccl-topology-transport.md) 与 [共享 NIC cut 算例](../../experiments/ring_cut/README.md)；不是新发布/生产性能核验 |
| KV 管理与分离式推理 | 容量、cache、KV 传输和 SLO 需联合设计 | 离线容量已建模；runtime/网络/硬件复现未开始 |
| 资源效率与可靠性 | 有效工作成本、checkpoint/恢复、机架功率决定系统效率 | 纳入 queue 的平台/经济性/设施题，不以 GPU utilization 代替 |

这是问题驱动的技术观察表，不是“已掌握所有最新成果”。正文、规范、实现和测量成熟度分别登记。网页 hash 变化只产生监控事件，须人工/代理阅读相关正文才能更新趋势。

2026-10-13：深化已有 NCCL commit，没有新增经核验的发布结论。
既有来源监控 17 项中 4 个 body hash 变化，不能解释成四项新技术；
JEDEC 目录再次仅有壳，HBM 比较仍 blocked。UEC 1.1 已发布分析保留
announcement 边界，不由 NCCL 代码可读推断 UET/INC 实现支持。
