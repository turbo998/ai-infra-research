# AI Infra 全栈地图

本表是研究范围及问题地图，不是全部完成清单。首轮完成 GPU/HBM 基础与互连分层；其余按 [queue](../research/queue.json) 逐题推进。

| 层 | 关键组件/技术栈 | 实现研究问题 | 架构产出/状态 |
|---|---|---|---|
| 0. 工作负载 | Dense/MoE、训练/微调/推理、张量、精度、质量/SLO | 计算量、状态、序列和并发怎样决定数据流？ | 首轮 [GPU/HBM](components/gpu-hbm.md)；不是全模型分析 |
| 1. 计算与封装 | GPU SM/CU、SIMT、Tensor Core、warp/wavefront、CPU/NUMA、chiplet/封装、TSP/WSE | 指令、调度、寄存器、共享内存与全 SRAM/晶圆级架构如何限制有效吞吐？ | GPU 执行模型基础已读；[Groq TSP 与 Cerebras WSE 架构对比](components/ai-infra-innovators.md)已完成 |
| 2. 内存 | SRAM/L1/L2、HBM、DDR、CXL、KV、量化 | 容量、带宽、访问合并、数据复用、spill 的关系？ | 基础与算术实验；HBM 标准/代际完整比较未完成 |
| 3. Scale-up | PCIe、NVLink/NVSwitch、P2P、coherence、开放互连 | 通信域、访问语义和拓扑怎样影响 TP/EP？ | [分层与历史结构样例](components/interconnect-network.md)；最新产品规格待核验 |
| 4. Scale-out | IB、Ethernet/RoCEv2、RDMA、NIC/HCA/DPU、verbs、GPUDirect RDMA、Clos/fat-tree/rail | 注册/队列/完成、超售、拥塞、路由怎样影响 collectives？ | [接口生命周期](components/rdma-lifecycle.md)、[NCCL 核心静态路径](components/nccl-topology-transport.md)、[图搜索/Channel 映射](components/nccl-graph-channel-mapping.md)与 [IB Port-Failover/异步恢复不变量](components/nccl-port-recovery-invariants.md)；硬件实测待授权 |
| 5. 存储 | NVMe、对象存储、Lustre、checkpoint、预取、GPUDirect Storage | 数据加载、持久化和恢复是否串行化训练？ | [GDS `cuFile` DMA 与 PyTorch DCP 分布式 Checkpoint/恢复契约](components/storage-checkpoint.md)及 [Checkpoint 恢复离线算例](../experiments/checkpoint_recovery/README.md)已完成；并行文件系统实机压测待授权 |
| 6. 执行软件 | CUDA/ROCm、compiler、cuBLAS/hipBLAS、CUTLASS/Triton/CuTe-DSL、CSL/GroqAPI | 形状/布局如何决定 dispatch、融合、精度与实际 kernel？ | [张量 Layout/Dtype 到 CuTe AOT 与 SM100 `tcgen05` 分派链路](components/kernel-dispatch-layout.md)及 [FA3 / Fireworks SM100 M3 / CSL / GroqAPI 源码分析](components/ai-infra-innovators.md)已完成 |
| 7. 训练系统 | DP/DDP、FSDP/ZeRO、TP/PP/CP/EP、NCCL/RCCL、Megatron/DeepSpeed | 状态分片、通信和拓扑如何联合设计？ | [TorchTitan/PyTorch 6D 并行、FSDP2、TP+SP、CP Ring Attention 与 DeepEP v2 联合账本](components/training-parallelism.md)及 [Training parallelism 算例](../experiments/training_parallelism/README.md)已完成；千卡实机训练评测待授权 |
| 8. 推理系统 | vLLM/SGLang/TensorRT-LLM、continuous batching、PagedAttention、speculation、P/D 分离、Fireworks/Together | TTFT、TPOT、质量、KV 传输与成本怎样权衡？ | [vLLM V1 统一调度、KV Cache 生命周期与 SLO/Goodput 边界](components/inference-slo.md)、[Inference SLO 离线算例](../experiments/inference_slo_scheduler/README.md)及 [Sequoia / StripedHyena / M3 分析](components/ai-infra-innovators.md)已完成 |
| 9. 调度与运维 | Kubernetes/Slurm、Operator/device plugin、GPU 分区、quota、DCGM/ROCm tools/Prometheus | 碎片、多租户、拓扑感知、兼容、故障/升级如何处理？ | Queue 待研究；控制面与运维 runbook |
| 10. 经济性与设施 | 容量/TCO、功率、机架密度、液冷、供电、故障域 | 何时更多 GPU 无效？SLO 如何映射成完整机架与资源预算？ | Queue 待研究；设计约束，不做施工/芯片制造认证 |

## 不可混淆的关系

HBM 是存储层；NVLink 是互连；NVSwitch 是交换实现；RDMA 是访问能力；IB 与 Ethernet/RoCE 是网络路径；NCCL/RCCL 是通信库。它们不是一个竞品列表。

Scale-up/scale-out 不严格等于机内/机外。跨节点 NVLink 域、机架级系统与开放 Ethernet scale-up 都需要按具体访问/通信语义分析。

## 贯穿的三张图

每个组件逐步补齐：工作负载数据流图、物理/逻辑拓扑图、失败/恢复图。三张图必须绑定假设，不能将厂商产品宣传图当成自己的部署证据。
