# 分布式训练与互连：TE 设备端 Grouped-Split 对齐校验、NCCL AllGather A2A 调优与独立网络后端 XML 拓扑（Phase 178）

## 1. 核心架构问题与源码证据

Phase 178（`priority: 187`）审计了 TransformerEngine、NCCL、Megatron-LM 与 Mooncake 的 9 项上游提交：
- **TransformerEngine 设备端 Grouped-Split 对齐校验与 MoE 架构指南**：`6f6b8f7680bc` (#3745) 在设备端验证 grouped split 切分对齐，避免在 Host 侧同步 tensor 元数据引发流水线气泡；`5f3abe14458c` (#3772) 在不支持的架构上跳过 FA 2.8.4 测试；`a0a7e920ccf7` (#3494) 发布官方 Mixture of Experts 训练指南。
- **NCCL AllGather A2A 模型选择调优与独立网络后端 XML 拓扑节点**：`9ef5381a1b51` 改进 AllGather All-to-All (A2A) 算法的代价模型选择（`1.17x` 带宽增益）；`ff6d58bb2975` 在拓扑探测中为不同网络后端创建独立的 XML 节点，防止多网卡/多传输插件混用时发生拓扑树覆盖。
- **Mooncake NPU 预发布 Wheel 构建与 Megatron-LM 全量单测回退**：Mooncake `4c78bb6cc7e0` (#4845) 在 pre-release 管线中构建 NPU wheels；Megatron-LM `6463d38242b8` (#8614) 在非 main 分支回退至全量单元测试保障覆盖率（同时审计 `babb33e40bdd` 与 `2cfbc2ca2b81`）。

## 2. 离线确定性实验验证

通过 `experiments/te_grouped_split_device_align_nccl_allgather_a2a_topo_phase178/run.py --check` 验证：
- `te_grouped_split_device_side_alignment_verified = True`
- `nccl_allgather_a2a_tuning_bandwidth_gain = 1.17`
- `nccl_topo_separate_net_backend_xml_nodes = True`
