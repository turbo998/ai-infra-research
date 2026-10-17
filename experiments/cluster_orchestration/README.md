# 集群调度、拓扑感知与弹性容错离线验证（`cluster_orchestration`）

状态：已验证的 `own-offline` 标准库算例（基于固定版本 Kubernetes `v1.37.1`、Kueue `v0.20.2`、Slurm `slurm-26-05-5-1`、NVIDIA `k8s-device-plugin` `v0.20.1` 与 PyTorch `v2.6.0` `torchelastic` 源码不变量）。
相关正文：[集群调度、拓扑感知与多租户运维深挖](../../docs/components/cluster-orchestration.md)。

## 1. 模型覆盖的源码不变量

1. **Kubernetes Kubelet Topology Manager 位掩码交集与准入策略（`pkg/kubelet/cm/topologymanager/policy.go`、`policy_single_numa_node.go`、`policy_restricted.go`）**：
   - 合并 CPU、GPU、RDMA NIC 三类 HintProvider 的 `NUMANodeAffinity` 位掩码（`merged_mask = default_affinity & cpu_mask & gpu_mask & rdma_mask`）；
   - 仅当所有 Provider 掩码完全相等（`all_provider_masks_equal`）、交集非空且所有 `Preferred == true` 时，`merged_preferred = true`；
   - `single-numa-node` 要求 `popcount(merged_mask) == 1` 且 `merged_preferred == true`，否则拒绝 Pod（`TopologyAffinityError`）；`restricted` 允许跨多 NUMA 的 `Preferred` 分配（如整机 8-GPU 申请跨 2 个 NUMA 节点）但拒绝 `merged_preferred == false`。
2. **Slurm `topology/block` 聚合块取整与 `_choose_best_bblock` 反碎片选择（`src/plugins/topology/block/eval_nodes_block.c`）**：
   - 根据 `bblock_node_cnt` 计算所需基础块数 $B_{\text{req}} = \lceil N_{\text{req}} / \text{bblock\_node\_cnt} \rceil$ 与原始层级 $L_{\text{raw}} = \lceil \log_2(B_{\text{req}}) \rceil$，再向上吸附到配置的 `block_levels` 允许集合；
   - 校验 `--segment` 整除与单基础块容纳不变量（`requested_nodes % segment_size == 0` 且 `segment_size <= bblock_node_cnt`）；
   - 复现 `_choose_best_bblock`：当存在可单块满足的 `bblock` 时优先选择可用节点数**最少**（最紧致）的 `bblock` 以保护完整空闲域；当单块均不足时优先选择可用节点数**最多**的 `bblock` 以减少跨越的 `bblock` 数量。
3. **NVIDIA `k8s-device-plugin` NVML XID 过滤与 MIG 故障爆炸半径（`internal/rm/health.go`）**：
   - 默认忽略应用级 XID `{13, 31, 43, 45, 68, 109}`（不会摘除 GPU）；
   - 对非忽略的 Critical XID，若 `nvmlEventData.gpuInstanceId != 0xFFFFFFFF` 且 `computeInstanceId != 0xFFFFFFFF`（例如单 MIG 实例触发 XID 79/48 子上下文故障），仅将该 `<parentUUID, gi, ci>` MIG 切片标记为 `Unhealthy`；若为 `0xFFFFFFFF`（整卡级 ECC / 总线故障），则整张物理 GPU 下所有 $M$ 个 MIG 切片全部失效。
4. **PyTorch `torchelastic` 同步重排（Gang Restart）与热备节点（Hot Standby）有效训练吞吐（Goodput）**：
   - 计算 $N$ 节点集群聚合故障间隔 $\text{MTBF}_{\text{cluster}} = \text{MTBF}_{\text{node}} / N$ 与 Young-Daly 最优周期 Checkpoint 间隔 $\tau_{\text{opt}} = \sqrt{2 T_{\text{ckpt}} \cdot \text{MTBF}_{\text{cluster}}}$；
   - 对比冷节点替换重拉起（`cold_node_reprovision_sec + rdzv_rejoin_sec + 0.5 * checkpoint_interval_sec`）与热备池被 `rdzv_handler.num_nodes_waiting()` 纳管时的 Goodput 差异，并扣除热备节点自身占用的算力分母计算单配置节点净效率（`effective_goodput_per_provisioned_node_hot`）。

## 2. 运行与自检

```powershell
python experiments\cluster_orchestration\run.py --check
python -m unittest tests\test_cluster_orchestration.py
```

## 3. 严格限制声明

本实验为 `own-offline` 纯标准库解析与状态机验证，未连接真实 Kubernetes/Slurm 集群或物理 GPU/NVML 驱动，不代表生产集群实测基准。
