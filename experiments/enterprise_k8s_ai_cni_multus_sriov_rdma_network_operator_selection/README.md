# Enterprise Kubernetes AI Network Virtualization (`Multus + SR-IOV + NVIDIA Network Operator` vs. `hostNetwork` vs. `Overlay VXLAN`) (`#479`)

Reproducible analytical model comparing Standard `VXLAN` Overlay CNI (`4.2 GB/s`), `hostNetwork=true` (zero tenant isolation), `Multus + IPvlan + RDMA Shared Device Plugin`, and `NVIDIA Network Operator (Multus + SR-IOV VF + Topology Manager single-numa-node PIX Alignment)` (`47.6 GB/s` `busbw`, `1.8 us` latency).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_k8s_ai_cni_multus_sriov_rdma_network_operator_selection/run.py --check`
