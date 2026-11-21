# Kubernetes DRA & GPU-RDMA-NUMA Topology-Aware Scheduling Selection Experiment (`K8s DRA` vs. `Device Plugin + Topology Manager` vs. `Kueue TAS / Volcano`)

Reproducible performance and cluster utilization model comparing 4 Kubernetes GPU+RDMA scheduling architectures across GPUDirect RDMA P2P bandwidth (`PIX` vs. `SYS/UPI`), PD KV transfer / AllReduce `P99` latency, multi-node Goodput, and fragmentation waste.

## Run
```bash
python experiments/kubernetes_dra_topology_aware_gpu_nic_numa_scheduling_kueue_volcano_selection/run.py
python experiments/kubernetes_dra_topology_aware_gpu_nic_numa_scheduling_kueue_volcano_selection/run.py --check
```
