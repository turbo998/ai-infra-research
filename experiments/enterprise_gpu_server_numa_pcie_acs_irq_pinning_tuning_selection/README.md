# Enterprise 8-GPU Server CPU/NUMA Topology, `PCIe ACS Disable` & `IRQ/Tokenizer Core Pinning` Selection (`#506`)

Reproducible analytical model comparing 4 host BIOS/OS/K8s topology tuning profiles on an 8-GPU `HGX H200 / B200` server:
- **Tier 1 (`Factory Default BIOS: PCIe ACS Enabled + Unpinned NUMA`)**: Forces GPUDirect RDMA traffic through the CPU Root Complex (`14.2 GB/s` vs. `47.8 GB/s`), inflating `P99 TPOT` to `46.5ms` and capping server throughput at `72.0%`.
- **Tier 2 (`PCIe ACS Disabled Only`)**: Restores RDMA to `44.5 GB/s`, but cross-socket NUMA memory/thread bouncing leaves `P99 TPOT` at `31.0ms` (`86.5%` throughput).
- **Tier 3 (`ACS Disabled + K8s TopologyManager single-numa-node`)**: `47.2 GB/s` RDMA, `21.5ms P99 TPOT`, `95.2%` throughput.
- **Tier 4 (`ACS Disabled + single-numa-node + NIC IRQ Core Isolation + Async Tokenizer Pinning`)**: Reaches **`47.8 GB/s` GPUDirect RDMA (`3.37x` faster)**, cuts `P99 TPOT` by **`61.7%` (`17.8ms`)**, and unlocks **`100.0%` (`+38.9%` net gain)** of rated 8-GPU server performance at `$0` hardware cost.

## Run
```bash
python experiments/enterprise_gpu_server_numa_pcie_acs_irq_pinning_tuning_selection/run.py
```
