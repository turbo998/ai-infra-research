# Agentic RL 10,000-MicroVM UFFD & GPU KV Cache Co-Scheduling Model

This offline stdlib analytical model (`evidence_class: own-offline`) evaluates the GPU HBM `GiB·s` cost and host PCIe/NVMe bandwidth contention when `10,000+` concurrent SWE/Coding Agent RL rollouts (`verl` `ToolAgentLoop` + Firecracker MicroVM `userfaultfd` snapshot resume) execute external tool calls lasting `1.0–60.0 s`:

1. **Break-Even Thresholds (`32,768` context tokens, `4.0 GiB` KV footprint)**:
   - Pinning KV in HBM is optimal only for short tool calls (`t_tool < 2.56 s`).
   - GPUDirect Storage / 3FS KV offload breaks even at `t_tool >= 2.56 s` (`0.160 s` round-trip swap vs `2.56 s` normalized HBM hold).
   - Evict-and-recompute breaks even vs HBM pinning at `t_tool >= 8.00 s` (`0.500 s` prefill recompute).
2. **Firecracker UFFD Page-Fault Storm Contention**:
   - At `500 VMs/s` snapshot resume (`96 MiB` active working set faulted via `UFFDIO_COPY`), unthrottled UFFD consumes `50.33 GB/s` of the `54.40 GB/s` host NVMe/PCIe pool, inflating GDS KV swap latency from `0.160 s` to `2.139 s` (`13.4x` degradation).
   - Token-bucket co-scheduling (capping UFFD ingress at `35%` of shared NVMe bandwidth during KV swap bursts) keeps GDS swap latency at `0.246 s` and increases effective active rollouts per 8-GPU node by `13.23x` (at `30 s` tool duration) to `25.58x` (at `60 s` tool duration).

## Usage

```bash
python experiments/agentic_sandbox_coscheduler/run.py
python experiments/agentic_sandbox_coscheduler/run.py --check
```
