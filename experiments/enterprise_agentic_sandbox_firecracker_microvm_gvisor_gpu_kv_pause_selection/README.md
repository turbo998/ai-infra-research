# Enterprise Coding Agent Sandbox (`Firecracker MicroVM` vs. `gVisor` vs. `Kata`) & GPU KV Cache Pause-on-Tool-Wait Sizing (`#471`)

Reproducible analytical model comparing shared-kernel `runc` Docker, `gVisor` (`runsc`), Cold-Boot `Kata/QEMU`, and `Firecracker MicroVM` `125ms` UFFD Copy-on-Write Snapshot Restoration paired with L2 Host DRAM KV Evict-on-Tool-Wait for multi-step Coding & Computer-Use Agents.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_agentic_sandbox_firecracker_microvm_gvisor_gpu_kv_pause_selection/run.py --check`
