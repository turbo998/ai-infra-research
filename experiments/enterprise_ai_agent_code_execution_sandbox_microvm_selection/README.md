# Enterprise AI Agent Code Execution Sandbox (Docker vs. gVisor vs. Snapshot MicroVM) Evaluator 

Deterministic analytical evaluator comparing four enterprise sandboxing architectures for executing `3,000,000` untrusted LLM-generated Python/Bash scripts per month:

1. **Tier 1 (`tier1_standard_docker_runc_shared_kernel`)**: Standard Docker Container (`runc`, `450 ms` start + `import pandas`, `0.0%` kernel isolation [shares host Linux kernel — critical RCE escape risk], `$1,800/mo`).
2. **Tier 2 (`tier2_per_session_cold_boot_vm`)**: Per-Session Cold-Boot Cloud VM (`18,500 ms` [18.5s] cold boot + imports, `100.0%` hardware KVM isolation, `$34,000/mo`).
3. **Tier 3 (`tier3_gvisor_runsc_userspace_kernel`)**: User-Space Kernel Sandbox (`gVisor runsc`, `680 ms` start + imports, `92.0%` isolation, `28.0%` file I/O syscall overhead, `$4,600/mo` — best when cloud VMs lack nested `/dev/kvm`).
4. **Tier 4 (`tier4_firecracker_snapshot_microvm_pool`)**: Memory-Snapshot MicroVM Pool (`Firecracker / Kata Cloud-Hypervisor / Azure Container Apps Dynamic Sessions / E2B`, `115 ms` ready time with pre-imported Pandas in RAM snapshot [`160.9x` faster than cold VM and `3.9x` faster than cold Docker `import pandas`!], `100.0%` hardware KVM isolation, `1.5%` overhead, `$5,400/mo` [`-84.1%` vs. Tier 2]).

## Usage

```bash
python experiments/enterprise_ai_agent_code_execution_sandbox_microvm_selection/run.py
python experiments/enterprise_ai_agent_code_execution_sandbox_microvm_selection/run.py --check
```
