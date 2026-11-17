from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # Analytical model for Enterprise AI Agent Code Execution Sandbox (Docker vs. Cold VM vs. gVisor vs. Snapshot MicroVM) (Round 395)
    # Workload: 3,000,000 untrusted LLM-generated Python/Bash code executions per month (Data Analysis & SWE Agents).
    monthly_executions = 3_000_000

    # Tier 1: Standard Docker Container (runc sharing host Linux kernel)
    t1_cold_start_plus_import_ms = 450.0  # Container start (45ms) + cold `import pandas, scipy` (405ms)
    t1_kernel_isolation_score_pct = 0.0  # Shared host kernel -> High container escape & SSRF risk
    t1_io_compute_overhead_pct = 0.0
    t1_monthly_cost_usd = 1_800

    # Tier 2: Dedicated Cold-Boot Cloud VM per Session
    t2_cold_start_plus_import_ms = 18_500.0  # 18.5s OS boot + Python runtime initialization
    t2_kernel_isolation_score_pct = 100.0
    t2_io_compute_overhead_pct = 0.0
    t2_monthly_cost_usd = 34_000

    # Tier 3: User-Space Kernel Sandbox (gVisor runsc on Kubernetes)
    t3_cold_start_plus_import_ms = 680.0  # Slower Python library file-stat imports through Sentry syscall proxy
    t3_kernel_isolation_score_pct = 92.0  # User-space Go kernel intercepts syscalls
    t3_io_compute_overhead_pct = 28.0  # Heavy file I/O syscall interception overhead
    t3_monthly_cost_usd = 4_600

    # Tier 4: Pre-Warmed Memory-Snapshot MicroVM Pool (Firecracker / Cloud-Hypervisor / ACA Dynamic Sessions / E2B)
    t4_cold_start_plus_import_ms = 115.0  # Restores pre-imported Python+Pandas RAM snapshot in 115ms!
    t4_kernel_isolation_score_pct = 100.0  # Full hardware KVM virtualization + dedicated guest kernel
    t4_io_compute_overhead_pct = 1.5
    t4_monthly_cost_usd = 5_400

    startup_speedup_t4_vs_t2_x = round(t2_cold_start_plus_import_ms / t4_cold_start_plus_import_ms, 1)
    startup_speedup_t4_vs_t3_x = round(t3_cold_start_plus_import_ms / t4_cold_start_plus_import_ms, 2)
    cost_savings_t4_vs_t2_pct = round((t2_monthly_cost_usd - t4_monthly_cost_usd) / t2_monthly_cost_usd * 100.0, 1)

    tiers = [
        {
            "tier_id": "tier1_standard_docker_runc_shared_kernel",
            "name": "Tier 1: 普通 Docker 容器裸跑 (runc 共享宿主机 Linux 内核，高危逃逸与云元数据 SSRF 风险)",
            "ready_with_pandas_latency_ms": t1_cold_start_plus_import_ms,
            "kernel_isolation_score_pct": t1_kernel_isolation_score_pct,
            "syscall_io_overhead_pct": t1_io_compute_overhead_pct,
            "monthly_cost_3m_runs_usd": t1_monthly_cost_usd,
            "isolation_boundary": "Shared Host Kernel (CISO Security Blocker!)",
            "status": "Critical RCE Escape Risk (0% Kernel Isolation)",
        },
        {
            "tier_id": "tier2_per_session_cold_boot_vm",
            "name": "Tier 2: 每次会话冷启动独立云虚拟机 (100% 硬件虚拟化，但开机+导包慢达 18.5 秒)",
            "ready_with_pandas_latency_ms": t2_cold_start_plus_import_ms,
            "kernel_isolation_score_pct": t2_kernel_isolation_score_pct,
            "syscall_io_overhead_pct": t2_io_compute_overhead_pct,
            "monthly_cost_3m_runs_usd": t2_monthly_cost_usd,
            "isolation_boundary": "Hardware KVM / Hyper-V",
            "status": "Too Slow for Interactive Agents (18.5s)",
        },
        {
            "tier_id": "tier3_gvisor_runsc_userspace_kernel",
            "name": "Tier 3: 用户态内核拦截沙箱 (K8s gVisor runsc，免 KVM 嵌套虚拟化，但文件 I/O 损耗 28%)",
            "ready_with_pandas_latency_ms": t3_cold_start_plus_import_ms,
            "kernel_isolation_score_pct": t3_kernel_isolation_score_pct,
            "syscall_io_overhead_pct": t3_io_compute_overhead_pct,
            "monthly_cost_3m_runs_usd": t3_monthly_cost_usd,
            "isolation_boundary": "Go User-Space Sentry Kernel Proxy",
            "status": "Best When Cloud VMs Lack Nested /dev/kvm",
        },
        {
            "tier_id": "tier4_firecracker_snapshot_microvm_pool",
            "name": "Tier 4: 内存快照恢复微虚拟机池 (Firecracker MicroVM Snapshot / ACA Dynamic Sessions / E2B)",
            "ready_with_pandas_latency_ms": t4_cold_start_plus_import_ms,
            "kernel_isolation_score_pct": t4_kernel_isolation_score_pct,
            "syscall_io_overhead_pct": t4_io_compute_overhead_pct,
            "monthly_cost_3m_runs_usd": t4_monthly_cost_usd,
            "isolation_boundary": "Hardware KVM + Dedicated Guest Kernel + Zero-Egress VPC",
            "status": "Enterprise Code Interpreter Gold Standard",
        },
    ]

    return {
        "experiment_id": "enterprise_ai_agent_code_execution_sandbox_microvm_selection",
        "round": 395,
        "title": "Enterprise AI Agent Code Execution Sandbox (Docker vs. gVisor vs. Snapshot MicroVM) Evaluation",
        "provenance": "synthetic_analytical_model_calibrated_to_published_firecracker_snapshot_gvisor_and_e2b_benchmarks",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "monthly_code_executions": monthly_executions,
            "preloaded_python_stack": "Python 3.12 + Pandas + Polars + NumPy + Matplotlib",
        },
        "headline_findings": {
            "tier4_ready_latency_ms": t4_cold_start_plus_import_ms,
            "startup_speedup_t4_vs_t2_x": startup_speedup_t4_vs_t2_x,
            "startup_speedup_t4_vs_t3_x": startup_speedup_t4_vs_t3_x,
            "cost_savings_t4_vs_t2_pct": cost_savings_t4_vs_t2_pct,
            "tier4_kernel_isolation_pct": t4_kernel_isolation_score_pct,
        },
        "architecture_tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic results.json")
    args = parser.parse_args()

    results = build_results()
    out_path = Path(__file__).with_name("results.json")

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist; run without --check first.")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != results:
            raise SystemExit("results.json mismatch with deterministic build_results().")
        print("OK: enterprise_ai_agent_code_execution_sandbox_microvm_selection verified.")
        return

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
