from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "cloud_browser_fleet_playwright_cdp_pool_firecracker_snapshot_isolation_selection",
        "title": "Enterprise Cloud Browser Fleet & CDP Pool Selection: Firecracker CoW Snapshot MicroVMs vs. Shared BrowserContexts & Monolithic Containers",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "concurrent_browser_sessions": 1000,
            "avg_session_duration_min": 15,
            "spa_complexity": "Heavy Enterprise React/Canvas ERP & Multi-Tab Research",
        },
        "tiers": [
            {
                "tier": "Tier 1: Monolithic Cold Chromium per Container (Default 64MB /dev/shm)",
                "browser_ready_startup_ms": 3500.0,
                "memory_gb_per_1000_sessions": 1800.0,
                "session_crash_or_oom_rate_pct": 14.5,
                "cross_tenant_kernel_isolation": False,
                "monthly_fleet_infra_cost_usd": 42000,
                "recommended_scenario": "Anti-pattern (64MB /dev/shm crashes tabs; zombie renderers leak 1.8GB/session)",
            },
            {
                "tier": "Tier 2: Shared Chromium Process with 25 Playwright BrowserContexts",
                "browser_ready_startup_ms": 85.0,
                "memory_gb_per_1000_sessions": 340.0,
                "session_crash_or_oom_rate_pct": 4.8,
                "cross_tenant_kernel_isolation": False,
                "monthly_fleet_infra_cost_usd": 9800,
                "recommended_scenario": "Internal single-tenant test runners (1 runaway tab crashes 24 neighbor sessions)",
            },
            {
                "tier": "Tier 3: Tuned K8s Pod Pool (2GB /dev/shm + dumb-init + Warm Standby Replicas)",
                "browser_ready_startup_ms": 220.0,
                "memory_gb_per_1000_sessions": 950.0,
                "session_crash_or_oom_rate_pct": 0.45,
                "cross_tenant_kernel_isolation": False,
                "monthly_fleet_infra_cost_usd": 24500,
                "recommended_scenario": "Standard Kubernetes enterprise browser pool without bare-metal KVM access",
            },
            {
                "tier": "Tier 4 (Recommended): Firecracker / Cloud Hypervisor CoW Memory-Snapshot Browser Pool + CDP Proxy + Idle Pause",
                "browser_ready_startup_ms": 140.0,
                "memory_gb_per_1000_sessions": 480.0,
                "session_crash_or_oom_rate_pct": 0.05,
                "cross_tenant_kernel_isolation": True,
                "monthly_fleet_infra_cost_usd": 14800,
                "recommended_scenario": "Multi-tenant Enterprise Browser Agents (140ms CoW restore, 99.95% reliability, hardware VM isolation)",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
