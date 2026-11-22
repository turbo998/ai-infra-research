from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Factory Default BIOS (PCIe ACS Enabled + Unpinned NUMA)",
            "gpudirect_rdma_busbw_gbps": 14.2,
            "p99_tpot_ms": 46.5,
            "server_throughput_efficiency_pct": 72.0,
            "host_dma_pinned_bw_gbps": 24.5,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 2: PCIe ACS Disabled Only (Cross-Socket NUMA Bouncing)",
            "gpudirect_rdma_busbw_gbps": 44.5,
            "p99_tpot_ms": 31.0,
            "server_throughput_efficiency_pct": 86.5,
            "host_dma_pinned_bw_gbps": 32.0,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 3: ACS Disabled + K8s TopologyManager single-numa-node",
            "gpudirect_rdma_busbw_gbps": 47.2,
            "p99_tpot_ms": 21.5,
            "server_throughput_efficiency_pct": 95.2,
            "host_dma_pinned_bw_gbps": 54.0,
            "action": "Confirm",
        },
        {
            "tier": "Tier 4: ACS Disabled + single-numa-node + Isolated IRQ/Tokenizer Cores (Recommended)",
            "gpudirect_rdma_busbw_gbps": 47.8,
            "p99_tpot_ms": 17.8,
            "server_throughput_efficiency_pct": 100.0,
            "host_dma_pinned_bw_gbps": 58.5,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_gpu_server_numa_pcie_acs_irq_pinning_tuning_selection",
        "title": "Enterprise 8-GPU Server CPU/NUMA Topology, PCIe ACS Disable & IRQ/Tokenizer Core Pinning Selection (#506)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "server_8gpu_baseline": {
            "rdma_speedup_factor": round(47.8 / 14.2, 2),
            "p99_tpot_reduction_pct": round((1.0 - 17.8 / 46.5) * 100, 1),
            "net_throughput_gain_pct": round((100.0 / 72.0 - 1.0) * 100, 1),
        },
        "tiers": tiers,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
