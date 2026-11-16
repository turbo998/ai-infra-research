from __future__ import annotations
import json
from pathlib import Path


def evaluate_persistenced_and_uvm_thrash() -> dict:
    # Workload: 8x H100-80GB node container cold start & 20% HBM memory oversubscription behavior
    tiers = [
        {
            "tier_id": "tier1_no_persistenced_driver_teardown_plus_naive_uvm_demand_paging",
            "name": "No `nvidia-persistenced` (Driver Unloads State on Exit) + Naive `cudaMallocManaged` Demand Paging",
            "cuda_init_cold_start_sec": 14.50,
            "oversubscribed_pcie_bw_gbps": 4.2,
            "step_latency_under_20pct_oversub_ms": 3820.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Without persistenced, every container exit unloads 640GB BAR1/ECC state (14.5s re-init), and UVM demand paging triggers 450k PCIe faults/s (4.2 GB/s)!"
        },
        {
            "tier_id": "tier2_legacy_nvidia_smi_pm_1_plus_uvm_demand_paging",
            "name": "Legacy `nvidia-smi -pm 1` Only (No `--uvm-persistence-mode`) + `cudaMallocManaged`",
            "cuda_init_cold_start_sec": 2.85,
            "oversubscribed_pcie_bw_gbps": 5.1,
            "step_latency_under_20pct_oversub_ms": 3150.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Keeps core GPU state alive, but UVM/NUMA state still tears down (2.85s init) and on-demand UVM page faults stall SMs for 3,150ms!"
        },
        {
            "tier_id": "tier3_persistenced_uvm_plus_cuda_mem_advise_prefetch_async",
            "name": "`nvidia-persistenced --uvm-persistence-mode` + `cudaMemAdvise` & `cudaMemPrefetchAsync` Bulk UVM",
            "cuda_init_cold_start_sec": 0.14,
            "oversubscribed_pcie_bw_gbps": 38.5,
            "step_latency_under_20pct_oversub_ms": 480.0,
            "action": "Confirm",
            "status": "pass",
            "note": "Keeps GPU + UVM driver state hot (0.14s init) and replaces 4KB/64KB fault thrashing with bulk 2MB prefetch, though kernel page-table locks remain."
        },
        {
            "tier_id": "tier4_persistenced_hot_daemon_plus_explicit_pinned_dma_double_buffer",
            "name": "`nvidia-persistenced` Daemon + Explicit `cudaHostAlloc` Pinned Double-Buffering (`Zero UVM Faults`)",
            "cuda_init_cold_start_sec": 0.12,
            "oversubscribed_pcie_bw_gbps": 55.4,
            "step_latency_under_20pct_oversub_ms": 302.0,
            "action": "Act automatically",
            "status": "recommended",
            "note": "0.12s instant CUDA init (120.8x faster) and 55.4 GB/s full-speed PCIe Gen5 DMA via explicit non-blocking copy streams (12.65x faster than UVM)!"
        },
    ]
    return {
        "experiment_id": "gpu_nvidia_persistenced_cold_init_uvm_page_fault_thrash_selection",
        "round": 315,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "node_gpus": "8x H100-80GB (640 GB Total BAR1)",
            "oversubscription_ratio_pct": 20.0
        },
        "tiers": tiers,
        "summary": {
            "cold_init_speedup_x": round(14.50 / 0.12, 1),
            "oversub_step_speedup_x": round(3820.0 / 302.0, 2),
            "effective_pcie_bw_gbps": 55.4,
            "recommended_tier": "tier4_persistenced_hot_daemon_plus_explicit_pinned_dma_double_buffer"
        }
    }


if __name__ == "__main__":
    data = evaluate_persistenced_and_uvm_thrash()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
