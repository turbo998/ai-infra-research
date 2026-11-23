from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier_id": "hbm_only_evict_and_recompute",
            "name": "Anti-Pattern: HBM-Only KV Cache (Evict on Tool Pause & Recompute 64K Prefill Every Turn)",
            "storage_medium": "Tier 1 Only: GPU HBM3e (420 GB free KV/node)",
            "concurrent_64k_sessions_supported": 40,
            "cache_hit_rate_across_500_agents_pct": 8.0,
            "turn_resume_p95_ttft_ms": 1840.0,
            "required_h200_gpus": 36,
            "annual_infra_tco_usd": 964000.0,
            "action": "Send to a human",
        },
        {
            "tier_id": "local_host_dram_offload_lmcache",
            "name": "Single-Node Offload: GPU HBM + Local Host Pinned DDR5 DRAM (2 TB/node via PCIe Gen5)",
            "storage_medium": "Tier 1 HBM + Tier 2 Local Pinned DRAM",
            "concurrent_64k_sessions_supported": 235,
            "cache_hit_rate_across_500_agents_pct": 64.0,
            "turn_resume_p95_ttft_ms": 245.0,
            "required_h200_gpus": 20,
            "annual_infra_tco_usd": 548000.0,
            "action": "Confirm",
        },
        {
            "tier_id": "three_tier_rdma_dram_3fs_nixl_mooncake",
            "name": "Optimal 3-Tier Pool: HBM + Distributed RDMA DRAM Pool + 3FS NVMe GDS (Mooncake / Dynamo NIXL)",
            "storage_medium": "Tier 1 HBM + Tier 2 RDMA DRAM + Tier 3 3FS NVMe",
            "concurrent_64k_sessions_supported": 2500,
            "cache_hit_rate_across_500_agents_pct": 96.5,
            "turn_resume_p95_ttft_ms": 118.0,
            "required_h200_gpus": 12,
            "annual_infra_tco_usd": 342000.0,
            "action": "Act automatically",
        },
    ]

    engine_comparison = [
        {"engine": "LMCache (vLLM Production Stack)", "best_for": "Drop-in vLLM V1 + CacheGen KV Compression over 25G/100G TCP/RoCE"},
        {"engine": "Mooncake Store (SGLang / Kimi Native)", "best_for": "Multi-NIC Aggregated RDMA Global DRAM Pool + RadixAttention"},
        {"engine": "NVIDIA Dynamo KV Block Manager + NIXL", "best_for": "Unified HBM <-> Host DRAM <-> GDS NVMe 3-Tier Lifecycle on K8s"},
    ]

    return {
        "experiment_id": "enterprise_milestone_515_tiered_kv_cache_lmcache_mooncake_nixl_sizing_selection",
        "title": "Milestone #515: Enterprise 3-Tier KV Cache Pool (HBM <-> Host DRAM <-> RDMA/3FS NVMe) Selection Guide — LMCache vs. Mooncake vs. Dynamo NIXL",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "resume_ttft_speedup_factor_vs_recompute": round(1840.0 / 118.0, 2),
            "gpu_saving_pct_vs_hbm_only": round((1.0 - 12 / 36) * 100, 1),
            "tco_saving_pct_vs_hbm_only": round((1.0 - 342000.0 / 964000.0) * 100, 1),
        },
        "engine_comparison": engine_comparison,
        "tiers": tiers,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
