from __future__ import annotations

import json
from pathlib import Path


def evaluate_meta_device_init_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Standard 8-Rank CPU Model(config) Before FSDP Sharding",
            "peak_node_ram_gb": 1120.0,
            "init_latency_sec": 185.0,
            "step1_nan_rate_pct": 0.0,
            "fits_512gb_node": False,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: meta-Device + to_empty('cuda') Without Non-Persistent Buffer Re-Init",
            "peak_node_ram_gb": 0.20,
            "init_latency_sec": 1.6,
            "step1_nan_rate_pct": 100.0,
            "fits_512gb_node": True,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: Rank-0 CPU Instantiation + sync_module_states Broadcast",
            "peak_node_ram_gb": 140.0,
            "init_latency_sec": 34.0,
            "step1_nan_rate_pct": 0.0,
            "fits_512gb_node": True,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: meta-Device + fully_shard() First + to_empty('cuda') + Buffer/Param Reset",
            "peak_node_ram_gb": 0.20,
            "init_latency_sec": 1.8,
            "step1_nan_rate_pct": 0.0,
            "fits_512gb_node": True,
            "decision": "Act automatically",
        },
    ]

    model_scaling = [
        {"model_b": 14, "cpu_8rank_ram_gb": 224.0, "rank0_ram_gb": 28.0, "tier4_ram_gb": 0.20},
        {"model_b": 32, "cpu_8rank_ram_gb": 512.0, "rank0_ram_gb": 64.0, "tier4_ram_gb": 0.20},
        {"model_b": 70, "cpu_8rank_ram_gb": 1120.0, "rank0_ram_gb": 140.0, "tier4_ram_gb": 0.20},
        {"model_b": 405, "cpu_8rank_ram_gb": 6480.0, "rank0_ram_gb": 810.0, "tier4_ram_gb": 0.25},
    ]

    summary = {
        "benchmark": "pytorch_meta_device_init_to_empty_uninitialized_nan_fsdp2_reset_parameters_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "model_size_b": 70,
        "local_ranks": 8,
        "naive_host_ram_gb": 1120.0,
        "tier4_host_ram_gb": 0.20,
        "ram_saved_gb": round(1120.0 - 0.20, 2),
        "naive_init_sec": 185.0,
        "tier4_init_sec": 1.8,
        "init_speedup_x": round(185.0 / 1.8, 1),
        "tiers": tiers,
        "model_scaling": model_scaling,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_meta_device_init_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
