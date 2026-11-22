from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    serving_modes = [
        {
            "mode": "Tier 1: Dedicated Merged Fine-Tuned Pods (1 Replica per Tenant for 128 Tenants on Llama-70B FP8)",
            "gpus_required_128_tenants": 256,
            "cold_adapter_swap_ms": 0.0,
            "mixed_batch_tpot_p99_ms": 14.5,
            "shared_base_prefix_hit_pct": 0.0,
            "monthly_cluster_cost_usd": 460800.0,
        },
        {
            "mode": "Tier 2: Unbatched Weight Merge / Unmerge Hot-Swap on Shared Base Pods",
            "gpus_required_128_tenants": 16,
            "cold_adapter_swap_ms": 340.0,
            "mixed_batch_tpot_p99_ms": 92.0,
            "shared_base_prefix_hit_pct": 35.0,
            "monthly_cluster_cost_usd": 28800.0,
        },
        {
            "mode": "Tier 3: vLLM / SGLang Multi-LoRA Heterogeneous Batching (32 Hot HBM Slots + CPU Pinned DRAM Pool)",
            "gpus_required_128_tenants": 8,
            "cold_adapter_swap_ms": 8.2,
            "mixed_batch_tpot_p99_ms": 16.8,
            "shared_base_prefix_hit_pct": 74.0,
            "monthly_cluster_cost_usd": 14400.0,
        },
        {
            "mode": "Tier 4: S-LoRA Unified Paging + Locality-Aware AI Gateway Routing + Shared Base Prefix Cache",
            "gpus_required_128_tenants": 8,
            "cold_adapter_swap_ms": 1.9,
            "mixed_batch_tpot_p99_ms": 14.9,
            "shared_base_prefix_hit_pct": 89.5,
            "monthly_cluster_cost_usd": 14400.0,
        },
    ]
    return {
        "experiment_id": "enterprise_multi_lora_serving_slora_punica_hbm_cache_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "base_model": "Llama-3.3-70B-FP8",
        "lora_rank_32_size_mb": 335.5,
        "tenant_count": 128,
        "serving_modes": serving_modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_multi_lora_serving_slora_punica_hbm_cache_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
