from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    batch_pipelines = [
        {
            "pipeline": "Tier 1: Unsorted FIFO Requests Sent to Online Real-Time Serving Pool (SLA-Constrained Batch<=32)",
            "prefix_cache_hit_rate_pct": 18.5,
            "throughput_tokens_per_sec_per_gpu": 1420,
            "effective_cost_per_1m_tokens_usd": 0.34,
            "spot_preemption_work_lost_pct": 0.0,
        },
        {
            "pipeline": "Tier 2: Cloud Managed 50%-Discount Batch API (24-Hour Turnaround JSONL Queue)",
            "prefix_cache_hit_rate_pct": 50.0,
            "throughput_tokens_per_sec_per_gpu": 2800,
            "effective_cost_per_1m_tokens_usd": 0.17,
            "spot_preemption_work_lost_pct": 0.0,
        },
        {
            "pipeline": "Tier 3: Dedicated On-Demand GPU Batch Pool without Input Sorting (High Batch=256, Random Order)",
            "prefix_cache_hit_rate_pct": 24.0,
            "throughput_tokens_per_sec_per_gpu": 3650,
            "effective_cost_per_1m_tokens_usd": 0.11,
            "spot_preemption_work_lost_pct": 0.0,
        },
        {
            "pipeline": "Tier 4: Ray Data + Lexicographical Radix Prefix Sorting + Length Bucketing on Spot/Tidal GPUs (1,000-Row Micro-Checkpointing)",
            "prefix_cache_hit_rate_pct": 94.2,
            "throughput_tokens_per_sec_per_gpu": 9840,
            "effective_cost_per_1m_tokens_usd": 0.028,
            "spot_preemption_work_lost_pct": 0.4,
        },
    ]
    return {
        "experiment_id": "enterprise_offline_batch_llm_inference_radix_prefix_sorting_spot_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "dataset_tokens_billions": 10.0,
        "batch_pipelines": batch_pipelines,
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
        print("OK: enterprise_offline_batch_llm_inference_radix_prefix_sorting_spot_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
