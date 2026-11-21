from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "offline_batch_inference_document_extraction_length_bucketing_spot_tco_selection",
        "title": "High-Throughput Offline Batch Inference & Document Extraction: Prefix/Length Sorting, Max-Batch Roofline & Spot GPU Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "documents_per_night": 2000000,
            "avg_input_tokens_per_doc": 3500,
            "avg_output_json_tokens": 450,
            "total_tokens_billions": 7.9,
        },
        "tiers": [
            {
                "tier_id": "tier1_sending_offline_docs_to_realtime_online_paygo_or_http_endpoint",
                "name": "Tier 1: Firing 2M Offline Docs at Real-Time Online HTTP Endpoint (Unsorted Random Order, Online B=48 Cap)",
                "sustained_output_tps_per_gpu": 1120,
                "prefix_cache_hit_rate_pct": 38.0,
                "nightly_2m_docs_completion_hours": 14.8,
                "cost_per_2m_docs_batch_run_usd": 18900,
                "verdict": "Anti-Pattern: Random arrival order thrash-evicts Prefix Cache; online latency cap wastes 74% batch throughput",
            },
            {
                "tier_id": "tier2_cloud_managed_24h_async_batch_api_50pct_discount",
                "name": "Tier 2: Cloud Managed 24h Batch API (50% Token Discount over PAYGO, Zero GPU Ops, 6-18h Turnaround)",
                "sustained_output_tps_per_gpu": 1950,
                "prefix_cache_hit_rate_pct": 72.0,
                "nightly_2m_docs_completion_hours": 9.5,
                "cost_per_2m_docs_batch_run_usd": 9450,
                "verdict": "Great zero-ops choice for ad-hoc or <1B-token batch runs ($9,450/run), but still 3.3x pricier than Spot batch",
            },
            {
                "tier_id": "tier3_on_demand_gpu_vllm_offline_without_prefix_or_length_sorting",
                "name": "Tier 3: Dedicated On-Demand GPU `vLLM LLM.generate()` (High Batch B=256, Unsorted Input Stream)",
                "sustained_output_tps_per_gpu": 2840,
                "prefix_cache_hit_rate_pct": 49.5,
                "nightly_2m_docs_completion_hours": 5.8,
                "cost_per_2m_docs_batch_run_usd": 6200,
                "verdict": "High batch size boosts TPS to 2,840, but mixing 500-tok and 32K-tok docs in the same batch creates tail bubbles",
            },
            {
                "tier_id": "tier4_prefix_and_length_sorted_bucketing_plus_spot_gpu_and_parquet_ckpt",
                "name": "Tier 4: Prefix-Grouped & Length-Sorted Bucketing + Max-Batch FP8 (`B=384`) on Spot GPUs + Shard Checkpointing",
                "sustained_output_tps_per_gpu": 4680,
                "prefix_cache_hit_rate_pct": 99.4,
                "nightly_2m_docs_completion_hours": 3.4,
                "cost_per_2m_docs_batch_run_usd": 2850,
                "verdict": "Recommended: 4,680 Out tok/s/GPU (4.18x vs. Tier 1), 99.4% Prefix Hit, 3.4h completion & $2,850/run (-84.9%)",
            },
        ],
        "headline_findings": {
            "output_tps_per_gpu": 4680,
            "tps_speedup_vs_tier1_x": 4.18,
            "prefix_hit_rate_pct": 99.4,
            "cost_reduction_vs_tier1_pct": 84.9,
        },
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
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
