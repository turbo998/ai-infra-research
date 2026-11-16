from __future__ import annotations
import json
from pathlib import Path


def evaluate_cudagraph_bucketing_and_shared_pool() -> dict:
    # Workload: LLM serving capturing CUDAGraphs up to max_batch_size = 256
    tiers = [
        {
            "tier_id": "tier1_unshared_per_graph_private_mempool_40_buckets_hbm_bloat",
            "name": "40 Fine-Grained CUDAGraph Buckets Without Shared `graph_pool_handle` (Private Pools)",
            "cudagraph_reserved_hbm_gb": 8.50,
            "avg_batch_padding_compute_waste_pct": 3.1,
            "decode_step_latency_at_b129_ms": 14.8,
            "action": "Send to a human",
            "status": "fail",
            "note": "Capturing 40 separate CUDAGraphs without passing a shared `pool=graph_pool_handle()` allocates 40 private activation pools -> wastes 8.50 GB/GPU of HBM!"
        },
        {
            "tier_id": "tier2_coarse_power_of_two_buckets_1_2_4_8_16_32_64_128_256",
            "name": "Coarse Power-of-Two Buckets (`1, 2, 4, 8, 16, 32, 64, 128, 256`) With Shared Pool",
            "cudagraph_reserved_hbm_gb": 1.10,
            "avg_batch_padding_compute_waste_pct": 28.4,
            "decode_step_latency_at_b129_ms": 20.6,
            "action": "Send to a human",
            "status": "fail",
            "note": "When active requests = 129, pads with 127 dummy tokens up to the 256 bucket (49.6% worst-case / 28.4% avg padding waste -> 20.6ms step)!"
        },
        {
            "tier_id": "tier3_shared_mempool_reverse_capture_order_40_decode_buckets",
            "name": "Shared `graph_pool_handle` + Reverse Capture Order (`256 -> 1`) Across 35 Decode Buckets",
            "cudagraph_reserved_hbm_gb": 1.05,
            "avg_batch_padding_compute_waste_pct": 3.4,
            "decode_step_latency_at_b129_ms": 14.5,
            "action": "Confirm",
            "status": "pass",
            "note": "Capturing largest bucket (256) first and sharing its memory pool across all smaller buckets caps HBM at 1.05 GB, though mixed prefill-decode still falls back to Eager."
        },
        {
            "tier_id": "tier4_shared_mempool_plus_piecewise_cudagraph_for_mixed_chunked_prefill",
            "name": "Shared `graph_pool_handle` (`256 -> 1`) + `Piecewise CUDAGraph` (Covers Mixed Prefill-Decode Too!)",
            "cudagraph_reserved_hbm_gb": 0.95,
            "avg_batch_padding_compute_waste_pct": 2.8,
            "decode_step_latency_at_b129_ms": 13.8,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Shares 1 physical memory pool across all graphs (0.95 GB, 8.95x smaller) and uses Piecewise CUDAGraph to keep static GEMMs graphed even during mixed prefill-decode!"
        },
    ]
    return {
        "experiment_id": "llm_serving_cudagraph_batch_bucketing_padding_waste_shared_mempool_selection",
        "round": 325,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "max_batch_size": 256,
            "fine_grained_buckets": "[1, 2, 4, 8, 16, 24, 32, ..., 136, ..., 256] (35 buckets)"
        },
        "tiers": tiers,
        "summary": {
            "graph_hbm_reduction_x": round(8.50 / 0.95, 2),
            "b129_step_speedup_vs_pow2_x": round(20.6 / 13.8, 2),
            "recommended_hbm_gb": 0.95,
            "recommended_tier": "tier4_shared_mempool_plus_piecewise_cudagraph_for_mixed_chunked_prefill"
        }
    }


if __name__ == "__main__":
    data = evaluate_cudagraph_bucketing_and_shared_pool()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
