from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    index_engines = [
        {
            "engine": "Tier 1: Uncompressed In-Memory HNSW (FP32 1536d Vectors + M=32 Graph in Host DRAM)",
            "dram_footprint_1b_vectors_tb": 8.42,
            "nvme_footprint_1b_vectors_tb": 8.42,
            "recall_at_10_pct": 98.4,
            "search_p99_latency_ms": 4.2,
            "index_build_hours_1b": 38.0,
            "annual_infra_tco_1b_usd": 412000.0,
        },
        {
            "engine": "Tier 2: Quantized In-Memory HNSW (SQ8 / Binary Quantization + 3x Oversampling Rerank)",
            "dram_footprint_1b_vectors_tb": 2.15,
            "nvme_footprint_1b_vectors_tb": 6.80,
            "recall_at_10_pct": 96.8,
            "search_p99_latency_ms": 5.8,
            "index_build_hours_1b": 29.0,
            "annual_infra_tco_1b_usd": 118000.0,
        },
        {
            "engine": "Tier 3: DiskANN / Vamana Hybrid SSD Index (64B PQ in DRAM + 4KB Aligned Graph/FP16 on NVMe RAID0)",
            "dram_footprint_1b_vectors_tb": 0.14,
            "nvme_footprint_1b_vectors_tb": 4.20,
            "recall_at_10_pct": 96.2,
            "search_p99_latency_ms": 11.4,
            "index_build_hours_1b": 22.0,
            "annual_infra_tco_1b_usd": 29500.0,
        },
        {
            "engine": "Tier 4: GPU-Accelerated cuVS CAGRA Index Build + Tiered Hot-SQ8 / Warm-DiskANN Serving",
            "dram_footprint_1b_vectors_tb": 0.38,
            "nvme_footprint_1b_vectors_tb": 4.20,
            "recall_at_10_pct": 97.6,
            "search_p99_latency_ms": 3.6,
            "index_build_hours_1b": 2.8,
            "annual_infra_tco_1b_usd": 38400.0,
        },
    ]
    return {
        "experiment_id": "enterprise_rag_vector_search_hnsw_diskann_cuvs_memory_tco_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "vector_dimension": 1536,
        "vector_count_billions": 1.0,
        "index_engines": index_engines,
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
        print("OK: enterprise_rag_vector_search_hnsw_diskann_cuvs_memory_tco_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
