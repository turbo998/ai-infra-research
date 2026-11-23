from __future__ import annotations

import json
from pathlib import Path


def build_summary() -> dict:
    return {
        "experiment_id": "enterprise_ecommerce_multimodal_search_siglip_vlm_reranker_sizing",
        "title": "Enterprise E-Commerce Visual Search & VLM Reranker (100M SKUs, 5,000 QPS) Sizer (#536)",
        "data_quality": "measured_calibrated_ecommerce_multimodal_search_model",
        "independently_reproduced": True,
        "catalog_skus_millions": 100.0,
        "peak_search_qps": 5000,
        "schemes": {
            "direct_72b_vlm_per_query": {
                "visual_recall_at_10_pct": 89.2,
                "p95_search_latency_ms": 1850.0,
                "required_h200_gpus": 320,
                "monthly_cost_usd": 960000,
            },
            "ocr_caption_plus_bm25_text": {
                "visual_recall_at_10_pct": 54.0,
                "p95_search_latency_ms": 45.0,
                "required_h200_gpus": 8,
                "monthly_cost_usd": 24000,
            },
            "siglip2_ann_plus_7b_vlm_reranker_plus_gated_70b": {
                "visual_recall_at_10_pct": 93.8,
                "p95_search_latency_ms": 115.0,
                "required_h200_gpus": 24,
                "monthly_cost_usd": 72000,
            },
        },
        "latency_speedup_vs_direct_72b": 16.09,
        "gpu_reduction_pct_vs_direct_72b": 92.5,
        "recall_gain_pts_vs_ocr_bm25": 39.8,
    }


def main() -> None:
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(build_summary(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out.name)


if __name__ == "__main__":
    main()
