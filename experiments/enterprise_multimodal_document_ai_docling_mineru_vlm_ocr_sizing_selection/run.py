from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Legacy PyPDF / Basic CPU OCR (Scrambled Tables)",
            "table_teds_accuracy_pct": 42.5,
            "effective_pages_per_sec_per_gpu": 0.0,
            "gpus_for_10m_pages_per_day": 0,
            "cost_per_1000_pages_usd": 0.05,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 2: Brute-Force 72B VLM on Every Page (1,540 Visual Tokens/Page)",
            "table_teds_accuracy_pct": 95.8,
            "effective_pages_per_sec_per_gpu": 1.85,
            "gpus_for_10m_pages_per_day": 64,
            "cost_per_1000_pages_usd": 8.50,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 3: Uniform 7B Document VLM (olmOCR / Qwen2.5-VL-7B FP8) on Every Page",
            "table_teds_accuracy_pct": 94.2,
            "effective_pages_per_sec_per_gpu": 14.2,
            "gpus_for_10m_pages_per_day": 10,
            "cost_per_1000_pages_usd": 1.15,
            "action": "Confirm",
        },
        {
            "tier": "Tier 4: Complexity-Gated Hybrid (65% Digital + 25% MinerU/Docling + 10% Cropped 7B VLM)",
            "table_teds_accuracy_pct": 96.1,
            "effective_pages_per_sec_per_gpu": 74.0,
            "gpus_for_10m_pages_per_day": 2,
            "cost_per_1000_pages_usd": 0.18,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_multimodal_document_ai_docling_mineru_vlm_ocr_sizing_selection",
        "title": "Enterprise Multimodal Document Intelligence (MinerU/Docling + Region-Cropped 7B VLM vs. 72B VLM) Sizer (#496)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_10m_pages_day": {
            "daily_pages": 10000000,
            "required_cluster_pages_per_sec": round(10000000 / 86400.0, 1),
            "throughput_speedup_vs_72b_vlm": round(74.0 / 1.85, 1),
            "cost_reduction_pct_vs_72b_vlm": round((1.0 - 0.18 / 8.50) * 100, 1),
            "gpu_saving_pct_vs_72b_vlm": round((1.0 - 2 / 64) * 100, 1),
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
