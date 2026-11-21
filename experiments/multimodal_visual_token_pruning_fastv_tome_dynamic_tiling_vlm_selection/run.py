from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "multimodal_visual_token_pruning_fastv_tome_dynamic_tiling_vlm_selection",
        "title": "Multimodal Visual Token Pruning & Compression Selection: Progressive Layer-Wise FastV/PyramidDrop vs. Unpruned Tiling & Pre-LLM Resamplers",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "input_modality": "4K Multi-Page Document / 32-Frame Video (8,192 Raw Visual Patches)",
            "vlm_backbone": "72B VLM (80 Decoder Layers)",
            "eval_benchmark": "Enterprise DocVQA + ChartQA + Video-MME Composite Accuracy",
        },
        "tiers": [
            {
                "tier": "Tier 1: Unpruned Dynamic High-Res Tiling (8,192 Visual Tokens Kept Across All 80 Layers)",
                "prefill_ttft_ms": 1820.0,
                "visual_kv_cache_gb_per_request": 2.62,
                "docvqa_chart_accuracy_pct": 94.2,
                "concurrent_requests_per_node": 22,
                "recommended_scenario": "Baseline when latency and KV Cache memory are unconstrained",
            },
            {
                "tier": "Tier 2: Fixed 64-Token Pre-LLM Perceiver Resampler (Drop 92% Before LLM Layer 1)",
                "prefill_ttft_ms": 240.0,
                "visual_kv_cache_gb_per_request": 0.21,
                "docvqa_chart_accuracy_pct": 68.5,
                "concurrent_requests_per_node": 180,
                "recommended_scenario": "Coarse natural-image captioning only (destroys fine-grained OCR, tables & numbers!)",
            },
            {
                "tier": "Tier 3: 2x2 Pixel-Shuffle / Token Merging (ToMe) Before LLM (Keep 100% Across 80 Layers)",
                "prefill_ttft_ms": 680.0,
                "visual_kv_cache_gb_per_request": 1.15,
                "docvqa_chart_accuracy_pct": 92.8,
                "concurrent_requests_per_node": 50,
                "recommended_scenario": "Standard Qwen2.5-VL / InternVL 2x2 spatial patch merging",
            },
            {
                "tier": "Tier 4 (Recommended): 2x2 Merger + Progressive Layer-Wise Attention Pruning (FastV / PyramidDrop: 100% L1-2 -> 50% L3-16 -> 20% L17-80)",
                "prefill_ttft_ms": 480.0,
                "visual_kv_cache_gb_per_request": 0.58,
                "docvqa_chart_accuracy_pct": 94.0,
                "concurrent_requests_per_node": 98,
                "recommended_scenario": "Production high-res Document/Video VLM serving (3.8x faster TTFT, -77.9% KV Cache, 94.0% accuracy)",
            },
        ],
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
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
