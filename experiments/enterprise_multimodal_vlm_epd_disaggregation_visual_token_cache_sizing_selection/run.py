from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Monolithic Colocated VLM (ViT + Prefill + Decode on 32x H200 · No Image Embedding Cache)",
            "p95_ttft_ms": 2480.0,
            "p99_tpot_ms": 74.5,
            "cluster_goodput_req_s": 8.4,
            "cost_per_1k_multimodal_turns_usd": 4.20,
        },
        {
            "tier": "Tier 2: Colocated VLM + SHA-256 Image Embedding Cache (Skips Redundant Multi-Turn ViT)",
            "p95_ttft_ms": 1240.0,
            "p99_tpot_ms": 48.2,
            "cluster_goodput_req_s": 16.2,
            "cost_per_1k_multimodal_turns_usd": 2.18,
        },
        {
            "tier": "Tier 3: 2-Stage PD Disaggregation (ViT Bundled on H100 Prefill + H200 Decode)",
            "p95_ttft_ms": 620.0,
            "p99_tpot_ms": 16.8,
            "cluster_goodput_req_s": 29.5,
            "cost_per_1k_multimodal_turns_usd": 1.15,
        },
        {
            "tier": "Tier 4: 3-Stage Heterogeneous E-P-D Disaggregation (8x L40S ViT Pool + 12x H100 Prefill + 12x H200 Decode + AnyRes Token Pruning & Embedding Cache)",
            "p95_ttft_ms": 310.0,
            "p99_tpot_ms": 14.1,
            "cluster_goodput_req_s": 46.8,
            "cost_per_1k_multimodal_turns_usd": 0.62,
        },
    ]
    return {
        "experiment_id": "enterprise_multimodal_vlm_epd_disaggregation_visual_token_cache_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "visual_token_formula": {
            "patch_merge_Equation": "N_vis = ceil(H / 28) * ceil(W / 28) * (1 + N_tiles)",
            "hd_1080p_unpruned_tokens": 2760,
            "video_60s_2fps_tokens": 30720,
        },
        "tiers": tiers,
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
        print("OK: enterprise_multimodal_vlm_epd_disaggregation_visual_token_cache_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
