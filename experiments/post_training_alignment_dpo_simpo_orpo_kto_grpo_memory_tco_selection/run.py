from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "post_training_alignment_dpo_simpo_orpo_kto_grpo_memory_tco_selection",
        "title": "Post-Training Preference Alignment Selection: DPO vs. SimPO vs. ORPO vs. KTO vs. Two-Stage SimPO + Online GRPO",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "model": "Llama-3.3-70B Full-Parameter Alignment at 8K-16K Context (16x H100 80GB)",
            "preference_dataset_pairs": 50000,
        },
        "tiers": [
            {
                "tier": "Tier 1: Standard Online-Reference DPO (Dual 70B Models in HBM + Materialized Logits)",
                "peak_hbm_gb_per_gpu": 102.5,
                "relative_training_speedup": 1.00,
                "domain_reasoning_and_alignment_score_pct": 81.2,
                "run_compute_cost_usd": 4600,
                "recommended_scenario": "Anti-pattern (holding frozen 70B Reference Model + materialized logits causes OOM on 80GB GPUs)",
            },
            {
                "tier": "Tier 2: Precomputed-Ref DPO / Adapter-Toggle LoRA DPO",
                "peak_hbm_gb_per_gpu": 72.0,
                "relative_training_speedup": 1.48,
                "domain_reasoning_and_alignment_score_pct": 81.5,
                "run_compute_cost_usd": 3100,
                "recommended_scenario": "Standard DPO when strict KL divergence to SFT reference is required",
            },
            {
                "tier": "Tier 3: Reference-Free SimPO / KTO + Liger Fused Chunked Preference Loss",
                "peak_hbm_gb_per_gpu": 54.2,
                "relative_training_speedup": 1.89,
                "domain_reasoning_and_alignment_score_pct": 84.6,
                "run_compute_cost_usd": 1850,
                "recommended_scenario": "Fast tone, style, safety & conciseness alignment (eliminates DPO length bias)",
            },
            {
                "tier": "Tier 4 (Recommended): Two-Stage Pipeline (Stage 1: SimPO + Liger Fused Loss -> Stage 2: Targeted Online Verifiable GRPO)",
                "peak_hbm_gb_per_gpu": 56.8,
                "relative_training_speedup": 1.72,
                "domain_reasoning_and_alignment_score_pct": 92.8,
                "run_compute_cost_usd": 3450,
                "recommended_scenario": "Enterprise agents requiring both human-preferred tone/safety AND verifiable SQL/code/math reasoning",
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
