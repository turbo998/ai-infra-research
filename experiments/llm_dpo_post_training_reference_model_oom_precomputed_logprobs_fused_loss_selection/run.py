from __future__ import annotations

import json
from pathlib import Path


def evaluate_dpo_reference_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Dual Model in HBM (Policy + Ref) + Padded [2B, L_max] + Materialized [N, V] Logits",
            "peak_hbm_per_gpu_gb": 88.40,
            "throughput_tok_per_sec_gpu": 1820,
            "padding_compute_waste_pct": 43.8,
            "fits_80gb_h100": False,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Offline Precomputed Ref Logprobs + Padded [2B, L_max] + Materialized Logits",
            "peak_hbm_per_gpu_gb": 70.90,
            "throughput_tok_per_sec_gpu": 2640,
            "padding_compute_waste_pct": 43.8,
            "fits_80gb_h100": True,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 3: Precomputed Ref Logprobs + Ragged cu_seqlens Pair Packing (Unfused Logits)",
            "peak_hbm_per_gpu_gb": 68.50,
            "throughput_tok_per_sec_gpu": 3790,
            "padding_compute_waste_pct": 0.0,
            "fits_80gb_h100": True,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Precomputed Ref Logprobs + Ragged cu_seqlens Packing + Fused Linear DPO Loss",
            "peak_hbm_per_gpu_gb": 51.20,
            "throughput_tok_per_sec_gpu": 4460,
            "padding_compute_waste_pct": 0.0,
            "fits_80gb_h100": True,
            "decision": "Act automatically",
        },
    ]

    context_scaling = [
        {"max_len": 2048, "naive_hbm_gb": 72.10, "tier4_hbm_gb": 48.00},
        {"max_len": 4096, "naive_hbm_gb": 78.80, "tier4_hbm_gb": 49.40},
        {"max_len": 8192, "naive_hbm_gb": 88.40, "tier4_hbm_gb": 51.20},
        {"max_len": 16384, "naive_hbm_gb": 108.60, "tier4_hbm_gb": 55.00},
    ]

    summary = {
        "benchmark": "llm_dpo_post_training_reference_model_oom_precomputed_logprobs_fused_loss_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "model_size": "70B (8x H100-80GB FSDP)",
        "vocab_size": 152064,
        "naive_peak_hbm_gb": 88.40,
        "tier4_peak_hbm_gb": 51.20,
        "hbm_saved_gb": round(88.40 - 51.20, 2),
        "naive_tps": 1820,
        "tier4_tps": 4460,
        "throughput_speedup_x": round(4460 / 1820, 2),
        "tiers": tiers,
        "context_scaling": context_scaling,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_dpo_reference_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
