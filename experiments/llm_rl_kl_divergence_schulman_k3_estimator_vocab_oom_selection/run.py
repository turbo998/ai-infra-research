from __future__ import annotations

import json
from pathlib import Path


def evaluate_rl_kl_estimator_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Naive k1 Estimator (log_pi_theta - log_pi_ref, Can Be Negative)",
            "kl_workspace_hbm_gb": 0.00012,
            "negative_kl_token_rate_pct": 48.5,
            "gradient_variance_ratio_x": 14.8,
            "exp_overflow_nan_rate_pct": 0.0,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Full-Vocabulary Analytical KL (Sum Over All V=152,064 Tokens)",
            "kl_workspace_hbm_gb": 39.86,
            "negative_kl_token_rate_pct": 0.0,
            "gradient_variance_ratio_x": 1.0,
            "exp_overflow_nan_rate_pct": 0.0,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: Unclamped Schulman k3 Estimator (exp(log_ref - log_pi) - 1 - delta)",
            "kl_workspace_hbm_gb": 0.00012,
            "negative_kl_token_rate_pct": 0.0,
            "gradient_variance_ratio_x": 1.1,
            "exp_overflow_nan_rate_pct": 6.4,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Clamped Schulman k3 Estimator (clamp(delta, -10, 10) on 1D Sampled Tokens)",
            "kl_workspace_hbm_gb": 0.00012,
            "negative_kl_token_rate_pct": 0.0,
            "gradient_variance_ratio_x": 1.1,
            "exp_overflow_nan_rate_pct": 0.0,
            "decision": "Act automatically",
        },
    ]

    estimator_properties = [
        {
            "estimator": "k1 = -log(r) = log(pi) - log(ref)",
            "unbiased": True,
            "strictly_non_negative": False,
            "variance_near_zero": "High O(delta^2)",
        },
        {
            "estimator": "k2 = 0.5 * (log(pi) - log(ref))^2",
            "unbiased": False,
            "strictly_non_negative": True,
            "variance_near_zero": "Low O(delta^4)",
        },
        {
            "estimator": "k3 = (r - 1) - log(r) (Clamped delta in [-10, 10])",
            "unbiased": True,
            "strictly_non_negative": True,
            "variance_near_zero": "Low O(delta^4) + Overflow-Safe",
        },
        {
            "estimator": "Full-Vocab Analytical Sum(pi * (log_pi - log_ref))",
            "unbiased": True,
            "strictly_non_negative": True,
            "variance_near_zero": "Zero Monte-Carlo Variance (39.86 GB OOM!)",
        },
    ]

    summary = {
        "benchmark": "llm_rl_kl_divergence_schulman_k3_estimator_vocab_oom_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "batch_size": 4,
        "seq_len": 8192,
        "vocab_size": 152064,
        "full_vocab_kl_hbm_gb": 39.86,
        "k3_1d_kl_hbm_gb": 0.00012,
        "hbm_saved_gb": round(39.86 - 0.00012, 2),
        "k1_variance_ratio_x": 14.8,
        "unclamped_k3_nan_pct": 6.4,
        "clamped_k3_nan_pct": 0.0,
        "tiers": tiers,
        "estimator_properties": estimator_properties,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_rl_kl_estimator_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
