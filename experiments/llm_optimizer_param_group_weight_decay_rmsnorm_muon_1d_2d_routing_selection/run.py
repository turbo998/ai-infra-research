from __future__ import annotations
import json
from pathlib import Path


def evaluate_optimizer_param_group_routing() -> dict:
    # Workload: 8B LLM pretraining (50B tokens) comparing parameter-group weight decay & Muon/AdamW routing
    tiers = [
        {
            "tier_id": "tier1_single_param_group_uniform_weight_decay_on_rmsnorm_and_bias",
            "name": "Single Unfiltered Parameter Group (`AdamW(model.parameters(), weight_decay=0.1)`)",
            "validation_perplexity_ppl": 8.45,
            "rmsnorm_gamma_mean_norm": 0.14,
            "steps_to_target_loss_speedup_x": 1.00,
            "action": "Send to a human",
            "status": "fail",
            "note": "Applying weight_decay=0.1 to 1D RMSNorm gamma and biases shrinks gamma from 1.0 toward 0.14, amplifying effective linear layer step sizes and degrading PPL by +1.85!"
        },
        {
            "tier_id": "tier2_naive_muon_newton_schulz_applied_to_all_params_including_embed_lmhead",
            "name": "Unrouted `Muon` Newton-Schulz Orthogonalization Applied to `Embedding`, `lm_head` & `1D Norm`",
            "validation_perplexity_ppl": 11.20,
            "rmsnorm_gamma_mean_norm": 1.00,
            "steps_to_target_loss_speedup_x": 0.62,
            "action": "Send to a human",
            "status": "fail",
            "note": "Running Newton-Schulz orthogonalization on the sparse [152K, 4096] Embedding/lm_head matrices and 1D vectors equalizes rare-token gradients and causes severe instability!"
        },
        {
            "tier_id": "tier3_adamw_split_2d_decay_vs_1d_no_decay_groups",
            "name": "Decoupled `AdamW` Parameter Groups (`ndim >= 2 -> WD=0.1` vs `ndim < 2 / Norm / Bias -> WD=0.0`)",
            "validation_perplexity_ppl": 6.95,
            "rmsnorm_gamma_mean_norm": 0.98,
            "steps_to_target_loss_speedup_x": 1.15,
            "action": "Confirm",
            "status": "pass",
            "note": "Preserves RMSNorm gamma scale (~0.98) and cuts PPL by -1.50 vs Tier 1, serving as the standard AdamW baseline."
        },
        {
            "tier_id": "tier4_three_way_hybrid_muon_2d_hidden_plus_adamw_embed_plus_zero_wd_1d",
            "name": "3-Way Hybrid Routing (`2D Hidden -> Muon + RMS Scale` | `Embed/Head -> AdamW` | `1D Norm/Bias -> AdamW WD=0`)",
            "validation_perplexity_ppl": 6.60,
            "rmsnorm_gamma_mean_norm": 1.00,
            "steps_to_target_loss_speedup_x": 1.42,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Routes internal 2D matrices to Muon (with 0.2*sqrt(max(A,B)) RMS scaling), keeps Embed/lm_head on AdamW, and sets WD=0.0 on 1D RMSNorm -> 6.60 PPL & 1.42x faster convergence!"
        },
    ]
    return {
        "experiment_id": "llm_optimizer_param_group_weight_decay_rmsnorm_muon_1d_2d_routing_selection",
        "round": 323,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "model": "8B LLM (Vocab V=152,064, Hidden D=4,096)",
            "training_tokens": "50B Tokens"
        },
        "tiers": tiers,
        "summary": {
            "ppl_improvement_vs_unfiltered_pts": round(8.45 - 6.60, 2),
            "convergence_speedup_x": 1.42,
            "recommended_ppl": 6.60,
            "recommended_tier": "tier4_three_way_hybrid_muon_2d_hidden_plus_adamw_embed_plus_zero_wd_1d"
        }
    }


if __name__ == "__main__":
    data = evaluate_optimizer_param_group_routing()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
