from __future__ import annotations

import json
from pathlib import Path


def evaluate_wgrad_fusion_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Standard PyTorch Autograd (Unfused dw_bf16 + Separate add_ + Sync wgrad)",
            "extra_grad_hbm_gb": 14.00,
            "wgrad_epilogue_overhead_ms": 18.60,
            "pp_bubble_ratio_pct": 23.8,
            "step_time_ms": 412.0,
            "relative_tflops_pct": 100.0,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Fused wgrad_gemm_accum (beta=1.0 into FP32 main_grad, Sync wgrad)",
            "extra_grad_hbm_gb": 0.00,
            "wgrad_epilogue_overhead_ms": 1.20,
            "pp_bubble_ratio_pct": 21.5,
            "step_time_ms": 378.0,
            "relative_tflops_pct": 109.0,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 3: dgrad-First Deferred wgrad Without main_grad Accumulation Fusion",
            "extra_grad_hbm_gb": 14.00,
            "wgrad_epilogue_overhead_ms": 18.60,
            "pp_bubble_ratio_pct": 5.8,
            "step_time_ms": 348.0,
            "relative_tflops_pct": 118.4,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Fused FP32 main_grad Accum (beta=1.0) + dgrad-First Deferred wgrad",
            "extra_grad_hbm_gb": 0.00,
            "wgrad_epilogue_overhead_ms": 1.20,
            "pp_bubble_ratio_pct": 4.2,
            "step_time_ms": 323.4,
            "relative_tflops_pct": 127.4,
            "decision": "Act automatically",
        },
    ]

    pp_stages = [
        {"pp_size": 4, "sync_bubble_pct": 14.2, "deferred_wgrad_bubble_pct": 2.4},
        {"pp_size": 8, "sync_bubble_pct": 23.8, "deferred_wgrad_bubble_pct": 4.2},
        {"pp_size": 16, "sync_bubble_pct": 36.5, "deferred_wgrad_bubble_pct": 6.8},
    ]

    summary = {
        "benchmark": "llm_linear_backward_wgrad_main_grad_accumulation_fusion_dgrad_first_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "shard_params_billions": 7.0,
        "grad_accum_steps": 16,
        "pp_size": 8,
        "extra_hbm_saved_gb": 14.00,
        "naive_epilogue_ms": 18.60,
        "fused_epilogue_ms": 1.20,
        "epilogue_speedup_x": round(18.60 / 1.20, 1),
        "naive_pp_bubble_pct": 23.8,
        "deferred_pp_bubble_pct": 4.2,
        "throughput_gain_pct": 27.4,
        "tiers": tiers,
        "pp_stages": pp_stages,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_wgrad_fusion_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
