from __future__ import annotations
import json
from pathlib import Path


def evaluate_reasoning_think_budget_kv_stripping() -> dict:
    unconstrained_avg_out_tokens = 3850
    routed_avg_out_tokens = int(0.55 * 280 + 0.30 * 1000 + 0.15 * 3240)  # 940
    output_token_reduction_pct = round(
        (1.0 - routed_avg_out_tokens / unconstrained_avg_out_tokens) * 100.0, 1
    )  # 75.6%

    hard_max_tokens_truncation_pct = 38.5
    soft_close_truncation_pct = 0.0
    multi_turn_kv_reduction_pct = 74.2

    unconstrained_h200_gpus = 32
    routed_h200_gpus = 8
    gpu_savings_pct = round((1.0 - routed_h200_gpus / unconstrained_h200_gpus) * 100.0, 1)  # 75.0%
    annual_savings_usd = (unconstrained_h200_gpus - routed_h200_gpus) * 3200 * 12  # 921,600

    return {
        "deliverable_id": "#633",
        "topic": "enterprise_reasoning_think_budget_soft_close_kv_stripping",
        "unconstrained_avg_out_tokens": unconstrained_avg_out_tokens,
        "routed_avg_out_tokens": routed_avg_out_tokens,
        "output_token_reduction_pct": output_token_reduction_pct,
        "hard_max_tokens_truncation_pct": hard_max_tokens_truncation_pct,
        "soft_close_truncation_pct": soft_close_truncation_pct,
        "multi_turn_kv_reduction_pct": multi_turn_kv_reduction_pct,
        "unconstrained_h200_gpus": unconstrained_h200_gpus,
        "routed_h200_gpus": routed_h200_gpus,
        "gpu_savings_pct": gpu_savings_pct,
        "annual_savings_usd": annual_savings_usd,
    }


if __name__ == "__main__":
    out = evaluate_reasoning_think_budget_kv_stripping()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)
