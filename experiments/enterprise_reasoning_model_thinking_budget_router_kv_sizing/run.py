"""Reproducible model for #592 Enterprise Reasoning Model <think> Budget Router & Multi-Turn KV Stripping."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_reasoning_budget(peak_qps: int = 50) -> dict:
    unconstrained_out_tokens = 3850
    routed_out_tokens = round(0.55 * 350 + 0.30 * 990 + 0.15 * 3000)
    token_reduction_pct = round((1.0 - routed_out_tokens / unconstrained_out_tokens) * 100, 1)

    turn6_unstripped_tokens = 28400
    turn6_stripped_tokens = 7320
    kv_reduction_pct = round((1.0 - turn6_stripped_tokens / turn6_unstripped_tokens) * 100, 1)

    unconstrained_h200_gpus = max(4, math.ceil((peak_qps * unconstrained_out_tokens) / 6000.0))
    routed_h200_gpus = max(2, math.ceil((peak_qps * routed_out_tokens) / 6000.0))

    unconstrained_monthly_usd = unconstrained_h200_gpus * 3400
    routed_monthly_usd = routed_h200_gpus * 3400

    return {
        "topic_id": 592,
        "peak_qps": peak_qps,
        "unconstrained_out_tokens": unconstrained_out_tokens,
        "routed_out_tokens": routed_out_tokens,
        "token_reduction_pct": token_reduction_pct,
        "turn6_unstripped_tokens": turn6_unstripped_tokens,
        "turn6_stripped_tokens": turn6_stripped_tokens,
        "kv_reduction_pct": kv_reduction_pct,
        "unconstrained_h200_gpus": unconstrained_h200_gpus,
        "routed_h200_gpus": routed_h200_gpus,
        "unconstrained_monthly_usd": unconstrained_monthly_usd,
        "routed_monthly_usd": routed_monthly_usd,
        "gpu_savings_pct": round((1.0 - routed_h200_gpus / unconstrained_h200_gpus) * 100, 1),
    }


def main() -> None:
    res = evaluate_reasoning_budget()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
