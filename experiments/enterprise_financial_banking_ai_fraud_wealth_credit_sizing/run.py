"""Reproducible model for #601 Enterprise Financial & Banking AI Platform Sizer."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_banking_ai(payment_tps: int = 2500, wealth_rms: int = 3000) -> dict:
    stage1_ms = 11
    stage2_slm_logprob_ms = 18
    inline_p99_ms = stage1_ms + stage2_slm_logprob_ms

    grey_zone_qps = round(payment_tps * 0.06)
    held_sar_qps = round(payment_tps * 0.012)

    l40s_gpus = max(2, math.ceil(grey_zone_qps / 80.0)) + 2
    h100_gpus = 2
    h200_gpus = max(4, math.ceil(held_sar_qps / 5.0)) + max(2, math.ceil(wealth_rms / 1500.0))
    total_gpus = l40s_gpus + h100_gpus + h200_gpus

    unified_monthly_usd = l40s_gpus * 950 + h100_gpus * 2300 + h200_gpus * 3400
    naive_inline_70b_gpus = math.ceil(payment_tps / 14.0) + 17
    naive_monthly_usd = naive_inline_70b_gpus * 3300

    return {
        "topic_id": 601,
        "payment_tps": payment_tps,
        "wealth_rms": wealth_rms,
        "inline_p99_ms": inline_p99_ms,
        "grey_zone_qps": grey_zone_qps,
        "held_sar_qps": held_sar_qps,
        "l40s_gpus": l40s_gpus,
        "h100_gpus": h100_gpus,
        "h200_gpus": h200_gpus,
        "total_gpus": total_gpus,
        "unified_monthly_usd": unified_monthly_usd,
        "naive_inline_70b_gpus": naive_inline_70b_gpus,
        "naive_monthly_usd": naive_monthly_usd,
        "savings_pct": round((1.0 - unified_monthly_usd / naive_monthly_usd) * 100, 1),
    }


def main() -> None:
    res = evaluate_banking_ai()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
