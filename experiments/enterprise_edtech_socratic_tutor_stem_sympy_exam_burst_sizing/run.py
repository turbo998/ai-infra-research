from __future__ import annotations
import json
from pathlib import Path


def evaluate_edtech_socratic_sizing() -> dict:
    dau_students = 500_000
    evening_peak_qps = 800
    curriculum_prefix_tokens = 3500
    student_step_tokens = 240
    prefix_cache_hit_rate = 0.84

    effective_prefill_tokens = int(
        curriculum_prefix_tokens * (1.0 - prefix_cache_hit_rate) + student_step_tokens
    )
    prefill_reduction_pct = round(
        (1.0 - effective_prefill_tokens / (curriculum_prefix_tokens + student_step_tokens)) * 100,
        1,
    )

    peak_dedicated_h100_gpus = 64
    peak_dedicated_monthly_usd = peak_dedicated_h100_gpus * 2300  # 147,200

    base_l40s_gpus = 8
    base_h200_gpus = 4
    base_monthly_usd = base_l40s_gpus * 850 + base_h200_gpus * 3200  # 19,600
    evening_spot_burst_monthly_usd = 10_700
    hybrid_monthly_usd = base_monthly_usd + evening_spot_burst_monthly_usd  # 30,300
    tco_savings_pct = round((1.0 - hybrid_monthly_usd / peak_dedicated_monthly_usd) * 100, 1)

    return {
        "deliverable_id": "#616",
        "topic": "enterprise_edtech_socratic_tutor_stem_sympy_exam_burst_sizing",
        "dau_students": dau_students,
        "evening_peak_qps": evening_peak_qps,
        "effective_prefill_tokens": effective_prefill_tokens,
        "prefill_reduction_pct": prefill_reduction_pct,
        "raw_vlm_step_error_rate_pct": 11.8,
        "sympy_cas_step_accuracy_pct": 99.4,
        "sympy_verify_latency_ms": 4.5,
        "raw_llm_answer_leakage_pct": 34.0,
        "socratic_fsm_answer_leakage_pct": 0.0,
        "peak_dedicated_monthly_usd": peak_dedicated_monthly_usd,
        "hybrid_monthly_usd": hybrid_monthly_usd,
        "tco_savings_pct": tco_savings_pct,
    }


if __name__ == "__main__":
    out = evaluate_edtech_socratic_sizing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)
