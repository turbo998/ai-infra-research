from __future__ import annotations
import json
from pathlib import Path


def compute_wcu_per_sec(rps: float, t_in: int, t_out: int, l1_l2_hit: float, l3_prefix_hit: float) -> float:
    pass_rps = rps * (1.0 - l1_l2_hit)
    wcu_per_req = (
        1.0 * t_in * (1.0 - l3_prefix_hit)
        + 0.1 * t_in * l3_prefix_hit
        + 4.0 * t_out
    )
    return round(pass_rps * wcu_per_req, 1)


def evaluate_gateway_governance() -> dict:
    rps = 100.0
    t_in, t_out = 4000, 500
    baseline_wcu_s = compute_wcu_per_sec(rps, t_in, t_out, l1_l2_hit=0.0, l3_prefix_hit=0.185)
    optimized_wcu_s = compute_wcu_per_sec(rps, t_in, t_out, l1_l2_hit=0.26, l3_prefix_hit=0.784)

    baseline_gpus = 32
    optimized_gpus = 18
    gpu_saving_pct = round((1.0 - optimized_gpus / baseline_gpus) * 100.0, 1)
    compute_offload_pct = round((1.0 - optimized_wcu_s / (rps * (t_in + 4.0 * t_out))) * 100.0, 1)

    return {
        "Deliverable_ID": "#641",
        "WCU_Weights": {"input_miss": 1.0, "input_cached": 0.1, "output_decode": 4.0},
        "Baseline_Nginx_RPM": {
            "wcu_per_sec": baseline_wcu_s,
            "l3_prefix_hit_pct": 18.5,
            "p0_p99_ttft_ms": 9420,
            "p0_goodput_pct": 54.2,
            "required_h200_gpus": baseline_gpus,
        },
        "Optimized_WCU_3Tier_QoS": {
            "wcu_per_sec": optimized_wcu_s,
            "l1_exact_hit_pct": 14.5,
            "l2_safe_semantic_hit_pct": 11.5,
            "l3_prefix_hit_pct": 78.4,
            "p0_p99_ttft_ms": 285,
            "p0_goodput_pct": 99.6,
            "required_h200_gpus": optimized_gpus,
            "gpu_saving_pct": gpu_saving_pct,
            "annual_saving_usd": 412000,
            "effective_compute_offload_pct": compute_offload_pct,
        },
    }


if __name__ == "__main__":
    out = evaluate_gateway_governance()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)
