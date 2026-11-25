from __future__ import annotations
import json
from pathlib import Path


def evaluate_paygo_ptu_k8s_tco_gateway_radar() -> dict:
    monthly_tokens_millions = 450_000
    paygo_monthly_usd = 216_000
    peak_ptu_monthly_usd = 142_000
    p80_ptu_hybrid_monthly_usd = 78_800
    self_hosted_k8s_tidal_monthly_usd = 23_400

    paygo_unit_usd_per_1m = round(paygo_monthly_usd / monthly_tokens_millions, 3)  # 0.480
    self_hosted_unit_usd_per_1m = round(self_hosted_k8s_tidal_monthly_usd / monthly_tokens_millions, 3)  # 0.052
    cost_reduction_x = round(paygo_unit_usd_per_1m / self_hosted_unit_usd_per_1m, 2)  # 9.23x
    net_savings_pct = round((1.0 - self_hosted_k8s_tidal_monthly_usd / paygo_monthly_usd) * 100.0, 1)  # 89.2%
    annual_savings_usd = (paygo_monthly_usd - self_hosted_k8s_tidal_monthly_usd) * 12  # 2,311,200

    ptu_crossover_utilization_pct = 28.0
    self_hosted_crossover_utilization_pct = 65.0
    round_robin_prefix_hit_pct = 22.0
    envoy_gie_prefix_hit_pct = 86.0
    envoy_gie_proxy_overhead_ms = 0.8

    return {
        "deliverable_id": "#627",
        "topic": "enterprise_paygo_ptu_k8s_tco_ai_gateway_breakthrough_radar",
        "monthly_tokens_millions": monthly_tokens_millions,
        "paygo_monthly_usd": paygo_monthly_usd,
        "peak_ptu_monthly_usd": peak_ptu_monthly_usd,
        "p80_ptu_hybrid_monthly_usd": p80_ptu_hybrid_monthly_usd,
        "self_hosted_k8s_tidal_monthly_usd": self_hosted_k8s_tidal_monthly_usd,
        "paygo_unit_usd_per_1m": paygo_unit_usd_per_1m,
        "self_hosted_unit_usd_per_1m": self_hosted_unit_usd_per_1m,
        "cost_reduction_x": cost_reduction_x,
        "net_savings_pct": net_savings_pct,
        "annual_savings_usd": annual_savings_usd,
        "ptu_crossover_utilization_pct": ptu_crossover_utilization_pct,
        "self_hosted_crossover_utilization_pct": self_hosted_crossover_utilization_pct,
        "round_robin_prefix_hit_pct": round_robin_prefix_hit_pct,
        "envoy_gie_prefix_hit_pct": envoy_gie_prefix_hit_pct,
        "envoy_gie_proxy_overhead_ms": envoy_gie_proxy_overhead_ms,
    }


if __name__ == "__main__":
    out = evaluate_paygo_ptu_k8s_tco_gateway_radar()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)
