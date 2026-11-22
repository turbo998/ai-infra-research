from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    strategies = [
        {
            "strategy": "Strategy 1: 3-Year Rip-and-Replace Air-Cooled Hall (Scrap + Rebuild)",
            "five_year_total_tco_musd": 68.4,
            "blended_cost_per_1m_tokens_usd": 0.145,
            "gpu_economic_lifespan_years": 3.0,
            "stranded_facility_capex_pct": 42.0,
            "action": "Send to a human",
        },
        {
            "strategy": "Strategy 2: Freeze on Hopper-Only Air-Cooled Fleet (No Blackwell/Rubin)",
            "five_year_total_tco_musd": 54.0,
            "blended_cost_per_1m_tokens_usd": 0.240,
            "gpu_economic_lifespan_years": 4.0,
            "stranded_facility_capex_pct": 0.0,
            "action": "Send to a human",
        },
        {
            "strategy": "Strategy 3: Unplanned Mixed-Generation Silos (Isolated H200 & GB200)",
            "five_year_total_tco_musd": 49.2,
            "blended_cost_per_1m_tokens_usd": 0.072,
            "gpu_economic_lifespan_years": 3.8,
            "stranded_facility_capex_pct": 18.5,
            "action": "Confirm",
        },
        {
            "strategy": "Strategy 4: 150kW DLC Facility + 5-Year 3-Stage Role Cascade (H200 Prefill/FIM + GB200/Rubin Decode)",
            "five_year_total_tco_musd": 36.8,
            "blended_cost_per_1m_tokens_usd": 0.035,
            "gpu_economic_lifespan_years": 5.0,
            "stranded_facility_capex_pct": 0.0,
            "action": "Act automatically",
        },
    ]

    generations = [
        {
            "gen": "Hopper HGX H200 (8-GPU NVL8)",
            "hbm_bw_tbps_per_gpu": 4.8,
            "nvlink_domain_gpus": 8,
            "rack_power_kw": 10.2,
            "cost_per_1m_tokens_usd": 0.240,
            "cascade_year_3_5_role": "Prefill Pool / 32B-8B Student / IDE FIM / Reranker",
        },
        {
            "gen": "Blackwell HGX B200 (8-GPU NVL8)",
            "hbm_bw_tbps_per_gpu": 8.0,
            "nvlink_domain_gpus": 8,
            "rack_power_kw": 15.5,
            "cost_per_1m_tokens_usd": 0.110,
            "cascade_year_3_5_role": "High-Throughput Prefill + Dense 70B/405B Serving",
        },
        {
            "gen": "Blackwell GB200 NVL72 (72-GPU Rack)",
            "hbm_bw_tbps_per_gpu": 8.0,
            "nvlink_domain_gpus": 72,
            "rack_power_kw": 132.0,
            "cost_per_1m_tokens_usd": 0.038,
            "cascade_year_3_5_role": "671B/1T MoE Ultra-Low-TPOT Decode & RL Rollout",
        },
        {
            "gen": "Vera Rubin NVL144 Ready (HBM4 + 1.6T CPO)",
            "hbm_bw_tbps_per_gpu": 16.0,
            "nvlink_domain_gpus": 144,
            "rack_power_kw": 180.0,
            "cost_per_1m_tokens_usd": 0.018,
            "cascade_year_3_5_role": "Next-Gen 2T+ MoE & Million-Token Reasoning Decode",
        },
    ]

    return {
        "experiment_id": "enterprise_nextgen_ai_factory_h200_gb200_rubin_5yr_cascade_tco_selection",
        "title": "Milestone #495: Enterprise 2026-2027 Next-Gen AI Factory Evolution (H200 -> GB200 NVL72 -> Rubin NVL144) & 5-Year Role-Cascade TCO Blueprint",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "five_year_tco_savings_musd": round(68.4 - 36.8, 1),
            "five_year_tco_reduction_pct": round((1.0 - 36.8 / 68.4) * 100, 1),
            "token_cost_reduction_factor": round(0.240 / 0.035, 2),
        },
        "strategies": strategies,
        "generations": generations,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
