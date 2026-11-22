from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    levels = [
        {
            "level": "Level 1: Ad-Hoc Siloed GPU Prototypes (Colocated BF16, No Cache, Raw PAYGO)",
            "blended_cost_per_1m_tokens_usd": 0.680,
            "cluster_effective_goodput_pct": 18.5,
            "p99_sla_compliance_pct": 64.0,
            "three_year_enterprise_tco_musd": 28.5,
            "action": "Send to a human",
        },
        {
            "level": "Level 2: Basic Containerized vLLM Cluster (Static Quotas, Standard FP16)",
            "blended_cost_per_1m_tokens_usd": 0.240,
            "cluster_effective_goodput_pct": 46.5,
            "p99_sla_compliance_pct": 88.5,
            "three_year_enterprise_tco_musd": 16.4,
            "action": "Send to a human",
        },
        {
            "level": "Level 3: Disaggregated PD + FP8 + Radix Prefix Cache",
            "blended_cost_per_1m_tokens_usd": 0.068,
            "cluster_effective_goodput_pct": 78.0,
            "p99_sla_compliance_pct": 97.2,
            "three_year_enterprise_tco_musd": 9.2,
            "action": "Confirm",
        },
        {
            "level": "Level 4: Milestone #500 Full-Stack AI Factory (PD + FP8/KIVI + MTP + Cascade + WCU QoS + Tidal)",
            "blended_cost_per_1m_tokens_usd": 0.028,
            "cluster_effective_goodput_pct": 91.2,
            "p99_sla_compliance_pct": 99.6,
            "three_year_enterprise_tco_musd": 5.8,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_milestone_500_ai_infra_architect_master_encyclopedia_selection",
        "title": "Milestone #500: Cloud & AI Infrastructure Solution Architect 500-Playbook Master Encyclopedia & Universal Decision Engine",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "milestone_500_summary": {
            "total_bilingual_playbooks": 500,
            "token_cost_reduction_factor_vs_level1": round(0.680 / 0.028, 2),
            "three_year_tco_saving_pct_vs_level1": round((1.0 - 5.8 / 28.5) * 100, 1),
            "three_year_tco_saving_musd_vs_level1": round(28.5 - 5.8, 1),
        },
        "levels": levels,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
