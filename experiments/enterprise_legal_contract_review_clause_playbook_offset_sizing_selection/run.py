from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    architectures = [
        {
            "arch_id": "naive_128k_full_contract_stuffing",
            "name": "Anti-Pattern: Stuffing Entire 100-Page Contract (75K Tokens) into 128K LLM Prompt",
            "risk_clause_recall_pct": 80.6,
            "hallucinated_citation_rate_pct": 11.4,
            "contract_review_p95_seconds": 68.0,
            "required_h200_gpus_for_5000_contracts_per_day": 32,
            "annual_tco_usd": 857000.0,
            "action": "Send to a human",
        },
        {
            "arch_id": "serial_clause_by_clause_70b_loop",
            "name": "Baseline: Serial Clause-by-Clause 70B Review without Playbook Prefix Caching",
            "risk_clause_recall_pct": 94.8,
            "hallucinated_citation_rate_pct": 2.4,
            "contract_review_p95_seconds": 95.0,
            "required_h200_gpus_for_5000_contracts_per_day": 24,
            "annual_tco_usd": 643000.0,
            "action": "Confirm",
        },
        {
            "arch_id": "parallel_clause_dag_playbook_radix_offset_grounding",
            "name": "Optimal: Parallel Clause DAG + Playbook Radix Prefix Cache + 8B Filter / 70B Redline + Offset Grounding",
            "risk_clause_recall_pct": 98.4,
            "hallucinated_citation_rate_pct": 0.0,
            "contract_review_p95_seconds": 8.2,
            "required_h200_gpus_for_5000_contracts_per_day": 8,
            "annual_tco_usd": 214000.0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_legal_contract_review_clause_playbook_offset_sizing_selection",
        "title": "Enterprise LegalTech Contract Review & Clause Redlining Selection Guide: 128K Stuffing vs. Serial Loop vs. Parallel Clause DAG + Playbook Radix Cache + Offset Grounding (#531)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "review_speedup_factor_vs_128k": round(68.0 / 8.2, 2),
            "gpu_saving_pct_vs_128k_stuffing": round((1.0 - 8 / 32) * 100, 1),
            "risk_recall_gain_pts": round(98.4 - 80.6, 1),
        },
        "architectures": architectures,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
