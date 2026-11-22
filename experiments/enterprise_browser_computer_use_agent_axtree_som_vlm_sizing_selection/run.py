from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Pure 1080p Screenshot Loop + 72B VLM (Every Step)",
            "workflow_success_rate_pct": 71.2,
            "workflow_e2el_sec": 48.5,
            "tokens_per_15step_workflow": 38500,
            "required_h200_gpus_500_sessions": 64,
            "cost_per_1000_workflows_usd": 145.0,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 2: Raw Unpruned HTML Dump (32K HTML/Step) + Text LLM",
            "workflow_success_rate_pct": 64.5,
            "workflow_e2el_sec": 34.0,
            "tokens_per_15step_workflow": 185000,
            "required_h200_gpus_500_sessions": 48,
            "cost_per_1000_workflows_usd": 110.0,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 3: Uniform Set-of-Marks (SoM) Screenshot + 7B UI-VLM",
            "workflow_success_rate_pct": 88.4,
            "workflow_e2el_sec": 14.2,
            "tokens_per_15step_workflow": 22000,
            "required_h200_gpus_500_sessions": 16,
            "cost_per_1000_workflows_usd": 28.0,
            "action": "Confirm",
        },
        {
            "tier": "Tier 4: Hybrid Pruned AXTree (85%) + SoM 7B UI-VLM (15%) + Warm CDP Pool (Recommended)",
            "workflow_success_rate_pct": 95.6,
            "workflow_e2el_sec": 7.8,
            "tokens_per_15step_workflow": 7400,
            "required_h200_gpus_500_sessions": 6,
            "cost_per_1000_workflows_usd": 8.2,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_browser_computer_use_agent_axtree_som_vlm_sizing_selection",
        "title": "Enterprise Browser-Use & GUI Computer-Use Agent (Pruned AXTree + SoM 7B UI-VLM vs. 1080p VLM Loop) Sizer (#502)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_500_concurrent_browsers": {
            "concurrent_sessions": 500,
            "steps_per_workflow": 15,
            "speedup_factor_vs_1080p_vlm": round(48.5 / 7.8, 2),
            "success_rate_gain_pts": round(95.6 - 71.2, 1),
            "gpu_saving_pct_vs_1080p_vlm": round((1.0 - 6 / 64) * 100, 1),
        },
        "tiers": tiers,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
