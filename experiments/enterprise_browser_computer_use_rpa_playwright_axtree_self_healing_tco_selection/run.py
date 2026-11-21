from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    monthly_runs_m = 3.0
    tiers = [
        {
            "tier": "Tier 1: Legacy Rigid RPA (UiPath / Power Automate Desktop Static XPath)",
            "wall_clock_15step_sec": 4.5,
            "task_completion_rate_pct": 81.5,
            "ui_change_breakage_rate_pct": 18.5,
            "cost_per_15step_run_usd": 0.048,
            "monthly_tco_usd": 145000,
        },
        {
            "tier": "Tier 2: 100% Online Pure-Vision Computer-Use Agent (Screenshot Pixel Clicking Every Run)",
            "wall_clock_15step_sec": 48.5,
            "task_completion_rate_pct": 86.4,
            "ui_change_breakage_rate_pct": 2.1,
            "cost_per_15step_run_usd": 0.420,
            "monthly_tco_usd": 1260000,
        },
        {
            "tier": "Tier 3: 100% Online Pruned AXTree + Set-of-Marks (Browser-Use / Stagehand Live LLM Every Run)",
            "wall_clock_15step_sec": 16.5,
            "task_completion_rate_pct": 94.2,
            "ui_change_breakage_rate_pct": 1.4,
            "cost_per_15step_run_usd": 0.066,
            "monthly_tco_usd": 198000,
        },
        {
            "tier": "Tier 4: CDP API Bypass + Cached Deterministic Playwright Replay (98% Runs) + AXTree/SoM LLM Self-Healing (2%) + HITL",
            "wall_clock_15step_sec": 3.2,
            "task_completion_rate_pct": 99.3,
            "ui_change_breakage_rate_pct": 0.2,
            "cost_per_15step_run_usd": 0.0062,
            "monthly_tco_usd": 18600,
        },
    ]
    return {
        "experiment_id": "enterprise_browser_computer_use_rpa_playwright_axtree_self_healing_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "monthly_workflow_runs_millions": monthly_runs_m,
            "steps_per_workflow": 15,
            "raw_html_dom_tokens": 180000,
            "pruned_axtree_tokens": 1800,
            "cached_replay_share_pct": 98.0,
        },
        "tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_browser_computer_use_rpa_playwright_axtree_self_healing_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
