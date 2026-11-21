from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    developers = 2000
    tiers = [
        {
            "tier": "Tier 1: Legacy Rule-Based SAST + Manual Security Triage",
            "false_positive_rate_pct": 78.0,
            "patch_accept_compile_pass_pct": 14.0,
            "mttr_hours": 1008.0,
            "monthly_tco_usd": 145000,
        },
        {
            "tier": "Tier 2: Raw LLM File Prompting Without CPG or Compiler Verification",
            "false_positive_rate_pct": 45.0,
            "patch_accept_compile_pass_pct": 66.0,
            "mttr_hours": 336.0,
            "monthly_tco_usd": 68000,
        },
        {
            "tier": "Tier 3: Managed Cloud AppSec AI (GitHub Copilot Autofix + GHAS CodeQL / Snyk DeepCode)",
            "false_positive_rate_pct": 12.5,
            "patch_accept_compile_pass_pct": 76.0,
            "mttr_hours": 3.5,
            "monthly_tco_usd": 98000,
        },
        {
            "tier": "Tier 4: CodeQL/Joern CPG Taint Slice + Self-Hosted 32B Triage + Sandbox Compiler-Verified Autofix",
            "false_positive_rate_pct": 4.2,
            "patch_accept_compile_pass_pct": 91.8,
            "mttr_hours": 0.3,
            "monthly_tco_usd": 14800,
        },
    ]
    return {
        "experiment_id": "enterprise_devsecops_sast_copilot_autofix_codeql_cpg_agent_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "enterprise_developers": developers,
            "monthly_prs_scanned": 40000,
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
        print("OK: enterprise_devsecops_sast_copilot_autofix_codeql_cpg_agent_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
