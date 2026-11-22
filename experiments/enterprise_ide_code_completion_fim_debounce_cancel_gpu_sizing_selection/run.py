from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    ide_tiers = [
        {
            "tier": "Tier 1: Un-Debounced Keystroke FIM Routed to Shared 70B Chat Pool (No In-Flight Cancel)",
            "fim_p99_e2e_latency_ms": 1840.0,
            "developer_suggestion_acceptance_rate_pct": 11.2,
            "gpus_required_for_10k_developers": 96,
            "wasted_obsolete_keystroke_compute_pct": 78.5,
        },
        {
            "tier": "Tier 2: Fixed 300ms Timer Debounce on Shared 32B Pool (No Syntax Boundary Awareness)",
            "fim_p99_e2e_latency_ms": 590.0,
            "developer_suggestion_acceptance_rate_pct": 21.5,
            "gpus_required_for_10k_developers": 40,
            "wasted_obsolete_keystroke_compute_pct": 42.0,
        },
        {
            "tier": "Tier 3: Dedicated 7B/14B FP8 FIM Pool + 150ms Debounce + HTTP/2 In-Flight Cancel",
            "fim_p99_e2e_latency_ms": 215.0,
            "developer_suggestion_acceptance_rate_pct": 33.4,
            "gpus_required_for_10k_developers": 16,
            "wasted_obsolete_keystroke_compute_pct": 4.8,
        },
        {
            "tier": "Tier 4: Tree-Sitter Syntax-Aware Debounce + In-Flight Cancel + 7B/14B FP8 + PLD Speculative + File-Prefix Cache",
            "fim_p99_e2e_latency_ms": 138.0,
            "developer_suggestion_acceptance_rate_pct": 38.6,
            "gpus_required_for_10k_developers": 8,
            "wasted_obsolete_keystroke_compute_pct": 0.9,
        },
    ]
    return {
        "experiment_id": "enterprise_ide_code_completion_fim_debounce_cancel_gpu_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "active_developers": 10000,
        "ide_tiers": ide_tiers,
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
        print("OK: enterprise_ide_code_completion_fim_debounce_cancel_gpu_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
