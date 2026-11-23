from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    engines = [
        {
            "engine_id": "unconstrained_prompt_validation_retry_loop",
            "name": "Anti-Pattern 1: Unconstrained Prompting + JSON Parse Error Retry Loop (Up to 3x)",
            "schema_compliance_pct": 98.6,
            "ten_step_agent_success_pct": 86.8,
            "cpu_mask_overhead_ms_per_token": 0.0,
            "p99_tpot_ms": 14.2,
            "p99_e2e_latency_ms": 6420.0,
            "required_h200_gpus_at_50rps": 16,
            "action": "Send to a human",
        },
        {
            "engine_id": "legacy_python_fsm_logit_masking",
            "name": "Anti-Pattern 2: Legacy Synchronous Python FSM Logit Masking (128K Vocab Scan)",
            "schema_compliance_pct": 100.0,
            "ten_step_agent_success_pct": 100.0,
            "cpu_mask_overhead_ms_per_token": 11.8,
            "p99_tpot_ms": 25.3,
            "p99_e2e_latency_ms": 2890.0,
            "required_h200_gpus_at_50rps": 14,
            "action": "Send to a human",
        },
        {
            "engine_id": "xgrammar_llguidance_jump_forward",
            "name": "Optimal: XGrammar / llguidance C++/Rust PDA Bitmask (0.04ms) + Jump-Forward Token Splicing",
            "schema_compliance_pct": 100.0,
            "ten_step_agent_success_pct": 100.0,
            "cpu_mask_overhead_ms_per_token": 0.04,
            "p99_tpot_ms": 9.8,
            "p99_e2e_latency_ms": 1120.0,
            "required_h200_gpus_at_50rps": 8,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_structured_output_xgrammar_llguidance_constrained_decoding_selection",
        "title": "Enterprise 100% Structured Output & Constrained Decoding Selection Guide: Prompt Retry Loop vs. Legacy FSM vs. XGrammar / llguidance + Jump-Forward (#518)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "p99_e2e_speedup_vs_retry_loop": round(6420.0 / 1120.0, 2),
            "p99_tpot_speedup_vs_legacy_fsm": round(25.3 / 9.8, 2),
            "gpu_saving_pct_vs_retry_loop": round((1.0 - 8 / 16) * 100, 1),
        },
        "engines": engines,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
