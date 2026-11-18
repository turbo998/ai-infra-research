from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "enterprise_mainframe_cobol_legacy_modernization_swe_agent_selection_v1",
        "title": "Enterprise Mainframe COBOL/CICS & Legacy Modernization SWE-Agent: Spec-First Dependency Graph, COMP-3 BigDecimal & Golden-Master Shadow Replay Evaluator",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_baseline": {
            "legacy_cobol_cics_jcl_lines_millions": 8.5,
            "legacy_programs_and_copybooks": 5900,
            "daily_core_ledger_transactions_millions": 25.0,
            "sla_target": "Functional & penny-exact equivalence >= 99.8%, COMP-3 decimal / REDEFINES regression rate == 0.0%, migration velocity >= 120 KLOC/month",
        },
        "tiers": [
            {
                "tier_id": "tier1_mechanical_line_by_line_transpiler_and_manual_rewrite",
                "name": "Tier 1: Mechanical Line-by-Line Transpiler (JOBOL) + Manual SI Rewrite",
                "functional_and_penny_equivalence_pass_rate_pct": 82.0,
                "comp3_decimal_or_redefines_regression_rate_pct": 4.5,
                "migration_velocity_kloc_per_month": 18.0,
                "monthly_modernization_and_mips_dual_run_cost_usd": 540000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier2_file_by_file_prompt_llm_direct_java_python_translation",
                "name": "Tier 2: File-by-File Prompted LLM Directly Translating COBOL to Java/Python",
                "functional_and_penny_equivalence_pass_rate_pct": 59.5,
                "comp3_decimal_or_redefines_regression_rate_pct": 18.4,
                "migration_velocity_kloc_per_month": 110.0,
                "monthly_modernization_and_mips_dual_run_cost_usd": 690000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier3_cobol_ast_rag_plus_llm_unit_test_generation",
                "name": "Tier 3: COBOL AST Chunk RAG + Self-Generated LLM Unit Tests (No Shadow Replay)",
                "functional_and_penny_equivalence_pass_rate_pct": 84.6,
                "comp3_decimal_or_redefines_regression_rate_pct": 5.2,
                "migration_velocity_kloc_per_month": 65.0,
                "monthly_modernization_and_mips_dual_run_cost_usd": 245000,
                "action_decision": "Confirm",
            },
            {
                "tier_id": "tier4_program_dependency_graph_spec_first_bigdecimal_shadow_replay_agent",
                "name": "Tier 4: JCL/CICS/Copybook Dependency Graph + Spec-First Domain Extraction + COMP-3 BigDecimal Compiler + Golden-Master Shadow Replay Agent",
                "functional_and_penny_equivalence_pass_rate_pct": 99.9,
                "comp3_decimal_or_redefines_regression_rate_pct": 0.0,
                "migration_velocity_kloc_per_month": 145.0,
                "monthly_modernization_and_mips_dual_run_cost_usd": 82000,
                "action_decision": "Act automatically",
            },
        ],
        "perturbation_check": {
            "scenario": "Core Banking Compound Interest Accrual Batch (PIC S9(9)V99 COMP-3 Truncation vs ROUNDED + Overlapping REDEFINES Copybook Union)",
            "tier2_penny_drift_or_redefines_corruption_pct": 27.6,
            "tier4_bigdecimal_shadow_replay_drift_pct": 0.0,
            "tier1_velocity_kloc_mo": 18.0,
            "tier4_velocity_kloc_mo": 145.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = build_payload()
    out_path = Path(__file__).resolve().parent / "results.json"
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            return 1
        return 0 if out_path.read_text(encoding="utf-8") == rendered else 1

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
