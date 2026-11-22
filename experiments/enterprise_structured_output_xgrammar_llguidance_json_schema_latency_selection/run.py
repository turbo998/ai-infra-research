from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Unconstrained Prompting + Client Post-Hoc JSON Parse Retry (18.4% Syntax Failure)",
            "schema_compile_ttft_overhead_ms": 0.0,
            "per_token_mask_overhead_ms": 0.0,
            "json_schema_compliance_pct": 81.6,
            "effective_goodput_tok_s": 1180,
        },
        {
            "tier": "Tier 2: First-Gen Regex FSM (Uncached Outlines Dynamic FSM Compilation on Every New Schema)",
            "schema_compile_ttft_overhead_ms": 1420.0,
            "per_token_mask_overhead_ms": 6.8,
            "json_schema_compliance_pct": 100.0,
            "effective_goodput_tok_s": 1420,
        },
        {
            "tier": "Tier 3: XGrammar (Context-Independent Adaptive Pushdown Automaton + Precompiled Bitmask Cache)",
            "schema_compile_ttft_overhead_ms": 8.5,
            "per_token_mask_overhead_ms": 0.12,
            "json_schema_compliance_pct": 100.0,
            "effective_goodput_tok_s": 2450,
        },
        {
            "tier": "Tier 4: XGrammar / llguidance + Compressed Jump-Forward Decoding (Deterministic Boilerplate Fast-Forward) + Async Overlap",
            "schema_compile_ttft_overhead_ms": 1.8,
            "per_token_mask_overhead_ms": 0.04,
            "json_schema_compliance_pct": 100.0,
            "effective_goodput_tok_s": 3120,
        },
    ]
    return {
        "experiment_id": "enterprise_structured_output_xgrammar_llguidance_json_schema_latency_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "jump_forward_token_savings_pct": 28.5,
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
        print("OK: enterprise_structured_output_xgrammar_llguidance_json_schema_latency_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
