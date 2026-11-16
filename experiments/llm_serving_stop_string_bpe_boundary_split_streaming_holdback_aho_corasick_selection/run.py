from __future__ import annotations

import json
from pathlib import Path


def evaluate_stop_string_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Token-ID Only Stop Matching (tokenizer.encode(stop))",
            "missed_stop_rate_pct": 31.4,
            "streaming_prefix_leak_pct": 0.0,
            "cpu_match_overhead_ms": 0.08,
            "subtoken_overrun_chars": 8.4,
            "react_agent_success_pct": 64.2,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Full-String O(L) Python .find() Without Holdback",
            "missed_stop_rate_pct": 0.0,
            "streaming_prefix_leak_pct": 100.0,
            "cpu_match_overhead_ms": 18.40,
            "subtoken_overrun_chars": 0.0,
            "react_agent_success_pct": 72.5,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: Fixed Max-Length Blind Holdback + Tail Slice",
            "missed_stop_rate_pct": 0.0,
            "streaming_prefix_leak_pct": 0.0,
            "cpu_match_overhead_ms": 1.45,
            "subtoken_overrun_chars": 0.0,
            "react_agent_success_pct": 94.0,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Incremental Aho-Corasick + Dynamic Prefix-Overlap Holdback",
            "missed_stop_rate_pct": 0.0,
            "streaming_prefix_leak_pct": 0.0,
            "cpu_match_overhead_ms": 0.12,
            "subtoken_overrun_chars": 0.0,
            "react_agent_success_pct": 99.6,
            "decision": "Act automatically",
        },
    ]

    bpe_boundary_cases = [
        {
            "case": "Standalone '\\nObservation:'",
            "token_pieces": ["\\n", "Observation", ":"],
            "token_id_match": True,
            "tail_automaton_match": True,
        },
        {
            "case": "Merged Prefix '.\\nObservation:' ('.\\n' merged into 1 token)",
            "token_pieces": [".\\n", "Observation", ":"],
            "token_id_match": False,
            "tail_automaton_match": True,
        },
        {
            "case": "Merged Suffix '\\nObservation:\\n' (':\\n' merged into 1 token)",
            "token_pieces": ["\\n", "Observation", ":\\n"],
            "token_id_match": False,
            "tail_automaton_match": True,
        },
        {
            "case": "Sub-Word Split '</tool_call>\\n' ('</', 'tool', '_call', '>\\n')",
            "token_pieces": ["</", "tool", "_call", ">\\n"],
            "token_id_match": False,
            "tail_automaton_match": True,
        },
    ]

    summary = {
        "benchmark": "llm_serving_stop_string_bpe_boundary_split_streaming_holdback_aho_corasick_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "batch_size": 256,
        "stop_strings_per_request": 8,
        "context_chars": 4096,
        "token_id_missed_stop_pct": 31.4,
        "full_scan_cpu_ms": 18.40,
        "aho_corasick_cpu_ms": 0.12,
        "cpu_speedup_x": round(18.40 / 0.12, 1),
        "tiers": tiers,
        "bpe_boundary_cases": bpe_boundary_cases,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_stop_string_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
