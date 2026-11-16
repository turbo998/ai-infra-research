from __future__ import annotations

import json
from pathlib import Path


def evaluate_streaming_tool_parser_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Per-Token Full-String Regex + partial_json_parser + Diff (O(L^2))",
            "cpu_parse_latency_4k_ms": 42.50,
            "argument_stream_corruption_pct": 16.4,
            "parallel_tool_index_acc_pct": 83.6,
            "complexity_per_step": "O(B * L)",
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Pre-Compiled Regex on Full Buffer + Naive String-Diff Slicing",
            "cpu_parse_latency_4k_ms": 14.80,
            "argument_stream_corruption_pct": 8.2,
            "parallel_tool_index_acc_pct": 91.8,
            "complexity_per_step": "O(B * L)",
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: Buffered Non-Streaming Tool Extraction (Emit Only at </tool_call>)",
            "cpu_parse_latency_4k_ms": 0.45,
            "argument_stream_corruption_pct": 0.0,
            "parallel_tool_index_acc_pct": 100.0,
            "complexity_per_step": "O(1) (6.5s Freeze)",
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: O(delta_l) Incremental Pushdown JSON Lexer + Escape Holdback",
            "cpu_parse_latency_4k_ms": 0.09,
            "argument_stream_corruption_pct": 0.0,
            "parallel_tool_index_acc_pct": 100.0,
            "complexity_per_step": "O(B * delta_l)",
            "decision": "Act automatically",
        },
    ]

    arg_length_scaling = [
        {"arg_tokens": 256, "quadratic_reparse_ms": 1.85, "incremental_lexer_ms": 0.08},
        {"arg_tokens": 1024, "quadratic_reparse_ms": 8.40, "incremental_lexer_ms": 0.08},
        {"arg_tokens": 4096, "quadratic_reparse_ms": 42.50, "incremental_lexer_ms": 0.09},
        {"arg_tokens": 8192, "quadratic_reparse_ms": 158.00, "incremental_lexer_ms": 0.09},
    ]

    summary = {
        "benchmark": "llm_serving_streaming_tool_call_parser_quadratic_regex_partial_json_lexer_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "batch_size": 64,
        "arg_tokens": 4096,
        "quadratic_4k_ms": 42.50,
        "incremental_4k_ms": 0.09,
        "speedup_4k_x": round(42.50 / 0.09, 1),
        "naive_corruption_pct": 16.4,
        "lexer_corruption_pct": 0.0,
        "tiers": tiers,
        "arg_length_scaling": arg_length_scaling,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_streaming_tool_parser_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
