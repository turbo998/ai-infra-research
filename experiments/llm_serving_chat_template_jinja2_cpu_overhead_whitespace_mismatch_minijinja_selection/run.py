from __future__ import annotations

import json
from pathlib import Path


def evaluate_chat_template_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Python Jinja2 AST + Uncached Tool JSON + add_special_tokens=True",
            "template_cpu_latency_ms": 16.80,
            "single_core_max_qps": 59.5,
            "prefix_cache_hit_rate_pct": 62.0,
            "tool_call_accuracy_pct": 89.2,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Compiled Python Jinja2 + add_special_tokens=False (Unsorted Tool Dicts)",
            "template_cpu_latency_ms": 7.40,
            "single_core_max_qps": 135.1,
            "prefix_cache_hit_rate_pct": 62.0,
            "tool_call_accuracy_pct": 93.3,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: GIL-Free Rust minijinja + Canonical Sorted JSON (No Token Prefix Cache)",
            "template_cpu_latency_ms": 1.15,
            "single_core_max_qps": 869.6,
            "prefix_cache_hit_rate_pct": 96.0,
            "tool_call_accuracy_pct": 96.4,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Rust minijinja + Canonical Pre-Tokenized Tool Cache + Trailing-Space Guard",
            "template_cpu_latency_ms": 0.30,
            "single_core_max_qps": 3333.3,
            "prefix_cache_hit_rate_pct": 96.0,
            "tool_call_accuracy_pct": 96.4,
            "decision": "Act automatically",
        },
    ]

    edge_cases = [
        {
            "case": "encode(chat_str, add_special_tokens=True) on Llama-3 / DeepSeek",
            "naive_defect": "Prepends duplicate <|begin_of_text|> token",
            "accuracy_drop_pt": -4.1,
        },
        {
            "case": "Trailing space after '<|im_start|>assistant\\n '",
            "naive_defect": "Forces model to avoid leading space token (' Let' vs 'Let')",
            "accuracy_drop_pt": -3.1,
        },
        {
            "case": "Assistant prefill with add_generation_prompt=True instead of continue_final_message=True",
            "naive_defect": "Appends <|eot_id|><|start_header_id|>assistant closing turn prematurely",
            "accuracy_drop_pt": -18.5,
        },
        {
            "case": "40 MCP tool JSON schemas with non-canonical dict key ordering",
            "naive_defect": "Invalidates GPU prefix cache across identical tool sets",
            "prefix_hit_drop_pt": -34.0,
        },
    ]

    summary = {
        "benchmark": "llm_serving_chat_template_jinja2_cpu_overhead_whitespace_mismatch_minijinja_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "tools_per_request": 40,
        "conversation_turns": 12,
        "python_jinja2_ms": 16.80,
        "rust_minijinja_cached_ms": 0.30,
        "speedup_x": round(16.80 / 0.30, 1),
        "naive_accuracy_pct": 89.2,
        "clean_accuracy_pct": 96.4,
        "accuracy_gain_pt": round(96.4 - 89.2, 1),
        "tiers": tiers,
        "edge_cases": edge_cases,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_chat_template_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
