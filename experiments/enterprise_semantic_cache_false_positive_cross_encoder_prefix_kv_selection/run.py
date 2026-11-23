from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    architectures = [
        {
            "arch_id": "naive_cosine_085_semantic_cache",
            "name": "Anti-Pattern: Naive Dense Cosine >= 0.85 Semantic Cache (No Entity/Negation Check)",
            "response_cache_hit_rate_pct": 42.0,
            "false_positive_wrong_answer_pct": 14.2,
            "p95_ttft_ms": 185.0,
            "required_h200_gpus_at_60rps": 12,
            "annual_tco_usd": 338000.0,
            "action": "Send to a human",
        },
        {
            "arch_id": "no_cache_full_llm_every_request",
            "name": "Baseline: Zero Semantic Cache & Zero Prefix KV Cache (Recompute Every Request)",
            "response_cache_hit_rate_pct": 0.0,
            "false_positive_wrong_answer_pct": 0.0,
            "p95_ttft_ms": 920.0,
            "required_h200_gpus_at_60rps": 32,
            "annual_tco_usd": 857000.0,
            "action": "Send to a human",
        },
        {
            "arch_id": "two_layer_entity_cross_encoder_plus_radix_kv_cache",
            "name": "Optimal Dual-Layer: Tenant/Entity-Guarded Cross-Encoder (>=0.96) + SGLang Radix Prefix KV Cache",
            "response_cache_hit_rate_pct": 28.5,
            "false_positive_wrong_answer_pct": 0.08,
            "p95_ttft_ms": 145.0,
            "required_h200_gpus_at_60rps": 10,
            "annual_tco_usd": 282000.0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_semantic_cache_false_positive_cross_encoder_prefix_kv_selection",
        "title": "Enterprise Semantic Cache False-Positive Governance & Exact Prefix KV Cache Dual-Layer Selection Guide (#524)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "false_positive_reduction_factor": round(14.2 / 0.08, 1),
            "gpu_saving_pct_vs_no_cache": round((1.0 - 10 / 32) * 100, 1),
            "ttft_speedup_factor_vs_no_cache": round(920.0 / 145.0, 2),
        },
        "architectures": architectures,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
