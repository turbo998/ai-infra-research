from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: 128K Full-History Replay (Lost-in-the-Middle Contradictions)",
            "temporal_state_accuracy_pct": 62.0,
            "prompt_tokens_per_turn": 128000,
            "p95_ttft_ms": 3450.0,
            "required_h200_gpus_at_50rps": 80,
            "monthly_serving_cost_usd": 230000,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 2: Rolling Recursive Summary Only (Entity Detail Loss)",
            "temporal_state_accuracy_pct": 68.5,
            "prompt_tokens_per_turn": 3200,
            "p95_ttft_ms": 280.0,
            "required_h200_gpus_at_50rps": 6,
            "monthly_serving_cost_usd": 17250,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 3: Pure Vector Top-10 Message RAG (Stale Fact Collisions)",
            "temporal_state_accuracy_pct": 54.5,
            "prompt_tokens_per_turn": 4500,
            "p95_ttft_ms": 340.0,
            "required_h200_gpus_at_50rps": 8,
            "monthly_serving_cost_usd": 23000,
            "action": "Confirm",
        },
        {
            "tier": "Tier 4: 3-Tier Bi-Temporal Memory (6-Turn Buffer + Async 8B + Zep/Mem0 Graph)",
            "temporal_state_accuracy_pct": 94.8,
            "prompt_tokens_per_turn": 1850,
            "p95_ttft_ms": 195.0,
            "required_h200_gpus_at_50rps": 4,
            "monthly_serving_cost_usd": 11500,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_agentic_memory_mem0_zep_bitemporal_graph_kv_sizing_selection",
        "title": "Enterprise Long-Horizon Agentic Memory (Mem0/Zep Bi-Temporal Graph vs. Vector History vs. 128K Replay) Sizer (#501)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_50rps_100k_users": {
            "active_users": 100000,
            "token_reduction_pct_vs_128k": round((1.0 - 1850 / 128000) * 100, 1),
            "accuracy_gain_pts_vs_vector_rag": round(94.8 - 54.5, 1),
            "gpu_saving_pct_vs_128k": round((1.0 - 4 / 80) * 100, 1),
        },
        "tiers": tiers,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
