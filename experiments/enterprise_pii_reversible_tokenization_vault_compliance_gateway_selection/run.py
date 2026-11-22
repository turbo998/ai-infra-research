from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Destructive [REDACTED] Masking (Regex Only)",
            "pii_detection_recall_pct": 74.0,
            "downstream_llm_task_accuracy_pct": 52.0,
            "gateway_overhead_ms": 1.2,
            "extra_scrubber_gpus": 0,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 2: 8B/70B LLM-Based PII Scrubber (Destructive)",
            "pii_detection_recall_pct": 97.5,
            "downstream_llm_task_accuracy_pct": 61.0,
            "gateway_overhead_ms": 385.0,
            "extra_scrubber_gpus": 16,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 3: Static Pseudonymization (No Streaming SSE Re-Hydration)",
            "pii_detection_recall_pct": 96.8,
            "downstream_llm_task_accuracy_pct": 84.5,
            "gateway_overhead_ms": 8.5,
            "extra_scrubber_gpus": 0,
            "action": "Confirm",
        },
        {
            "tier": "Tier 4: Hybrid Regex + GLiNER ONNX + Redis Vault + Streaming Aho-Corasick Re-Hydration (Recommended)",
            "pii_detection_recall_pct": 98.6,
            "downstream_llm_task_accuracy_pct": 94.2,
            "gateway_overhead_ms": 4.5,
            "extra_scrubber_gpus": 0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_pii_reversible_tokenization_vault_compliance_gateway_selection",
        "title": "Enterprise Reversible PII Tokenization (Presidio + GLiNER ONNX + Redis Vault vs. Destructive [REDACTED]) Selection (#504)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "gateway_100rps_baseline": {
            "rps": 100,
            "accuracy_gain_pts_vs_destructive": round(94.2 - 52.0, 1),
            "latency_speedup_vs_llm_scrubber": round(385.0 / 4.5, 1),
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
