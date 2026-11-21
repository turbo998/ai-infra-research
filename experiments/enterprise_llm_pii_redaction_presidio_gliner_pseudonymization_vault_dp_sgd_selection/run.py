from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "enterprise_llm_pii_redaction_presidio_gliner_pseudonymization_vault_dp_sgd_selection",
        "title": "Enterprise LLM PII Redaction & Reversible Pseudonymization Selection: Presidio + GLiNER Bi-Directional Vault vs. Destructive Masking & DP-SGD",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "prompt_tokens": 2048,
            "entities_per_document": 18,
            "compliance_scope": "GDPR / HIPAA / PIPL Cross-Border & Cloud LLM Gateway",
        },
        "tiers": [
            {
                "tier": "Tier 1: Naive Regex Destructive Masking (Replace All Matches With '[REDACTED]')",
                "pii_detection_recall_pct": 66.0,
                "downstream_llm_task_utility_pct": 61.5,
                "gateway_added_latency_ms": 2.5,
                "supports_reversible_streaming_restore": False,
                "recommended_scenario": "Anti-pattern (misses unstructured names/addresses and breaks multi-entity reasoning)",
            },
            {
                "tier": "Tier 2: Synchronous 8B LLM Prompt Scrubber Before Cloud API Call",
                "pii_detection_recall_pct": 95.2,
                "downstream_llm_task_utility_pct": 82.0,
                "gateway_added_latency_ms": 420.0,
                "supports_reversible_streaming_restore": False,
                "recommended_scenario": "Offline batch dataset scrubbing where +420ms latency is acceptable",
            },
            {
                "tier": "Tier 3: Differential Privacy SFT Training (DP-SGD epsilon=3.0 Per-Sample Grad Clipping)",
                "pii_detection_recall_pct": 91.0,
                "downstream_llm_task_utility_pct": 88.2,
                "gateway_added_latency_ms": 0.0,
                "supports_reversible_streaming_restore": False,
                "recommended_scenario": "Preventing training-set memorization in custom fine-tuned weights (still needs PII scrubbing first)",
            },
            {
                "tier": "Tier 4 (Recommended): Reversible Pseudonymization Vault (Presidio Checksums + GLiNER ONNX + Aho-Corasick Stream Restore)",
                "pii_detection_recall_pct": 98.6,
                "downstream_llm_task_utility_pct": 94.8,
                "gateway_added_latency_ms": 11.5,
                "supports_reversible_streaming_restore": True,
                "recommended_scenario": "Production AI Gateway protecting cloud LLM calls with 98.6% PII recall & 94.8% utility in +11.5ms",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
