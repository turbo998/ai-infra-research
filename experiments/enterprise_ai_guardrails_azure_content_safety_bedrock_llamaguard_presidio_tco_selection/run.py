from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    monthly_requests_m = 50
    tiers = [
        {
            "tier": "Tier 1: Synchronous Serial 70B LLM Self-Critique Guardrail",
            "added_ttft_latency_ms": 460,
            "injection_block_rate_pct": 84.0,
            "pii_leak_rate_pct": 3.2,
            "monthly_guardrail_tco_usd": 82000,
        },
        {
            "tier": "Tier 2: Managed Cloud Guardrails (Azure AI Content Safety Prompt Shields / Bedrock Guardrails)",
            "added_ttft_latency_ms": 65,
            "injection_block_rate_pct": 97.8,
            "pii_leak_rate_pct": 0.15,
            "monthly_guardrail_tco_usd": 38500,
        },
        {
            "tier": "Tier 3: Self-Hosted Serial Open-Source Guardrails (Llama-Guard-3-8B + DeBERTa on AKS)",
            "added_ttft_latency_ms": 48,
            "injection_block_rate_pct": 95.4,
            "pii_leak_rate_pct": 0.20,
            "monthly_guardrail_tco_usd": 9800,
        },
        {
            "tier": "Tier 4: Hybrid Async Speculative Pipeline (Gateway Presidio Vault + Async Llama-Guard + Azure Shield Escalation)",
            "added_ttft_latency_ms": 12,
            "injection_block_rate_pct": 99.4,
            "pii_leak_rate_pct": 0.0,
            "monthly_guardrail_tco_usd": 9650,
        },
    ]
    return {
        "experiment_id": "enterprise_ai_guardrails_azure_content_safety_bedrock_llamaguard_presidio_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "monthly_requests_millions": monthly_requests_m,
            "streaming_window_tokens": 64,
            "presidio_reversible_tokenization_ms": 12,
        },
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
        print("OK: enterprise_ai_guardrails_azure_content_safety_bedrock_llamaguard_presidio_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
