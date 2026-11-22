from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    guardrail_architectures = [
        {
            "architecture": "Tier 1: Synchronous Blocking LLM-as-a-Judge Input + Full-Response Output Buffer (Wait for Full Completion)",
            "added_ttft_p99_ms": 2140.0,
            "jailbreak_prompt_injection_recall_pct": 98.6,
            "wasted_main_llm_compute_pct": 0.0,
            "streaming_sse_supported": False,
        },
        {
            "architecture": "Tier 2: Synchronous Blocking Input Guardrail (Llama-Guard-3 8B Before Main LLM) + Sentence-Buffered Output",
            "added_ttft_p99_ms": 185.0,
            "jailbreak_prompt_injection_recall_pct": 97.8,
            "wasted_main_llm_compute_pct": 0.0,
            "streaming_sse_supported": True,
        },
        {
            "architecture": "Tier 3: CPU Regex / Keyword Filter Only (Fast but Easily Bypassed by Base64 / Multi-Turn Jailbreaks)",
            "added_ttft_p99_ms": 2.4,
            "jailbreak_prompt_injection_recall_pct": 41.0,
            "wasted_main_llm_compute_pct": 0.0,
            "streaming_sse_supported": True,
        },
        {
            "architecture": "Tier 4: Speculative Parallel Input Shield (0ms Added TTFT) + 16-Token Sliding-Window Async Output Circuit Breaker",
            "added_ttft_p99_ms": 0.8,
            "jailbreak_prompt_injection_recall_pct": 98.4,
            "wasted_main_llm_compute_pct": 0.9,
            "streaming_sse_supported": True,
        },
    ]
    return {
        "experiment_id": "enterprise_ai_safety_guardrails_speculative_parallel_shield_ttft_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "attack_traffic_rate_pct": 1.2,
        "guardrail_architectures": guardrail_architectures,
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
        print("OK: enterprise_ai_safety_guardrails_speculative_parallel_shield_ttft_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
