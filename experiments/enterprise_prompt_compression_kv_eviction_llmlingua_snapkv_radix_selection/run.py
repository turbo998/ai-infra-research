from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    compression_modes = [
        {
            "mode": "Tier 1: Uncompressed 32K Prompt without Prefix Caching or Reranker",
            "effective_prefill_ttft_ms": 1480.0,
            "prefix_cache_hit_rate_pct": 0.0,
            "exact_entity_code_accuracy_pct": 98.8,
            "kv_memory_per_req_gb": 5.24,
        },
        {
            "mode": "Tier 2: Query-Aware LLMLingua Compression Applied Across Shared System Prefix (Breaks Radix Hash!)",
            "effective_prefill_ttft_ms": 620.0,
            "prefix_cache_hit_rate_pct": 4.5,
            "exact_entity_code_accuracy_pct": 86.4,
            "kv_memory_per_req_gb": 1.65,
        },
        {
            "mode": "Tier 3: StreamingLLM Attention Sinks (4 Sink Tokens + 4K Window Eviction, Drops Middle Context)",
            "effective_prefill_ttft_ms": 340.0,
            "prefix_cache_hit_rate_pct": 68.0,
            "exact_entity_code_accuracy_pct": 54.0,
            "kv_memory_per_req_gb": 0.66,
        },
        {
            "mode": "Tier 4: Prefix-Preserving Architecture (Uncompressed Static Prefix in RadixCache + L40S Reranker + LLMLingua-2 on Dynamic Cold RAG Only)",
            "effective_prefill_ttft_ms": 192.0,
            "prefix_cache_hit_rate_pct": 92.5,
            "exact_entity_code_accuracy_pct": 98.4,
            "kv_memory_per_req_gb": 1.12,
        },
    ]
    return {
        "experiment_id": "enterprise_prompt_compression_kv_eviction_llmlingua_snapkv_radix_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "raw_context_tokens": 32768,
        "compression_modes": compression_modes,
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
        print("OK: enterprise_prompt_compression_kv_eviction_llmlingua_snapkv_radix_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
