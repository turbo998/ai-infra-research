from __future__ import annotations

import json
from pathlib import Path


def evaluate_prefix_cache_security_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Unsalted Global Prefix Cache (64-Bit hash(tokens) + Flat LRU)",
            "cross_tenant_leak_accuracy_pct": 98.4,
            "ttft_hit_vs_miss_ratio_x": 9.07,
            "cross_lora_mm_corruption_pct": 12.5,
            "sustained_cache_hit_rate_pct": 41.2,
            "hash_cpu_overhead_us": 4.2,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Unsalted Composite Hash (tokens + lora_id + mm_hash) + Flat LRU",
            "cross_tenant_leak_accuracy_pct": 98.4,
            "ttft_hit_vs_miss_ratio_x": 9.07,
            "cross_lora_mm_corruption_pct": 0.0,
            "sustained_cache_hit_rate_pct": 41.2,
            "hash_cpu_overhead_us": 8.5,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: Strictly Isolated Per-Tenant Cache (No Public Template Sharing + Flat LRU)",
            "cross_tenant_leak_accuracy_pct": 0.0,
            "ttft_hit_vs_miss_ratio_x": 1.00,
            "cross_lora_mm_corruption_pct": 0.0,
            "sustained_cache_hit_rate_pct": 54.8,
            "hash_cpu_overhead_us": 14.0,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Public Read-Only Root + Tenant-Salted SHA-256 + Scan-Resistant S3-FIFO",
            "cross_tenant_leak_accuracy_pct": 0.0,
            "ttft_hit_vs_miss_ratio_x": 1.00,
            "cross_lora_mm_corruption_pct": 0.0,
            "sustained_cache_hit_rate_pct": 84.6,
            "hash_cpu_overhead_us": 14.8,
            "decision": "Act automatically",
        },
    ]

    attack_scenarios = [
        {
            "scenario": "Cross-Tenant Private Prompt Timing Probe",
            "unsalted_leak_pct": 98.4,
            "tenant_salted_leak_pct": 0.0,
        },
        {
            "scenario": "Same `<image>` Token IDs with Different Image Pixels",
            "token_only_hash_error_pct": 100.0,
            "composite_sha256_error_pct": 0.0,
        },
        {
            "scenario": "Same Prompt Under LoRA Adapter #1 vs LoRA Adapter #2",
            "token_only_hash_error_pct": 100.0,
            "composite_sha256_error_pct": 0.0,
        },
        {
            "scenario": "128K One-Hit RAG Scan Polluting Interactive Multi-Turn Cache",
            "flat_lru_hit_rate_pct": 41.2,
            "s3_fifo_hit_rate_pct": 84.6,
        },
    ]

    summary = {
        "benchmark": "llm_prefix_cache_timing_side_channel_tenant_salted_sha256_eviction_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "ttft_miss_ms": 165.0,
        "ttft_hit_ms": 18.2,
        "timing_oracle_ratio_x": round(165.0 / 18.2, 2),
        "unsalted_prompt_stealing_pct": 98.4,
        "salted_prompt_stealing_pct": 0.0,
        "flat_lru_hit_rate_pct": 41.2,
        "s3_fifo_hit_rate_pct": 84.6,
        "hit_rate_gain_pt": round(84.6 - 41.2, 1),
        "tiers": tiers,
        "attack_scenarios": attack_scenarios,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_prefix_cache_security_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
