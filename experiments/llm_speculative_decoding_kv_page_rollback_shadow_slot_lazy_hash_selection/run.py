from __future__ import annotations
import json
from pathlib import Path


def evaluate_spec_decode_kv_rollback_lazy_hash() -> dict:
    # Workload: EAGLE-3 / DeepSeek-V3 MTP Speculative Decoding (K=5 draft tokens/step, average accepted = 3.4 tokens)
    tiers = [
        {
            "tier_id": "tier1_sync_cpu_per_token_block_alloc_and_eager_prefix_rehash",
            "name": "Synchronous CPU Per-Draft-Token Block Allocation/Free + Eager Prefix Block Hashing",
            "cpu_kv_manager_overhead_ms_per_step": 6.80,
            "net_speculative_speedup_x": 1.38,
            "prefix_cache_stale_slot_corruption": False,
            "action": "Send to a human",
            "status": "fail",
            "note": "Allocating, freeing, and hashing KV pages on the CPU for every accepted/rejected draft token adds 6.80ms/step, eating 58% of speculative gains!"
        },
        {
            "tier_id": "tier2_device_seqlen_rollback_without_lazy_full_block_hash_guard",
            "name": "In-Place `kv_seqlens` Rollback Without Full-Block Commit Boundary Guard (Prefix Hash Poisoning)",
            "cpu_kv_manager_overhead_ms_per_step": 1.15,
            "net_speculative_speedup_x": 2.18,
            "prefix_cache_stale_slot_corruption": True,
            "action": "Send to a human",
            "status": "fail",
            "note": "Commits partial or unverified speculative KV blocks into the global Radix/Prefix Cache before target verification -> corrupts future cache hits!"
        },
        {
            "tier_id": "tier3_preallocated_k_plus_1_shadow_slots_with_host_sync_rollback",
            "name": "Pre-Allocated `K+1` Shadow Slots + Host-Synchronized `accepted_len` Page Truncation",
            "cpu_kv_manager_overhead_ms_per_step": 1.95,
            "net_speculative_speedup_x": 2.04,
            "prefix_cache_stale_slot_corruption": False,
            "action": "Confirm",
            "status": "pass",
            "note": "Reserves K+1 slots up front and only hashes verified full 16-token blocks, but still syncs accepted_len to CPU on every verification step."
        },
        {
            "tier_id": "tier4_preallocated_shadow_slots_device_atomic_seqlen_rewind_plus_lazy_block_hash",
            "name": "Pre-Allocated `+K` Shadow Slots + Device-Side Atomic `kv_seqlens` Rewind + Lazy Full-Block Hash Commit",
            "cpu_kv_manager_overhead_ms_per_step": 0.28,
            "net_speculative_speedup_x": 2.45,
            "prefix_cache_stale_slot_corruption": False,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Pre-reserves +K slots, rewinds rejected KV offsets purely on GPU in CUDAGraph (`seqlen += num_accepted`), and hashes only verified 16-token blocks -> 2.45x speedup!"
        },
    ]
    return {
        "experiment_id": "llm_speculative_decoding_kv_page_rollback_shadow_slot_lazy_hash_selection",
        "round": 319,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "speculative_method": "EAGLE-3 / DeepSeek-V3 MTP (K=5 Draft Tokens)",
            "mean_accepted_tokens_per_step": 3.4,
            "paged_kv_block_size": 16
        },
        "tiers": tiers,
        "summary": {
            "kv_manager_overhead_reduction_x": round(6.80 / 0.28, 2),
            "net_speedup_improvement_x": 2.45,
            "baseline_naive_speedup_x": 1.38,
            "recommended_tier": "tier4_preallocated_shadow_slots_device_atomic_seqlen_rewind_plus_lazy_block_hash"
        }
    }


if __name__ == "__main__":
    data = evaluate_spec_decode_kv_rollback_lazy_hash()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
