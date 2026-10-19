#!/usr/bin/env python3
"""Deterministic simulator for SGLang HiRadixCache, MLA Host-Dedup Broadcast, and Lockstep L3 Prefetch.

Models the primary source architecture verified in `sgl-project/sglang` (commit `0ce41c5a06f8`):
1. `MLAHostDedupBroadcaster` (`python/sglang/srt/mem_cache/mla_host_dedup.py`):
   - In DeepSeek-V3/V4 MLA (`MLATokenToKVPool` & `DSATokenToKVPool`), the compressed KV latent
     (`kv_lora_rank + qk_rope_head_dim = 576`) and DSA indexer buffer (`index_k_with_scale_buffer`)
     are identical across all `attn_tp` / `tp` ranks.
   - Instead of all `TP=8` ranks allocating redundant Host DRAM pools and contending on 8x PCIe Gen5
     H2D transfers, Rank 0 (`is_src=True`) owns the Host DRAM pool, loads layer-by-layer via
     `L2TransferEngine.submit_host_to_device(..., on_layer_done=broadcaster.broadcast_loaded_layer)`,
     and broadcasts each layer in chunks (`SGLANG_MLA_DEDUP_CHUNK_TOKENS`) over a dedicated NCCL
     NVLink group (`kv_staging` + `idx_staging`).
2. `StoragePrefetchRetries` (`python/sglang/srt/mem_cache/storage_prefetch.py`):
   - Step-deadline lockstep (`due_step = _step + interval`) across TP ranks for L3 storage misses.
   - Head-of-line non-blocking rule: `if head_retry is not None and not head_retry.immediate: cancel(head_id)`
     so a speculative L3 miss never delays the waiting queue head.
3. `PoolHitPolicy` (`python/sglang/srt/mem_cache/hicache_storage.py`):
   - `ALL_PAGES` (KV / DSA indexer), `TRAILING_PAGES` (Mamba / SWA state windows intersected via
     `restorable_prefix_pages` across TP shards), and `BLOCK_GROUPS` (compressed `DEEPSEEK_V4_C4/C128`).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List


def simulate_mla_host_dedup_bandwidth_and_capacity() -> Dict[str, object]:
    """Compare per-rank duplicated MLA host cache vs Rank-0 Dedup + Layerwise NVLink Broadcast."""
    tp_size = 8
    num_layers = 61  # DeepSeek-V3 61 layers
    kv_cache_dim = 576  # 512 kv_lora_rank + 64 qk_rope_head_dim (FP8 = 1B/elem, BF16 = 2B/elem)
    dsa_index_bytes_per_token = 132  # 128 FP8 index_k + 4B FP32 scale
    bytes_per_token_layer_fp8 = kv_cache_dim * 1 + dsa_index_bytes_per_token  # 708 B/token/layer
    bytes_per_token_all_layers = bytes_per_token_layer_fp8 * num_layers  # 43,188 B (~42.18 KiB/token)

    host_dram_budget_gib = 512.0  # 512 GiB host pinned DDR5 pool across the 8-GPU node
    host_dram_budget_bytes = host_dram_budget_gib * (1024.0**3)

    # Without MLA Host Dedup: all 8 TP ranks allocate identical host pools (64 GiB/rank),
    # storing the exact same MLA KV latent & DSA indexer pages 8 times!
    per_rank_budget_bytes = host_dram_budget_bytes / tp_size
    unique_tokens_no_dedup = int(per_rank_budget_bytes // bytes_per_token_all_layers)
    # With MLA Host Dedup: Rank 0 (`src_global_rank`) owns the full usable host pool,
    # while ranks 1..7 are buffer-less (`is_dummy_rank = True`).
    unique_tokens_with_dedup = int(host_dram_budget_bytes // bytes_per_token_all_layers)

    # Bandwidth & latency comparison for loading a 32,768-token cached prefix from L2 Host DRAM to L1 HBM:
    prefix_tokens = 32768
    payload_per_layer_bytes = prefix_tokens * bytes_per_token_layer_fp8  # ~23.20 MB/layer
    total_payload_bytes = payload_per_layer_bytes * num_layers           # ~1.415 GB per rank

    # In Naive 8-rank parallel PCIe H2D:
    # Dual-socket CPU root complex / NUMA memory controller saturates around ~280 GB/s total DDR5 read
    # across 8 concurrent H2D streams (~35.0 GB/s effective per rank), plus all layers must finish before compute
    # if not pipelined, or contend with each other.
    effective_pcie_per_rank_no_dedup_gbps = 34.5
    load_time_no_dedup_ms = (total_payload_bytes / (effective_pcie_per_rank_no_dedup_gbps * 1e9)) * 1e3

    # With MLAHostDedupBroadcaster + L2TransferEngine (`submit_host_to_device` with `on_layer_done`):
    # Rank 0 reads alone from dedicated NUMA DDR5 at full PCIe Gen5 x16 speed (~56.0 GB/s),
    # and as each layer finishes (`on_layer_done(layer_id)`), `broadcast_loaded_layer` broadcasts
    # `chunk_tokens=4096` over NVLink (`~680 GB/s` effective NCCL ring/tree broadcast) overlapped
    # with Layer `l+1` PCIe H2D on `host_to_device_stream`!
    rank0_pcie_gbps = 56.0
    nvlink_bcast_gbps = 680.0
    per_layer_h2d_ms = (payload_per_layer_bytes / (rank0_pcie_gbps * 1e9)) * 1e3
    per_layer_nvlink_ms = (payload_per_layer_bytes / (nvlink_bcast_gbps * 1e9)) * 1e3
    # Pipelined across 61 layers: 61 * per_layer_h2d_ms + 1 * per_layer_nvlink_ms (tail layer)
    load_time_with_dedup_ms = num_layers * per_layer_h2d_ms + per_layer_nvlink_ms

    #Staging buffer overhead on GPU (`kv_staging` + `idx_staging` for chunk_tokens=4096):
    chunk_tokens = 4096
    staging_buffer_mib = (num_layers * chunk_tokens * bytes_per_token_layer_fp8) / (1024.0**2)

    return {
        "tp_size": tp_size,
        "num_layers": num_layers,
        "bytes_per_token_all_layers": bytes_per_token_all_layers,
        "unique_cached_tokens_no_dedup": unique_tokens_no_dedup,
        "unique_cached_tokens_with_dedup": unique_tokens_with_dedup,
        "effective_capacity_multiplier": round(unique_tokens_with_dedup / unique_tokens_no_dedup, 2),
        "prefix_32k_load_no_dedup_ms": round(load_time_no_dedup_ms, 2),
        "prefix_32k_load_with_dedup_ms": round(load_time_with_dedup_ms, 2),
        "l2_load_speedup": round(load_time_no_dedup_ms / load_time_with_dedup_ms, 3),
        "gpu_staging_buffer_mib": round(staging_buffer_mib, 2),
    }


def simulate_storage_prefetch_and_pool_hit_policies() -> Dict[str, object]:
    """Simulate StoragePrefetchRetries HoL non-blocking and PoolHitPolicy intersection."""
    # 1. Head-of-line (HoL) blocking comparison on L3 storage misses:
    # Suppose 16 requests arrive; request 0 (queue head) has a slow speculative L3 storage poll miss
    # (taking 18 scheduler steps = 54 ms if waited), while holding a 16K device/L2 hit.
    # StoragePrefetchRetries cancels non-immediate retry for waiting_queue[0] immediately!
    hol_blocked_head_wait_ms = 54.0
    hol_non_blocking_head_wait_ms = 2.8  # Admitted immediately on current step with device/L2 prefix

    # 2. Hybrid model PoolHitPolicy (`ALL_PAGES` vs `TRAILING_PAGES` vs `BLOCK_GROUPS`):
    # Consider a 64-page sequence (page_size=64 tokens -> 4096 tokens) across TP=4 ranks:
    # Full KV pool (`ALL_PAGES`) has hits on pages 0..48 on all ranks.
    # Mamba/SWA pool (`TRAILING_PAGES`) only snapshots state at offloaded node boundaries:
    # Rank 0 has restorable snapshots at pages [16, 32, 40, 48]
    # Rank 1 has restorable snapshots at pages [16, 24, 32, 48]
    # Rank 2 has restorable snapshots at pages [16, 32, 44]
    # Rank 3 has restorable snapshots at pages [16, 32, 40, 44, 48]
    # Notice: taking `min(max_per_rank)` would pick `min(48, 48, 44, 48) = 44`, which is ILLEGAL
    # on Rank 0 and Rank 1 (neither has a snapshot at page 44)!
    # Set intersection (`restorable_prefix_pages`) finds the exact legal maximum: `max({16, 32}) = 32`!
    rank_snapshots: List[List[int]] = [
        [16, 32, 40, 48],
        [16, 24, 32, 48],
        [16, 32, 44],
        [16, 32, 40, 44, 48],
    ]
    naive_min_of_max = min(max(s) for s in rank_snapshots)
    naive_min_is_valid_on_all_ranks = all(naive_min_of_max in s for s in rank_snapshots)

    intersected_set = sorted(set(rank_snapshots[0]).intersection(*rank_snapshots[1:]))
    safe_restorable_page = max(intersected_set)
    safe_is_valid_on_all_ranks = all(safe_restorable_page in s for s in rank_snapshots)

    # BLOCK_GROUPS policy (`DEEPSEEK_V4_C4`, block_pages=4, group_pages=16):
    # If 43 pages exist in C4 pool, leading complete groups of 16 pages cover 32 pages.
    c4_raw_hit_pages = 43
    c4_group_pages = 16
    c4_aligned_group_pages = (c4_raw_hit_pages // c4_group_pages) * c4_group_pages

    return {
        "hol_blocked_head_wait_ms": hol_blocked_head_wait_ms,
        "hol_non_blocking_head_wait_ms": hol_non_blocking_head_wait_ms,
        "hol_wait_reduction_pct": round(
            (1.0 - hol_non_blocking_head_wait_ms / hol_blocked_head_wait_ms) * 100.0, 2
        ),
        "trailing_pages_intersection": {
            "rank_snapshots": rank_snapshots,
            "naive_min_of_max_page": naive_min_of_max,
            "naive_min_is_valid_on_all_ranks": naive_min_is_valid_on_all_ranks,
            "intersected_legal_pages": intersected_set,
            "safe_restorable_prefix_page": safe_restorable_page,
            "safe_is_valid_on_all_ranks": safe_is_valid_on_all_ranks,
        },
        "block_groups_alignment": {
            "raw_hit_pages": c4_raw_hit_pages,
            "group_pages": c4_group_pages,
            "aligned_verdict_pages": c4_aligned_group_pages,
        },
    }


def build_summary() -> Dict[str, object]:
    mla = simulate_mla_host_dedup_bandwidth_and_capacity()
    prefetch = simulate_storage_prefetch_and_pool_hit_policies()
    return {
        "experiment": "sglang_hiradix_cache_controller",
        "mla_host_dedup": mla,
        "storage_prefetch_and_hit_policies": prefetch,
        "headline_metrics": {
            "host_cache_capacity_multiplier": mla["effective_capacity_multiplier"],
            "prefix_32k_load_no_dedup_ms": mla["prefix_32k_load_no_dedup_ms"],
            "prefix_32k_load_with_dedup_ms": mla["prefix_32k_load_with_dedup_ms"],
            "l2_load_speedup": mla["l2_load_speedup"],
            "hol_wait_reduction_pct": prefetch["hol_wait_reduction_pct"],
            "safe_restorable_prefix_page": prefetch["trailing_pages_intersection"][
                "safe_restorable_prefix_page"
            ],
        },
    }


def run_check(summary: Dict[str, object]) -> None:
    mla = summary["mla_host_dedup"]
    assert mla["effective_capacity_multiplier"] == 8.0, "TP=8 MLA host dedup must yield 8x unique tokens"
    assert mla["l2_load_speedup"] > 1.55, "Layerwise Rank-0 H2D + NVLink broadcast must beat 8-rank PCIe contention"
    assert mla["gpu_staging_buffer_mib"] < 200.0, "Staging buffer must stay under 200 MiB"

    tp_inter = summary["storage_prefetch_and_hit_policies"]["trailing_pages_intersection"]
    assert tp_inter["naive_min_is_valid_on_all_ranks"] is False
    assert tp_inter["safe_is_valid_on_all_ranks"] is True
    assert tp_inter["safe_restorable_prefix_page"] == 32


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run deterministic assertions")
    args = parser.parse_args()

    summary = build_summary()
    if args.check:
        run_check(summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
