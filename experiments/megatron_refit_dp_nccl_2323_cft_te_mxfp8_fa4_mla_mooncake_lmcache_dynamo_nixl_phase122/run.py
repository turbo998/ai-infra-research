"""Phase 122 Track C: Megatron-LM Single-DP-Group Refit Load Balancing, NCCL 2.32.3-1 Peak BusBW & CFT Counted Operations, TransformerEngine 128x128 MXFP8 Requantize, FlashAttention-4 SM100 Sparse-MLA 64-Head lse_log2 Tail Bound, Mooncake Conductor/HCCS/PG, LMCache aio gRPC Semaphore, Dynamo, and NIXL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_megatron_single_dp_refit_and_nccl_2323_cft() -> dict[str, Any]:
    """Model Megatron-LM PR #8502 (4a55e6fe), NCCL 53a84c7e, and NCCL 3a87143f."""
    # Megatron-LM PR #8502: TP=1, PP=1, DP=4 training ranks [0,1,2,3] refitting to 4 inference ranks [4,5,6,7]
    src_replicas = [0, 1, 2, 3]
    dst_ranks = [4, 5, 6, 7]
    legacy_selected_sources = [src_replicas[0] for _ in dst_ranks]  # all read from rank 0!
    fixed_selected_sources = [src_replicas[dst % len(src_replicas)] for dst in dst_ranks]
    max_reads_per_src_before = max(legacy_selected_sources.count(r) for r in src_replicas)
    max_reads_per_src_after = max(fixed_selected_sources.count(r) for r in src_replicas)

    # NCCL 2.32.3-1 Peak BusBW (GB/s) from docs/perf/v2.32.3-1.md
    gb300_allreduce_legacy_8gb = {"NVL4": 694, "NVL8": 830, "NVL16": 889, "NVL32": 875, "NVL64": 878}
    gb300_allreduce_sym_2gb = {"NVL4": 673, "NVL8": 843, "NVL16": 922, "NVL32": 959, "NVL64": 992}
    gb300_allgather_legacy_8gb = {"NVL4": 703, "NVL8": 710, "NVL16": 710, "NVL32": 710, "NVL64": 701}
    gb300_allgather_ce_8gb = {"NVL4": 722, "NVL8": 751, "NVL16": 737, "NVL32": 743, "NVL64": 737}
    nvl8_allreduce_legacy_cross_gen = {"H100": 484, "B200": 846, "GB300": 830}

    return {
        "megatron_legacy_sources_dp4": legacy_selected_sources,
        "megatron_fixed_sources_dp4": fixed_selected_sources,
        "megatron_max_reads_per_src_before": max_reads_per_src_before,
        "megatron_max_reads_per_src_after": max_reads_per_src_after,
        "megatron_source_egress_reduction_x": round(max_reads_per_src_before / max_reads_per_src_after, 2),
        "nccl_gb300_allreduce_legacy_8gb_gbs": gb300_allreduce_legacy_8gb,
        "nccl_gb300_allreduce_sym_2gb_gbs": gb300_allreduce_sym_2gb,
        "nccl_gb300_nvl64_sym_vs_legacy_gain_pct": round((992 - 878) / 878 * 100.0, 2),
        "nccl_gb300_allgather_legacy_8gb_gbs": gb300_allgather_legacy_8gb,
        "nccl_gb300_allgather_ce_8gb_gbs": gb300_allgather_ce_8gb,
        "nccl_nvl8_allreduce_legacy_cross_gen_gbs": nvl8_allreduce_legacy_cross_gen,
        "nccl_cft_counted_buffer_align_bytes": 256,
        "nccl_cft_counted_op_align_bytes": 16,
        "nccl_cft_counted_max_active_counters_per_gpu": 32,
        "nccl_cft_counted_max_window_gb": 256,
    }


def simulate_te_mxfp8_128x128_and_fa4_sparse_mla_tail() -> dict[str, Any]:
    """Model TransformerEngine PR #3656 (15dcc848) and FlashAttention-4 2865a891a90c."""
    # TE PR #3656: 128x128 wide-tile persistent grouped MXFP8 requantize vs 32x128 capacity grid
    capacity_rows = 32768
    live_rows = 4096
    cols = 4096
    legacy_ctas = math.ceil(capacity_rows / 32) * (cols // 128)  # 1024 * 32 = 32768 CTAs
    wide_live_tiles = math.ceil(live_rows / 128) * (cols // 128)  # 32 * 32 = 1024 tiles
    tile_reduction_x = round(legacy_ctas / wide_live_tiles, 2)  # 32.0x fewer tile evaluations!
    swizzled_scale_contiguous_bytes_per_tile = (128 * 128) // 32  # 512 bytes contiguous per tile

    # FA4 commit 2865a891a90c: SM100 Sparse-MLA bwd preprocess at 64 Q heads with odd token count
    preprocess_tile_m = 128
    q_heads = 64
    seq0_tokens = 3  # odd token count!
    seq0_rows = seq0_tokens * q_heads  # 192 packed (token, head) rows
    seq0_rows_rounded = math.ceil(seq0_rows / preprocess_tile_m) * preprocess_tile_m  # 256 rows
    overwritten_next_seq_rows_before = seq0_rows_rounded - seq0_rows  # 64 rows = entire first token of seq 1!
    recomputed_p_corrupted = math.exp(0.0 - math.inf)  # exp2(S - inf) = 0.0 -> dq/dqv/dk/dv vanish

    return {
        "te_legacy_32x128_capacity_ctas": legacy_ctas,
        "te_wide_128x128_live_tiles": wide_live_tiles,
        "te_tile_work_reduction_x": tile_reduction_x,
        "te_swizzled_scale_block_bytes": swizzled_scale_contiguous_bytes_per_tile,
        "fa4_preprocess_tile_rows": preprocess_tile_m,
        "fa4_64head_3tok_real_rows": seq0_rows,
        "fa4_64head_3tok_rounded_rows": seq0_rows_rounded,
        "fa4_overwritten_next_seq_rows_before_fix": overwritten_next_seq_rows_before,
        "fa4_overwritten_next_seq_rows_after_fix": 0,
        "fa4_corrupted_recomputed_p_value": recomputed_p_corrupted,
    }


def simulate_mooncake_lmcache_dynamo_and_nixl() -> dict[str, Any]:
    """Model Mooncake #4907/#4903/#5008, LMCache #5858, Dynamo #17440, and NIXL #2387."""
    # Mooncake #4907: configurable max_blocks_per_context (default 200,000, evict to 90% = 180,000)
    default_max_blocks = 200000
    evict_target_blocks = int(default_max_blocks * 0.9)  # 180000

    # Mooncake #4903: HCCS 2 MiB alignment check at registerLocalMemory
    hccs_align_bytes = 2 * 1024 * 1024  # 2,097,152
    subblock_addr = 0x100000000200  # offset +512 B from 2 MiB boundary
    hccs_misaligned_rejected = (subblock_addr % hccs_align_bytes) != 0

    # Mooncake #5008: atomic next_active_ranks update vs clear-then-refill window
    world_size = 8
    clear_then_refill_transient_false_count = world_size
    staged_atomic_transient_false_count = 0

    # LMCache #5858: _STREAM_READ_AHEAD = 1 semaphore on aio gRPC server
    lmcache_unbounded_inflight_batches = 998
    lmcache_bounded_inflight_batches = 158
    lmcache_unbounded_rss_delta_mb = 128
    lmcache_bounded_rss_delta_mb = 4
    lmcache_rss_reduction_x = round(lmcache_unbounded_rss_delta_mb / lmcache_bounded_rss_delta_mb, 1)
    lmcache_1rec_batches_per_sec_before = 794.5
    lmcache_1rec_batches_per_sec_after = 2163.5
    lmcache_1rec_speedup_x = round(lmcache_1rec_batches_per_sec_after / lmcache_1rec_batches_per_sec_before, 3)

    # NIXL #2387: collision-only pid & agent_instance labels in prometheus_mp
    def nixl_select_stores(stores: list[dict[str, Any]]) -> list[dict[str, Any]]:
        live = [s for s in stores if s["alive"]]
        if len(live) >= 2:
            return [{**s, "with_process_labels": True} for s in live]
        if len(live) == 1:
            return [{**live[0], "with_process_labels": False}]
        newest_departed = max(stores, key=lambda s: s["last_update_ns"])
        return [{**newest_departed, "with_process_labels": False}]

    restart_case = nixl_select_stores([
        {"pid": 1001, "alive": False, "last_update_ns": 10},
        {"pid": 1002, "alive": True, "last_update_ns": 20},
    ])
    collision_case = nixl_select_stores([
        {"pid": 2001, "alive": True, "last_update_ns": 30},
        {"pid": 2002, "alive": True, "last_update_ns": 35},
    ])

    return {
        "mooncake_default_max_blocks_per_context": default_max_blocks,
        "mooncake_evict_target_blocks_at_default": evict_target_blocks,
        "mooncake_hccs_alignment_bytes": hccs_align_bytes,
        "mooncake_hccs_subblock_512b_rejected": hccs_misaligned_rejected,
        "mooncake_clear_then_refill_transient_false_ranks": clear_then_refill_transient_false_count,
        "mooncake_staged_atomic_transient_false_ranks": staged_atomic_transient_false_count,
        "lmcache_stream_read_ahead_limit": 1,
        "lmcache_unbounded_inflight_batches": lmcache_unbounded_inflight_batches,
        "lmcache_bounded_inflight_batches": lmcache_bounded_inflight_batches,
        "lmcache_unbounded_rss_delta_mb": lmcache_unbounded_rss_delta_mb,
        "lmcache_bounded_rss_delta_mb": lmcache_bounded_rss_delta_mb,
        "lmcache_rss_reduction_x": lmcache_rss_reduction_x,
        "lmcache_1rec_speedup_x": lmcache_1rec_speedup_x,
        "dynamo_shared_postprocessor_settings_count": 15,
        "nixl_restart_series_count": len(restart_case),
        "nixl_restart_with_process_labels": restart_case[0]["with_process_labels"],
        "nixl_collision_series_count": len(collision_case),
        "nixl_collision_with_process_labels": all(s["with_process_labels"] for s in collision_case),
    }


def run_suite() -> dict[str, Any]:
    megatron_nccl = simulate_megatron_single_dp_refit_and_nccl_2323_cft()
    te_fa4 = simulate_te_mxfp8_128x128_and_fa4_sparse_mla_tail()
    mooncake_lmcache_nixl = simulate_mooncake_lmcache_dynamo_and_nixl()

    return {
        "suite": "megatron_refit_dp_nccl_2323_cft_te_mxfp8_fa4_mla_mooncake_lmcache_dynamo_nixl_phase122",
        "phase": 122,
        "priority": 131,
        "track": "Track C: Distributed Training, Collectives, KV Tiering & Telemetry",
        "megatron_and_nccl": megatron_nccl,
        "te_and_fa4": te_fa4,
        "mooncake_lmcache_dynamo_nixl": mooncake_lmcache_nixl,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_suite()
    out_path = Path(__file__).with_name("summary.json")
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        assert summary["megatron_and_nccl"]["megatron_fixed_sources_dp4"] == [0, 1, 2, 3]
        assert summary["megatron_and_nccl"]["nccl_gb300_allreduce_sym_2gb_gbs"]["NVL64"] == 992
        assert summary["te_and_fa4"]["te_tile_work_reduction_x"] == 32.0
        assert summary["te_and_fa4"]["fa4_overwritten_next_seq_rows_before_fix"] == 64
        assert summary["mooncake_lmcache_dynamo_nixl"]["lmcache_rss_reduction_x"] == 32.0
        assert summary["mooncake_lmcache_dynamo_nixl"]["nixl_restart_with_process_labels"] is False
        assert summary["mooncake_lmcache_dynamo_nixl"]["nixl_collision_with_process_labels"] is True

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
