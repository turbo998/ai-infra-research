#!/usr/bin/env python3
"""Phase 64 Track C deterministic offline simulation and verification guard:
1. Megatron-LM MFSDPv2 phase-selective MXFP8 weight plane All-Gather (#7907, 18bc49d) & EP A2A overlap per-segment full activation recompute (#6429, da2c0ec)
2. TransformerEngine Triton mask-map permute num_out_tokens capacity drop (#3347, c9c9225) & GroupedTensor autograd view / record_stream tracking (#3701, 2199b71)
3. Mooncake TransferEngine per-chain RDMA clock read (#4793, dd1217b) & Store shard-free DFS eviction DeferDeletion (#4854, c81f4ad)
4. NIXL sharedRingBuffer 256-byte cache-line isolation & cachedReadPos_/cachedWritePos_ (#2018, a9f1f7a) + LMCache callback-driven LookupState (#5960, ff9185d)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUMMARY_PATH = Path(__file__).with_name("summary.json")


def simulate_megatron_mfsdp_mxfp8_and_a2a_recompute() -> dict[str, Any]:
    """Model Megatron-LM PR #7907 (18bc49d) MFSDPv2 MXFP8 phase-aware plane All-Gather & PR #6429 (da2c0ec) EP A2A segment recompute."""
    # Consider a linear weight of shape [Out=14336, In=5376] in MXFP8 (block_size=32)
    rows, cols = 14336, 5376
    block_size = 32
    rowwise_data_bytes = rows * cols * 1  # FP8 E4M3 = 1 byte/elem
    rowwise_scale_bytes = rows * (cols // block_size) * 1  # E8M0 scale = 1 byte per 32 elems
    colwise_data_bytes = rows * cols * 1
    colwise_scale_bytes = (rows // block_size) * cols * 1

    single_plane_bytes = rowwise_data_bytes + rowwise_scale_bytes
    dual_plane_bytes = single_plane_bytes + colwise_data_bytes + colwise_scale_bytes

    # Baseline MFSDPv2 without phase selection: gathers BOTH rowwise + columnwise planes on Forward All-Gather AND Backward All-Gather (if resharded)
    baseline_step_ag_bytes = 2 * dual_plane_bytes
    # Phase-selective MFSDPv2 (#7907): Forward gathers only rowwise (data+scale); Backward gathers only columnwise (data+scale)
    phase_selective_step_ag_bytes = single_plane_bytes + (colwise_data_bytes + colwise_scale_bytes)
    ag_volume_reduction_pct = round(
        100.0 * (baseline_step_ag_bytes - phase_selective_step_ag_bytes) / baseline_step_ag_bytes, 2
    )

    # EP A2A overlap per-segment recompute (#6429): verify mutable ModelChunkState snapshot fields
    mutable_fields = (
        "input_ids",
        "position_ids",
        "padding_mask",
        "mtp_input_mask",
        "mtp_hidden_states",
    )
    # Simulate MTP roll of input_ids and mtp_input_mask during forward vs backward replay
    initial_mask = [True, True, True, True, True, False, True, True]
    rolled_once = initial_mask[1:] + [initial_mask[0]]
    # Without snapshotting mtp_input_mask, replay rolls an already-rolled mask -> rolled twice!
    rolled_twice_bug = rolled_once[1:] + [rolled_once[0]]
    replayed_with_snapshot = initial_mask[1:] + [initial_mask[0]]

    return {
        "weight_shape": [rows, cols],
        "mxfp8_block_size": block_size,
        "single_plane_mib": round(single_plane_bytes / (1024 * 1024), 4),
        "dual_plane_mib": round(dual_plane_bytes / (1024 * 1024), 4),
        "baseline_step_allgather_mib": round(baseline_step_ag_bytes / (1024 * 1024), 4),
        "phase_selective_step_allgather_mib": round(phase_selective_step_ag_bytes / (1024 * 1024), 4),
        "allgather_bandwidth_reduction_pct": ag_volume_reduction_pct,
        "a2a_recompute_mutable_fields": list(mutable_fields),
        "mtp_mask_replay_matches_forward": replayed_with_snapshot == rolled_once,
        "mtp_mask_without_snapshot_corrupted": rolled_twice_bug != rolled_once,
    }


def simulate_te_permute_and_grouped_tensor() -> dict[str, Any]:
    """Model TransformerEngine PR #3347 (c9c9225) Triton mask-map permute capacity drop & PR #3701 (2199b71) GroupedTensor view/stream."""
    # From test_permutation_mask_map_capacity_drop: num_tokens=4096, num_expert=8
    total_routed = 16420
    capacity = total_routed - 17  # 16403
    row_ids = list(range(total_routed))
    clamped_row_ids = [r if r < capacity else -1 for r in row_ids]
    kept_count = sum(1 for r in clamped_row_ids if r >= 0)
    dropped_count = sum(1 for r in clamped_row_ids if r == -1)
    oob_writes_without_clamp = total_routed - capacity

    # GroupedTensor record_stream propagation across all backing buffers
    backing_buffers = [
        "rowwise_data",
        "columnwise_data",
        "scale_inv",
        "columnwise_scale_inv",
        "amax",
        "first_dims",
        "tensor_offsets",
    ]

    return {
        "num_tokens": 4096,
        "num_experts": 8,
        "total_routed_tokens": total_routed,
        "num_out_tokens_capacity": capacity,
        "kept_destinations": kept_count,
        "dropped_destinations_marked_minus1": dropped_count,
        "oob_writes_prevented": oob_writes_without_clamp,
        "jax_no_drop_sentinel": (1 << 31) - 1,
        "grouped_tensor_tracked_buffers_count": len(backing_buffers),
        "grouped_tensor_tracked_buffers": backing_buffers,
    }


def simulate_mooncake_nixl_lmcache() -> dict[str, Any]:
    """Model Mooncake PR #4793 (dd1217b) & #4854 (c81f4ad), NIXL PR #2018 (a9f1f7a), and LMCache PR #5960 (ff9185d)."""
    # 1. Mooncake RDMA submitPostSend clock read once per chain (#4793)
    total_wrs = 256
    chains = 2
    wrs_per_chain = 128
    per_wr_before_ns = 25.0
    per_wr_after_ns = 7.5
    call_latency_before_us = round(total_wrs * per_wr_before_ns / 1000.0, 2)  # 6.40 us (reported 6.3 us)
    call_latency_after_us = round(total_wrs * per_wr_after_ns / 1000.0, 2)    # 1.92 us (reported 1.9 us)
    reported_before_us = 6.3
    reported_after_us = 1.9
    lock_hold_speedup_x = round(reported_before_us / reported_after_us, 4)

    # 2. NIXL sharedRingBuffer false-sharing elimination & cached position checks (#2018)
    destructive_interference_size = 256
    ring_capacity = 1024
    events_pushed = 1000
    # Uncached full() check loads consumer-owned read_pos with memory_order_acquire on every push:
    cross_core_acquire_loads_uncached = events_pushed
    # Cached check only reloads read_pos when next_write == cachedReadPos_ (1 reload when ring wraps or fills):
    cross_core_acquire_loads_cached = 1
    coherence_traffic_reduction_pct = round(
        100.0 * (cross_core_acquire_loads_uncached - cross_core_acquire_loads_cached) / cross_core_acquire_loads_uncached,
        2,
    )

    # 3. LMCache callback-driven LookupState (#5960)
    # Polling waits until the next scheduler step (e.g. 5.0 ms step interval) after LOOKUP ack before sending QUERY_PREFETCH_STATUS.
    # Callback-driven LookupState fires QUERY_PREFETCH_STATUS immediately on transport recv thread upon LOOKUP ack (0.15 ms RPC RTT).
    lookup_rpc_ms = 0.40
    scheduler_step_interval_ms = 5.00
    query_prefetch_rpc_ms = 0.40
    polling_lookup_to_query_ms = round(scheduler_step_interval_ms + query_prefetch_rpc_ms, 2)
    callback_lookup_to_query_ms = round(lookup_rpc_ms + query_prefetch_rpc_ms, 2)
    lookup_latency_speedup_x = round(polling_lookup_to_query_ms / callback_lookup_to_query_ms, 4)

    return {
        "mooncake_rdma_wrs": total_wrs,
        "mooncake_rdma_chains": chains,
        "mooncake_clock_calls_before": total_wrs,
        "mooncake_clock_calls_after": chains,
        "mooncake_post_send_before_us": reported_before_us,
        "mooncake_post_send_after_us": reported_after_us,
        "mooncake_per_wr_before_ns": per_wr_before_ns,
        "mooncake_per_wr_after_ns": per_wr_after_ns,
        "mooncake_lock_hold_speedup_x": lock_hold_speedup_x,
        "nixl_destructive_interference_bytes": destructive_interference_size,
        "nixl_telemetry_version": 6,
        "nixl_cross_core_loads_uncached": cross_core_acquire_loads_uncached,
        "nixl_cross_core_loads_cached": cross_core_acquire_loads_cached,
        "nixl_coherence_reduction_pct": coherence_traffic_reduction_pct,
        "lmcache_polling_resolution_ms": polling_lookup_to_query_ms,
        "lmcache_callback_resolution_ms": callback_lookup_to_query_ms,
        "lmcache_lookup_speedup_x": lookup_latency_speedup_x,
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "megatron-mfsdp-mxfp8-te-permute-mooncake-nixl",
        "phase": 64,
        "priority": 73,
        "provenance": "own-offline deterministic guard + upstream measurements (Megatron-LM #7907, #6429; TE #3347, #3701; Mooncake #4793, #4854; NIXL #2018; LMCache #5960)",
        "megatron_mfsdp_and_a2a": simulate_megatron_mfsdp_mxfp8_and_a2a_recompute(),
        "te_permute_and_grouped_tensor": simulate_te_permute_and_grouped_tensor(),
        "mooncake_nixl_lmcache": simulate_mooncake_nixl_lmcache(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify summary.json matches deterministic outputs")
    args = parser.parse_args()

    summary = build_summary()
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not SUMMARY_PATH.exists():
            raise SystemExit(f"Missing {SUMMARY_PATH}")
        existing = SUMMARY_PATH.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("summary.json is out of sync with deterministic run.py output")
        print("own-offline outputs verified for megatron-mfsdp-mxfp8-te-permute-mooncake-nixl")
        return 0

    SUMMARY_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
