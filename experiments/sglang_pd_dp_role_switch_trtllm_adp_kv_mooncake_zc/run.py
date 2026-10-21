#!/usr/bin/env python3
"""Deterministic reproduction of SGLang PD+DP Two-Phase Role Switch,
TRT-LLM Attention-DP KV Event Collective Alignment, and Mooncake Store Zero-Copy VMM Splitting.

Upstream references:
  - sgl-project/sglang PR #44215 (af49ada16d9b4e7535fd6ff542247f50b8c92c9f, 2026-10-21)
  - NVIDIA/TensorRT-LLM PR #20314 (ec782566f50ad9548e99818fa1d74c7c66cc572d, 2026-10-21)
  - NVIDIA/TensorRT-LLM PR #20108 (69e06b8afbffc4726a63b43fb4acb9fda340f3a7, 2026-10-21)
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


# --- 1. SGLang PR #44215: Two-phase PD role switch under DP & DP attention ---

def validate_sglang_pd_role_switch_config(
    disaggregation_mode: str,
    dp_size: int,
    enable_dp_attention: bool,
    enable_dp_attention_local_control_broadcast: bool,
) -> tuple[bool, str]:
    """Reproduce pd_disaggregation_hook.py validation logic from SGLang PR #44215."""
    if disaggregation_mode not in ("prefill", "decode"):
        return False, "role switch requires --disaggregation-mode prefill or decode"
    if enable_dp_attention and enable_dp_attention_local_control_broadcast:
        return (
            False,
            "PD role switching with --enable-dp-attention does not support "
            "--enable-dp-attention-local-control-broadcast",
        )
    return True, "ok"


def simulate_sglang_pd_role_switch(
    dp_rank_headroom_ok: list[bool],
    two_phase: bool,
) -> dict[str, Any]:
    """Compare single-phase teardown vs PR #44215 two-phase prepare_only preflight."""
    num_dp_ranks = len(dp_rank_headroom_ok)
    if two_phase and num_dp_ranks > 1:
        preflight_results = [
            {"dp_rank": r, "success": ok, "message": "ready" if ok else "Insufficient GPU memory"}
            for r, ok in enumerate(dp_rank_headroom_ok)
        ]
        if not all(item["success"] for item in preflight_results):
            # Abort before any rank tears down resources
            return {
                "overall_success": False,
                "ranks_torn_down": 0,
                "ranks_switched": 0,
                "split_brain": False,
                "safe_to_restore": True,
            }
        return {
            "overall_success": True,
            "ranks_torn_down": num_dp_ranks,
            "ranks_switched": num_dp_ranks,
            "split_brain": False,
            "safe_to_restore": True,
        }

    # Legacy single-phase broadcast: ranks with headroom tear down and switch while others fail
    torn_down = sum(1 for ok in dp_rank_headroom_ok if ok)
    overall_ok = torn_down == num_dp_ranks
    return {
        "overall_success": overall_ok,
        "ranks_torn_down": torn_down,
        "ranks_switched": torn_down,
        "split_brain": (0 < torn_down < num_dp_ranks),
        "safe_to_restore": overall_ok,
    }


# --- 2. TRT-LLM PR #20314: Attention-DP Overlap Loop KV Event Collective Order ---

def simulate_trtllm_adp_overlap_step(
    rank_states: tuple[str, ...],
    enable_kv_cache_events: bool,
    rebalance: bool,
    fixed: bool,
) -> dict[str, Any]:
    """Reproduce test_disagg_loop_transcript.py across ('absent', 'processed', 'deferred') ADP ranks."""
    transcripts: list[list[str]] = []
    for state in rank_states:
        ops: list[str] = []
        if rebalance:
            if not fixed and state == "absent":
                # Old _consume_previous_batch_for_rebalance returned early when previous_batch is None
                transcripts.append(ops)
                continue
            ops.append("flush_transfer_responses")
            if state != "absent":
                ops.append("enqueue_batch_responses")
            else:
                ops.append("enqueue_empty_responses")
            if enable_kv_cache_events:
                ops.append("kv_event_allgather")
            transcripts.append(ops)
            continue

        # Overlap loop step
        ops.append("flush_transfer_responses")
        if state == "processed":
            ops.append("enqueue_batch_responses")
            if enable_kv_cache_events and not fixed:
                ops.append("kv_event_allgather")
        else:
            # 'absent' or 'deferred'
            ops.append("enqueue_empty_responses")
        if enable_kv_cache_events and fixed:
            ops.append("kv_event_allgather")
        transcripts.append(ops)

    # Collective order is only deadlock-free if every rank executes identical collective sequence
    collectives_per_rank = [
        [op for op in ops if op in ("flush_transfer_responses", "enqueue_batch_responses", "enqueue_empty_responses", "kv_event_allgather")]
        for ops in transcripts
    ]
    # Normalize enqueue_batch_responses and enqueue_empty_responses to "responses_collective"
    norm = [
        ["responses_collective" if op.startswith("enqueue_") else op for op in seq]
        for seq in collectives_per_rank
    ]
    deadlock_free = all(seq == norm[0] for seq in norm)
    return {
        "rank_states": list(rank_states),
        "rebalance": rebalance,
        "fixed": fixed,
        "deadlock_free": deadlock_free,
        "normalized_collectives_per_rank": norm,
    }


# --- 3. TRT-LLM PR #20108: Mooncake Store Zero-Copy VMM Allocation Boundary Splitting ---

AllocationLookup = Callable[[int], tuple[int, int] | None]


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def allocation_spans(
    start: int, end: int, lookup: AllocationLookup
) -> list[tuple[int, int]] | None:
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        allocation = lookup(cursor)
        if allocation is None:
            return None
        base, size = allocation
        stop = min(base + size, end)
        if not (base <= cursor < stop):
            return None
        spans.append((cursor, stop))
        cursor = stop
    return spans


def split_at_cuts(start: int, size: int, cuts: Sequence[int]) -> list[tuple[int, int]]:
    end = start + size
    pieces: list[tuple[int, int]] = []
    address = start
    index = bisect_right(cuts, start)
    while index < len(cuts) and cuts[index] < end:
        pieces.append((address, cuts[index] - address))
        address = cuts[index]
        index += 1
    pieces.append((address, end - address))
    return pieces


def evaluate_mooncake_zerocopy() -> dict[str, Any]:
    # Simulate a V2 KV cache pool mapped across three 32 MiB cuMemCreate chunks starting at 0x1000_0000
    chunk_bytes = 32 * 1024 * 1024  # 33,554,432 B
    pool_base = 0x10000000
    allocations = [
        (pool_base + i * chunk_bytes, chunk_bytes) for i in range(3)
    ]

    def lookup(addr: int) -> tuple[int, int] | None:
        for base, size in allocations:
            if base <= addr < base + size:
                return (base, size)
        return None

    # Suppose two layer groups (e.g. full attention + sliding window or MiniMax index-K + K/V)
    # span overlapping regions inside [pool_base, pool_base + 96 MiB)
    raw_spans = [
        (pool_base, pool_base + 48 * 1024 * 1024),
        (pool_base + 16 * 1024 * 1024, pool_base + 96 * 1024 * 1024),
    ]
    merged = merge_intervals(raw_spans)
    reg_ranges: list[tuple[int, int]] = []
    cuts: list[int] = []
    for s, e in merged:
        pieces = allocation_spans(s, e, lookup)
        assert pieces is not None
        reg_ranges.extend(pieces)
        cuts.extend(piece_start for piece_start, _ in pieces[1:])

    # A 4 MiB page straddling the first cuMemCreate boundary (pool_base + 30 MiB .. pool_base + 34 MiB)
    straddling_page_start = pool_base + 30 * 1024 * 1024
    straddling_page_size = 4 * 1024 * 1024
    split_buffers = split_at_cuts(straddling_page_start, straddling_page_size, cuts)

    # Host staging memory saved on an 8-GPU node with bidirectional (send + recv) staging buffers (64 MiB each)
    ranks_on_node = 8
    directions = 2
    max_staging_buffer_bytes = 64 * 1024 * 1024
    pinned_host_bytes_saved = ranks_on_node * directions * max_staging_buffer_bytes

    return {
        "default_stage_through_host": False,
        "merged_span_count": len(merged),
        "registration_range_count": len(reg_ranges),
        "interior_cut_addresses": cuts,
        "straddling_page_split_sizes_mib": [sz // (1024 * 1024) for _, sz in split_buffers],
        "pinned_host_staging_saved_mib_8gpu": pinned_host_bytes_saved // (1024 * 1024),
    }


def run_experiment() -> dict[str, Any]:
    # 1. SGLang PD+DP role switch
    valid_dp_attn, _ = validate_sglang_pd_role_switch_config(
        disaggregation_mode="prefill",
        dp_size=4,
        enable_dp_attention=True,
        enable_dp_attention_local_control_broadcast=False,
    )
    invalid_local_broadcast, _ = validate_sglang_pd_role_switch_config(
        disaggregation_mode="prefill",
        dp_size=4,
        enable_dp_attention=True,
        enable_dp_attention_local_control_broadcast=True,
    )
    legacy_partial_fail = simulate_sglang_pd_role_switch(
        [True, True, False, True], two_phase=False
    )
    twophase_partial_fail = simulate_sglang_pd_role_switch(
        [True, True, False, True], two_phase=True
    )
    twophase_all_ready = simulate_sglang_pd_role_switch(
        [True, True, True, True], two_phase=True
    )

    # 2. TRT-LLM ADP KV cache event collective alignment
    scenarios = [
        ("absent", "processed"),
        ("processed", "deferred"),
        ("deferred", "absent"),
    ]
    buggy_overlap = [
        simulate_trtllm_adp_overlap_step(s, enable_kv_cache_events=True, rebalance=False, fixed=False)
        for s in scenarios
    ]
    fixed_overlap = [
        simulate_trtllm_adp_overlap_step(s, enable_kv_cache_events=True, rebalance=False, fixed=True)
        for s in scenarios
    ]
    buggy_rebalance = simulate_trtllm_adp_overlap_step(
        ("absent", "processed"), enable_kv_cache_events=True, rebalance=True, fixed=False
    )
    fixed_rebalance = simulate_trtllm_adp_overlap_step(
        ("absent", "processed"), enable_kv_cache_events=True, rebalance=True, fixed=True
    )

    # 3. Mooncake Store zero-copy VMM boundary splitting
    mooncake_zc = evaluate_mooncake_zerocopy()

    summary: dict[str, Any] = {
        "experiment_id": "sglang_pd_dp_role_switch_trtllm_adp_kv_mooncake_zc",
        "topic": "SGLang PD+DP Two-Phase Role Switch, TRT-LLM ADP KV Event Collective Alignment & Mooncake Zero-Copy VMM",
        "metrics": {
            "sglang_pd_dp_role_switch": {
                "dp_attention_allowed": valid_dp_attn,
                "local_control_broadcast_rejected": not invalid_local_broadcast,
                "legacy_single_phase_partial_fail": legacy_partial_fail,
                "two_phase_preflight_partial_fail": twophase_partial_fail,
                "two_phase_preflight_all_ready": twophase_all_ready,
            },
            "trtllm_adp_kv_event_alignment": {
                "buggy_overlap_deadlock_free_flags": [r["deadlock_free"] for r in buggy_overlap],
                "fixed_overlap_deadlock_free_flags": [r["deadlock_free"] for r in fixed_overlap],
                "buggy_rebalance_deadlock_free": buggy_rebalance["deadlock_free"],
                "fixed_rebalance_deadlock_free": fixed_rebalance["deadlock_free"],
            },
            "trtllm_mooncake_zerocopy": mooncake_zc,
        },
        "invariants": {
            "sglang_two_phase_prevents_split_brain": (
                legacy_partial_fail["split_brain"] is True
                and twophase_partial_fail["split_brain"] is False
                and twophase_partial_fail["ranks_torn_down"] == 0
                and twophase_all_ready["ranks_switched"] == 4
            ),
            "sglang_rejects_local_control_broadcast_only": valid_dp_attn and not invalid_local_broadcast,
            "trtllm_adp_overlap_eliminates_kv_event_hangs": (
                buggy_overlap[0]["deadlock_free"] is False
                and buggy_overlap[1]["deadlock_free"] is False
                and all(r["deadlock_free"] for r in fixed_overlap)
                and buggy_rebalance["deadlock_free"] is False
                and fixed_rebalance["deadlock_free"] is True
            ),
            "mooncake_zerocopy_splits_cumemcreate_boundaries": (
                mooncake_zc["default_stage_through_host"] is False
                and mooncake_zc["merged_span_count"] == 1
                and mooncake_zc["registration_range_count"] == 3
                and mooncake_zc["straddling_page_split_sizes_mib"] == [2, 2]
                and mooncake_zc["pinned_host_staging_saved_mib_8gpu"] == 1024
            ),
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_experiment()
    base_dir = Path(__file__).resolve().parent
    for path in (base_dir / "summary.json", base_dir / "results" / "summary.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        failed = [k for k, v in summary["invariants"].items() if not v]
        if failed:
            raise SystemExit(f"Invariant check failed: {failed}")
    print(json.dumps(summary["invariants"], indent=2))


if __name__ == "__main__":
    main()
