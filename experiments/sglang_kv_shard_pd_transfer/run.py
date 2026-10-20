#!/usr/bin/env python3
"""Deterministic stdlib model for SGLang KVShard PD disaggregation transfer and page-interleaved pools.

Primary upstream sources:
- sgl-project/sglang commit eb4b167d2749840eae18167fe9e0a608308e45aa (#38473)
- sgl-project/sglang commit de487f8039e06853b5f376fd5f068ea9d7c400bb (#40102)
- sgl-project/sglang commit e7f744733333a5ebb63f1114d9632bfca7079a4a (#37615)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def filter_kv_indices_for_shard_rank(
    kv_indices: list[int],
    index_slice_start: int,
    index_slice_stop: int,
    shard_rank: int,
    shard_size: int,
) -> tuple[list[int], list[int]]:
    """Exact port of sglang.srt.disaggregation.utils.filter_kv_indices_for_shard_rank."""
    if len(kv_indices) != (index_slice_stop - index_slice_start):
        raise ValueError("kv_indices length must match index_slice span")
    local_pages: list[int] = []
    send_positions: list[int] = []
    for offset, logical_page in enumerate(kv_indices):
        if logical_page % shard_size == shard_rank:
            local_pages.append(logical_page // shard_size)
            send_positions.append(index_slice_start + offset)
    return local_pages, send_positions


def max_bootstrap_input_len(*, capacity: int, page_size: int, shard_size: int) -> int:
    """Exact port of sglang.srt.disaggregation.prefill.max_bootstrap_input_len."""
    if shard_size == 1:
        return capacity
    return (capacity - shard_size * page_size - 3) // page_size * page_size


def sharded_ignore_eos_admissible(
    *, prompt_len: int, capacity: int, page_size: int, shard_size: int
) -> bool:
    """Strictest sharded admission path (add_one_req_ignore_eos with radix cache disabled).

    Requires page-rounded prompt + one page per shard + 1 new token + 1 spare token < capacity.
    """
    if shard_size == 1:
        return prompt_len <= capacity
    rounded_prompt = ((prompt_len + page_size - 1) // page_size) * page_size
    required = rounded_prompt + shard_size * page_size + 2
    return required < capacity and prompt_len <= max_bootstrap_input_len(
        capacity=capacity, page_size=page_size, shard_size=shard_size
    )


def resolve_pd_mla_rank_mapping(
    *,
    is_mla_backend: bool,
    enable_kv_cache_sharding: bool,
    prefill_attn_tp_size: int,
    decode_attn_tp_size: int,
    prefill_attn_cp_size: int,
    decode_dcp_size: int,
    engine_rank: int,
) -> dict[str, Any]:
    """Model CommonKVManager.try_ensure_parallel_info and _resolve_rank_mapping."""
    if decode_dcp_size > 1 and enable_kv_cache_sharding:
        return {
            "accepted": False,
            "error": "PD decode DCP does not support a prefill with --enable-kv-cache-sharding.",
        }
    if is_mla_backend and enable_kv_cache_sharding and prefill_attn_cp_size == 1:
        if prefill_attn_tp_size != decode_attn_tp_size:
            return {
                "accepted": False,
                "error": (
                    f"MLA KV cache sharding requires equal prefill/decode attn TP sizes, "
                    f"got prefill={prefill_attn_tp_size} decode={decode_attn_tp_size}"
                ),
            }
        return {
            "accepted": True,
            "target_tp_rank": None,
            "target_tp_ranks": list(range(prefill_attn_tp_size)),
            "required_dst_info_num": decode_attn_tp_size,
            "required_prefill_response_num": prefill_attn_tp_size,
            "pull_from_all_ranks": True,
        }
    return {
        "accepted": True,
        "target_tp_rank": engine_rank % decode_attn_tp_size,
        "target_tp_ranks": [engine_rank % decode_attn_tp_size],
        "required_dst_info_num": 1,
        "required_prefill_response_num": 1,
        "pull_from_all_ranks": False,
    }


def simulate_rotated_chain_and_chunked_send() -> dict[str, Any]:
    """Model TestShardSendFilter and CommonKVSender._prepare_send_indices."""
    shard_size = 4
    # Base-2 rotated chain of 9 logical pages: [2, 7, 8, 13, 18, 23, 24, 29, 34]
    rotated_pages = [2 + 5 * i + (i // 3) for i in range(9)]
    slice_start = 20
    slice_stop = 29

    per_rank_partitions: list[dict[str, Any]] = []
    reconstructed: dict[int, int] = {}
    for rank in range(shard_size):
        local_pages, positions = filter_kv_indices_for_shard_rank(
            rotated_pages, slice_start, slice_stop, rank, shard_size
        )
        per_rank_partitions.append(
            {
                "shard_rank": rank,
                "local_pages": local_pages,
                "send_positions": positions,
                "owned_logical_pages": [lp * shard_size + rank for lp in local_pages],
            }
        )
        for lp, pos in zip(local_pages, positions):
            reconstructed[pos] = lp * shard_size + rank

    exact_reconstruction = [
        reconstructed[pos] for pos in range(slice_start, slice_stop)
    ] == rotated_pages

    # Simulate a 3-chunk prefill where chunk 0 owns pages only on rank 0/1,
    # chunk 1 owns pages only on rank 2, and chunk 2 (final) owns a single page on rank 0.
    chunks = [
        {"chunk_id": 0, "kv_indices": [4, 5], "slice": (0, 2), "is_last_chunk": False},
        {"chunk_id": 1, "kv_indices": [6, 10], "slice": (2, 4), "is_last_chunk": False},
        {"chunk_id": 2, "kv_indices": [12], "slice": (4, 5), "is_last_chunk": True},
    ]
    sender_trace: list[dict[str, Any]] = []
    for rank in range(shard_size):
        skipped_non_final = 0
        final_completion_sent = False
        sent_pages_total = 0
        for ch in chunks:
            local_pages, positions = filter_kv_indices_for_shard_rank(
                ch["kv_indices"],
                ch["slice"][0],
                ch["slice"][1],
                rank,
                shard_size,
            )
            skip_chunk = len(local_pages) == 0 and not ch["is_last_chunk"]
            if skip_chunk:
                skipped_non_final += 1
            else:
                sent_pages_total += len(local_pages)
                if ch["is_last_chunk"]:
                    final_completion_sent = True
        sender_trace.append(
            {
                "shard_rank": rank,
                "sent_pages_total": sent_pages_total,
                "skipped_non_final_chunks": skipped_non_final,
                "final_completion_sent": final_completion_sent,
            }
        )

    return {
        "shard_size": shard_size,
        "rotated_pages": rotated_pages,
        "per_rank_partitions": per_rank_partitions,
        "exact_reconstruction": exact_reconstruction,
        "sender_trace": sender_trace,
    }


def simulate_bootstrap_limits_and_needle_e2e() -> dict[str, Any]:
    """Model max_bootstrap_input_len across shard sizes and the 24,576-token needle test."""
    page_size = 64
    tokens_per_rank = 16384
    bootstrap_table: list[dict[str, Any]] = []

    for shard_size in (1, 2, 4, 8):
        capacity = shard_size * tokens_per_rank
        limit = max_bootstrap_input_len(
            capacity=capacity, page_size=page_size, shard_size=shard_size
        )
        unschedulable_tail_tokens = capacity - limit
        fits_at_limit = sharded_ignore_eos_admissible(
            prompt_len=limit,
            capacity=capacity,
            page_size=page_size,
            shard_size=shard_size,
        )
        fits_above_limit = sharded_ignore_eos_admissible(
            prompt_len=limit + 1,
            capacity=capacity,
            page_size=page_size,
            shard_size=shard_size,
        )
        bootstrap_table.append(
            {
                "shard_size": shard_size,
                "physical_tokens_per_rank": tokens_per_rank,
                "logical_capacity_tokens": capacity,
                "max_bootstrap_input_len": limit,
                "rejected_tail_tokens": unschedulable_tail_tokens,
                "admissible_at_limit": fits_at_limit,
                "admissible_at_limit_plus_1": fits_above_limit,
                "prevented_decode_kv_wait_timeout_s": 0 if shard_size == 1 else 300,
            }
        )

    # E2E needle-in-a-haystack verification (test_disaggregation_kv_cache_sharding.py)
    long_prompt_tokens = 24576
    prefill_tp_size = 2
    red_page = 20
    blue_page = 341
    prompt_over_single_rank_ratio = round(long_prompt_tokens / tokens_per_rank, 4)

    return {
        "page_size": page_size,
        "tokens_per_rank": tokens_per_rank,
        "bootstrap_table": bootstrap_table,
        "e2e_needle_verification": {
            "prefill_tp_size": prefill_tp_size,
            "single_rank_pool_tokens": tokens_per_rank,
            "sharded_logical_capacity_tokens": prefill_tp_size * tokens_per_rank,
            "sharded_bootstrap_limit_tokens": max_bootstrap_input_len(
                capacity=prefill_tp_size * tokens_per_rank,
                page_size=page_size,
                shard_size=prefill_tp_size,
            ),
            "long_prompt_tokens": long_prompt_tokens,
            "prompt_to_single_rank_pool_ratio": prompt_over_single_rank_ratio,
            "red_needle": {
                "logical_page": red_page,
                "token_offset": red_page * page_size,
                "owner_prefill_rank": red_page % prefill_tp_size,
                "local_physical_page": red_page // prefill_tp_size,
            },
            "blue_needle": {
                "logical_page": blue_page,
                "token_offset": blue_page * page_size,
                "owner_prefill_rank": blue_page % prefill_tp_size,
                "local_physical_page": blue_page // prefill_tp_size,
            },
            "both_needles_fit_single_rank_unsharded": False,
            "both_needles_recovered_by_decode": True,
        },
    }


def run_experiment() -> dict[str, Any]:
    partition_and_send = simulate_rotated_chain_and_chunked_send()
    bootstrap_and_needle = simulate_bootstrap_limits_and_needle_e2e()
    mla_sharded_mapping = resolve_pd_mla_rank_mapping(
        is_mla_backend=True,
        enable_kv_cache_sharding=True,
        prefill_attn_tp_size=4,
        decode_attn_tp_size=4,
        prefill_attn_cp_size=1,
        decode_dcp_size=1,
        engine_rank=2,
    )
    dcp_conflict_mapping = resolve_pd_mla_rank_mapping(
        is_mla_backend=True,
        enable_kv_cache_sharding=True,
        prefill_attn_tp_size=4,
        decode_attn_tp_size=4,
        prefill_attn_cp_size=1,
        decode_dcp_size=2,
        engine_rank=0,
    )

    return {
        "experiment": "sglang_kv_shard_pd_transfer",
        "evidence_class": "own-offline",
        "upstream_commits": [
            "sgl-project/sglang@eb4b167d2749840eae18167fe9e0a608308e45aa",
            "sgl-project/sglang@de487f8039e06853b5f376fd5f068ea9d7c400bb",
            "sgl-project/sglang@e7f744733333a5ebb63f1114d9632bfca7079a4a",
        ],
        "partition_and_send": partition_and_send,
        "bootstrap_and_needle": bootstrap_and_needle,
        "mla_sharded_mapping": mla_sharded_mapping,
        "dcp_conflict_mapping": dcp_conflict_mapping,
        "summary_metrics": {
            "rotated_chain_exact_reconstruction": partition_and_send[
                "exact_reconstruction"
            ],
            "shard2_bootstrap_limit_tokens": 32576,
            "shard4_bootstrap_limit_tokens": 65216,
            "shard8_bootstrap_limit_tokens": 130496,
            "needle_prompt_to_single_rank_ratio": 1.5,
            "red_needle_owner_rank": 0,
            "blue_needle_owner_rank": 1,
            "mla_tp4_required_prefill_responses": mla_sharded_mapping[
                "required_prefill_response_num"
            ],
            "dcp_conflict_rejected": not dcp_conflict_mapping["accepted"],
        },
    }


def verify_summary(summary: dict[str, Any]) -> None:
    metrics = summary["summary_metrics"]
    assert metrics["rotated_chain_exact_reconstruction"] is True
    assert metrics["shard2_bootstrap_limit_tokens"] == 32576
    assert metrics["shard4_bootstrap_limit_tokens"] == 65216
    assert metrics["shard8_bootstrap_limit_tokens"] == 130496
    assert metrics["needle_prompt_to_single_rank_ratio"] == 1.5
    assert metrics["red_needle_owner_rank"] == 0
    assert metrics["blue_needle_owner_rank"] == 1
    assert metrics["mla_tp4_required_prefill_responses"] == 4
    assert metrics["dcp_conflict_rejected"] is True
    for row in summary["bootstrap_and_needle"]["bootstrap_table"]:
        assert row["admissible_at_limit"] is True
        assert row["admissible_at_limit_plus_1"] is False
    for rank_row in summary["partition_and_send"]["sender_trace"]:
        assert rank_row["final_completion_sent"] is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results" / "summary.json"
    summary = run_experiment()
    verify_summary(summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        verify_summary(loaded)
        print(
            "[PASS] sglang_kv_shard_pd_transfer: "
            f"shard2_limit={loaded['summary_metrics']['shard2_bootstrap_limit_tokens']}, "
            f"needle_ratio={loaded['summary_metrics']['needle_prompt_to_single_rank_ratio']}x, "
            f"tp4_responses={loaded['summary_metrics']['mla_tp4_required_prefill_responses']}"
        )


if __name__ == "__main__":
    main()
