"""Deterministic simulation of SGLang KVShard page-interleaved PD transfer
(`filter_kv_indices_for_shard_rank`, `max_bootstrap_input_len`), `sgl-router`
uncached-token prefill bucket routing (`--route-by-uncached-tokens`,
`idle_spill_domains`), and Qwen3 MoE EPLB post-rebalance `ExpertLocationDispatchInfo`.

Verified against `sgl-project/sglang` commits:
- `eb4b167d2749840eae18167fe9e0a608308e45aa`:
  - `python/sglang/srt/disaggregation/common/conn.py`
  - `python/sglang/srt/disaggregation/prefill.py`
  - `python/sglang/srt/disaggregation/utils.py`
- `df13d5dce31abfeff0e701a62e8c39a136758e6c`:
  - `experimental/sgl-router/src/policies/buckets.rs`
  - `experimental/sgl-router/src/policies/selection.rs`
- `9db58ab547516435b544ae050dd6c2f91b52bcf3`:
  - `python/sglang/srt/models/qwen3_moe.py`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SUMMARY_PATH = Path(__file__).resolve().parent / "results" / "summary.json"


def max_bootstrap_input_len(*, capacity: int, page_size: int, shard_size: int) -> int:
    """Exact mirror of `max_bootstrap_input_len` in `sglang/srt/disaggregation/prefill.py`."""
    if shard_size == 1:
        return capacity
    return (capacity - shard_size * page_size - 3) // page_size * page_size


def filter_kv_indices_for_shard_rank(
    *,
    kv_indices: list[int],
    slice_start: int,
    slice_stop: int,
    shard_rank: int,
    shard_size: int,
) -> tuple[list[int], list[int]]:
    """Exact mirror of `filter_kv_indices_for_shard_rank` in `sglang/srt/disaggregation/utils.py`."""
    positions = list(range(slice_start, slice_stop))
    local_pages: list[int] = []
    send_positions: list[int] = []
    for logical_page, pos in zip(kv_indices, positions):
        if logical_page % shard_size == shard_rank:
            local_pages.append(logical_page // shard_size)
            send_positions.append(pos)
    return local_pages, send_positions


def simulate_kvshard_pd_transfer() -> dict[str, Any]:
    """Simulate 4-way page-interleaved KVShard PD disaggregation transfer."""
    shard_size = 4
    page_size = 64
    per_rank_physical_tokens = 16_384
    widened_capacity = per_rank_physical_tokens * shard_size  # 65,536
    bootstrap_limit = max_bootstrap_input_len(
        capacity=widened_capacity, page_size=page_size, shard_size=shard_size
    )  # (65536 - 256 - 3) // 64 * 64 = 65,216

    # Simulate a 12-page prompt transferred in 3 chunks of 4 pages each, with non-contiguous logical page IDs:
    logical_pages = [12, 16, 20, 24, 5, 6, 7, 8, 9, 10, 11, 13]
    # Notice chunk 0 ([12, 16, 20, 24]) has all pages % 4 == 0!
    # So shard_rank 0 owns all 4 pages in chunk 0, while shard_ranks 1, 2, 3 have 0 pages in chunk 0
    # and skip non-final chunk 0 (`skip_non_final_empty = True`).
    chunks = [(0, 4), (4, 8), (8, 12)]
    rank_transfers: dict[str, Any] = {}
    reconstructed: dict[int, tuple[int, int]] = {}
    skipped_empty_non_final_chunks = 0

    for r in range(shard_size):
        rank_chunks = []
        for c_idx, (s, e) in enumerate(chunks):
            is_last_chunk = e == len(logical_pages)
            local_ids, send_pos = filter_kv_indices_for_shard_rank(
                kv_indices=logical_pages[s:e],
                slice_start=s,
                slice_stop=e,
                shard_rank=r,
                shard_size=shard_size,
            )
            skip_chunk = len(local_ids) == 0 and not is_last_chunk
            if skip_chunk:
                skipped_empty_non_final_chunks += 1
            for lid, pos in zip(local_ids, send_pos):
                reconstructed[pos] = (r, lid * shard_size + r)
            rank_chunks.append(
                {
                    "chunk_idx": c_idx,
                    "local_ids": local_ids,
                    "send_positions": send_pos,
                    "is_last_chunk": is_last_chunk,
                    "skipped": skip_chunk,
                }
            )
        rank_transfers[f"rank_{r}"] = rank_chunks

    lossless_reconstruction = [reconstructed[i][1] for i in range(len(logical_pages))] == logical_pages

    return {
        "shard_size": shard_size,
        "page_size": page_size,
        "per_rank_physical_tokens": per_rank_physical_tokens,
        "widened_logical_capacity_tokens": widened_capacity,
        "max_bootstrap_input_len": bootstrap_limit,
        "admission_guard_margin_tokens": widened_capacity - bootstrap_limit,
        "skipped_empty_non_final_chunks": skipped_empty_non_final_chunks,
        "lossless_page_reconstruction": lossless_reconstruction,
        "rank_transfers": rank_transfers,
    }


def select_prefill_bucket_route(
    *,
    sequence_tokens: int,
    input_tokens: int,
    reusable_tokens: int | None,
    short_max_context: int | None,
    short_ranks: list[tuple[int, int]],
    long_ranks: list[tuple[int, int]],
    route_by_uncached_tokens: bool,
) -> dict[str, Any]:
    """Exact mirror of `Selector::run` and `idle_spill_domains` in `sgl-router/src/policies/selection.rs`.
    `short` bucket: `max_extend_tokens = 8192`, `max_context_tokens = short_max_context`.
    `long` bucket: `min_extend_tokens = 8193`, `max_context_tokens = None`.
    Each rank entry is `(waiting_reqs, waiting_uncached_tokens)`.
    """
    short_max_extend = 8192
    long_min_extend = 8193

    def prefill_compatible(bucket: str, extend: int | None, inp: int) -> bool:
        if bucket == "short":
            ext_ok = extend is None or (0 <= extend <= short_max_extend)
            ctx_ok = short_max_context is None or inp <= short_max_context
            return ext_ok and ctx_ok
        else:
            return extend is None or extend >= long_min_extend

    def min_rank_pending(ranks: list[tuple[int, int]]) -> int:
        # Pending > 0 if waiting > 0 or uncached > 0
        return min(w + u for w, u in ranks)

    uncached_tokens: int | None = None
    if route_by_uncached_tokens and reusable_tokens is not None and reusable_tokens > 0:
        uncached_tokens = max(0, sequence_tokens - reusable_tokens)

    cold = route_by_uncached_tokens and uncached_tokens is None

    # Determine extend_bucket_workers
    extend_workers: list[str] = []
    if uncached_tokens is not None:
        for b in ("short", "long"):
            if prefill_compatible(b, uncached_tokens, sequence_tokens):
                extend_workers.append(b)

    # Check idle_spill_domains
    rank_map = {"short": short_ranks, "long": long_ranks}
    idle_spill_workers: list[str] = []
    if extend_workers and all(min_rank_pending(rank_map[w]) > 0 for w in extend_workers):
        for b in ("short", "long"):
            if (
                b not in extend_workers
                and prefill_compatible(b, None, sequence_tokens)
                and min_rank_pending(rank_map[b]) == 0
            ):
                idle_spill_workers.append(b)

    if idle_spill_workers:
        return {"selected": idle_spill_workers[0], "mode": "idle_spill", "uncached_tokens": uncached_tokens}

    if uncached_tokens is not None:
        if extend_workers:
            return {"selected": extend_workers[0], "mode": "warm", "uncached_tokens": uncached_tokens}
        # Fallback to input-length matching when no bucket admits E (e.g., short_max_context rejected it)
        for b in ("short", "long"):
            if prefill_compatible(b, sequence_tokens, sequence_tokens):
                return {"selected": b, "mode": "warm_fallback_input_len", "uncached_tokens": uncached_tokens}

    if cold:
        admitted = [b for b in ("short", "long") if prefill_compatible(b, None, sequence_tokens)]
        # Pick the less loaded worker
        selected = min(admitted, key=lambda w: min_rank_pending(rank_map[w]))
        return {"selected": selected, "mode": "cold", "uncached_tokens": None}

    # Legacy route_by_uncached_tokens = False: matches raw sequence_tokens
    for b in ("short", "long"):
        if prefill_compatible(b, sequence_tokens, sequence_tokens):
            return {"selected": b, "mode": "legacy_input_len", "uncached_tokens": None}
    return {"selected": "long", "mode": "legacy_input_len", "uncached_tokens": None}


def simulate_router_and_eplb() -> dict[str, Any]:
    """Verify all test cases from `routes_prefill_buckets_by_uncached_tokens`,
    `spills_to_a_worker_with_an_idle_dp_rank`, and Qwen3 MoE EPLB dispatch mapping.
    """
    cases = [
        {
            "name": "cold_20k_short_idle_long_busy",
            "reusable": None,
            "short_max_ctx": None,
            "short_ranks": [(0, 0)],
            "long_ranks": [(3, 900_000)],
        },
        {
            "name": "warm_20k_reusable_15k_e_5k",
            "reusable": 15_000,
            "short_max_ctx": None,
            "short_ranks": [(0, 0)],
            "long_ranks": [(0, 0)],
        },
        {
            "name": "warm_20k_reusable_10k_e_10k",
            "reusable": 10_000,
            "short_max_ctx": None,
            "short_ranks": [(0, 0)],
            "long_ranks": [(0, 0)],
        },
        {
            "name": "warm_20k_e_5k_rejected_by_short_ctx_cap_10k",
            "reusable": 15_000,
            "short_max_ctx": 10_000,
            "short_ranks": [(0, 0)],
            "long_ranks": [(0, 0)],
        },
        {
            "name": "spill_long_busy_to_short_with_idle_dp_rank",
            "reusable": 5_000,  # E = 15,000 -> long
            "short_max_ctx": None,
            "short_ranks": [(3, 9), (0, 0)],
            "long_ranks": [(2, 9)],
        },
        {
            "name": "no_spill_when_short_rank_prefilling_chunked_40k",
            "reusable": 5_000,  # E = 15,000 -> long
            "short_max_ctx": None,
            "short_ranks": [(3, 9), (0, 40_000)],
            "long_ranks": [(2, 9)],
        },
    ]

    results = []
    for c in cases:
        with_uncached = select_prefill_bucket_route(
            sequence_tokens=20_000,
            input_tokens=40_000,
            reusable_tokens=c["reusable"],
            short_max_context=c["short_max_ctx"],
            short_ranks=c["short_ranks"],
            long_ranks=c["long_ranks"],
            route_by_uncached_tokens=True,
        )
        without_uncached = select_prefill_bucket_route(
            sequence_tokens=20_000,
            input_tokens=40_000,
            reusable_tokens=c["reusable"],
            short_max_context=c["short_max_ctx"],
            short_ranks=c["short_ranks"],
            long_ranks=c["long_ranks"],
            route_by_uncached_tokens=False,
        )
        results.append(
            {
                "case": c["name"],
                "with_uncached_routing": with_uncached["selected"],
                "without_uncached_routing": without_uncached["selected"],
                "mode": with_uncached["mode"],
                "uncached_tokens": with_uncached["uncached_tokens"],
            }
        )

    # Simulate Qwen3 MoE EPLB rebalance permutation (`9db58ab547`)
    # After EPLB rebalance, logical expert `e` is relocated to physical slot `logical_to_physical[e]`.
    logical_to_physical = [5, 2, 7, 0, 6, 1, 3, 4]
    physical_weights = [10.0 * (e + 1) for e in range(8)]
    # Inverted table: physical slot `p` holds weight of logical expert `e` where `logical_to_physical[e] == p`
    rebalanced_physical_weights = [0.0] * 8
    for logical_e, phys_slot in enumerate(logical_to_physical):
        rebalanced_physical_weights[phys_slot] = physical_weights[logical_e]

    topk_logical = [0, 3, 7]
    # Without `expert_location_dispatch_info`, topk returns raw logical IDs -> indexes wrong physical weights!
    unmapped_out = sum(rebalanced_physical_weights[i] for i in topk_logical)
    mapped_out = sum(rebalanced_physical_weights[logical_to_physical[i]] for i in topk_logical)
    expected_out = sum(physical_weights[i] for i in topk_logical)

    return {
        "routing_cases": results,
        "eplb_qwen3_moe": {
            "expected_output": expected_out,
            "without_dispatch_info_output": unmapped_out,
            "with_dispatch_info_output": mapped_out,
            "corrupted_without_dispatch_info": unmapped_out != expected_out,
            "exact_with_dispatch_info": mapped_out == expected_out,
        },
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "sglang_kvshard_uncached_bucket_sim",
        "source_commits": [
            "sgl-project/sglang@eb4b167d2749840eae18167fe9e0a608308e45aa",
            "sgl-project/sglang@df13d5dce31abfeff0e701a62e8c39a136758e6c",
            "sgl-project/sglang@9db58ab547516435b544ae050dd6c2f91b52bcf3",
        ],
        "kvshard_pd": simulate_kvshard_pd_transfer(),
        "router_and_eplb": simulate_router_and_eplb(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify against committed summary.json")
    args = parser.parse_args()

    summary = build_summary()
    assert summary["kvshard_pd"]["max_bootstrap_input_len"] == 65_216
    assert summary["kvshard_pd"]["skipped_empty_non_final_chunks"] == 3
    assert summary["kvshard_pd"]["lossless_page_reconstruction"] is True
    cases = {c["case"]: c for c in summary["router_and_eplb"]["routing_cases"]}
    assert cases["cold_20k_short_idle_long_busy"]["with_uncached_routing"] == "short"
    assert cases["warm_20k_reusable_15k_e_5k"]["with_uncached_routing"] == "short"
    assert cases["warm_20k_e_5k_rejected_by_short_ctx_cap_10k"]["with_uncached_routing"] == "long"
    assert cases["spill_long_busy_to_short_with_idle_dp_rank"]["with_uncached_routing"] == "short"
    assert cases["no_spill_when_short_rank_prefilling_chunked_40k"]["with_uncached_routing"] == "long"
    assert summary["router_and_eplb"]["eplb_qwen3_moe"]["exact_with_dispatch_info"] is True

    if args.check:
        if not SUMMARY_PATH.exists():
            print(f"Missing {SUMMARY_PATH}", file=sys.stderr)
            return 1
        committed = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        if committed != summary:
            print("Mismatch between computed and committed summary.json", file=sys.stderr)
            return 1
        print("sglang_kvshard_uncached_bucket_sim: OK")
        return 0

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
