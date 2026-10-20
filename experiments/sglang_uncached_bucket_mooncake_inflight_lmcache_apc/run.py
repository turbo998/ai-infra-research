from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_sglang_uncached_bucket_routing() -> dict[str, Any]:
    def route_request(
        *,
        input_tokens: int,
        query_blocks: int,
        matched_blocks_by_worker: dict[str, int],
        short_max_context: int | None,
        rank_pending_by_worker: dict[str, list[tuple[int, int]]],
        route_by_uncached_tokens: bool,
    ) -> dict[str, Any]:
        reusable_tokens = 0
        for w, mb in matched_blocks_by_worker.items():
            capped = min(query_blocks, mb)
            est = (input_tokens * capped) // query_blocks if query_blocks > 0 else 0
            reusable_tokens = max(reusable_tokens, est)

        uncached_tokens = (
            max(0, input_tokens - reusable_tokens)
            if (route_by_uncached_tokens and reusable_tokens > 0)
            else None
        )
        cold = route_by_uncached_tokens and uncached_tokens is None

        buckets = [
            {
                "id": "short",
                "min_extend": 0,
                "max_extend": 8192,
                "max_context": short_max_context,
            },
            {
                "id": "long",
                "min_extend": 8193,
                "max_extend": None,
                "max_context": None,
            },
        ]

        def admits(b: dict[str, Any], extend: int | None, ctx: int) -> bool:
            if b["max_context"] is not None and ctx > b["max_context"]:
                return False
            if extend is None:
                return True
            if extend < b["min_extend"]:
                return False
            if b["max_extend"] is not None and extend > b["max_extend"]:
                return False
            return True

        min_rank_pending = {
            w: min(wait_req + wait_uncached for wait_req, wait_uncached in ranks)
            for w, ranks in rank_pending_by_worker.items()
        }

        if not route_by_uncached_tokens:
            eligible = [b["id"] for b in buckets if admits(b, input_tokens, input_tokens)]
            picked = min(eligible, key=lambda w: min_rank_pending[w])
            return {
                "route_by_uncached_tokens": False,
                "uncached_tokens": None,
                "mode": "legacy_input_length",
                "selected_worker": picked,
                "decision": "cache_hit" if matched_blocks_by_worker.get(picked, 0) > 0 else "cache_miss",
            }

        if cold:
            eligible = [b["id"] for b in buckets if admits(b, None, input_tokens)]
            picked = min(eligible, key=lambda w: min_rank_pending[w])
            return {
                "route_by_uncached_tokens": True,
                "uncached_tokens": None,
                "mode": "cold",
                "selected_worker": picked,
                "decision": "cache_miss",
            }

        assert uncached_tokens is not None
        target_buckets = [
            b["id"] for b in buckets if admits(b, uncached_tokens, input_tokens)
        ]
        if not target_buckets:
            target_buckets = [
                b["id"] for b in buckets if admits(b, input_tokens, input_tokens)
            ]

        all_targets_busy = all(min_rank_pending[w] > 0 for w in target_buckets)
        if all_targets_busy:
            spill_candidates = [
                b["id"]
                for b in buckets
                if admits(b, None, input_tokens)
                and b["id"] not in target_buckets
                and min_rank_pending[b["id"]] == 0
            ]
            if spill_candidates:
                return {
                    "route_by_uncached_tokens": True,
                    "uncached_tokens": uncached_tokens,
                    "mode": "idle_spill",
                    "selected_worker": spill_candidates[0],
                    "decision": "idle_spill",
                }

        picked = target_buckets[0]
        return {
            "route_by_uncached_tokens": True,
            "uncached_tokens": uncached_tokens,
            "mode": "warm",
            "selected_worker": picked,
            "decision": "cache_hit",
        }

    return {
        "warm_20k_15of20_cached_routes_short": route_request(
            input_tokens=20000,
            query_blocks=20,
            matched_blocks_by_worker={"long": 15},
            short_max_context=None,
            rank_pending_by_worker={"short": [(0, 0)], "long": [(0, 0)]},
            route_by_uncached_tokens=True,
        ),
        "warm_20k_15of20_legacy_routes_long": route_request(
            input_tokens=20000,
            query_blocks=20,
            matched_blocks_by_worker={"long": 15},
            short_max_context=None,
            rank_pending_by_worker={"short": [(0, 0)], "long": [(0, 0)]},
            route_by_uncached_tokens=False,
        ),
        "warm_long_busy_spills_to_idle_short_rank": route_request(
            input_tokens=20000,
            query_blocks=20,
            matched_blocks_by_worker={"long": 5},
            short_max_context=None,
            rank_pending_by_worker={"long": [(2, 9)], "short": [(3, 9), (0, 0)]},
            route_by_uncached_tokens=True,
        ),
        "warm_long_busy_no_spill_when_short_chunked_prefill_active": route_request(
            input_tokens=20000,
            query_blocks=20,
            matched_blocks_by_worker={"long": 5},
            short_max_context=None,
            rank_pending_by_worker={"long": [(2, 9)], "short": [(3, 9), (0, 40000)]},
            route_by_uncached_tokens=True,
        ),
        "warm_short_context_cap_falls_back_to_long": route_request(
            input_tokens=20000,
            query_blocks=20,
            matched_blocks_by_worker={"long": 15},
            short_max_context=10000,
            rank_pending_by_worker={"short": [(0, 0)], "long": [(0, 0)]},
            route_by_uncached_tokens=True,
        ),
    }


def simulate_mooncake_bounded_inflight_reshard() -> dict[str, Any]:
    total_batches = 8
    submit_overhead_ms = 2.0
    transfer_duration_ms = 12.0

    def timeline_ms(max_inflight: int) -> float:
        completion_times: list[float] = []
        now = 0.0
        for idx in range(total_batches):
            if len(completion_times) >= max_inflight:
                oldest_done = completion_times.pop(0)
                now = max(now, oldest_done)
            now += submit_overhead_ms
            completion_times.append(now + transfer_duration_ms)
        if completion_times:
            now = max(now, max(completion_times))
        return round(now, 2)

    sync_w1_ms = timeline_ms(1)
    async_w4_ms = timeline_ms(4)

    child_consume_ms = [400, 350, 500]
    budget_ms = 1000
    elapsed = 0
    remaining_passed: list[int] = []
    for c_ms in child_consume_ms:
        rem = max(0, budget_ms - elapsed)
        remaining_passed.append(rem)
        elapsed += min(rem, c_ms)

    return {
        "total_batches": total_batches,
        "window_1_sync_total_ms": sync_w1_ms,
        "window_4_inflight_total_ms": async_w4_ms,
        "wall_time_reduction_pct": round(
            (sync_w1_ms - async_w4_ms) * 100.0 / sync_w1_ms, 2
        ),
        "composite_deadline_budget_ms": budget_ms,
        "child_remaining_budgets_ms": remaining_passed,
        "composite_total_waited_ms": elapsed,
        "quarantined_undrainable_retains_owner_ref": True,
    }


def simulate_lmcache_apc_covered_lookup_metrics() -> dict[str, Any]:
    chunk_size = 256
    prompt_tokens = 2100
    num_chunks = prompt_tokens // chunk_size
    key_start = 512
    engine_computed_tokens_hint = 600
    l1_found_in_subrange = 2
    l2_found_in_subrange = 3

    start_chunk = min(max(0, key_start // chunk_size), num_chunks)
    apc_covered_tokens = start_chunk * chunk_size
    total_chunk_aligned_tokens = num_chunks * chunk_size

    legacy_requested_tokens = total_chunk_aligned_tokens
    legacy_l1_hit_tokens = (start_chunk + l1_found_in_subrange) * chunk_size
    legacy_l2_hit_tokens = l2_found_in_subrange * chunk_size
    legacy_hit_tokens = legacy_l1_hit_tokens + legacy_l2_hit_tokens

    new_requested_tokens = total_chunk_aligned_tokens - apc_covered_tokens
    new_l1_hit_tokens = l1_found_in_subrange * chunk_size
    new_l2_hit_tokens = l2_found_in_subrange * chunk_size
    new_hit_tokens = new_l1_hit_tokens + new_l2_hit_tokens
    request_hit_tokens = max(
        engine_computed_tokens_hint, apc_covered_tokens + new_hit_tokens
    )

    return {
        "chunk_size": chunk_size,
        "prompt_tokens": prompt_tokens,
        "start_chunk": start_chunk,
        "apc_covered_tokens": apc_covered_tokens,
        "engine_hit_tokens": engine_computed_tokens_hint,
        "legacy_metrics": {
            "requested_tokens": legacy_requested_tokens,
            "l1_hit_tokens": legacy_l1_hit_tokens,
            "l2_hit_tokens": legacy_l2_hit_tokens,
            "hit_tokens": legacy_hit_tokens,
            "lookup_hit_rate": round(legacy_hit_tokens / legacy_requested_tokens, 4),
        },
        "decoupled_metrics": {
            "requested_tokens": new_requested_tokens,
            "l1_hit_tokens": new_l1_hit_tokens,
            "l2_hit_tokens": new_l2_hit_tokens,
            "hit_tokens": new_hit_tokens,
            "lookup_range_hit_rate": round(new_hit_tokens / new_requested_tokens, 4),
            "request_hit_tokens": request_hit_tokens,
            "request_hit_rate": round(request_hit_tokens / prompt_tokens, 4),
        },
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "sglang_uncached_bucket_mooncake_inflight_lmcache_apc",
        "evidence_class": "own-offline",
        "sglang_router_uncached_buckets": simulate_sglang_uncached_bucket_routing(),
        "mooncake_bounded_inflight_reshard": simulate_mooncake_bounded_inflight_reshard(),
        "lmcache_apc_covered_lookup_metrics": simulate_lmcache_apc_covered_lookup_metrics(),
    }


def verify_summary(summary: dict[str, Any]) -> None:
    sgl = summary["sglang_router_uncached_buckets"]
    assert (
        sgl["warm_20k_15of20_cached_routes_short"]["selected_worker"] == "short"
    )
    assert sgl["warm_20k_15of20_cached_routes_short"]["uncached_tokens"] == 5000
    assert (
        sgl["warm_20k_15of20_legacy_routes_long"]["selected_worker"] == "long"
    )
    assert (
        sgl["warm_long_busy_spills_to_idle_short_rank"]["decision"]
        == "idle_spill"
    )
    assert (
        sgl["warm_long_busy_spills_to_idle_short_rank"]["selected_worker"]
        == "short"
    )
    assert (
        sgl["warm_long_busy_no_spill_when_short_chunked_prefill_active"][
            "selected_worker"
        ]
        == "long"
    )
    assert (
        sgl["warm_short_context_cap_falls_back_to_long"]["selected_worker"]
        == "long"
    )

    mc = summary["mooncake_bounded_inflight_reshard"]
    assert mc["window_1_sync_total_ms"] == 112.0
    assert mc["window_4_inflight_total_ms"] == 34.0
    assert mc["wall_time_reduction_pct"] == 69.64
    assert mc["child_remaining_budgets_ms"] == [1000, 600, 250]
    assert mc["composite_total_waited_ms"] == 1000

    lmc = summary["lmcache_apc_covered_lookup_metrics"]
    assert lmc["apc_covered_tokens"] == 512
    assert lmc["legacy_metrics"]["l1_hit_tokens"] == 1024
    assert lmc["decoupled_metrics"]["l1_hit_tokens"] == 512
    assert lmc["decoupled_metrics"]["requested_tokens"] == 1536
    assert lmc["decoupled_metrics"]["hit_tokens"] == 1280
    assert lmc["decoupled_metrics"]["request_hit_tokens"] == 1792


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    summary = build_summary()
    verify_summary(summary)

    out_paths = [base_dir / "summary.json", base_dir / "results" / "summary.json"]
    if args.check:
        for p in out_paths:
            existing = json.loads(p.read_text(encoding="utf-8"))
            verify_summary(existing)
            assert existing == summary, f"Mismatch in {p}"
        print("sglang_uncached_bucket_mooncake_inflight_lmcache_apc: check passed")
        return

    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rendered, encoding="utf-8")
    print("sglang_uncached_bucket_mooncake_inflight_lmcache_apc: generated summary.json")


if __name__ == "__main__":
    main()
