"""Deterministic simulation of LMCache Device-DAX/CXL L1 range-offset partitioning
(`check_dax_range`, `_DevDaxArena.overlaps`), APC-covered prefix exclusion in
lookup hit attribution (`lookup.py`), and Dynamo Cuckoo DC relay snapshot-tail
removal replay (`cuckoo/dc.rs`) & post-migration prompt usage rebasing (`migration.rs`).

Verified against:
- `LMCache/LMCache` commits:
  - `e8dc1a1daeb918aa5a1797d2ecc2c6b2cdeab1a3` (`lmcache/utils.py`, `devdax_memory_allocator.py`)
  - `168130df9c8fafa376b650814598179c212d3e66` (`lmcache/v1/multiprocess/modules/lookup.py`)
- `ai-dynamo/dynamo` commits:
  - `909ebee0c4b63988bea6ab474a46913073e4a2dc` (`lib/kv-router/src/indexer/cuckoo/dc.rs`)
  - `cc02d8b1c2b60a9cbe291b5264456897c64d7e7c` (`lib/llm/src/migration.rs`)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SUMMARY_PATH = Path(__file__).resolve().parent / "results" / "summary.json"

GIB = 1 << 30
MIB = 1 << 20
ALLOC_GRANULARITY = 65_536  # 64 KiB


def validate_devdax_arena_add(
    *,
    existing_arenas: list[tuple[int, int]],
    device_capacity: int,
    device_align: int,
    offset: int,
    size: int,
) -> dict[str, Any]:
    """Mirror `_check_devdax_offset`, `check_dax_range`, and `_DevDaxArena.overlaps`."""
    if offset < 0 or offset % ALLOC_GRANULARITY != 0:
        return {"admitted": False, "reason": "offset_not_alloc_granularity_aligned"}
    if offset + size > device_capacity:
        return {"admitted": False, "reason": "exceeds_device_capacity"}
    if offset % device_align != 0 or size % device_align != 0:
        return {"admitted": False, "reason": "breaks_device_alignment"}
    for ar_offset, ar_size in existing_arenas:
        if offset < ar_offset + ar_size and ar_offset < offset + size:
            return {"admitted": False, "reason": "overlaps_existing_arena"}
    return {"admitted": True, "reason": "ok"}


def simulate_lmcache_devdax_and_apc() -> dict[str, Any]:
    """Simulate a 512 GiB CXL Device-DAX (`align = 2 MiB`) shared by multiple arenas
    and verify APC-covered prefix exclusion in `lookup.py`.
    """
    capacity = 512 * GIB
    align = 2 * MIB
    arenas: list[tuple[int, int]] = [(0, 256 * GIB)]

    probes = [
        ("second_256gib_disjoint", 256 * GIB, 256 * GIB),
        ("overlapping_range_128gib_to_384gib", 128 * GIB, 256 * GIB),
        ("unaligned_1mib_offset", 256 * GIB + 1 * MIB, 64 * GIB),
        ("past_capacity_384gib_plus_256gib", 384 * GIB, 256 * GIB),
    ]
    dax_results = {}
    for name, off, sz in probes:
        res = validate_devdax_arena_add(
            existing_arenas=arenas,
            device_capacity=capacity,
            device_align=align,
            offset=off,
            size=sz,
        )
        dax_results[name] = res

    # APC-covered prefix exclusion (`lookup.py` commit 168130df9c):
    # Suppose chunk_size = 256 tokens, prompt_tokens = 2,100 (8 full chunks = 2,048 tokens + 52 tail tokens).
    # Engine APC already holds 3 chunks (`start_chunk = 3`, `covered_tokens = 768`), plus 100 sub-chunk tokens
    # reported in `engine_computed_tokens = 868`.
    # Case A: LMCache finds 0 additional chunks (`found_count = 0, l1_found_count = 0`).
    # Legacy bug: `requested_tokens = 2048`, `l1_found_count += start_chunk (3)` -> reported `l1_hit_tokens = 768`
    # (`37.5%` fake L1 hit rate even though LMCache missed 100% of looked-up chunks!).
    # Patched (`168130df9c`): `looked_up_tokens = 2048 - 768 = 1280`, `l1_hit_tokens = 0` (`0.0%` L1 hit rate),
    # `apc_covered_tokens = 768`, `request_hit_tokens = max(868, 768 + 0) = 868`.
    # Case B: LMCache finds 3 chunks (`l1_found_count = 2, found_count = 3` -> `l1_chunks = 2, l2_chunks = 1`).
    # `hit_tokens = 768`, `request_hit_tokens = max(868, 768 + 768) = 1536`.
    chunk_size = 256
    requested_tokens = 8 * chunk_size  # 2048
    prompt_tokens = 2100
    start_chunk = 3
    engine_hit_tokens = 868
    covered_tokens = start_chunk * chunk_size  # 768
    looked_up_tokens = requested_tokens - covered_tokens  # 1280

    # Case A: LMCache miss (found=0)
    legacy_miss_l1_hit_rate = round((start_chunk * chunk_size) / requested_tokens, 4)
    patched_miss_l1_hit_rate = round(0 / looked_up_tokens, 4)
    patched_miss_request_hit_tokens = max(engine_hit_tokens, covered_tokens + 0)

    # Case B: LMCache hits 2 L1 chunks + 1 L2 chunk
    l1_chunks, l2_chunks = 2, 1
    hit_tokens = (l1_chunks + l2_chunks) * chunk_size  # 768
    patched_hit_l1_rate = round((l1_chunks * chunk_size) / looked_up_tokens, 4)  # 512 / 1280 = 0.4
    patched_hit_l2_rate = round((l2_chunks * chunk_size) / looked_up_tokens, 4)  # 256 / 1280 = 0.2
    patched_hit_request_hit_tokens = max(engine_hit_tokens, covered_tokens + hit_tokens)  # 1536
    naive_sum_overcount_tokens = (engine_hit_tokens + hit_tokens) - patched_hit_request_hit_tokens  # 1636 - 1536 = 100

    return {
        "dax_probes": dax_results,
        "apc_attribution": {
            "prompt_tokens": prompt_tokens,
            "requested_tokens_raw": requested_tokens,
            "apc_covered_tokens": covered_tokens,
            "looked_up_tokens": looked_up_tokens,
            "legacy_miss_fake_l1_hit_rate": legacy_miss_l1_hit_rate,
            "patched_miss_l1_hit_rate": patched_miss_l1_hit_rate,
            "patched_miss_request_hit_tokens": patched_miss_request_hit_tokens,
            "patched_hit_l1_rate": patched_hit_l1_rate,
            "patched_hit_l2_rate": patched_hit_l2_rate,
            "patched_hit_request_hit_tokens": patched_hit_request_hit_tokens,
            "subchunk_overlap_prevented_tokens": naive_sum_overcount_tokens,
        },
    }


def simulate_dynamo_relay_and_migration() -> dict[str, Any]:
    """Simulate Dynamo `DcCkfRankReplacement` (`cuckoo/dc.rs` commit `909ebee0c4`)
    and `RetryManager::rebase_usage` (`migration.rs` commit `cc02d8b1c2`).
    """
    # 1. `DcCkfRankReplacement` replay of mixed snapshot + buffered tail:
    # Events: Stored([11, 12, 13]), Removed([12, 13, 99]), Stored([14] with parent 11)
    events = [
        ("Stored", [(11, 1011), (12, 1012), (13, 1013)]),
        ("Removed", [12, 13, 99]),
        ("Stored", [(14, 1014)]),
    ]
    source_lineage: dict[int, int] = {}
    canonical_candidates: list[int] = []
    seen_candidates: set[int] = set()

    for kind, payload in events:
        if kind == "Stored":
            for ext, can in payload:
                source_lineage[ext] = can
                if can not in seen_candidates:
                    seen_candidates.add(can)
                    canonical_candidates.append(can)
        elif kind == "Removed":
            for ext in payload:
                source_lineage.pop(ext, None)
        elif kind == "Cleared":
            source_lineage.clear()
            canonical_candidates.clear()
            seen_candidates.clear()

    owner_increments: dict[int, int] = {}
    for can in source_lineage.values():
        owner_increments[can] = owner_increments.get(can, 0) + 1

    unfiltered_candidates = list(canonical_candidates)  # [1011, 1012, 1013, 1014]
    surviving_candidates = [c for c in canonical_candidates if c in owner_increments]  # [1011, 1014]

    # 2. `RetryManager::rebase_usage` across 2 migrations (`test_retry_manager_rebases_usage_across_two_migrations`)
    # Client prompt = 3 tokens, target max_tokens = 6, fail_after = 2 tokens on attempts 0 and 1.
    # Attempt 0 sends 3 prompt tokens, delivers 2 tokens, disconnects.
    # Attempt 1 sends 3 + 2 = 5 prompt tokens, delivers 2 tokens, disconnects.
    # Attempt 2 sends 3 + 4 = 7 prompt tokens (`replayed_tokens = 4`), generates remaining 2 tokens,
    # and worker reports: prompt_tokens = 7, completion_tokens = 2, total_tokens = 9, cached_tokens = 7, reasoning_tokens = 2.
    raw_worker_usage = {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
        "cached_tokens": 7,
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    replayed_tokens = 4
    rebased_prompt = raw_worker_usage["prompt_tokens"] - replayed_tokens  # 3
    rebased_completion = raw_worker_usage["completion_tokens"] + replayed_tokens  # 6
    rebased_cached = min(raw_worker_usage["cached_tokens"], rebased_prompt)  # 3
    rebased_usage = {
        "prompt_tokens": rebased_prompt,
        "completion_tokens": rebased_completion,
        "total_tokens": rebased_prompt + rebased_completion,
        "cached_tokens": rebased_cached,
        "completion_tokens_details": None,
    }

    return {
        "cuckoo_dc_replacement": {
            "surviving_external_hashes": sorted(source_lineage.keys()),
            "unfiltered_canonical_candidates": unfiltered_candidates,
            "surviving_canonical_candidates": surviving_candidates,
            "zombie_candidates_pruned": len(unfiltered_candidates) - len(surviving_candidates),
        },
        "migration_usage_rebase": {
            "replayed_tokens": replayed_tokens,
            "raw_worker_usage": raw_worker_usage,
            "rebased_client_usage": rebased_usage,
        },
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "lmcache_devdax_apc_dynamo_relay_sim",
        "source_commits": [
            "LMCache/LMCache@e8dc1a1daeb918aa5a1797d2ecc2c6b2cdeab1a3",
            "LMCache/LMCache@168130df9c8fafa376b650814598179c212d3e66",
            "ai-dynamo/dynamo@909ebee0c4b63988bea6ab474a46913073e4a2dc",
            "ai-dynamo/dynamo@cc02d8b1c2b60a9cbe291b5264456897c64d7e7c",
        ],
        "lmcache": simulate_lmcache_devdax_and_apc(),
        "dynamo": simulate_dynamo_relay_and_migration(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify against committed summary.json")
    args = parser.parse_args()

    summary = build_summary()
    probes = summary["lmcache"]["dax_probes"]
    assert probes["second_256gib_disjoint"]["admitted"] is True
    assert probes["overlapping_range_128gib_to_384gib"]["admitted"] is False
    assert probes["unaligned_1mib_offset"]["admitted"] is False
    assert probes["past_capacity_384gib_plus_256gib"]["admitted"] is False
    apc = summary["lmcache"]["apc_attribution"]
    assert apc["legacy_miss_fake_l1_hit_rate"] == 0.375
    assert apc["patched_miss_l1_hit_rate"] == 0.0
    assert apc["patched_hit_request_hit_tokens"] == 1536
    assert apc["subchunk_overlap_prevented_tokens"] == 100
    dyn = summary["dynamo"]
    assert dyn["cuckoo_dc_replacement"]["surviving_canonical_candidates"] == [1011, 1014]
    assert dyn["cuckoo_dc_replacement"]["zombie_candidates_pruned"] == 2
    assert dyn["migration_usage_rebase"]["rebased_client_usage"]["prompt_tokens"] == 3
    assert dyn["migration_usage_rebase"]["rebased_client_usage"]["completion_tokens"] == 6
    assert dyn["migration_usage_rebase"]["rebased_client_usage"]["cached_tokens"] == 3

    if args.check:
        if not SUMMARY_PATH.exists():
            print(f"Missing {SUMMARY_PATH}", file=sys.stderr)
            return 1
        committed = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        if committed != summary:
            print("Mismatch between computed and committed summary.json", file=sys.stderr)
            return 1
        print("lmcache_devdax_apc_dynamo_relay_sim: OK")
        return 0

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
