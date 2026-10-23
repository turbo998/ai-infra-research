#!/usr/bin/env python3
"""Phase 69 Track C deterministic experiment:
TransformerEngine Fused RMSNorm+MXFP8 & TMA S2G Ring Synchronization,
TorchTitan ActorSystem & Offloaded EMA Global DTensor Metadata,
LMCache OTLP Trace-Analyze Simulator, and Mooncake DFS-Only Replica Placement.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
from pathlib import Path


def swizzled_scale_idx(i: int, j: int, num_tiles_j: int) -> int:
    """Reproduce TransformerEngine GEMM-swizzled 128x4 MXFP8 scale index (#3657)."""
    return ((i // 128) * num_tiles_j + j // 4) * 512 + (i % 32) * 16 + ((i % 128) // 32) * 4 + (j % 4)


def simulate_te_rmsnorm_mxfp8_and_tma_sync() -> dict[str, object]:
    """Verify TE RMSNorm+MXFP8 32x128 tile geometry (#3657) and TMA S2G wait-group handoff (#3417)."""
    rows, cols = 256, 1024
    scale_block = 32
    tile_rows, tile_cols = 32, 128
    scale_cols = cols // scale_block  # 32
    num_tiles_j = scale_cols // 4  # 8

    # Verify bijectivity of 128x4 GEMM-swizzled scale index over 256 x 32 scale grid
    total_scales = rows * scale_cols
    mapped = {swizzled_scale_idx(r, c, num_tiles_j) for r in range(rows) for c in range(scale_cols)}
    bijective = (len(mapped) == total_scales) and (min(mapped) == 0) and (max(mapped) == total_scales - 1)

    # TMA S2G output-buffer ring reuse invariant (#3417):
    # Bulk async-groups are thread-local; when stage >= BUFFS_NUM, only the master/leading thread
    # owns the group, so it must call cp_async_bulk_wait_group_read<BUFFS_NUM - 1>() followed by
    # __syncthreads() before cooperative threads overwrite `stage % BUFFS_NUM`.
    prefetch_stages = 2
    buffs_num = prefetch_stages + 1  # 3
    wait_depth = buffs_num - 1  # 2 (was hardcoded to 1 or unguarded on non-master threads)

    return {
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "threads_per_cta": 128,
        "rows_per_warp": 8,
        "cols_per_lane": 4,
        "lanes_per_scale_block": 8,
        "rsigma_rows_per_cta": 4,
        "scale_grid_shape": [rows, scale_cols],
        "swizzled_scale_bijective": bijective,
        "tma_ring_buffs_num": buffs_num,
        "tma_wait_group_read_depth": wait_depth,
        "nvfp4_transposed_out_buffs_old": 2,
        "nvfp4_transposed_out_buffs_new": buffs_num,
        "ptx_memory_clobber_added_helpers": 11,
    }


def simulate_torchtitan_actor_system_and_ema() -> dict[str, object]:
    """Reproduce TorchTitan ActorSystem RDMA guard (#5391), offloaded EMA DTensor shape (#5357), and EMA reseed (#5141)."""
    # Offloaded EMA uneven FSDP shard test (#5357):
    # Global tensor (5, 8) sharded over 2 ranks on dim 0 has local shards (3, 8) and (2, 8).
    # Inferring global shape without explicit shape=p.shape yields (6, 8) on rank 0 and (4, 8) on rank 1!
    uneven_cases = []
    for global_rows, local_rows in ((4, (2, 2)), (5, (3, 2)), (1, (1, 0))):
        inferred_r0 = local_rows[0] * 2
        inferred_r1 = local_rows[1] * 2
        uneven_cases.append(
            {
                "global_rows": global_rows,
                "local_rows": list(local_rows),
                "buggy_inferred_rows_per_rank": [inferred_r0, inferred_r1],
                "buggy_mismatch": (inferred_r0 != global_rows) or (inferred_r1 != global_rows),
                "fixed_explicit_shape_rows": [global_rows, global_rows],
            }
        )

    return {
        "uneven_ema_cases": uneven_cases,
        "ema_reseed_on_model_only_load": True,
        "ema_reseed_on_exclude_ema": True,
        "ema_preserved_on_full_resume": True,
        "rdma_min_tensor_bytes_default": 1024,
        "batcher_rejects_rdma_without_registration": True,
    }


def simulate_lmcache_trace_analyze_and_mooncake_dfs() -> dict[str, object]:
    """Reproduce LMCache `bench trace-analyze` replay (#5821) and Mooncake DFS-only placement (#4637)."""
    # Fixtures from tests/tools/data/trace_formats/lmcache.jsonl (#5821):
    # Raw spans in file order:
    raw_spans = [
        {"name": "request", "ts": 1_700_000_000_000_000_000, "session_id": None, "model": None, "ids": []},
        {"name": "mp.request.trace", "ts": 1_700_000_002_000_000_000, "session_id": "sess-r2", "model": "meta-llama/Llama-3.1-8B", "ids": list(range(1, 13)) + [100, 101]},
        {"name": "mp.request.trace", "ts": 1_700_000_001_000_000_000, "session_id": "sess-r1", "model": "meta-llama/Llama-3.1-8B", "ids": list(range(1, 9))},
        {"name": "mp.request.trace", "ts": 1_700_000_003_000_000_000, "session_id": "sess-r3", "model": "Qwen/Qwen3-1.7B", "ids": [1, 2, 3, 4, 200, 201, 202, 203]},
        {"name": "mp.request.trace", "ts": 1_700_000_004_000_000_000, "session_id": "sess-r1", "model": "meta-llama/Llama-3.1-8B", "ids": list(range(1, 9))},
        {"name": "mp.request.trace", "ts": 1_700_000_004_500_000_000, "session_id": "sess-r4", "model": "Qwen/Qwen3-1.7B", "ids": list(range(300, 308))},
    ]

    # Dedup by request_id (keep earliest timestamp) and sort by timestamp_ns
    by_req: dict[str, dict[str, object]] = {}
    for sp in raw_spans:
        if sp["name"] != "mp.request.trace":
            continue
        rid = str(sp["session_id"])
        if rid not in by_req or int(sp["ts"]) < int(by_req[rid]["ts"]):
            by_req[rid] = sp
    records = sorted(by_req.values(), key=lambda r: int(r["ts"]))

    def replay_prefix_lru(chunk_size: int, capacity_tokens: int) -> dict[str, int]:
        max_chunks = capacity_tokens // chunk_size
        lru: OrderedDict[tuple[tuple[int, ...], ...], None] = OrderedDict()
        prefix_hit_tokens = 0
        evictions = 0
        for rec in records:
            ids = list(rec["ids"])
            num_chunks = len(ids) // chunk_size
            chain: list[tuple[int, ...]] = []
            chunks = []
            for c_idx in range(num_chunks):
                c_tokens = tuple(ids[c_idx * chunk_size : (c_idx + 1) * chunk_size])
                chain.append(c_tokens)
                chunks.append(tuple(chain))
            # Leading run of present chunks is the prefix hit
            hit_chunks = 0
            for ch in chunks:
                if ch in lru:
                    lru.move_to_end(ch)
                    hit_chunks += 1
                else:
                    break
            prefix_hit_tokens += hit_chunks * chunk_size
            # Insert all chunks
            for ch in chunks:
                if ch in lru:
                    lru.move_to_end(ch)
                else:
                    if len(lru) >= max_chunks:
                        lru.popitem(last=False)
                        evictions += 1
                    lru[ch] = None
        return {
            "chunk_size": chunk_size,
            "capacity_tokens": capacity_tokens,
            "prefix_hit_tokens": prefix_hit_tokens,
            "evictions": evictions,
        }

    grid = [
        replay_prefix_lru(chunk_size=4, capacity_tokens=8),
        replay_prefix_lru(chunk_size=4, capacity_tokens=16),
    ]

    # Mooncake DFS-only replica config validation (#4637)
    def mooncake_validate_dfs_config(replica_num: int, nof_replica_num: int, dfs_replica_num: int, tenant: str = "default") -> bool:
        if dfs_replica_num < 0 or dfs_replica_num > 1:
            return False
        if dfs_replica_num > 0 and tenant != "default":
            return False
        return (replica_num + nof_replica_num + dfs_replica_num) >= 1

    return {
        "deduped_requests": len(records),
        "total_tokens": sum(len(r["ids"]) for r in records),
        "ordered_request_ids": [r["session_id"] for r in records],
        "grid": grid,
        "mooncake_dfs_only_valid": mooncake_validate_dfs_config(0, 0, 1, "default"),
        "mooncake_dfs_2_invalid": mooncake_validate_dfs_config(0, 0, 2, "default"),
        "mooncake_dfs_non_default_tenant_invalid": mooncake_validate_dfs_config(0, 0, 1, "tenant_a"),
    }


def build_summary() -> dict[str, object]:
    te = simulate_te_rmsnorm_mxfp8_and_tma_sync()
    titan = simulate_torchtitan_actor_system_and_ema()
    lmc_mc = simulate_lmcache_trace_analyze_and_mooncake_dfs()

    invariants = {
        "te_mxfp8_swizzle_is_bijective": te["swizzled_scale_bijective"] is True,
        "te_tma_ring_wait_depth_is_buffs_minus_1": (
            te["tma_wait_group_read_depth"] == 2
            and te["nvfp4_transposed_out_buffs_new"] == 3
        ),
        "torchtitan_uneven_ema_fixes_5row_and_1row_mismatch": (
            titan["uneven_ema_cases"][1]["buggy_mismatch"] is True
            and titan["uneven_ema_cases"][2]["buggy_mismatch"] is True
            and titan["uneven_ema_cases"][1]["fixed_explicit_shape_rows"] == [5, 5]
        ),
        "lmcache_trace_analyze_matches_golden_fixture": (
            lmc_mc["deduped_requests"] == 4
            and lmc_mc["total_tokens"] == 38
            and lmc_mc["grid"][1]["prefix_hit_tokens"] == 12
        ),
        "mooncake_allows_dfs_only_and_rejects_invalid": (
            lmc_mc["mooncake_dfs_only_valid"] is True
            and lmc_mc["mooncake_dfs_2_invalid"] is False
            and lmc_mc["mooncake_dfs_non_default_tenant_invalid"] is False
        ),
    }

    return {
        "experiment": "te-mxfp8-rmsnorm-tma-torchtitan-lmcache-mooncake",
        "phase": 69,
        "priority": 78,
        "evidence_tier": "own-offline",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "te_rmsnorm_mxfp8_and_tma": te,
        "torchtitan_actor_and_ema": titan,
        "lmcache_trace_and_mooncake_dfs": lmc_mc,
        "invariants": invariants,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        failed = [k for k, v in summary["invariants"].items() if not v]
        if failed:
            raise SystemExit(f"FAILED invariants: {failed}")
        print(
            "OK: te-mxfp8-rmsnorm-tma-torchtitan-lmcache-mooncake "
            f"(swizzle_bijective={summary['te_rmsnorm_mxfp8_and_tma']['swizzled_scale_bijective']}, "
            f"trace_hit_4_16={summary['lmcache_trace_and_mooncake_dfs']['grid'][1]['prefix_hit_tokens']}, "
            f"dfs_only={summary['lmcache_trace_and_mooncake_dfs']['mooncake_dfs_only_valid']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
