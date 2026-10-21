from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
SUMMARY_PATHS = [ROOT / "summary.json", ROOT / "results" / "summary.json"]

LATENT = 512
ROPE = 64
QK_DIM = LATENT + ROPE
V_DIM = LATENT
SF_VEC = 16
CKV_BYTES = LATENT // 2
SF_BYTES = LATENT // SF_VEC
PAGE = 64
ROWS_TILE = 256
KEY_TILE = 2 * PAGE
CLUSTER = 4
ITEM_WORDS = 6
ROWS_GRANULE = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
ALIGN = 256
MAX_SPLITS = 64
MLA2_ITEM_COST = 7
MERGE_BATCH_COST = 1.5
UNITS_PAIRS_BY_SM_COUNT: dict[int, tuple[int, int]] = {208: (50, 4), 212: (46, 14)}

AUTHOR_BENCHMARKS = [
    {"config": "H128-Q1-B16-KV16K", "cake_us": 59.1, "ts2_us": 58.6, "speedup": 0.992},
    {"config": "H128-Q1-B16-KV32K", "cake_us": 115.6, "ts2_us": 115.6, "speedup": 1.000},
    {"config": "H128-Q1-B16-KV64K", "cake_us": 227.5, "ts2_us": 229.0, "speedup": 1.007},
    {"config": "H128-Q1-B16-KV128K", "cake_us": 449.9, "ts2_us": 456.6, "speedup": 1.015},
    {"config": "H128-Q1-B64-KV64K", "cake_us": 850.5, "ts2_us": 841.3, "speedup": 0.989},
    {"config": "H128-Q1-B64-KV128K", "cake_us": 1705.2, "ts2_us": 1689.1, "speedup": 0.991},
    {"config": "H128-Q2-B16-KV64K", "cake_us": 236.2, "ts2_us": 235.4, "speedup": 0.997},
    {"config": "H128-Q2-B64-KV64K", "cake_us": 886.9, "ts2_us": 891.5, "speedup": 1.005},
    {"config": "H128-Q4-B16-KV16K", "cake_us": 74.0, "ts2_us": 81.9, "speedup": 1.107},
    {"config": "H128-Q4-B16-KV64K", "cake_us": 261.2, "ts2_us": 286.6, "speedup": 1.097},
    {"config": "H128-Q4-B16-KV128K", "cake_us": 511.2, "ts2_us": 556.0, "speedup": 1.088},
    {"config": "H128-Q4-B64-KV16K", "cake_us": 270.0, "ts2_us": 292.9, "speedup": 1.085},
    {"config": "H128-Q4-B64-KV64K", "cake_us": 1003.4, "ts2_us": 1084.6, "speedup": 1.081},
    {"config": "H128-Q4-B64-KV128K", "cake_us": 1995.1, "ts2_us": 2158.9, "speedup": 1.082},
    {"config": "H96-Q4-B64-KV64K", "cake_us": 661.2, "ts2_us": 818.9, "speedup": 1.239},
    {"config": "H96-Q4-B64-KV128K", "cake_us": 1300.8, "ts2_us": 1623.8, "speedup": 1.248},
]


def round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def mla2_plan(
    kv_lens: Sequence[int],
    q_lens: Sequence[int],
    heads: int,
    units: int,
    pairs: int = 0,
    item_cost: float = MLA2_ITEM_COST,
    merge_rows_per_tile: float = 3900,
    slack: float = 0.03,
    pair_cost: float = 1.4,
    split: int | tuple | int = 0,
) -> tuple[list[tuple], list[tuple], int, int]:
    tile = 2 * PAGE
    tiles_of = [max(1, -(-kv // tile)) for kv in kv_lens]
    tiles = []
    for b, ql in enumerate(q_lens):
        rows = ql * heads
        for m in range(-(-rows // ROWS_TILE)):
            tiles.append((b, m, rows - m * ROWS_TILE <= ROWS_TILE // 2))
    full_tiles = [t for t in tiles if not t[2]]
    half_tiles = [t for t in tiles if t[2]]
    rows_full = sum(
        min(ROWS_TILE, q_lens[b] * heads - m * ROWS_TILE) for b, m, _ in full_tiles
    )
    rows_half = sum(
        min(ROWS_TILE, q_lens[b] * heads - m * ROWS_TILE) for b, m, _ in half_tiles
    )
    k_max = max(tiles_of, default=0)
    couples = pairs // 2 if pairs > 1 else 0

    def makespan(costs: list[tuple[float, bool]]) -> float:
        free = [(0.0, 0, r) for r in range(units)] + [
            (0.0, 1, r) for r in range(couples)
        ]
        heapq.heapify(free)
        span = 0.0
        for c, dbl in costs:
            t, kind, r = heapq.heappop(free)
            t += c if (kind == 0 or dbl) else pair_cost * c
            span = max(span, t)
            heapq.heappush(free, (t, kind, r))
        return span

    def full_ranges(s_f: int, scale: float) -> tuple[list[list[tuple[int, int, bool]]], list[tuple[int, int]]]:
        n_tiles = len(full_tiles)
        n_p = (
            min(couples, max(0, n_tiles * s_f - units))
            if scale != 1.0 and n_tiles
            else 0
        )
        per_tile = [
            n_p // n_tiles + (1 if t < n_p % n_tiles else 0) for t in range(n_tiles)
        ]
        ranges = []
        for t, (b, _m, _) in enumerate(full_tiles):
            k = tiles_of[b]
            p_t = min(per_tile[t], s_f)
            q_t = s_f - p_t
            if p_t:
                long = k / (q_t + p_t * scale)
                lens = [long] * q_t + [long * scale] * p_t
            else:
                lens = [k / s_f] * s_f
            base = [int(x) for x in lens]
            for i in sorted(range(s_f), key=lambda i: -(lens[i] - base[i]))[
                : k - sum(base)
            ]:
                base[i] += 1
            lo, rs = 0, []
            for i, n in enumerate(base):
                rs.append((lo, n, i >= q_t))
                lo += n
            ranges.append(rs)
        q_list = [
            (t, i) for t in range(n_tiles) for i, r in enumerate(ranges[t]) if not r[2]
        ]
        p_list = [
            (t, i) for t in range(n_tiles) for i, r in enumerate(ranges[t]) if r[2]
        ]
        return ranges, q_list[:units] + p_list + q_list[units:]

    def mixed_ranges(s_t: int) -> tuple[list[list[tuple[int, int, bool]]], list[tuple[int, int]]]:
        ranges, order = [], []
        for t, (b, _m, _) in enumerate(full_tiles):
            k = tiles_of[b]
            if t < units:
                ranges.append([(0, k, False)])
                continue
            base = [k // s_t + (1 if i < k % s_t else 0) for i in range(s_t)]
            lo, rs = 0, []
            for n in base:
                rs.append((lo, n, False))
                lo += n
            ranges.append(rs)
        order = [(t, 0) for t in range(min(units, len(full_tiles)))]
        order += [
            (t, i) for t in range(units, len(full_tiles)) for i in range(len(ranges[t]))
        ]
        return ranges, order

    def cost(s_f: int, s_h: int, scale: float, s_t: int = 0) -> float:
        ranges, order = full_ranges(s_f, scale) if not s_t else mixed_ranges(s_t)
        tps_h = -(-k_max // s_h) if half_tiles else 0
        costs = [(ranges[t][i][1] + item_cost, False) for t, i in order]
        costs += [(0.6 * tps_h + item_cost, True)] * (len(half_tiles) * s_h)
        merge = 0.0
        if full_tiles and s_t:
            rows_tail = sum(
                min(ROWS_TILE, q_lens[b] * heads - m * ROWS_TILE)
                for b, m, _ in full_tiles[units:]
            )
            merge += (2 * s_t + 1) * rows_tail / merge_rows_per_tile
        elif full_tiles and s_f > 1:
            merge += (2 * s_f + 1) * rows_full / merge_rows_per_tile
        if half_tiles:
            merge += (4 * s_h + 1) * rows_half / merge_rows_per_tile
        if max(s_t or (s_f if full_tiles else 0), 2 * s_h if half_tiles else 0) > 32:
            merge += MERGE_BATCH_COST
        return makespan(costs) + merge

    scales = (1.0, 1.0 / pair_cost, pair_cost)
    s_t = 0
    if isinstance(split, tuple) and split and split[0] == "mixed":
        s_f, s_h, s_t, scale = 1, 1, int(split[1]), 1.0
    elif split:
        s_f, s_h = (split, split) if isinstance(split, int) else tuple(split)
        scale = min(scales, key=lambda sc: (cost(s_f, s_h, sc), sc != 1.0))
    else:
        cap = min(max(k_max, 1), 64)
        cands = set()
        for s_f in range(1, cap + 1):
            if not half_tiles:
                cands.add((s_f, 1))
                continue
            for f in (0.5, 0.6, 0.7, 0.85, 1.0, 1.2):
                s_h = max(1, round(s_f * f))
                if s_h <= cap and 2 * s_h <= MAX_SPLITS:
                    cands.add((s_f if full_tiles else 1, s_h))
            if s_f <= 2:
                cands |= {(s_f if full_tiles else 1, 1), (s_f if full_tiles else 1, 2)}
        if half_tiles and len(full_tiles) > units + couples:
            cands = {c for c in cands if c[0] >= 2}
        costs = {(c, sc): cost(*c, sc) for c in cands for sc in scales}
        best = min(costs.values())
        (s_f, s_h), scale = min(
            (k for k, v in costs.items() if v <= (1 + slack) * best),
            key=lambda k: (k[1] != 1.0, k[0][0] + k[0][1], k[0]),
        )
        if scale == 1.0:
            alt = min(
                (sc for sc in scales if sc != 1.0),
                key=lambda sc: costs[((s_f, s_h), sc)],
            )
            if costs[((s_f, s_h), alt)] < 0.99 * costs[((s_f, s_h), 1.0)]:
                scale = alt
        if len(full_tiles) > units:
            s_h_m = s_h if half_tiles else 1
            mixed = {st: cost(1, s_h_m, 1.0, st) for st in (2, 3, 4, 6, 8) if st <= cap}
            st_best = min(mixed, key=lambda st: (mixed[st], st))
            if mixed[st_best] < costs[((s_f, s_h), scale)] / (1 + slack):
                s_f, s_h, s_t, scale = 1, s_h_m, st_best, 1.0
    ranges, order = full_ranges(s_f, scale) if not s_t else mixed_ranges(s_t)
    tps_h = -(-k_max // s_h)
    n_f = (s_t or s_f) if full_tiles else 0
    n_h = 2 * s_h if half_tiles else 0
    items, table = [], []
    for t, i in order:
        b, m, _ = full_tiles[t]
        pages = -(-kv_lens[b] // PAGE)
        lo, n, _ = ranges[t][i]
        s_tile = len(ranges[t])
        items.append((b, m, 2 * lo, min(2 * (lo + n), pages), i, s_tile, 1 if s_tile == 1 else 0, 0))
        table.append((b, m, lo, n, i, s_tile))
    for b, m, _ in half_tiles:
        pages = -(-kv_lens[b] // PAGE)
        k_tiles = tiles_of[b]
        for z in range(s_h):
            lo, hi = min(z * tps_h, k_tiles), min(k_tiles, (z + 1) * tps_h)
            mid = lo + (hi - lo) // 2
            items.append((b, m, 2 * lo, min(2 * mid, pages), 2 * z, 2 * s_h, 0, 0))
            items.append((b, m, 2 * mid, min(2 * hi, pages), 2 * z + 1, 2 * s_h, 0, 0))
            table.append((b, m, lo, hi - lo, z, 2 * s_h))
    return items, table, max(n_f, n_h), len(order)


def visible_keys(bound: int, kv_len: int, world: int, rank: int, interleave: int) -> int:
    bnd = max(int(bound), -1)
    blocks_to = (bnd + 1) // interleave - rank - 1
    n_full = blocks_to // world + 1 if blocks_to >= 0 else 0
    g0 = (n_full * world + rank) * interleave
    partial = min(max(bnd - g0 + 1, 0), interleave)
    return min(n_full * interleave + partial, int(kv_len))


def compute_workspace_bytes(rows: int, max_splits: int, items: int, batch: int) -> int:
    plan_words = items * ITEM_WORDS + batch + 1 + rows
    specs = (
        rows * max_splits * V_DIM * 2,
        rows * max_splits * 4,
        rows * 4,
        plan_words * 4,
    )
    offset = 0
    for nbytes in specs:
        offset += round_up(nbytes, ALIGN)
    return round_up(offset, ALIGN)


def select_merge_record(rows: int, max_splits: int) -> str | None:
    if max_splits <= 1:
        return None
    if rows <= 1024:
        return "merge_w4"
    if rows <= 2048:
        return "merge_w2"
    return "merge_w1"


def simulate_rubin_mla_nvfp4_decode_plans() -> dict[str, object]:
    units, pairs = UNITS_PAIRS_BY_SM_COUNT[208]  # R200 (sm_107a): 50 fours, 4 pairs

    # Case 1: H96-Q4-B64-KV128K (384 rows/req -> 1 full tile [256 rows] + 1 half tile [128 rows])
    kv_h96 = [131072] * 64
    q_h96 = [4] * 64
    _items_h96, table_h96, max_splits_h96, n_full_h96 = mla2_plan(
        kv_h96, q_h96, 96, units, pairs
    )
    rows_h96 = sum(q_h96) * 96  # 24576 rows
    decode_record_h96 = "decode_full" if n_full_h96 == len(table_h96) else "decode"
    merge_record_h96 = select_merge_record(rows_h96, max_splits_h96)
    ws_bytes_h96 = compute_workspace_bytes(rows_h96, max_splits_h96, len(table_h96), 64)

    # Case 2: H128-Q2-B16-KV64K (256 rows/req -> 1 full tile [256 rows], no half tiles)
    kv_h128 = [65536] * 16
    q_h128 = [2] * 16
    _items_h128, table_h128, max_splits_h128, n_full_h128 = mla2_plan(
        kv_h128, q_h128, 128, units, pairs
    )
    rows_h128 = sum(q_h128) * 128  # 4096 rows
    decode_record_h128 = "decode_full" if n_full_h128 == len(table_h128) else "decode"
    merge_record_h128 = select_merge_record(rows_h128, max_splits_h128)

    # Case 3: DCP visible_keys partitioning verification (W=4, I=64, global_kv=1024, q_len=4)
    # Sum of visible keys across all 4 DCP ranks for each query token t must equal global causal bound + 1
    dcp_world = 4
    dcp_interleave = 64
    global_kv = 1024
    local_kv = global_kv // dcp_world
    dcp_token_sums = []
    for t in range(4):
        bound = global_kv - 4 + t
        rank_vis = [
            visible_keys(bound, local_kv, dcp_world, r, dcp_interleave)
            for r in range(dcp_world)
        ]
        dcp_token_sums.append({"t": t, "bound": bound, "per_rank": rank_vis, "sum": sum(rank_vis)})

    # KV cache per-token footprint comparison
    nvfp4_kv_bytes_per_token = CKV_BYTES + SF_BYTES + ROPE  # 256 + 32 + 64 = 352 B
    fp8_mla_kv_bytes_per_token = 512 + 64 * 2  # 640 B (or 576 B all-FP8)
    bf16_mla_kv_bytes_per_token = 576 * 2  # 1152 B
    query_quantized_bytes_per_row = CKV_BYTES + SF_BYTES + ROPE + 4  # 356 B

    q4_speedups = [b["speedup"] for b in AUTHOR_BENCHMARKS if "-Q4-" in b["config"]]
    all_speedups = [b["speedup"] for b in AUTHOR_BENCHMARKS]
    geomean_all = 1.0
    for s in all_speedups:
        geomean_all *= s
    geomean_all = round(geomean_all ** (1.0 / len(all_speedups)), 3)

    geomean_q4 = 1.0
    for s in q4_speedups:
        geomean_q4 *= s
    geomean_q4 = round(geomean_q4 ** (1.0 / len(q4_speedups)), 3)

    return {
        "r200_sm208_units_pairs": [units, pairs],
        "r200_sm212_units_pairs": list(UNITS_PAIRS_BY_SM_COUNT[212]),
        "nvfp4_kv_bytes_per_token": nvfp4_kv_bytes_per_token,
        "bf16_mla_kv_bytes_per_token": bf16_mla_kv_bytes_per_token,
        "kv_memory_reduction_vs_bf16_pct": round(
            (1.0 - nvfp4_kv_bytes_per_token / bf16_mla_kv_bytes_per_token) * 100.0, 2
        ),
        "fp8_mla_kv_bytes_per_token": fp8_mla_kv_bytes_per_token,
        "query_quantized_bytes_per_row": query_quantized_bytes_per_row,
        "h96_q4_b64_kv128k_plan": {
            "rows_total": rows_h96,
            "items": len(table_h96),
            "n_full": n_full_h96,
            "max_splits": max_splits_h96,
            "decode_record": decode_record_h96,
            "merge_record": merge_record_h96,
            "grid_decode": [CLUSTER * len(table_h96), 1, 1],
            "workspace_bytes": ws_bytes_h96,
        },
        "h128_q2_b16_kv64k_plan": {
            "rows_total": rows_h128,
            "items": len(table_h128),
            "n_full": n_full_h128,
            "max_splits": max_splits_h128,
            "decode_record": decode_record_h128,
            "merge_record": merge_record_h128,
            "grid_decode": [CLUSTER * len(table_h128), 1, 1],
        },
        "dcp_visible_keys_check": dcp_token_sums,
        "geomean_speedup_all_16": geomean_all,
        "geomean_speedup_q4_8": geomean_q4,
        "peak_speedup_h96_q4_b64_kv128k": 1.248,
    }


def simulate_mnnvl_collective_handle_selection() -> dict[str, object]:
    # Mirrors choose_mnnvl_handle_type and allocate_mnnvl_handles in flashinfer/comm/mnnvl.py (#6734, a48bf8af3c)
    def run_allocation_scenario(
        ranks: list[dict[str, object]],
        fabric_create_or_import_succeeds: list[bool],
        posix_create_or_import_succeeds: list[bool] | None = None,
    ) -> dict[str, object]:
        hosts = {r["host"] for r in ranks}
        single_host = len(hosts) == 1
        if single_host:
            if all(r["reason"] is None for r in ranks):
                chosen = "FABRIC"
            else:
                chosen = "POSIX_FD"
        else:
            problems = []
            for idx, r in enumerate(ranks):
                if r["reason"] is not None:
                    problems.append(f"rank {idx} on {r['host']}: {r['reason']}")
                elif not r["cluster_uuid"] or all(b == 0 for b in r["cluster_uuid"]):
                    problems.append(
                        f"rank {idx} on {r['host']}: the GPU is not in a completed NVLink fabric cluster (NVML clusterUuid unset)"
                    )
            if not problems and len({r["cluster_uuid"] for r in ranks}) > 1:
                problems.append("the ranks are in different NVLink domains")
            if problems:
                return {
                    "single_host": False,
                    "initial_choice": None,
                    "final_handle_type": None,
                    "raised": True,
                    "problems": problems,
                    "cft_enabled": False,
                }
            chosen = "FABRIC"

        if chosen == "FABRIC":
            if all(fabric_create_or_import_succeeds):
                return {
                    "single_host": single_host,
                    "initial_choice": "FABRIC",
                    "final_handle_type": "FABRIC",
                    "fallback_to_posix": False,
                    "raised": False,
                    "cft_enabled": True,
                }
            if single_host:
                posix_ok = (
                    all(posix_create_or_import_succeeds)
                    if posix_create_or_import_succeeds is not None
                    else True
                )
                if posix_ok:
                    return {
                        "single_host": True,
                        "initial_choice": "FABRIC",
                        "final_handle_type": "POSIX_FD",
                        "fallback_to_posix": True,
                        "raised": False,
                        "cft_enabled": False,
                    }
            return {
                "single_host": single_host,
                "initial_choice": "FABRIC",
                "final_handle_type": None,
                "fallback_to_posix": False,
                "raised": True,
                "cft_enabled": False,
            }

        return {
            "single_host": True,
            "initial_choice": "POSIX_FD",
            "final_handle_type": "POSIX_FD",
            "fallback_to_posix": False,
            "raised": False,
            "cft_enabled": False,
        }

    # 1. Single-host 8xB200 with zero NVML clusterUuid but working IMEX -> FABRIC succeeds
    b200_single_host_imex = run_allocation_scenario(
        [{"host": "node0", "reason": None, "cluster_uuid": b"\x00" * 16} for _ in range(8)],
        [True] * 8,
    )
    # 2. Single-host 8xB200 with driver 595 reporting FABRIC=1 but no IMEX channel -> collective fallback to POSIX_FD
    b200_single_host_no_imex = run_allocation_scenario(
        [{"host": "node0", "reason": None, "cluster_uuid": b"\x00" * 16} for _ in range(8)],
        [False] * 8,
        [True] * 8,
    )
    # 3. Multi-host GB200 NVL72 with matching non-zero clusterUuid -> FABRIC succeeds
    gb200_multihost_ok = run_allocation_scenario(
        [
            {"host": f"node{i // 4}", "reason": None, "cluster_uuid": b"\xaa" * 16}
            for i in range(8)
        ],
        [True] * 8,
    )
    # 4. Multi-host with mismatched NVLink domains -> immediate collective error with per-rank UUIDs
    multihost_domain_mismatch = run_allocation_scenario(
        [
            {
                "host": f"node{i // 4}",
                "reason": None,
                "cluster_uuid": (b"\xaa" * 16 if i < 4 else b"\xbb" * 16),
            }
            for i in range(8)
        ],
        [True] * 8,
    )

    return {
        "b200_single_host_imex": b200_single_host_imex,
        "b200_single_host_no_imex_fallback": b200_single_host_no_imex,
        "gb200_multihost_ok": gb200_multihost_ok,
        "multihost_domain_mismatch": multihost_domain_mismatch,
    }


def build_summary() -> dict[str, object]:
    rubin = simulate_rubin_mla_nvfp4_decode_plans()
    mnnvl = simulate_mnnvl_collective_handle_selection()

    invariants = {
        "rubin_nvfp4_kv_footprint_352b_verified": (
            rubin["nvfp4_kv_bytes_per_token"] == 352
            and rubin["query_quantized_bytes_per_row"] == 356
        ),
        "rubin_h96_mixed_vs_h128_full_dispatch_verified": (
            rubin["h96_q4_b64_kv128k_plan"]["decode_record"] == "decode"
            and rubin["h96_q4_b64_kv128k_plan"]["merge_record"] == "merge_w1"
            and rubin["h128_q2_b16_kv64k_plan"]["decode_record"] == "decode_full"
        ),
        "rubin_dcp_visible_keys_exact_partition": all(
            entry["sum"] == entry["bound"] + 1 for entry in rubin["dcp_visible_keys_check"]
        ),
        "mnnvl_single_host_no_imex_falls_back_to_posix_and_gates_cft": (
            mnnvl["b200_single_host_imex"]["final_handle_type"] == "FABRIC"
            and mnnvl["b200_single_host_no_imex_fallback"]["final_handle_type"] == "POSIX_FD"
            and mnnvl["b200_single_host_no_imex_fallback"]["fallback_to_posix"] is True
            and mnnvl["b200_single_host_no_imex_fallback"]["cft_enabled"] is False
            and mnnvl["multihost_domain_mismatch"]["raised"] is True
        ),
    }

    return {
        "experiment": "flashinfer_rubin_nvfp4_mla_mnnvl_handle",
        "phase": 41,
        "priority": 50,
        "author_benchmarks": AUTHOR_BENCHMARKS,
        "rubin_mla_nvfp4_decode": rubin,
        "mnnvl_collective_handle": mnnvl,
        "invariants": invariants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    assert all(summary["invariants"].values()), f"Invariant failed: {summary['invariants']}"

    for path in SUMMARY_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "experiment": summary["experiment"],
                    "invariants": summary["invariants"],
                }
            )
        )


if __name__ == "__main__":
    main()
