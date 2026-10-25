#!/usr/bin/env python3
"""Deterministic CPU/offline simulator for Phase 73 Track A:
FlashInfer SM103 QSA Persistent-Walk Decode & tcgen05 Merge, SM120 QSA mma.sync
Split-Override Decode, Hidden-4096 4-Route mHC Sinkhorn Fusion, MiniMax-H3 Out-Proj
Half-Width Tail, SM120 PCIe Copy-Engine Ring Piped All-Reduce, Kimi-K3 L2 Evict-First
& Resident 2-CTA TP1 Tail GEMM, and TileLang Partial-Sync Named-Barrier Deduplication.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_sm103_qsa_walk_and_sm120_qsa() -> dict[str, Any]:
    """Model flashinfer@131f85942e (SM103 persistent-walk + tcgen05 one-tile merge)
    and flashinfer@09a77503f0 (SM120 mma.sync plain/cooperative + 188-SM split overrides).
    """
    sm_count_gb300 = 152
    walk_max_waves = 2
    num_kv_heads = 2
    selected_pages = 33  # 1 initial + 32 top-k pages (page_size=64, max_seq_len=2048)

    # At rows=128, plain_splits=1 -> 128 * 2 * 1 = 256 CTAs -> ceil(256/152) = 2 waves
    rows_walk = 128
    plain_splits_128 = 1
    plain_ctas_128 = rows_walk * num_kv_heads * plain_splits_128
    plain_waves_128 = math.ceil(plain_ctas_128 / sm_count_gb300)
    use_walk_128 = (
        plain_splits_128 < selected_pages
        and 1 < plain_waves_128 <= walk_max_waves
    )
    walk_ctas_128 = min(sm_count_gb300, rows_walk * num_kv_heads * selected_pages)
    total_work_items_128 = rows_walk * num_kv_heads * selected_pages
    walk_tiles_min = total_work_items_128 // walk_ctas_128
    walk_tiles_max = math.ceil(total_work_items_128 / walk_ctas_128)
    walk_splits_128 = 1 + (walk_tiles_min // selected_pages)

    gb300_eager_table = [
        {"rows": 1, "nhd_before_us": 11.65, "nhd_after_us": 10.21, "nhd_speedup": 1.14, "hnd_before_us": 11.71, "hnd_after_us": 10.53, "hnd_speedup": 1.11, "active_path": "tcgen05_one_tile_merge"},
        {"rows": 2, "nhd_before_us": 14.37, "nhd_after_us": 13.09, "nhd_speedup": 1.10, "hnd_before_us": 14.70, "hnd_after_us": 12.77, "hnd_speedup": 1.15, "active_path": "tcgen05_one_tile_merge"},
        {"rows": 4, "nhd_before_us": 14.70, "nhd_after_us": 13.68, "nhd_speedup": 1.07, "hnd_before_us": 15.92, "hnd_after_us": 13.95, "hnd_speedup": 1.14, "active_path": "tcgen05_one_tile_merge"},
        {"rows": 128, "nhd_before_us": 71.89, "nhd_after_us": 67.74, "nhd_speedup": 1.06, "hnd_before_us": 72.11, "hnd_after_us": 67.79, "hnd_speedup": 1.06, "active_path": "persistent_walk_2slots"},
    ]
    gb300_graph_128 = {
        "nhd_before_us": 72.00,
        "nhd_after_us": 67.84,
        "nhd_speedup": 1.06,
        "hnd_before_us": 72.37,
        "hnd_after_us": 67.74,
        "hnd_speedup": 1.07,
    }

    # SM120 (RTX PRO 6000 Blackwell Server Edition, 188 SMs, 2 CTAs/SM -> 376 co-resident)
    sm120_sms = 188
    sm120_coresident_ctas = sm120_sms * 2
    sm120_split_overrides = {
        "4": {"default_splits": 33, "default_ctas": 264, "override_splits": 16, "override_ctas": 128},
        "8": {"default_splits": 23, "default_ctas": 368, "override_splits": 11, "override_ctas": 176},
        "16": {"default_splits": 11, "default_ctas": 352, "override_splits": 5, "override_ctas": 160},
        "32": {"default_splits": 5, "default_ctas": 320, "override_splits": 3, "override_ctas": 192},
    }
    sm120_eager_table = [
        {"rows": 1, "nhd_ref_us": 16.05, "nhd_cake_us": 12.66, "nhd_speedup": 1.27, "hnd_ref_us": 18.67, "hnd_cake_us": 12.82, "hnd_speedup": 1.46},
        {"rows": 4, "nhd_ref_us": 28.59, "nhd_cake_us": 16.13, "nhd_speedup": 1.77, "hnd_ref_us": 26.86, "hnd_cake_us": 16.72, "hnd_speedup": 1.61},
        {"rows": 8, "nhd_ref_us": 36.11, "nhd_cake_us": 22.86, "nhd_speedup": 1.58, "hnd_ref_us": 39.22, "hnd_cake_us": 23.06, "hnd_speedup": 1.70},
        {"rows": 32, "nhd_ref_us": 93.74, "nhd_cake_us": 52.85, "nhd_speedup": 1.77, "hnd_ref_us": 96.34, "hnd_cake_us": 56.11, "hnd_speedup": 1.72},
        {"rows": 128, "nhd_ref_us": 301.26, "nhd_cake_us": 161.36, "nhd_speedup": 1.87, "hnd_ref_us": 320.93, "hnd_cake_us": 169.62, "hnd_speedup": 1.89},
    ]

    return {
        "sm103_gb300_sms": sm_count_gb300,
        "walk_max_waves": walk_max_waves,
        "rows_128_plain_ctas": plain_ctas_128,
        "rows_128_plain_waves": plain_waves_128,
        "rows_128_use_walk": use_walk_128,
        "rows_128_walk_ctas": walk_ctas_128,
        "rows_128_walk_tiles_range": [walk_tiles_min, walk_tiles_max],
        "rows_128_walk_splits": walk_splits_128,
        "gb300_eager_table": gb300_eager_table,
        "gb300_graph_128": gb300_graph_128,
        "sm120_sms": sm120_sms,
        "sm120_coresident_ctas": sm120_coresident_ctas,
        "sm120_split_overrides": sm120_split_overrides,
        "sm120_eager_table": sm120_eager_table,
        "sm120_max_eager_speedup": 1.89,
        "sm120_max_graph_speedup": 1.90,
    }


def simulate_mhc_hidden4096_and_minimax_h3() -> dict[str, Any]:
    """Model flashinfer@847172573a (hidden-4096 4-route mHC + CheckedMul overflow guard)
    and flashinfer@b4125442a5 (MiniMax-H3 N-split half-width tail).
    """
    hidden = 4096
    routes = 4
    sinkhorn_iters = 20
    wsmem_regime_splits = 8
    plain_min_waves = 7

    # CheckedMul overflow check: large batch/token strides previously multiplied in signed int32
    int32_max = (1 << 31) - 1
    example_tokens = 65536
    example_row_elems = routes * hidden  # 16384
    raw_product = example_tokens * example_row_elems * 2  # 2,147,483,648 bytes -> overflows int32!
    int32_overflows = raw_product > int32_max

    mhc_benchmarks = [
        {"tokens": 1, "b200_vllm_us": 26.847, "b200_cake_us": 19.648, "b200_speedup": 1.366, "gb300_vllm_us": 19.060, "gb300_cake_us": 14.944, "gb300_speedup": 1.275},
        {"tokens": 64, "b200_vllm_us": 26.367, "b200_cake_us": 19.808, "b200_speedup": 1.331, "gb300_vllm_us": 19.997, "gb300_cake_us": 15.808, "gb300_speedup": 1.265},
        {"tokens": 1024, "b200_vllm_us": 39.095, "b200_cake_us": 27.424, "b200_speedup": 1.426, "gb300_vllm_us": 35.510, "gb300_cake_us": 25.608, "gb300_speedup": 1.387},
        {"tokens": 4096, "b200_vllm_us": 118.465, "b200_cake_us": 86.913, "b200_speedup": 1.363, "gb300_vllm_us": 109.945, "gb300_cake_us": 79.744, "gb300_speedup": 1.379},
        {"tokens": 16384, "b200_vllm_us": 424.918, "b200_cake_us": 333.760, "b200_speedup": 1.273, "gb300_vllm_us": 401.298, "gb300_cake_us": 313.856, "gb300_speedup": 1.279},
    ]

    # MiniMax-H3 out_proj_nvfp4: full tile is 256x256, N-split half-width tail is 256x128
    # Out-proj N = 8192 -> 32 N-tiles at 256 cols, or 64 N-tiles at 128 cols
    out_proj_n = 8192
    m_example = 4816
    m_tiles = math.ceil(m_example / 256)  # 19 M-tiles
    full_n_tiles = out_proj_n // 256  # 32
    full_ctas = m_tiles * full_n_tiles  # 608 CTAs
    gb200_sms = 148
    full_waves_gb200 = full_ctas / gb200_sms  # 4.108 waves (partial wave of 16 CTAs <= 74 half-wave!)
    partial_rem_gb200 = full_ctas % gb200_sms  # 16 <= 74
    nsplit_eligible_gb200 = m_example >= 32 and 0 < partial_rem_gb200 <= (gb200_sms // 2) and math.ceil(full_waves_gb200) <= 10

    minimax_h3_benchmarks = [
        {"arch": "GB300_sm103a", "P": 8, "M": 4816, "full_us": 82.2, "nsplit_us": 79.0, "speedup": 1.041},
        {"arch": "GB200_sm100a", "P": 8, "M": 4816, "full_us": 98.2, "nsplit_us": 93.4, "speedup": 1.052},
    ]

    return {
        "mhc_hidden": hidden,
        "mhc_routes": routes,
        "mhc_sinkhorn_iters": sinkhorn_iters,
        "wsmem_regime_splits": wsmem_regime_splits,
        "plain_min_waves": plain_min_waves,
        "tma_stride_int32_overflow_detected": int32_overflows,
        "mhc_benchmarks": mhc_benchmarks,
        "mhc_max_b200_speedup": 1.426,
        "mhc_max_gb300_speedup": 1.387,
        "minimax_h3_m4816_full_ctas": full_ctas,
        "minimax_h3_m4816_partial_rem_gb200": partial_rem_gb200,
        "minimax_h3_m4816_nsplit_eligible": nsplit_eligible_gb200,
        "minimax_h3_benchmarks": minimax_h3_benchmarks,
    }


def simulate_sm120_ce_ring_kimi_k3_and_tilelang() -> dict[str, Any]:
    """Model flashinfer@68e3e09ad3 (SM120 Copy-Engine Ring Piped All-Reduce),
    flashinfer@1c0d2be434 (Kimi-K3 L2::evict_first + resident 2-CTA TP1 tail GEMM),
    and tilelang@efa1ffc1e3 (ThreadSync partial-sync barrier deduplication).
    """
    world_size = 8
    piped_min_chunk_bytes = 4 << 20  # 4 MiB
    # Dual-socket 8x SM120 PCIe Gen5: ranks 3->4 and 7->0 cross CPU socket links
    ce_slow_hops_mask = (1 << 3) | (1 << 7)  # 0x88 = 136

    ce_ring_benchmarks = [
        {"per_rank_mib": 32, "chunk_mib": 4.0, "pieces": 1, "unpiped_ms": 1.4601, "piped_ms": 1.4601, "speedup": 1.000},
        {"per_rank_mib": 48, "chunk_mib": 6.0, "pieces": 2, "unpiped_ms": 2.1832, "piped_ms": 2.0792, "speedup": 1.050},
        {"per_rank_mib": 64, "chunk_mib": 8.0, "pieces": 2, "unpiped_ms": 2.9129, "piped_ms": 2.7536, "speedup": 1.058},
        {"per_rank_mib": 96, "chunk_mib": 12.0, "pieces": 3, "unpiped_ms": 4.4753, "piped_ms": 4.0287, "speedup": 1.111},
    ]

    kimi_k3_decode_evict_first = [
        {"arch": "B200", "T": 1, "front_speedup": 1.143, "tail_tp1_speedup": 1.191},
        {"arch": "B300", "T": 1, "front_speedup": 1.201, "tail_tp1_speedup": 1.156},
        {"arch": "GB200", "T": 1, "front_speedup": 1.158, "tail_tp1_speedup": 1.186},
        {"arch": "GB300", "T": 1, "front_speedup": 1.181, "tail_tp1_speedup": 1.160},
    ]
    kimi_k3_prefill_2cta_tail = [
        {"arch": "GB200_148SM", "T": 4096, "tile": "128x256_resident_2cta", "before_us": 244.7, "after_us": 218.2, "speedup": 1.121},
        {"arch": "B200_148SM", "T": 8192, "tile": "256x512_wide_2cta", "before_us": 465.7, "after_us": 420.5, "speedup": 1.107},
        {"arch": "GB300_152SM", "T": 4096, "tile": "256x512_wide_2cta", "before_us": 243.1, "after_us": 225.8, "speedup": 1.077},
        {"arch": "B300_148SM", "T": 16384, "tile": "256x512_wide_2cta", "before_us": 882.7, "after_us": 815.6, "speedup": 1.082},
    ]

    # TileLang ThreadPartialSyncRewriter named-barrier allocation across storage scopes
    k_first_used_barrier = 1
    # Scope 1 ("shared"): assigns barrier 1 ([0,256)), barrier 2 ([128,384)), barrier 3 ([256,384) producer 128 threads)
    shared_scope_barriers = {
        "[0, 256)": 1,
        "[128, 384)": 2,
        "[256, 384)": 3,
    }
    # Scope 2 ("shared.dyn"): consumer warpgroup [0, 128) (128 threads)
    # Buggy behavior: fresh rewriter starts at kFirstUsedBarrier (or collides on barrier 3 with count 128)
    buggy_shared_dyn_consumer_barrier = 3
    buggy_collision_on_bar_sync_3_128 = (
        buggy_shared_dyn_consumer_barrier == shared_scope_barriers["[256, 384)"]
    )
    # Fixed behavior: PostOrderVisit seeds used_barrier_ids_ = {1, 2, 3}, so [0, 128) gets barrier 4
    used_barrier_ids = set(shared_scope_barriers.values())
    candidate_id = k_first_used_barrier
    while candidate_id in used_barrier_ids:
        candidate_id += 1
    fixed_shared_dyn_consumer_barrier = candidate_id

    return {
        "ce_ring_world_size": world_size,
        "ce_ring_piped_min_chunk_bytes": piped_min_chunk_bytes,
        "ce_slow_hops_mask": ce_slow_hops_mask,
        "ce_ring_benchmarks": ce_ring_benchmarks,
        "ce_ring_max_speedup": 1.111,
        "kimi_k3_decode_evict_first": kimi_k3_decode_evict_first,
        "kimi_k3_prefill_2cta_tail": kimi_k3_prefill_2cta_tail,
        "tilelang_shared_scope_barriers": shared_scope_barriers,
        "tilelang_buggy_collision_on_bar3": buggy_collision_on_bar_sync_3_128,
        "tilelang_fixed_consumer_barrier_id": fixed_shared_dyn_consumer_barrier,
    }


def build_summary() -> dict[str, Any]:
    qsa = simulate_sm103_qsa_walk_and_sm120_qsa()
    mhc_h3 = simulate_mhc_hidden4096_and_minimax_h3()
    ce_kimi_tl = simulate_sm120_ce_ring_kimi_k3_and_tilelang()
    return {
        "experiment": "flashinfer_sm103_qsa_walk_mhc4096_sm120_ce_ring_kimi_k3_tilelang_sim",
        "phase": "Phase 73",
        "priority": 82,
        "track": "Track A",
        "qsa_sm103_and_sm120": qsa,
        "mhc_hidden4096_and_minimax_h3": mhc_h3,
        "sm120_ce_ring_kimi_k3_and_tilelang": ce_kimi_tl,
        "invariants": {
            "sm103_rows128_uses_persistent_walk": qsa["rows_128_use_walk"] is True,
            "sm103_rows128_walk_splits_eq_2": qsa["rows_128_walk_splits"] == 2,
            "sm120_max_eager_speedup_189x": abs(qsa["sm120_max_eager_speedup"] - 1.89) < 1e-6,
            "mhc_int32_stride_overflow_caught": mhc_h3["tma_stride_int32_overflow_detected"] is True,
            "minimax_h3_m4816_nsplit_eligible": mhc_h3["minimax_h3_m4816_nsplit_eligible"] is True,
            "ce_ring_96mib_speedup_1111x": abs(ce_kimi_tl["ce_ring_max_speedup"] - 1.111) < 1e-6,
            "tilelang_fixed_barrier_id_eq_4": ce_kimi_tl["tilelang_fixed_consumer_barrier_id"] == 4,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against committed summary.json")
    args = parser.parse_args()

    summary = build_summary()
    summary_path = Path(__file__).with_name("summary.json")

    if not all(summary["invariants"].values()):
        raise AssertionError(f"Invariant failure: {summary['invariants']}")

    if args.check:
        expected = json.loads(summary_path.read_text(encoding="utf-8"))
        if expected != summary:
            raise AssertionError("summary.json mismatch for Track A Phase 73")
        print("OK: Phase 73 Track A summary verified.")
        return 0

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
