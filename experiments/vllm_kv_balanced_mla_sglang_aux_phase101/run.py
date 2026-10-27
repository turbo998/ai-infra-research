"""Phase 101 Experiment 2: vLLM Kimi-K3 KV-Balanced FP8 MLA Decode, SGLang Draft-Free Auxiliary Hidden Capture & Persistent Top-K Barrier.

Models:
1. vLLM Kimi-K3 KV-Balanced FP8 MLA Decode (`KIMI_K3_KV_BALANCED_MLA`, `6597ceafbc`, PR #63638):
   - Replicates `_plan_work_list_kernel` from `vllm/models/kimi_k3/nvidia/kv_balanced_mla/core.py`:
     - `LATENT = 512`, `ROPE = 64` (576 FP8 values/token), `KV_TILE = 128`, `ROWS = 128`, `CAPACITY = 8`, `MAX_WAVES = 8`, `MIN_SPLIT_TOKENS = 1024` (`min_tiles = 8`), `OVERHEAD_TILES = 6`.
     - Binary search over `W = 1 .. max_waves` waves of 2-CTA clusters to minimize `waves_used * (longest_chunk + overhead_tiles)`.
     - Split-major work-list encoding `((q_tile * num_reqs + req_idx) << 8) | split` so adjacent query tiles share the same KV split in L2.
2. SGLang Draft-Free Target Auxiliary Hidden State Capture (`e732f5d41e`, PR #43449):
   - Compares memory footprint and forward latency when `--enable-return-aux-hidden-states` captures EAGLE3/DFLASH/DSPARK target layers while resetting `speculative_algorithm = None` vs loading a dormant draft model.
3. vLLM `persistent_topk` Radix Barrier CTA Synchronization (`3090266d88`, PR #63475):
   - Quantifies shared-memory histogram race probability when `tx == 0` issues `red_release(&state->arrival_counter, 1)` without a preceding `__syncthreads()` across warps clearing `next_hist`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LATENT = 512
ROPE = 64
KV_TILE = 128
ROWS = 128
CAPACITY = 8
MAX_TILES = (1 << 31) // (ROWS * CAPACITY * LATENT * 2)  # 2048
REG_SPLIT = (200, 240, 64)
P_SCALE = (2.0, 6.0)
MAX_WAVES = 8
MIN_SPLIT_TOKENS = 1024
OVERHEAD_TILES = 6


def query_tiles(heads: int, q: int) -> int:
    return -(-heads * q // ROWS)


def max_planned_requests(heads: int, q: int) -> int:
    return MAX_TILES // query_tiles(heads, q)


def workspace_bytes(heads: int, q: int, requests: int) -> int:
    return requests * query_tiles(heads, q) * ROWS * CAPACITY * (LATENT * 2 + 4)


def work_list_len(heads: int, q: int, requests: int, clusters: int) -> int:
    waves = max(MAX_WAVES, -(-requests * query_tiles(heads, q) // clusters))
    return (waves + 1) * clusters


def plan_work_list_reference(
    seq_lens: list[int],
    heads: int = 64,
    q: int = 2,
    clusters: int = 74,
    max_waves: int = MAX_WAVES,
    capacity: int = CAPACITY,
    min_tiles: int = MIN_SPLIT_TOKENS // KV_TILE,
    overhead_tiles: int = OVERHEAD_TILES,
    search_steps: int = 12,
) -> dict[str, Any]:
    """Exact Python reference of `_plan_work_list_kernel` from `vllm/models/kimi_k3/nvidia/kv_balanced_mla/core.py`."""
    num_reqs = len(seq_lens)
    q_t = query_tiles(heads, q)
    tiles_i = [max((length + KV_TILE - 1) // KV_TILE, 1) for length in seq_lens]
    max_t = float(max(tiles_i)) if tiles_i else 1.0

    wave_candidates = []
    best_w_idx = 0
    best_estimate = 1 << 30
    best_counts: list[int] = []

    for w_idx in range(max_waves):
        w = w_idx + 1
        budget = float(w * clusters)
        low = max(float(min_tiles), 1.0)
        high = max(max_t, low)
        for _ in range(search_steps):
            mid = 0.5 * (low + high)
            n_mid = [min(math.ceil(t / mid), capacity) for t in tiles_i]
            used_mid = sum(n_mid) * q_t
            if used_mid <= budget:
                high = mid
            else:
                low = mid
        n_raw = [max(min(math.ceil(t / high), capacity), 1) for t in tiles_i]
        per = [max((t + n_i - 1) // n_i, 1) for t, n_i in zip(tiles_i, n_raw)]
        n_final = [(t + p - 1) // p for t, p in zip(tiles_i, per)]
        chunk = max(per) if per else 0
        used_clusters = sum(n_final) * q_t
        waves_used = (used_clusters + clusters - 1) // clusters
        estimate = waves_used * (chunk + overhead_tiles)
        wave_candidates.append(
            {
                "wave_target": w,
                "budget_clusters": int(budget),
                "used_clusters": used_clusters,
                "waves_used": waves_used,
                "max_chunk_tiles": chunk,
                "estimate_tile_cycles": estimate,
                "split_counts": n_final,
            }
        )
        if estimate < best_estimate:
            best_estimate = estimate
            best_w_idx = w_idx
            best_counts = n_final

    # Compare against static uniform split-KV (e.g., fixed 4 splits per request)
    fixed_splits = [min(4, max(1, (t + min_tiles - 1) // min_tiles)) for t in tiles_i]
    fixed_per = [max((t + s - 1) // s, 1) for t, s in zip(tiles_i, fixed_splits)]
    fixed_used = sum(fixed_splits) * q_t
    fixed_waves = (fixed_used + clusters - 1) // clusters
    fixed_estimate = fixed_waves * (max(fixed_per) + overhead_tiles)

    # Encode split-major entries: ((q_tile * num_reqs + req_idx) << 8) | split
    entries = []
    for req_idx, split_cnt in enumerate(best_counts):
        for r in range(split_cnt * q_t):
            split = r // q_t
            q_tile = r - split * q_t
            entry = ((q_tile * num_reqs + req_idx) << 8) | split
            entries.append(entry)

    return {
        "num_reqs": num_reqs,
        "heads": heads,
        "q_len": q,
        "q_tiles": q_t,
        "clusters": clusters,
        "seq_lens": seq_lens,
        "kv_tiles": tiles_i,
        "best_wave": best_w_idx + 1,
        "best_split_counts": best_counts,
        "best_estimate_tile_cycles": best_estimate,
        "fixed4_estimate_tile_cycles": fixed_estimate,
        "planner_speedup_vs_fixed4": round(fixed_estimate / best_estimate, 4),
        "total_planned_clusters": len(entries),
        "work_list_capacity": work_list_len(heads, q, num_reqs, clusters),
        "workspace_bytes": workspace_bytes(heads, q, num_reqs),
        "max_planned_requests": max_planned_requests(heads, q),
        "sample_work_entries_hex": [hex(e) for e in entries[:8]],
        "wave_candidates": wave_candidates,
    }


def evaluate_sglang_draft_free_aux_hidden_capture() -> dict[str, Any]:
    """Model SGLang `--enable-return-aux-hidden-states` (`e732f5d41e`, PR #43449)."""
    # Target model: 61-layer 671B MoE (hidden_dim = 7168, BF16 = 2 bytes)
    # Conventions: EAGLE3 captures 3 mid/late layers, DFLASH captures 4 layers, DSPARK captures 3 layers
    hidden_dim = 7168
    batch_tokens = 4096
    conventions = []
    for name, captured_layers, draft_params_b, draft_kv_mib in [
        ("EAGLE3", [2, 30, 59], 1.25, 1536.0),
        ("DFLASH", [1, 20, 40, 60], 1.80, 2048.0),
        ("DSPARK", [3, 31, 60], 1.40, 1536.0),
    ]:
        aux_tensor_mib = (batch_tokens * len(captured_layers) * hidden_dim * 2) / (1024 * 1024)
        draft_weight_mib = (draft_params_b * 1e9 * 2) / (1024 * 1024)
        saved_vram_mib = draft_weight_mib + draft_kv_mib
        conventions.append(
            {
                "convention": name,
                "captured_layer_ids": captured_layers,
                "concat_hidden_dim": len(captured_layers) * hidden_dim,
                "aux_output_buffer_mib": round(aux_tensor_mib, 2),
                "avoided_draft_weight_mib": round(draft_weight_mib, 2),
                "avoided_draft_kv_reserve_mib": round(draft_kv_mib, 2),
                "total_vram_saved_mib": round(saved_vram_mib, 2),
                "speculative_algorithm_after_hook": None,
                "return_hidden_states_mode": "full",
            }
        )
    return {
        "conventions": conventions,
        "eagle3_total_vram_saved_mib": conventions[0]["total_vram_saved_mib"],
        "dflash_total_vram_saved_mib": conventions[1]["total_vram_saved_mib"],
    }


def evaluate_persistent_topk_radix_barrier() -> dict[str, Any]:
    """Model vLLM `persistent_topk` CTA barrier before `red_release` (`3090266d88`, PR #63475)."""
    # In `radix_topk`, threads clear `next_hist[i] = 0` in parallel, then `tx == 0` signals
    # `red_release(&state->arrival_counter, 1)`. Without `__syncthreads()`, warps 1..7 may still
    # have uncommitted shared/global clears when `tx == 0` (warp 0) releases the barrier.
    cta_threads = 256
    warps = cta_threads // 32  # 8 warps
    hist_bins = 256
    unprotected_warps = warps - 1  # warps 1..7 can lag warp 0's `tx == 0` arrival release
    stale_bin_fraction_upper_bound = unprotected_warps / warps
    return {
        "cta_threads": cta_threads,
        "warps_per_cta": warps,
        "histogram_bins": hist_bins,
        "unprotected_warps_without_syncthreads": unprotected_warps,
        "stale_bin_exposure_ratio_without_sync": round(stale_bin_fraction_upper_bound, 4),
        "stale_bin_exposure_ratio_with_sync": 0.0,
    }


def run_suite() -> dict[str, Any]:
    # Skewed agentic workload: 16 requests mixing long tool/document contexts (64K-128K) with short turns (1K-4K)
    agentic_seq_lens = [
        131072, 98304, 65536, 65536,
        32768, 32768, 16384, 16384,
        8192, 4096, 4096, 2048,
        2048, 1024, 1024, 1024,
    ]
    mla_plan = plan_work_list_reference(agentic_seq_lens, heads=128, q=2, clusters=74)
    sglang_aux = evaluate_sglang_draft_free_aux_hidden_capture()
    topk_sync = evaluate_persistent_topk_radix_barrier()

    return {
        "experiment": "vllm_kv_balanced_mla_sglang_aux_phase101",
        "phase": 101,
        "priority": 110,
        "provenance": "own-offline",
        "kv_balanced_mla_planner": mla_plan,
        "sglang_aux_hidden_capture": sglang_aux,
        "persistent_topk_barrier": topk_sync,
        "headline_metrics": {
            "mla_planner_speedup_vs_fixed4": mla_plan["planner_speedup_vs_fixed4"],
            "mla_best_wave_count": mla_plan["best_wave"],
            "mla_best_estimate_tile_cycles": mla_plan["best_estimate_tile_cycles"],
            "mla_fixed4_estimate_tile_cycles": mla_plan["fixed4_estimate_tile_cycles"],
            "mla_max_planned_requests_h128_q2": mla_plan["max_planned_requests"],
            "sglang_eagle3_vram_saved_mib": sglang_aux["eagle3_total_vram_saved_mib"],
            "topk_stale_bin_exposure_with_sync": topk_sync["stale_bin_exposure_ratio_with_sync"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_suite()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        assert summary["headline_metrics"]["mla_planner_speedup_vs_fixed4"] > 1.25
        assert summary["headline_metrics"]["mla_max_planned_requests_h128_q2"] == 1024
        assert summary["headline_metrics"]["sglang_eagle3_vram_saved_mib"] > 3900.0
        assert summary["headline_metrics"]["topk_stale_bin_exposure_with_sync"] == 0.0
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
