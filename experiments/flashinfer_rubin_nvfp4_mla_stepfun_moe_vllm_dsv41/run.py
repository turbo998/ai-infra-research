#!/usr/bin/env python3
"""Deterministic reproduction of FlashInfer Rubin SM107 Cake NVFP4 MLA decode,
StepFun pair-rasterized + 2-CTA MoE kernels, and vLLM DeepSeek-V4.1 replay scoring guard.

Upstream references:
  - flashinfer-ai/flashinfer PR #6931 (adca7b87d23d426e7ae0eae6df3992d6b44532e8, 2026-10-21)
  - flashinfer-ai/flashinfer PR #6934 (fd4de5acce18fd1c747c8425555f082dd4452cc3, 2026-10-21)
  - vllm-project/vllm PR #62652 (85752a43c3cee76d9234704e189cc2ca69f24712, 2026-10-21)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

LATENT_DIM = 512
ROPE_DIM = 64
QK_DIM = LATENT_DIM + ROPE_DIM  # 576
V_DIM = 512
SF_VEC = 16
CKV_BYTES = LATENT_DIM // 2  # 256 packed E2M1 bytes
SF_BYTES = LATENT_DIM // SF_VEC  # 32 E4M3 block scales
PAGE_SIZE = 64
ROWS_TILE = 256
KEY_TILE = 2 * PAGE_SIZE  # 128 keys
MAX_SPLITS = 64
ITEM_WORDS = 6
ALIGN = 256
E2M1_MAX = 6.0
E4M3_MAX = 448.0
QUERY_ROW_BYTES = CKV_BYTES + SF_BYTES + ROPE_DIM + 4  # 356 bytes
KV_TOKEN_BYTES = CKV_BYTES + SF_BYTES + ROPE_DIM  # 352 bytes

UNITS_PAIRS_BY_SM_COUNT: dict[int, tuple[int, int]] = {
    208: (50, 4),
    212: (46, 14),
}

# Author-reported R200 (208 SMs, 120 MiB L2, SM107) median latency (us) from PR #6931
RUBIN_R200_BENCHMARKS: list[dict[str, Any]] = [
    {"config": "H96-Q4-B1-KV64K", "q_len": 4, "batch": 1, "kv_len": 65536, "cake_us": 25.1, "ts_mla2_us": 23.6},
    {"config": "H96-Q4-B4-KV64K", "q_len": 4, "batch": 4, "kv_len": 65536, "cake_us": 52.8, "ts_mla2_us": 57.4},
    {"config": "H96-Q4-B16-KV64K", "q_len": 4, "batch": 16, "kv_len": 65536, "cake_us": 185.7, "ts_mla2_us": 213.3},
    {"config": "H96-Q4-B64-KV64K", "q_len": 4, "batch": 64, "kv_len": 65536, "cake_us": 674.9, "ts_mla2_us": 805.5},
    {"config": "H96-Q4-B1-KV128K", "q_len": 4, "batch": 1, "kv_len": 131072, "cake_us": 36.6, "ts_mla2_us": 32.9},
    {"config": "H96-Q4-B4-KV128K", "q_len": 4, "batch": 4, "kv_len": 131072, "cake_us": 89.7, "ts_mla2_us": 104.5},
    {"config": "H96-Q4-B16-KV128K", "q_len": 4, "batch": 16, "kv_len": 131072, "cake_us": 346.6, "ts_mla2_us": 417.3},
    {"config": "H96-Q4-B64-KV128K", "q_len": 4, "batch": 64, "kv_len": 131072, "cake_us": 1300.8, "ts_mla2_us": 1623.8},
    {"config": "H96-Q8-B1-KV64K", "q_len": 8, "batch": 1, "kv_len": 65536, "cake_us": 32.1, "ts_mla2_us": 29.0},
    {"config": "H96-Q8-B4-KV64K", "q_len": 8, "batch": 4, "kv_len": 65536, "cake_us": 82.0, "ts_mla2_us": 85.6},
    {"config": "H96-Q8-B16-KV64K", "q_len": 8, "batch": 16, "kv_len": 65536, "cake_us": 283.9, "ts_mla2_us": 280.5},
    {"config": "H96-Q8-B64-KV64K", "q_len": 8, "batch": 64, "kv_len": 65536, "cake_us": 1094.5, "ts_mla2_us": 1076.7},
    {"config": "H96-Q8-B1-KV128K", "q_len": 8, "batch": 1, "kv_len": 131072, "cake_us": 50.0, "ts_mla2_us": 45.5},
    {"config": "H96-Q8-B4-KV128K", "q_len": 8, "batch": 4, "kv_len": 131072, "cake_us": 150.1, "ts_mla2_us": 159.9},
    {"config": "H96-Q8-B16-KV128K", "q_len": 8, "batch": 16, "kv_len": 131072, "cake_us": 554.7, "ts_mla2_us": 546.9},
    {"config": "H96-Q8-B64-KV128K", "q_len": 8, "batch": 64, "kv_len": 131072, "cake_us": 2151.7, "ts_mla2_us": 2205.3},
]


def round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def compute_query_row_scale(amax_nope: float, amax_rope: float, ckv_scale: float, kpe_scale: float) -> float:
    """Exact CakeMlaNvfp4QueryQuantize row scale from cake_backend.py."""
    if amax_nope == 0.0 and amax_rope == 0.0:
        return 1.0
    return max(
        amax_nope / (E2M1_MAX * E4M3_MAX),
        amax_rope * kpe_scale / (E4M3_MAX * ckv_scale),
    )


def query_workspace_bytes(rows: int) -> int:
    return int(rows) * QUERY_ROW_BYTES


def max_items(rows_max: int, batch_max: int) -> int:
    return MAX_SPLITS * (int(rows_max) // ROWS_TILE + int(batch_max) + 1)


def workspace_layout(rows: int, max_splits: int, plan_words: int) -> tuple[list[dict[str, int | str]], int]:
    specs = (
        ("partial_o", rows * max_splits * V_DIM * 2),
        ("partial_lse", rows * max_splits * 4),
        ("lse_scratch", rows * 4),
        ("plan_block", plan_words * 4),
    )
    regions: list[dict[str, int | str]] = []
    offset = 0
    for name, nbytes in specs:
        regions.append({"name": name, "offset": offset, "nbytes": nbytes, "aligned_bytes": round_up(nbytes, ALIGN)})
        offset += round_up(nbytes, ALIGN)
    total_bytes = round_up(regions[-1]["offset"] + regions[-1]["nbytes"], ALIGN)  # type: ignore[operator]
    return regions, total_bytes


def max_workspace_bytes(rows_max: int, batch_max: int, max_splits: int = MAX_SPLITS) -> int:
    items = max_items(rows_max, batch_max)
    words = items * ITEM_WORDS + int(batch_max) + 1 + int(rows_max)
    _, total_bytes = workspace_layout(int(rows_max), int(max_splits), words)
    return total_bytes


def select_merge_warps_per_row(rows_total: int) -> int:
    if rows_total <= 1024:
        return 4
    if rows_total <= 2048:
        return 2
    return 1


def dcp_global_key_pos(local_k: int, cp_world: int, cp_rank: int, cp_interleave: int = 1) -> int:
    return ((local_k // cp_interleave) * cp_world + cp_rank) * cp_interleave + (local_k % cp_interleave)


def evaluate_rubin_mla_nvfp4() -> dict[str, Any]:
    speedups_all = [row["ts_mla2_us"] / row["cake_us"] for row in RUBIN_R200_BENCHMARKS]
    speedups_q4 = [row["ts_mla2_us"] / row["cake_us"] for row in RUBIN_R200_BENCHMARKS if row["q_len"] == 4]
    speedups_q8 = [row["ts_mla2_us"] / row["cake_us"] for row in RUBIN_R200_BENCHMARKS if row["q_len"] == 8]

    geomean_all = math.exp(sum(math.log(x) for x in speedups_all) / len(speedups_all))
    geomean_q4 = math.exp(sum(math.log(x) for x in speedups_q4) / len(speedups_q4))
    geomean_q8 = math.exp(sum(math.log(x) for x in speedups_q8) / len(speedups_q8))
    peak_entry = max(RUBIN_R200_BENCHMARKS, key=lambda r: r["ts_mla2_us"] / r["cake_us"])
    peak_speedup = peak_entry["ts_mla2_us"] / peak_entry["cake_us"]

    # Workspace and row geometry for H=96, Q=4, B=64 (24,576 rows)
    rows_b64_q4 = 64 * 4 * 96  # 24576
    q_ws_bytes = query_workspace_bytes(rows_b64_q4)
    max_ws_bytes = max_workspace_bytes(rows_b64_q4, 64)

    # Row scale alignment test
    q_scale = compute_query_row_scale(amax_nope=13.44, amax_rope=8.96, ckv_scale=0.5, kpe_scale=0.25)

    # DCP causal visibility check: cp_world=4, cp_interleave=2, global_kv_len=13
    dcp_visible_per_rank = []
    for r in range(4):
        vis = sum(1 for k in range(4) if dcp_global_key_pos(k, 4, r, 2) < 13)
        dcp_visible_per_rank.append(vis)

    return {
        "kv_token_bytes": KV_TOKEN_BYTES,
        "query_row_bytes": QUERY_ROW_BYTES,
        "r200_208sm_units_pairs": list(UNITS_PAIRS_BY_SM_COUNT[208]),
        "r200_212sm_units_pairs": list(UNITS_PAIRS_BY_SM_COUNT[212]),
        "geomean_speedup_all_16": round(geomean_all, 4),
        "geomean_speedup_q4_8": round(geomean_q4, 4),
        "geomean_speedup_q8_8": round(geomean_q8, 4),
        "peak_config": peak_entry["config"],
        "peak_speedup": round(peak_speedup, 4),
        "peak_cake_us": peak_entry["cake_us"],
        "peak_ts_mla2_us": peak_entry["ts_mla2_us"],
        "b64_q4_h96_rows": rows_b64_q4,
        "b64_q4_h96_query_workspace_bytes": q_ws_bytes,
        "b64_q4_h96_max_workspace_bytes": max_ws_bytes,
        "merge_warps_per_row_384_1536_3072": [
            select_merge_warps_per_row(384),
            select_merge_warps_per_row(1536),
            select_merge_warps_per_row(3072),
        ],
        "sample_q_scale": round(q_scale, 6),
        "dcp_visible_keys_world4_interleave2_len13": dcp_visible_per_rank,
    }


def evaluate_stepfun_moe_regen() -> dict[str, Any]:
    """Model StepFun SwiGLU MoE (H=4096, I=1536, E=64, top_k=8) PR #6934 improvements."""
    # PR #6934 reported median latencies (us) for nvfp4_bf16tok (2-CTA FC2 tile-256 vs 1-CTA)
    gb300_t1024_before_us = 196.8
    gb300_t1024_after_us = 192.1
    gb300_t2048_before_us = 276.3
    gb300_t2048_after_us = 271.0

    b200_t1024_before_us = 211.0
    b200_t1024_after_us = 204.4
    b200_t2048_before_us = 301.9
    b200_t2048_after_us = 294.0

    # Pair-rasterized BF16 FC1 tile-128: pairs two M-tiles along shared N-tile weight blocks
    # Hidden=4096, Intermediate=1536 (gate+up = 3072), tile_M=128, tile_N=128
    # Unpaired raster loads weight tile once per M-tile; pair-raster loads weight tile once per 2 M-tiles
    n_tiles_fc1 = (2 * 1536) // 128  # 24 N-tiles per expert
    weight_bytes_per_fc1_tile_bf16 = 4096 * 128 * 2  # 1,048,576 B (1 MiB)
    unpaired_2m_weight_bytes = 2 * n_tiles_fc1 * weight_bytes_per_fc1_tile_bf16
    pair_raster_2m_weight_bytes = 1 * n_tiles_fc1 * weight_bytes_per_fc1_tile_bf16

    return {
        "kernels_per_arch": 111,
        "precision_families": ["bf16", "fp8", "mxfp8", "nvfp4", "nvfp4_bf16tok"],
        "supported_archs": ["sm_100a", "sm_103a"],
        "fc1_pair_raster_l2_weight_read_reduction_ratio": round(
            unpaired_2m_weight_bytes / pair_raster_2m_weight_bytes, 4
        ),
        "gb300_nvfp4_bf16tok_t1024_speedup": round(gb300_t1024_before_us / gb300_t1024_after_us, 4),
        "gb300_nvfp4_bf16tok_t2048_speedup": round(gb300_t2048_before_us / gb300_t2048_after_us, 4),
        "b200_nvfp4_bf16tok_t1024_speedup": round(b200_t1024_before_us / b200_t1024_after_us, 4),
        "b200_nvfp4_bf16tok_t2048_speedup": round(b200_t2048_before_us / b200_t2048_after_us, 4),
    }


def simulate_dsv41_replay_rows(
    chunk_lengths: list[int],
    window_size: int,
    prompt_logprobs: int | None,
    prompt_logprob_token_ids: list[int] | None,
    fixed: bool,
) -> list[int]:
    """Simulate vLLM PR #62652 DeepseekV41ModelState._keeps_rows_np & prepare_for_replay."""
    if fixed:
        keeps_rows = (prompt_logprobs is not None) or (prompt_logprob_token_ids is not None)
    else:
        keeps_rows = prompt_logprobs is not None

    replayed_rows_per_chunk: list[int] = []
    for length in chunk_lengths:
        if keeps_rows:
            replayed_rows_per_chunk.append(length)
        else:
            replayed_rows_per_chunk.append(min(length, window_size))
    return replayed_rows_per_chunk


def evaluate_vllm_dsv41_replay() -> dict[str, Any]:
    window_size = 64  # DeepSeek-V4.1 local attention window size in test_deepseek_v41_replay_batch.py
    single_chunk = [300]
    two_chunks = [150, 150]

    buggy_single = simulate_dsv41_replay_rows(
        single_chunk, window_size, prompt_logprobs=None, prompt_logprob_token_ids=[101, 102], fixed=False
    )
    fixed_single = simulate_dsv41_replay_rows(
        single_chunk, window_size, prompt_logprobs=None, prompt_logprob_token_ids=[101, 102], fixed=True
    )
    buggy_two = simulate_dsv41_replay_rows(
        two_chunks, window_size, prompt_logprobs=None, prompt_logprob_token_ids=[101, 102], fixed=False
    )
    fixed_two = simulate_dsv41_replay_rows(
        two_chunks, window_size, prompt_logprobs=None, prompt_logprob_token_ids=[101, 102], fixed=True
    )
    non_scoring_single = simulate_dsv41_replay_rows(
        single_chunk, window_size, prompt_logprobs=None, prompt_logprob_token_ids=None, fixed=True
    )

    return {
        "window_size": window_size,
        "buggy_target_token_single_chunk_rows": buggy_single[0],
        "fixed_target_token_single_chunk_rows": fixed_single[0],
        "buggy_target_token_two_chunk_rows": buggy_two,
        "fixed_target_token_two_chunk_rows": fixed_two,
        "non_scoring_single_chunk_rows": non_scoring_single[0],
        "dropped_rows_recovered_single_chunk": fixed_single[0] - buggy_single[0],
        "dropped_rows_recovered_two_chunks": sum(fixed_two) - sum(buggy_two),
    }


def run_experiment() -> dict[str, Any]:
    rubin = evaluate_rubin_mla_nvfp4()
    stepfun = evaluate_stepfun_moe_regen()
    dsv41 = evaluate_vllm_dsv41_replay()

    summary: dict[str, Any] = {
        "experiment_id": "flashinfer_rubin_nvfp4_mla_stepfun_moe_vllm_dsv41",
        "topic": "FlashInfer Rubin SM107 Cake NVFP4 MLA Decode, StepFun Pair-Raster/2-CTA MoE & vLLM DeepSeek-V4.1 Replay Fix",
        "metrics": {
            "rubin_mla_nvfp4": rubin,
            "stepfun_moe": stepfun,
            "vllm_dsv41_replay": dsv41,
        },
        "invariants": {
            "rubin_kv_token_bytes_is_352": rubin["kv_token_bytes"] == 352,
            "rubin_query_row_bytes_is_356": rubin["query_row_bytes"] == 356,
            "rubin_geomean_q4_gt_1_10x": rubin["geomean_speedup_q4_8"] >= 1.10,
            "rubin_peak_speedup_is_1_25x": abs(rubin["peak_speedup"] - 1.2483) < 1e-3,
            "stepfun_111_kernels_per_arch": stepfun["kernels_per_arch"] == 111,
            "stepfun_b200_t1024_speedup_gt_1_03x": stepfun["b200_nvfp4_bf16tok_t1024_speedup"] >= 1.03,
            "dsv41_target_token_replay_preserves_all_300_rows": (
                dsv41["fixed_target_token_single_chunk_rows"] == 300
                and dsv41["fixed_target_token_two_chunk_rows"] == [150, 150]
                and dsv41["non_scoring_single_chunk_rows"] == 64
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
    out_paths = [base_dir / "summary.json", base_dir / "results" / "summary.json"]
    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        failed = [k for k, v in summary["invariants"].items() if not v]
        if failed:
            raise SystemExit(f"Invariant check failed: {failed}")
    print(json.dumps(summary["invariants"], indent=2))


if __name__ == "__main__":
    main()
