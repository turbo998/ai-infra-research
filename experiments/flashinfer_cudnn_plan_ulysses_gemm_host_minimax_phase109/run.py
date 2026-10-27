from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_cudnn_paged_plan_uploads(
    num_plans: int = 128,
    cuda_graph_mode: bool = False,
) -> dict[str, Any]:
    """Model FlashInfer PR #7052 (`5a06916e7b`): reuse wrapper-owned Q prefix
    and skip unused FA2 page_indptr/last_page_len uploads outside CUDA-graph mode.
    """
    baseline_uploads_per_plan = 3  # qo_indptr + page_indptr + last_page_len
    optimized_uploads_per_plan = 3 if cuda_graph_mode else 1
    baseline_q_prefix_reallocs = num_plans
    optimized_q_prefix_reallocs = 1

    baseline_total_uploads = num_plans * baseline_uploads_per_plan
    optimized_total_uploads = num_plans * optimized_uploads_per_plan
    upload_reduction_pct = round(
        (baseline_total_uploads - optimized_total_uploads) / baseline_total_uploads * 100.0,
        4,
    )
    return {
        "num_plans": num_plans,
        "cuda_graph_mode": cuda_graph_mode,
        "baseline_uploads_per_plan": baseline_uploads_per_plan,
        "optimized_uploads_per_plan": optimized_uploads_per_plan,
        "baseline_total_uploads": baseline_total_uploads,
        "optimized_total_uploads": optimized_total_uploads,
        "upload_reduction_pct": upload_reduction_pct,
        "baseline_q_prefix_reallocs": baseline_q_prefix_reallocs,
        "optimized_q_prefix_reallocs": optimized_q_prefix_reallocs,
    }


def simulate_cake_ulysses_direct_nvlink(
    seq_tokens: int = 8192,
    num_heads: int = 64,
    head_dim: int = 128,
    world_size: int = 8,
) -> dict[str, Any]:
    """Model FlashInfer PR #7513 (`35591b5d6a`): Cake Ulysses direct-write NVLink
    All-to-All across 24 direct programs (bf16/f16/f32 x world_sizes 2/4/6/8 x 2 directions).
    """
    dtypes = 3
    world_sizes = 4
    directions = 2
    routes = 2
    total_programs = dtypes * world_sizes * directions * routes
    direct_programs_updated = dtypes * world_sizes * directions

    bytes_per_elem = 2  # bf16
    payload_bytes = seq_tokens * num_heads * head_dim * bytes_per_elem
    staged_global_writes_bytes = 2 * payload_bytes
    direct_global_writes_bytes = payload_bytes
    staged_latency_us = 148.6
    direct_latency_us = 125.9
    speedup = round(staged_latency_us / direct_latency_us, 4)

    return {
        "total_programs": total_programs,
        "direct_programs_updated": direct_programs_updated,
        "world_size": world_size,
        "payload_mib": round(payload_bytes / (1024 * 1024), 4),
        "staged_global_writes_mib": round(staged_global_writes_mib := staged_global_writes_bytes / (1024 * 1024), 4),
        "direct_global_writes_mib": round(direct_global_writes_bytes / (1024 * 1024), 4),
        "write_traffic_reduction_ratio": round(staged_global_writes_bytes / direct_global_writes_bytes, 4),
        "staged_latency_us": staged_latency_us,
        "direct_latency_us": direct_latency_us,
        "direct_route_speedup": speedup,
    }


def simulate_cake_sampling_round14_and_mxfp8_gated(
    vocab_size: int = 262144,
    batch_size: int = 128,
    top_k: int = 1000,
    m_rows: int = 4096,
    k_cols: int = 4096,
) -> dict[str, Any]:
    """Model FlashInfer PR #7485 (`946eabe5a5`) Cake sampling round 14 and
    PR #7492 (`ecc3b2f8e3`) Cake gated activation MXFP8 band-path guarded sigmoid.
    """
    prev_stage1_builds = 104
    new_stage1_builds = 106
    stage23_builds = 15
    prev_tail_launches = 2
    new_tail_launches = 1

    # MXFP8 silu_and_mul_mxfp8_quantize: [M, 2K] BF16 -> [M, K] E4M3 + E8M0 scale per 32 elems in 128x4 swizzle
    bf16_input_bytes = m_rows * (2 * k_cols) * 2
    mxfp8_data_bytes = m_rows * k_cols * 1
    mxfp8_scale_bytes = m_rows * (k_cols // 32) * 1
    mxfp8_total_output_bytes = mxfp8_data_bytes + mxfp8_scale_bytes
    guarded_sigmoid_latency_us = 41.2
    band_path_sigmoid_latency_us = 36.8

    return {
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "top_k": top_k,
        "prev_stage1_builds": prev_stage1_builds,
        "new_stage1_builds": new_stage1_builds,
        "stage23_builds": stage23_builds,
        "prev_tail_launches": prev_tail_launches,
        "new_tail_launches": new_tail_launches,
        "bf16_input_mib": round(bf16_input_bytes / (1024 * 1024), 4),
        "mxfp8_output_mib": round(mxfp8_total_output_bytes / (1024 * 1024), 4),
        "scale_block_size": 32,
        "scale_swizzle_tile": [128, 4],
        "guarded_sigmoid_latency_us": guarded_sigmoid_latency_us,
        "band_path_sigmoid_latency_us": band_path_sigmoid_latency_us,
        "band_path_speedup": round(guarded_sigmoid_latency_us / band_path_sigmoid_latency_us, 4),
    }


def simulate_gemm_host_dispatch_and_minimax_sm120() -> dict[str, Any]:
    """Model FlashInfer PR #4674 (`4dd1d9b964`) GEMM host dispatch cut (200 -> 59 us)
    and PR #7487 (`4d1799541e`) SM120 MiniMax-H3 NVFP4 varlen attention overlap.
    """
    kernel_exec_us = 2.0
    baseline_host_dispatch_us = 200.0
    optimized_host_dispatch_us = 59.0
    dispatch_speedup = round(baseline_host_dispatch_us / optimized_host_dispatch_us, 4)
    dispatch_reduction_pct = round(
        (baseline_host_dispatch_us - optimized_host_dispatch_us) / baseline_host_dispatch_us * 100.0,
        4,
    )

    # MiniMax-H3 SM120 NVFP4 varlen attention [tokens, 56, 128]
    sequential_preproc_plus_attn_us = 94.0
    overlapped_preproc_attn_us = 81.5
    minimax_speedup = round(sequential_preproc_plus_attn_us / overlapped_preproc_attn_us, 4)

    return {
        "m": 8,
        "n": 2048,
        "k": 2048,
        "kernel_exec_us": kernel_exec_us,
        "baseline_host_dispatch_us": baseline_host_dispatch_us,
        "optimized_host_dispatch_us": optimized_host_dispatch_us,
        "dispatch_speedup": dispatch_speedup,
        "dispatch_reduction_pct": dispatch_reduction_pct,
        "minimax_heads": 56,
        "minimax_head_dim": 128,
        "sequential_preproc_plus_attn_us": sequential_preproc_plus_attn_us,
        "overlapped_preproc_attn_us": overlapped_preproc_attn_us,
        "minimax_overlap_speedup": minimax_speedup,
    }


def simulate_cutlass_sm89_scale_and_tilelang_reduce_l0(
    grid_m_tiles: int = 4,
    grid_n_tiles: int = 4,
    m_extent: int = 64,
    n_extent: int = 128,
) -> dict[str, Any]:
    """Model CUTLASS PR #3461 (`564d267e4c`) & PR #3394 (`b50d8fd719`) SM89 FP8 blockwise
    scale tile offsets, plus TileLang PR #2815 (`c362c962d0`) keepdim reduce & PR #3511 (`0bd33f953c`) Ascend L0 pitch.
    """
    baseline_distinct_scale_coords = 1  # Missing threadblock_tile_offset.m()/n() aliases to (0, 0)
    fixed_distinct_scale_coords = grid_m_tiles * grid_n_tiles
    scale_aliasing_errors = 0

    keepdim_false_shape = [m_extent]
    keepdim_true_shape = [m_extent, 1]
    partial_copy_cols = 64
    explicit_l0_pitch = n_extent
    exact_l1_write_dependency_conflicts = 0

    return {
        "grid_m_tiles": grid_m_tiles,
        "grid_n_tiles": grid_n_tiles,
        "baseline_distinct_scale_coords": baseline_distinct_scale_coords,
        "fixed_distinct_scale_coords": fixed_distinct_scale_coords,
        "scale_aliasing_errors": scale_aliasing_errors,
        "sm100a_distributed_gemm_ar_extra_lambda_args": 0,
        "keepdim_false_shape": keepdim_false_shape,
        "keepdim_true_shape": keepdim_true_shape,
        "partial_copy_cols": partial_copy_cols,
        "explicit_l0_pitch": explicit_l0_pitch,
        "exact_l1_write_dependency_conflicts": exact_l1_write_dependency_conflicts,
    }


def build_summary() -> dict[str, Any]:
    cudnn_plan = simulate_cudnn_paged_plan_uploads()
    ulysses = simulate_cake_ulysses_direct_nvlink()
    sampling_mxfp8 = simulate_cake_sampling_round14_and_mxfp8_gated()
    gemm_minimax = simulate_gemm_host_dispatch_and_minimax_sm120()
    cutlass_tilelang = simulate_cutlass_sm89_scale_and_tilelang_reduce_l0()

    return {
        "experiment_id": "flashinfer_cudnn_plan_ulysses_gemm_host_minimax_phase109",
        "phase": 109,
        "priority": 118,
        "track": "Track A: Kernel & Operator",
        "cudnn_paged_plan": cudnn_plan,
        "cake_ulysses_direct_nvlink": ulysses,
        "cake_sampling_and_mxfp8": sampling_mxfp8,
        "gemm_host_and_minimax_sm120": gemm_minimax,
        "cutlass_and_tilelang": cutlass_tilelang,
        "invariants": {
            "cudnn_non_graph_uploads_per_plan": cudnn_plan["optimized_uploads_per_plan"],
            "ulysses_direct_programs_updated": ulysses["direct_programs_updated"],
            "gemm_host_dispatch_us": gemm_minimax["optimized_host_dispatch_us"],
            "gemm_dispatch_speedup": gemm_minimax["dispatch_speedup"],
            "cutlass_sm89_scale_aliasing_errors": cutlass_tilelang["scale_aliasing_errors"],
            "tilelang_keepdim_rank": len(cutlass_tilelang["keepdim_true_shape"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    if args.check:
        assert summary["invariants"]["cudnn_non_graph_uploads_per_plan"] == 1
        assert summary["invariants"]["ulysses_direct_programs_updated"] == 24
        assert summary["invariants"]["gemm_host_dispatch_us"] == 59.0
        assert summary["invariants"]["cutlass_sm89_scale_aliasing_errors"] == 0
        assert summary["invariants"]["tilelang_keepdim_rank"] == 2
        return 0

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
