#!/usr/bin/env python3
"""Phase 59 Track A deterministic offline models for FlashInfer & TileLang kernel commits.

Models:
1. NVFP4 Attention Sink JIT Module-Name Disambiguation & Scale Folding (`flashinfer@5be1442520`)
2. Cake FMHA Aligned hd128 Single-Query Paged Decode Cluster Band Selector (`flashinfer@8656005c6a`)
3. Cake MoK Fused-Kernel Issue-Path & TMEM Hand-Off Step Latency (`flashinfer@4a35122d9e`)
4. Cake DeepGEMM Dense MQA 64-Head Whole-Grid KV-Split Extension (`flashinfer@4b88078d61`)
5. CuTe-DSL BF16 Direct GEMM Re-Gating & Split-K Tactic Pruning (`flashinfer@afb249ac04`)
6. Cake Mega-mHC Hidden-4096 BF16 Inference Program on SM107a (`flashinfer@9ecd6e31b9`)
7. Cake Causal Conv1d Two-Launch Non-Cooperative Backward with PDL (`flashinfer@608772bd85`)
8. TileLang Signed Int64 `cuda::atomic_ref<T1>` Max/Min & AtomicStore (`tilelang@2573d47079`)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def model_nvfp4_attention_sink_jit() -> dict[str, Any]:
    """Model JIT module-name disambiguation and scale folding (`flashinfer@5be1442520`)."""
    legacy_module_name = "batch_attention_sink"
    variants = [
        {"kv_dtype": "bfloat16", "head_dim": 64, "pos_enc": 0, "use_fp16_qk_reduction": False},
        {"kv_dtype": "bfloat16", "head_dim": 128, "pos_enc": 0, "use_fp16_qk_reduction": False},
        {"kv_dtype": "uint8_nvfp4", "head_dim": 128, "pos_enc": 0, "use_fp16_qk_reduction": False},
        {"kv_dtype": "uint8_nvfp4", "head_dim": 64, "pos_enc": 1, "use_fp16_qk_reduction": True},
    ]

    disambiguated_names = []
    for v in variants:
        name = (
            f"batch_attention_sink_kv_{v['kv_dtype']}_hd_{v['head_dim']}"
            f"_pos_{v['pos_enc']}_fp16qk_{int(v['use_fp16_qk_reduction'])}"
        )
        disambiguated_names.append(name)

    head_dim = 128
    q_scale = 0.75
    k_scale = 0.50
    base_sm_scale = 1.0 / math.sqrt(head_dim)
    folded_sm_scale = base_sm_scale * q_scale * k_scale

    return {
        "commit": "5be1442520",
        "legacy_module_name": legacy_module_name,
        "legacy_unique_names_count": 1,
        "disambiguated_unique_names_count": len(set(disambiguated_names)),
        "disambiguated_names": disambiguated_names,
        "legacy_second_head_dim_rel_error_min": 0.6,
        "legacy_second_head_dim_rel_error_max": 1.2,
        "fixed_nvfp4_decode_rel_error": 1.8e-3,
        "base_sm_scale": round(base_sm_scale, 8),
        "folded_sm_scale": round(folded_sm_scale, 8),
        "fa2_uint8_declares_kv_cache_sf": True,
        "fa3_rejects_packed_fp4": True,
    }


def model_cake_fmha_aligned_hd128() -> dict[str, Any]:
    """Model Cake FMHA aligned hd128 q1 paged decode on SM100a/SM103a (`flashinfer@8656005c6a`)."""
    b200_rows = [
        {"batch": 1, "kv_heads": 8, "seq_len": 512, "dtype": "bf16", "prev_us": 24.97, "new_us": 4.87, "speedup": 5.13, "routed": True},
        {"batch": 1, "kv_heads": 8, "seq_len": 512, "dtype": "fp16", "prev_us": 19.49, "new_us": 5.13, "speedup": 3.80, "routed": True},
        {"batch": 1, "kv_heads": 8, "seq_len": 1024, "dtype": "bf16", "prev_us": 27.99, "new_us": 5.40, "speedup": 5.18, "routed": True},
        {"batch": 4, "kv_heads": 8, "seq_len": 4096, "dtype": "bf16", "prev_us": 49.71, "new_us": 17.73, "speedup": 2.80, "routed": True},
        {"batch": 32, "kv_heads": 1, "seq_len": 4096, "dtype": "bf16", "prev_us": 52.87, "new_us": 17.82, "speedup": 2.97, "routed": True},
        {"batch": 8, "kv_heads": 8, "seq_len": 4096, "dtype": "bf16", "prev_us": 56.48, "new_us": 56.38, "speedup": 1.00, "routed": False},
    ]
    gb300_rows = [
        {"batch": 1, "kv_heads": 8, "seq_len": 512, "dtype": "bf16", "prev_us": 46.56, "new_us": 4.64, "speedup": 10.03, "routed": True},
        {"batch": 1, "kv_heads": 8, "seq_len": 512, "dtype": "fp16", "prev_us": 35.97, "new_us": 4.89, "speedup": 7.35, "routed": True},
        {"batch": 1, "kv_heads": 8, "seq_len": 1024, "dtype": "bf16", "prev_us": 49.12, "new_us": 5.14, "speedup": 9.55, "routed": True},
        {"batch": 4, "kv_heads": 8, "seq_len": 4096, "dtype": "bf16", "prev_us": 72.03, "new_us": 16.96, "speedup": 4.25, "routed": True},
        {"batch": 32, "kv_heads": 1, "seq_len": 4096, "dtype": "bf16", "prev_us": 72.60, "new_us": 16.94, "speedup": 4.28, "routed": True},
        {"batch": 2, "kv_heads": 8, "seq_len": 1153, "dtype": "bf16", "prev_us": 51.65, "new_us": 6.53, "speedup": 7.91, "routed": True},
    ]

    cluster_sizes = [8, 4, 2, 1]
    dtypes = ["bf16", "fp16"]

    return {
        "commit": "8656005c6a",
        "cluster_sizes": cluster_sizes,
        "dtypes": dtypes,
        "num_aligned_variants_per_arch": len(cluster_sizes) * len(dtypes),
        "tensor_map_abi_sizeof_bytes": 128,
        "tensor_map_abi_alignof_bytes": 64,
        "b200_max_speedup_x": max(r["speedup"] for r in b200_rows),
        "gb300_max_speedup_x": max(r["speedup"] for r in gb300_rows),
        "b200_rows": b200_rows,
        "gb300_rows": gb300_rows,
    }


def model_cake_mok_issue_tmem() -> dict[str, Any]:
    """Model Cake MoK fused-kernel issue-path and TMEM hand-off (`flashinfer@4a35122d9e`)."""
    fused_variants = 16
    scheduler_epilogue_kernels = 32
    architectures = ["sm_100a", "sm_103a", "sm_107a"]
    total_libraries = (fused_variants + scheduler_epilogue_kernels) * len(architectures)

    benchmarks = [
        {"gpu": "B200", "ep": 8, "shape": "glm52_ep8_bf16_full_defaults", "prev_ms": 30.284, "new_ms": 29.668, "speedup": 1.021},
        {"gpu": "B200", "ep": 8, "shape": "glm52_ep8_bf16_full_ragged_defaults", "prev_ms": 20.447, "new_ms": 19.869, "speedup": 1.029},
        {"gpu": "B200", "ep": 8, "shape": "glm52_ep8_mxfp8_full_mb393216_defaults", "prev_ms": 22.176, "new_ms": 21.866, "speedup": 1.014},
        {"gpu": "B200", "ep": 4, "shape": "glm52_ep4_bf16_full_skew4_defaults", "prev_ms": 104.165, "new_ms": 101.561, "speedup": 1.026},
        {"gpu": "GB300", "ep": 4, "shape": "glm52_ep4_bf16_full_ragged_defaults", "prev_ms": 17.459, "new_ms": 17.073, "speedup": 1.023},
        {"gpu": "GB300", "ep": 8, "shape": "glm52_ep8_bf16_full_imbal_defaults", "prev_ms": 69.292, "new_ms": 68.014, "speedup": 1.019},
    ]

    return {
        "commit": "4a35122d9e",
        "fused_variants_per_arch": fused_variants,
        "scheduler_epilogue_per_arch": scheduler_epilogue_kernels,
        "total_libraries_per_arch": fused_variants + scheduler_epilogue_kernels,
        "total_libraries_3_archs": total_libraries,
        "max_speedup_x": max(b["speedup"] for b in benchmarks),
        "benchmarks": benchmarks,
    }


def model_cake_deepgemm_dense_mqa() -> dict[str, Any]:
    """Model DeepGEMM dense MQA 32-head rebind and 64-head KV-split tier (`flashinfer@4b88078d61`)."""
    gb300_highlights = [
        {"shape": "dense_fp8_q1_k4096_h32", "prev_us": 3.97, "new_us": 3.75, "speedup": 1.0613},
        {"shape": "dense_fp8_q1_k32768_h32", "prev_us": 4.98, "new_us": 4.77, "speedup": 1.0419},
        {"shape": "dense_fp8_q2053_k2048_h32", "prev_us": 23.01, "new_us": 22.29, "speedup": 1.0316},
        {"shape": "dense_q1024_k33803_p32779_h64", "prev_us": 221.64, "new_us": 205.51, "speedup": 1.0785},
        {"shape": "dense_q1024_k132101_p131077_h64", "prev_us": 876.60, "new_us": 817.43, "speedup": 1.0726},
    ]
    return {
        "commit": "4b88078d61",
        "programs_added_32h": 7,
        "programs_removed_32h": 4,
        "retired_64h_routes": ["le1024_tier_a", "le1024_tier_b"],
        "extended_64h_q_ceiling": 1536,
        "extended_64h_k_threshold": 2048,
        "gb300_geomean_86_shapes": 1.0053,
        "gb300_max_speedup_x": 1.0785,
        "b200_geomean_86_shapes": 1.0043,
        "b200_max_speedup_x": 1.0577,
        "gb300_highlights": gb300_highlights,
    }


def model_cute_dsl_bf16_gemm_regating() -> dict[str, Any]:
    """Model CuTe-DSL BF16 direct GEMM re-gating and split-K pruning (`flashinfer@afb249ac04`)."""
    vector_width = 8  # 16-byte BF16 vector = 8 elements
    candidate_blocks = (256, 192, 128, 96, 64, 32)

    def is_direct_candidate(m: int, n: int) -> bool:
        return m <= 3 or (m * n) <= 4096

    def prefer_direct_sm100(m: int, n: int, k: int) -> bool:
        if not (1 <= m <= 32 and k % 256 == 0):
            return False
        return (m * n <= 2048) or (m == 1 and n <= 8192 and 5120 <= k <= 16384)

    def select_default_tactic(m: int, n: int, k: int) -> dict[str, int]:
        blocks = [b for b in candidate_blocks if k % (b * vector_width) == 0]
        block_size = blocks[0]
        if m <= 4:
            block_size = next(
                (b for b in blocks if k // (b * vector_width) >= 2),
                blocks[-1],
            )
        outputs_per_block = 1 if (n * k < (1 << 21)) else (2 if n % 2 == 0 else 1)
        return {
            "block_size": block_size,
            "outputs_per_block": outputs_per_block,
            "prefetched_vectors_per_thread": k // (block_size * vector_width),
        }

    test_shapes = [
        {"m": 1, "n": 128, "k": 4096, "b300_main_us": 3.75, "b300_pr_us": 2.34},
        {"m": 2, "n": 1000, "k": 8192, "b300_main_us": 8.35, "b300_pr_us": 5.70},
        {"m": 2, "n": 256, "k": 2048, "b300_main_us": 3.40, "b300_pr_us": 2.95},
        {"m": 8, "n": 128, "k": 4096, "b300_main_us": 4.20, "b300_pr_us": 3.50},
    ]
    evaluated = []
    for s in test_shapes:
        tactic = select_default_tactic(s["m"], s["n"], s["k"])
        evaluated.append({
            **s,
            "is_candidate": is_direct_candidate(s["m"], s["n"]),
            "prefer_direct": prefer_direct_sm100(s["m"], s["n"], s["k"]),
            **tactic,
            "speedup_x": round(s["b300_main_us"] / s["b300_pr_us"], 3),
        })

    return {
        "commit": "afb249ac04",
        "warp_min_bytes_in_flight": 96 * 1024,
        "cluster_weight_bytes_in_flight": 9 * 64 * 128 * 2,  # 147,456 B (~144 KiB)
        "small_weight_threshold_elements": 1 << 21,
        "evaluated_shapes": evaluated,
    }


def model_cake_mega_mhc_h4096() -> dict[str, Any]:
    """Model Cake Mega-mHC hidden-4096 BF16 program on SM107a (`flashinfer@9ecd6e31b9`)."""
    r200_table = [
        {"tokens": 1, "existing_us": 17.28, "pr_us": 9.25, "speedup_x": 1.869},
        {"tokens": 4, "existing_us": 24.74, "pr_us": 10.08, "speedup_x": 2.454},
        {"tokens": 16, "existing_us": 25.63, "pr_us": 10.46, "speedup_x": 2.450},
        {"tokens": 64, "existing_us": 29.22, "pr_us": 11.09, "speedup_x": 2.635},
        {"tokens": 256, "existing_us": 40.74, "pr_us": 12.38, "speedup_x": 3.289},
        {"tokens": 1024, "existing_us": 105.23, "pr_us": 18.91, "speedup_x": 5.564},
        {"tokens": 2048, "existing_us": 226.30, "pr_us": 30.02, "speedup_x": 7.539},
        {"tokens": 4096, "existing_us": 424.53, "pr_us": 56.58, "speedup_x": 7.504},
        {"tokens": 8192, "existing_us": 970.10, "pr_us": 112.83, "speedup_x": 8.598},
    ]
    return {
        "commit": "9ecd6e31b9",
        "existing_h5120_fp8_variants": 14,
        "new_h4096_bf16_variants": 1,
        "total_mega_mhc_programs": 15,
        "target_arch": "sm_107a",
        "residual_routes": 4,
        "fast_divmod_sizeof_bytes": 12,
        "min_speedup_x": 1.869,
        "max_speedup_x": 8.598,
        "r200_table": r200_table,
    }


def model_cake_causal_conv1d_non_coop_bwd() -> dict[str, Any]:
    """Model Cake causal conv1d two-launch non-cooperative backward (`flashinfer@608772bd85`)."""
    stream_co_residency_rows = [
        {"gpu": "B200", "S": 250340, "C": 512, "prev_alone_ms": 0.168, "pr_alone_ms": 0.169, "prev_1thread_ms": 4.740, "pr_1thread_ms": 0.255, "speedup_1thread_x": 18.6, "prev_8spin_ms": 20.009, "pr_8spin_ms": 0.306, "speedup_8spin_x": 65.4},
        {"gpu": "B200", "S": 250340, "C": 2048, "prev_alone_ms": 0.627, "pr_alone_ms": 0.611, "prev_1thread_ms": 5.208, "pr_1thread_ms": 0.968, "speedup_1thread_x": 5.4, "prev_8spin_ms": 20.478, "pr_8spin_ms": 0.978, "speedup_8spin_x": 20.9},
        {"gpu": "GB300", "S": 250340, "C": 512, "prev_alone_ms": 0.150, "pr_alone_ms": 0.150, "prev_1thread_ms": 4.490, "pr_1thread_ms": 0.231, "speedup_1thread_x": 19.4, "prev_8spin_ms": 18.983, "pr_8spin_ms": 0.277, "speedup_8spin_x": 68.5},
        {"gpu": "R200", "S": 250340, "C": 512, "prev_alone_ms": 0.098, "pr_alone_ms": 0.097, "prev_1thread_ms": 3.814, "pr_1thread_ms": 0.149, "speedup_1thread_x": 25.6, "prev_8spin_ms": 16.217, "pr_8spin_ms": 0.149, "speedup_8spin_x": 108.8},
    ]

    # Workspace bytes formula: rows = ceil(grid / n_ch_tiles), FP32 [rows, C, 4]
    sm_count = 148
    ctas_per_sm = 2
    grid_mult = 2
    grid = grid_mult * ctas_per_sm * sm_count  # 592 CTAs
    channels = 2048
    n_ch_tiles = channels // 64  # 32
    rows = math.ceil(grid / n_ch_tiles)  # 19 rows
    workspace_bytes = rows * channels * 4 * 4  # FP32 = 4 bytes

    return {
        "commit": "608772bd85",
        "backward_launches_per_call": 2,
        "uses_cooperative_grid": False,
        "uses_pdl_reduce": True,
        "bitwise_deterministic_dweight": True,
        "example_workspace_rows": rows,
        "example_workspace_bytes": workspace_bytes,
        "max_speedup_1thread_x": max(r["speedup_1thread_x"] for r in stream_co_residency_rows),
        "max_speedup_8spin_x": max(r["speedup_8spin_x"] for r in stream_co_residency_rows),
        "stream_co_residency_rows": stream_co_residency_rows,
    }


def model_tilelang_int64_atomics() -> dict[str, Any]:
    """Model TileLang signed int64 atomic max/min & AtomicStore (`tilelang@2573d47079`)."""
    a_vals = [1 << 40, -(1 << 40), -7, 7, -9, 9, 0, 0]
    initial_vals = [-(1 << 40), 1 << 40, -3, 3, -9, 9, -1, 1]

    def unsigned_64(x: int) -> int:
        return x & ((1 << 64) - 1)

    signed_max = [max(i, a) for i, a in zip(initial_vals, a_vals)]
    signed_min = [min(i, a) for i, a in zip(initial_vals, a_vals)]
    # If cast to unsigned uint64 (NT1 bug), negative values wrap to 2^64 + x > positive values
    buggy_unsigned_max = [
        i if unsigned_64(i) >= unsigned_64(a) else a
        for i, a in zip(initial_vals, a_vals)
    ]
    mismatch_count = sum(1 for s, u in zip(signed_max, buggy_unsigned_max) if s != u)

    return {
        "commit": "2573d47079",
        "test_vector_len": len(a_vals),
        "signed_max_results": signed_max,
        "signed_min_results": signed_min,
        "buggy_unsigned_cast_mismatches": mismatch_count,
        "preserves_signed_t1_atomic_ref": True,
        "preserves_dest_type_in_atomic_store": True,
    }


def build_summary() -> dict[str, Any]:
    return {
        "phase": 59,
        "priority": 68,
        "track": "Track A — GPU Kernel, Quantization & Attention Architecture",
        "nvfp4_attention_sink_jit": model_nvfp4_attention_sink_jit(),
        "cake_fmha_aligned_hd128": model_cake_fmha_aligned_hd128(),
        "cake_mok_issue_tmem": model_cake_mok_issue_tmem(),
        "cake_deepgemm_dense_mqa": model_cake_deepgemm_dense_mqa(),
        "cute_dsl_bf16_gemm_regating": model_cute_dsl_bf16_gemm_regating(),
        "cake_mega_mhc_h4096": model_cake_mega_mhc_h4096(),
        "cake_causal_conv1d_non_coop_bwd": model_cake_causal_conv1d_non_coop_bwd(),
        "tilelang_int64_atomics": model_tilelang_int64_atomics(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify summary.json matches model")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    rendered = json.dumps(summary, indent=2) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit(f"Missing {out_path}")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise SystemExit("summary.json is out of sync with run.py")
        print("Phase 59 Track A summary.json verified.")
        return 0

    out_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
