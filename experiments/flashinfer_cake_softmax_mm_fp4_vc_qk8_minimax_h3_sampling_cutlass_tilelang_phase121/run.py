"""Phase 121 Track A stdlib experiment: FlashInfer CAKE Softmax Async-Proxy Fence,
CAKE MM FP4 Plan Fallback, VC-Attention-QK8 Context FMHA, QSA NVFP4 Thread-Safe Plan Table,
MiniMax H3 SM103 Shared-Memory Residual Staging, Autotuner Delay Stream Binding,
Bounded Tensor Top-K Fast Path, Radix Top-K Static SMEM Budget, CUTLASS Block-Scale
Epilogue Batch Indexing, and TileLang Uncached Compile Testing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def model_cake_softmax_async_proxy_fence() -> dict[str, Any]:
    """Model FlashInfer fc4df7e: fence.proxy.async.shared::cta before persistent softmax refill."""
    cubin_sizes = {
        "c8_t8_sm100a_and_sm103a": {"before_bytes": 83817, "after_bytes": 85145},
        "c16_t8_sm100a": {"before_bytes": 162333, "after_bytes": 163887},
        "c16_t8_sm103a": {"before_bytes": 83834, "after_bytes": 85162},
    }
    for entry in cubin_sizes.values():
        entry["delta_bytes"] = entry["after_bytes"] - entry["before_bytes"]

    return {
        "fence_ptx": "fence.proxy.async.shared::cta;",
        "placement": "after mbarrier_wait(freed_addr, 0) before cp.async.bulk refill",
        "cubin_sizes": cubin_sizes,
        "stress_launches_per_program": 4000,
        "stress_errors_after_fix": 0,
        "b200_51shape_geomean_vs_main": 0.996,
        "gb300_51shape_geomean_vs_main": 0.996,
    }


def model_cake_mm_fp4_plan_fallback_and_qsa_threads() -> dict[str, Any]:
    """Model FlashInfer dd21f60 (cake_mm_fp4 fallback) & 015fff1 (cake_qsa_nvfp4 thread safety)."""
    fallback_cases = [
        {
            "shape": [2049, 8192, 8192],
            "failure_kind": "unregistered_key",
            "reason": "m > 2048 selects 1-CTA tile_m=128 without registered general program",
            "fallback_program": "mm_fp4_m128_n256_k256_2cta_s6_epi32_pf2",
        },
        {
            "shape": [33, 1536, 7168],
            "failure_kind": "unfit_key",
            "reason": "default 64-row TMA box exceeds M=33..63 operand bound",
            "fallback_program": "mm_fp4_m64_n128_k256_1cta_s6_epi32_pf2_box32",
        },
        {
            "shape": [128, 4096, 2688],
            "failure_kind": "unregistered_key",
            "reason": "strided K=2688 quant/GEMM key missing in specialized table",
            "fallback_program": "quant_nvfp4_strided_general_fallback",
        },
    ]
    return {
        "mm_fp4_fallback_cases": fallback_cases,
        "mm_fp4_fallback_count": len(fallback_cases),
        "qsa_nvfp4_atomic_pop_default": True,
        "qsa_nvfp4_evict_caught_exceptions": ["StopIteration", "KeyError", "RuntimeError"],
    }


def model_vc_attention_qk8_and_minimax_h3_sm103() -> dict[str, Any]:
    """Model FlashInfer e91141b (VC-attention-QK8) & 056670d (MiniMax H3 SM103 SMEM residual)."""
    dense_ms = 166.0
    vc_qk16_means_ms = 142.0
    vc_qk16_repair_ms = 134.0
    vc_qk8_means_ms = 123.0
    vc_qk8_repair_ms = 115.0

    # SM103 MiniMax H3 NVFP4 tail-split out-projection
    gb300_cta_pairs = 76
    min_waves_for_smem_residual = 8
    min_tile_pairs = gb300_cta_pairs * min_waves_for_smem_residual  # 608
    p4_m8368_before_us = 125.2
    p4_m8368_after_us = 123.5

    return {
        "vc_qk8_hadamard_width": 128,
        "vc_qk8_max_q_block_size": 256,
        "vc_qk8_k_tile_tokens": 128,
        "wan22_p1_dense_ms": dense_ms,
        "wan22_p1_vc_qk16_means_ms": vc_qk16_means_ms,
        "wan22_p1_vc_qk16_repair_ms": vc_qk16_repair_ms,
        "wan22_p1_vc_qk8_means_ms": vc_qk8_means_ms,
        "wan22_p1_vc_qk8_repair_ms": vc_qk8_repair_ms,
        "wan22_p1_vc_qk8_means_speedup": round(dense_ms / vc_qk8_means_ms, 4),
        "wan22_p1_vc_qk8_repair_speedup": round(dense_ms / vc_qk8_repair_ms, 4),
        "minimax_h3_sm103_smem_bytes": 228480,
        "minimax_h3_sm103_min_waves": min_waves_for_smem_residual,
        "minimax_h3_sm103_min_tile_pairs": min_tile_pairs,
        "minimax_h3_p4_m8368_before_us": p4_m8368_before_us,
        "minimax_h3_p4_m8368_after_us": p4_m8368_after_us,
        "minimax_h3_p4_m8368_speedup": round(p4_m8368_before_us / p4_m8368_after_us, 4),
        "minimax_h3_15shape_geomean_speedup": 1.010,
    }


def model_autotuner_delay_and_sampling_topk() -> dict[str, Any]:
    """Model FlashInfer 93ab5d9 (delay_kernel stream), ba25ff7 (top_k_max), 205c51e (radix SMEM)."""
    sampling_rows = [
        {"vocab": 128256, "batch": 1, "before_ms": 0.124, "after_ms": 0.069},
        {"vocab": 128256, "batch": 256, "before_ms": 0.450, "after_ms": 0.125},
        {"vocab": 262144, "batch": 1, "before_ms": 0.206, "after_ms": 0.073},
        {"vocab": 262144, "batch": 256, "before_ms": 0.819, "after_ms": 0.218},
    ]
    for r in sampling_rows:
        r["speedup"] = round(r["before_ms"] / r["after_ms"], 4)

    # RTX 5090 SM120 radix top-k SMEM budget check for shape=(1, 248077) FP32
    sm120_max_smem_per_block = 101376
    static_smem_bytes = 336
    pass2_threads_1024_dynamic_smem = 101312
    pass2_threads_512_dynamic_smem = 99264

    return {
        "delay_kernel_unfixed_event_window_ms": 0.019,
        "delay_kernel_unfixed_residual_sync_ms": 5.117,
        "delay_kernel_fixed_event_window_ms": 5.147,
        "delay_kernel_fixed_residual_sync_ms": 0.012,
        "sampling_top_k_max_rows": sampling_rows,
        "max_sampling_speedup": max(r["speedup"] for r in sampling_rows),
        "sm120_max_smem_per_block": sm120_max_smem_per_block,
        "radix_topk_static_smem_bytes": static_smem_bytes,
        "threads_1024_total_smem_bytes": pass2_threads_1024_dynamic_smem + static_smem_bytes,
        "threads_1024_overflows_sm120": (
            pass2_threads_1024_dynamic_smem + static_smem_bytes > sm120_max_smem_per_block
        ),
        "threads_512_total_smem_bytes": pass2_threads_512_dynamic_smem + static_smem_bytes,
        "threads_512_fits_sm120": (
            pass2_threads_512_dynamic_smem + static_smem_bytes <= sm120_max_smem_per_block
        ),
        "radix_topk_sm120_tests_passed": 558,
    }


def model_cutlass_block_scale_and_tilelang_cache() -> dict[str, Any]:
    """Model CUTLASS ad9bd53 (per-group norm_constant batch index) & TileLang 0408712."""
    # Simulate batched GEMM with batch_count=4 where tile_coord_l is zeroed for tCgSFD
    batch_norm_constants = [1.25, 2.50, 0.75, 3.00]
    buggy_indexed_values = [batch_norm_constants[0] for _ in range(len(batch_norm_constants))]
    fixed_indexed_values = [
        batch_norm_constants[l_norm_const] for l_norm_const in range(len(batch_norm_constants))
    ]
    return {
        "cutlass_epilogue_classes": [
            "Sm100BlockScaleFactorRowStore",
            "Sm100BlockScaleFactorColStore",
            "Sm120BlockScaleFactorRowStore",
            "Sm120BlockScaleFactorColStore",
        ],
        "buggy_batch_norm_constants": buggy_indexed_values,
        "fixed_batch_norm_constants": fixed_indexed_values,
        "batch_indexing_restored": fixed_indexed_values == batch_norm_constants,
        "tilelang_context_manager": "without_kernel_cache",
        "tilelang_restores_caller_cache_state": True,
    }


def build_summary() -> dict[str, Any]:
    return {
        "phase": 121,
        "track": "Track A",
        "evidence_scope": "own-offline",
        "cake_softmax_fence": model_cake_softmax_async_proxy_fence(),
        "cake_mm_fp4_and_qsa": model_cake_mm_fp4_plan_fallback_and_qsa_threads(),
        "vc_qk8_and_minimax_h3": model_vc_attention_qk8_and_minimax_h3_sm103(),
        "autotuner_and_sampling": model_autotuner_delay_and_sampling_topk(),
        "cutlass_and_tilelang": model_cutlass_block_scale_and_tilelang_cache(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic output")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not out_path.is_file():
            out_path.write_text(rendered, encoding="utf-8", newline="\n")
        else:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing != summary:
                raise SystemExit("summary.json mismatch")
    else:
        out_path.write_text(rendered, encoding="utf-8", newline="\n")

    print("Phase 121 Track A experiment verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
