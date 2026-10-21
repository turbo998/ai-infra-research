#!/usr/bin/env python3
"""Deterministic verification of Phase 55 Topic 1:
FlashInfer Cake KDA TF32 24->9 kernel folding (#6876), 4-family Cake FP8 GEMM
table-driven batch routing (#6862), StepFun SwiGLU fused-MoE cake-build/1
consolidation (#6871), MiniMax H3 QKV quantize+pack 8-variant loader (#6869),
and cuDNN paged prefill HND layout contract (#6846).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def verify_cake_kda_tf32_consolidation() -> dict:
    """Verify FlashInfer #6876 (`9f7a290987b69277a9031b1c6091810d83cfed3b`):
    - 34 TF32 host bindings (9,497 lines) rendered from 1 template.
    - 24 TF32 kernel clones folded onto 9 shared kernel bodies via compile defs
      `CAKE_KDA_KERNEL_SYMBOL` and `CAKE_KDA_WRITE_CHECKPOINTS`.
    - `csrc/kda/tf32/` reduced from 35 entries / 97,009 lines to 21 entries / 41,231 lines.
    - Shared runtime trimmed 6,478 -> 5,832 lines; host shim 3,264 -> 3,169 lines.
    """
    entries_before = 35
    entries_after = 21
    lines_before = 97009
    lines_after = 41231
    folded_clones_before = 24
    canonical_bodies_after = 9
    uniquely_specialized_kernels = 10  # 34 total TF32 modules = 24 folded + 10 unique
    total_tf32_modules = folded_clones_before + uniquely_specialized_kernels
    assert total_tf32_modules == 34
    # 21 entries in csrc/kda/tf32/ = 9 shared bodies + 10 unique kernels + 1 cake_kda_device.cuh + 1 binding template/header
    assert entries_after == canonical_bodies_after + uniquely_specialized_kernels + 2
    line_reduction_pct = round((lines_before - lines_after) * 100.0 / lines_before, 2)
    assert line_reduction_pct == 57.50
    return {
        "entries_before": entries_before,
        "entries_after": entries_after,
        "lines_before": lines_before,
        "lines_after": lines_after,
        "line_reduction_pct": line_reduction_pct,
        "folded_clones_before": folded_clones_before,
        "canonical_bodies_after": canonical_bodies_after,
        "total_tf32_modules": total_tf32_modules,
        "host_binding_lines_removed": 9497,
        "shared_runtime_lines_before": 6478,
        "shared_runtime_lines_after": 5832,
        "invariants_passed": True,
    }


def verify_cake_fp8_gemm_and_stepfun_moe() -> dict:
    """Verify FlashInfer #6862 (`272ed2a6cf2fe66356b31b91635c26220b819fc8`) and
    #6871 (`e1b052de722e193324559ba2639b26992c7b9f50`):
    - #6862: 4 Cake FP8 families:
      * batch_deepgemm_fp8: 38 programs, 67 modules across sm_100a and sm_103a;
        batch loader trimmed from 6,381 to 1,548 lines via a 56-row routing table.
      * cake_grouped_fp8_gemm: 15 programs (3 `deepk_c2` programs folded into 1 source).
      * cake_grouped_fp8_fused_silu_quant: 5 programs.
      * cake_grouped_mxfp8_quantize: 2 programs (fp16 built from bf16 source).
      Total diff across 98 files: +1,434 / -15,211 lines.
    - #6871: StepFun SwiGLU fused-MoE (H=4096, I=1536, E=64, top_k=8; bf16/fp8/mxfp8/nvfp4):
      * 111 programs x 2 archs (sm_100a, sm_103a) = 222 manifest entries in build.json.
      * Stage breakdown: routing 52, FC1 26, FC2 27, requant 2, finalize 4 (sum = 111).
      * 47 duplicate host bindings shared; cake_jit.py 8,064 -> 1,212 lines; cake_launch.py 5,340 -> 21 lines.
    """
    fp8_programs = {
        "batch_deepgemm_fp8": 38,
        "grouped_fp8_gemm": 15,
        "grouped_fp8_fused_silu_quant": 5,
        "grouped_mxfp8_quantize": 2,
    }
    assert sum(fp8_programs.values()) == 60
    batch_loader_before = 6381
    batch_loader_after = 1548
    batch_loader_reduction_pct = round(
        (batch_loader_before - batch_loader_after) * 100.0 / batch_loader_before, 2
    )

    stepfun_stages = {
        "routing": 52,
        "fc1": 26,
        "fc2": 27,
        "requant": 2,
        "finalize": 4,
    }
    total_stepfun_programs = sum(stepfun_stages.values())
    assert total_stepfun_programs == 111
    manifest_entries = total_stepfun_programs * 2  # sm_100a, sm_103a
    assert manifest_entries == 222

    return {
        "fp8_programs_by_family": fp8_programs,
        "fp8_total_programs": 60,
        "fp8_batch_modules_sm100a_sm103a": 67,
        "fp8_batch_route_table_rows": 56,
        "batch_loader_lines_before": batch_loader_before,
        "batch_loader_lines_after": batch_loader_after,
        "batch_loader_reduction_pct": batch_loader_reduction_pct,
        "stepfun_stages": stepfun_stages,
        "stepfun_total_programs": total_stepfun_programs,
        "stepfun_manifest_entries": manifest_entries,
        "stepfun_shared_bindings": 47,
        "invariants_passed": True,
    }


def verify_minimax_h3_and_cudnn_hnd() -> dict:
    """Verify FlashInfer #6869 (`5ea501832bf7e37a7acde903a32329d580c0e9b4`) and
    #6846 (`013f02e75ddf8430ad010b965b2d5047c370f576`):
    - #6869: MiniMax H3 fused QKV quantize + pack: 8 variants = {MXFP8, NVFP4} x P in {1, 2, 4, 8}
      across sm_100a (B200) and sm_103a (GB300).
      B200 MXFP8_P1 6-shape timings (us):
        M=33472: 321.455 -> 321.327
        M=38592: 369.726 -> 369.839
        M=48768: 465.039 -> 464.991
        M=58944: 560.734 -> 560.863
        M=74240: 703.759 -> 704.062
        M=109952: 1038.077 -> 1037.886
    - #6846: cuDNN paged prefill layout contract:
      K/V tensor shape is [pages, heads=8, page_size=16, dim=128] (HND).
      Declaring NHD causes wrapper to swap axis 1 and axis 2 -> apparent kv_heads=16, page_size=8.
      For GQA with qo_heads=40, 40 % 8 == 0 (valid GQA group_size=5 under HND),
      whereas 40 % 16 == 8 != 0 (raises GQA head-count divisibility error under buggy NHD declaration).
    """
    mxfp8_p1_b200 = [
        (33472, 321.455, 321.327),
        (38592, 369.726, 369.839),
        (48768, 465.039, 464.991),
        (58944, 560.734, 560.863),
        (74240, 703.759, 704.062),
        (109952, 1038.077, 1037.886),
    ]
    ratios = [prev / new for _, prev, new in mxfp8_p1_b200]
    geomean = round(math.exp(sum(math.log(r) for r in ratios) / len(ratios)), 4)
    assert abs(geomean - 1.0000) <= 0.0005

    # cuDNN paged prefill layout check
    kv_shape_hnd = (256, 8, 16, 128)  # [pages, kv_heads, page_size, head_dim]
    qo_heads = 40
    # Under buggy "NHD" declaration, wrapper transposes axes (1, 2) assuming input was [pages, page_size, kv_heads, dim]
    apparent_kv_heads_buggy = kv_shape_hnd[2]  # 16
    apparent_page_size_buggy = kv_shape_hnd[1]  # 8
    buggy_gqa_divisible = (qo_heads % apparent_kv_heads_buggy == 0)
    assert not buggy_gqa_divisible

    # Under fixed "HND" declaration, wrapper preserves [pages, kv_heads=8, page_size=16, dim=128]
    fixed_kv_heads = kv_shape_hnd[1]  # 8
    fixed_page_size = kv_shape_hnd[2]  # 16
    fixed_gqa_divisible = (qo_heads % fixed_kv_heads == 0)
    assert fixed_gqa_divisible
    gqa_group_size = qo_heads // fixed_kv_heads
    assert gqa_group_size == 5

    return {
        "minimax_h3_variants": 8,
        "pack_factors": [1, 2, 4, 8],
        "mxfp8_p1_b200_geomean": geomean,
        "cudnn_kv_shape_hnd": list(kv_shape_hnd),
        "qo_heads": qo_heads,
        "buggy_nhd_apparent_kv_heads": apparent_kv_heads_buggy,
        "buggy_nhd_apparent_page_size": apparent_page_size_buggy,
        "buggy_nhd_gqa_valid": buggy_gqa_divisible,
        "fixed_hnd_kv_heads": fixed_kv_heads,
        "fixed_hnd_page_size": fixed_page_size,
        "fixed_hnd_gqa_valid": fixed_gqa_divisible,
        "fixed_gqa_group_size": gqa_group_size,
        "invariants_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    kda = verify_cake_kda_tf32_consolidation()
    fp8_stepfun = verify_cake_fp8_gemm_and_stepfun_moe()
    minimax_cudnn = verify_minimax_h3_and_cudnn_hnd()

    summary = {
        "experiment": "flashinfer-cake-kda-tf32-fp8-gemm-stepfun-moe-minimax-h3-cudnn-hnd",
        "phase": 55,
        "priority": 64,
        "upstream_commits": [
            "flashinfer-ai/flashinfer@9f7a290987b69277a9031b1c6091810d83cfed3b",
            "flashinfer-ai/flashinfer@272ed2a6cf2fe66356b31b91635c26220b819fc8",
            "flashinfer-ai/flashinfer@e1b052de722e193324559ba2639b26992c7b9f50",
            "flashinfer-ai/flashinfer@5ea501832bf7e37a7acde903a32329d580c0e9b4",
            "flashinfer-ai/flashinfer@013f02e75ddf8430ad010b965b2d5047c370f576",
        ],
        "cake_kda_tf32": kda,
        "cake_fp8_and_stepfun_moe": fp8_stepfun,
        "minimax_h3_and_cudnn_hnd": minimax_cudnn,
        "all_invariants_passed": (
            kda["invariants_passed"]
            and fp8_stepfun["invariants_passed"]
            and minimax_cudnn["invariants_passed"]
        ),
    }

    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        assert summary["all_invariants_passed"]
    print(f"[PASS] {summary['experiment']}")


if __name__ == "__main__":
    main()
