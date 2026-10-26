#!/usr/bin/env python3
"""Phase 89 Track A offline model:
FlashInfer NVFP4 MLA Short-Unit Decode & Split-KV Combine, Batched DeepGEMM FP8,
GDN Training Fused dQ/dK/dG, SM90 VSA Transposer Retuning, PrimTS Keeps Softmax
& 2-CTA MLA Page-ID Window Prefetch, SM107 Split-K=8 BF16 GEMM, Kimi K3 Held-Release,
and TileLang Bulk TMA Cache Hints / NVRTC TVM-FFI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_summary() -> dict:
    # 1. cake_nvfp4_mla_decode (#7418): short-unit decode (<= 4096 on SM100, <= 8192 on SM103)
    # and general split-KV combine (>= 3 splits per row)
    nvfp4_mla = {
        "sm100_short_unit_threshold_tokens": 4096,
        "sm103_short_unit_threshold_tokens": 8192,
        "general_combine_min_splits": 3,
        "gb300_3split_before_us": 12.53,
        "gb300_3split_after_us": 11.30,
        "gb300_3split_speedup": round(12.53 / 11.30, 3),
        "gb300_2split_before_us": 11.94,
        "gb300_2split_after_us": 11.20,
        "gb300_2split_speedup": round(11.94 / 11.20, 3),
        "b200_3split_before_us": 13.04,
        "b200_3split_after_us": 11.68,
        "b200_3split_speedup": round(13.04 / 11.68, 3),
        "b200_2split_before_us": 12.32,
        "b200_2split_after_us": 11.71,
        "b200_2split_speedup": round(12.32 / 11.71, 3),
    }

    # 2. cake_batch_deepgemm_fp8 (#7425): masked batched FP8 GEMM on SM100/SM103 across 164 shapes
    batch_deepgemm_fp8 = {
        "total_shapes_evaluated": 164,
        "b200_faster_shapes": 51,
        "gb300_1_faster_shapes": 54,
        "gb300_2_faster_shapes": 51,
        "slower_shapes_any_gpu": 0,
        "gb300_b128_m128_n512_k128_before_us": 49.41,
        "gb300_b128_m128_n512_k128_after_us": 13.86,
        "gb300_b128_m128_n512_k128_ratio": round(13.86 / 49.41, 4),
        "gb300_b128_m128_n512_k128_speedup": round(49.41 / 13.86, 3),
        "b200_b128_m128_n512_k128_before_us": 34.62,
        "b200_b128_m128_n512_k128_after_us": 14.11,
        "b200_b128_m128_n512_k128_speedup": round(34.62 / 14.11, 3),
        "b200_sglang_p10_b32_m256_n4096_k7168_before_us": 5.18,
        "b200_sglang_p10_b32_m256_n4096_k7168_after_us": 2.46,
        "b200_sglang_p10_b32_m256_n4096_k7168_speedup": round(5.18 / 2.46, 3),
    }

    # 3. cake_gdn (#7423): Gated DeltaNet training forward prep, tape stores, fused dq/dk/dg
    gdn_train = {
        "total_benchmark_rows": 60,
        "dynamic_smem_before_bytes": 91136,
        "dynamic_smem_after_bytes": 99328,
        "gb300_forward_geomean_speedup": 1.064,
        "gb300_train_step_geomean_speedup": 1.042,
        "gb200_forward_geomean_speedup": 1.067,
        "gb200_train_step_geomean_speedup": 1.043,
        "checked_mul_int64_overflow_guard": True,
    }

    # 4. cake_vsa_sm90 (#7412): FP8 grouped-query compact and shared-transposer queue retuning
    vsa_sm90 = {
        "compact_l2_prefetch_distance_before": 2,
        "compact_l2_prefetch_distance_after": 1,
        "v_transposer_ldmatrix_chunks_before": 4,
        "v_transposer_ldmatrix_chunks_after": 1,
        "queue_l2_prefetch_distance_after": 0,
        "g8_before_us": 51.824,
        "g8_after_us": 49.824,
        "g8_speedup": round(51.824 / 49.824, 4),
    }

    # 5. prims-ts Keeps softmax chain & grouped-Q paged decode (#6937)
    prims_ts_keeps = {
        "bf16_pv_k32_fragments_per_row": 4,
        "dense_bf16_b200_before_mcycles": 336.6,
        "dense_bf16_b200_after_mcycles": 261.9,
        "dense_bf16_b200_speedup": round(336.6 / 261.9, 2),
        "dense_bf16_gb300_before_mcycles": 293.6,
        "dense_bf16_gb300_after_mcycles": 206.5,
        "dense_bf16_gb300_speedup": round(293.6 / 206.5, 2),
        "paged_dense_bf16_hq8_sq256_b64_b200_before_us": 26372.0,
        "paged_dense_bf16_hq8_sq256_b64_b200_after_us": 4786.0,
        "paged_dense_bf16_hq8_sq256_b64_b200_speedup": round(26372.0 / 4786.0, 2),
        "paged_dense_bf16_hq8_sq256_b64_gb300_before_us": 25797.0,
        "paged_dense_bf16_hq8_sq256_b64_gb300_after_us": 4626.0,
        "paged_dense_bf16_hq8_sq256_b64_gb300_speedup": round(25797.0 / 4626.0, 2),
    }

    # 6. prims-ts 2-CTA dense FP8 MLA K/V page-ID window prefetch (#7019)
    prims_ts_mla_2cta = {
        "warp_window_entries": 32,
        "k_tile_tokens": 128,
        "page16_b128_k8192_before_us": 339.73,
        "page16_b128_k8192_after_us": 141.71,
        "page16_b128_k8192_speedup": round(339.73 / 141.71, 3),
        "page32_b128_k8192_before_us": 198.31,
        "page32_b128_k8192_after_us": 135.78,
        "page32_b128_k8192_speedup": round(198.31 / 135.78, 3),
        "qk_mma_wait_k_stall_before_pct": 49.0,
        "k_loader_pagetable_stall_before_pct": 19.0,
    }

    # 7. SM107 BF16 cluster split-K=8 GEMM (#7064) & 8. Kimi K3 held-release (#7411)
    # 9-10. TileLang TMA cache hints (#3683) & NVRTC TVM-FFI (#3285)
    sm107_kimi_tilelang = {
        "sm107_supported_split_k": [1, 2, 3, 4, 8],
        "sm107_max_ab_stages_before": 12,
        "sm107_max_ab_stages_after": 18,
        "sm107_pooler_tp4_m8_before_us": 4.94,
        "sm107_pooler_tp4_m8_after_us": 4.61,
        "sm107_pooler_tp4_m8_speedup": round(4.94 / 4.61, 3),
        "kimi_k3_gb300_m1024_k5_pdl_on_before_us": 25.888,
        "kimi_k3_gb300_m1024_k5_pdl_on_after_us": 25.248,
        "kimi_k3_gb300_m1024_k5_pdl_on_ratio": 1.0236,
        "tilelang_tma_eviction_policies": ["evict_normal", "evict_first", "evict_last"],
        "tilelang_nvrtc_cache_key_includes_compiler_version": True,
    }

    return {
        "experiment_id": "flashinfer_nvfp4_mla_deepgemm_gdn_vsa_prims_sm107_tilelang_phase89",
        "phase": 89,
        "priority": 98,
        "track": "A",
        "nvfp4_mla_decode": nvfp4_mla,
        "batch_deepgemm_fp8": batch_deepgemm_fp8,
        "gdn_train": gdn_train,
        "vsa_sm90": vsa_sm90,
        "prims_ts_keeps": prims_ts_keeps,
        "prims_ts_mla_2cta": prims_ts_mla_2cta,
        "sm107_kimi_tilelang": sm107_kimi_tilelang,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic summary.json")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit(f"Missing {out_path}")
        existing = out_path.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit(f"Drift detected in {out_path}")
        print("verified")
        return 0

    out_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
