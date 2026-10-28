"""Phase 124 Track A: FlashInfer KDA Plan Caching / Rubin SM107 SSD Combined / Muse Spark Norm / MM FP4 2x2 Multicast / Grouped FP8 SiLU CTA-Cap / BF16-FP4 PDL Alpha / SM120 Weight-First / GDN Det Train / Softmax Balanced Chunks / mHC dY Release & TileLang Coalesced / ReduceLowerer / NonRestrict Verification."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def build_summary() -> dict[str, Any]:
    # 1. FlashInfer cake_kda_train repeated plan cache & non-blocking arena (`bd60c11809`)
    # Problem: 4 packed docs (120001, 70707, 65537, 5148) -> S = 261393, H = 4, D = 128, BF16
    kda_doc_lengths = [120001, 70707, 65537, 5148]
    kda_s_total = sum(kda_doc_lengths)
    kda_plan_cache = {
        "commit": "bd60c118092e66514e1fbd3299dddc06e44440e8",
        "s_total": kda_s_total,
        "num_docs": len(kda_doc_lengths),
        "cache_key": "(261393, 4, 4, 152, 'sm_103a')",
        "select_record_calls_before": 63,
        "select_record_calls_after": 0,
        "gb200_sm100a": {
            "plan_ms_before": 1.044,
            "plan_ms_after": 0.0016,
            "plan_speedup_x": round(1.044 / 0.0016, 2),
            "fwd_bwd_enqueue_ms_before": 3.707,
            "fwd_bwd_enqueue_ms_default_after": 1.173,
            "fwd_bwd_enqueue_ms_workspace_after": 0.969,
            "host_syncs_lean_tail4_before": 2,
            "host_syncs_lean_tail4_batched_before": 6,
            "host_syncs_after": 0,
            "host_wait_98ms_spin_lean_before_ms": 101.45,
            "host_wait_98ms_spin_lean_after_ms": 1.73,
            "host_wait_98ms_spin_batched_before_ms": 105.88,
            "host_wait_98ms_spin_batched_after_ms": 4.33,
            "peak_device_memory_gb": 6.97,
            "retained_memory_after_step_bytes": 0,
        },
        "gb300_sm103a": {
            "plan_ms_before": 1.047,
            "plan_ms_after": 0.0016,
            "fwd_bwd_enqueue_ms_before": 3.750,
            "fwd_bwd_enqueue_ms_default_after": 1.275,
            "fwd_bwd_enqueue_ms_workspace_after": 1.031,
            "host_wait_98ms_spin_lean_before_ms": 101.31,
            "host_wait_98ms_spin_lean_after_ms": 1.68,
            "host_wait_98ms_spin_batched_before_ms": 105.62,
            "host_wait_98ms_spin_batched_after_ms": 4.17,
            "default_arch_selected": "sm_103a",
        },
        "spec3_probe_modes": ["off", "record", "warn", "assert"],
    }

    # 2. FlashInfer cake_ssd_combined Rubin sm_107a 1-CTA cluster & exactpre tailone (`ecd4d71d89`)
    ssd_sm107 = {
        "commit": "ecd4d71d89be194f04195de4f9595826a75f28df",
        "arch": "sm_107a",
        "total_sm107_programs": 56,
        "one_cta_cluster_programs": 20,
        "new_tailone_programs": [
            "exactpre_bf16_varlen_tailone",
            "exactpre_f16_varlen_tailone",
            "exactpre_f32_varlen_tailone",
        ],
        "varlen_seq_table_broadcast_program": "exactpre_f32_varlen_claim",
        "shapes_evaluated": 93,
        "shapes_changed": 45,
        "shapes_unchanged": 48,
        "changed_speedup_min_x": 1.004,
        "changed_speedup_max_x": 1.084,
        "changed_speedup_geomean_x": 1.022,
        "overall_93_geomean_x": 1.011,
        "representative_rows": {
            "f32_b2_l4096_varlen_c128_h64_p64_g1_n128": {
                "before_us": 566.5,
                "after_us": 522.5,
                "speedup_x": 1.084,
            },
            "bf16_b16_l2048_varlen_c256_h64_p64_g1_n128": {
                "before_us": 621.8,
                "after_us": 579.7,
                "speedup_x": 1.073,
            },
        },
    }

    # 3. FlashInfer cake_muse_spark_norm GB300 & B200 gated residual RMSNorm & next-RMSNorm (`d9447704cb`)
    muse_spark_norm = {
        "commit": "d9447704cb55a40a1df31d917feb6e28a175dda6",
        "hidden_size": 8192,
        "records_before": 78,
        "records_after": 79,
        "libraries_before": 143,
        "libraries_after": 147,
        "gb300_pdl_highlights": {
            "mode0_m192_graph_replay": {"before_us": 3.439, "after_us": 3.027, "speedup_x": 1.1360},
            "mode0_m512_kernel_alone": {"before_us": 9.847, "after_us": 8.722, "speedup_x": 1.1289},
            "mode1_m192_graph_replay": {"before_us": 3.466, "after_us": 3.135, "speedup_x": 1.1057},
            "mode1_m512_kernel_alone": {"before_us": 9.951, "after_us": 8.936, "speedup_x": 1.1135},
            "mode1_m768_graph_replay": {"before_us": 8.970, "after_us": 8.242, "speedup_x": 1.0883},
        },
        "b200_next_rmsnorm_highlights": {
            "alpha_tensor_post_weight_m1536": {"before_us": 28.224, "after_us": 26.936, "speedup_x": 1.0478},
            "alpha_scalar_no_post_m1024": {"before_us": 18.656, "after_us": 18.016, "speedup_x": 1.0355},
        },
    }

    # 4. FlashInfer cake_mm_fp4 GB300 cluster-of-8 2x2 multicast scale-factor ring (`c9fc6cab40`)
    mm_fp4_multicast = {
        "commit": "c9fc6cab4029bd7a3998170dfda623ce0e2c7d78",
        "arch": "sm_103a",
        "sm_count": 152,
        "cluster_size": 8,
        "multicast_topology": "2x2_sfr_mc42",
        "m_band": [1793, 2048],
        "geometries_k_x_n": ["8192x8192", "7168x18432", "8192x28672"],
        "rows_retuned": 12,
        "speedup_min_x": 1.038,
        "speedup_max_x": 1.066,
        "representative_rows": {
            "m2048_n8192_k8192_nvfp4_bf16": {"before_us": 44.32, "after_us": 42.11, "speedup_x": 1.052},
            "m1920_n18432_k7168_nvfp4_bf16": {"before_us": 73.12, "after_us": 68.61, "speedup_x": 1.066},
            "m2048_n28672_k8192_nvfp4_bf16": {"before_us": 128.32, "after_us": 123.36, "speedup_x": 1.040},
        },
    }

    # 5. FlashInfer cake_grouped_fp8_fused_silu_quant CTA-cap rule & cake_blackwell_bf16_fp4 PDL alpha (`a0384efcba`, `464ca3c813`)
    grouped_fp8_and_bf16_fp4 = {
        "grouped_fp8_silu_commit": "a0384efcba47887cee9a9dfba8b34a56001199f4",
        "default_cta_cap": 128,
        "elevated_cta_caps": [144, 148, 152],
        "calibration_geometry": {"two_h": 2048, "k": 4096},
        "row15_m8192_e16_640x8_384x8": {
            "selected_cta_cap": 148,
            "two_cta_clusters": 148 // 2,
            "b200_graph_us_before": 70.86,
            "b200_graph_us_after": 66.54,
            "b200_graph_speedup_x": 1.064,
            "b200_eager_us_before": 71.66,
            "b200_eager_us_after": 66.78,
            "b200_eager_speedup_x": 1.073,
            "gb300_fast_graph_speedup_x": 1.032,
            "gb300_fast_eager_speedup_x": 1.043,
        },
        "bf16_fp4_pdl_commit": "464ca3c8139cc2de83197d1630356e1b0d4bcb19",
        "bf16_fp4_programs_total": 97,
        "bf16_fp4_programs_regenerated": 2,
        "pdl_coherent_alpha_load": "ld.global.b32 after griddepcontrol.wait",
        "shape_16_1024_1024_gb300_fp16_us": {"before": 2.824, "after": 2.804, "speedup_x": 1.007},
        "shape_16_1024_1024_b200_out_us": {"before": 2.984, "after": 2.930, "speedup_x": 1.018},
    }

    # 6. FlashInfer SM120 weight-first, GDN det train, Softmax balanced chunks, mHC dY release (`1e7a60d232`, `4cc027c249`, `f4be6e9df5`, `1fb63709b8`)
    # Softmax chunk calculation: ceil(vocab / CTAs) rounded up to 32 vs whole 2048-element tiles
    def compute_softmax_chunks(vocab: int, ctas: int) -> tuple[int, int]:
        whole_tiles_per_cta = math.ceil(math.ceil(vocab / 2048) / ctas) * 2048
        balanced_chunk = math.ceil(math.ceil(vocab / ctas) / 32) * 32
        return whole_tiles_per_cta, balanced_chunk

    sm_67x81928 = compute_softmax_chunks(81928, 8)
    sm_28x48000 = compute_softmax_chunks(48000, 16)
    sm_30x140000 = compute_softmax_chunks(140000, 16)
    sm_70x201088 = compute_softmax_chunks(201088, 16)

    # mHC dY tensor bytes at T = 16263, n = 4, C = 4096, BF16 (2 bytes) for 60 layers
    mhc_dy_single_bytes = 16263 * 4 * 4096 * 2
    mhc_dy_saved_bytes_59_layers = 59 * mhc_dy_single_bytes

    sm120_gdn_softmax_mhc = {
        "sm120_weight_first_commit": "1e7a60d2320987fd14b9e6d045e3e1b448d1f7aa",
        "sm120_kernel": "fp8-grouped-tc-w8-weight-first-v6",
        "sm120_geometry": {"hidden_size": 2048, "intermediate_size": 512, "top_k": 10},
        "gdn_det_train_commit": "4cc027c249427ee1a4b59a4bd05eb2387ab38eec",
        "gdn_b200_4x4096_fwd_bwd": {"before_ms": 0.8435, "after_ms": 0.6544, "speedup_x": 1.289},
        "gdn_gb300_3seqs_12750_fwd": {"before_ms": 0.2407, "after_ms": 0.1882, "speedup_x": 1.279},
        "softmax_balanced_commit": "f4be6e9df5003ed3f8bd5a74790ad64409cf3d68",
        "softmax_changed_shapes_count": 219,
        "softmax_changed_geomean_speedup_x": 1.045,
        "softmax_examples": {
            "67x81928_8cta": {"before_chunk": sm_67x81928[0], "after_chunk": sm_67x81928[1], "before_us": 10.37, "after_us": 9.50, "speedup_x": 1.094},
            "28x48000_16cta": {"before_chunk": sm_28x48000[0], "after_chunk": sm_28x48000[1], "before_us": 4.38, "after_us": 4.00, "speedup_x": 1.093},
            "30x140000_16cta": {"before_chunk": sm_30x140000[0], "after_chunk": sm_30x140000[1], "before_us": 9.10, "after_us": 8.45, "speedup_x": 1.078},
            "70x201088_16cta": {"before_chunk": sm_70x201088[0], "after_chunk": sm_70x201088[1], "before_us": 19.62, "after_us": 18.69, "speedup_x": 1.051},
        },
        "mhc_dy_release_commit": "1fb63709b88dc08b82fc7661f12a40602057a1c0",
        "mhc_dy_single_layer_mib": round(mhc_dy_single_bytes / (1024 * 1024), 3),
        "mhc_dy_saved_60_layers_gib": round(mhc_dy_saved_bytes_59_layers / (1024**3), 3),
    }

    # 7. TileLang T.Parallel coalesced_width, ReduceLowerer thread_bounds guard & NonRestrict (`714cbf150d`, `dbabf135f9`, `a366af801a`)
    # In dbabf135f9: block has 160 threads; gemm uses [0, 128), reduce uses [32, 160) with extent=128.
    # Unconditional run_batch allowed threads [0, 32) (32 threads) to compute unsigned underflow (tid - 32)
    # and overwrite 64 floats = 256 bytes immediately preceding reduce workspace in shared memory.
    tilelang_verification = {
        "parallel_coalesced_commit": "714cbf150d862da1d1b1b6d1649354534040a6bd",
        "reduce_bounds_commit": "dbabf135f9ca436e5fa1d1aa99c7291f30784e5e",
        "non_restrict_commit": "a366af801ade96b2de274f8245dc3cf87d7bc793",
        "coalesced_width_conversion": "tirx.convert(coalesced_width) -> IntImmNode",
        "reduce_total_threads": 160,
        "reduce_thread_offset": 32,
        "reduce_thread_extent": 128,
        "non_participating_threads": 32,
        "corrupted_floats_before_fix": 64,
        "corrupted_shared_bytes_before_fix": 64 * 4,
        "guarded_thread_interval": [32, 160],
        "preserved_restrict_suffix": "_handle",
    }

    return {
        "phase": 124,
        "track": "A",
        "question_id": "q-flashinfer-kda-plan-ssd-sm107-muse-mm-fp4-silu-bf16fp4-sm120-gdn-softmax-mhc-tilelang-phase124",
        "kda_plan_cache": kda_plan_cache,
        "ssd_sm107": ssd_sm107,
        "muse_spark_norm": muse_spark_norm,
        "mm_fp4_multicast": mm_fp4_multicast,
        "grouped_fp8_and_bf16_fp4": grouped_fp8_and_bf16_fp4,
        "sm120_gdn_softmax_mhc": sm120_gdn_softmax_mhc,
        "tilelang_verification": tilelang_verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic summary.json")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).with_name("summary.json")

    if args.check:
        if not out_path.exists():
            raise SystemExit("summary.json missing; run without --check first")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise SystemExit("summary.json mismatch")
        print("Phase 124 Track A check passed.")
        return 0

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
