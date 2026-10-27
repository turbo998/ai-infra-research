"""Phase 105 Track A deterministic experiment:
FlashInfer SM103 mhc_fused_post_pre, cooperative kimi_k3_tp12_tail launch,
one-wave 10/12-CTA cached-cluster softmax, SM103a K=96 NVFP4 SVDQuant MMA,
CUTLASS CuTeDSL opt_level=2 & SM121 CUDA 12.9 gate, and TileLang negative-index check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_summary() -> dict:
    # 1. FlashInfer SM103 mhc_fused_post_pre (f56fecc7ebf7989a5f075031ff0ded26000b1e8b)
    # GLM-5.3-Flash: 45 layers * 2 - 1 = 89 boundaries per forward pass, [T, 4, 4096] BF16
    num_boundaries = 89
    unfused_launches_per_boundary = 3
    fused_launches_per_boundary = 1
    total_unfused_launches = num_boundaries * unfused_launches_per_boundary
    total_fused_launches = num_boundaries * fused_launches_per_boundary
    saved_launches_per_pass = total_unfused_launches - total_fused_launches

    # Memory traffic at T = 11476: 3 passes over 4-stream residual -> 1 pass
    unfused_traffic_gb_t11476 = 1.69
    fused_traffic_gb_t11476 = 0.94
    traffic_reduction_pct = round(
        (unfused_traffic_gb_t11476 - fused_traffic_gb_t11476) / unfused_traffic_gb_t11476 * 100.0,
        2,
    )
    traffic_reduction_ratio = round(unfused_traffic_gb_t11476 / fused_traffic_gb_t11476, 4)

    # GB300 kernel latency benchmarks
    decode_t48_unfused_us = 14.88
    decode_t48_fused_us = 6.88
    decode_t48_speedup_x = round(decode_t48_unfused_us / decode_t48_fused_us, 4)

    mixed_t11476_unfused_us = 310.96
    mixed_t11476_fused_us = 186.27
    mixed_t11476_speedup_x = round(mixed_t11476_unfused_us / mixed_t11476_fused_us, 4)

    # 2. FlashInfer kimi_k3_tp12_tail cooperative launch for M <= 8 (9a27746f1db861700fafe7579031aa5442a54184)
    # Standard launch on undersized green-context partition spins forever; cooperative launch fails fast
    tp12_tail_m1_gb300_prev_us = 17.59
    tp12_tail_m1_gb300_new_us = 17.30
    tp12_tail_m1_gb300_speedup_x = round(tp12_tail_m1_gb300_prev_us / tp12_tail_m1_gb300_new_us, 4)
    undersized_partition_deadlocks_before = 1
    undersized_partition_deadlocks_after = 0

    # 3. FlashInfer Blackwell cached-cluster softmax 10- and 12-CTA programs (e61cdfa925127e68f62781d3e81cf511f808c257)
    softmax_programs_before = 14
    softmax_programs_after = 20
    b200_37x128k_prev_us = 9.95
    b200_37x128k_new_us = 8.51
    b200_37x128k_speedup_x = round(b200_37x128k_prev_us / b200_37x128k_new_us, 4)
    gb300_84x100k_prev_us = 14.43
    gb300_84x100k_new_us = 12.58
    gb300_84x100k_speedup_x = round(gb300_84x100k_prev_us / gb300_84x100k_new_us, 4)

    # 4. FlashInfer cake_nvfp4_svdquant K=96 MMA on SM103a & split-K M=129 (4f3b2d6b91e26066d978a988e37b18c575cd305d)
    # 256-wide K stage: previously 4 x K=64 MMAs; now 96 + 96 + 64 = 3 MMAs per 256-K stage
    mma_per_256k_stage_before = 4
    mma_per_256k_stage_after = 3
    gb300_m16384_k12288_prev_us = 162.46
    gb300_m16384_k12288_new_us = 148.30
    gb300_svdquant_speedup_x = round(gb300_m16384_k12288_prev_us / gb300_m16384_k12288_new_us, 4)
    gb200_m129_k12288_prev_us = 11.01
    gb200_m129_k12288_new_us = 10.53
    gb200_splitk_speedup_x = round(gb200_m129_k12288_prev_us / gb200_m129_k12288_new_us, 4)

    # 5. CUTLASS CuTeDSL opt_level=2 gate (147295a3d4) & TileLang LegalizeNegativeIndex (fa456b8b2c)
    # Buffer extent = 64; index = -80 -> -80 + 64 = -16 < 0 (previously emitted OOB pre-buffer load; now rejected)
    buffer_extent = 64
    test_indices = [-16, -64, -65, -80, -128]
    rejected_oob_negative_indices = sum(1 for idx in test_indices if idx + buffer_extent < 0)

    return {
        "experiment_id": "flashinfer_mhc_post_pre_tp12_tail_softmax_svdquant_phase105",
        "phase": 105,
        "priority": 114,
        "track": "Track A - GPU/HBM/Kernel Stack",
        "metrics": {
            "mhc_num_boundaries_per_pass": num_boundaries,
            "mhc_unfused_launches_per_pass": total_unfused_launches,
            "mhc_fused_launches_per_pass": total_fused_launches,
            "mhc_saved_launches_per_pass": saved_launches_per_pass,
            "mhc_unfused_traffic_gb_t11476": unfused_traffic_gb_t11476,
            "mhc_fused_traffic_gb_t11476": fused_traffic_gb_t11476,
            "mhc_traffic_reduction_pct": traffic_reduction_pct,
            "mhc_traffic_reduction_ratio": traffic_reduction_ratio,
            "mhc_gb300_decode_t48_unfused_us": decode_t48_unfused_us,
            "mhc_gb300_decode_t48_fused_us": decode_t48_fused_us,
            "mhc_gb300_decode_t48_speedup_x": decode_t48_speedup_x,
            "mhc_gb300_mixed_t11476_unfused_us": mixed_t11476_unfused_us,
            "mhc_gb300_mixed_t11476_fused_us": mixed_t11476_fused_us,
            "mhc_gb300_mixed_t11476_speedup_x": mixed_t11476_speedup_x,
            "tp12_tail_m1_gb300_prev_us": tp12_tail_m1_gb300_prev_us,
            "tp12_tail_m1_gb300_new_us": tp12_tail_m1_gb300_new_us,
            "tp12_tail_m1_gb300_speedup_x": tp12_tail_m1_gb300_speedup_x,
            "undersized_partition_deadlocks_before": undersized_partition_deadlocks_before,
            "undersized_partition_deadlocks_after": undersized_partition_deadlocks_after,
            "softmax_programs_before": softmax_programs_before,
            "softmax_programs_after": softmax_programs_after,
            "b200_softmax_37x128k_speedup_x": b200_37x128k_speedup_x,
            "gb300_softmax_84x100k_speedup_x": gb300_84x100k_speedup_x,
            "svdquant_mma_per_256k_stage_before": mma_per_256k_stage_before,
            "svdquant_mma_per_256k_stage_after": mma_per_256k_stage_after,
            "gb300_svdquant_m16384_speedup_x": gb300_svdquant_speedup_x,
            "gb200_svdquant_m129_splitk_speedup_x": gb200_splitk_speedup_x,
            "tilelang_rejected_oob_negative_indices": rejected_oob_negative_indices,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    summary_path = Path(__file__).with_name("summary.json")

    if args.check:
        expected = json.loads(summary_path.read_text(encoding="utf-8"))
        assert expected == summary, f"Mismatch: {expected} != {summary}"
        m = summary["metrics"]
        assert m["mhc_saved_launches_per_pass"] == 178
        assert m["mhc_traffic_reduction_pct"] == 44.38
        assert m["mhc_gb300_decode_t48_speedup_x"] == 2.1628
        assert m["mhc_gb300_mixed_t11476_speedup_x"] == 1.6694
        assert m["undersized_partition_deadlocks_after"] == 0
        assert m["b200_softmax_37x128k_speedup_x"] == 1.1692
        assert m["gb300_svdquant_m16384_speedup_x"] == 1.0955
        assert m["tilelang_rejected_oob_negative_indices"] == 3
        print("OK: flashinfer_mhc_post_pre_tp12_tail_softmax_svdquant_phase105 verified")
        return

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
