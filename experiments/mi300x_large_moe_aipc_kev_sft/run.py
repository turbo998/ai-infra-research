"""Deterministic verification of AIPC Kev-0.8B/4B SFT/calibration/Q8_0 GGUF and Azure MI300X Large MoE BF16 vs MXFP4 bounds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compute_metrics() -> dict[str, object]:
    # 1. AIPC Kev-0.8B / 4B Counterfactual SFT & Calibration & Q8_0 GGUF
    kev_08b_base_acc = 29.6
    kev_08b_stage1_orig = 88.9
    kev_08b_stage1_cf = 11.1
    kev_08b_stage2_orig = 73.5
    kev_08b_stage2_cf = 88.9
    kev_08b_stage2_combined = round((kev_08b_stage2_orig + kev_08b_stage2_cf) / 2.0, 1)
    kev_08b_gain_pts = round(kev_08b_stage2_combined - kev_08b_base_acc, 1)

    kev_4b_base_acc = 59.4
    kev_4b_stage2_orig = 89.2
    kev_4b_stage2_cf = 92.6
    kev_4b_stage2_combined = round((kev_4b_stage2_orig + kev_4b_stage2_cf) / 2.0, 1)
    kev_4b_gain_pts = round(kev_4b_stage2_combined - kev_4b_base_acc, 1)

    # Temperature scaling T = 2.83 logit dampening check
    raw_logits = [2.65, 0.20, -1.10]
    temperature = 2.83
    raw_exp = [math.exp(z) for z in raw_logits]
    raw_top_prob = round(raw_exp[0] / sum(raw_exp), 4)
    cal_exp = [math.exp(z / temperature) for z in raw_logits]
    cal_top_prob = round(cal_exp[0] / sum(cal_exp), 4)

    ece_before = 0.112
    ece_after = 0.047
    ece_reduction_pct = round((ece_before - ece_after) / ece_before * 100.0, 1)

    # Q8_0 GGUF memory footprint model (0.8B parameters)
    q8_weight_mib = 775.0
    q8_weight_gib = round(q8_weight_mib / 1024.0, 2)
    kv_and_runtime_overhead_arm64_gib = 1.39
    snapdragon_x_elite_ram_gib = round(q8_weight_gib + kv_and_runtime_overhead_arm64_gib, 2)

    # 2. Azure AMD MI300X Large MoE BF16 vs MXFP4 Memory & Throughput
    mi300x_hbm_per_gpu_gib = 192.0
    mi300x_node_hbm_gib = mi300x_hbm_per_gpu_gib * 8

    models = {
        "muse_spark_816b_a42b": {
            "total_params_b": 816,
            "active_params_b": 42,
            "bf16_tp8_pp2_weight_gib_per_gpu": 95.1,
            "bf16_single_node_tp8_weight_gib_per_gpu": 190.1,
            "mxfp4_tp8_weight_gib_per_gpu": 60.2,
            "bf16_c1_tps": 38.9,
            "bf16_c32_tps": 336.4,
            "mxfp4_c1_tps": 6.3,
            "gsm8k_bf16_pct": 90.0,
            "gsm8k_mxfp4_pct": 40.0,
        },
        "nemotron3_ultra_550b_a55b": {
            "total_params_b": 550,
            "active_params_b": 55,
            "bf16_single_node_tp8_weight_gib_per_gpu": 137.4,
            "mxfp4_tp8_weight_gib_per_gpu": 44.1,
            "bf16_c1_tps": 43.2,
            "bf16_c32_tps": 348.6,
            "mxfp4_c1_tps": 10.9,
            "gsm8k_bf16_pct": 90.0,
            "gsm8k_mxfp4_pct": 0.0,
        },
        "inkling_975b_a41b": {
            "total_params_b": 975,
            "active_params_b": 41,
            "bf16_tp8_pp2_weight_gib_per_gpu": 113.3,
            "bf16_single_node_tp8_weight_gib_per_gpu": 226.6,
            "mxfp4_tp8_weight_gib_per_gpu": 64.6,
            "bf16_c1_tps": 35.8,
            "bf16_c32_tps": 317.4,
            "mxfp4_c1_tps": 14.3,
            "mxfp4_c32_tps": 112.5,
            "gsm8k_bf16_pct": 90.0,
            "gsm8k_mxfp4_w4a16_pct": 90.0,
        },
    }

    muse_mem_saving_pct = round(
        (1.0 - models["muse_spark_816b_a42b"]["mxfp4_tp8_weight_gib_per_gpu"]
         / models["muse_spark_816b_a42b"]["bf16_single_node_tp8_weight_gib_per_gpu"]) * 100.0,
        1,
    )
    inkling_mem_saving_pct = round(
        (1.0 - models["inkling_975b_a41b"]["mxfp4_tp8_weight_gib_per_gpu"]
         / models["inkling_975b_a41b"]["bf16_single_node_tp8_weight_gib_per_gpu"]) * 100.0,
        1,
    )

    return {
        "suite": "mi300x_large_moe_aipc_kev_sft",
        "aipc_kev_sft": {
            "kev_08b_base_acc_pct": kev_08b_base_acc,
            "kev_08b_stage1_counterfactual_acc_pct": kev_08b_stage1_cf,
            "kev_08b_stage2_combined_acc_pct": kev_08b_stage2_combined,
            "kev_08b_gain_pts": kev_08b_gain_pts,
            "kev_4b_base_acc_pct": kev_4b_base_acc,
            "kev_4b_stage2_combined_acc_pct": kev_4b_stage2_combined,
            "kev_4b_gain_pts": kev_4b_gain_pts,
            "calibration_temperature": temperature,
            "uncalibrated_top_prob": raw_top_prob,
            "calibrated_top_prob": cal_top_prob,
            "ece_reduction_pct": ece_reduction_pct,
            "q8_0_weight_gib": q8_weight_gib,
            "snapdragon_x_elite_cpu_ram_gib": snapdragon_x_elite_ram_gib,
        },
        "mi300x_large_moe": {
            "mi300x_hbm_per_gpu_gib": mi300x_hbm_per_gpu_gib,
            "mi300x_node_hbm_gib": mi300x_node_hbm_gib,
            "muse_spark_mxfp4_weight_saving_pct": muse_mem_saving_pct,
            "inkling_mxfp4_weight_saving_pct": inkling_mem_saving_pct,
            "inkling_fits_single_node_mxfp4": (
                models["inkling_975b_a41b"]["mxfp4_tp8_weight_gib_per_gpu"] < mi300x_hbm_per_gpu_gib
            ),
            "nemotron_fits_single_node_bf16": (
                models["nemotron3_ultra_550b_a55b"]["bf16_single_node_tp8_weight_gib_per_gpu"]
                < mi300x_hbm_per_gpu_gib
            ),
            "models": models,
        },
    }


def verify(payload: dict[str, object]) -> None:
    aipc = payload["aipc_kev_sft"]
    assert aipc["kev_08b_gain_pts"] == 51.6
    assert aipc["kev_4b_gain_pts"] == 31.5
    assert aipc["ece_reduction_pct"] == 58.0
    assert aipc["uncalibrated_top_prob"] > 0.80
    assert 0.45 <= aipc["calibrated_top_prob"] < 0.80
    assert 2.0 <= aipc["snapdragon_x_elite_cpu_ram_gib"] <= 3.0

    moe = payload["mi300x_large_moe"]
    assert moe["muse_spark_mxfp4_weight_saving_pct"] == 68.3
    assert moe["inkling_mxfp4_weight_saving_pct"] == 71.5
    assert moe["inkling_fits_single_node_mxfp4"] is True
    assert moe["nemotron_fits_single_node_bf16"] is True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify against committed results.json")
    args = parser.parse_args()

    payload = compute_metrics()
    verify(payload)
    out_path = Path(__file__).resolve().parent / "results.json"

    if args.check:
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        assert saved == payload, f"Mismatch between saved and computed metrics:\n{saved}\n!=\n{payload}"
        print("OK: mi300x_large_moe_aipc_kev_sft verified")
        return 0

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
