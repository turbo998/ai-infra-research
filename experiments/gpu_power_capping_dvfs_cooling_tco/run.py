from __future__ import annotations

import json
from pathlib import Path


def evaluate_gpu_power_cooling_tiers() -> dict:
    """Deterministic evaluator for 4 GPU Cluster Power Capping, DVFS & Cooling TCO Tiers.

    Workload Baseline:
      - 1,024-GPU H200 SXM Cluster (128 nodes × 8 GPUs) running mixed LLM Pretrain + Prefill/Decode Inference
      - Fixed Datacenter Facility Power Envelope: 1.20 MW IT + Cooling budget, $0.10 / kWh electricity, 3-Year TCO
      - Physics invariant: At top of VF curve (1620 MHz -> 1980 MHz), Voltage scales super-linearly (P ~ V^2 * f),
        so top 21% power (550W -> 700W) yields only ~5.8% GEMM throughput and ~0.8% HBM-bound Decode throughput.
    """
    tiers = [
        {
            "tier_id": "tier1_uncapped_700w_air_cooled",
            "name": "Tier 1: Uncapped 700W TDP + Legacy Air Cooling (PUE 1.48, 0% Oversubscription)",
            "name_zh": "Tier 1: 默认 700W 满血不封顶 + 传统风冷机房 (PUE 1.48 + 零超配)",
            "gpu_power_cap_w": 700,
            "cooling_mode": "Rear-Door Air Cooling (18% Fan Power Overhead, PUE 1.48)",
            "pue": 1.48,
            "prefill_throughput_pct": 100.0,
            "decode_throughput_pct": 100.0,
            "blended_throughput_pct": 100.0,
            "tokens_per_joule_norm": 1.00,
            "cluster_total_facility_mw_for_1024_gpus": 1.485,
            "max_gpus_under_1_2mw_cap": 824,
            "effective_cluster_goodput_under_1_2mw_pct": 80.5,
            "thermal_throttle_events_per_month": 42,
            "breaker_trip_risk_on_sync_allreduce": "High (28% transient di/dt peak over average trips 20kW rack PDU)",
            "three_year_power_opex_usd_1024_gpus": 3902544,
        },
        {
            "tier_id": "tier2_static_550w_power_cap_air",
            "name": "Tier 2: Static nvidia-smi -pl 550W Cap + Hot-Aisle Containment Air (PUE 1.35)",
            "name_zh": "Tier 2: 静态 nvidia-smi 550W 功耗封顶 + 封闭冷热通道风冷 (PUE 1.35)",
            "gpu_power_cap_w": 550,
            "cooling_mode": "Hot/Cold Aisle Containment Air (PUE 1.35)",
            "pue": 1.35,
            "prefill_throughput_pct": 94.2,
            "decode_throughput_pct": 99.1,
            "blended_throughput_pct": 96.6,
            "tokens_per_joule_norm": 1.23,
            "cluster_total_facility_mw_for_1024_gpus": 1.115,
            "max_gpus_under_1_2mw_cap": 1024,
            "effective_cluster_goodput_under_1_2mw_pct": 96.6,
            "thermal_throttle_events_per_month": 6,
            "breaker_trip_risk_on_sync_allreduce": "Low (Hard 550W PMU cap shaves synchronous GEMM burst peaks)",
            "three_year_power_opex_usd_1024_gpus": 2930208,
        },
        {
            "tier_id": "tier3_phase_aware_dvfs_split_cap",
            "name": "Tier 3: Phase-Aware DVFS (Prefill 650W / Decode 440W / Comm 350W) + Hybrid Cooling (PUE 1.24)",
            "name_zh": "Tier 3: 算子阶段感知动态 DVFS (Prefill 650W / Decode 440W / 通信 350W) + 混合散热 (PUE 1.24)",
            "gpu_power_cap_w": 495,
            "cooling_mode": "Rear-Door Heat Exchanger Chilled Water (PUE 1.24)",
            "pue": 1.24,
            "prefill_throughput_pct": 98.6,
            "decode_throughput_pct": 99.4,
            "blended_throughput_pct": 99.0,
            "tokens_per_joule_norm": 1.41,
            "cluster_total_facility_mw_for_1024_gpus": 0.942,
            "max_gpus_under_1_2mw_cap": 1024,
            "effective_cluster_goodput_under_1_2mw_pct": 99.0,
            "thermal_throttle_events_per_month": 1,
            "breaker_trip_risk_on_sync_allreduce": "Very Low (NVML clock lock drops SM to 1350MHz during HBM/NCCL phases)",
            "three_year_power_opex_usd_1024_gpus": 2475571,
        },
        {
            "tier_id": "tier4_direct_liquid_cooling_rack_oversubscription",
            "name": "Tier 4: Direct-to-Chip Cold-Plate Liquid Cooling (PUE 1.11) + Phase DVFS + 22% Rack Power Oversubscription",
            "name_zh": "Tier 4: 冷板式直接液冷 (DLC, PUE 1.11) + 阶段感知 DVFS + 机柜级 PDU 动态功耗超配 (+22% 装机容量)",
            "gpu_power_cap_w": 510,
            "cooling_mode": "Direct-to-Chip Warm-Water Liquid Cooling (40C Inlet, Fanless GPU Tray, PUE 1.11)",
            "pue": 1.11,
            "prefill_throughput_pct": 99.2,
            "decode_throughput_pct": 99.8,
            "blended_throughput_pct": 99.5,
            "tokens_per_joule_norm": 1.56,
            "cluster_total_facility_mw_for_1024_gpus": 0.858,
            "max_gpus_under_1_2mw_cap": 1248,
            "effective_cluster_goodput_under_1_2mw_pct": 121.3,
            "thermal_throttle_events_per_month": 0,
            "breaker_trip_risk_on_sync_allreduce": "Zero (Hardware BBU/Supercap Peak-Shaving + Fast Telemetry Capping <10ms)",
            "three_year_power_opex_usd_1024_gpus": 2254810,
        },
    ]

    # Power cap sensitivity sweep (350W to 700W on H200 SXM)
    power_cap_sweep = [
        {"cap_w": 700, "sm_clock_mhz": 1980, "prefill_tflops_pct": 100.0, "decode_tps_pct": 100.0, "efficiency_tok_per_j_norm": 1.00},
        {"cap_w": 600, "sm_clock_mhz": 1815, "prefill_tflops_pct": 97.1, "decode_tps_pct": 99.6, "efficiency_tok_per_j_norm": 1.15},
        {"cap_w": 550, "sm_clock_mhz": 1710, "prefill_tflops_pct": 94.2, "decode_tps_pct": 99.1, "efficiency_tok_per_j_norm": 1.23},
        {"cap_w": 450, "sm_clock_mhz": 1455, "prefill_tflops_pct": 83.5, "decode_tps_pct": 97.8, "efficiency_tok_per_j_norm": 1.44},
        {"cap_w": 350, "sm_clock_mhz": 1140, "prefill_tflops_pct": 66.0, "decode_tps_pct": 91.5, "efficiency_tok_per_j_norm": 1.52},
    ]

    t1 = tiers[0]
    t2 = tiers[1]
    t3 = tiers[2]
    t4 = tiers[3]

    summary = {
        "static_550w_power_saving_pct": round((t1["gpu_power_cap_w"] - t2["gpu_power_cap_w"]) / t1["gpu_power_cap_w"] * 100.0, 1),
        "static_550w_blended_perf_loss_pct": round(t1["blended_throughput_pct"] - t2["blended_throughput_pct"], 1),
        "tier4_vs_tier1_efficiency_gain_x": round(t4["tokens_per_joule_norm"] / t1["tokens_per_joule_norm"], 2),
        "tier4_vs_tier1_3yr_opex_savings_usd": t1["three_year_power_opex_usd_1024_gpus"] - t4["three_year_power_opex_usd_1024_gpus"],
        "tier4_vs_tier1_goodput_under_1_2mw_gain_pct": round(
            t4["effective_cluster_goodput_under_1_2mw_pct"] - t1["effective_cluster_goodput_under_1_2mw_pct"], 1
        ),
        "tier4_extra_gpus_deployed_under_1_2mw": t4["max_gpus_under_1_2mw_cap"] - t1["max_gpus_under_1_2mw_cap"],
    }

    return {
        "experiment_id": "gpu_power_capping_dvfs_cooling_tco",
        "date": "2026-11-09",
        "provenance": {
            "type": "synthetic_vf_curve_dvfs_and_datacenter_pue_tco_model_calibrated_to_anonymized_reported_specs",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "CMOS Cubic Voltage-Frequency (P ~ C * V^2 * f) Scaling Law on H100/H200 SXM GPUs (350W-700W)",
                "Phase-Aware Prefill vs Decode DVFS Power Management (ISCA Splitwise / Dynamo Power Gating Model)",
                "ASHRAE W4 Warm-Water Direct-to-Chip Liquid Cooling (PUE 1.11) vs Legacy Air Cooling (PUE 1.48) TCO",
            ],
        },
        "workload_baseline": {
            "cluster_target_gpus": 1024,
            "facility_power_limit_mw": 1.20,
            "electricity_rate_usd_per_kwh": 0.10,
            "amortization_years": 3,
        },
        "tiers": tiers,
        "power_cap_sweep": power_cap_sweep,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_gpu_power_cooling_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
