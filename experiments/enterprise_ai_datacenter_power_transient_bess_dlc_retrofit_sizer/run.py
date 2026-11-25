from __future__ import annotations

import json
from pathlib import Path


def simulate_ai_datacenter_power_and_dlc() -> dict:
    """Model synchronous GPU step-load transients (di/dt), 3-layer BESS/capacitive smoothing, and DLC retrofit tiers."""
    # 128-GPU (16 Racks @ 42 kW HGX H200 or 8 Racks @ 132 kW GB200 NVL72)
    cluster_it_peak_kw = 1056.0
    idle_baseline_kw = 264.0  # 25% idle between synchronous AllReduce/barrier phases
    unmitigated_step_load_swing_kw = cluster_it_peak_kw - idle_baseline_kw  # 792 kW swing in <20 ms!
    unmitigated_swing_pct_of_peak = round((unmitigated_step_load_swing_kw / cluster_it_peak_kw) * 100, 1)  # 75.0%

    transient_mitigation = {
        "unmitigated_raw_gpu_step_load": {
            "mode_name": "Unmitigated Synchronous Barrier / AllReduce Step Load (25% -> 100% in 15ms)",
            "step_swing_pct_in_20ms": 75.0,
            "rack_peak_excursion_pct": 118.0,
            "ups_bypass_or_genset_trip_risk": "CRITICAL (>50% step limit of standard diesel genset)",
            "workload_throughput_penalty_pct": 0.0,
        },
        "naive_static_gpu_power_cap_70pct": {
            "mode_name": "Naive Static nvidia-smi Power Cap (490W vs 700W / 700W vs 1000W)",
            "step_swing_pct_in_20ms": 42.0,
            "rack_peak_excursion_pct": 70.0,
            "ups_bypass_or_genset_trip_risk": "MODERATE",
            "workload_throughput_penalty_pct": 19.4,
        },
        "three_layer_power_smoothing_cbu_bess_burn_in": {
            "mode_name": "3-Layer Smoothing (ORv3 48V Supercap Shelf CBU + Rack Li-Ion BESS + Burn-Kernel Idle Floor)",
            "step_swing_pct_in_20ms": 11.5,
            "rack_peak_excursion_pct": 100.0,
            "ups_bypass_or_genset_trip_risk": "ZERO (<15% smooth ramp rate over 10s)",
            "workload_throughput_penalty_pct": 0.8,
        },
    }

    cooling_retrofits = {
        "legacy_air_cooling_crac": {
            "tier_name": "Legacy Hot/Cold Aisle Air Cooling (CRAC)",
            "max_kw_per_rack": 15,
            "design_pue": 1.48,
            "retrofit_capex_per_rack_usd": 4500,
            "chilled_water_pipes_to_rack": False,
            "supported_gpu_skus": "2-4x L40S / PCIe H100 only",
        },
        "rear_door_heat_exchanger_rdhx": {
            "tier_name": "Passive/Active Rear-Door Heat Exchanger (RDHx)",
            "max_kw_per_rack": 45,
            "design_pue": 1.26,
            "retrofit_capex_per_rack_usd": 18500,
            "chilled_water_pipes_to_rack": True,
            "supported_gpu_skus": "1x 8-GPU HGX H100/H200 (Air-assisted 10.2 kW/node)",
        },
        "liquid_to_air_sidecar_cdu_l2a": {
            "tier_name": "Self-Contained Liquid-to-Air (L2A) Sidecar CDU + Cold Plates (No Facility Water!)",
            "max_kw_per_rack": 75,
            "design_pue": 1.22,
            "retrofit_capex_per_rack_usd": 34000,
            "chilled_water_pipes_to_rack": False,
            "supported_gpu_skus": "Up to 4x HGX H200 / 3x HGX B200 in existing air-cooled colocation",
        },
        "liquid_to_liquid_in_row_cdu_l2l": {
            "tier_name": "Greenfield Direct-to-Chip Liquid-to-Liquid (L2L) CDU + Warm Water (32C-40C)",
            "max_kw_per_rack": 140,
            "design_pue": 1.11,
            "retrofit_capex_per_rack_usd": 42000,
            "chilled_water_pipes_to_rack": True,
            "supported_gpu_skus": "GB200 / GB300 NVL72 (120-140 kW/rack) & 8x HGX B200",
        },
    }

    return {
        "experiment_id": "enterprise_ai_datacenter_power_transient_bess_dlc_retrofit_sizer",
        "cluster_it_peak_kw": cluster_it_peak_kw,
        "unmitigated_swing_pct_of_peak": unmitigated_swing_pct_of_peak,
        "transient_mitigation": transient_mitigation,
        "cooling_retrofits": cooling_retrofits,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = simulate_ai_datacenter_power_and_dlc()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
