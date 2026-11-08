#!/usr/bin/env python3
"""Deterministic thermal, PUE, copper-vs-optics BOM, and 3-year TCO evaluator for Round 10:
AI Datacenter Rack Density, Power Delivery & Cooling Architecture Selection
(Forced-Air vs. Rear-Door Heat Exchanger RDHx vs. Single-Phase Direct-to-Chip DLC vs. Two-Phase DLC / Immersion).

Provenance Note:
- All cluster power, PUE, rack count, copper-vs-optics interconnect BOM, and 3-year TCO figures
  are deterministic synthetic calculations (`provenance_type: "synthetic"`, `independently_reproduced: false`)
  calibrated against publicly documented hardware specifications (`anonymized_reported`:
  NVIDIA DGX/HGX H100/H200 ~10.2-12.5 kW/node & GB200 NVL72 ~120-132 kW/rack reference architectures,
  OCP ORv3 48V/54V DC busbar specifications, and ASHRAE TC 9.9 liquid cooling guidelines).
- No physical 1,152-GPU datacenter hall was constructed or metered during this run (`physical_datacenter_measurement: "not_measured"`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def compute_results() -> dict[str, Any]:
    # Cluster baseline: 1,152 GPUs (16 x 72-GPU NVL72-equivalent scale-up domains)
    total_gpus = 1152
    domain_gpus = 72
    num_domains = total_gpus // domain_gpus  # 16 domains
    hours_3yr = 3 * 365 * 24  # 26,280 hours over 3 years
    electricity_rate_usd_kwh = 0.10

    # Base IT compute power (excluding server chassis fans & inter-rack optical DSP overhead):
    # 16 domains * 124.0 kW base compute/switch silicon load = 1,984.0 kW
    base_silicon_it_kw = 1984.0

    # Tier 1: Forced-Air Hot/Cold Aisle Containment (33 kW/rack max practical density)
    # Only 18 GPUs/rack -> 64 racks required. Cannot fit 72 GPUs in 1 rack (copper NVLink reach ~1.5m),
    # forcing multi-rack optical transceivers (+4.5 kW optical DSP per 72-GPU domain = +72.0 kW)
    # and high-RPM chassis fans (+15.5% of silicon IT power = +307.52 kW).
    t1_racks = 64
    t1_gpus_per_rack = 18
    t1_fan_overhead_kw = round(base_silicon_it_kw * 0.155, 2)  # 307.52 kW
    t1_optical_dsp_kw = round(num_domains * 4.5, 2)  # 72.0 kW
    t1_total_it_kw = round(base_silicon_it_kw + t1_fan_overhead_kw + t1_optical_dsp_kw, 2)  # 2,363.52 kW
    t1_kw_per_rack = round(t1_total_it_kw / t1_racks, 2)  # 36.93 kW/rack
    t1_pue = 1.48
    t1_facility_kw = round(t1_total_it_kw * t1_pue, 2)  # 3,498.01 kW
    t1_3yr_power_cost_usd = round(t1_facility_kw * hours_3yr * electricity_rate_usd_kwh, 2)  # $9,192,770.28
    # Multi-rack scale-up optical transceiver BOM penalty ($280,000 per 72-GPU domain * 16 = $4,480,000)
    t1_inter_rack_optics_bom_usd = 4480000.00
    t1_cooling_retrofit_capex_usd = 640000.00  # $10k/rack air containment & CRAC
    t1_3yr_infra_tco_usd = round(
        t1_3yr_power_cost_usd + t1_inter_rack_optics_bom_usd + t1_cooling_retrofit_capex_usd, 2
    )  # $14,312,770.28

    # Tier 2: Rear-Door Heat Exchanger (RDHx Brownfield Retrofit, 36 GPUs/rack -> 32 racks)
    # Halves rack count (64 -> 32 racks), room-neutral exhaust, lowers PUE to 1.28,
    # still requires 2 racks per 72-GPU domain (half optical BOM penalty = $2,240,000) and server fans (+11.0%).
    t2_racks = 32
    t2_gpus_per_rack = 36
    t2_fan_overhead_kw = round(base_silicon_it_kw * 0.11, 2)  # 218.24 kW
    t2_optical_dsp_kw = round(num_domains * 2.25, 2)  # 36.0 kW
    t2_total_it_kw = round(base_silicon_it_kw + t2_fan_overhead_kw + t2_optical_dsp_kw, 2)  # 2,238.24 kW
    t2_kw_per_rack = round(t2_total_it_kw / t2_racks, 2)  # 69.95 kW/rack (practical 50-65 kW HGX H200 sweet spot)
    t2_pue = 1.28
    t2_facility_kw = round(t2_total_it_kw * t2_pue, 2)  # 2,864.95 kW
    t2_3yr_power_cost_usd = round(t2_facility_kw * hours_3yr * electricity_rate_usd_kwh, 2)  # $7,529,088.60
    t2_inter_rack_optics_bom_usd = 2240000.00
    t2_cooling_retrofit_capex_usd = 1120000.00  # $35k/rack RDHx coil doors + manifolds
    t2_3yr_infra_tco_usd = round(
        t2_3yr_power_cost_usd + t2_inter_rack_optics_bom_usd + t2_cooling_retrofit_capex_usd, 2
    )  # $10,889,088.60

    # Tier 3: Single-Phase Direct-to-Chip Liquid Cooling (1-Phase DLC + CDU, 72 GPUs/rack -> 16 racks)
    # Fits a full 72-GPU NVL72 domain inside a single 132 kW rack!
    # Unlocks 100% passive copper NVLink-5 backplane ($0 inter-rack scale-up optics, 0 kW optical DSP penalty),
    # cuts server fan power to 3.5% (+69.44 kW for residual 18% air-cooled components), and reaches PUE = 1.12.
    t3_racks = 16
    t3_gpus_per_rack = 72
    t3_fan_overhead_kw = round(base_silicon_it_kw * 0.035, 2)  # 69.44 kW
    t3_optical_dsp_kw = 0.0
    t3_total_it_kw = round(base_silicon_it_kw + t3_fan_overhead_kw, 2)  # 2,053.44 kW
    t3_kw_per_rack = round(t3_total_it_kw / t3_racks, 2)  # 128.34 kW/rack (~132 kW peak NVL72 envelope)
    t3_pue = 1.12
    t3_facility_kw = round(t3_total_it_kw * t3_pue, 2)  # 2,299.85 kW
    t3_3yr_power_cost_usd = round(t3_facility_kw * hours_3yr * electricity_rate_usd_kwh, 2)  # $6,044,005.80
    t3_inter_rack_optics_bom_usd = 0.00  # 100% passive copper backplane within each rack!
    t3_cooling_retrofit_capex_usd = 1680000.00  # Cold plates, quick-disconnect manifolds, N+1 CDUs
    t3_3yr_infra_tco_usd = round(
        t3_3yr_power_cost_usd + t3_inter_rack_optics_bom_usd + t3_cooling_retrofit_capex_usd, 2
    )  # $7,724,005.80
    t3_3yr_power_savings_vs_air_usd = round(t1_3yr_power_cost_usd - t3_3yr_power_cost_usd, 2)  # $3,148,764.48
    t3_3yr_total_tco_savings_vs_air_usd = round(t1_3yr_infra_tco_usd - t3_3yr_infra_tco_usd, 2)  # $6,588,764.48

    # Tier 4: Two-Phase Direct-to-Chip / Full Dielectric Immersion Cooling (>150-240 kW/pod, 16 racks/tanks)
    # 100% liquid heat capture (0.8% pump/control overhead), PUE = 1.05, lowest power bill,
    # but higher specialty dielectric fluid & sealed tank CapEx ($3,100,000) and 2.5x longer hardware swap MTTR.
    t4_racks = 16
    t4_gpus_per_rack = 72
    t4_fan_overhead_kw = round(base_silicon_it_kw * 0.008, 2)  # 15.87 kW
    t4_optical_dsp_kw = 0.0
    t4_total_it_kw = round(base_silicon_it_kw + t4_fan_overhead_kw, 2)  # 1,999.87 kW
    t4_kw_per_rack = round(t4_total_it_kw / t4_racks, 2)  # 124.99 kW/tank (supports up to 240 kW/tank)
    t4_pue = 1.05
    t4_facility_kw = round(t4_total_it_kw * t4_pue, 2)  # 2,099.86 kW
    t4_3yr_power_cost_usd = round(t4_facility_kw * hours_3yr * electricity_rate_usd_kwh, 2)  # $5,518,432.08
    t4_inter_rack_optics_bom_usd = 0.00
    t4_cooling_retrofit_capex_usd = 3100000.00  # Dielectric fluid inventory + sealed tanks + crane tooling
    t4_3yr_infra_tco_usd = round(
        t4_3yr_power_cost_usd + t4_inter_rack_optics_bom_usd + t4_cooling_retrofit_capex_usd, 2
    )  # $8,618,432.08

    return {
        "metadata": {
            "experiment_id": "ai_datacenter_cooling_power_selection",
            "title": "AI Datacenter Rack Density, Power Delivery & Cooling Selection (Forced-Air vs. RDHx vs. Single-Phase DLC vs. Two-Phase/Immersion)",
            "provenance_type": "synthetic",
            "independently_reproduced": False,
            "physical_datacenter_measurement": "not_measured",
            "local_cpu_harness_measurement": "measured",
            "calibration_sources": [
                "NVIDIA GB200 NVL72 & HGX H100/H200 Datacenter Facility Reference Design (120-132 kW/rack liquid-cooled NVL72 with passive copper NVLink backplane)",
                "OCP Open Rack v3 (ORv3) 48V/54V High-Power Busbar & Blind-Mate Liquid Manifold Specification",
                "ASHRAE TC 9.9 Water-Cooled Servers (W3/W4 Warm-Water Cooling Classes, 30C-45C facility supply)"
            ],
            "cluster_baseline": {
                "total_gpus": total_gpus,
                "scale_up_domain_gpus": domain_gpus,
                "num_scale_up_domains": num_domains,
                "base_silicon_it_kw": base_silicon_it_kw,
                "horizon_hours_3yr": hours_3yr,
                "electricity_rate_usd_kwh": electricity_rate_usd_kwh,
                "volumetric_heat_capacity_ratio_water_vs_air": 3450,
            },
        },
        "tiers": {
            "tier1_forced_air_aisle_containment": {
                "label": "Tier 1: Forced-Air Hot/Cold Aisle Containment (<=35-40 kW/rack)",
                "racks_required": t1_racks,
                "gpus_per_rack": t1_gpus_per_rack,
                "liquid_heat_capture_pct": 0.0,
                "chassis_fan_overhead_pct": 15.5,
                "chassis_fan_overhead_kw": t1_fan_overhead_kw,
                "inter_rack_optical_dsp_kw": t1_optical_dsp_kw,
                "total_it_power_kw": t1_total_it_kw,
                "kw_per_rack": t1_kw_per_rack,
                "pue": t1_pue,
                "total_facility_power_kw": t1_facility_kw,
                "sustained_clock_performance_pct": 92.0,
                "power_cost_3yr_usd": t1_3yr_power_cost_usd,
                "inter_rack_scale_up_optics_bom_usd": t1_inter_rack_optics_bom_usd,
                "cooling_capex_usd": t1_cooling_retrofit_capex_usd,
                "total_3yr_infra_tco_usd": t1_3yr_infra_tco_usd,
                "serviceability_mttr_min": 25.0,
            },
            "tier2_rear_door_heat_exchanger_rdhx": {
                "label": "Tier 2: Rear-Door Heat Exchanger (RDHx Brownfield Retrofit, 40-65 kW/rack)",
                "racks_required": t2_racks,
                "gpus_per_rack": t2_gpus_per_rack,
                "liquid_heat_capture_pct": 0.0,
                "room_exhaust_heat_neutralized_pct": 100.0,
                "chassis_fan_overhead_pct": 11.0,
                "chassis_fan_overhead_kw": t2_fan_overhead_kw,
                "inter_rack_optical_dsp_kw": t2_optical_dsp_kw,
                "total_it_power_kw": t2_total_it_kw,
                "kw_per_rack": t2_kw_per_rack,
                "pue": t2_pue,
                "total_facility_power_kw": t2_facility_kw,
                "sustained_clock_performance_pct": 96.5,
                "power_cost_3yr_usd": t2_3yr_power_cost_usd,
                "inter_rack_scale_up_optics_bom_usd": t2_inter_rack_optics_bom_usd,
                "cooling_capex_usd": t2_cooling_retrofit_capex_usd,
                "total_3yr_infra_tco_usd": t2_3yr_infra_tco_usd,
                "serviceability_mttr_min": 30.0,
            },
            "tier3_single_phase_direct_to_chip_dlc": {
                "label": "Tier 3: Single-Phase Direct-to-Chip Liquid Cooling (1-Phase PG25 DLC + CDU, 80-140 kW/rack)",
                "racks_required": t3_racks,
                "gpus_per_rack": t3_gpus_per_rack,
                "liquid_heat_capture_pct": 82.0,
                "chassis_fan_overhead_pct": 3.5,
                "chassis_fan_overhead_kw": t3_fan_overhead_kw,
                "inter_rack_optical_dsp_kw": t3_optical_dsp_kw,
                "total_it_power_kw": t3_total_it_kw,
                "kw_per_rack": t3_kw_per_rack,
                "pue": t3_pue,
                "total_facility_power_kw": t3_facility_kw,
                "sustained_clock_performance_pct": 100.0,
                "power_cost_3yr_usd": t3_3yr_power_cost_usd,
                "inter_rack_scale_up_optics_bom_usd": t3_inter_rack_optics_bom_usd,
                "cooling_capex_usd": t3_cooling_retrofit_capex_usd,
                "total_3yr_infra_tco_usd": t3_3yr_infra_tco_usd,
                "power_savings_3yr_vs_air_usd": t3_3yr_power_savings_vs_air_usd,
                "total_tco_savings_3yr_vs_air_usd": t3_3yr_total_tco_savings_vs_air_usd,
                "total_tco_savings_pct_vs_air": round(
                    (t3_3yr_total_tco_savings_vs_air_usd / t1_3yr_infra_tco_usd) * 100.0, 1
                ),  # 46.0%
                "rack_footprint_reduction_factor": round(t1_racks / t3_racks, 1),  # 4.0x
                "serviceability_mttr_min": 35.0,
            },
            "tier4_two_phase_or_immersion_cooling": {
                "label": "Tier 4: Two-Phase Direct-to-Chip / Dielectric Immersion Cooling (150-240+ kW/pod)",
                "racks_required": t4_racks,
                "gpus_per_rack": t4_gpus_per_rack,
                "liquid_heat_capture_pct": 99.0,
                "chassis_fan_overhead_pct": 0.8,
                "chassis_fan_overhead_kw": t4_fan_overhead_kw,
                "inter_rack_optical_dsp_kw": t4_optical_dsp_kw,
                "total_it_power_kw": t4_total_it_kw,
                "kw_per_rack": t4_kw_per_rack,
                "pue": t4_pue,
                "total_facility_power_kw": t4_facility_kw,
                "sustained_clock_performance_pct": 100.0,
                "power_cost_3yr_usd": t4_3yr_power_cost_usd,
                "inter_rack_scale_up_optics_bom_usd": t4_inter_rack_optics_bom_usd,
                "cooling_capex_usd": t4_cooling_retrofit_capex_usd,
                "total_3yr_infra_tco_usd": t4_3yr_infra_tco_usd,
                "serviceability_mttr_min": 88.0,
            },
        },
        "sensitivity_analysis": {
            "w4_warm_water_40c_free_cooling_hours_pct": 94.0,
            "chiller_less_dry_cooler_pue_improvement": 0.06,
            "orv3_54v_dc_busbar_conversion_efficiency_pct": 97.5,
            "legacy_208v_ac_pdu_conversion_efficiency_pct": 91.5,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or check Round 10 AI datacenter cooling & power evaluator.")
    parser.add_argument("--check", action="store_true", help="Verify results.json matches deterministic model.")
    args = parser.parse_args()

    computed = compute_results()
    if args.check:
        if not RESULTS_PATH.exists():
            raise SystemExit(f"Missing {RESULTS_PATH}")
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if existing != computed:
            raise SystemExit("MISMATCH: experiments/ai_datacenter_cooling_power_selection/results.json is out of sync.")
        print("OK: experiments/ai_datacenter_cooling_power_selection/results.json matches deterministic model.")
        return 0

    RESULTS_PATH.write_text(json.dumps(computed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
