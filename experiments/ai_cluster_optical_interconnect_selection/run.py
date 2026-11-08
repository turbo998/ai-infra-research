#!/usr/bin/env python3
"""Deterministic Round 12 evaluator: AI Cluster Optical & Copper Interconnect Selection.

Compares 4 interconnect architectures across a 2,048-GPU 800G Rail-Optimized 2-Tier Leaf-Spine cluster
(4,096 bidirectional 800G links = 8,192 transceiver/cable endpoints; 3-year = 26,280-hour horizon):
  - Tier 1: Full-Reach Retimed DSP Pluggable Optics (800G OSFP DR8/2xFR4 on all 8,192 endpoints)
  - Tier 2: Hybrid In-Rack DAC/AEC Copper (4,096 Server-Leaf endpoints) + Spine DSP Pluggable Optics (4,096 Leaf-Spine endpoints)
  - Tier 3: Hybrid In-Rack DAC/AEC Copper (4,096 Server-Leaf endpoints) + Short-Reach Linear Pluggable Optics LPO (4,096 Leaf-Spine endpoints)
  - Tier 4: Next-Gen Co-Packaged Optics (CPO Silicon Photonics + ELS) + 3D MEMS Optical Circuit Switch (OCS Spine Bypass)

Provenance classification:
  - provenance_type: "synthetic" (deterministic analytical power, latency, link-flap reliability, and 3-year TCO model)
  - independently_reproduced: false (calibrated against anonymized_reported NVIDIA Quantum-X800/Spectrum-X800 CPO,
    Broadcom Bailly 51.2T CPO, OIF/OFC 800G LPO/LRO MSA, and Google TPU v4/v5p Palomar 3D MEMS OCS specifications)
  - physical_datacenter_fabric_measurement: "not_measured"
  - local_cpu_harness_measurement: "measured"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def compute_model() -> dict:
    # Cluster & TCO baseline parameters
    cluster_gpus = 2048
    nodes = 256
    gpus_per_node = 8
    port_speed_gbps = 800
    server_leaf_endpoints = 4096  # 2,048 GPUs * 2 ends (NIC + Leaf downlink)
    leaf_spine_endpoints = 4096   # 2,048 uplinks * 2 ends (Leaf uplink + Spine port)
    total_endpoints = server_leaf_endpoints + leaf_spine_endpoints  # 8,192 endpoints

    horizon_years = 3
    hours_3yr = 26280  # 3 * 365 * 24
    electricity_rate_usd_kwh = 0.10
    cooling_pue_multiplier = 1.15
    effective_power_rate_usd_kwh = round(electricity_rate_usd_kwh * cooling_pue_multiplier, 4)  # $0.115/kWh

    gpu_hourly_rate_usd = 2.50
    cluster_hourly_rate_usd = round(cluster_gpus * gpu_hourly_rate_usd, 2)  # $5,120.00/hr
    flap_rollback_restart_hours = 0.20  # 12.0 minutes average NCCL job rollback + restart penalty per unmasked flap

    def build_tier(
        tier_id: str,
        name_en: str,
        name_zh: str,
        server_leaf_tech: str,
        leaf_spine_tech: str,
        sl_power_w: float,
        ls_power_w: float,
        sl_unit_cost_usd: float,
        ls_unit_cost_usd: float,
        spine_phy_latency_ns: float,
        avg_endpoint_latency_ns: float,
        sl_annual_flap_rate_pct: float,
        ls_annual_flap_rate_pct: float,
        max_spine_reach_m: int,
        baseline_tco_usd: float | None = None,
        baseline_flaps_3yr: float | None = None,
    ) -> dict:
        total_power_kw = round(
            (server_leaf_endpoints * sl_power_w + leaf_spine_endpoints * ls_power_w) / 1000.0,
            2,
        )
        capex_usd = round(
            server_leaf_endpoints * sl_unit_cost_usd + leaf_spine_endpoints * ls_unit_cost_usd,
            2,
        )
        power_cost_3yr_usd = round(
            total_power_kw * hours_3yr * effective_power_rate_usd_kwh,
            2,
        )
        flaps_3yr = round(
            (
                server_leaf_endpoints * (sl_annual_flap_rate_pct / 100.0)
                + leaf_spine_endpoints * (ls_annual_flap_rate_pct / 100.0)
            )
            * horizon_years,
            1,
        )
        flap_downtime_hours_3yr = round(flaps_3yr * flap_rollback_restart_hours, 2)
        flap_downtime_cost_3yr_usd = round(
            flaps_3yr * flap_rollback_restart_hours * cluster_hourly_rate_usd,
            2,
        )
        total_3yr_tco_usd = round(capex_usd + power_cost_3yr_usd + flap_downtime_cost_3yr_usd, 2)

        if baseline_tco_usd is None:
            tco_savings_vs_tier1_usd = 0.0
            tco_savings_vs_tier1_pct = 0.0
        else:
            tco_savings_vs_tier1_usd = round(baseline_tco_usd - total_3yr_tco_usd, 2)
            tco_savings_vs_tier1_pct = round((tco_savings_vs_tier1_usd / baseline_tco_usd) * 100.0, 1)

        if baseline_flaps_3yr is None:
            flap_reduction_factor_x = 1.0
        else:
            flap_reduction_factor_x = round(baseline_flaps_3yr / flaps_3yr, 2)

        return {
            "tier_id": tier_id,
            "name_en": name_en,
            "name_zh": name_zh,
            "server_leaf_tech": server_leaf_tech,
            "leaf_spine_tech": leaf_spine_tech,
            "server_leaf_power_w_per_end": sl_power_w,
            "leaf_spine_power_w_per_end": ls_power_w,
            "total_interconnect_power_kw": total_power_kw,
            "spine_module_phy_latency_ns": spine_phy_latency_ns,
            "avg_endpoint_phy_latency_ns": avg_endpoint_latency_ns,
            "max_spine_reach_m": max_spine_reach_m,
            "capex_usd": capex_usd,
            "power_cost_3yr_usd": power_cost_3yr_usd,
            "expected_link_flaps_3yr": flaps_3yr,
            "flap_downtime_hours_3yr": flap_downtime_hours_3yr,
            "flap_downtime_cost_3yr_usd": flap_downtime_cost_3yr_usd,
            "total_3yr_tco_usd": total_3yr_tco_usd,
            "tco_savings_vs_tier1_usd": tco_savings_vs_tier1_usd,
            "tco_savings_vs_tier1_pct": tco_savings_vs_tier1_pct,
            "flap_reduction_factor_x": flap_reduction_factor_x,
        }

    # Tier 1: All-Optical Retimed DSP Pluggable (8,192 x 800G OSFP DR8/2xFR4)
    tier1 = build_tier(
        tier_id="tier1_all_dsp_pluggable_optics",
        name_en="Tier 1: Full-Reach Retimed DSP Pluggable Optics (All 8,192 Endpoints 800G OSFP)",
        name_zh="Tier 1：全光 DSP 重定时可插拔模块（8,192 端口全采用 800G OSFP DR8/2xFR4）",
        server_leaf_tech="800G OSFP DSP Pluggable (16.0W, $1,050)",
        leaf_spine_tech="800G OSFP DSP Pluggable (16.0W, $1,050)",
        sl_power_w=16.0,
        ls_power_w=16.0,
        sl_unit_cost_usd=1050.0,
        ls_unit_cost_usd=1050.0,
        spine_phy_latency_ns=115.0,
        avg_endpoint_latency_ns=115.0,
        sl_annual_flap_rate_pct=1.80,
        ls_annual_flap_rate_pct=1.80,
        max_spine_reach_m=500,
    )

    t1_tco = tier1["total_3yr_tco_usd"]
    t1_flaps = tier1["expected_link_flaps_3yr"]

    # Tier 2: Hybrid In-Rack DAC/AEC Copper (<=3-7m) + Spine DSP Pluggable Optics
    tier2 = build_tier(
        tier_id="tier2_inrack_copper_plus_spine_dsp",
        name_en="Tier 2: Hybrid In-Rack DAC/AEC Copper (<=7m) + Spine Retimed DSP Optics",
        name_zh="Tier 2：柜内无源/有源铜缆 DAC/AEC (<=7m) + 脊叶层 DSP 可插拔光模块",
        server_leaf_tech="800G OSFP DAC/AEC Copper (2.5W avg, $180)",
        leaf_spine_tech="800G OSFP DSP Pluggable (16.0W, $1,050)",
        sl_power_w=2.5,
        ls_power_w=16.0,
        sl_unit_cost_usd=180.0,
        ls_unit_cost_usd=1050.0,
        spine_phy_latency_ns=115.0,
        avg_endpoint_latency_ns=60.0,
        sl_annual_flap_rate_pct=0.15,
        ls_annual_flap_rate_pct=1.80,
        max_spine_reach_m=500,
        baseline_tco_usd=t1_tco,
        baseline_flaps_3yr=t1_flaps,
    )

    # Tier 3: Hybrid In-Rack DAC/AEC Copper + Short-Reach Linear Pluggable Optics (LPO <=30-50m)
    tier3 = build_tier(
        tier_id="tier3_inrack_copper_plus_spine_lpo",
        name_en="Tier 3: Hybrid In-Rack DAC/AEC Copper + DSP-Less Linear Pluggable Optics (800G LPO <=50m)",
        name_zh="Tier 3：柜内铜缆 DAC/AEC + 脊叶层去 DSP 线性直驱光模块（800G LPO <=50m）",
        server_leaf_tech="800G OSFP DAC/AEC Copper (2.5W avg, $180)",
        leaf_spine_tech="800G OSFP Linear Pluggable Optics LPO (7.5W, $720)",
        sl_power_w=2.5,
        ls_power_w=7.5,
        sl_unit_cost_usd=180.0,
        ls_unit_cost_usd=720.0,
        spine_phy_latency_ns=8.0,
        avg_endpoint_latency_ns=6.5,
        sl_annual_flap_rate_pct=0.15,
        ls_annual_flap_rate_pct=0.75,
        max_spine_reach_m=50,
        baseline_tco_usd=t1_tco,
        baseline_flaps_3yr=t1_flaps,
    )

    # Tier 4: Next-Gen Co-Packaged Optics (CPO Silicon Photonics on Cold Plate) + 3D MEMS OCS Spine Bypass
    # Note: MEMS OCS eliminates active Spine packet switch transceivers; amortized per leaf-spine endpoint equivalent:
    # 2,048 Leaf CPO uplinks (4.8W) + 2,048 passive MEMS OCS ports (0.4W equiv) => avg 2.6W per leaf_spine_endpoint slot,
    # unit cost $850/slot equiv, annual optical flap rate 0.15% (matching cold-plate ELS reliability).
    tier4 = build_tier(
        tier_id="tier4_cpo_silicon_photonics_plus_mems_ocs",
        name_en="Tier 4: Liquid-Cooled Co-Packaged Optics (CPO + ELS) + 3D MEMS Optical Circuit Switch (OCS)",
        name_zh="Tier 4：冷板共封装光学（CPO 硅光引擎 + 外置光源 ELS）+ 3D MEMS 全光交换（OCS 脊层直通）",
        server_leaf_tech="800G OSFP DAC/AEC Copper (2.5W avg, $180)",
        leaf_spine_tech="Leaf 800G CPO + ELS (4.8W) + Passive 3D MEMS OCS Spine (0.4W) -> 2.6W/slot equiv ($850/slot)",
        sl_power_w=2.5,
        ls_power_w=2.6,
        sl_unit_cost_usd=180.0,
        ls_unit_cost_usd=850.0,
        spine_phy_latency_ns=4.5,
        avg_endpoint_latency_ns=4.8,
        sl_annual_flap_rate_pct=0.15,
        ls_annual_flap_rate_pct=0.15,
        max_spine_reach_m=2000,
        baseline_tco_usd=t1_tco,
        baseline_flaps_3yr=t1_flaps,
    )

    # Sensitivity across cluster scales (512, 2,048, 8,192 GPUs)
    scale_sensitivity = [
        {
            "cluster_gpus": 512,
            "total_800g_endpoints": 2048,
            "tier1_all_dsp_3yr_tco_usd": round(t1_tco * 0.25, 2),
            "tier2_copper_dsp_3yr_tco_usd": round(tier2["total_3yr_tco_usd"] * 0.25, 2),
            "tier3_copper_lpo_3yr_tco_usd": round(tier3["total_3yr_tco_usd"] * 0.25, 2),
            "tier4_cpo_ocs_3yr_tco_usd": round(tier4["total_3yr_tco_usd"] * 0.25, 2),
            "recommended_tier": "Tier 3 (In-Rack Copper + Short-Reach 800G LPO within single 50m pod)",
        },
        {
            "cluster_gpus": 2048,
            "total_800g_endpoints": 8192,
            "tier1_all_dsp_3yr_tco_usd": t1_tco,
            "tier2_copper_dsp_3yr_tco_usd": tier2["total_3yr_tco_usd"],
            "tier3_copper_lpo_3yr_tco_usd": tier3["total_3yr_tco_usd"],
            "tier4_cpo_ocs_3yr_tco_usd": tier4["total_3yr_tco_usd"],
            "recommended_tier": "Tier 3 (<=50m compact pod) or Tier 2 (>50m multi-hall SMF); Tier 4 CPO+OCS for mission-critical zero-flap SLA",
        },
        {
            "cluster_gpus": 8192,
            "total_800g_endpoints": 32768,
            "tier1_all_dsp_3yr_tco_usd": round(t1_tco * 4.0, 2),
            "tier2_copper_dsp_3yr_tco_usd": round(tier2["total_3yr_tco_usd"] * 4.0, 2),
            "tier3_copper_lpo_3yr_tco_usd": round(tier3["total_3yr_tco_usd"] * 4.0, 2),
            "tier4_cpo_ocs_3yr_tco_usd": round(tier4["total_3yr_tco_usd"] * 4.0, 2),
            "recommended_tier": "Tier 4 (CPO + 3D MEMS OCS at 1.6T/800G: cuts 3-yr link flaps from 1,769.6 -> 147.6 across 32,768 endpoints)",
        },
    ]

    return {
        "experiment_id": "ai_cluster_optical_interconnect_selection_v1",
        "round": 12,
        "provenance": {
            "provenance_type": "synthetic",
            "independently_reproduced": False,
            "physical_datacenter_fabric_measurement": "not_measured",
            "local_cpu_harness_measurement": "measured",
            "calibration_sources": [
                "NVIDIA Quantum-X800 / Spectrum-X800 Co-Packaged Optics (TSMC COUPE silicon photonics MRM + ELS, GTC 2025/2026 reported architecture)",
                "OIF / OFC 800G-LPO & 1.6T-LRO Linear Pluggable Optics MSA power and BER specifications",
                "Google TPU v4 / v5p Palomar 3D MEMS Optical Circuit Switch (OCS) SIGCOMM/ISCA published fabric availability & power data",
            ],
        },
        "baseline_cluster": {
            "cluster_gpus": cluster_gpus,
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "port_speed_gbps": port_speed_gbps,
            "server_leaf_endpoints": server_leaf_endpoints,
            "leaf_spine_endpoints": leaf_spine_endpoints,
            "total_endpoints": total_endpoints,
            "horizon_years": horizon_years,
            "hours_3yr": hours_3yr,
            "electricity_rate_usd_kwh": electricity_rate_usd_kwh,
            "cooling_pue_multiplier": cooling_pue_multiplier,
            "effective_power_rate_usd_kwh": effective_power_rate_usd_kwh,
            "gpu_hourly_rate_usd": gpu_hourly_rate_usd,
            "cluster_hourly_rate_usd": cluster_hourly_rate_usd,
            "flap_rollback_restart_hours": flap_rollback_restart_hours,
        },
        "tiers": {
            "tier1_all_dsp_pluggable_optics": tier1,
            "tier2_inrack_copper_plus_spine_dsp": tier2,
            "tier3_inrack_copper_plus_spine_lpo": tier3,
            "tier4_cpo_silicon_photonics_plus_mems_ocs": tier4,
        },
        "scale_sensitivity": scale_sensitivity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or check Round 12 optical interconnect model.")
    parser.add_argument("--check", action="store_true", help="Verify results.json matches deterministic model.")
    args = parser.parse_args()

    computed = compute_model()
    if args.check:
        if not RESULTS_PATH.exists():
            print(f"ERROR: {RESULTS_PATH} does not exist.", file=sys.stderr)
            return 1
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if existing != computed:
            print("ERROR: results.json drifted from deterministic compute_model().", file=sys.stderr)
            return 1
        print(f"OK: {RESULTS_PATH} matches deterministic Round 12 optical interconnect model.")
        return 0

    RESULTS_PATH.write_text(json.dumps(computed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
