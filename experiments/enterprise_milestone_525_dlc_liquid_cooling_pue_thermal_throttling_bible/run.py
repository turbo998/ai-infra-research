from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    cooling_options = [
        {
            "option_id": "legacy_air_cooling_forced_on_high_density",
            "name": "Anti-Pattern: Forcing 80kW+ High-Density H200/B200 Racks onto Legacy Hot-Aisle Air Cooling",
            "rack_power_density_kw": 42.0,
            "facility_pue": 1.54,
            "gpu_thermal_throttle_loss_pct": 18.5,
            "racks_required_for_256_gpus": 32,
            "optical_cable_reach_penalty_pct": 42.0,
            "three_year_power_and_cooling_tco_usd": 2180000.0,
            "action": "Send to a human",
        },
        {
            "option_id": "rear_door_heat_exchanger_rdhx_hybrid",
            "name": "Transitional: Rear-Door Heat Exchanger (RDHx Chilled Water) at 45–55 kW/Rack",
            "rack_power_density_kw": 52.0,
            "facility_pue": 1.32,
            "gpu_thermal_throttle_loss_pct": 4.2,
            "racks_required_for_256_gpus": 16,
            "optical_cable_reach_penalty_pct": 18.0,
            "three_year_power_and_cooling_tco_usd": 1540000.0,
            "action": "Confirm",
        },
        {
            "option_id": "direct_to_chip_dlc_cold_plate_cdu_120kw",
            "name": "Optimal (>45kW–130kW): Direct-to-Chip Liquid Cooling (DLC Cold Plate + In-Row Redundant CDU)",
            "rack_power_density_kw": 120.0,
            "facility_pue": 1.12,
            "gpu_thermal_throttle_loss_pct": 0.0,
            "racks_required_for_256_gpus": 8,
            "optical_cable_reach_penalty_pct": 0.0,
            "three_year_power_and_cooling_tco_usd": 1190000.0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_milestone_525_dlc_liquid_cooling_pue_thermal_throttling_bible",
        "title": "Milestone #525: Enterprise AI Datacenter Direct-to-Chip Liquid Cooling (DLC >45kW–130kW/Rack), PUE & Thermal Throttling Bible",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "pue_improvement": round(1.54 - 1.12, 2),
            "three_year_power_tco_saving_pct": round((1.0 - 1190000.0 / 2180000.0) * 100, 1),
            "rack_footprint_reduction_pct": round((1.0 - 8 / 32) * 100, 1),
        },
        "cooling_options": cooling_options,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
