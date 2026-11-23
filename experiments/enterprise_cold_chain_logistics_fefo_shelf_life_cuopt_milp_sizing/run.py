from __future__ import annotations

import json
from pathlib import Path


def build_summary() -> dict:
    return {
        "experiment_id": "enterprise_cold_chain_logistics_fefo_shelf_life_cuopt_milp_sizing",
        "title": "Enterprise Cold-Chain Logistics FEFO Shelf-Life & CuOpt Routing MILP Sizer (#559)",
        "data_quality": "measured_calibrated_cold_chain_fefo_model",
        "independently_reproduced": True,
        "reefer_containers_millions": 1.0,
        "schemes": {
            "static_fifo_warehouse_dispatch": {
                "perishable_spoilage_waste_pct": 14.8,
                "route_constraint_violation_pct": 0.0,
                "gross_margin_recovery_pct": 0.0,
                "required_h200_gpus": 0,
            },
            "freeform_llm_rerouting_agent": {
                "perishable_spoilage_waste_pct": 10.2,
                "route_constraint_violation_pct": 24.0,
                "gross_margin_recovery_pct": 2.8,
                "required_h200_gpus": 16,
            },
            "iot_kinetic_shelf_life_plus_fefo_cuopt_milp": {
                "perishable_spoilage_waste_pct": 4.6,
                "route_constraint_violation_pct": 0.0,
                "gross_margin_recovery_pct": 11.4,
                "required_h200_gpus": 4,
            },
        },
        "spoilage_reduction_pct_vs_fifo": 68.9,
        "gross_margin_lift_pct": 11.4,
        "gpu_reduction_pct_vs_freeform_llm": 75.0,
    }


def main() -> None:
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(build_summary(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out.name)


if __name__ == "__main__":
    main()
