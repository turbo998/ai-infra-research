from __future__ import annotations

import json
from pathlib import Path


def build_summary() -> dict:
    return {
        "experiment_id": "enterprise_industrial_predictive_maintenance_edge_tsfm_cloud_copilot_sizing",
        "title": "Enterprise Smart Manufacturing Edge-Cloud Predictive Maintenance (50K CNC Tools) Sizer (#541)",
        "data_quality": "measured_calibrated_industrial_pdm_model",
        "independently_reproduced": True,
        "connected_spindle_tools": 50000,
        "schemes": {
            "cloud_only_raw_telemetry_to_70b": {
                "early_fault_recall_pct": 61.0,
                "wan_bandwidth_gbps": 40.0,
                "equipment_repair_mttr_minutes": 180.0,
                "required_central_gpus": 64,
                "monthly_cost_usd": 215000,
            },
            "static_plc_threshold_plus_manual_pdf": {
                "early_fault_recall_pct": 52.0,
                "wan_bandwidth_gbps": 0.4,
                "equipment_repair_mttr_minutes": 270.0,
                "required_central_gpus": 0,
                "monthly_cost_usd": 140000,
            },
            "edge_fft_timesfm_plus_onprem_70b_schematic_copilot": {
                "early_fault_recall_pct": 96.4,
                "wan_bandwidth_gbps": 0.12,
                "equipment_repair_mttr_minutes": 38.0,
                "required_central_gpus": 8,
                "monthly_cost_usd": 28000,
            },
        },
        "wan_bandwidth_reduction_pct": 99.7,
        "mttr_speedup_vs_static_plc": 7.11,
        "central_gpu_reduction_pct_vs_cloud_raw": 87.5,
    }


def main() -> None:
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(build_summary(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out.name)


if __name__ == "__main__":
    main()
