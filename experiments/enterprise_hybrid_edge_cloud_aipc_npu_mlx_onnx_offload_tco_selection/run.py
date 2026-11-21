from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    employee_seats = 20000
    tiers = [
        {
            "tier": "Tier 1: 100% Cloud LLM API Only (Zero On-Device AI PC / Mac Offload)",
            "edge_offload_ratio_pct": 0.0,
            "local_task_ttft_ms": 420,
            "laptop_power_draw_watts": 0.0,
            "annual_cloud_gpu_tco_usd": 2400000,
        },
        {
            "tier": "Tier 2: Naive On-Device CPU-Only GGUF (No NPU / Metal Offload — Thermal Throttling)",
            "edge_offload_ratio_pct": 30.0,
            "local_task_ttft_ms": 380,
            "laptop_power_draw_watts": 38.0,
            "annual_cloud_gpu_tco_usd": 1680000,
        },
        {
            "tier": "Tier 3: OS-Native NPU/Metal Runtime (Windows Foundry Local ONNX QNN/OpenVINO & Apple MLX)",
            "edge_offload_ratio_pct": 55.0,
            "local_task_ttft_ms": 45,
            "laptop_power_draw_watts": 7.8,
            "annual_cloud_gpu_tco_usd": 1080000,
        },
        {
            "tier": "Tier 4: 66/34 Hybrid Edge-Cloud Router (On-Device 3.8B/7B INT4 NPU/MLX 66% + Local PII Scrubbing + 34% Cloud 70B/R1)",
            "edge_offload_ratio_pct": 66.8,
            "local_task_ttft_ms": 42,
            "laptop_power_draw_watts": 7.5,
            "annual_cloud_gpu_tco_usd": 796000,
        },
    ]
    return {
        "experiment_id": "enterprise_hybrid_edge_cloud_aipc_npu_mlx_onnx_offload_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "employee_laptop_seats": employee_seats,
            "max_safe_ram_budget_16gb_laptop_gb": 3.5,
            "phi4_mini_int4_footprint_gb": 2.4,
        },
        "tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_hybrid_edge_cloud_aipc_npu_mlx_onnx_offload_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
