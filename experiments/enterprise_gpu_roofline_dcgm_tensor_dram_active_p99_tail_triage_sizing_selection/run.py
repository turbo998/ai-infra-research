from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Legacy nvidia-smi GPU_UTIL & Host CPU Monitoring Only (Blind to False-Busy Spin)",
            "root_cause_identification_accuracy_pct": 31.0,
            "mean_time_to_diagnose_min": 145.0,
            "true_tensor_plus_dram_efficiency_pct": 28.4,
            "p99_itl_tail_ms": 68.5,
        },
        {
            "tier": "Tier 2: Basic Prometheus + vLLM Queue Metrics (No Hardware Counter Correlation)",
            "root_cause_identification_accuracy_pct": 64.0,
            "mean_time_to_diagnose_min": 42.0,
            "true_tensor_plus_dram_efficiency_pct": 52.0,
            "p99_itl_tail_ms": 34.0,
        },
        {
            "tier": "Tier 3: DCGM Profiling Counters (TENSOR_ACTIVE, DRAM_ACTIVE, NVLINK/PCIE, Xid/Clocks)",
            "root_cause_identification_accuracy_pct": 91.5,
            "mean_time_to_diagnose_min": 8.5,
            "true_tensor_plus_dram_efficiency_pct": 76.5,
            "p99_itl_tail_ms": 17.8,
        },
        {
            "tier": "Tier 4: DCGM 4-Quadrant Roofline Triage + OpenTelemetry Token-Span Correlation + Auto Straggler Quarantine",
            "root_cause_identification_accuracy_pct": 98.6,
            "mean_time_to_diagnose_min": 2.4,
            "true_tensor_plus_dram_efficiency_pct": 88.2,
            "p99_itl_tail_ms": 13.4,
        },
    ]
    quadrants = {
        "q1_compute_bound": {"tensor_active_pct": 82, "dram_active_pct": 35, "diagnosis": "Healthy Prefill Compute-Bound"},
        "q2_memory_bw_bound": {"tensor_active_pct": 14, "dram_active_pct": 86, "diagnosis": "Healthy Decode Memory-Bandwidth-Bound"},
        "q3_comm_or_cpu_stall": {"tensor_active_pct": 6, "dram_active_pct": 8, "diagnosis": "False-Busy! Waiting on NCCL AllReduce or CPU Python GIL"},
        "q4_thermal_throttle": {"tensor_active_pct": 45, "dram_active_pct": 40, "diagnosis": "SW_THERMAL_SLOWDOWN / Row-Remap Hardware Straggler"},
    }
    return {
        "experiment_id": "enterprise_gpu_roofline_dcgm_tensor_dram_active_p99_tail_triage_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "dcgm_four_quadrants": quadrants,
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
        print("OK: enterprise_gpu_roofline_dcgm_tensor_dram_active_p99_tail_triage_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
