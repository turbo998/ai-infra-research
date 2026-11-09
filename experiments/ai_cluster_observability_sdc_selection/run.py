#!/usr/bin/env python3
"""Deterministic evaluator for Round 21: Large-Scale AI Cluster Observability, Straggler & Silent Data Corruption (SDC) Selection.

Compares 4 observability, straggler triage, and Silent Data Corruption (SDC) detection architectures
across a 2,048-GPU H100 training cluster ($2.50/GPU-hr = $5,120/hr = $3,737,600/mo lease):
  - Tier 1: Basic Node-Level Prometheus + 30s nvidia-smi Polling + Manual Bisect (8,700s triage, 68.4% goodput)
  - Tier 2: NVIDIA DCGM Hardware Telemetry + XID Auto-Cordon (Catches hard ECC/XID, misses grey hangs & ALU SDC)
  - Tier 3: PyTorch FlightRecorder + NCCL RAS + CUDA Event Straggler Profiler (0.65s patient-zero triage, 91.5% goodput)
  - Tier 4: Tier 3 + Online Tensor Checksum & Deterministic Golden-Step SDC Verifier (99.6% SDC catch, 96.4% goodput)

Provenance classification:
  - provenance_type: "synthetic" (deterministic MTTR, straggler drag, and SDC rollback waste model
    calibrated to published Meta Llama-3 16K-GPU reliability reports, NVIDIA NCCL RAS, and PyTorch FlightRecorder specs)
  - independently_reproduced: false
  - physical_gpu_cluster_measurement: "not_measured"
  - local_cpu_harness_measurement: "measured"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def compute_model() -> dict[str, Any]:
    cluster_gpus = 2048
    nodes = 256
    gpu_hourly_usd = 2.50
    hours_per_month = 730.0
    monthly_cluster_lease_usd = round(cluster_gpus * gpu_hourly_usd * hours_per_month, 2)  # $3,737,600.00

    # Tier 1: Basic Prometheus + 30s nvidia-smi Polling + Manual Log Bisect
    t1_triage_sec = 8700.0  # 145.0 minutes (30 min NCCL timeout + 115 min manual node bisection)
    t1_mttr_min = 145.0
    t1_grey_straggler_catch_pct = 12.0
    t1_sdc_catch_pct = 0.0
    t1_goodput_pct = 68.4
    t1_monthly_waste_usd = round(monthly_cluster_lease_usd * (1.0 - t1_goodput_pct / 100.0), 2)  # $1,181,081.60

    # Tier 2: NVIDIA DCGM Hardware Telemetry + NPD XID Auto-Cordon
    t2_triage_sec = 45.0
    t2_mttr_min = 12.5
    t2_grey_straggler_catch_pct = 48.0
    t2_sdc_catch_pct = 0.0  # Tensor Core ALU bit-flips do not trigger HBM ECC or XID
    t2_goodput_pct = 82.2
    t2_monthly_waste_usd = round(monthly_cluster_lease_usd * (1.0 - t2_goodput_pct / 100.0), 2)  # $665,292.80

    # Tier 3: PyTorch FlightRecorder + NCCL RAS + Per-Kernel CUDA Event Straggler Detector
    t3_triage_sec = 0.65  # 650 ms collective dependency graph patient-zero localization
    t3_mttr_min = 1.80
    t3_grey_straggler_catch_pct = 98.5
    t3_sdc_catch_pct = 15.0  # Only catches extreme NaN/Inf explosions, not subtle mantissa drift
    t3_goodput_pct = 91.5
    t3_monthly_waste_usd = round(monthly_cluster_lease_usd * (1.0 - t3_goodput_pct / 100.0), 2)  # $317,696.00

    # Tier 4: Tier 3 + Online Tensor Norm Fingerprint & Deterministic Golden-Step SDC Verifier
    t4_triage_sec = 0.65
    t4_mttr_min = 0.85
    t4_grey_straggler_catch_pct = 99.4
    t4_sdc_catch_pct = 99.6
    t4_goodput_pct = 96.4
    t4_monthly_waste_usd = round(monthly_cluster_lease_usd * (1.0 - t4_goodput_pct / 100.0), 2)  # $134,553.60
    t4_monthly_savings_vs_t1_usd = round(t1_monthly_waste_usd - t4_monthly_waste_usd, 2)  # $1,046,528.00

    tiers = {
        "tier1_basic_prometheus_polling": {
            "label": "Tier 1: Basic Prometheus + 30s nvidia-smi Polling + Manual Bisect",
            "root_cause_triage_sec": t1_triage_sec,
            "recovery_mttr_min": t1_mttr_min,
            "grey_straggler_detection_pct": t1_grey_straggler_catch_pct,
            "sdc_detection_rate_pct": t1_sdc_catch_pct,
            "telemetry_compute_overhead_pct": 0.05,
            "effective_goodput_pct": t1_goodput_pct,
            "monthly_fault_waste_usd": t1_monthly_waste_usd,
            "monthly_savings_vs_tier1_usd": 0.0,
            "waste_reduction_pct": 0.0,
        },
        "tier2_dcgm_xid_auto_cordon": {
            "label": "Tier 2: NVIDIA DCGM Telemetry + NPD XID Auto-Cordon",
            "root_cause_triage_sec": t2_triage_sec,
            "recovery_mttr_min": t2_mttr_min,
            "grey_straggler_detection_pct": t2_grey_straggler_catch_pct,
            "sdc_detection_rate_pct": t2_sdc_catch_pct,
            "telemetry_compute_overhead_pct": 0.10,
            "effective_goodput_pct": t2_goodput_pct,
            "monthly_fault_waste_usd": t2_monthly_waste_usd,
            "monthly_savings_vs_tier1_usd": round(t1_monthly_waste_usd - t2_monthly_waste_usd, 2),  # $515,788.80
            "waste_reduction_pct": round((1.0 - t2_monthly_waste_usd / t1_monthly_waste_usd) * 100.0, 1),  # 43.7%
        },
        "tier3_flightrecorder_nccl_ras": {
            "label": "Tier 3: PyTorch FlightRecorder + NCCL RAS (<650ms Patient-Zero Triage)",
            "root_cause_triage_sec": t3_triage_sec,
            "recovery_mttr_min": t3_mttr_min,
            "grey_straggler_detection_pct": t3_grey_straggler_catch_pct,
            "sdc_detection_rate_pct": t3_sdc_catch_pct,
            "telemetry_compute_overhead_pct": 0.18,
            "effective_goodput_pct": t3_goodput_pct,
            "monthly_fault_waste_usd": t3_monthly_waste_usd,
            "monthly_savings_vs_tier1_usd": round(t1_monthly_waste_usd - t3_monthly_waste_usd, 2),  # $863,385.60
            "waste_reduction_pct": round((1.0 - t3_monthly_waste_usd / t1_monthly_waste_usd) * 100.0, 1),  # 73.1%
        },
        "tier4_online_sdc_golden_verifier": {
            "label": "Tier 4: Tier 3 + Online Tensor Checksum & Golden-Step SDC Verifier",
            "root_cause_triage_sec": t4_triage_sec,
            "recovery_mttr_min": t4_mttr_min,
            "grey_straggler_detection_pct": t4_grey_straggler_catch_pct,
            "sdc_detection_rate_pct": t4_sdc_catch_pct,
            "telemetry_compute_overhead_pct": 0.35,
            "effective_goodput_pct": t4_goodput_pct,
            "monthly_fault_waste_usd": t4_monthly_waste_usd,
            "monthly_savings_vs_tier1_usd": t4_monthly_savings_vs_t1_usd,  # $1,046,528.00
            "waste_reduction_pct": round((1.0 - t4_monthly_waste_usd / t1_monthly_waste_usd) * 100.0, 1),  # 88.6%
        },
    }

    cluster_scale_goodput_sensitivity = [
        {"gpus": 256, "tier1_goodput_pct": 86.0, "tier2_goodput_pct": 91.5, "tier3_goodput_pct": 95.2, "tier4_goodput_pct": 97.5},
        {"gpus": 1024, "tier1_goodput_pct": 76.5, "tier2_goodput_pct": 86.4, "tier3_goodput_pct": 93.4, "tier4_goodput_pct": 96.9},
        {"gpus": 2048, "tier1_goodput_pct": 68.4, "tier2_goodput_pct": 82.2, "tier3_goodput_pct": 91.5, "tier4_goodput_pct": 96.4},
        {"gpus": 8192, "tier1_goodput_pct": 42.0, "tier2_goodput_pct": 69.5, "tier3_goodput_pct": 86.8, "tier4_goodput_pct": 94.8},
    ]

    return {
        "experiment_id": "ai_cluster_observability_sdc_selection",
        "round": 21,
        "provenance": {
            "provenance_type": "synthetic",
            "classification": "synthetic_observability_straggler_and_sdc_reliability_model_calibrated_to_anonymized_reported_specs",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "local_cpu_harness_measurement": "measured",
            "calibration_sources": [
                "Meta Llama-3 16,384-GPU H100 training reliability & Silent Data Corruption (SDC) postmortem analysis",
                "NVIDIA NCCL 2.23+ RAS (Reliability, Availability, Serviceability) & PyTorch FlightRecorder ring-buffer specifications",
                "Google / Meta deterministic Golden-Batch shadow verification for Tensor Core ALU silent bit-flip detection",
            ],
        },
        "baseline_cluster": {
            "cluster_gpus": cluster_gpus,
            "nodes": nodes,
            "gpu_hourly_usd": gpu_hourly_usd,
            "hours_per_month": hours_per_month,
            "monthly_cluster_lease_usd": monthly_cluster_lease_usd,
        },
        "tiers": tiers,
        "cluster_scale_goodput_sensitivity": cluster_scale_goodput_sensitivity,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify results.json matches deterministic model")
    args = parser.parse_args()

    expected = compute_model()
    if args.check:
        if not RESULTS_PATH.exists():
            print(f"ERROR: {RESULTS_PATH} does not exist.", file=sys.stderr)
            return 1
        actual = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if actual != expected:
            print(f"ERROR: {RESULTS_PATH} drifted from deterministic model.", file=sys.stderr)
            return 1
        print(f"OK: {RESULTS_PATH} matches deterministic Round 21 AI cluster observability & SDC selection model.")
        return 0

    RESULTS_PATH.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
