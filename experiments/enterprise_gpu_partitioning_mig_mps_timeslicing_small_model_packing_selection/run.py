from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    partition_modes = [
        {
            "mode": "Tier 1: Whole-GPU Exclusive Allocation (1 Full H200 per Small Model: Embedding, Reranker, OCR, Guardrail)",
            "gpus_required_for_14_microservices": 14,
            "noisy_neighbor_p99_jitter_pct": 0.0,
            "hardware_fault_and_oom_isolation": True,
            "monthly_gpu_cost_usd": 25200.0,
        },
        {
            "mode": "Tier 2: Kubernetes GPU Time-Slicing (Context-Switch Preemption + Shared HBM without Memory Quota)",
            "gpus_required_for_14_microservices": 2,
            "noisy_neighbor_p99_jitter_pct": 310.0,
            "hardware_fault_and_oom_isolation": False,
            "monthly_gpu_cost_usd": 3600.0,
        },
        {
            "mode": "Tier 3: CUDA MPS (Multi-Process Service Spatial SM Sharing + Shared Fault Domain)",
            "gpus_required_for_14_microservices": 2,
            "noisy_neighbor_p99_jitter_pct": 24.5,
            "hardware_fault_and_oom_isolation": False,
            "monthly_gpu_cost_usd": 3600.0,
        },
        {
            "mode": "Tier 4: Hardware MIG Partitioning on H200 (7x 1g.18gb / 3g.71gb Hardware SM + L2 + HBM Crossbar Isolation)",
            "gpus_required_for_14_microservices": 2,
            "noisy_neighbor_p99_jitter_pct": 1.2,
            "hardware_fault_and_oom_isolation": True,
            "monthly_gpu_cost_usd": 3600.0,
        },
    ]
    return {
        "experiment_id": "enterprise_gpu_partitioning_mig_mps_timeslicing_small_model_packing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "h200_mig_slices_max": 7,
        "partition_modes": partition_modes,
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
        print("OK: enterprise_gpu_partitioning_mig_mps_timeslicing_small_model_packing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
