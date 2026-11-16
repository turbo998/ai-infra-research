from __future__ import annotations

import json
from pathlib import Path


def evaluate_liveness_watchdog_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Shallow FastAPI HTTP-Only /health (return 200 OK)",
            "gpu_hang_detection_rate_pct": 0.0,
            "false_positive_kill_under_burst_pct": 0.0,
            "probe_latency_p99_ms": 0.20,
            "blackhole_duration_sec": 3600.0,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Synthetic 1-Token Inference Probe Queued via Engine",
            "gpu_hang_detection_rate_pct": 100.0,
            "false_positive_kill_under_burst_pct": 42.0,
            "probe_latency_p99_ms": 3450.0,
            "blackhole_duration_sec": 15.0,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: CPU-Only ZMQ Control Ping + Process.is_alive()",
            "gpu_hang_detection_rate_pct": 45.0,
            "false_positive_kill_under_burst_pct": 4.5,
            "probe_latency_p99_ms": 12.0,
            "blackhole_duration_sec": 1800.0,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Shared-Memory cudaEventQuery Progress Lease + Dynamic State Gate",
            "gpu_hang_detection_rate_pct": 100.0,
            "false_positive_kill_under_burst_pct": 0.0,
            "probe_latency_p99_ms": 0.05,
            "blackhole_duration_sec": 10.0,
            "decision": "Act automatically",
        },
    ]

    fault_modes = [
        {
            "fault": "Worker Process Segfault / SIGKILL (OOM Killer)",
            "tier1_detected": False,
            "tier3_detected": True,
            "tier4_detected": True,
        },
        {
            "fault": "Sticky CUDA_ERROR_ILLEGAL_ADDRESS / ECC Uncorrectable",
            "tier1_detected": False,
            "tier3_detected": True,
            "tier4_detected": True,
        },
        {
            "fault": "Infinite NCCL Ring/Tree Spin Hang on Rank 3 GPU Stream",
            "tier1_detected": False,
            "tier3_detected": False,
            "tier4_detected": True,
        },
        {
            "fault": "Heavy 64K Prefill Burst (3.45s Legitimate Step Duration)",
            "tier2_false_kill": True,
            "tier4_false_kill": False,
        },
    ]

    summary = {
        "benchmark": "llm_serving_liveness_probe_false_200_ok_gpu_worker_watchdog_lease_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "tp_size": 8,
        "shallow_hang_detection_pct": 0.0,
        "synthetic_probe_false_kill_pct": 42.0,
        "shm_watchdog_detection_pct": 100.0,
        "shm_watchdog_false_kill_pct": 0.0,
        "shm_probe_latency_ms": 0.05,
        "probe_speedup_vs_synthetic_x": round(3450.0 / 0.05, 1),
        "tiers": tiers,
        "fault_modes": fault_modes,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_liveness_watchdog_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
