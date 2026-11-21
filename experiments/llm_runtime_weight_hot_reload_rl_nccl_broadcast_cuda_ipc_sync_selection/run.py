from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "llm_runtime_weight_hot_reload_rl_nccl_broadcast_cuda_ipc_sync_selection",
        "title": "Runtime Weight Hot-Reload & RL Actor-to-Rollout Weight Sync: Zero-Copy CUDA IPC & Bucketed FP8 NCCL Broadcast vs. Disk Staging",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "model": "70B (140 GB BF16 -> 70 GB FP8 Rollout Weights)",
            "cluster_size_gpus": 128,
            "rl_compute_and_rollout_sec_per_step": 30.0,
        },
        "tiers": [
            {
                "tier": "Tier 1: Shared Filesystem .safetensors Dump + vLLM Engine Reload",
                "weight_sync_latency_sec": 95.0,
                "preserves_cuda_graphs_in_place": False,
                "rl_step_goodput_pct": 24.0,
                "monthly_wasted_sync_gpu_usd": 308000,
                "recommended_scenario": "Anti-pattern (disk write/read + 22s CUDA Graph recapture stalls 76% of RL step)",
            },
            {
                "tier": "Tier 2: TCP/gRPC Host-Staged Parameter Streaming (In-Place copy_())",
                "weight_sync_latency_sec": 22.0,
                "preserves_cuda_graphs_in_place": True,
                "rl_step_goodput_pct": 57.7,
                "monthly_wasted_sync_gpu_usd": 171000,
                "recommended_scenario": "Legacy cross-subnet weight refresh without RDMA connectivity",
            },
            {
                "tier": "Tier 3: Disaggregated 400G RDMA NCCL Broadcast (StatelessProcessGroup BF16)",
                "weight_sync_latency_sec": 2.4,
                "preserves_cuda_graphs_in_place": True,
                "rl_step_goodput_pct": 92.6,
                "monthly_wasted_sync_gpu_usd": 29900,
                "recommended_scenario": "Disaggregated Actor-Rollout clusters over 400GbE RoCEv2 / InfiniBand",
            },
            {
                "tier": "Tier 4 (Recommended): Zero-Copy CUDA IPC Handles (Co-Located 0.45s) / Bucketed FP8 NCCL Broadcast (Disaggregated 1.15s)",
                "weight_sync_latency_sec": 0.45,
                "preserves_cuda_graphs_in_place": True,
                "rl_step_goodput_pct": 98.5,
                "monthly_wasted_sync_gpu_usd": 6050,
                "recommended_scenario": "Production veRL / SLIME / OpenRLHF GRPO training & sub-second canary hot-swapping",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
