from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "distributed_checkpointing_pytorch_dcp_megatron_async_orbax_storage_selection",
        "title": "Distributed Training Checkpointing Selection: PyTorch Async DCP vs. Megatron-Core DistCkpt vs. JAX Orbax & NVMe Burst Buffer",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "cluster_size_gpus": 1024,
            "model": "405B Dense / 671B MoE (BF16 Weights + FP32 Master Weights + FP32 AdamW States)",
            "total_checkpoint_bytes_tb": 6.4,
            "checkpoint_interval_minutes": 30,
        },
        "tiers": [
            {
                "tier": "Tier 1: Naive Gather-to-Rank-0 Blocking Save (torch.save on rank==0)",
                "gpu_blocking_stall_sec": 480.0,
                "effective_storage_write_gbps": 13.3,
                "supports_elastic_topology_resharding": False,
                "checkpoint_goodput_pct": 76.9,
                "monthly_wasted_gpu_compute_usd": 334000,
                "recommended_scenario": "Anti-pattern for multi-node training (causes Rank-0 OOM and 8-minute GPU stalls)",
            },
            {
                "tier": "Tier 2: Synchronous Sharded DCP / Orbax Direct-to-Parallel-FS",
                "gpu_blocking_stall_sec": 38.0,
                "effective_storage_write_gbps": 168.4,
                "supports_elastic_topology_resharding": True,
                "checkpoint_goodput_pct": 97.9,
                "monthly_wasted_gpu_compute_usd": 30400,
                "recommended_scenario": "Baseline sharded save when host RAM is constrained",
            },
            {
                "tier": "Tier 3: Two-Stage Pinned-RAM D2H + Background Async Flush (Async DCP / Orbax)",
                "gpu_blocking_stall_sec": 1.8,
                "effective_storage_write_gbps": 185.0,
                "supports_elastic_topology_resharding": True,
                "checkpoint_goodput_pct": 99.90,
                "monthly_wasted_gpu_compute_usd": 1450,
                "recommended_scenario": "Standard 128-512 GPU clusters with sufficient pinned host DRAM",
            },
            {
                "tier": "Tier 4 (Recommended): Zero-Overhead CUDA Stream D2H + Local NVMe Burst Buffer + Async DCP/Orbax",
                "gpu_blocking_stall_sec": 0.25,
                "effective_storage_write_gbps": 320.0,
                "supports_elastic_topology_resharding": True,
                "checkpoint_goodput_pct": 99.98,
                "monthly_wasted_gpu_compute_usd": 290,
                "recommended_scenario": "Production 1,024+ GPU/TPU pre-training with frequent 15-30 min elastic checkpoints",
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
