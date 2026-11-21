from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "compute_communication_overlap_userbuffers_fsdp2_nccl_sm_contention_selection",
        "title": "Compute-Communication Overlap & SM-Contention Governance: TransformerEngine Userbuffers (TMA) vs. FSDP2 Async vs. Standard NCCL SM Stealing",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "gpu_architecture": "NVIDIA H100 / H200 (132 SMs) with NVSwitch NVLS + 400G RoCEv2",
            "model": "70B / 405B Training & Prefill (TP=8 Intra-Node + FSDP2 Inter-Node)",
            "total_sms_per_gpu": 132,
        },
        "tiers": [
            {
                "tier": "Tier 1: Sequential Non-Overlapped GEMM -> NCCL Collective (Zero Overlap)",
                "sms_consumed_by_comm": 32,
                "gemm_wave_quantization_slowdown_pct": 0.0,
                "exposed_comm_overhead_pct": 32.0,
                "effective_cluster_mfu_pct": 38.5,
                "recommended_scenario": "Anti-pattern ( GPUs sit 100% idle during AllGather and ReduceScatter )",
            },
            {
                "tier": "Tier 2: Naive Multi-Stream GEMM + Standard NCCL Overlap (NCCL_MAX_NCHANNELS=32)",
                "sms_consumed_by_comm": 28,
                "gemm_wave_quantization_slowdown_pct": 21.5,
                "exposed_comm_overhead_pct": 14.5,
                "effective_cluster_mfu_pct": 46.2,
                "recommended_scenario": "Un-tuned overlap trap (NCCL steals 28/132 SMs, triggering GEMM tail-wave slowdown)",
            },
            {
                "tier": "Tier 3: FSDP2 Async Prefetch + NVLS Hardware Offload (NCCL_NVLS_ENABLE=1, 6 Channels)",
                "sms_consumed_by_comm": 6,
                "gemm_wave_quantization_slowdown_pct": 4.2,
                "exposed_comm_overhead_pct": 4.2,
                "effective_cluster_mfu_pct": 53.8,
                "recommended_scenario": "Standard multi-node FSDP2 training with NVSwitch SHARP offload",
            },
            {
                "tier": "Tier 4 (Recommended): Fine-Grained Pipelined Overlap (TransformerEngine Userbuffers + TMA Copy Engine)",
                "sms_consumed_by_comm": 2,
                "gemm_wave_quantization_slowdown_pct": 0.8,
                "exposed_comm_overhead_pct": 0.8,
                "effective_cluster_mfu_pct": 57.4,
                "recommended_scenario": "High-MFU TP=8 + FSDP2 pre-training & high-throughput prefill engines",
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
