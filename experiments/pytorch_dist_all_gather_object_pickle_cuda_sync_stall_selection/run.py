from __future__ import annotations

import json
from pathlib import Path


def evaluate() -> dict:
    # Workload: 64-rank distributed training/serving exchanging 4 KB step metadata per iteration
    world_size = 64
    metadata_payload_kb = 4.0

    # Tier 1: dist.all_gather_object(out, python_dict, group=nccl_group)
    # 2 NCCL AllGathers (sizes + padded uint8 bytes) + 2 hard .cpu() stream sync barriers + dynamic cudaMalloc!
    t1_step_stall_ms = 14.80
    t1_gpu_stream_syncs_per_call = 2
    t1_collectives_per_call = 2

    # Tier 2: dist.all_gather_object with unstripped CUDA tensor inside dict (cuda:0 deserialization spike)
    t2_step_stall_ms = 185.00
    t2_gpu_stream_syncs_per_call = 66
    t2_collectives_per_call = 2

    # Tier 3: dist.all_gather_object over CPU gloo group (backend="gloo")
    # 0 GPU stream syncs, but still 2 TCP rounds + pickle.dumps/loads overhead (3.40 ms)
    t3_step_stall_ms = 3.40
    t3_gpu_stream_syncs_per_call = 0
    t3_collectives_per_call = 2

    # Tier 4: Fixed-Schema Pre-Allocated Tensor Packing (dist.all_gather_into_tensor / SHM Ring)
    # 1 collective, 0 pickle overhead, 0 dynamic allocations, 0.08 ms step latency (185.0x faster than Tier 1!)
    t4_step_stall_ms = 0.08
    t4_gpu_stream_syncs_per_call = 0
    t4_collectives_per_call = 1

    speedup_vs_t1_x = round(t1_step_stall_ms / t4_step_stall_ms, 1)  # 185.0x
    speedup_vs_t3_x = round(t3_step_stall_ms / t4_step_stall_ms, 1)  # 42.5x

    return {
        "benchmark": "pytorch_dist_all_gather_object_pickle_cuda_sync_stall_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "world_size": world_size,
            "metadata_payload_kb": metadata_payload_kb,
        },
        "tiers": {
            "tier1_nccl_all_gather_object": {
                "step_stall_ms": t1_step_stall_ms,
                "gpu_stream_syncs_per_call": t1_gpu_stream_syncs_per_call,
                "collectives_per_call": t1_collectives_per_call,
                "dynamic_gpu_alloc": True,
            },
            "tier2_cuda_tensor_in_pickled_dict": {
                "step_stall_ms": t2_step_stall_ms,
                "gpu_stream_syncs_per_call": t2_gpu_stream_syncs_per_call,
                "collectives_per_call": t2_collectives_per_call,
                "dynamic_gpu_alloc": True,
            },
            "tier3_gloo_cpu_all_gather_object": {
                "step_stall_ms": t3_step_stall_ms,
                "gpu_stream_syncs_per_call": t3_gpu_stream_syncs_per_call,
                "collectives_per_call": t3_collectives_per_call,
                "dynamic_gpu_alloc": False,
            },
            "tier4_fixed_schema_all_gather_into_tensor": {
                "step_stall_ms": t4_step_stall_ms,
                "gpu_stream_syncs_per_call": t4_gpu_stream_syncs_per_call,
                "collectives_per_call": t4_collectives_per_call,
                "dynamic_gpu_alloc": False,
                "speedup_vs_tier1_x": speedup_vs_t1_x,
                "speedup_vs_tier3_x": speedup_vs_t3_x,
            },
        },
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
