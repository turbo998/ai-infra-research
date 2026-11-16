from __future__ import annotations
import json
from pathlib import Path


def evaluate_cuda_ipc_handle_pool() -> dict:
    # Workload: Sharing 512 weight/KV tensors per step across colocated GPU processes (RL Actor->Rollout or Intra-Node P/D)
    tiers = [
        {
            "tier_id": "tier1_per_tensor_cuda_ipc_open_close_every_step_mmu_stall",
            "name": "Per-Tensor `cudaIpcOpenMemHandle` & `cudaIpcCloseMemHandle` Every Step (`512` Calls/Step)",
            "ipc_mapping_overhead_ms_per_step": 18.40,
            "open_file_descriptors_per_rank": 512,
            "supports_vmm_and_gb200_mnnvl": False,
            "action": "Send to a human",
            "status": "fail",
            "note": "Calling cudaIpcOpenMemHandle + Close 512 times/step triggers GPU MMU page-table mapping & implicit device syncs (18.40ms stall)!"
        },
        {
            "tier_id": "tier2_uncapped_fd_handle_cache_without_segment_dedup_emfile_crash",
            "name": "Per-Tensor `POSIX FD` IPC Without Base-Segment Deduplication (`EMFILE: Too Many Open Files`)",
            "ipc_mapping_overhead_ms_per_step": 4.20,
            "open_file_descriptors_per_rank": 4096,
            "supports_vmm_and_gb200_mnnvl": False,
            "action": "Send to a human",
            "status": "fail",
            "note": "Exporting a new SCM_RIGHTS POSIX FD per tensor sub-slice quickly exceeds Linux default ulimit -n 1024 -> fatal EMFILE crash!"
        },
        {
            "tier_id": "tier3_legacy_cuda_ipc_base_allocation_cache_no_vmm_support",
            "name": "Cached Legacy `cudaIpcGetMemHandle` on Base `cudaMalloc` Segments (`BasePtr + ByteOffset`)",
            "ipc_mapping_overhead_ms_per_step": 0.015,
            "open_file_descriptors_per_rank": 16,
            "supports_vmm_and_gb200_mnnvl": False,
            "action": "Confirm",
            "status": "pass",
            "note": "Maps each 2GB CUDACachingAllocator base segment once and passes byte offsets (0.015ms), but fails with CUDA_ERROR_NOT_SUPPORTED when expandable_segments:True or GB200 MNNVL is used!"
        },
        {
            "tier_id": "tier4_vmm_cu_mem_export_shareable_fabric_fd_pool_plus_offset_registry",
            "name": "CUDA VMM `cuMemExportToShareableHandle` (`POSIX FD` / `FABRIC`) Base-Pool Registry + Offset Arithmetic",
            "ipc_mapping_overhead_ms_per_step": 0.002,
            "open_file_descriptors_per_rank": 8,
            "supports_vmm_and_gb200_mnnvl": True,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Maps physical VMM slabs once at init (closing temp FDs immediately after cuMemMap!) and exchanges pure (slab_id, byte_offset) in 0.002ms (9,200x faster, VMM + GB200 MNNVL ready)!"
        },
    ]
    return {
        "experiment_id": "cuda_ipc_memhandle_cache_leak_fd_exhaustion_fabric_handle_pool_selection",
        "round": 322,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "shared_tensors_per_step": 512,
            "colocated_ranks": 8
        },
        "tiers": tiers,
        "summary": {
            "ipc_speedup_x": round(18.40 / 0.002, 1),
            "fd_reduction_x": round(4096 / 8, 1),
            "recommended_overhead_ms": 0.002,
            "recommended_tier": "tier4_vmm_cu_mem_export_shareable_fabric_fd_pool_plus_offset_registry"
        }
    }


if __name__ == "__main__":
    data = evaluate_cuda_ipc_handle_pool()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
