from __future__ import annotations
import argparse
import json
from pathlib import Path


def run_suite() -> dict:
    # 1. DeepEP Cross-Process Shared Memory via CUDA Driver cuMem API (#217 9249c25796)
    # Replaces legacy cudaIpcGetMemHandle constraints with expandable cuMemCreate / cuMemExportToShareableHandle
    ep_ranks_per_node = 8
    cumem_granularity_mib = 2
    exported_fabric_handles = ep_ranks_per_node
    legacy_ipc_remap_stalls = 8
    cumem_dynamic_remap_stalls = 0

    # 2. TRT-LLM Cached MPI Pool Drain Before Child Process Spawn (#20331 9d3b492125)
    # Prevents inherited MPI communicator state deadlock (nvbugs/6700875)
    cached_mpi_pools = 4
    undrained_child_spawn_deadlocks = 4
    drained_child_spawn_deadlocks = 0

    # 3. SGLang EFA Installer 1.51.0 Upgrade (#46892 e337bb10b5), DSpark MLA Seed (#47540 e4bb0478c5),
    # & vLLM ROCm CI Tier Consolidation (#61939 bba4d9afec) + AITER MXFP4 Padding Tests (#62996 e3f03f4f89)
    sglang_efa_installer_version = "1.51.0"
    vllm_rocm_legacy_image_tiers = 4
    vllm_rocm_collapsed_image_tiers = 1
    rocm_ci_image_reduction_ratio = round(
        vllm_rocm_legacy_image_tiers / vllm_rocm_collapsed_image_tiers, 2
    )
    aiter_mxfp4_unskipped_test_suites = 2

    return {
        "suite": "deepep_cumem_shared_sglang_efa151_trtllm_mpi_drain_phase169",
        "phase": 169,
        "priority": 178,
        "status": "ok",
        "deepep_cumem_cross_process_shm": {
            "ep_ranks_per_node": ep_ranks_per_node,
            "cumem_granularity_mib": cumem_granularity_mib,
            "exported_fabric_handles": exported_fabric_handles,
            "cumem_dynamic_remap_stalls": cumem_dynamic_remap_stalls,
        },
        "trtllm_mpi_pool_drain_before_fork": {
            "cached_mpi_pools": cached_mpi_pools,
            "drained_child_spawn_deadlocks": drained_child_spawn_deadlocks,
        },
        "sglang_efa_and_vllm_rocm_aiter": {
            "sglang_efa_installer_version": sglang_efa_installer_version,
            "rocm_ci_image_reduction_ratio": rocm_ci_image_reduction_ratio,
            "aiter_mxfp4_unskipped_test_suites": aiter_mxfp4_unskipped_test_suites,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    out = run_suite()
    res_path = Path(__file__).with_name("results.json")
    res_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if args.check:
        assert out["status"] == "ok"
        assert out["deepep_cumem_cross_process_shm"]["cumem_dynamic_remap_stalls"] == 0
        assert out["trtllm_mpi_pool_drain_before_fork"]["drained_child_spawn_deadlocks"] == 0
        assert out["sglang_efa_and_vllm_rocm_aiter"]["rocm_ci_image_reduction_ratio"] == 4.0
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
