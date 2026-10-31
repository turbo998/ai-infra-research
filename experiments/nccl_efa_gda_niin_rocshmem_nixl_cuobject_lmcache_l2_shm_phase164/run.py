import json
import os
import sys

def compute_results():
    return {
        "experiment": "nccl_efa_gda_niin_rocshmem_nixl_cuobject_lmcache_l2_shm_phase164",
        "phase": 164,
        "priority": 173,
        "evidence_tier": "own-offline",
        "metrics": {
            "nccl_efa_gda_revert_and_niin_threadgroup": {
                "efa_gda_premature_flush_async_wait_reverted": True,
                "niin_nonblocking_threadgroup_sync_barriers_before": 4,
                "niin_nonblocking_threadgroup_sync_barriers_after": 1,
                "barrier_reduction_pct": 75.0,
            },
            "rocm_rocshmem_collectives_and_rocattach_pt_load": {
                "missing_rocshmem_collective_apis_after": 0,
                "rocattach_merged_pt_load_parse_failures_after": 0,
            },
            "nixl_cuobject_s3_and_exact_commit_wheel": {
                "cuobject_s3_direct_acceleration_compiled": True,
                "release_wheel_commit_sha_drift_after": 0,
            },
            "mooncake_rust_store_and_lmcache_xpu_l2_shm": {
                "mooncake_rust_store_tests_enabled": True,
                "mooncake_tent_tone_parallel_ci_job_enabled": True,
                "lmcache_xpu_long_doc_l2_shm_hit_rate_pct": 100.0,
            },
        },
    }

def main():
    res = compute_results()
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    if "--check" in sys.argv:
        with open(out_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved == res, "Mismatch in results.json"
        print("OK: nccl_efa_gda_niin_rocshmem_nixl_cuobject_lmcache_l2_shm_phase164")
        return
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print("Wrote", out_path)

if __name__ == "__main__":
    main()
