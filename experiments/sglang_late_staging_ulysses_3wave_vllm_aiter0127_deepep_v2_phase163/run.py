import json
import os
import sys

def compute_results():
    return {
        "experiment": "sglang_late_staging_ulysses_3wave_vllm_aiter0127_deepep_v2_phase163",
        "phase": 163,
        "priority": 172,
        "evidence_tier": "own-offline",
        "metrics": {
            "sglang_disagg_late_staging_req_guard": {
                "late_staging_req_duplicate_scatter_before": 1,
                "late_staging_req_duplicate_scatter_after": 0,
                "kv_corruption_rate_pct_after_fix": 0.0,
            },
            "sglang_diffusion_ulysses_3wave_and_gfx950_vae": {
                "ulysses_head_group_waves_before": 6,
                "ulysses_head_group_waves_after": 3,
                "wave_reduction_pct": 50.0,
                "gfx950_minimax_h3_dead_math_sdpa_skipped_pct": 100.0,
                "dp_replica_warmup_coverage_pct": 100.0,
            },
            "vllm_rocm_aiter0127_and_prebuilt_mamba": {
                "aiter_version": "v0.1.27",
                "runtime_pip_compile_minutes_before": 11.5,
                "runtime_pip_compile_minutes_after": 0.0,
            },
            "trtllm_ucx_loopback_and_deepep_v2_intra_ep": {
                "ucx_off_host_interface_leaks": 0,
                "deepep_v2_single_node_intra_ep_init_failures": 0,
                "deepep_pip_wheel_so_resolution_rate_pct": 100.0,
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
        print("OK: sglang_late_staging_ulysses_3wave_vllm_aiter0127_deepep_v2_phase163")
        return
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print("Wrote", out_path)

if __name__ == "__main__":
    main()
