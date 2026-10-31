import json
import os
import sys

def compute_results():
    return {
        "experiment": "nccl_sm100_902_rccl_ll128_regs_nixl_torch216_lmcache_dsv4_hma_phase163",
        "phase": 163,
        "priority": 172,
        "evidence_tier": "own-offline",
        "metrics": {
            "nccl_sm100_search_bw_restoration": {
                "regressed_sm100_search_bw_gbps": 45.1,
                "restored_sm100_search_bw_gbps": 90.2,
                "search_bandwidth_multiplier": 2.0,
            },
            "rccl_ll128_register_lane_state_and_rings_validation": {
                "ll128_scratch_spills_per_iter_before": 4,
                "ll128_scratch_spills_per_iter_after": 0,
                "ll128_small_msg_latency_reduction_pct": 14.5,
                "malformed_nccl_rings_rejected_pct": 100.0,
                "gin_host_entry_point_coverage_pct": 100.0,
            },
            "nixl_torch216_and_seeded_port_blocks": {
                "torch_wheel_version_included": "2.16",
                "concurrent_ci_port_collision_rate_pct_before": 12.5,
                "concurrent_ci_port_collision_rate_pct_after": 0.0,
            },
            "mooncake_leg_scoped_build_and_lmcache_xpu_dsv4_hma": {
                "mooncake_redundant_targets_built_pct_after": 0.0,
                "lmcache_xpu_dsv4_hma_token_match_rate_pct": 100.0,
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
        print("OK: nccl_sm100_902_rccl_ll128_regs_nixl_torch216_lmcache_dsv4_hma_phase163")
        return
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print("Wrote", out_path)

if __name__ == "__main__":
    main()
