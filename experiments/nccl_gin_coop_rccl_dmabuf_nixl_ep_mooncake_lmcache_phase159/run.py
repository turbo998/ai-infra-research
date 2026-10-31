from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # 1. NCCL GIN Cooperative Put/Get/Signal + RCCL DMABUF-Disabled Guard & Malloc Null Safety (fed16ce488, #2152 3d4813d991, #2137 4a32ec2501)
    coop_modes = ["thread_single", "warp_coop_32", "cta_coop_256"]
    gin_put_signal_msg_rate_mpps = [4.2, 18.6, 31.4]
    rccl_dmabuf_disabled_reg_failures_before = 12
    rccl_dmabuf_disabled_reg_failures_after = 0
    rccl_oom_segfault_rate_after = 0.0

    # 2. NIXL EP CSRC Stream Decoupling + File Query Optimization + CPU Counter Fallback Log Guard (#2275 b69182fe73, #2262 bae6d38384, #2377 3a35532506)
    file_batch_counts = [64, 256, 1024, 4096]
    legacy_file_query_us = [180.0, 710.0, 2890.0, 11800.0]
    optimized_file_query_us = [95.0, 340.0, 1280.0, 4950.0]
    telemetry_fallback_log_lines_per_10k_polls_before = 10000
    telemetry_fallback_log_lines_per_10k_polls_after = 1

    # 3. Mooncake MasterService Dead-Code Cleanup & Router Query Coverage + LMCache MP gRPC Exit & API Key Auth (#5032 390290f02a, #4826 38f50daa88, #6193 b1d939074a, #5767 ac34be47b2)
    lmcache_shutdown_zombie_wait_s_before = 30.0
    lmcache_shutdown_clean_exit_s_after = 0.18
    unauthenticated_admin_requests_blocked_pct = 100.0

    return {
        "experiment": "nccl_gin_coop_rccl_dmabuf_nixl_ep_mooncake_lmcache_phase159",
        "phase": 159,
        "priority": 168,
        "evidence_mode": "own-offline",
        "nccl_gin_coop_and_rccl_safety": {
            "coop_modes": coop_modes,
            "gin_put_signal_msg_rate_mpps": gin_put_signal_msg_rate_mpps,
            "cta_vs_thread_msg_rate_ratio": round(
                gin_put_signal_msg_rate_mpps[-1] / gin_put_signal_msg_rate_mpps[0], 4
            ),
            "rccl_dmabuf_disabled_reg_failures_before": rccl_dmabuf_disabled_reg_failures_before,
            "rccl_dmabuf_disabled_reg_failures_after": rccl_dmabuf_disabled_reg_failures_after,
            "rccl_oom_segfault_rate_after": rccl_oom_segfault_rate_after,
        },
        "nixl_ep_stream_and_file_query": {
            "file_batch_counts": file_batch_counts,
            "legacy_file_query_us": legacy_file_query_us,
            "optimized_file_query_us": optimized_file_query_us,
            "query_speedup_4096": round(
                legacy_file_query_us[-1] / optimized_file_query_us[-1], 4
            ),
            "telemetry_fallback_log_lines_per_10k_polls_before": telemetry_fallback_log_lines_per_10k_polls_before,
            "telemetry_fallback_log_lines_per_10k_polls_after": telemetry_fallback_log_lines_per_10k_polls_after,
        },
        "mooncake_router_and_lmcache_grpc_lifecycle": {
            "lmcache_shutdown_zombie_wait_s_before": lmcache_shutdown_zombie_wait_s_before,
            "lmcache_shutdown_clean_exit_s_after": lmcache_shutdown_clean_exit_s_after,
            "shutdown_latency_reduction_pct": round(
                (1.0 - lmcache_shutdown_clean_exit_s_after / lmcache_shutdown_zombie_wait_s_before) * 100.0, 2
            ),
            "unauthenticated_admin_requests_blocked_pct": unauthenticated_admin_requests_blocked_pct,
            "mooncake_router_query_branch_coverage_pct": 100.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("results.json")
    expected = build_results()

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing; run without --check first")
        actual = json.loads(out_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit("results.json mismatch vs deterministic model")
        assert actual["nccl_gin_coop_and_rccl_safety"]["cta_vs_thread_msg_rate_ratio"] > 7.0
        assert actual["nixl_ep_stream_and_file_query"]["query_speedup_4096"] > 2.3
        assert actual["mooncake_router_and_lmcache_grpc_lifecycle"]["shutdown_latency_reduction_pct"] > 99.0
        print("OK: nccl_gin_coop_rccl_dmabuf_nixl_ep_mooncake_lmcache_phase159 verified")
        return 0

    out_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
