#!/usr/bin/env python3
"""Phase 89 Track C offline model:
FlashAttention-4 SM90 Unmasked Interior Block Fast-Path & SM100 64-Head Sparse MLA Recompute-P,
verl vLLM >= 0.28 Multi-Node Hybrid-LB Leader Rank Guard, Mooncake Tiled COO Partial Reads,
HA SIGTERM Leader Lease Release & SSE4.2 CRC-32C, NIXL EP Barrier Put Completion & EFA Unsolicited Write Recv Disable,
and Dynamo KV-Router Custom Visitor Deserialization & Lazy OnceLock CRTC Leaf Storage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _crc32c_ref(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if (crc & 1) else 0)
    return crc ^ 0xFFFFFFFF


def build_summary() -> dict:
    # 1. FlashAttention-4 SM90 interior unmasked KV blocks (#2904) & SM100 64-head Sparse MLA recompute-P
    fa4_sm90_and_sm100 = {
        "sm90_fsel_saved_per_interior_kv_iter": 64,
        "h200_hdim128_causal_16k_before_tflops": 613,
        "h200_hdim128_causal_16k_after_tflops": 660,
        "h200_hdim128_causal_16k_speedup": round(660 / 613, 4),
        "h200_hdim64_causal_16k_before_tflops": 463,
        "h200_hdim64_causal_16k_after_tflops": 531,
        "h200_hdim64_causal_16k_speedup": round(531 / 463, 4),
        "sm100_sparse_mla_recompute_p_supported_q_heads": [64, 128],
        "gb200_64head_64k_load_p_step_ms": 144.3,
        "gb200_64head_64k_recompute_p_step_ms": 150.8,
        "gb200_64head_64k_load_p_peak_gib": 41.2,
        "gb200_64head_64k_recompute_p_peak_gib": 13.7,
        "gb200_64head_64k_peak_mem_reduction_pct": round((41.2 - 13.7) / 41.2 * 100.0, 2),
        "gb200_64head_64k_saved_activations_before_gib": 20.3,
        "gb200_64head_64k_saved_activations_after_gib": 8.0,
    }

    # 2. verl vLLM >= 0.28 multi-node rollout DP leader start_rank fix (#8234)
    verl_multinode_dp = {
        "leader_node_rank": 0,
        "leader_passes_data_parallel_start_rank": False,
        "follower_headless_passes_data_parallel_start_rank": True,
        "gb200_validated_replicas": 16,
        "gb200_validated_gpus_per_replica": 32,
        "gb200_validated_total_nodes": 128,
        "rollout_actor_logprob_pearson": 0.995,
    }

    # 3. Mooncake Tiled COO reshard (#3751), HA SIGTERM lease release (#4814), and SSE4.2 CRC-32C (#4821)
    sample_crc = _crc32c_ref(b"mooncake-phase89-oplog-record")
    mooncake_reshard_and_store = {
        "tiled_coo_max_pairwise_comparisons": 1_000_000,
        "tiled_coo_target_independent_directory": True,
        "ha_shutdown_deadline_seconds": 15,
        "ha_shutdown_unconfirmed_release_exit_status": 2,
        "ha_stops_renewal_guard_before_release": True,
        "crc32c_polynomial_hex": "0x82F63B78",
        "crc32c_slice_by_8_tables": 8,
        "crc32c_sample_digest_uint32": sample_crc,
    }

    # 4. NIXL EP barrier put wait (#2442), Libfabric EFA unsolicited write recv (#2494),
    # and Dynamo KV-Router custom visitor (#17238) & OnceLock CRTC leaf storage (#17259)
    nixl_and_dynamo = {
        "nixl_ep_barrier_polls_put_xfer_status": True,
        "nixl_ep_gb200_cpu_proxy_pass_rate_before": "0/5",
        "nixl_ep_gb200_cpu_proxy_pass_rate_after": "5/5",
        "nixl_efa_unsolicited_write_recv_option_offset": 8,
        "nixl_efa_min_libfabric_header_version": "2.5",
        "dynamo_kv_router_eliminates_serde_flatten_and_path_tracking": True,
        "dynamo_kv_router_strict_boolean_parsing": True,
        "dynamo_crtc_leaf_uses_oncelock_arcswap": True,
        "dynamo_crtc_dashmap_promotion_fanout_threshold": 4,
    }

    return {
        "experiment_id": "fa4_sparse_mla_verl_mooncake_reshard_nixl_ep_dynamo_kvrouter_phase89",
        "phase": 89,
        "priority": 98,
        "track": "C",
        "fa4_sm90_and_sm100": fa4_sm90_and_sm100,
        "verl_multinode_dp": verl_multinode_dp,
        "mooncake_reshard_and_store": mooncake_reshard_and_store,
        "nixl_and_dynamo": nixl_and_dynamo,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic summary.json")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit(f"Missing {out_path}")
        existing = out_path.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit(f"Drift detected in {out_path}")
        print("verified")
        return 0

    out_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
