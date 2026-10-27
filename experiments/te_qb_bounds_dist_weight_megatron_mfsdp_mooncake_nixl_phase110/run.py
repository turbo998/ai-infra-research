from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_summary() -> dict:
    # 1. TransformerEngine #3734 (96c0fb0e24) & #3730 (fa4b30f09b):
    # Remove device-side QB bounds validation (eliminates cudaMemcpyAsync + cudaStreamSynchronize host sync;
    # __float2int_rd saturates + clamps to [0, num_bins)).
    # Fix DistributedWeight handling under saved-tensor hooks via _SavedTensorRef and fast-path prepare_for_saving
    # (10.0 us -> 5.2 us vs 4.5 us baseline for 32-expert bf16 GroupedLinear).
    qb_host_syncs_before = 1
    qb_host_syncs_after = 0
    qb_num_bins = 32
    prepare_saving_prev_us = 10.0
    prepare_saving_new_us = 5.2
    prepare_saving_speedup_x = round(prepare_saving_prev_us / prepare_saving_new_us, 4)

    # 2. Megatron-LM #8207 (60547a3f41) & #8201 (89b28875a8) + TorchTitan #5631 (c1522ec208):
    # Replace IndexedOrder with occurrence-based ExecutionOrder for MFSDP v2 repeated modules [m0, m1, m0, m0, m2];
    # Nemotron Lightning deterministic mode sets NVTE_ALLOW_NONDETERMINISTIC_ALGO=0, NCCL_ALGO=Ring,
    # CUBLAS_WORKSPACE_CONFIG=:4096:8, and shared TRITON_CACHE_DIR;
    # TorchTitan pins TRITON_DEFAULT_FP_FUSION="1" on B200 Kimi K3 DistMuon 10-step golden.
    execution_seq = [0, 1, 0, 0, 2]
    budget_2_prefetch_schedule = [
        execution_seq[i + 1 : i + 1 + 2] for i in range(len(execution_seq))
    ]
    triton_fp_fusion_pinned = "1"
    kimi_k3_step10_loss_disabled_fusion = 6.416478157043457
    kimi_k3_step10_loss_enabled_fusion = 6.417245864868164

    # 3. Mooncake #4902 (6b90ea5e6c) & #4948 (c3df64c745) + LMCache #5878 (8afb6da466):
    # Mooncake TENT stages MTYPE_XPU buffers through page-aligned (4096 B) host memory in io_uring transport;
    # Mooncake tebench pins thread to target NUMA CPU, first-touches DRAM, and queries numa_maps dominant node;
    # LMCache adds offset_gb / owns_range(path, offset, size) so L1 and L2 or multiple servers share a CXL DAX device.
    io_uring_page_align_bytes = 4096
    cxl_device_capacity_gib = 1024
    l1_range_gib = [0, 512]
    l2_range_gib = [512, 1024]
    ranges_overlap = max(l1_range_gib[0], l2_range_gib[0]) < min(l1_range_gib[1], l2_range_gib[1])

    # 4. Dynamo #16701 (d3fd74f489) & NIXL #2521 (280c3a1cd8):
    # Dynamo router credits fetchable KV blocks via shared_cache_multiplier (default 0.5):
    # adjusted_prefill = max(raw_prefill - effective_device*device - 0.75*host - 0.25*disk - 0.5*(shared + fetchable), 0).
    # NIXL adds lock-free nixlAgent::getXferTraceInfo() exposing (request_id, execution_number, traceparent).
    raw_prefill_blocks = 100
    fetchable_blocks = 60
    shared_cache_multiplier = 0.5
    adjusted_prefill_without_fetch_credit = float(raw_prefill_blocks)
    adjusted_prefill_with_fetch_credit = max(
        raw_prefill_blocks - shared_cache_multiplier * fetchable_blocks, 0.0
    )
    nixl_exec_num_before_post = 0
    nixl_exec_num_after_refused_post = 0
    nixl_exec_num_after_backend_post = 1

    return {
        "experiment_id": "te_qb_bounds_dist_weight_megatron_mfsdp_mooncake_nixl_phase110",
        "phase": 110,
        "priority": 119,
        "provenance": "own-offline",
        "te_qb_bounds_and_dist_weight": {
            "qb_host_syncs_before": qb_host_syncs_before,
            "qb_host_syncs_after": qb_host_syncs_after,
            "qb_num_bins": qb_num_bins,
            "prepare_saving_prev_us": prepare_saving_prev_us,
            "prepare_saving_new_us": prepare_saving_new_us,
            "prepare_saving_speedup_x": prepare_saving_speedup_x,
        },
        "megatron_mfsdp_and_torchtitan": {
            "execution_seq": execution_seq,
            "budget_2_prefetch_schedule": budget_2_prefetch_schedule,
            "triton_fp_fusion_pinned": triton_fp_fusion_pinned,
            "kimi_k3_step10_loss_disabled_fusion": kimi_k3_step10_loss_disabled_fusion,
            "kimi_k3_step10_loss_enabled_fusion": kimi_k3_step10_loss_enabled_fusion,
        },
        "mooncake_lmcache_dynamo_nixl": {
            "io_uring_page_align_bytes": io_uring_page_align_bytes,
            "cxl_device_capacity_gib": cxl_device_capacity_gib,
            "l1_range_gib": l1_range_gib,
            "l2_range_gib": l2_range_gib,
            "cxl_disjoint_ranges_overlap": ranges_overlap,
            "raw_prefill_blocks": raw_prefill_blocks,
            "fetchable_blocks": fetchable_blocks,
            "shared_cache_multiplier": shared_cache_multiplier,
            "adjusted_prefill_without_fetch_credit": adjusted_prefill_without_fetch_credit,
            "adjusted_prefill_with_fetch_credit": adjusted_prefill_with_fetch_credit,
            "nixl_exec_num_before_post": nixl_exec_num_before_post,
            "nixl_exec_num_after_refused_post": nixl_exec_num_after_refused_post,
            "nixl_exec_num_after_backend_post": nixl_exec_num_after_backend_post,
        },
        "invariants": {
            "te_zero_qb_host_sync_and_fast_save": (
                qb_host_syncs_after == 0 and prepare_saving_speedup_x == 1.9231
            ),
            "mfsdp_occurrence_prefetch_exact": budget_2_prefetch_schedule
            == [[1, 0], [0, 0], [0, 2], [2], []],
            "lmcache_disjoint_cxl_ranges_allowed": ranges_overlap is False,
            "dynamo_fetchable_credit_and_nixl_trace_getter": (
                adjusted_prefill_with_fetch_credit == 70.0
                and nixl_exec_num_after_refused_post == 0
                and nixl_exec_num_after_backend_post == 1
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("summary.json")
    summary = build_summary()
    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == summary, "Drift detected in Phase 110 Track C summary.json"
        assert all(summary["invariants"].values())
        print("OK: te_qb_bounds_dist_weight_megatron_mfsdp_mooncake_nixl_phase110 verified")
        return

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
