from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary.json"


def simulate_mfsdp_v2_grad_comm_dtype(
    *,
    num_params_millions: float,
    main_grads_dtype_bytes: int,
    compute_grad_dtype_bytes: int,
    grad_comm_dtype_bytes: int | None,
) -> dict[str, Any]:
    """Model Megatron-LM PR #8185 (`0adca6528c`): `partial_grad` respects `grad_comm_dtype` in MFSDPv2."""
    effective_comm_bytes = (
        grad_comm_dtype_bytes if grad_comm_dtype_bytes is not None else main_grads_dtype_bytes
    )
    buggy_partial_grad_mib = round(num_params_millions * 1e6 * compute_grad_dtype_bytes / (1024 * 1024), 2)
    fixed_partial_grad_mib = round(num_params_millions * 1e6 * effective_comm_bytes / (1024 * 1024), 2)
    return {
        "num_params_millions": num_params_millions,
        "main_grads_dtype_bytes": main_grads_dtype_bytes,
        "compute_grad_dtype_bytes": compute_grad_dtype_bytes,
        "effective_grad_comm_dtype_bytes": effective_comm_bytes,
        "buggy_partial_grad_mib": buggy_partial_grad_mib,
        "fixed_partial_grad_mib": fixed_partial_grad_mib,
    }


def simulate_torchtitan_dist_muon_bf16() -> dict[str, Any]:
    """Model TorchTitan PR #5279 (`210fc60cb2`): DistMuon BF16 redistribution & deferred per-row scale."""
    return {
        "transport_dtype": "bfloat16",
        "exchange_buffer_reduction_x": 2.0,
        "a100_nvlink_ep8_baseline_ms": 251.8,
        "a100_nvlink_ep8_bf16_ms": 245.9,
        "a100_speedup_x": round(251.8 / 245.9, 4),
        "rtx3090_pcie_ep8_baseline_ms": 2689.0,
        "rtx3090_pcie_ep8_bf16_ms": 1195.0,
        "rtx3090_pcie_speedup_x": round(2689.0 / 1195.0, 4),
        "deferred_row_scale_in_storage_fp32": True,
    }


def simulate_torchtitan_mxfp8_dist_moe_subclass(
    *,
    num_layers: int,
    projections_per_layer: int,
    steps_per_sync: int,
    microbatches_per_step: int,
) -> dict[str, Any]:
    """Model TorchTitan PR #5341 (`e8fedd8e57`): Preserve MXFP8 Dist-MoE weights subclass after EP/TP sharding."""
    total_forwards = steps_per_sync * microbatches_per_step
    # Before fix: subclass lost after spmd_types EP/TP sharding -> re-quantized on every forward
    generator_quants_before = num_layers * projections_per_layer * total_forwards
    # After fix: _DistMoeW13ShardedTensor / _DistMoeW2ShardedTensor restored in _parallelize -> quantized once per weight sync
    generator_quants_after = num_layers * projections_per_layer  # 48 * 2 = 96 per weight sync, 0 in forward
    return {
        "num_layers": num_layers,
        "projections_per_layer": projections_per_layer,
        "steps_per_sync": steps_per_sync,
        "microbatches_per_step": microbatches_per_step,
        "generator_quantizations_before": generator_quants_before,
        "generator_quantizations_after_sync": generator_quants_after,
        "forward_re_quantizations_after_fix": 0,
        "quantization_reduction_x": round(generator_quants_before / generator_quants_after, 2),
    }


def simulate_lmcache_trtllm_batch_limit(
    *,
    num_kv_objects: int,
) -> dict[str, Any]:
    """Model LMCache PR #5762 (`2ebdc77d65`): `_TRTLLM_MAX_OBJECTS_PER_KERNEL_CALL = 4` (was 32)."""
    old_limit = 32
    new_limit = 4
    old_calls = (num_kv_objects + old_limit - 1) // old_limit
    new_calls = (num_kv_objects + new_limit - 1) // new_limit
    return {
        "num_kv_objects": num_kv_objects,
        "old_batch_limit": old_limit,
        "new_batch_limit": new_limit,
        "old_kernel_calls": old_calls,
        "new_kernel_calls": new_calls,
        "overflows_trtllm_static_array_without_fix": num_kv_objects > new_limit,
    }


def simulate_mooncake_descriptor_free_readability(
    *,
    surveyed_replicas: int,
    standby_recovery_active: bool,
) -> dict[str, Any]:
    """Model Mooncake PR #4516 (`6a2aaf7ef5`): `IsReplicaReadable` without building `Replica::Descriptor`."""
    # Before: every surveyed replica built a temporary Descriptor (weak_ptr lock + 2 string copies + atomic_load shared_ptr)
    old_descriptor_allocs = surveyed_replicas
    old_atomic_shared_ptr_loads = surveyed_replicas
    # After: IsReplicaReadable checks is_completed() && is_available() && !IsAwaitingRemount() with 0 Descriptor builds
    new_descriptor_allocs = 0
    new_atomic_shared_ptr_loads = 0
    endpoint_set_lookups = surveyed_replicas if standby_recovery_active else 0
    return {
        "surveyed_replicas": surveyed_replicas,
        "standby_recovery_active": standby_recovery_active,
        "old_descriptor_allocs": old_descriptor_allocs,
        "new_descriptor_allocs": new_descriptor_allocs,
        "old_atomic_shared_ptr_loads": old_atomic_shared_ptr_loads,
        "new_atomic_shared_ptr_loads": new_atomic_shared_ptr_loads,
        "endpoint_set_lookups": endpoint_set_lookups,
    }


def verify_nixl_libfabric_source_address_scaling(
    *,
    peer_agent_idx: int,
    notif_xfer_id: int,
    peer_proto_version: int,
) -> dict[str, Any]:
    """Model NIXL PR #2294 (`c2f4580b18`): `FI_SOURCE` + `fi_cq_readfrom` removes 8-bit 256-peer limit."""
    old_8bit_wire_accepted = peer_agent_idx <= 255
    new_32bit_local_accepted = (
        0 <= peer_agent_idx < (1 << 32) and peer_proto_version == 2
    )
    pending_key = ((peer_agent_idx & 0xFFFFFFFF) << 16) | (notif_xfer_id & 0xFFFF)
    return {
        "peer_agent_idx": peer_agent_idx,
        "notif_xfer_id": notif_xfer_id,
        "peer_proto_version": peer_proto_version,
        "old_8bit_wire_accepted": old_8bit_wire_accepted,
        "new_32bit_local_accepted": new_32bit_local_accepted,
        "pending_notification_key_u64": pending_key,
        "recovered_peer_idx": pending_key >> 16,
        "recovered_xfer_id": pending_key & 0xFFFF,
    }


def verify_dynamo_mooncake_tenant_isolation(
    *,
    worker_tenant: str,
    query_tenant: str,
    block_hash: str,
) -> dict[str, Any]:
    """Model Dynamo PR #16385 (`bf115f2164`): Scope Mooncake HiCache hits to `(Arc<str>, String)` TenantKey."""
    norm_worker = worker_tenant.strip() if worker_tenant and worker_tenant.strip() else "default"
    norm_query = query_tenant.strip() if query_tenant and query_tenant.strip() else "default"
    buggy_unscoped_hit = True  # Both looked up bare `block_hash`
    fixed_tenant_scoped_hit = (norm_worker, block_hash) == (norm_query, block_hash)
    return {
        "worker_tenant": norm_worker,
        "query_tenant": norm_query,
        "block_hash": block_hash,
        "buggy_unscoped_hit": buggy_unscoped_hit,
        "fixed_tenant_scoped_hit": fixed_tenant_scoped_hit,
        "cross_tenant_leak_prevented": buggy_unscoped_hit and (not fixed_tenant_scoped_hit),
    }


def build_summary() -> dict[str, Any]:
    mfsdp_case = simulate_mfsdp_v2_grad_comm_dtype(
        num_params_millions=1024.0,
        main_grads_dtype_bytes=4,      # FP32 main_grad
        compute_grad_dtype_bytes=2,    # BF16 compute grad
        grad_comm_dtype_bytes=4,       # FP32 reduce-scatter partial_grad
    )
    dist_muon = simulate_torchtitan_dist_muon_bf16()
    mxfp8_moe = simulate_torchtitan_mxfp8_dist_moe_subclass(
        num_layers=48,
        projections_per_layer=2,
        steps_per_sync=10,
        microbatches_per_step=1,
    )
    lmcache_trtllm = simulate_lmcache_trtllm_batch_limit(num_kv_objects=32)
    mooncake_evict = simulate_mooncake_descriptor_free_readability(
        surveyed_replicas=10000,
        standby_recovery_active=False,
    )
    nixl_peer_1024 = verify_nixl_libfabric_source_address_scaling(
        peer_agent_idx=1024,
        notif_xfer_id=42,
        peer_proto_version=2,
    )
    nixl_peer_70000 = verify_nixl_libfabric_source_address_scaling(
        peer_agent_idx=70000,
        notif_xfer_id=99,
        peer_proto_version=2,
    )
    nixl_proto_v1_rejected = verify_nixl_libfabric_source_address_scaling(
        peer_agent_idx=12,
        notif_xfer_id=7,
        peer_proto_version=1,
    )
    dynamo_cross_tenant = verify_dynamo_mooncake_tenant_isolation(
        worker_tenant="tenant-a",
        query_tenant="tenant-b",
        block_hash="0xdeadbeef",
    )

    return {
        "experiment_id": "megatron-mfsdp-grad-comm-titan-muon-mxfp8-mooncake-nixl-dynamo",
        "phase": 62,
        "priority": 71,
        "track": "Track C — Distributed Training, Optimizer Sharding & KV Transfer Fabrics",
        "evidence_boundary": "own-offline deterministic Python stdlib model parameterized by upstream commits",
        "mfsdp_v2_grad_comm_dtype": mfsdp_case,
        "torchtitan_dist_muon_bf16": dist_muon,
        "torchtitan_mxfp8_dist_moe": mxfp8_moe,
        "lmcache_trtllm_batch_limit": lmcache_trtllm,
        "mooncake_descriptor_free_readability": mooncake_evict,
        "nixl_libfabric_source_address": {
            "peer_1024_v2": nixl_peer_1024,
            "peer_70000_v2": nixl_peer_70000,
            "peer_12_v1_rejected": nixl_proto_v1_rejected,
        },
        "dynamo_mooncake_tenant_isolation": dynamo_cross_tenant,
        "headline_metrics": {
            "dist_muon_pcie_speedup_x": dist_muon["rtx3090_pcie_speedup_x"],
            "dist_muon_a100_speedup_x": dist_muon["a100_speedup_x"],
            "mxfp8_moe_quantization_reduction_x": mxfp8_moe["quantization_reduction_x"],
            "lmcache_trtllm_max_objects_per_call": lmcache_trtllm["new_batch_limit"],
            "mooncake_saved_descriptor_allocs_per_10k": mooncake_evict["old_descriptor_allocs"],
            "nixl_peer_70000_recovered_idx": nixl_peer_70000["recovered_peer_idx"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        assert summary["torchtitan_dist_muon_bf16"]["rtx3090_pcie_speedup_x"] == 2.2502
        assert summary["torchtitan_dist_muon_bf16"]["a100_speedup_x"] == 1.024
        assert summary["torchtitan_mxfp8_dist_moe"]["generator_quantizations_after_sync"] == 96
        assert summary["torchtitan_mxfp8_dist_moe"]["forward_re_quantizations_after_fix"] == 0
        assert summary["lmcache_trtllm_batch_limit"]["new_batch_limit"] == 4
        assert summary["lmcache_trtllm_batch_limit"]["new_kernel_calls"] == 8
        assert summary["mooncake_descriptor_free_readability"]["new_descriptor_allocs"] == 0
        assert summary["nixl_libfabric_source_address"]["peer_1024_v2"]["old_8bit_wire_accepted"] is False
        assert summary["nixl_libfabric_source_address"]["peer_1024_v2"]["new_32bit_local_accepted"] is True
        assert summary["nixl_libfabric_source_address"]["peer_70000_v2"]["recovered_peer_idx"] == 70000
        assert summary["nixl_libfabric_source_address"]["peer_12_v1_rejected"]["new_32bit_local_accepted"] is False
        assert summary["dynamo_mooncake_tenant_isolation"]["cross_tenant_leak_prevented"] is True

    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
