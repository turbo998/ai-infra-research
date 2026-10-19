#!/usr/bin/env python3
"""Deterministic simulation of DeepEP V2.5 HybridElasticSymmetricMemory & Engram GPU-Initiated RDMA Read.

Models:
1. `HybridElasticSymmetricMemory` (`csrc/kernels/comm/symmetric.hpp`):
   - Reserves contiguous VA of `num_sym_bytes + num_rdma_storage_bytes * num_scaleup_ranks`.
   - Maps GPU symmetric segment `[0, num_sym_bytes)` and imports all intra-node (`num_scaleup_ranks = 8`)
     CPU NUMA / GPU RDMA storage shards via Linux `pidfd_open` + `pidfd_getfd` + `cuMemImportFromShareableHandle`
     + `cuMemMap`, enabling a 100+ GiB parametric N-gram table per node at zero HBM capacity tax when
     `use_cpu_rdma_storage = True`.
2. `engram_fetch_impl` (`deep_ep/include/deep_ep/impls/engram/engram_fetch.cuh`):
   - Compares CPU-staged scatter-gather vs Uncoalesced GPU RDMA Read vs Warp-Ballot Coalesced
     (`__ballot_sync` + circular `pending_requests[2*32]` + `ncclGinOptFlagsAggregateRequests` doorbell
     threshold `kFlushDepth / num_active_peer_context_warps`) + atomic last-warp `gin.flush_async`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_deepep_engram_hybrid_symmetric() -> dict[str, Any]:
    num_nodes = 8
    gpus_per_node = 8
    total_gpus = num_nodes * gpus_per_node

    # Symmetric memory & Engram table parameters per rank
    gpu_sym_bytes_gib = 4.0
    engram_shard_per_rank_gib = 16.0  # 128 GiB per 8-GPU node, 1,024 GiB (1 TiB) cluster-wide table
    node_mapped_va_gib = gpu_sym_bytes_gib + engram_shard_per_rank_gib * gpus_per_node  # 132 GiB contiguous VA/rank

    # Lookup workload: BS=64 tokens/rank, kNumEntriesPerToken=8 N-gram entries/token, kNumLayers=4 Engram layers
    num_tokens = 64
    entries_per_token = 8
    num_layers = 4
    hidden_bytes = 512  # FP8 + SF pack per N-gram embedding entry
    total_entries_per_layer = num_tokens * entries_per_token  # 512 entries/layer
    total_entries_all_layers = total_entries_per_layer * num_layers  # 2,048 entries/step

    # Traffic distribution across 8 nodes: 1/8 local node (NVLink / local NUMA), 7/8 remote RDMA peers
    local_fraction = 1.0 / num_nodes
    remote_fraction = 1.0 - local_fraction
    remote_entries = int(round(total_entries_all_layers * remote_fraction))  # 1,792 remote 512B reads

    # Doorbell & network parameters (400 Gbps CX7/CX8 per GPU = 45.0 GB/s effective payload, 28 Mmsg/s doorbell limit)
    nic_bw_gbps = 45.0
    base_rdma_read_rtt_us = 2.40
    uncoalesced_doorbell_rate_mops = 6.5  # PCIe MMIO 64B doorbell bottleneck without aggregation
    coalesced_flush_depth = 32  # kWarpSize = 32 circular batch before ringing doorbell
    active_warps_per_peer_ctx = 2
    doorbell_threshold = coalesced_flush_depth // active_warps_per_peer_ctx  # 16 requests per doorbell

    # 1. Pure HBM Symmetric Memory (GPUSymmetricMemory): OOMs or steals 16 GiB HBM per GPU
    hbm_capacity_gib = 80.0
    pure_gpu_hbm_tax_gib = gpu_sym_bytes_gib + engram_shard_per_rank_gib  # 20.0 GiB/GPU
    hybrid_cpu_rdma_hbm_tax_gib = gpu_sym_bytes_gib  # 4.0 GiB/GPU (16 GiB offloaded to pinned Host NUMA VMM)
    hbm_saved_per_gpu_gib = pure_gpu_hbm_tax_gib - hybrid_cpu_rdma_hbm_tax_gib
    hbm_saved_cluster_gib = hbm_saved_per_gpu_gib * total_gpus

    # 2. Fetch latency comparison across 4 Engram layers (2,048 * 512B = 1.0 MiB/rank)
    payload_bytes_remote = remote_entries * hidden_bytes  # 917,504 B
    payload_wire_us = (payload_bytes_remote / (nic_bw_gbps * 1e9)) * 1e6  # 20.389 us

    # (A) CPU Proxy Scatter-Gather + AllToAll: kernel launch + D2H indices + CPU thread pool + H2D
    cpu_proxy_overhead_per_layer_us = 95.0
    cpu_proxy_total_us = num_layers * cpu_proxy_overhead_per_layer_us + (
        remote_entries / uncoalesced_doorbell_rate_mops
    ) + payload_wire_us

    # (B) Uncoalesced GPU IBGDA gin.get (rings doorbell on every single 512B entry, no AggregateRequests)
    uncoalesced_doorbells = remote_entries
    uncoalesced_issue_us = uncoalesced_doorbells / uncoalesced_doorbell_rate_mops
    uncoalesced_total_us = base_rdma_read_rtt_us * num_layers + uncoalesced_issue_us + payload_wire_us

    # (C) DeepEP Engram Warp-Ballot Coalesced gin.get (ncclGinOptFlagsAggregateRequests + last-warp flush_async)
    # Coalesces up to doorbell_threshold=16 requests per WQE doorbell ring + 1 final flush per (layer, peer_ctx)
    num_rdma_peers = num_nodes - 1  # 7 remote nodes
    num_qps = 2
    num_peer_contexts = num_rdma_peers * num_qps  # 14 peer contexts
    coalesced_doorbells = math.ceil(remote_entries / doorbell_threshold) + (num_layers * num_peer_contexts)
    coalesced_issue_us = coalesced_doorbells / (uncoalesced_doorbell_rate_mops * 2.4)  # multi-QP parallel issue
    # Atomic last-warp flush_async avoids (active_warps - 1) redundant CQE waits per peer_ctx
    redundant_flushes_avoided = num_layers * num_peer_contexts * (active_warps_per_peer_ctx - 1)
    coalesced_total_us = base_rdma_read_rtt_us + coalesced_issue_us + payload_wire_us

    doorbell_reduction_x = round(uncoalesced_doorbells / coalesced_doorbells, 2)
    speedup_vs_cpu_proxy_x = round(cpu_proxy_total_us / coalesced_total_us, 2)
    speedup_vs_uncoalesced_x = round(uncoalesced_total_us / coalesced_total_us, 2)

    return {
        "experiment": "deepep_engram_hybrid_symmetric",
        "cluster_config": {
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "total_gpus": total_gpus,
            "engram_table_total_tib": round((engram_shard_per_rank_gib * total_gpus) / 1024.0, 2),
            "contiguous_va_per_rank_gib": node_mapped_va_gib,
            "num_tokens": num_tokens,
            "entries_per_token": entries_per_token,
            "num_layers": num_layers,
            "entry_hidden_bytes": hidden_bytes,
        },
        "symmetric_vmm_memory": {
            "pure_gpu_sym_hbm_gib": pure_gpu_hbm_tax_gib,
            "hybrid_elastic_gpu_hbm_gib": hybrid_cpu_rdma_hbm_tax_gib,
            "host_numa_rdma_shard_per_rank_gib": engram_shard_per_rank_gib,
            "intra_node_mapped_storage_gib": engram_shard_per_rank_gib * gpus_per_node,
            "hbm_saved_per_gpu_gib": hbm_saved_per_gpu_gib,
            "hbm_saved_cluster_gib": hbm_saved_cluster_gib,
            "uses_pidfd_getfd_zero_copy_import": True,
            "nccl_elastic_buffer_register_enabled": True,
        },
        "engram_rdma_fetch": {
            "remote_entries_per_step": remote_entries,
            "remote_payload_bytes": payload_bytes_remote,
            "doorbell_threshold": doorbell_threshold,
            "uncoalesced_doorbells": uncoalesced_doorbells,
            "coalesced_doorbells": coalesced_doorbells,
            "doorbell_reduction_x": doorbell_reduction_x,
            "redundant_flushes_avoided": redundant_flushes_avoided,
            "cpu_proxy_scatter_gather_us": round(cpu_proxy_total_us, 2),
            "uncoalesced_gpu_ibgda_us": round(uncoalesced_total_us, 2),
            "deepep_engram_coalesced_us": round(coalesced_total_us, 2),
            "speedup_vs_cpu_proxy_x": speedup_vs_cpu_proxy_x,
            "speedup_vs_uncoalesced_x": speedup_vs_uncoalesced_x,
        },
    }


def _check(results: dict[str, Any]) -> None:
    vmm = results["symmetric_vmm_memory"]
    fetch = results["engram_rdma_fetch"]
    assert vmm["hbm_saved_per_gpu_gib"] == 16.0
    assert vmm["hbm_saved_cluster_gib"] == 1024.0
    assert vmm["intra_node_mapped_storage_gib"] == 128.0
    assert fetch["doorbell_reduction_x"] > 10.0
    assert fetch["deepep_engram_coalesced_us"] < 35.0
    assert fetch["speedup_vs_cpu_proxy_x"] > 20.0
    assert fetch["speedup_vs_uncoalesced_x"] > 9.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate deterministic invariants")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON results")
    args = parser.parse_args()

    results = simulate_deepep_engram_hybrid_symmetric()
    if args.check:
        _check(results)
        print("deepep_engram_hybrid_symmetric: OK")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if not args.check and not args.output:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
