#!/usr/bin/env python3
"""Deterministic simulation of Mooncake Transfer Engine ConnectX-8 Data Direct DMA-BUF & BatchRdmaDeviceCache.

Models `kvcache-ai/Mooncake` (`commit 1f65cf13c9f886f8db8b88b7733367b18340ebf0`):
- `mooncake-transfer-engine/include/transport/rdma_transport/data_direct_topology.h` & `.cpp`:
  ConnectX-8 (`CX8`, 800 Gbps) "Data Direct" companion PCI function (`mlx5dv_get_data_direct_sysfs_path@MLX5_1.25`
  + `mlx5dv_reg_dmabuf_mr` with `MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT`) pairs a NIC with GPUs under the exact
  same PCI root (`firstPciRootSegment`, e.g., `"pci0009:00"`), failing with `ENOTSUPP` on cross-root GPUs.
- `mooncake-transfer-engine/include/topology.h`: `selectDeviceByLocalHca` (`resolved_hca_peer_affinity_by_local_`).
- `mooncake-transfer-engine/include/transport/rdma_transport/rdma_batch_cache.h`: `BatchRdmaDeviceCache::select`
  (`bufferCoversRange` zero-lookup fast path across paged KV blocks).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_mooncake_cx8_data_direct_rdma() -> dict[str, Any]:
    # 64K-token disaggregated KV transfer across 8 GPUs (e.g., 16 KiB per paged KV block, 4,096 blocks total = 64 MiB/rank)
    num_kv_blocks = 4096
    block_bytes = 16384  # 16 KiB paged KV block
    total_kv_bytes = num_kv_blocks * block_bytes  # 67,108,864 B (64.0 MiB)

    # PCIe roots on an 8-GPU node: 4 PCI roots ("pci0009:00", "pci000a:00", "pci000b:00", "pci000c:00"), 2 GPUs + 2 CX8 NICs per root
    num_pci_roots = 4
    cx8_line_rate_gbps = 92.0  # 800 Gbps ConnectX-8 Data Direct effective payload bandwidth (GB/s)
    standard_gpudirect_pcie_gbps = 56.0  # Standard PCIe Gen5 x16 GPUDirect RDMA without CX8 Data Direct companion engine
    cross_root_qpi_hairpin_gbps = 24.5  # Cross-PCI-root CPU Root Complex / UPI hairpin bandwidth

    # CPU metadata resolution overhead per block in submitTransfer:
    # Uncached binary-tree / range-index + topology resolve vs BatchRdmaDeviceCache (`bufferCoversRange` O(1) hit)
    uncached_resolve_ns_per_block = 420.0  # 420 ns per 16 KiB block = 1.72 ms across 4,096 blocks
    cached_resolve_ns_per_block = 12.0  # snapshot_ == desc && bufferCoversRange fast path = 0.049 ms

    # 1. Topology-Unaware Round-Robin (75% cross-root misses -> ENOTSUPP on Data Direct, falls back to cross-root hairpin)
    cross_root_miss_rate = (num_pci_roots - 1) / num_pci_roots  # 0.75
    rr_eff_bw_gbps = 1.0 / (
        (1.0 - cross_root_miss_rate) / standard_gpudirect_pcie_gbps
        + cross_root_miss_rate / cross_root_qpi_hairpin_gbps
    )
    rr_cpu_submit_ms = (num_kv_blocks * uncached_resolve_ns_per_block) / 1e6
    rr_wire_ms = (total_kv_bytes / (rr_eff_bw_gbps * 1e9)) * 1e3
    rr_total_ms = rr_cpu_submit_ms + rr_wire_ms

    # 2. PCI-Root Aligned Standard GPUDirect RDMA (Uncached per-block resolve, no CX8 Data Direct)
    std_cpu_submit_ms = (num_kv_blocks * uncached_resolve_ns_per_block) / 1e6
    std_wire_ms = (total_kv_bytes / (standard_gpudirect_pcie_gbps * 1e9)) * 1e3
    std_total_ms = std_cpu_submit_ms + std_wire_ms

    # 3. Mooncake CX8 Data Direct (`firstPciRootSegment` matched `MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT`) + `BatchRdmaDeviceCache`
    # First block resolves & populates snapshot_; blocks 2..4096 hit `bufferCoversRange` (99.976% hit rate)
    cache_hits = num_kv_blocks - 1
    cache_hit_rate_pct = (cache_hits / num_kv_blocks) * 100.0
    cx8_cpu_submit_ms = (
        uncached_resolve_ns_per_block + cache_hits * cached_resolve_ns_per_block
    ) / 1e6
    cx8_wire_ms = (total_kv_bytes / (cx8_line_rate_gbps * 1e9)) * 1e3
    cx8_total_ms = cx8_cpu_submit_ms + cx8_wire_ms

    submit_cpu_speedup_x = round(std_cpu_submit_ms / cx8_cpu_submit_ms, 2)
    total_speedup_vs_rr_x = round(rr_total_ms / cx8_total_ms, 2)
    total_speedup_vs_std_x = round(std_total_ms / cx8_total_ms, 2)

    return {
        "experiment": "mooncake_cx8_data_direct_rdma",
        "workload": {
            "num_kv_blocks": num_kv_blocks,
            "block_bytes": block_bytes,
            "total_kv_mib": round(total_kv_bytes / (1024 * 1024), 2),
            "num_pci_roots": num_pci_roots,
        },
        "modes": {
            "topology_unaware_round_robin": {
                "cross_root_enotsupp_rate_pct": round(cross_root_miss_rate * 100.0, 2),
                "effective_bw_gbps": round(rr_eff_bw_gbps, 2),
                "cpu_submit_ms": round(rr_cpu_submit_ms, 3),
                "wire_transfer_ms": round(rr_wire_ms, 3),
                "total_transfer_ms": round(rr_total_ms, 3),
            },
            "pci_root_standard_gpudirect_uncached": {
                "effective_bw_gbps": standard_gpudirect_pcie_gbps,
                "cpu_submit_ms": round(std_cpu_submit_ms, 3),
                "wire_transfer_ms": round(std_wire_ms, 3),
                "total_transfer_ms": round(std_total_ms, 3),
            },
            "mooncake_cx8_data_direct_batch_cached": {
                "pci_root_match_segment": "pci0009:00",
                "mlx5_symbol": "mlx5dv_get_data_direct_sysfs_path@MLX5_1.25",
                "dmabuf_flag": "MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT",
                "batch_cache_hit_rate_pct": round(cache_hit_rate_pct, 3),
                "effective_bw_gbps": cx8_line_rate_gbps,
                "cpu_submit_ms": round(cx8_cpu_submit_ms, 3),
                "wire_transfer_ms": round(cx8_wire_ms, 3),
                "total_transfer_ms": round(cx8_total_ms, 3),
            },
        },
        "gains": {
            "submit_cpu_speedup_x": submit_cpu_speedup_x,
            "total_speedup_vs_round_robin_x": total_speedup_vs_rr_x,
            "total_speedup_vs_std_gpudirect_x": total_speedup_vs_std_x,
        },
    }


def _check(results: dict[str, Any]) -> None:
    modes = results["modes"]
    gains = results["gains"]
    assert modes["topology_unaware_round_robin"]["cross_root_enotsupp_rate_pct"] == 75.0
    assert modes["mooncake_cx8_data_direct_batch_cached"]["batch_cache_hit_rate_pct"] > 99.9
    assert gains["submit_cpu_speedup_x"] > 34.0
    assert gains["total_speedup_vs_round_robin_x"] > 4.8
    assert gains["total_speedup_vs_std_gpudirect_x"] > 3.6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate deterministic invariants")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON results")
    args = parser.parse_args()

    results = simulate_mooncake_cx8_data_direct_rdma()
    if args.check:
        _check(results)
        print("mooncake_cx8_data_direct_rdma: OK")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if not args.check and not args.output:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
