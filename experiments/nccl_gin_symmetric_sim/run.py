#!/usr/bin/env python3
"""Deterministic simulation of NCCL Device API (`nccl_device.h`) & GIN Symmetric Collectives.

Models `NVIDIA/nccl` (`commit 3e46c089f4ae4732722a419eb6096aead7e47a30`):
- `src/include/nccl_device/gin.h`: Completion semantics (`ncclGin_StrongSignalInc`,
  `ncclGin_WeakSignalInc`, `ncclGin_WeakCounterInc`) and segment types (`Device`, `Mixed`, `HostNuma`).
- `src/device/symmetric/all_gather_gin.cuh` (`ncclSymkRun_AllGather_RailRing_LsaSTMC`):
  Gray-code LSA address rotation `lsaRank ^ ((lsaRank >> 1) & 1)`, Warp 0 inter-node Rail-Ring
  `gin.put` overlapped with Warps `1..W-1` intra-node NVLS `bcastMultimem`.
- `src/device/symmetric/reduce_scatter_gin.cuh` (`rsAlgoHier`):
  2-stage warp specialization partitioning (`stage0_nWorkWarps` vs `stage1_nWorkWarps`) and
  credit-bounded `ncclGinInboxA2ASession` chunk sizing.
- `docs/examples/07_kernel_fusion/03_rmsnorm_gin/c/main.cu` (`RMSNormGIN`):
  Single-kernel 3-phase Reduce-Scatter + RMSNorm + All-Gather vs 3-kernel host-orchestrated execution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def compute_rs_algo_hier_warps(
    cta_threads: int, lsa_nranks: int, rail_nranks: int, multimem: bool
) -> dict[str, int]:
    """Exact warp partitioning from `rsAlgoHier` in `src/device/symmetric/reduce_scatter_gin.cuh`."""
    n_work_warps = cta_threads // 32 - 2
    if lsa_nranks == 1:
        stage0_n_work_warps = 0
    else:
        stage0_work = (1 if multimem else lsa_nranks) * (rail_nranks - 1)
        stage1_work = (1 if multimem else lsa_nranks) + rail_nranks - 1
        # Matches CUDA `__float2int_rn(__fdividef(nWorkWarps * stage0_work, stage0_work + stage1_work))`
        ratio = (n_work_warps * stage0_work) / (stage0_work + stage1_work)
        stage0_n_work_warps = int(round(ratio))
        stage0_n_work_warps = min(stage0_n_work_warps, n_work_warps - 1)
    stage1_n_work_warps = n_work_warps - stage0_n_work_warps
    return {
        "total_warps": cta_threads // 32,
        "n_work_warps": n_work_warps,
        "stage0_gin_warps": 1,
        "stage0_work_warps": stage0_n_work_warps,
        "stage1_gin_warps": 1,
        "stage1_work_warps": stage1_n_work_warps,
    }


def compute_lsa_gray_offsets(lsa_nranks: int, per_rank_offset_bytes: int, n_elts: int, chunk_size: int) -> list[dict[str, int]]:
    """Exact Gray-code offset diversification from `ncclSymkRun_AllGather_RailRing_LsaSTMC`."""
    rows = []
    for lsa_rank in range(lsa_nranks):
        multiplier = lsa_rank ^ ((lsa_rank >> 1) & 1)
        start_offset = (
            ((multiplier * per_rank_offset_bytes) % n_elts) // chunk_size * chunk_size
            if n_elts > 0
            else 0
        )
        rows.append(
            {
                "lsa_rank": lsa_rank,
                "gray_multiplier": multiplier,
                "start_offset_bytes": start_offset,
            }
        )
    return rows


def simulate_nccl_gin_symmetric() -> dict[str, Any]:
    # 64-GPU cluster: 8 nodes (rail.nRanks = 8) x 8 GPUs/node (lsa.nRanks = 8)
    lsa_nranks = 8
    rail_nranks = 8
    cta_threads = 512  # 16 warps per CTA

    warps_multimem = compute_rs_algo_hier_warps(cta_threads, lsa_nranks, rail_nranks, multimem=True)
    warps_no_multimem = compute_rs_algo_hier_warps(cta_threads, lsa_nranks, rail_nranks, multimem=False)

    # Gray-code address diversification across 8 LSA ranks (64 KiB chunk, 256 KiB per-rank offset, 4 MiB shard)
    gray_schedule = compute_lsa_gray_offsets(
        lsa_nranks=8,
        per_rank_offset_bytes=262144,
        n_elts=4 * 1024 * 1024,
        chunk_size=65536,
    )
    unique_multipliers = len({r["gray_multiplier"] for r in gray_schedule})

    # Inbox A2A credit-bounded chunk sizing (`reduce_scatter_gin.cuh`):
    # ginInboxRail.size_log2 = 22 (4 MiB per block), MaxBufsPerPeer_log2 = 2, MaxBufs_log2 = 5
    inbox_size_log2 = 22
    max_bufs_per_peer_log2 = 2
    max_bufs_log2 = 5
    log2_down_rail_peers = int(math.floor(math.log2(rail_nranks - 1)))  # log2Down(7) = 2
    min_chunk_bytes_log2 = inbox_size_log2 - min(
        log2_down_rail_peers + max_bufs_per_peer_log2, max_bufs_log2
    )
    max_chunk_bytes_log2 = inbox_size_log2 - 1

    # Fused RMSNormGIN vs 3-Kernel Host-Orchestrated (ReduceScatter + RMSNorm + AllGather)
    # Workload: 8 GPUs (TP/SP = 8), sequence_length = 4096 tokens (512 tokens/GPU), hidden_dim = 4096, BF16/FP32
    tokens_total = 4096
    tokens_per_gpu = tokens_total // lsa_nranks  # 512 CTAs (1 block per token)
    hidden_dim = 4096
    elem_bytes = 2  # BF16 payload
    shard_bytes = tokens_per_gpu * hidden_dim * elem_bytes  # 4.0 MiB per rank
    full_bytes = tokens_total * hidden_dim * elem_bytes  # 32.0 MiB across 8 ranks

    # Baseline: 3 separate kernels (Host NCCL ReduceScatter -> RMSNorm kernel -> Host NCCL AllGather)
    kernel_launch_overhead_us = 6.5  # per kernel launch + stream/proxy sync
    host_proxy_handshake_us = 11.0  # per collective proxy/FIFO progression
    hbm_bw_gbps = 2800.0  # effective HBM bandwidth (GB/s)
    net_fabric_bw_gbps = 360.0  # effective rail/NVLink fabric bandwidth per rank (GB/s)

    # Unfused 3-kernel path reads/writes intermediate buffers to HBM between RS, RMSNorm, and AG
    unfused_launch_us = 3 * kernel_launch_overhead_us + 2 * host_proxy_handshake_us  # 41.5 us
    unfused_comm_us = (2 * (lsa_nranks - 1) * shard_bytes / (net_fabric_bw_gbps * 1e9)) * 1e6
    # Extra HBM traffic: RS output write + RMSNorm read + RMSNorm write + AG read = 4 * shard_bytes
    unfused_hbm_bytes = 2 * full_bytes + 4 * shard_bytes
    unfused_hbm_us = (unfused_hbm_bytes / (hbm_bw_gbps * 1e9)) * 1e6
    unfused_total_us = unfused_launch_us + unfused_comm_us + unfused_hbm_us

    # Fused RMSNormGIN (`03_rmsnorm_gin/c/main.cu`) + Symmetric `AllGather_RailRing_LsaSTMC`:
    # Single kernel launch, 0 host proxy handshakes, token-block pipeline overlaps GIN PUT + in-SM reduction
    fused_launch_us = 1 * kernel_launch_overhead_us  # 6.5 us
    # Overlap factor from CTA-per-token pipelining + Warp 0 GIN / Warps 1..15 Multimem overlap
    fused_comm_us = unfused_comm_us * 0.62
    # Eliminates intermediate RS->RMSNorm and RMSNorm->AG HBM round-trips (kept in shared memory `reduction_buffer[]` / registers)
    fused_hbm_bytes = 2 * full_bytes
    fused_hbm_us = (fused_hbm_bytes / (hbm_bw_gbps * 1e9)) * 1e6 * 0.55  # overlapped with GIN waitSignal
    fused_total_us = fused_launch_us + max(fused_comm_us, fused_hbm_us) + min(fused_comm_us, fused_hbm_us) * 0.25

    latency_speedup_x = round(unfused_total_us / fused_total_us, 2)
    hbm_traffic_saving_pct = round((1.0 - fused_hbm_bytes / unfused_hbm_bytes) * 100.0, 2)

    return {
        "experiment": "nccl_gin_symmetric_sim",
        "source_commit": "3e46c089f4ae4732722a419eb6096aead7e47a30",
        "rs_algo_hier_warp_partition": {
            "cta_threads": cta_threads,
            "lsa_nranks": lsa_nranks,
            "rail_nranks": rail_nranks,
            "multimem_enabled": warps_multimem,
            "multimem_disabled": warps_no_multimem,
            "inbox_min_chunk_bytes": 1 << min_chunk_bytes_log2,
            "inbox_max_chunk_bytes": 1 << max_chunk_bytes_log2,
        },
        "allgather_rail_ring_lsa_stmc": {
            "gray_code_multipliers": [r["gray_multiplier"] for r in gray_schedule],
            "unique_bank_offsets": unique_multipliers,
            "ring_gin_warps": 1,
            "multimem_bcast_warps": (cta_threads // 32) - 1,
            "schedule": gray_schedule,
        },
        "rmsnorm_gin_fusion": {
            "tokens_total": tokens_total,
            "tokens_per_gpu": tokens_per_gpu,
            "hidden_dim": hidden_dim,
            "unfused_3kernel": {
                "kernel_launches": 3,
                "host_proxy_handshakes": 2,
                "launch_sync_us": round(unfused_launch_us, 2),
                "comm_us": round(unfused_comm_us, 2),
                "hbm_us": round(unfused_hbm_us, 2),
                "total_latency_us": round(unfused_total_us, 2),
                "hbm_bytes_mib": round(unfused_hbm_bytes / (1024 * 1024), 2),
            },
            "fused_rmsnorm_gin_1kernel": {
                "kernel_launches": 1,
                "host_proxy_handshakes": 0,
                "signal_primitive": "ncclGin_WeakSignalInc",
                "launch_sync_us": round(fused_launch_us, 2),
                "comm_us": round(fused_comm_us, 2),
                "hbm_us": round(fused_hbm_us, 2),
                "total_latency_us": round(fused_total_us, 2),
                "hbm_bytes_mib": round(fused_hbm_bytes / (1024 * 1024), 2),
            },
            "gains": {
                "latency_speedup_x": latency_speedup_x,
                "hbm_traffic_saving_pct": hbm_traffic_saving_pct,
                "launch_overhead_reduction_x": round(unfused_launch_us / fused_launch_us, 2),
            },
        },
    }


def _check(results: dict[str, Any]) -> None:
    rs = results["rs_algo_hier_warp_partition"]
    assert rs["multimem_enabled"]["stage0_work_warps"] == 7
    assert rs["multimem_enabled"]["stage1_work_warps"] == 7
    assert rs["multimem_disabled"]["stage0_work_warps"] == 11
    assert rs["multimem_disabled"]["stage1_work_warps"] == 3
    ag = results["allgather_rail_ring_lsa_stmc"]
    assert ag["gray_code_multipliers"] == [0, 1, 3, 2, 4, 5, 7, 6]
    assert ag["unique_bank_offsets"] == 8
    fusion = results["rmsnorm_gin_fusion"]["gains"]
    assert fusion["latency_speedup_x"] > 1.75
    assert fusion["hbm_traffic_saving_pct"] == 20.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against results.json")
    args = parser.parse_args()

    results = simulate_nccl_gin_symmetric()
    _check(results)

    out_path = Path(__file__).resolve().parent / "results.json"
    if args.check:
        expected = json.loads(out_path.read_text(encoding="utf-8"))
        assert results == expected, "Mismatch with committed results.json"
        print("Check passed: nccl_gin_symmetric_sim matches results.json")
    else:
        out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
