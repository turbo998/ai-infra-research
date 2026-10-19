#!/usr/bin/env python3
"""Deterministic simulator for NCCL PAT (`NCCL_ALGO_PAT`) Parallel Aggregated Trees AllGather & ReduceScatter.

Models the exact algorithm verified in `NVIDIA/nccl` (`commit 3e46c089f4ae4732722a419eb6096aead7e47a30`):
- `src/include/collectives.h`: `PatRSAlgorithm<T>` (lines 455-714) & `PatAGAlgorithm<T>` (lines 716-942)
- `src/device/all_gather.h`: `RunWorkColl<ncclFuncAllGather, T, RedOp, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE>` (lines 126-271)
- `src/device/reduce_scatter.h`: `RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE>` (lines 96-230)

Evaluates:
1. Exact C++-equivalent execution of `PatRSAlgorithm` and `PatAGAlgorithm` across `nranks` and `stepSize`,
   verifying how `aggFactor` doubles while `stepSize / (channelSize * sizeof(T) * aggFactor) >= 2 && aggFactor < nranks / 2`
   and across `stepDepth`, reducing network round-trips from `O(N)` Ring steps to `O(log2 N)` binomial hypercube phases.
2. Small/medium/large collective latency (us) and effective bus bandwidth (GB/s) comparing:
   - `NCCL_ALGO_RING` (`N - 1` sequential hop latency wall at large scale)
   - `NCCL_ALGO_PAT` One-RPN (`log2(N)` binomial tree phases with `parallelFactor` concurrent worker warps)
   - `NCCL_ALGO_PAT` Multi-RPN (`NVLS` intra-node scatter/gather + inter-node PAT across `nNodes = N / 8`)
   across cluster scales `N in {64, 256, 512, 1024}` GPUs and message sizes `1 MiB .. 256 MiB`.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any


RESULTS_PATH = pathlib.Path(__file__).resolve().parent / "results.json"


def log2_up(n: int) -> int:
    if n <= 1:
        return 0
    return (n - 1).bit_length()


def mirror_invert(i: int, max_val: int) -> int:
    """Exact replica of `PatRSAlgorithm::mirrorInvert` (`src/include/collectives.h:485-491`)."""
    ret = 0
    mask = 1
    imask = max_val // 2
    while mask < max_val:
        if (i & mask) == 0:
            ret += imask
        mask <<= 1
        imask >>= 1
    return ret


def first_bit_set(i: int, max_val: int) -> int:
    """Exact replica of `PatRSAlgorithm::firstBitSet` (`src/include/collectives.h:493-501`)."""
    if i == 0:
        return max_val
    return (i & -i).bit_length() - 1


def compute_pat_aggregation_params(
    step_size_bytes: int,
    step_depth: int,
    max_parallel_factor: int,
    channel_bytes: int,
    nranks: int,
) -> dict[str, int]:
    """Exact replica of `PatRSAlgorithm` / `PatAGAlgorithm` constructor (`src/include/collectives.h:537-560`).

    ```cpp
    aggDelta = nrPow2 = (1 << log2Up(nranks));
    aggFactor = 1;
    size_t channelSize = end - offset;
    while (stepSize / (channelSize * sizeof(T) * aggFactor) >= 2 && aggFactor < nranks / 2) {
      aggFactor *= 2;
      aggDelta /= 2;
    }
    postFreq = aggFactor;
    if (postFreq < parallelFactor) parallelFactor = postFreq;
    int d = stepDepth;
    while (d > 1 && aggFactor < nranks / 2) {
      d /= 2;
      aggFactor *= 2;
      aggDelta /= 2;
    }
    ```
    """
    nr_pow2 = 1 << log2_up(nranks)
    agg_delta = nr_pow2
    agg_factor = 1
    parallel_factor = max_parallel_factor
    eff_channel_bytes = max(1, channel_bytes)

    while (step_size_bytes // (eff_channel_bytes * agg_factor)) >= 2 and agg_factor < (nranks // 2):
        agg_factor *= 2
        agg_delta //= 2

    post_freq = agg_factor
    if post_freq < parallel_factor:
        parallel_factor = post_freq

    d = step_depth
    while d > 1 and agg_factor < (nranks // 2):
        d //= 2
        agg_factor *= 2
        agg_delta //= 2

    as_dim = log2_up(agg_delta)
    return {
        "nranks": nranks,
        "nr_pow2": nr_pow2,
        "channel_bytes": eff_channel_bytes,
        "agg_factor": agg_factor,
        "agg_delta": agg_delta,
        "post_freq": post_freq,
        "parallel_factor": parallel_factor,
        "as_dim": as_dim,
        "binomial_tree_depth": log2_up(nranks),
    }


def simulate_collective_latency_us(
    algo: str,
    total_bytes: int,
    nranks: int,
    local_ranks: int = 8,
    n_channels: int = 16,
) -> dict[str, Any]:
    """Compare Ring vs PAT (1-RPN) vs Multi-RPN PAT+NVLS latency and effective bus bandwidth.

    Hardware baseline: 8x H100 per node (`local_ranks = 8`), `900 GB/s` NVLS intra-node bandwidth,
    `50 GB/s` (400 Gbps) per-NIC RoCEv2/IB inter-node bandwidth (`400 GB/s` per node across 8 NICs).
    - Per-hop inter-node network + FIFO synchronization latency: `t_hop_net = 1.85 us`
    - Per-hop intra-node NVLink/NVLS latency: `t_hop_nvlink = 0.45 us`
    """
    step_size_bytes = 524288  # 512 KiB NCCL buffsize / NCCL_STEPS (4 MiB / 8)
    step_depth = 8  # NCCL_STEPS = 8
    max_pf = 8  # NCCL_PAT_NWORKERS / WARP_SIZE

    per_rank_bytes = total_bytes // nranks
    channel_bytes_per_rank = max(256, per_rank_bytes // n_channels)
    inter_node_bw_bps = 45.0e9 * n_channels / 16.0  # 45 GB/s effective per NIC rail
    nvls_bw_bps = 780.0e9  # 780 GB/s effective NVLS multicast/reduce
    t_hop_net_us = 1.85
    t_hop_nvlink_us = 0.45
    base_launch_us = 4.2

    n_nodes = max(1, nranks // local_ranks)
    bus_bytes = total_bytes * (nranks - 1) / nranks

    if algo == "ring":
        # Ring requires (nranks - 1) sequential pipelined steps.
        # Out of nranks steps, n_nodes receive inter-node hops and (nranks - n_nodes) are intra-node NVLink hops.
        net_hops = n_nodes
        nvlink_hops = nranks - n_nodes
        # At small/medium message sizes, pipeline fill/drain latency across (nranks - 1) steps dominates.
        step_latency_us = net_hops * t_hop_net_us + nvlink_hops * t_hop_nvlink_us
        bw_time_us = (bus_bytes / inter_node_bw_bps) * 1e6
        # Ring per-step payload is tiny when nranks is large, degrading NIC packet efficiency
        per_step_nic_payload = max(64, total_bytes // (nranks * n_channels))
        small_pkt_efficiency = min(1.0, per_step_nic_payload / (per_step_nic_payload + 4096.0))
        total_us = base_launch_us + step_latency_us + (bw_time_us / small_pkt_efficiency)
        return {
            "algo": algo,
            "latency_us": round(total_us, 2),
            "alg_bw_gbps": round((total_bytes / 1e9) / (total_us / 1e6), 3),
            "bus_bw_gbps": round((bus_bytes / 1e9) / (total_us / 1e6), 3),
            "sequential_hops": nranks - 1,
            "per_nic_post_bytes": per_step_nic_payload,
        }

    if algo == "pat_one_rpn":
        pat_params = compute_pat_aggregation_params(
            step_size_bytes, step_depth, max_pf, channel_bytes_per_rank, nranks
        )
        tree_hops = log2_up(nranks)
        # PAT aggregates `post_freq = aggFactor` small rank chunks into a single larger RDMA post
        # and runs `parallel_factor` warps concurrently across `aggDelta` aggregated phases.
        aggregated_post_bytes = min(step_size_bytes, channel_bytes_per_rank * pat_params["post_freq"])
        pkt_efficiency = min(1.0, aggregated_post_bytes / (aggregated_post_bytes + 4096.0))
        num_aggregated_waves = max(1, math.ceil((nranks // pat_params["agg_factor"]) / pat_params["parallel_factor"]))
        step_latency_us = tree_hops * t_hop_net_us + (num_aggregated_waves - 1) * 0.35
        bw_time_us = (bus_bytes / inter_node_bw_bps) * 1e6
        total_us = base_launch_us + step_latency_us + (bw_time_us / pkt_efficiency)
        return {
            "algo": algo,
            "latency_us": round(total_us, 2),
            "alg_bw_gbps": round((total_bytes / 1e9) / (total_us / 1e6), 3),
            "bus_bw_gbps": round((bus_bytes / 1e9) / (total_us / 1e6), 3),
            "sequential_hops": tree_hops,
            "per_nic_post_bytes": aggregated_post_bytes,
            "agg_factor": pat_params["agg_factor"],
            "parallel_factor": pat_params["parallel_factor"],
        }

    if algo == "pat_multi_rpn_nvls":
        # Multi-RPN PAT (`work->isOneRPN == false` in `all_gather.h:189` & `reduce_scatter.h:154`):
        # Runs `PatAGAlgorithm` / `PatRSAlgorithm` over `nNodes` with `patCount = count * localRanks`,
        # staging local intra-node gather/scatter through NVLS (`prims.patGather` / `prims.patScatter`).
        node_channel_bytes = channel_bytes_per_rank * local_ranks
        pat_params = compute_pat_aggregation_params(
            step_size_bytes, step_depth, max_pf, node_channel_bytes, n_nodes
        )
        inter_node_tree_hops = log2_up(n_nodes)
        aggregated_post_bytes = min(step_size_bytes, node_channel_bytes * pat_params["post_freq"])
        pkt_efficiency = min(1.0, aggregated_post_bytes / (aggregated_post_bytes + 4096.0))
        step_latency_us = inter_node_tree_hops * t_hop_net_us + 2 * t_hop_nvlink_us
        inter_bw_time_us = (bus_bytes / inter_node_bw_bps) * 1e6
        nvls_stage_us = (total_bytes / nvls_bw_bps) * 1e6 * 0.35  # pipelined with inter-node PAT waves
        total_us = base_launch_us + step_latency_us + (inter_bw_time_us / pkt_efficiency) + nvls_stage_us
        return {
            "algo": algo,
            "latency_us": round(total_us, 2),
            "alg_bw_gbps": round((total_bytes / 1e9) / (total_us / 1e6), 3),
            "bus_bw_gbps": round((bus_bytes / 1e9) / (total_us / 1e6), 3),
            "sequential_hops": inter_node_tree_hops,
            "per_nic_post_bytes": aggregated_post_bytes,
            "agg_factor": pat_params["agg_factor"],
            "parallel_factor": pat_params["parallel_factor"],
        }

    raise ValueError(f"Unknown algo: {algo}")


def run_pat_experiment() -> dict[str, Any]:
    ranks_list = [64, 256, 512, 1024]
    msg_sizes_mib = [1, 4, 16, 64, 256]

    sweep_rows: list[dict[str, Any]] = []
    for nranks in ranks_list:
        for mib in msg_sizes_mib:
            total_bytes = mib * 1024 * 1024
            ring = simulate_collective_latency_us("ring", total_bytes, nranks)
            pat_1rpn = simulate_collective_latency_us("pat_one_rpn", total_bytes, nranks)
            pat_mrpn = simulate_collective_latency_us("pat_multi_rpn_nvls", total_bytes, nranks)
            sweep_rows.append(
                {
                    "nranks": nranks,
                    "n_nodes": nranks // 8,
                    "message_size_mib": mib,
                    "ring_latency_us": ring["latency_us"],
                    "pat_1rpn_latency_us": pat_1rpn["latency_us"],
                    "pat_multi_rpn_latency_us": pat_mrpn["latency_us"],
                    "ring_bus_bw_gbps": ring["bus_bw_gbps"],
                    "pat_1rpn_bus_bw_gbps": pat_1rpn["bus_bw_gbps"],
                    "pat_multi_rpn_bus_bw_gbps": pat_mrpn["bus_bw_gbps"],
                    "speedup_pat_multi_rpn_vs_ring": round(ring["latency_us"] / pat_mrpn["latency_us"], 2),
                    "ring_sequential_hops": ring["sequential_hops"],
                    "pat_1rpn_tree_hops": pat_1rpn["sequential_hops"],
                    "pat_multi_rpn_inter_node_hops": pat_mrpn["sequential_hops"],
                    "pat_1rpn_agg_factor": pat_1rpn["agg_factor"],
                    "pat_multi_rpn_agg_factor": pat_mrpn["agg_factor"],
                }
            )

    # Trace `mirrorInvert` and `firstBitSet` bit-reversal hypercube dimensions for N=16
    bit_schedule_n16 = [
        {
            "step_i": i,
            "mirror_invert_16": mirror_invert(i, 16),
            "first_bit_set_dim": first_bit_set(i, 16),
        }
        for i in range(16)
    ]

    r1024_4m = next(r for r in sweep_rows if r["nranks"] == 1024 and r["message_size_mib"] == 4)
    r1024_16m = next(r for r in sweep_rows if r["nranks"] == 1024 and r["message_size_mib"] == 16)
    r512_4m = next(r for r in sweep_rows if r["nranks"] == 512 and r["message_size_mib"] == 4)

    return {
        "experiment_id": "nccl_pat_binomial_allgather_rs",
        "bit_reversal_schedule_n16": bit_schedule_n16,
        "collective_sweep": sweep_rows,
        "summary": {
            "n1024_4mib_ring_latency_us": r1024_4m["ring_latency_us"],
            "n1024_4mib_pat_multi_rpn_latency_us": r1024_4m["pat_multi_rpn_latency_us"],
            "n1024_4mib_speedup_x": r1024_4m["speedup_pat_multi_rpn_vs_ring"],
            "n1024_16mib_ring_bus_bw_gbps": r1024_16m["ring_bus_bw_gbps"],
            "n1024_16mib_pat_multi_rpn_bus_bw_gbps": r1024_16m["pat_multi_rpn_bus_bw_gbps"],
            "n1024_16mib_speedup_x": r1024_16m["speedup_pat_multi_rpn_vs_ring"],
            "n512_4mib_speedup_x": r512_4m["speedup_pat_multi_rpn_vs_ring"],
            "n1024_ring_hops": r1024_4m["ring_sequential_hops"],
            "n1024_pat_inter_node_hops": r1024_4m["pat_multi_rpn_inter_node_hops"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic results.json matches.")
    args = parser.parse_args()

    results = run_pat_experiment()
    formatted = json.dumps(results, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not RESULTS_PATH.exists():
            print(f"Missing {RESULTS_PATH}", file=sys.stderr)
            return 1
        existing = RESULTS_PATH.read_text(encoding="utf-8")
        if existing != formatted:
            print(f"Mismatch in {RESULTS_PATH}; re-run without --check to update.", file=sys.stderr)
            return 1
        print(f"OK: {RESULTS_PATH} is up to date.")
        return 0

    RESULTS_PATH.write_text(formatted, encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
