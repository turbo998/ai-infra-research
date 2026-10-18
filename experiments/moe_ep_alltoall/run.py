#!/usr/bin/env python3
"""CPU/offline analytical and discrete-event model for MoE Expert Parallelism (EP) & DeepEP All-to-All.

Covers exact primary-source layouts and algorithms from `deepseek-ai/DeepEP`
(`b5a33eefda39485f438dc764695b71ba7b9249b6`):
1. `deep_ep/include/deep_ep/layout/ep/token.cuh:10-41` (`TokenLayout::get_num_bytes<kWithMBarrier>()`):
   Exact TMA (`kNumTMAAlignmentBytes = 32`) + RDMA (`kNumRDMAAlignmentBytes = 64`) aligned token packet sizing
   for Dispatch (with FP8 scale factors `sf_pack_t` + metadata) vs Combine (BF16 without SF/metadata).
2. `deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh:120-135, 408-452`:
   Hierarchical 2-tier Scale-out (RDMA) -> Scale-up (NVLink) deduplication vs Flat All-to-All RDMA when
   each token selects `num_topk` experts across `num_scaleout_ranks * num_scaleup_ranks`.
3. `deep_ep/include/deep_ep/layout/ep/eplb.cuh:15-64, 91-136` & `prefetch_weights.cuh:25-86`:
   Expert Parallelism Load Balancing (`EPLBTask`, `EPWeightIterator`, `lb_prefetch_weights`) neutralizing Zipfian
   expert skew via redundant expert slots and atomic chunk-claimed NVLink TMA weight prefetching.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "input.json"
DEFAULT_JSON = ROOT / "results.json"
DEFAULT_CSV = ROOT / "results.csv"

K_NUM_TMA_ALIGNMENT_BYTES = 32
K_NUM_RDMA_ALIGNMENT_BYTES = 64
SIZEOF_MBARRIER = 8


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _validate_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _validate_positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return val


def compute_token_layout_bytes(
    num_hidden_bytes: int,
    num_sf_bytes: int,
    num_topk: int,
    with_metadata: bool,
    with_mbarrier: bool,
) -> dict[str, int]:
    """Exact port of `deep_ep::layout::TokenLayout` (`token.cuh:10-41`)."""
    if num_hidden_bytes <= 0 or num_hidden_bytes % K_NUM_TMA_ALIGNMENT_BYTES != 0:
        raise ValueError(
            f"num_hidden_bytes must be > 0 and divisible by {K_NUM_TMA_ALIGNMENT_BYTES}, got {num_hidden_bytes}"
        )
    if num_sf_bytes < 0:
        raise ValueError(f"num_sf_bytes must be >= 0, got {num_sf_bytes}")
    if num_topk <= 0 or num_topk > 32:
        raise ValueError(f"num_topk must be in [1, 32], got {num_topk}")

    # `token.cuh:25-26`:
    # num_metadata_bytes = num_topk * (sizeof(int) + sizeof(float)) + (with_metadata ? (1 + num_topk) * sizeof(int) : 0)
    num_metadata_bytes = num_topk * (4 + 4) + ((1 + num_topk) * 4 if with_metadata else 0)

    aligned_hidden = _align(num_hidden_bytes, K_NUM_TMA_ALIGNMENT_BYTES)
    aligned_sf = _align(num_sf_bytes, K_NUM_TMA_ALIGNMENT_BYTES)
    aligned_meta = _align(num_metadata_bytes, K_NUM_TMA_ALIGNMENT_BYTES)

    num_data_bytes = _align(
        aligned_hidden + aligned_sf + aligned_meta,
        K_NUM_RDMA_ALIGNMENT_BYTES,
    )
    mbarrier_bytes = _align(SIZEOF_MBARRIER if with_mbarrier else 0, K_NUM_TMA_ALIGNMENT_BYTES)
    total_bytes = num_data_bytes + mbarrier_bytes

    return {
        "num_hidden_bytes": num_hidden_bytes,
        "num_sf_bytes": num_sf_bytes,
        "num_metadata_bytes": num_metadata_bytes,
        "num_data_bytes_rdma_aligned": num_data_bytes,
        "smem_slot_bytes_with_mbarrier": total_bytes,
    }


def evaluate_token_and_smem_sizing(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate `TokenLayout` and warp capacity per SM (`dispatch.hpp:75-85`, `combine.hpp:44-58`)."""
    hidden = _validate_positive_int("hidden", case.get("hidden"))
    dispatch_elem_size = _validate_positive_int("dispatch_elem_size", case.get("dispatch_elem_size"))
    num_sf_packs = _validate_non_negative_int("num_sf_packs", case.get("num_sf_packs"))
    num_topk = _validate_positive_int("num_topk", case.get("num_topk"))
    num_ranks = _validate_positive_int("num_ranks", case.get("num_ranks"))
    num_experts = _validate_positive_int("num_experts", case.get("num_experts"))
    smem_capacity_bytes = _validate_positive_int("smem_capacity_bytes", case.get("smem_capacity_bytes", 232448))
    num_channels_per_sm = _validate_positive_int("num_channels_per_sm", case.get("num_channels_per_sm", 8))

    # `dispatch.hpp:20-29`:
    notify_warps = 4
    notify_smem_bytes = _align(num_ranks + num_experts, notify_warps * 32) * 4

    dispatch_token = compute_token_layout_bytes(
        num_hidden_bytes=hidden * dispatch_elem_size,
        num_sf_bytes=num_sf_packs * 4,
        num_topk=num_topk,
        with_metadata=True,
        with_mbarrier=True,
    )
    combine_token = compute_token_layout_bytes(
        num_hidden_bytes=hidden * 2,  # nv_bfloat16
        num_sf_bytes=0,
        num_topk=num_topk,
        with_metadata=False,
        with_mbarrier=True,
    )

    scaleup_only_dispatch_warps = min(
        (smem_capacity_bytes - notify_smem_bytes) // dispatch_token["smem_slot_bytes_with_mbarrier"],
        32 - notify_warps,
    )
    hybrid_dispatch_smem_used = (
        notify_smem_bytes + (2 * num_channels_per_sm) * dispatch_token["smem_slot_bytes_with_mbarrier"]
    )
    hybrid_combine_smem_used = (2 * num_channels_per_sm) * combine_token["smem_slot_bytes_with_mbarrier"]

    return {
        "case_id": str(case["case_id"]),
        "model": str(case["model"]),
        "hidden": hidden,
        "num_topk": num_topk,
        "dispatch_rdma_bytes_per_token": dispatch_token["num_data_bytes_rdma_aligned"],
        "dispatch_smem_bytes_per_warp": dispatch_token["smem_slot_bytes_with_mbarrier"],
        "combine_rdma_bytes_per_token": combine_token["num_data_bytes_rdma_aligned"],
        "combine_smem_bytes_per_warp": combine_token["smem_slot_bytes_with_mbarrier"],
        "notify_smem_bytes": notify_smem_bytes,
        "scaleup_only_dispatch_warps": scaleup_only_dispatch_warps,
        "hybrid_dispatch_smem_bytes": hybrid_dispatch_smem_used,
        "hybrid_combine_smem_bytes": hybrid_combine_smem_used,
    }


def evaluate_hierarchical_deduplication(case: dict[str, Any]) -> dict[str, Any]:
    """Compare Flat All-to-All RDMA vs DeepEP `ptx::deduplicate` 2-Tier Hybrid Dispatch (`hybrid_dispatch.cuh`)."""
    num_tokens_per_rank = _validate_positive_int("num_tokens_per_rank", case.get("num_tokens_per_rank"))
    num_scaleout_ranks = _validate_positive_int("num_scaleout_ranks", case.get("num_scaleout_ranks"))
    num_scaleup_ranks = _validate_positive_int("num_scaleup_ranks", case.get("num_scaleup_ranks"))
    num_experts = _validate_positive_int("num_experts", case.get("num_experts"))
    num_topk = _validate_positive_int("num_topk", case.get("num_topk"))
    max_scaleout_groups_per_token = _validate_positive_int(
        "max_scaleout_groups_per_token", case.get("max_scaleout_groups_per_token", num_scaleout_ranks)
    )
    hidden = _validate_positive_int("hidden", case.get("hidden", 7168))
    dispatch_elem_size = _validate_positive_int("dispatch_elem_size", case.get("dispatch_elem_size", 1))
    num_sf_packs = _validate_non_negative_int("num_sf_packs", case.get("num_sf_packs", 56))
    rdma_bw_gbps_per_rank = _validate_positive_float("rdma_bw_gbps_per_rank", case.get("rdma_bw_gbps_per_rank", 388.0))
    nvlink_bw_gbps_per_rank = _validate_positive_float(
        "nvlink_bw_gbps_per_rank", case.get("nvlink_bw_gbps_per_rank", 3200.0)
    )

    num_ranks = num_scaleout_ranks * num_scaleup_ranks
    if num_experts % num_ranks != 0:
        raise ValueError("num_experts must be divisible by num_scaleout_ranks * num_scaleup_ranks")

    dispatch_token = compute_token_layout_bytes(
        num_hidden_bytes=hidden * dispatch_elem_size,
        num_sf_bytes=num_sf_packs * 4,
        num_topk=num_topk,
        with_metadata=True,
        with_mbarrier=False,
    )
    combine_token = compute_token_layout_bytes(
        num_hidden_bytes=hidden * 2,
        num_sf_bytes=0,
        num_topk=num_topk,
        with_metadata=False,
        with_mbarrier=False,
    )
    dispatch_bytes = dispatch_token["num_data_bytes_rdma_aligned"]
    combine_bytes = combine_token["num_data_bytes_rdma_aligned"]

    # 1. Flat All-to-All (without node-level scale-out deduplication):
    # Every top-k selection targeting a remote node sends an independent RDMA packet.
    remote_node_fraction = (num_scaleout_ranks - 1) / float(num_scaleout_ranks)
    flat_remote_sends_per_token = num_topk * remote_node_fraction

    # 2. DeepEP 2-tier Hybrid Dispatch (`hybrid_dispatch.cuh:395, 412`):
    # `stored_dst_scaleout_rank_idx = dst_expert_idx / kNumExpertsPerScaleout`
    # `ptx::deduplicate(stored_dst_scaleout_rank_idx, lane_idx)` sends at most 1 copy per distinct scale-out node!
    # Under uniform random selection across `G = min(num_scaleout_ranks, max_scaleout_groups_per_token)` candidate
    # nodes, expected distinct nodes hit by `k` draws from `G` nodes (with `E_node = num_experts / num_scaleout_ranks`
    # experts per node):
    experts_per_node = num_experts // num_scaleout_ranks
    active_nodes = min(num_scaleout_ranks, max_scaleout_groups_per_token)
    active_pool_experts = active_nodes * experts_per_node
    if active_pool_experts < num_topk:
        raise ValueError("active_pool_experts must be >= num_topk")

    # Hypergeometric probability that a specific active node is hit at least once:
    prob_miss_one_active_node = math.comb(active_pool_experts - experts_per_node, num_topk) / float(
        math.comb(active_pool_experts, num_topk)
    )
    expected_distinct_nodes = active_nodes * (1.0 - prob_miss_one_active_node)
    # Fraction of those distinct nodes that are remote (`(num_scaleout_ranks - 1) / num_scaleout_ranks`):
    deepep_remote_sends_per_token = expected_distinct_nodes * remote_node_fraction

    flat_rdma_mib_per_rank = (
        num_tokens_per_rank * flat_remote_sends_per_token * (dispatch_bytes + combine_bytes)
    ) / (1024.0 * 1024.0)
    deepep_rdma_mib_per_rank = (
        num_tokens_per_rank * deepep_remote_sends_per_token * (dispatch_bytes + combine_bytes)
    ) / (1024.0 * 1024.0)

    # Local NVLink forwarding volume per rank (`hybrid_dispatch.cuh:584-595`):
    experts_per_rank = num_experts // num_ranks
    prob_miss_one_rank_given_node = math.comb(experts_per_node - experts_per_rank, 1) / float(experts_per_node)
    # Approximate local NVLink fanout per arrived node packet:
    avg_hits_per_active_node = num_topk / expected_distinct_nodes
    expected_nvlink_ranks_per_node_packet = num_scaleup_ranks * (
        1.0 - (1.0 - 1.0 / num_scaleup_ranks) ** avg_hits_per_active_node
    )
    nvlink_mib_per_rank = (
        num_tokens_per_rank
        * expected_distinct_nodes
        * expected_nvlink_ranks_per_node_packet
        * (dispatch_bytes + combine_bytes)
    ) / (1024.0 * 1024.0)

    flat_rdma_us = (flat_rdma_mib_per_rank * 1024.0 * 1024.0 * 8.0) / (rdma_bw_gbps_per_rank * 1e3)
    deepep_rdma_us = (deepep_rdma_mib_per_rank * 1024.0 * 1024.0 * 8.0) / (rdma_bw_gbps_per_rank * 1e3)
    nvlink_forward_us = (nvlink_mib_per_rank * 1024.0 * 1024.0 * 8.0) / (nvlink_bw_gbps_per_rank * 1e3)
    # In `hybrid_dispatch.cuh`, scale-out warps and forward warps pipeline concurrently across `kNumChannels`
    deepep_pipelined_us = max(deepep_rdma_us, nvlink_forward_us) + 0.15 * min(deepep_rdma_us, nvlink_forward_us)

    return {
        "case_id": str(case["case_id"]),
        "num_scaleout_ranks": num_scaleout_ranks,
        "num_scaleup_ranks": num_scaleup_ranks,
        "num_topk": num_topk,
        "max_scaleout_groups_per_token": active_nodes,
        "flat_remote_sends_per_token": round(flat_remote_sends_per_token, 4),
        "deepep_remote_sends_per_token": round(deepep_remote_sends_per_token, 4),
        "flat_rdma_mib_per_rank": round(flat_rdma_mib_per_rank, 3),
        "deepep_rdma_mib_per_rank": round(deepep_rdma_mib_per_rank, 3),
        "rdma_traffic_reduction_x": round(flat_rdma_mib_per_rank / deepep_rdma_mib_per_rank, 2),
        "flat_rdma_us": round(flat_rdma_us, 2),
        "deepep_rdma_us": round(deepep_rdma_us, 2),
        "nvlink_forward_us": round(nvlink_forward_us, 2),
        "deepep_pipelined_us": round(deepep_pipelined_us, 2),
        "end_to_end_speedup_x": round(flat_rdma_us / deepep_pipelined_us, 2),
    }


def evaluate_eplb_and_skew(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate Zipfian expert load skew, `kExpertAlignment`, and DeepEP `EPLB` (`eplb.cuh`, `prefetch_weights.cuh`)."""
    num_ranks = _validate_positive_int("num_ranks", case.get("num_ranks"))
    num_local_experts = _validate_positive_int("num_local_experts", case.get("num_local_experts"))
    num_redundant_experts_per_rank = _validate_non_negative_int(
        "num_redundant_experts_per_rank", case.get("num_redundant_experts_per_rank")
    )
    total_tokens_across_cluster = _validate_positive_int(
        "total_tokens_across_cluster", case.get("total_tokens_across_cluster")
    )
    num_topk = _validate_positive_int("num_topk", case.get("num_topk"))
    zipf_s = _validate_positive_float("zipf_s", case.get("zipf_s"))
    expert_alignment = _validate_positive_int("expert_alignment", case.get("expert_alignment", 128))
    bytes_per_expert_weight = _validate_positive_int(
        "bytes_per_expert_weight", case.get("bytes_per_expert_weight", 44040192)
    )
    chunk_bytes = _validate_positive_int("chunk_bytes", case.get("chunk_bytes", 16384))
    nvlink_bw_gbps = _validate_positive_float("nvlink_bw_gbps", case.get("nvlink_bw_gbps", 3200.0))
    compute_tokens_per_ms_per_rank = _validate_positive_float(
        "compute_tokens_per_ms_per_rank", case.get("compute_tokens_per_ms_per_rank", 18000.0)
    )

    num_experts = num_ranks * num_local_experts
    total_routed_assignments = total_tokens_across_cluster * num_topk

    # Zipfian popularity across `num_experts` experts
    raw_weights = [1.0 / ((i + 1) ** zipf_s) for i in range(num_experts)]
    weight_sum = sum(raw_weights)
    expert_tokens = [int(round(total_routed_assignments * (w / weight_sum))) for w in raw_weights]

    # Without EPLB: contiguous assignment of `num_local_experts` per rank
    # Worst-case rank holds the hottest experts (or even under cyclic placement, hot experts concentrate on a few ranks)
    rank_loads_unbalanced = [0] * num_ranks
    for e_idx, count in enumerate(expert_tokens):
        r_idx = e_idx // num_local_experts
        # Apply `math::align(count, kExpertAlignment)` (`hybrid_dispatch.cuh:269-270`)
        aligned_count = _align(count, expert_alignment) if count > 0 else 0
        rank_loads_unbalanced[r_idx] += aligned_count

    mean_rank_load = sum(rank_loads_unbalanced) / float(num_ranks)
    max_rank_load_no_eplb = max(rank_loads_unbalanced)
    imbalance_ratio_no_eplb = max_rank_load_no_eplb / max(1.0, mean_rank_load)

    # With DeepEP EPLB (`eplb.cuh:15-64` & `prefetch_weights.cuh`):
    # Total redundant slots across the NVLink/EP domain = `num_ranks * num_redundant_experts_per_rank`.
    # Greedy replication assigns redundant replicas to the hottest experts (dividing their token load across `1 + r_i` replicas)
    # and packs replicas onto the least-loaded ranks.
    replicas = [1] * num_experts
    total_redundant_slots = num_ranks * num_redundant_experts_per_rank
    for _ in range(total_redundant_slots):
        # Find expert with largest per-replica load
        best_e = max(range(num_experts), key=lambda e: expert_tokens[e] / float(replicas[e]))
        replicas[best_e] += 1

    # Pack original experts (fixed on `e // num_local_experts`) + redundant replicas onto least-loaded ranks
    rank_loads_eplb = [0] * num_ranks
    for e_idx in range(num_experts):
        per_rep = int(math.ceil(expert_tokens[e_idx] / float(replicas[e_idx])))
        aligned_rep = _align(per_rep, expert_alignment) if per_rep > 0 else 0
        rank_loads_eplb[e_idx // num_local_experts] += aligned_rep

    # Place each redundant replica onto the currently least-loaded rank (up to `num_redundant_experts_per_rank` per rank)
    redundant_used_per_rank = [0] * num_ranks
    for e_idx in range(num_experts):
        extra = replicas[e_idx] - 1
        if extra <= 0:
            continue
        per_rep = int(math.ceil(expert_tokens[e_idx] / float(replicas[e_idx])))
        aligned_rep = _align(per_rep, expert_alignment) if per_rep > 0 else 0
        for _ in range(extra):
            candidates = [r for r in range(num_ranks) if redundant_used_per_rank[r] < num_redundant_experts_per_rank]
            dst_r = min(candidates, key=lambda r: rank_loads_eplb[r])
            rank_loads_eplb[dst_r] += aligned_rep
            redundant_used_per_rank[dst_r] += 1

    max_rank_load_eplb = max(rank_loads_eplb)
    imbalance_ratio_eplb = max_rank_load_eplb / max(1.0, sum(rank_loads_eplb) / float(num_ranks))

    # Weight prefetch cost (`EPWeightIterator` in `eplb.cuh:116` & `prefetch_weights.cuh:72-86`):
    chunks_per_expert = (bytes_per_expert_weight + chunk_bytes - 1) // chunk_bytes
    max_tasks_on_one_source_rank = sum(replicas[e] - 1 for e in range(num_local_experts))
    total_chunks_on_hottest_source = chunks_per_expert * max_tasks_on_one_source_rank
    prefetch_bytes_on_hottest_source = max_tasks_on_one_source_rank * bytes_per_expert_weight
    prefetch_time_ms = (prefetch_bytes_on_hottest_source * 8.0) / (nvlink_bw_gbps * 1e6)

    compute_ms_no_eplb = max_rank_load_no_eplb / compute_tokens_per_ms_per_rank
    compute_ms_with_eplb = (max_rank_load_eplb / compute_tokens_per_ms_per_rank) + prefetch_time_ms

    return {
        "case_id": str(case["case_id"]),
        "zipf_s": zipf_s,
        "num_ranks": num_ranks,
        "num_experts": num_experts,
        "num_redundant_experts_per_rank": num_redundant_experts_per_rank,
        "max_rank_tokens_no_eplb": max_rank_load_no_eplb,
        "max_rank_tokens_with_eplb": max_rank_load_eplb,
        "imbalance_ratio_no_eplb": round(imbalance_ratio_no_eplb, 3),
        "imbalance_ratio_with_eplb": round(imbalance_ratio_eplb, 3),
        "hottest_source_eplb_tasks": max_tasks_on_one_source_rank,
        "hottest_source_tma_chunks": total_chunks_on_hottest_source,
        "weight_prefetch_ms": round(prefetch_time_ms, 3),
        "step_compute_ms_no_eplb": round(compute_ms_no_eplb, 3),
        "step_compute_ms_with_eplb": round(compute_ms_with_eplb, 3),
        "step_speedup_x": round(compute_ms_no_eplb / compute_ms_with_eplb, 2),
    }


def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(fixture, dict) or fixture.get("schema_version") != 1:
        raise ValueError("Fixture schema_version must be 1")

    token_cases = fixture.get("token_and_smem_sizing_cases")
    dedup_cases = fixture.get("hierarchical_deduplication_cases")
    eplb_cases = fixture.get("eplb_and_skew_cases")
    if not isinstance(token_cases, list) or not token_cases:
        raise ValueError("token_and_smem_sizing_cases must be a non-empty list")
    if not isinstance(dedup_cases, list) or not dedup_cases:
        raise ValueError("hierarchical_deduplication_cases must be a non-empty list")
    if not isinstance(eplb_cases, list) or not eplb_cases:
        raise ValueError("eplb_and_skew_cases must be a non-empty list")

    return {
        "schema_version": 1,
        "experiment_id": "moe_ep_alltoall",
        "evidence_type": "own-offline",
        "token_and_smem_sizing": [evaluate_token_and_smem_sizing(c) for c in token_cases],
        "hierarchical_deduplication": [evaluate_hierarchical_deduplication(c) for c in dedup_cases],
        "eplb_and_skew": [evaluate_eplb_and_skew(c) for c in eplb_cases],
    }


def _to_csv_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results["token_and_smem_sizing"]:
        rows.append(
            {
                "section": "token_and_smem_sizing",
                "case_id": item["case_id"],
                "primary_metric": item["dispatch_rdma_bytes_per_token"],
                "secondary_metric": item["combine_rdma_bytes_per_token"],
                "delta_or_ratio": item["scaleup_only_dispatch_warps"],
                "note": f"hybrid_dispatch_smem={item['hybrid_dispatch_smem_bytes']}",
            }
        )
    for item in results["hierarchical_deduplication"]:
        rows.append(
            {
                "section": "hierarchical_deduplication",
                "case_id": item["case_id"],
                "primary_metric": item["deepep_rdma_mib_per_rank"],
                "secondary_metric": item["flat_rdma_mib_per_rank"],
                "delta_or_ratio": item["rdma_traffic_reduction_x"],
                "note": f"speedup={item['end_to_end_speedup_x']}x",
            }
        )
    for item in results["eplb_and_skew"]:
        rows.append(
            {
                "section": "eplb_and_skew",
                "case_id": item["case_id"],
                "primary_metric": item["step_compute_ms_with_eplb"],
                "secondary_metric": item["step_compute_ms_no_eplb"],
                "delta_or_ratio": item["step_speedup_x"],
                "note": f"imbalance={item['imbalance_ratio_no_eplb']}->{item['imbalance_ratio_with_eplb']}",
            }
        )
    return rows


def _format_csv(rows: list[dict[str, Any]]) -> str:
    fieldnames = ["section", "case_id", "primary_metric", "secondary_metric", "delta_or_ratio", "note"]
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--check", action="store_true", help="Verify committed outputs match without writing")
    args = parser.parse_args(argv)

    fixture = json.loads(args.input.read_text(encoding="utf-8"))
    results = evaluate_fixture(fixture)
    json_text = json.dumps(results, indent=2, sort_keys=True) + "\n"
    csv_text = _format_csv(_to_csv_rows(results))

    if args.check:
        if not args.json_out.exists() or not args.csv_out.exists():
            raise SystemExit("Committed results.json or results.csv missing")
        if args.json_out.read_text(encoding="utf-8") != json_text:
            raise SystemExit("results.json does not match deterministic evaluation")
        if args.csv_out.read_text(encoding="utf-8") != csv_text:
            raise SystemExit("results.csv does not match deterministic evaluation")
        print("own-offline moe_ep_alltoall outputs verified")
        return 0

    args.json_out.write_text(json_text, encoding="utf-8")
    args.csv_out.write_text(csv_text, encoding="utf-8")
    print("own-offline moe_ep_alltoall outputs generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
