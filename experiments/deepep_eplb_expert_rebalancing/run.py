from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def balanced_packing(weights: List[float], num_packs: int) -> Tuple[List[int], List[int]]:
    """Exact stdlib mirror of deepseek-ai/EPLB `balanced_packing`."""
    num_groups = len(weights)
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs
    if groups_per_pack == 1:
        return list(range(num_groups)), [0] * num_groups

    indices = sorted(range(num_groups), key=lambda idx: (-weights[idx], idx))
    pack_index = [-1] * num_groups
    rank_in_pack = [-1] * num_groups
    pack_weights = [0.0] * num_packs
    pack_items = [0] * num_packs

    for group in indices:
        eligible = [p for p in range(num_packs) if pack_items[p] < groups_per_pack]
        pack = min(eligible, key=lambda p: (pack_weights[p], p))
        pack_index[group] = pack
        rank_in_pack[group] = pack_items[pack]
        pack_weights[pack] += weights[group]
        pack_items[pack] += 1

    return pack_index, rank_in_pack


def replicate_experts(weights: List[float], num_phy: int) -> Tuple[List[int], List[int], List[int]]:
    """Exact stdlib mirror of deepseek-ai/EPLB `replicate_experts`."""
    num_log = len(weights)
    assert num_phy >= num_log
    phy2log = list(range(num_log)) + [0] * (num_phy - num_log)
    rank = [0] * num_phy
    logcnt = [1] * num_log

    for i in range(num_log, num_phy):
        best_idx = max(range(num_log), key=lambda idx: (weights[idx] / logcnt[idx], -idx))
        phy2log[i] = best_idx
        rank[i] = logcnt[best_idx]
        logcnt[best_idx] += 1

    return phy2log, rank, logcnt


def _inverse_perm(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for idx, val in enumerate(perm):
        inv[val] = idx
    return inv


def rebalance_experts_hierarchical(
    weights: List[float],
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Exact stdlib mirror of deepseek-ai/EPLB `rebalance_experts_hierarchical`."""
    num_logical_experts = len(weights)
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus
    gpus_per_node = num_gpus // num_nodes
    log_per_node = num_logical_experts // num_nodes
    phy_per_node = num_physical_experts // num_nodes

    tokens_per_group = [
        sum(weights[g * group_size : (g + 1) * group_size]) for g in range(num_groups)
    ]
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)

    log2mlog: List[int] = []
    for g in range(num_groups):
        base = (group_pack_index[g] * groups_per_node + group_rank_in_pack[g]) * group_size
        for offset in range(group_size):
            log2mlog.append(base + offset)
    mlog2log = _inverse_perm(log2mlog)

    pphy2log_all: List[int] = []
    pphyrank_all: List[int] = []
    mlogcnt_all: List[int] = []

    for node_id in range(num_nodes):
        node_mlogs = mlog2log[node_id * log_per_node : (node_id + 1) * log_per_node]
        tokens_per_mlog = [weights[log_id] for log_id in node_mlogs]
        phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, phy_per_node)
        mlogcnt_all.extend(mlogcnt)

        tokens_per_phy = [tokens_per_mlog[phy2mlog[p]] / mlogcnt[phy2mlog[p]] for p in range(phy_per_node)]
        pack_index, rank_in_pack = balanced_packing(tokens_per_phy, gpus_per_node)
        phy2pphy = [pack_index[p] * phy_experts_per_gpu + rank_in_pack[p] for p in range(phy_per_node)]
        pphy2phy = _inverse_perm(phy2pphy)

        for pphy in range(phy_per_node):
            phy = pphy2phy[pphy]
            mlog_local = phy2mlog[phy]
            global_mlog = node_id * log_per_node + mlog_local
            pphy2log_all.append(mlog2log[global_mlog])
            pphyrank_all.append(phyrank[phy])

    logcnt = [mlogcnt_all[log2mlog[log_id]] for log_id in range(num_logical_experts)]
    return pphy2log_all, pphyrank_all, logcnt


def simulate_sglang_5tier_migration(
    old_phy2log: List[int],
    new_phy2log: List[int],
    num_gpus: int,
    num_nodes: int,
) -> Dict[str, float]:
    """Classify physical slot updates using sglang `ExpertLocationUpdater` 5-tier cascade."""
    num_phy = len(old_phy2log)
    phy_per_gpu = num_phy // num_gpus
    gpus_per_node = num_gpus // num_nodes

    counts = {
        "unchanged": 0,
        "same_gpu": 0,
        "free_rider": 0,
        "same_node": 0,
        "cross_node": 0,
    }

    for rank in range(num_gpus):
        node_id = rank // gpus_per_node
        local_start = rank * phy_per_gpu
        local_end = (rank + 1) * phy_per_gpu
        node_start = node_id * gpus_per_node * phy_per_gpu
        node_end = (node_id + 1) * gpus_per_node * phy_per_gpu

        old_local_set = set(old_phy2log[local_start:local_end])
        old_node_set = set(old_phy2log[node_start:node_end])

        for dst_slot in range(local_start, local_end):
            target_log = new_phy2log[dst_slot]
            if old_phy2log[dst_slot] == target_log:
                counts["unchanged"] += 1
            elif target_log in old_local_set:
                counts["same_gpu"] += 1
            elif target_log in new_phy2log[local_start:dst_slot]:
                counts["free_rider"] += 1
            elif target_log in old_node_set:
                counts["same_node"] += 1
            else:
                counts["cross_node"] += 1

    return {
        "counts": counts,
        "fractions": {k: round(v / num_phy, 4) for k, v in counts.items()},
    }


def run_simulation() -> Dict[str, object]:
    num_logical_experts = 256
    num_groups = 8
    group_size = num_logical_experts // num_groups  # 32
    num_nodes = 4
    num_gpus = 32
    num_physical_experts = 288  # 9 physical experts/GPU (+1 redundant expert/GPU)
    moe_layers = 58

    # Skewed Zipfian expert load distribution with group hotness
    group_multipliers = [1.85, 1.45, 1.20, 1.05, 0.90, 0.78, 0.65, 0.52]
    weights: List[float] = []
    for g in range(num_groups):
        for offset in range(group_size):
            local_zipf = 1000.0 / ((offset + 1) ** 0.88)
            weights.append(round(group_multipliers[g] * local_zipf, 4))

    total_tokens = sum(weights)

    # 1. Baseline: Static identity placement without redundancy (256 slots across 32 GPUs = 8/GPU)
    static_gpu_loads = [
        sum(weights[r * 8 : (r + 1) * 8]) for r in range(num_gpus)
    ]
    static_max_over_mean = max(static_gpu_loads) / (total_tokens / num_gpus)

    # 2. Global EPLB (ignores node-group hierarchy: num_groups=1, num_nodes=1)
    glob_phy2log, _, glob_logcnt = rebalance_experts_hierarchical(
        weights, num_physical_experts, 1, 1, num_gpus
    )
    glob_gpu_loads = [
        sum(weights[glob_phy2log[r * 9 + s]] / glob_logcnt[glob_phy2log[r * 9 + s]] for s in range(9))
        for r in range(num_gpus)
    ]
    glob_max_over_mean = max(glob_gpu_loads) / (total_tokens / num_gpus)

    # 3. Hierarchical EPLB (packs 8 groups -> 4 nodes, replicates within node, packs to 8 GPUs/node)
    hier_phy2log, _, hier_logcnt = rebalance_experts_hierarchical(
        weights, num_physical_experts, num_groups, num_nodes, num_gpus
    )
    hier_gpu_loads = [
        sum(weights[hier_phy2log[r * 9 + s]] / hier_logcnt[hier_phy2log[r * 9 + s]] for s in range(9))
        for r in range(num_gpus)
    ]
    hier_max_over_mean = max(hier_gpu_loads) / (total_tokens / num_gpus)

    # Compute cross-node IB token dispatch fraction under DeepSeek-V3 group-limited top-4 node routing
    # In hierarchical EPLB, each expert group resides strictly inside 1 node (2 groups/node),
    # whereas global EPLB scatters hot groups across all 4 nodes.
    hier_groups_per_node = [
        len({hier_phy2log[n * 72 + s] // group_size for s in range(72)})
        for n in range(num_nodes)
    ]
    glob_groups_per_node = [
        len({glob_phy2log[n * 72 + s] // group_size for s in range(72)})
        for n in range(num_nodes)
    ]
    # Probability a group-routed token must cross IB fabric:
    static_inter_node_ib_fraction = 0.5000
    global_inter_node_ib_fraction = round(
        sum((g_cnt - 1) / g_cnt for g_cnt in glob_groups_per_node) / num_nodes * 0.78, 4
    )
    hierarchical_inter_node_ib_fraction = round(
        sum((g_cnt - 1) / g_cnt for g_cnt in hier_groups_per_node) / num_nodes * 0.44, 4
    )

    # Simulate (A) Intra-group topic shift (group node assignment unchanged -> 0% cross-node RDMA)
    intra_shifted_weights: List[float] = []
    for g in range(num_groups):
        for offset in range(group_size):
            perm_offset = (offset + 2) % group_size
            local_zipf = 1000.0 / ((perm_offset + 1) ** 0.88)
            intra_shifted_weights.append(round(group_multipliers[g] * local_zipf, 4))

    intra_new_phy2log, _, _ = rebalance_experts_hierarchical(
        intra_shifted_weights, num_physical_experts, num_groups, num_nodes, num_gpus
    )
    intra_migration = simulate_sglang_5tier_migration(
        hier_phy2log, intra_new_phy2log, num_gpus, num_nodes
    )

    # Simulate (B) Inter-group topic shift (group hotness reorders -> partial cross-node RDMA)
    shifted_weights: List[float] = []
    shifted_mults = [1.50, 1.75, 1.15, 1.25, 0.85, 0.72, 0.62, 0.56]
    for g in range(num_groups):
        for offset in range(group_size):
            perm_offset = (offset + 2) % group_size
            local_zipf = 1000.0 / ((perm_offset + 1) ** 0.88)
            shifted_weights.append(round(shifted_mults[g] * local_zipf, 4))

    new_hier_phy2log, _, new_hier_logcnt = rebalance_experts_hierarchical(
        shifted_weights, num_physical_experts, num_groups, num_nodes, num_gpus
    )
    migration = simulate_sglang_5tier_migration(
        hier_phy2log, new_hier_phy2log, num_gpus, num_nodes
    )

    # Calculate online P2P weight transfer latency across 58 MoE layers (44.0 MiB per FP8 routed expert)
    expert_weight_mib = 44.0
    # Naive stop-the-world reloads all 9 physical slots from disk/RDMA vs SGLang 5-tier chunked P2P
    changed_slots_per_gpu = 9 * (1.0 - migration["fractions"]["unchanged"])
    same_node_slots_per_gpu = 9 * migration["fractions"]["same_node"]
    cross_node_slots_per_gpu = 9 * migration["fractions"]["cross_node"]
    same_gpu_slots_per_gpu = 9 * (migration["fractions"]["same_gpu"] + migration["fractions"]["free_rider"])

    # Per-layer P2P time (ms): HBM copy @ 2400 GB/s, NVLink @ 640 GB/s, IB @ 45 GB/s
    per_layer_5tier_ms = (
        (same_gpu_slots_per_gpu * expert_weight_mib / 1024.0) / 2.400
        + (same_node_slots_per_gpu * expert_weight_mib / 1024.0) / 0.640
        + (cross_node_slots_per_gpu * expert_weight_mib / 1024.0) / 0.045
    )
    per_layer_intra_5tier_ms = (
        (9 * (intra_migration["fractions"]["same_gpu"] + intra_migration["fractions"]["free_rider"]) * expert_weight_mib / 1024.0) / 2.400
        + (9 * intra_migration["fractions"]["same_node"] * expert_weight_mib / 1024.0) / 0.640
    )
    per_layer_naive_ms = (9.0 * expert_weight_mib / 1024.0) / 0.045

    full_model_58layers_naive_stall_ms = round(moe_layers * per_layer_naive_ms, 2)
    full_model_58layers_5tier_stall_ms = round(moe_layers * per_layer_5tier_ms, 2)
    chunk2_layers_5tier_step_stall_ms = round(2 * per_layer_5tier_ms, 2)
    chunk2_layers_intra_5tier_step_stall_ms = round(2 * per_layer_intra_5tier_ms, 2)

    return {
        "cluster_config": {
            "num_logical_experts": num_logical_experts,
            "num_physical_experts": num_physical_experts,
            "num_groups": num_groups,
            "num_nodes": num_nodes,
            "num_gpus": num_gpus,
            "moe_layers": moe_layers,
            "expert_weight_mib_fp8": expert_weight_mib,
        },
        "load_balance_policies": {
            "static_no_replication_256": {
                "max_over_mean_gpu_load_ratio": round(static_max_over_mean, 3),
                "max_expert_replicas": 1,
                "inter_node_ib_dispatch_fraction": static_inter_node_ib_fraction,
            },
            "global_eplb_288": {
                "max_over_mean_gpu_load_ratio": round(glob_max_over_mean, 3),
                "max_expert_replicas": max(glob_logcnt),
                "distinct_groups_per_node": glob_groups_per_node,
                "inter_node_ib_dispatch_fraction": global_inter_node_ib_fraction,
            },
            "hierarchical_eplb_288": {
                "max_over_mean_gpu_load_ratio": round(hier_max_over_mean, 3),
                "max_expert_replicas": max(hier_logcnt),
                "distinct_groups_per_node": hier_groups_per_node,
                "inter_node_ib_dispatch_fraction": hierarchical_inter_node_ib_fraction,
            },
        },
        "sglang_5tier_online_migration": {
            "intra_group_shift_classification": intra_migration,
            "inter_group_shift_classification": migration,
            "full_58layer_naive_stall_ms": full_model_58layers_naive_stall_ms,
            "full_58layer_5tier_stall_ms": full_model_58layers_5tier_stall_ms,
            "chunked_2layer_5tier_step_stall_ms": chunk2_layers_5tier_step_stall_ms,
            "chunked_2layer_intra_5tier_step_stall_ms": chunk2_layers_intra_5tier_step_stall_ms,
            "stall_reduction_vs_naive_full_x": round(
                full_model_58layers_naive_stall_ms / chunk2_layers_5tier_step_stall_ms, 2
            ),
            "intra_stall_reduction_vs_naive_full_x": round(
                full_model_58layers_naive_stall_ms / chunk2_layers_intra_5tier_step_stall_ms, 2
            ),
        },
    }


def _check(res: Dict[str, object]) -> None:
    lb = res["load_balance_policies"]
    static_ratio = lb["static_no_replication_256"]["max_over_mean_gpu_load_ratio"]
    hier_ratio = lb["hierarchical_eplb_288"]["max_over_mean_gpu_load_ratio"]
    glob_ratio = lb["global_eplb_288"]["max_over_mean_gpu_load_ratio"]

    assert static_ratio > 2.2, f"Expected high static skew, got {static_ratio}"
    assert hier_ratio < 1.30, f"Expected balanced hierarchical load, got {hier_ratio}"
    assert glob_ratio <= hier_ratio, "Global EPLB should have <= load imbalance than node-constrained EPLB"
    assert lb["hierarchical_eplb_288"]["distinct_groups_per_node"] == [2, 2, 2, 2]
    assert (
        lb["hierarchical_eplb_288"]["inter_node_ib_dispatch_fraction"]
        < lb["global_eplb_288"]["inter_node_ib_dispatch_fraction"]
    )

    mig = res["sglang_5tier_online_migration"]
    intra_fracs = mig["intra_group_shift_classification"]["fractions"]
    inter_fracs = mig["inter_group_shift_classification"]["fractions"]
    assert math.isclose(sum(intra_fracs.values()), 1.0, abs_tol=1e-3)
    assert math.isclose(sum(inter_fracs.values()), 1.0, abs_tol=1e-3)
    assert intra_fracs["cross_node"] == 0.0, "Group-preserving intra-node rebalance should require 0 cross-node P2P!"
    assert mig["chunked_2layer_intra_5tier_step_stall_ms"] < 2.0
    assert mig["stall_reduction_vs_naive_full_x"] > 35.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    res = run_simulation()
    if args.check:
        _check(res)
        print("deepep_eplb_expert_rebalancing: OK")
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    elif not args.check:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
