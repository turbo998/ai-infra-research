"""Evaluate Kubernetes Topology Manager hint merging, Slurm topology/block level & segment fitting, NVIDIA k8s-device-plugin XID/MIG isolation, and PyTorch torchelastic goodput."""

import argparse
import json
import math
from pathlib import Path
import sys

if __package__:
    from ..capacity_roofline.budget import integer
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.capacity_roofline.budget import integer


HERE = Path(__file__).resolve().parent
NVML_DEVICE_WIDE_SENTINEL = 0xFFFFFFFF
DEFAULT_IGNORED_APPLICATION_XIDS = frozenset({13, 31, 43, 45, 68, 109})
SUPPORTED_K8S_POLICIES = frozenset({"none", "best-effort", "restricted", "single-numa-node"})
LIMITATIONS = [
    "own-offline synthetic model; no live Kubernetes cluster, Slurm controller, NVML GPU hardware, or torchelastic process group execution",
    "reproduces static algorithms from kubernetes@f78e722310e50bcaca9276be22276d9e91d91308, slurm@ad46cad8e0969e8e5945b88f0a21f6bed0f9a3bf, k8s-device-plugin@487d7a93894b0f3ad08097169593972eb3ca8206, and pytorch@1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340",
    "goodput formula uses first-order Poisson failure rate lambda = N / MTBF_node and deterministic checkpoint/rendezvous overheads",
]


def _ascii_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or not value.replace("_", "").isalnum():
        raise ValueError(f"{name}: expected non-empty ASCII alphanumeric/underscore label")
    return value


def _popcount(mask: int) -> int:
    return bin(mask).count("1")


def evaluate_k8s_topology_merge(
    *,
    numa_nodes: int,
    cpu_hint_mask: int,
    gpu_hint_mask: int,
    rdma_nic_hint_mask: int,
    cpu_hint_preferred: bool,
    gpu_hint_preferred: bool,
    rdma_nic_hint_preferred: bool,
    k8s_policy: str,
) -> dict:
    """Reproduce Kubernetes pkg/kubelet/cm/topologymanager/policy.go mergePermutation & admission."""
    n_numa = integer(numa_nodes, "numa_nodes")
    if n_numa > 8:
        raise ValueError("numa_nodes: kubelet supports up to 8 NUMA nodes before state explosion")
    default_affinity = (1 << n_numa) - 1
    masks = []
    for name, val in (
        ("cpu_hint_mask", cpu_hint_mask),
        ("gpu_hint_mask", gpu_hint_mask),
        ("rdma_nic_hint_mask", rdma_nic_hint_mask),
    ):
        m = integer(val, name)
        if m > default_affinity:
            raise ValueError(f"{name}: bitmask exceeds numa_nodes")
        masks.append(m)

    for name, flag in (
        ("cpu_hint_preferred", cpu_hint_preferred),
        ("gpu_hint_preferred", gpu_hint_preferred),
        ("rdma_nic_hint_preferred", rdma_nic_hint_preferred),
    ):
        if type(flag) is not bool:
            raise ValueError(f"{name}: expected bool")

    if k8s_policy not in SUPPORTED_K8S_POLICIES:
        raise ValueError("k8s_policy: unsupported policy")

    # Bitwise AND across all provider affinities (policy.go:46-57)
    merged_mask = default_affinity
    for m in masks:
        merged_mask &= m

    all_equal = (masks[0] == masks[1] == masks[2])
    all_preferred = cpu_hint_preferred and gpu_hint_preferred and rdma_nic_hint_preferred
    merged_preferred = bool(merged_mask > 0 and all_equal and all_preferred)
    merged_numa_count = _popcount(merged_mask) if merged_mask > 0 else n_numa

    if k8s_policy == "single-numa-node":
        # policy_single_numa_node.go:46-76 requires Count()==1 and Preferred==true
        single_numa_valid = (merged_mask > 0 and merged_numa_count == 1 and merged_preferred)
        admitted = single_numa_valid
    elif k8s_policy == "restricted":
        # policy_restricted.go:43-48 admits iff bestHint.Preferred is true
        admitted = merged_preferred
    else:
        admitted = True

    return {
        "default_affinity_mask": default_affinity,
        "merged_affinity_mask": merged_mask,
        "merged_numa_count": merged_numa_count,
        "all_provider_masks_equal": all_equal,
        "merged_preferred": merged_preferred,
        "pod_admitted": admitted,
    }


def evaluate_slurm_block_topology(
    *,
    bblock_node_cnt: int,
    block_levels: list[int],
    requested_nodes: int,
    segment_size: int,
    available_nodes_per_bblock: list[int],
) -> dict:
    """Reproduce Slurm src/plugins/topology/block/eval_nodes_block.c _get_block_level, segment check, and _choose_best_bblock."""
    bblock_size = integer(bblock_node_cnt, "bblock_node_cnt")
    req_nodes = integer(requested_nodes, "requested_nodes")
    seg_size = integer(segment_size, "segment_size")

    if not isinstance(block_levels, list) or not block_levels:
        raise ValueError("block_levels: expected non-empty list of non-negative ints")
    levels = sorted(set(int(lv) for lv in block_levels))
    if any(type(lv) is not int or lv < 0 or lv > 15 for lv in block_levels):
        raise ValueError("block_levels: invalid level in [0, 15]")

    if not isinstance(available_nodes_per_bblock, list) or not available_nodes_per_bblock:
        raise ValueError("available_nodes_per_bblock: expected non-empty list")
    avail_bblocks = []
    for x in available_nodes_per_bblock:
        if type(x) is not int or x < 0 or x > bblock_size:
            raise ValueError("available_nodes_per_bblock: must be int in [0, bblock_node_cnt]")
        avail_bblocks.append(x)

    segment_divisible = (req_nodes % seg_size == 0) and (seg_size <= bblock_size)
    segment_cnt = (req_nodes // seg_size) if segment_divisible else 0

    # _get_block_level (eval_nodes_block.c:248-266)
    required_bblocks = (req_nodes + bblock_size - 1) // bblock_size
    raw_level = math.ceil(math.log2(max(1, required_bblocks)))
    selected_level = -1
    for lv in levels:
        if lv >= raw_level:
            selected_level = lv
            break

    if selected_level >= 0:
        bblock_per_block = 1 << selected_level
        selected_block_capacity_nodes = bblock_per_block * bblock_size
    else:
        bblock_per_block = 0
        selected_block_capacity_nodes = 0

    # _choose_best_bblock best-fit selection for first pass (eval_nodes_block.c:114-149)
    best_bblock_index = -1
    best_fit = False
    for idx, nodes_on_bblock in enumerate(avail_bblocks):
        if nodes_on_bblock == 0:
            continue
        cur_fit = nodes_on_bblock >= req_nodes
        if best_bblock_index == -1:
            best_bblock_index = idx
            best_fit = cur_fit
            continue
        if cur_fit and best_fit and (nodes_on_bblock < avail_bblocks[best_bblock_index]):
            best_bblock_index = idx
        elif cur_fit and not best_fit:
            best_bblock_index = idx
            best_fit = True
        elif not cur_fit and not best_fit and (nodes_on_bblock > avail_bblocks[best_bblock_index]):
            best_bblock_index = idx

    return {
        "required_bblocks": required_bblocks,
        "raw_ceil_log2_level": raw_level,
        "selected_block_level": selected_level,
        "bblock_per_block": bblock_per_block,
        "selected_block_capacity_nodes": selected_block_capacity_nodes,
        "segment_divisible_and_fits_bblock": segment_divisible,
        "segment_cnt": segment_cnt,
        "best_initial_bblock_index": best_bblock_index,
        "best_initial_bblock_avail_nodes": avail_bblocks[best_bblock_index] if best_bblock_index >= 0 else 0,
        "best_initial_bblock_single_fit": best_fit,
    }


def evaluate_nvml_xid_isolation(
    *,
    nvml_xid_code: int,
    nvml_gi_id: int,
    nvml_ci_id: int,
    total_mig_slices_on_gpu: int,
) -> dict:
    """Reproduce NVIDIA k8s-device-plugin internal/rm/health.go XID filtering and MIG slice isolation."""
    xid = integer(nvml_xid_code, "nvml_xid_code")
    if type(nvml_gi_id) is not int or nvml_gi_id < 0 or nvml_gi_id > NVML_DEVICE_WIDE_SENTINEL:
        raise ValueError("nvml_gi_id: expected non-negative 32-bit int")
    if type(nvml_ci_id) is not int or nvml_ci_id < 0 or nvml_ci_id > NVML_DEVICE_WIDE_SENTINEL:
        raise ValueError("nvml_ci_id: expected non-negative 32-bit int")
    mig_slices = integer(total_mig_slices_on_gpu, "total_mig_slices_on_gpu")

    ignored_by_default = xid in DEFAULT_IGNORED_APPLICATION_XIDS
    is_mig_scoped_event = (
        nvml_gi_id != NVML_DEVICE_WIDE_SENTINEL and nvml_ci_id != NVML_DEVICE_WIDE_SENTINEL
    )

    if ignored_by_default:
        unhealthy_mig_slices = 0
        blast_radius_scope = "ignored_application_xid"
    elif is_mig_scoped_event:
        unhealthy_mig_slices = 1
        blast_radius_scope = "single_mig_slice"
    else:
        unhealthy_mig_slices = mig_slices
        blast_radius_scope = "entire_parent_gpu"

    return {
        "xid_code": xid,
        "ignored_by_default_policy": ignored_by_default,
        "is_mig_scoped_event": is_mig_scoped_event,
        "blast_radius_scope": blast_radius_scope,
        "unhealthy_mig_slices": unhealthy_mig_slices,
        "healthy_remaining_mig_slices": mig_slices - unhealthy_mig_slices,
    }


def evaluate_elastic_goodput(
    *,
    world_size_nodes: int,
    node_mtbf_hours: float,
    checkpoint_interval_sec: float,
    checkpoint_write_sec: float,
    rdzv_rejoin_sec: float,
    cold_node_reprovision_sec: float,
    hot_standby_nodes: int,
) -> dict:
    """Compare training goodput under cold node replacement vs torchelastic hot-standby rendezvous."""
    nodes = integer(world_size_nodes, "world_size_nodes")
    if type(hot_standby_nodes) is not int or hot_standby_nodes < 0:
        raise ValueError("hot_standby_nodes: expected non-negative int")
    for name, val in (
        ("node_mtbf_hours", node_mtbf_hours),
        ("checkpoint_interval_sec", checkpoint_interval_sec),
        ("checkpoint_write_sec", checkpoint_write_sec),
        ("rdzv_rejoin_sec", rdzv_rejoin_sec),
        ("cold_node_reprovision_sec", cold_node_reprovision_sec),
    ):
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name}: expected finite positive number")

    cluster_mtbf_hours = node_mtbf_hours / nodes
    cluster_mtbf_sec = cluster_mtbf_hours * 3600.0
    failure_rate_per_sec = 1.0 / cluster_mtbf_sec

    # Optimal periodic checkpoint interval (Young-Daly approximation): sqrt(2 * T_ckpt * MTBF_cluster)
    young_daly_optimal_ckpt_interval_sec = math.sqrt(2.0 * checkpoint_write_sec * cluster_mtbf_sec)
    expected_rework_sec = 0.5 * checkpoint_interval_sec

    cold_recovery_downtime_sec = cold_node_reprovision_sec + rdzv_rejoin_sec + expected_rework_sec
    hot_standby_recovery_downtime_sec = rdzv_rejoin_sec + expected_rework_sec

    checkpoint_overhead_fraction = checkpoint_write_sec / (checkpoint_interval_sec + checkpoint_write_sec)
    cold_failure_lost_fraction = failure_rate_per_sec * cold_recovery_downtime_sec
    hot_failure_lost_fraction = failure_rate_per_sec * hot_standby_recovery_downtime_sec

    goodput_cold_reprovision = max(0.0, (1.0 - checkpoint_overhead_fraction) * (1.0 - cold_failure_lost_fraction))
    goodput_hot_standby = max(0.0, (1.0 - checkpoint_overhead_fraction) * (1.0 - hot_failure_lost_fraction))

    # Effective goodput per total provisioned node (accounting for standby node capex/opex)
    effective_goodput_per_provisioned_node_hot = (
        goodput_hot_standby * nodes / (nodes + hot_standby_nodes)
    )

    return {
        "cluster_mtbf_hours": round(cluster_mtbf_hours, 4),
        "young_daly_optimal_ckpt_interval_sec": round(young_daly_optimal_ckpt_interval_sec, 2),
        "checkpoint_overhead_fraction": round(checkpoint_overhead_fraction, 4),
        "cold_recovery_downtime_sec": round(cold_recovery_downtime_sec, 2),
        "hot_standby_recovery_downtime_sec": round(hot_standby_recovery_downtime_sec, 2),
        "goodput_cold_reprovision": round(goodput_cold_reprovision, 4),
        "goodput_hot_standby": round(goodput_hot_standby, 4),
        "goodput_gain_pts": round((goodput_hot_standby - goodput_cold_reprovision) * 100.0, 2),
        "effective_goodput_per_provisioned_node_hot": round(effective_goodput_per_provisioned_node_hot, 4),
    }


def evaluate_cluster_orchestration_case(
    *,
    case_id: str,
    numa_nodes: int,
    cpu_hint_mask: int,
    gpu_hint_mask: int,
    rdma_nic_hint_mask: int,
    cpu_hint_preferred: bool,
    gpu_hint_preferred: bool,
    rdma_nic_hint_preferred: bool,
    k8s_policy: str,
    bblock_node_cnt: int,
    block_levels: list[int],
    requested_nodes: int,
    segment_size: int,
    available_nodes_per_bblock: list[int],
    nvml_xid_code: int,
    nvml_gi_id: int,
    nvml_ci_id: int,
    total_mig_slices_on_gpu: int,
    world_size_nodes: int,
    node_mtbf_hours: float,
    checkpoint_interval_sec: float,
    checkpoint_write_sec: float,
    rdzv_rejoin_sec: float,
    cold_node_reprovision_sec: float,
    hot_standby_nodes: int,
) -> dict:
    """Evaluate a complete cluster orchestration scenario across K8s, Slurm, NVML XID, and torchelastic."""
    label = _ascii_label(case_id, "case_id")
    k8s = evaluate_k8s_topology_merge(
        numa_nodes=numa_nodes,
        cpu_hint_mask=cpu_hint_mask,
        gpu_hint_mask=gpu_hint_mask,
        rdma_nic_hint_mask=rdma_nic_hint_mask,
        cpu_hint_preferred=cpu_hint_preferred,
        gpu_hint_preferred=gpu_hint_preferred,
        rdma_nic_hint_preferred=rdma_nic_hint_preferred,
        k8s_policy=k8s_policy,
    )
    slurm = evaluate_slurm_block_topology(
        bblock_node_cnt=bblock_node_cnt,
        block_levels=block_levels,
        requested_nodes=requested_nodes,
        segment_size=segment_size,
        available_nodes_per_bblock=available_nodes_per_bblock,
    )
    xid = evaluate_nvml_xid_isolation(
        nvml_xid_code=nvml_xid_code,
        nvml_gi_id=nvml_gi_id,
        nvml_ci_id=nvml_ci_id,
        total_mig_slices_on_gpu=total_mig_slices_on_gpu,
    )
    elastic = evaluate_elastic_goodput(
        world_size_nodes=world_size_nodes,
        node_mtbf_hours=node_mtbf_hours,
        checkpoint_interval_sec=checkpoint_interval_sec,
        checkpoint_write_sec=checkpoint_write_sec,
        rdzv_rejoin_sec=rdzv_rejoin_sec,
        cold_node_reprovision_sec=cold_node_reprovision_sec,
        hot_standby_nodes=hot_standby_nodes,
    )
    return {
        "case_id": label,
        "k8s_policy": k8s_policy,
        "k8s_topology": k8s,
        "slurm_block": slurm,
        "nvml_xid_isolation": xid,
        "elastic_goodput": elastic,
    }


def evaluate_fixture(fixture: dict) -> dict:
    if not isinstance(fixture, dict) or fixture.get("experiment_id") != "cluster_orchestration":
        raise ValueError("fixture: expected experiment_id 'cluster_orchestration'")
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture.cases: expected non-empty list")
    evaluated = []
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case: expected object")
        result = evaluate_cluster_orchestration_case(**case)
        if result["case_id"] in seen:
            raise ValueError(f"duplicate case_id: {result['case_id']}")
        seen.add(result["case_id"])
        evaluated.append(result)
    return {
        "schema_version": 1,
        "experiment_id": "cluster_orchestration",
        "evidence_state": "own-offline",
        "limitations": list(LIMITATIONS),
        "cases": evaluated,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "input.json")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    fixture = json.loads(args.input.read_text(encoding="utf-8"))
    results = evaluate_fixture(fixture)
    rendered = json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        expected = args.output.read_text(encoding="utf-8")
        if expected != rendered:
            print("results.json is out of date with input.json and run.py", file=sys.stderr)
            return 1
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
