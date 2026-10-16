"""Evaluate PyTorch DCP deduplication, LPT thread bucketing, D2H loader invariants, and GDS vs POSIX recovery bounds."""

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
DEFAULT_COPY_AHEAD_BYTES = 10_000_000
GDS_BLOCK_ALIGN_BYTES = 4096
LIMITATIONS = [
    "own-offline synthetic arithmetic; no NVMe array, Lustre/WEKA parallel filesystem, PCIe, or GPU execution",
    "models static invariants in pytorch/pytorch@1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340 torch/distributed/checkpoint",
    "models GPUDirect Storage (cuFile) DMA vs POSIX page-cache bounce paths per NVIDIA GDS Overview Guide",
    "Young-Daly checkpoint interval is a first-order analytical approximation assuming Poisson failures",
]


def _ascii_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or not value.replace("_", "").isalnum():
        raise ValueError(f"{name}: expected non-empty ASCII alphanumeric/underscore label")
    return value


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name}: expected finite positive number")
    return float(value)


def _dedup_replicated_bytes_by_rank(
    world_size: int,
    sharded_bytes_per_rank: list[int],
    replicated_tensor_bytes: list[int],
    *,
    save_to_lowest_rank: bool,
) -> list[int]:
    """Reproduce torch/distributed/checkpoint/_dedup_save_plans.py:dedup_save_plans."""
    plan_to_size = list(sharded_bytes_per_rank)
    for item_bytes in replicated_tensor_bytes:
        if save_to_lowest_rank:
            select_rank = 0
        else:
            select_rank = min(range(world_size), key=lambda r: plan_to_size[r])
        plan_to_size[select_rank] += item_bytes
    return plan_to_size


def _lpt_thread_buckets(item_sizes_bytes: list[int], thread_count: int) -> list[int]:
    """Reproduce torch/distributed/checkpoint/filesystem.py:_split_by_size_and_type."""
    buckets = [0] * thread_count
    for size in sorted(item_sizes_bytes, reverse=True):
        target = min(range(thread_count), key=lambda idx: buckets[idx])
        buckets[target] += size
    return buckets


def evaluate_checkpoint_case(
    *,
    case_id: str,
    world_size: int,
    sharded_tensor_bytes_per_rank: list[int],
    replicated_tensor_bytes: list[int],
    writer_thread_count: int,
    per_thread_copy_ahead_bytes: int = DEFAULT_COPY_AHEAD_BYTES,
    async_staged_to_pinned_cpu: bool = False,
    gds_enabled: bool = False,
    unaligned_4k_io: bool = False,
    uvm_managed_memory: bool = False,
    exceeds_bar1_aperture: bool = False,
    pcie_d2h_bytes_per_sec_per_rank: int,
    host_ram_Copy_bytes_per_sec_per_rank: int,
    storage_write_bytes_per_sec_per_rank: int,
    cluster_aggregate_storage_bytes_per_sec: int,
    metadata_fsync_barrier_seconds: float,
    mtbf_hours: float,
) -> dict:
    """Evaluate DCP deduplication, thread bucketing, loader selection, and GDS/POSIX step stall vs recovery."""
    label = _ascii_label(case_id, "case_id")
    ranks = integer(world_size, "world_size")
    threads = integer(writer_thread_count, "writer_thread_count")
    copy_ahead = int(per_thread_copy_ahead_bytes)
    if type(per_thread_copy_ahead_bytes) is not int or copy_ahead < 0:
        raise ValueError("per_thread_copy_ahead_bytes: expected non-negative int")
    pcie_bw = integer(pcie_d2h_bytes_per_sec_per_rank, "pcie_d2h_bytes_per_sec_per_rank")
    ram_bw = integer(host_ram_Copy_bytes_per_sec_per_rank, "host_ram_Copy_bytes_per_sec_per_rank")
    rank_storage_bw = integer(storage_write_bytes_per_sec_per_rank, "storage_write_bytes_per_sec_per_rank")
    cluster_storage_bw = integer(cluster_aggregate_storage_bytes_per_sec, "cluster_aggregate_storage_bytes_per_sec")
    fsync_s = _positive_float(metadata_fsync_barrier_seconds, "metadata_fsync_barrier_seconds")
    mtbf_h = _positive_float(mtbf_hours, "mtbf_hours")

    if (
        not isinstance(sharded_tensor_bytes_per_rank, list)
        or not sharded_tensor_bytes_per_rank
        or any(type(x) is not int or x <= 0 for x in sharded_tensor_bytes_per_rank)
    ):
        raise ValueError("sharded_tensor_bytes_per_rank: expected non-empty list of positive ints")
    if (
        not isinstance(replicated_tensor_bytes, list)
        or any(type(x) is not int or x <= 0 for x in replicated_tensor_bytes)
    ):
        raise ValueError("replicated_tensor_bytes: expected list of positive ints")
    for flag_name, flag_val in (
        ("async_staged_to_pinned_cpu", async_staged_to_pinned_cpu),
        ("gds_enabled", gds_enabled),
        ("unaligned_4k_io", unaligned_4k_io),
        ("uvm_managed_memory", uvm_managed_memory),
        ("exceeds_bar1_aperture", exceeds_bar1_aperture),
    ):
        if type(flag_val) is not bool:
            raise ValueError(f"{flag_name}: expected bool")

    sharded_per_rank = sum(sharded_tensor_bytes_per_rank)
    replicated_total = sum(replicated_tensor_bytes)
    base_by_rank = [sharded_per_rank] * ranks

    naive_total_bytes = ranks * (sharded_per_rank + replicated_total)
    lowest_rank_plans = _dedup_replicated_bytes_by_rank(
        ranks, base_by_rank, replicated_tensor_bytes, save_to_lowest_rank=True
    )
    balanced_plans = _dedup_replicated_bytes_by_rank(
        ranks, base_by_rank, replicated_tensor_bytes, save_to_lowest_rank=False
    )
    dedup_total_bytes = sum(balanced_plans)
    max_rank_bytes_lowest = max(lowest_rank_plans)
    max_rank_bytes_balanced = max(balanced_plans)

    # Rank 0 under balanced deduplication owns its sharded items plus its assigned subset of replicated items.
    rank0_items = list(sharded_tensor_bytes_per_rank)
    sim_sizes = list(base_by_rank)
    for item_bytes in replicated_tensor_bytes:
        sel = min(range(ranks), key=lambda r: sim_sizes[r])
        sim_sizes[sel] += item_bytes
        if sel == 0:
            rank0_items.append(item_bytes)

    thread_buckets = _lpt_thread_buckets(rank0_items, threads)
    max_bucket_bytes = max(thread_buckets)
    min_bucket_bytes = min(thread_buckets)
    bucket_imbalance_ratio = round(max_bucket_bytes / max(1, (sum(thread_buckets) / threads)), 6)

    # Loader selection invariant in filesystem.py:249-265:
    # _OverlappingCpuLoader is only selected when thread_count == 1 and copy_ahead > 0 (and not already on CPU).
    gds_compat_fallback = gds_enabled and (unaligned_4k_io or uvm_managed_memory or exceeds_bar1_aperture)
    effective_gds_direct_dma = gds_enabled and not gds_compat_fallback

    if effective_gds_direct_dma:
        loader_mode = "cufile_gds_direct_dma"
    elif async_staged_to_pinned_cpu:
        loader_mode = "async_blocking_stager_then_cpu_threads"
    elif threads == 1 and copy_ahead > 0:
        loader_mode = "overlapping_cpu_loader_single_thread"
    else:
        loader_mode = "serial_cpu_loader_multi_thread_fallback"

    # Effective per-rank storage bandwidth is capped by both per-rank storage ceiling and cluster aggregate share
    effective_rank_storage_bw = min(rank_storage_bw, cluster_storage_bw / ranks)
    effective_thread_bw = effective_rank_storage_bw / threads
    storage_write_time_s = max_bucket_bytes / effective_thread_bw
    cluster_bandwidth_floor_s = dedup_total_bytes / cluster_storage_bw
    effective_storage_phase_s = max(storage_write_time_s, cluster_bandwidth_floor_s)

    d2h_time_s = max_rank_bytes_balanced / pcie_bw
    posix_page_cache_bounce_s = (2.0 * max_rank_bytes_balanced) / ram_bw

    if effective_gds_direct_dma:
        host_memory_bounce_bytes_per_rank = 0
        persist_completion_time_s = max(d2h_time_s, effective_storage_phase_s) + fsync_s
        step_blocking_time_s = persist_completion_time_s
    elif async_staged_to_pinned_cpu:
        # BlockingAsyncStager copies GPU -> pinned CPU RAM synchronously before returning to training loop
        host_memory_bounce_bytes_per_rank = 2 * max_rank_bytes_balanced
        stage_d2h_time_s = d2h_time_s + (max_rank_bytes_balanced / ram_bw)
        step_blocking_time_s = stage_d2h_time_s
        persist_completion_time_s = stage_d2h_time_s + effective_storage_phase_s + posix_page_cache_bounce_s + fsync_s
    elif loader_mode == "overlapping_cpu_loader_single_thread":
        host_memory_bounce_bytes_per_rank = 2 * max_rank_bytes_balanced
        persist_completion_time_s = max(d2h_time_s, effective_storage_phase_s + posix_page_cache_bounce_s) + fsync_s
        step_blocking_time_s = persist_completion_time_s
    else:
        # _SerialCpuLoader: each worker thread synchronously copies D2H to CPU and writes through POSIX page cache
        host_memory_bounce_bytes_per_rank = 2 * max_rank_bytes_balanced
        persist_completion_time_s = d2h_time_s + posix_page_cache_bounce_s + effective_storage_phase_s + fsync_s
        step_blocking_time_s = persist_completion_time_s

    # Young-Daly first-order optimal checkpoint interval: tau_opt = sqrt(2 * T_stall * MTBF)
    mtbf_seconds = mtbf_h * 3600.0
    optimal_interval_seconds = math.sqrt(2.0 * step_blocking_time_s * mtbf_seconds)
    # Goodput fraction under optimal interval: 1 - sqrt(2 * T_stall / MTBF) - (T_recovery / MTBF)
    load_recovery_time_s = persist_completion_time_s
    steady_state_goodput_fraction = max(
        0.0,
        1.0 - math.sqrt((2.0 * step_blocking_time_s) / mtbf_seconds) - (load_recovery_time_s / mtbf_seconds),
    )

    return {
        "case_id": label,
        "world_size": ranks,
        "writer_thread_count": threads,
        "naive_replicated_total_bytes": naive_total_bytes,
        "dedup_total_bytes": dedup_total_bytes,
        "dedup_saved_bytes": naive_total_bytes - dedup_total_bytes,
        "max_rank_write_bytes_lowest_rank": max_rank_bytes_lowest,
        "max_rank_write_bytes_balanced": max_rank_bytes_balanced,
        "rank0_skew_reduction_ratio": round(max_rank_bytes_lowest / max_rank_bytes_balanced, 6),
        "rank0_thread_bucket_bytes": thread_buckets,
        "max_thread_bucket_bytes": max_bucket_bytes,
        "min_thread_bucket_bytes": min_bucket_bytes,
        "bucket_imbalance_ratio": bucket_imbalance_ratio,
        "loader_mode": loader_mode,
        "gds_compatibility_fallback": gds_compat_fallback,
        "effective_gds_direct_dma": effective_gds_direct_dma,
        "host_memory_bounce_bytes_per_rank": host_memory_bounce_bytes_per_rank,
        "d2h_transfer_time_seconds": round(d2h_time_s, 6),
        "effective_storage_phase_seconds": round(effective_storage_phase_s, 6),
        "step_blocking_time_seconds": round(step_blocking_time_s, 6),
        "persist_completion_time_seconds": round(persist_completion_time_s, 6),
        "optimal_checkpoint_interval_minutes": round(optimal_interval_seconds / 60.0, 4),
        "steady_state_goodput_fraction": round(steady_state_goodput_fraction, 6),
    }


def calculate(fixture: dict) -> dict:
    """Evaluate all synthetic distributed checkpoint and storage recovery cases."""
    if (
        not isinstance(fixture, dict)
        or type(fixture.get("schema_version")) is not int
        or fixture["schema_version"] != 1
        or fixture.get("synthetic") is not True
        or not isinstance(fixture.get("cases"), list)
        or not fixture["cases"]
        or any(not isinstance(case, dict) for case in fixture["cases"])
    ):
        raise ValueError("fixture: expected version-one nonempty synthetic cases")
    return {
        "schema_version": 1,
        "evidence_status": "own-offline",
        "synthetic": True,
        "cases": [{"inputs": case, "result": evaluate_checkpoint_case(**case)} for case in fixture["cases"]],
        "limitations": LIMITATIONS,
    }


def main(argv: list[str] | None = None) -> int:
    """Generate results.json or verify exact match with --check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        fixture = json.loads((HERE / "input.json").read_text(encoding="utf-8"))
        content = json.dumps(calculate(fixture), indent=2, sort_keys=True, allow_nan=False) + "\n"
        output = HERE / "results.json"
        if args.check:
            if not output.is_file() or output.read_bytes() != content.encode("utf-8"):
                print("Checkpoint recovery check failed: missing or different results.json", file=sys.stderr)
                return 1
        else:
            with output.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        print("Checkpoint recovery failed: invalid fixture or output", file=sys.stderr)
        return 1
    print(
        "own-offline checkpoint recovery verified"
        if args.check
        else "own-offline checkpoint recovery regenerated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
