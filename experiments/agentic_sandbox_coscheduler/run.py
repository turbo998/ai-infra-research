#!/usr/bin/env python3
"""Offline analytical model for Agentic RL 10,000-MicroVM UFFD & GPU KV Cache Co-Scheduling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "input.json"
RESULTS_JSON_PATH = ROOT / "results.json"
RESULTS_CSV_PATH = ROOT / "results.csv"


def load_input(path: Path = INPUT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_breakeven_thresholds(params: dict[str, Any]) -> dict[str, float]:
    tokens = int(params["active_kv_tokens"])
    bytes_per_tok = int(params["kv_bytes_per_token"])
    kv_gib = (tokens * bytes_per_tok) / (1024.0 ** 3)
    prefill_tps = float(params["prefill_tokens_per_sec_on_gpu"])
    gds_rt_bw_gib_s = float(params["gds_roundtrip_bw_gib_s"])

    # Time on GPU required to recompute the prefix from scratch
    recompute_gpu_sec = tokens / prefill_tps
    # Time on GDS/3FS required to offload + reload the KV block (round-trip = 2 * kv_gib / bw_per_dir,
    # where gds_roundtrip_bw_gib_s is the effective round-trip GiB/s rate = bw_per_dir / 2)
    gds_swap_gpu_sec = kv_gib / gds_rt_bw_gib_s

    # Break-even tool duration where holding KV in HBM (cost = kv_gib * t_tool)
    # equals active GPU stall cost of GDS swap or recompute (normalized to full-GPU HBM capacity of 80 GiB)
    gpu_usable_hbm_gib = 64.0
    t_breakeven_gds_sec = gds_swap_gpu_sec * (gpu_usable_hbm_gib / kv_gib)
    t_breakeven_recompute_sec = recompute_gpu_sec * (gpu_usable_hbm_gib / kv_gib)

    return {
        "kv_footprint_gib": round(kv_gib, 2),
        "recompute_gpu_sec": round(recompute_gpu_sec, 3),
        "gds_swap_gpu_sec": round(gds_swap_gpu_sec, 3),
        "breakeven_gds_offload_sec": round(t_breakeven_gds_sec, 2),
        "breakeven_recompute_sec": round(t_breakeven_recompute_sec, 2),
    }


def evaluate_tool_duration_and_uffd(
    params: dict[str, Any],
    tool_duration_sec: float,
    uffd_vms_per_sec: int,
) -> dict[str, Any]:
    thresholds = compute_breakeven_thresholds(params)
    kv_gib = thresholds["kv_footprint_gib"]
    gpu_usable_hbm_gib = 64.0

    # HBM GiB-seconds consumed per trajectory step under each policy
    pin_hbm_gib_sec = kv_gib * tool_duration_sec
    recompute_hbm_gib_sec = gpu_usable_hbm_gib * thresholds["recompute_gpu_sec"]

    # Calculate UFFD page-fault storm bandwidth demand on host NVMe/PCIe pool
    shared_pool_gbps = float(params["shared_pcie_nvme_pool_gbps"])
    working_set_mib = float(params["uffd_working_set_mib_per_vm"])
    uffd_demand_gbps = (uffd_vms_per_sec * working_set_mib * 1048576.0) / 1e9

    # Remaining PCIe/NVMe bandwidth available for GDS KV swap under unthrottled vs token-bucket co-scheduled UFFD
    unthrottled_gds_avail_gbps = max(4.0, shared_pool_gbps - uffd_demand_gbps)
    # Co-scheduled token bucket caps UFFD page-fault ingress at 35% of shared pool during KV offload bursts
    coscheduled_uffd_cap_gbps = shared_pool_gbps * 0.35
    coscheduled_gds_avail_gbps = shared_pool_gbps - min(uffd_demand_gbps, coscheduled_uffd_cap_gbps)

    base_gds_rt_gib_s = float(params["gds_roundtrip_bw_gib_s"])
    unthrottled_swap_sec = thresholds["gds_swap_gpu_sec"] * (shared_pool_gbps / unthrottled_gds_avail_gbps)
    coscheduled_swap_sec = thresholds["gds_swap_gpu_sec"] * (shared_pool_gbps / coscheduled_gds_avail_gbps)

    unthrottled_gds_hbm_gib_sec = gpu_usable_hbm_gib * unthrottled_swap_sec
    coscheduled_gds_hbm_gib_sec = gpu_usable_hbm_gib * coscheduled_swap_sec

    best_policy = "pin_in_hbm"
    best_cost = pin_hbm_gib_sec
    if coscheduled_gds_hbm_gib_sec < best_cost:
        best_policy = "gds_offload_coscheduled"
        best_cost = coscheduled_gds_hbm_gib_sec
    if recompute_hbm_gib_sec < best_cost:
        best_policy = "evict_and_recompute"
        best_cost = recompute_hbm_gib_sec

    # Effective concurrent rollouts supported per 8-GPU node (512 GiB usable KV HBM)
    # Pinning holds kv_gib for (decode_active_sec + tool_duration_sec), whereas co-scheduled offload holds it only during active decode + swap
    active_decode_sec = 2.2
    pin_concurrency_per_node = (512.0 / kv_gib) * (active_decode_sec / (active_decode_sec + tool_duration_sec))
    cosched_concurrency_per_node = (512.0 / kv_gib) * (
        active_decode_sec / (active_decode_sec + coscheduled_swap_sec)
    )

    return {
        "tool_duration_sec": tool_duration_sec,
        "uffd_vms_per_sec": uffd_vms_per_sec,
        "uffd_demand_gbps": round(uffd_demand_gbps, 2),
        "unthrottled_gds_avail_gbps": round(unthrottled_gds_avail_gbps, 2),
        "coscheduled_gds_avail_gbps": round(coscheduled_gds_avail_gbps, 2),
        "pin_hbm_gib_sec": round(pin_hbm_gib_sec, 2),
        "recompute_hbm_gib_sec": round(recompute_hbm_gib_sec, 2),
        "unthrottled_gds_hbm_gib_sec": round(unthrottled_gds_hbm_gib_sec, 2),
        "coscheduled_gds_hbm_gib_sec": round(coscheduled_gds_hbm_gib_sec, 2),
        "unthrottled_swap_sec": round(unthrottled_swap_sec, 3),
        "coscheduled_swap_sec": round(coscheduled_swap_sec, 3),
        "optimal_policy": best_policy,
        "effective_active_rollouts_pin": round(pin_concurrency_per_node, 1),
        "effective_active_rollouts_cosched": round(cosched_concurrency_per_node, 1),
    }


def generate_results(config: dict[str, Any]) -> dict[str, Any]:
    params = config["cluster_parameters"]
    thresholds = compute_breakeven_thresholds(params)

    rows: list[dict[str, Any]] = []
    for t_sec in config["tool_durations_sec"]:
        for vms_sec in config["uffd_burst_rates_vms_per_sec"]:
            rows.append(evaluate_tool_duration_and_uffd(params, float(t_sec), int(vms_sec)))

    row_30s_500 = next(
        r for r in rows if r["tool_duration_sec"] == 30.0 and r["uffd_vms_per_sec"] == 500
    )
    row_60s_500 = next(
        r for r in rows if r["tool_duration_sec"] == 60.0 and r["uffd_vms_per_sec"] == 500
    )

    summary = {
        **thresholds,
        "tool_30s_pin_hbm_gib_sec": row_30s_500["pin_hbm_gib_sec"],
        "tool_30s_cosched_gds_hbm_gib_sec": row_30s_500["coscheduled_gds_hbm_gib_sec"],
        "tool_30s_unthrottled_gds_hbm_gib_sec": row_30s_500["unthrottled_gds_hbm_gib_sec"],
        "tool_30s_rollout_gain_x": round(
            row_30s_500["effective_active_rollouts_cosched"]
            / row_30s_500["effective_active_rollouts_pin"],
            2,
        ),
        "tool_60s_rollout_gain_x": round(
            row_60s_500["effective_active_rollouts_cosched"]
            / row_60s_500["effective_active_rollouts_pin"],
            2,
        ),
        "uffd_500vms_unthrottled_swap_sec": row_30s_500["unthrottled_swap_sec"],
        "uffd_500vms_coscheduled_swap_sec": row_30s_500["coscheduled_swap_sec"],
    }

    return {
        "experiment_id": "agentic_sandbox_coscheduler",
        "evidence_class": "own-offline",
        "cluster_parameters": params,
        "rows": rows,
        "summary": summary,
    }


def write_outputs(results: dict[str, Any]) -> None:
    RESULTS_JSON_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(results["rows"][0].keys())
    with RESULTS_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results["rows"])


def check_outputs(expected: dict[str, Any]) -> bool:
    if not RESULTS_JSON_PATH.exists() or not RESULTS_CSV_PATH.exists():
        return False
    current = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
    return current == expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify outputs are up-to-date")
    args = parser.parse_args()

    config = load_input()
    results = generate_results(config)
    if args.check:
        if not check_outputs(results):
            print("agentic_sandbox_coscheduler outputs are stale; re-run without --check.")
            return 1
        print("agentic_sandbox_coscheduler outputs verified.")
        return 0

    write_outputs(results)
    print(f"Wrote {RESULTS_JSON_PATH} and {RESULTS_CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
