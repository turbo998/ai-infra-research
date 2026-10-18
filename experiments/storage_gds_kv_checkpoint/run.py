#!/usr/bin/env python3
"""Offline analytical model for Storage I/O Paths: POSIX vs O_DIRECT vs GDS cuFile vs 3FS USRBIO."""

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


def evaluate_io_path(
    workload: dict[str, Any],
    hw: dict[str, Any],
    io_path: dict[str, Any],
) -> dict[str, Any]:
    direction = str(workload["direction"])
    payload_gib = float(workload["payload_gib"])
    payload_gb = payload_gib * 1.073741824

    nvme_count = int(hw["nvme_drives_per_node"])
    local_nvme_bw_gbps = nvme_count * (
        float(hw["nvme_read_gbps_per_drive"])
        if direction == "read"
        else float(hw["nvme_write_gbps_per_drive"])
    )
    rdma_bw_gbps = int(hw["rdma_nics_per_node"]) * float(hw["rdma_gbps_per_nic_unidir"])

    # Determine media/network ceiling
    if bool(io_path["uses_remote_3fs_craq"]):
        if direction == "read":
            # CRAQ write-all-read-any saturates client RDMA ingress
            media_ceiling_gbps = rdma_bw_gbps
        else:
            # CRAQ chain replication + RocksDB/chunk COW commit overhead across 3 replicas
            media_ceiling_gbps = rdma_bw_gbps * 0.72
    else:
        media_ceiling_gbps = local_nvme_bw_gbps

    rc_passes = float(io_path["root_complex_passes"])
    dram_passes = float(io_path["dram_memcpy_passes"])
    rc_cap = float(hw["pcie_root_complex_cap_gbps"])
    p2p_cap = float(hw["pcie_switch_p2p_cap_gbps"])
    dram_bw = float(hw["host_dram_mem_bw_gbps"])

    rc_limit_gbps = (rc_cap / rc_passes) if rc_passes > 0 else p2p_cap
    dram_limit_gbps = (dram_bw / (2.0 * dram_passes)) if dram_passes > 0 else p2p_cap
    bus_limit_gbps = min(p2p_cap, rc_limit_gbps, dram_limit_gbps)

    effective_bw_gbps = min(media_ceiling_gbps, bus_limit_gbps)
    effective_bw_gib_s = effective_bw_gbps / 1.073741824

    median_duration_sec = payload_gb / effective_bw_gbps
    p99_duration_sec = median_duration_sec * float(io_path["p99_tail_multiplier"])
    root_complex_traffic_gib = payload_gib * rc_passes

    return {
        "workload_id": workload["workload_id"],
        "direction": direction,
        "payload_gib": payload_gib,
        "path_id": io_path["path_id"],
        "label": io_path["label"],
        "effective_bw_gbps": round(effective_bw_gbps, 2),
        "effective_bw_gib_s": round(effective_bw_gib_s, 2),
        "median_duration_sec": round(median_duration_sec, 3),
        "p99_duration_sec": round(p99_duration_sec, 3),
        "root_complex_passes": rc_passes,
        "root_complex_traffic_gib": round(root_complex_traffic_gib, 1),
        "cpu_cores_utilized": int(io_path["cpu_cores_utilized"]),
    }


def generate_results(config: dict[str, Any]) -> dict[str, Any]:
    hw = config["node_hardware"]
    rows: list[dict[str, Any]] = []
    for wl in config["workloads"]:
        for path in config["io_paths"]:
            rows.append(evaluate_io_path(wl, hw, path))

    kv_rows = {r["path_id"]: r for r in rows if r["workload_id"] == "kv_read_100gib"}
    ckpt_rows = {r["path_id"]: r for r in rows if r["workload_id"] == "ckpt_write_1024gib"}

    summary = {
        "kv_read_posix_bw_gbps": kv_rows["posix_page_cache"]["effective_bw_gbps"],
        "kv_read_odirect_bw_gbps": kv_rows["o_direct_pinned_bounce"]["effective_bw_gbps"],
        "kv_read_gds_bw_gbps": kv_rows["gds_cufile_bar1_p2p"]["effective_bw_gbps"],
        "kv_read_3fs_shm_bw_gbps": kv_rows["hf3fs_usrbio_shm_memcpy"]["effective_bw_gbps"],
        "kv_read_3fs_p2p_bw_gbps": kv_rows["hf3fs_usrbio_gpudirect_p2p"]["effective_bw_gbps"],
        "kv_read_100gib_posix_p99_sec": kv_rows["posix_page_cache"]["p99_duration_sec"],
        "kv_read_100gib_gds_p99_sec": kv_rows["gds_cufile_bar1_p2p"]["p99_duration_sec"],
        "kv_read_100gib_3fs_p2p_p99_sec": kv_rows["hf3fs_usrbio_gpudirect_p2p"]["p99_duration_sec"],
        "ckpt_1tib_posix_p99_sec": ckpt_rows["posix_page_cache"]["p99_duration_sec"],
        "ckpt_1tib_gds_p99_sec": ckpt_rows["gds_cufile_bar1_p2p"]["p99_duration_sec"],
        "ckpt_1tib_3fs_p2p_p99_sec": ckpt_rows["hf3fs_usrbio_gpudirect_p2p"]["p99_duration_sec"],
        "gds_vs_posix_kv_speedup_x": round(
            kv_rows["gds_cufile_bar1_p2p"]["effective_bw_gbps"]
            / kv_rows["posix_page_cache"]["effective_bw_gbps"],
            2,
        ),
        "hf3fs_p2p_vs_shm_kv_speedup_x": round(
            kv_rows["hf3fs_usrbio_gpudirect_p2p"]["effective_bw_gbps"]
            / kv_rows["hf3fs_usrbio_shm_memcpy"]["effective_bw_gbps"],
            2,
        ),
    }

    return {
        "experiment_id": "storage_gds_kv_checkpoint",
        "evidence_class": "own-offline",
        "node_hardware": hw,
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
            print("storage_gds_kv_checkpoint outputs are stale; re-run without --check.")
            return 1
        print("storage_gds_kv_checkpoint outputs verified.")
        return 0

    write_outputs(results)
    print(f"Wrote {RESULTS_JSON_PATH} and {RESULTS_CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
