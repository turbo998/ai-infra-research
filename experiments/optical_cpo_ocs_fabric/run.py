#!/usr/bin/env python3
"""Offline analytical model for 100k-GPU Optical Interconnects: Pluggable DSP vs LPO vs CPO vs OCS."""

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


def evaluate_architecture_at_scale(
    arch: dict[str, Any],
    gpu_count: int,
    port_rate_gbps: float,
    checkpoint_interval_sec: float,
    restart_recovery_sec: float,
) -> dict[str, Any]:
    if gpu_count <= 0 or port_rate_gbps <= 0:
        raise ValueError("gpu_count and port_rate_gbps must be positive")
    endpoints_per_gpu = float(arch["transceiver_endpoints_per_gpu"])
    pj_per_bit = float(arch["pj_per_bit"])
    mtbf_hours = float(arch["mtbf_hours_per_endpoint"])
    oe_shared = int(arch["oe_shared_ports"])
    elsfp_shared = int(arch["elsfp_shared_ports"])
    reconfig_ms = float(arch["reconfig_latency_ms"])

    total_endpoints = int(round(gpu_count * endpoints_per_gpu))
    # 1 pJ/bit at 1 Gbps = 1 mW; at 800 Gbps, Watts per endpoint = 0.8 * pj_per_bit
    watts_per_endpoint = (port_rate_gbps * 1e9) * (pj_per_bit * 1e-12)
    optics_power_mw = (total_endpoints * watts_per_endpoint) / 1e6

    # Cluster link-flap failure rate per hour and per day
    cluster_flaps_per_hour = total_endpoints / mtbf_hours
    cluster_flaps_per_day = cluster_flaps_per_hour * 24.0
    cluster_mtbf_hours = 1.0 / cluster_flaps_per_hour

    # Expected rollback loss per flap = half of checkpoint interval + restart recovery time
    expected_loss_sec_per_flap = 0.5 * checkpoint_interval_sec + restart_recovery_sec
    lost_sec_per_hour = cluster_flaps_per_hour * expected_loss_sec_per_flap
    # Availability / synchronous training goodput factor: MTBF / (MTBF + MTTR_effective)
    mttr_hours = expected_loss_sec_per_flap / 3600.0
    sync_goodput_pct = 100.0 * (cluster_mtbf_hours / (cluster_mtbf_hours + mttr_hours))

    # Blast radius when a single optical engine (OE) or ELSFP module goes into low-power/fault
    oe_blast_radius_gbps = oe_shared * port_rate_gbps
    elsfp_blast_radius_gbps = elsfp_shared * port_rate_gbps

    return {
        "arch_id": arch["arch_id"],
        "label": arch["label"],
        "gpu_count": gpu_count,
        "total_optical_endpoints": total_endpoints,
        "pj_per_bit": round(pj_per_bit, 2),
        "watts_per_endpoint": round(watts_per_endpoint, 2),
        "optics_power_mw": round(optics_power_mw, 3),
        "cluster_mtbf_hours": round(cluster_mtbf_hours, 3),
        "cluster_flaps_per_day": round(cluster_flaps_per_day, 2),
        "sync_goodput_pct": round(sync_goodput_pct, 2),
        "oe_shared_ports": oe_shared,
        "elsfp_shared_ports": elsfp_shared,
        "elsfp_blast_radius_gbps": round(elsfp_blast_radius_gbps, 1),
        "reconfig_latency_ms": round(reconfig_ms, 2),
    }


def generate_results(config: dict[str, Any]) -> dict[str, Any]:
    scales = [int(x) for x in config["cluster_scales"]]
    port_rate = float(config["port_rate_gbps"])
    ckpt_sec = float(config["checkpoint_interval_sec"])
    restart_sec = float(config["restart_recovery_sec"])

    rows: list[dict[str, Any]] = []
    for scale in scales:
        for arch in config["architectures"]:
            rows.append(
                evaluate_architecture_at_scale(
                    arch=arch,
                    gpu_count=scale,
                    port_rate_gbps=port_rate,
                    checkpoint_interval_sec=ckpt_sec,
                    restart_recovery_sec=restart_sec,
                )
            )

    rows_100k = {r["arch_id"]: r for r in rows if r["gpu_count"] == 100000}
    dsp_100k = rows_100k["pluggable_dsp_3tier_800g"]
    lpo_100k = rows_100k["pluggable_lpo_3tier_800g"]
    cpo_100k = rows_100k["cpo_elsfp_3tier_800g"]
    ocs_100k = rows_100k["ocs_mems_sp_ineliminated_800g"]
    hybrid_100k = rows_100k["ocs_cpo_hybrid_800g"]

    summary = {
        "dsp_100k_optics_power_mw": dsp_100k["optics_power_mw"],
        "lpo_100k_optics_power_mw": lpo_100k["optics_power_mw"],
        "cpo_100k_optics_power_mw": cpo_100k["optics_power_mw"],
        "ocs_100k_optics_power_mw": ocs_100k["optics_power_mw"],
        "hybrid_100k_optics_power_mw": hybrid_100k["optics_power_mw"],
        "cpo_vs_dsp_power_savings_mw": round(
            dsp_100k["optics_power_mw"] - cpo_100k["optics_power_mw"], 3
        ),
        "hybrid_vs_dsp_power_savings_mw": round(
            dsp_100k["optics_power_mw"] - hybrid_100k["optics_power_mw"], 3
        ),
        "dsp_100k_flaps_per_day": dsp_100k["cluster_flaps_per_day"],
        "cpo_100k_flaps_per_day": cpo_100k["cluster_flaps_per_day"],
        "hybrid_100k_flaps_per_day": hybrid_100k["cluster_flaps_per_day"],
        "dsp_100k_sync_goodput_pct": dsp_100k["sync_goodput_pct"],
        "cpo_100k_sync_goodput_pct": cpo_100k["sync_goodput_pct"],
        "hybrid_100k_sync_goodput_pct": hybrid_100k["sync_goodput_pct"],
        "cpo_vs_dsp_goodput_gain_pts": round(
            cpo_100k["sync_goodput_pct"] - dsp_100k["sync_goodput_pct"], 2
        ),
    }

    return {
        "experiment_id": "optical_cpo_ocs_fabric",
        "evidence_class": "own-offline",
        "parameters": {
            "port_rate_gbps": port_rate,
            "checkpoint_interval_sec": ckpt_sec,
            "restart_recovery_sec": restart_sec,
        },
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
            print("optical_cpo_ocs_fabric outputs are stale; re-run without --check.")
            return 1
        print("optical_cpo_ocs_fabric outputs verified.")
        return 0

    write_outputs(results)
    print(f"Wrote {RESULTS_JSON_PATH} and {RESULTS_CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
