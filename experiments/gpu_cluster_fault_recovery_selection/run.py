#!/usr/bin/env python3
"""Deterministic Availability, MTBF & Recovery Goodput Evaluator for 1024-GPU Training Clusters.

Compares four fault-tolerance & NCCL hang recovery architectures on a 1,024 x H100
(128 nodes x 8 GPUs, $2.50/GPU-hour) cluster where per-GPU hardware/link failure rate
yields a cluster Mean Time Between Failures (MTBF) of 6.0 hours (4.0 faults/day),
with 30-minute (1,800 s) asynchronous checkpoints:

1. traditional_stop_the_world_restart:
   - Default 1,800s (30 min) NCCL_TIMEOUT wait on silent NIC/GPU hang + Slurm/K8s pod teardown
     + hardware Cordon/Drain + container re-pull + 9TB checkpoint reload + average 900s (15 min)
     lost training rollback work.
2. fast_hang_detect_warm_spare_swap:
   - NCCL Flight Recorder + 120s watchdog timeout + automated node Cordon + pre-warmed standby
     node swap + Parallel FS GDS checkpoint reload (cuts recovery outage from 2,520s to 330s,
     plus 900s average rollback loss = 1,230s lost work per fault).
3. inplace_comm_shrink_elastic_async_ckpt:
   - Custom application-managed in-place recovery built on NCCL 2.27+ ncclCommShrink (which officially
     only removes excluded ranks among surviving processes) combined with a separate communicator
     grow/re-init + state/dataloader resharding loop and 5-minute (300s) two-phase async DCP checkpointing
     (45s detection + 12s custom shrink/re-init + 8s state restore = 65s outage + 150s average rollback
     loss = 215s lost work per fault; synthetic timing input assumptions, not official vendor guarantees).
     Note: standard PyTorch torchrun/TorchElastic stops and restarts all workers on membership change
     (https://docs.pytorch.org/docs/2.15/elastic/run.html) and does not preserve surviving CUDA contexts
     in-place without custom application-level communicator management.
4. torchft_in_memory_replica_peer_recovery:
   - TorchFT (Lighthouse + Manager) replica-group / Hybrid-FSDP in-memory state streaming from surviving
     DP peers over 400G RDMA + warm spare node (zero disk reload, zero step rollback; 15s + 5s + 8s = 28s
     total recovery stall per fault; synthetic timing input assumptions, not official vendor guarantees).

Provenance:
- Analytical availability & MTBF goodput simulation (`synthetic`) calibrated against published
  Llama-3 16K GPU reliability reports and NCCL 2.27+ / TorchFT specifications (`anonymized_reported`,
  `independently_reproduced=false`).
- No physical 1,024-GPU cluster was provisioned during this run (`gpu_cluster_hardware_measured = "not_measured"`).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def build_results() -> dict:
    cluster_gpus = 1024
    gpu_hourly_usd = 2.50
    cluster_hourly_usd = cluster_gpus * gpu_hourly_usd  # $2,560 / hour
    hours_per_month = 730.0
    cluster_monthly_budget_usd = cluster_hourly_usd * hours_per_month  # $1,868,800 / month
    cluster_mtbf_hours = 6.0  # 1 fault every 6.0 hours = 21,600 seconds

    # Each architecture defines:
    # - hang_detection_sec: time to detect & isolate a silent RDMA/GPU hang
    # - orchestrator_reinit_sec: time to replace/reconfigure communicator & processes
    # - state_restore_sec: time to restore model/optimizer state (disk or peer RDMA)
    # - checkpoint_interval_sec: interval between durable states
    # - avg_rollback_loss_sec: 0.5 * checkpoint_interval_sec (or 0 if in-memory peer recovery)
    architectures = [
        {
            "id": "traditional_stop_the_world_restart",
            "label": "1. Traditional Stop-the-World Job Restart (30m NCCL Timeout + Cold Pod Restart + 30m Ckpt)",
            "hang_detection_sec": 1800.0,
            "orchestrator_reinit_sec": 480.0,
            "state_restore_sec": 240.0,
            "checkpoint_interval_sec": 1800.0,
            "avg_rollback_loss_sec": 900.0,
            "warm_spare_nodes_required": 0,
        },
        {
            "id": "fast_hang_detect_warm_spare_swap",
            "label": "2. Flight Recorder Fast Triage (120s) + Warm-Spare Node Swap + GDS Reload (30m Ckpt)",
            "hang_detection_sec": 120.0,
            "orchestrator_reinit_sec": 150.0,
            "state_restore_sec": 60.0,
            "checkpoint_interval_sec": 1800.0,
            "avg_rollback_loss_sec": 900.0,
            "warm_spare_nodes_required": 2,
        },
        {
            "id": "inplace_comm_shrink_elastic_async_ckpt",
            "label": "3. Custom App-Managed In-Place Shrink/Re-init (ncclCommShrink + Separate Grow/Reshard) + 5m Async DCP",
            "hang_detection_sec": 45.0,
            "orchestrator_reinit_sec": 12.0,
            "state_restore_sec": 8.0,
            "checkpoint_interval_sec": 300.0,
            "avg_rollback_loss_sec": 150.0,
            "warm_spare_nodes_required": 2,
            "api_semantics_boundary": (
                "Official NVIDIA ncclCommShrink only removes specified ranks among survivors; adding a warm-spare "
                "rank requires a separate grow/ncclCommInitRank step. Official PyTorch torchrun/TorchElastic stops "
                "and restarts all workers on failure/membership change; keeping 127 surviving CUDA contexts alive "
                "requires custom application-level communicator shrink/re-init + state resharding. The 45s/12s/8s "
                "phase durations are synthetic model input assumptions, not official vendor SLA guarantees."
            ),
        },
        {
            "id": "torchft_in_memory_replica_peer_recovery",
            "label": "4. In-Memory Peer State Streaming (TorchFT Lighthouse + Replica-Group RDMA) + Online Comm Re-init",
            "hang_detection_sec": 15.0,
            "orchestrator_reinit_sec": 5.0,
            "state_restore_sec": 8.0,
            "checkpoint_interval_sec": 300.0,
            "avg_rollback_loss_sec": 0.0,
            "warm_spare_nodes_required": 2,
            "api_semantics_boundary": (
                "Uses TorchFT Lighthouse + Manager replica-group coordination so surviving replica groups continue "
                "or stream live HBM weights/optimizer shards over RDMA without full-cluster stop/restart. The "
                "15s/5s/8s phase durations are synthetic model input assumptions, not official vendor SLA guarantees."
            ),
        },
    ]

    cycle_sec = cluster_mtbf_hours * 3600.0  # 21,600.0 seconds per MTBF cycle
    evaluated = []
    baseline_waste_usd = None
    baseline_lost_sec = None

    for arch in architectures:
        recovery_outage_sec = round(
            arch["hang_detection_sec"] + arch["orchestrator_reinit_sec"] + arch["state_restore_sec"], 1
        )
        total_lost_sec_per_fault = round(recovery_outage_sec + arch["avg_rollback_loss_sec"], 1)
        effective_goodput_pct = round(max(0.0, (cycle_sec - total_lost_sec_per_fault) / cycle_sec) * 100.0, 2)
        wasted_fraction = total_lost_sec_per_fault / cycle_sec
        monthly_wasted_gpu_spend_usd = round(cluster_monthly_budget_usd * wasted_fraction, 2)

        if baseline_waste_usd is None:
            baseline_waste_usd = monthly_wasted_gpu_spend_usd
            baseline_lost_sec = total_lost_sec_per_fault

        monthly_savings_vs_baseline_usd = round(baseline_waste_usd - monthly_wasted_gpu_spend_usd, 2)
        gross_recovered_compute_usd = monthly_savings_vs_baseline_usd
        warm_spare_monthly_cost_usd = round(
            arch["warm_spare_nodes_required"] * 8 * gpu_hourly_usd * hours_per_month, 2
        )
        net_monthly_savings_after_spare_usd = round(
            gross_recovered_compute_usd - warm_spare_monthly_cost_usd, 2
        )
        if warm_spare_monthly_cost_usd > 0:
            gross_benefit_cost_ratio_x = round(
                gross_recovered_compute_usd / warm_spare_monthly_cost_usd, 4
            )
            net_roi_ratio_x = round(
                (gross_recovered_compute_usd - warm_spare_monthly_cost_usd) / warm_spare_monthly_cost_usd, 4
            )
        else:
            gross_benefit_cost_ratio_x = None
            net_roi_ratio_x = None
        recovery_speedup_x = round(baseline_lost_sec / total_lost_sec_per_fault, 2)

        row = dict(arch)
        row.update({
            "recovery_outage_sec": recovery_outage_sec,
            "total_lost_sec_per_fault": total_lost_sec_per_fault,
            "total_lost_min_per_fault": round(total_lost_sec_per_fault / 60.0, 2),
            "effective_goodput_pct": effective_goodput_pct,
            "recovery_speedup_vs_baseline_x": recovery_speedup_x,
            "monthly_wasted_gpu_spend_usd": monthly_wasted_gpu_spend_usd,
            "monthly_savings_vs_baseline_usd": monthly_savings_vs_baseline_usd,
            "gross_recovered_compute_usd": gross_recovered_compute_usd,
            "warm_spare_monthly_cost_usd": warm_spare_monthly_cost_usd,
            "net_monthly_savings_after_spare_usd": net_monthly_savings_after_spare_usd,
            "gross_benefit_cost_ratio_x": gross_benefit_cost_ratio_x,
            "net_roi_ratio_x": net_roi_ratio_x,
            "other_operational_overhead_usd": "not_measured",
        })
        evaluated.append(row)

    # Scale sensitivity across 256, 512, 1024, 2048 GPUs (MTBF scales inversely with GPU count:
    # 256 GPUs -> 24.0h MTBF, 512 GPUs -> 12.0h MTBF, 1024 GPUs -> 6.0h MTBF, 2048 GPUs -> 3.0h MTBF)
    scale_sensitivity = []
    for gpus in [256, 512, 1024, 2048]:
        mtbf_h = round(6.0 * (1024.0 / gpus), 1)
        cyc_s = mtbf_h * 3600.0
        scale_sensitivity.append({
            "gpu_count": gpus,
            "cluster_mtbf_hours": mtbf_h,
            "traditional_goodput_pct": round(max(0.0, (cyc_s - 3420.0) / cyc_s) * 100.0, 2),
            "warm_spare_goodput_pct": round(max(0.0, (cyc_s - 1230.0) / cyc_s) * 100.0, 2),
            "inplace_comm_shrink_goodput_pct": round(max(0.0, (cyc_s - 215.0) / cyc_s) * 100.0, 2),
            "torchft_peer_rdma_goodput_pct": round(max(0.0, (cyc_s - 28.0) / cyc_s) * 100.0, 2),
        })

    return {
        "schema_version": 1,
        "experiment_id": "gpu_cluster_fault_recovery_selection",
        "provenance": {
            "classification": "synthetic_availability_model_and_anonymized_reported_priors",
            "independently_reproduced": False,
            "gpu_cluster_hardware_measured": "not_measured",
            "local_cpu_harness_measured": "measured",
            "notes": (
                "Analytical MTBF & fault-recovery goodput simulation for a 1,024xH100 training cluster "
                "(MTBF=6.0h, $2.50/GPU-hour). No physical 1,024-GPU cluster was spun up during this run."
            ),
        },
        "baseline_cluster": {
            "gpu_count": cluster_gpus,
            "nodes": 128,
            "gpus_per_node": 8,
            "gpu_hourly_cost_usd": gpu_hourly_usd,
            "cluster_hourly_cost_usd": cluster_hourly_usd,
            "cluster_monthly_budget_usd": cluster_monthly_budget_usd,
            "cluster_mtbf_hours": cluster_mtbf_hours,
        },
        "architectures": evaluated,
        "scale_sensitivity": scale_sensitivity,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify sealed results.json matches deterministic model")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    computed = build_results()

    if args.check:
        if not out_path.exists():
            raise SystemExit(f"Missing {out_path}")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != computed:
            raise SystemExit("Mismatch between results.json and computed deterministic model")
        print("OK: experiments/gpu_cluster_fault_recovery_selection/results.json matches deterministic model.")
        return 0

    out_path.write_text(json.dumps(computed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
