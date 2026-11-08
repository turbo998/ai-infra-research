"""Deterministic analytical model for AI Cluster Storage, Two-Phase Async Checkpointing,
and Tiered KV Cache Offloading (HBM -> Pinned Host DRAM -> NVMe/RDMA GDS).

Provenance note:
- Hardware bandwidth baselines (PCIe Gen5 x16, 400G RoCEv2 / ConnectX-7, GPUDirect Storage,
  3FS / WEKA / Managed Lustre) are `anonymized_reported` (`independently_reproduced=false`).
- Checkpoint stall, cluster goodput, TCO savings, and KV reload TTFT equations are
  `synthetic` deterministic analytical models (`paid_cloud_gpu_launched=false`,
  `live_cluster_measurement="not_measured"`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS_PATH = Path(__file__).with_name("results.json")


def build_results() -> dict:
    # Scenario A: 512-GPU (64 nodes x 8 GPUs) training cluster saving 9,000 GB checkpoint every 1,800 s
    ckpt_payload_gb = 9000.0
    ckpt_interval_s = 1800.0
    cluster_gpus = 512
    cluster_nodes = 64
    gpu_hourly_usd = 4.00
    monthly_hours = 720.0

    checkpoint_modes = [
        {
            "mode": "sync_posix_lustre",
            "label": "Synchronous POSIX Checkpoint over Standard Parallel FS (40 GB/s agg)",
            "aggregate_write_gb_s": 40.0,
            "gpu_stall_s": round(ckpt_payload_gb / 40.0, 3),  # 225.0 s
            "background_flush_s": 0.0,
            "effective_goodput_pct": round((1.0 - (ckpt_payload_gb / 40.0) / ckpt_interval_s) * 100.0, 2),  # 87.50%
            "stall_speedup_vs_posix": 1.0,
            "provenance": "synthetic",
        },
        {
            "mode": "sync_gds_rdma_fs",
            "label": "Synchronous GPUDirect Storage (GDS) over RDMA All-Flash 3FS/WEKA (360 GB/s agg)",
            "aggregate_write_gb_s": 360.0,
            "gpu_stall_s": round(ckpt_payload_gb / 360.0, 3),  # 25.0 s
            "background_flush_s": 0.0,
            "effective_goodput_pct": round((1.0 - (ckpt_payload_gb / 360.0) / ckpt_interval_s) * 100.0, 2),  # 98.61%
            "stall_speedup_vs_posix": round(225.0 / (ckpt_payload_gb / 360.0), 2),  # 9.0x
            "provenance": "synthetic",
        },
        {
            "mode": "two_phase_async_dram_nvme",
            "label": "Two-Phase Async Checkpoint (Phase 1 HBM->Pinned DRAM @ 2,880 GB/s + Phase 2 Background Flush)",
            "aggregate_write_gb_s": 2880.0,
            "per_node_d2h_gb_s": 45.0,
            "per_node_payload_gb": round(ckpt_payload_gb / cluster_nodes, 4),  # 140.625 GB
            "gpu_stall_s": round(ckpt_payload_gb / 2880.0, 3),  # 3.125 s
            "background_flush_s": round(ckpt_payload_gb / 180.0, 1),  # 50.0 s async (0 s GPU stall)
            "effective_goodput_pct": round((1.0 - (ckpt_payload_gb / 2880.0) / ckpt_interval_s) * 100.0, 2),  # 99.83%
            "stall_speedup_vs_posix": round(225.0 / (ckpt_payload_gb / 2880.0), 2),  # 72.0x
            "provenance": "synthetic",
        },
    ]

    posix_goodput = checkpoint_modes[0]["effective_goodput_pct"] / 100.0
    async_goodput = checkpoint_modes[2]["effective_goodput_pct"] / 100.0
    goodput_gain_pct_pts = round((async_goodput - posix_goodput) * 100.0, 2)  # 12.33 pts
    monthly_cluster_cost_usd = round(cluster_gpus * gpu_hourly_usd * monthly_hours, 2)  # $1,474,560
    monthly_recovered_compute_usd = round(monthly_cluster_cost_usd * (async_goodput - posix_goodput), 2)  # $181,813.25

    # Scenario B: 65,536-token Multi-Turn Agent KV Cache Tiering (2.048 GB FP8 KV per session)
    kv_context_tokens = 65536
    kv_bytes_per_token = 32000  # 32 KB/token decimal -> 65,536 * 32,000 = 2,097,152,000 B, or 64K * 32 KB = 2.048 GB
    kv_payload_gb = 2.048
    kv_payload_gb_bits = 16.384
    delta_prefill_ms = 12.0
    recompute_ttft_ms = 780.0

    kv_tiering_modes = [
        {
            "tier": "hbm_evict_recompute",
            "label": "No Offload (Evict & Recompute 64K Prefill on Resume)",
            "effective_reload_gb_s": 0.0,
            "kv_reload_ms": 0.0,
            "resume_ttft_ms": recompute_ttft_ms,
            "ttft_speedup_vs_recompute": 1.0,
            "cost_per_1m_tokens_usd": 0.96,
            "cost_reduction_pct": 0.0,
            "provenance": "synthetic",
        },
        {
            "tier": "cloud_posix_tcp_nofs",
            "label": "Traditional POSIX/TCP Disk Offload (No GDS, CPU Bounce Buffer @ 3.2 GB/s)",
            "effective_reload_gb_s": 3.2,
            "kv_reload_ms": round((kv_payload_gb / 3.2) * 1000.0, 1),  # 640.0 ms
            "resume_ttft_ms": round((kv_payload_gb / 3.2) * 1000.0 + delta_prefill_ms, 1),  # 652.0 ms
            "ttft_speedup_vs_recompute": round(recompute_ttft_ms / 652.0, 2),  # 1.20x
            "cost_per_1m_tokens_usd": 0.84,
            "cost_reduction_pct": 12.5,
            "provenance": "synthetic",
        },
        {
            "tier": "shared_rdma_nvme_gds",
            "label": "Shared RDMA / NVMe-oF Pool (Mooncake Store / 3FS + GDS @ 40.96 GB/s)",
            "effective_reload_gb_s": 40.96,
            "kv_reload_ms": round((kv_payload_gb / 40.96) * 1000.0, 1),  # 50.0 ms
            "resume_ttft_ms": round((kv_payload_gb / 40.96) * 1000.0 + delta_prefill_ms, 1),  # 62.0 ms
            "ttft_speedup_vs_recompute": round(recompute_ttft_ms / 62.0, 2),  # 12.58x
            "cost_per_1m_tokens_usd": 0.48,
            "cost_reduction_pct": 50.0,
            "provenance": "synthetic",
        },
        {
            "tier": "host_dram_pinned_pcie5",
            "label": "Local Pinned Host DRAM Tier (LMCache / Dynamo KVBM over PCIe Gen5 @ 51.2 GB/s)",
            "effective_reload_gb_s": 51.2,
            "kv_reload_ms": round((kv_payload_gb / 51.2) * 1000.0, 1),  # 40.0 ms
            "resume_ttft_ms": round((kv_payload_gb / 51.2) * 1000.0 + delta_prefill_ms, 1),  # 52.0 ms
            "ttft_speedup_vs_recompute": round(recompute_ttft_ms / 52.0, 2),  # 15.0x
            "cost_per_1m_tokens_usd": 0.44,
            "cost_reduction_pct": 54.2,
            "provenance": "synthetic",
        },
    ]

    return {
        "experiment_id": "ai_storage_checkpoint_kv_offload",
        "version": "1.0.0",
        "provenance": "synthetic_analytical_model_calibrated_to_anonymized_reported_specs",
        "provenance_details": {
            "benchmark_harness_standard": "7-card-4x5-v1",
            "hardware_spec_reference": "anonymized_reported",
            "analytical_simulation": "synthetic",
            "local_harness_execution": "measured_local_python_unittest_and_check_cli_only",
            "independently_reproduced": False,
            "paid_cloud_gpu_launched": False,
            "live_512gpu_cluster_measurement": "not_measured",
        },
        "checkpoint_scenario": {
            "cluster_gpus": cluster_gpus,
            "cluster_nodes": cluster_nodes,
            "checkpoint_payload_gb": ckpt_payload_gb,
            "checkpoint_interval_s": ckpt_interval_s,
            "gpu_hourly_usd": gpu_hourly_usd,
            "monthly_cluster_cost_usd": monthly_cluster_cost_usd,
            "goodput_gain_pct_pts": goodput_gain_pct_pts,
            "monthly_recovered_compute_usd": monthly_recovered_compute_usd,
            "modes": checkpoint_modes,
        },
        "kv_tiering_scenario": {
            "context_tokens": kv_context_tokens,
            "kv_payload_gb": kv_payload_gb,
            "kv_payload_gb_bits": kv_payload_gb_bits,
            "delta_prefill_ms": delta_prefill_ms,
            "recompute_ttft_ms": recompute_ttft_ms,
            "tiers": kv_tiering_modes,
        },
        "summary_metrics": {
            "async_ckpt_stall_reduction_x": 72.0,
            "async_ckpt_stall_s": 3.125,
            "posix_ckpt_stall_s": 225.0,
            "async_goodput_pct": 99.83,
            "posix_goodput_pct": 87.50,
            "dram_kv_ttft_speedup_x": 15.0,
            "rdma_kv_ttft_speedup_x": 12.58,
            "dram_kv_cost_reduction_pct": 54.2,
            "rdma_kv_cost_reduction_pct": 50.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or verify AI Storage, Checkpoint & KV Offload analytical model.")
    parser.add_argument("--check", action="store_true", help="Verify sealed results.json matches deterministic model.")
    args = parser.parse_args()

    computed = build_results()
    if args.check:
        if not RESULTS_PATH.exists():
            print(f"ERROR: {RESULTS_PATH} missing", file=sys.stderr)
            return 1
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if existing != computed:
            print("ERROR: results.json out of sync with run.py", file=sys.stderr)
            return 1
        print("OK: ai_storage_checkpoint_kv_offload results.json verified.")
        return 0

    RESULTS_PATH.write_text(json.dumps(computed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
