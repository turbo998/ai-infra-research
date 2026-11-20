from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_ai_model_storage_p2p_distribution_nvme_cache_cold_start_selection",
        "title": "Enterprise AI Model Storage, P2P Weight Distribution & 32-Node GPU Cold-Start Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "model_weights_size_gb": 140,
            "concurrent_scaleout_gpu_nodes": 32,
            "active_model_versions_in_registry": 25,
            "daily_autoscaling_events": 12,
        },
        "tiers": [
            {
                "tier_id": "tier1_direct_s3_blob_http_pull",
                "name": "Tier 1: Direct S3/Blob HTTP Pull per Pod (No Local Cache, No P2P)",
                "thirty_two_node_cold_start_sec": 1110.0,
                "per_node_effective_throughput_gbps": 0.13,
                "object_store_egress_stampede_tb": 4.48,
                "monthly_storage_and_idle_gpu_buffer_tco_usd": 98000,
                "verdict": "Anti-Pattern: 18.5-minute cold start forces 16 idle H100 GPUs 24/7 just to survive traffic spikes",
            },
            {
                "tier_id": "tier2_standard_cloud_nas_nfs_efs",
                "name": "Tier 2: Shared Standard Cloud NAS (EFS / Azure Files Standard NFS)",
                "thirty_two_node_cold_start_sec": 380.0,
                "per_node_effective_throughput_gbps": 0.37,
                "object_store_egress_stampede_tb": 0.0,
                "monthly_storage_and_idle_gpu_buffer_tco_usd": 72000,
                "verdict": "Shared NAS bandwidth ceiling (~12 GB/s total) divides by 32 nodes -> 6.3-minute cold start",
            },
            {
                "tier_id": "tier3_dedicated_always_on_managed_lustre",
                "name": "Tier 3: Dedicated Always-On 100TB Managed Parallel FS (FSx Lustre / ANF Ultra)",
                "thirty_two_node_cold_start_sec": 28.0,
                "per_node_effective_throughput_gbps": 5.0,
                "object_store_egress_stampede_tb": 0.0,
                "monthly_storage_and_idle_gpu_buffer_tco_usd": 64000,
                "verdict": "Fast 28s load, but overpaying $64K/mo 24/7 for training-grade Lustre just to read static inference weights",
            },
            {
                "tier_id": "tier4_s3_dragonfly_p2p_local_nvme_safetensors_streamer",
                "name": "Tier 4: S3/Blob + Dragonfly P2P Ring + Free Node-Local NVMe RAID0 + Run:ai/Safetensors Streamer",
                "thirty_two_node_cold_start_sec": 13.5,
                "per_node_effective_throughput_gbps": 10.4,
                "object_store_egress_stampede_tb": 0.14,
                "monthly_storage_and_idle_gpu_buffer_tco_usd": 13500,
                "verdict": "Recommended: 13.5s cold start (82x faster), $0 extra NVMe hardware cost & -86.2% monthly TCO",
            },
        ],
        "headline_findings": {
            "cold_start_speedup_factor": 82.2,
            "thirty_two_node_cold_start_sec": 13.5,
            "s3_egress_reduction_pct": 96.9,
            "tco_savings_vs_tier1_pct": 86.2,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
