from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "k8s_llm_autoscaling_keda_queue_metrics_weight_streaming_cold_start_selection",
        "title": "K8s LLM Autoscaling & Cold-Start Optimization: KEDA Queue/KV Metrics, Run:ai Weight Streamer & GDS/Sleep-Wake Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "model": "Llama-3.3-70B-FP8 (70 GB) / BF16 (140 GB)",
            "k8s_cluster": "AKS / GKE GPU Node Pool (8x H200 Nodes)",
            "traffic_spike_multiplier": 3.5,
        },
        "tiers": [
            {
                "tier_id": "tier1_k8s_hpa_on_gpu_util_plus_monolithic_oci_image_pull",
                "name": "Tier 1: K8s HPA on `DCGM_FI_DEV_GPU_UTIL` + Monolithic 140GB Image Pull & Sequential Disk Read",
                "replica_scale_out_ready_sec": 640.0,
                "burst_p99_ttft_ms": 18500.0,
                "false_positive_or_late_scaling_rate_pct": 72.0,
                "monthly_cluster_elastic_tco_usd": 245000,
                "verdict": "Anti-Pattern: `GPU_UTIL` is 95% even at B=2; 640s (10.7m) cold start misses the entire traffic burst!",
            },
            {
                "tier_id": "tier2_keda_on_waiting_queue_and_kv_usage_with_nfs_weight_load",
                "name": "Tier 2: KEDA on `vllm:num_requests_waiting` & `gpu_cache_usage_perc` + Standard NFS/PVC `.safetensors` Load",
                "replica_scale_out_ready_sec": 165.0,
                "burst_p99_ttft_ms": 4200.0,
                "false_positive_or_late_scaling_rate_pct": 8.5,
                "monthly_cluster_elastic_tco_usd": 158000,
                "verdict": "Accurate scaling triggers, but single-thread Python `safetensors` read over shared NFS still takes 165s",
            },
            {
                "tier_id": "tier3_keda_plus_runai_model_streamer_s3_gcs_concurrent_dma",
                "name": "Tier 3: KEDA Queue/KV Triggers + Run:ai Model Streamer (32-Thread S3/Blob Byte-Range -> Pinned DMA -> HBM)",
                "replica_scale_out_ready_sec": 24.0,
                "burst_p99_ttft_ms": 920.0,
                "false_positive_or_late_scaling_rate_pct": 2.1,
                "monthly_cluster_elastic_tco_usd": 112000,
                "verdict": "Streams 70GB FP8 weights from object storage / local NVMe in 11s (24s total pod ready)!",
            },
            {
                "tier_id": "tier4_keda_predictive_kv_trigger_plus_nvme_gds_and_vllm_sleep_level1_warm_pool",
                "name": "Tier 4: KEDA KV/Queue Trigger + Local NVMe GDS + vLLM Sleep-Mode (`level=1` Host-DRAM Weight Wake in 4.2s)",
                "replica_scale_out_ready_sec": 12.5,
                "burst_p99_ttft_ms": 540.0,
                "false_positive_or_late_scaling_rate_pct": 0.4,
                "monthly_cluster_elastic_tco_usd": 88000,
                "verdict": "Recommended: 12.5s cold/warm scale-out (-51.2x vs. Tier 1), 540ms burst TTFT & -64.1% monthly TCO",
            },
        ],
        "headline_findings": {
            "scale_out_ready_sec": 12.5,
            "cold_start_speedup_vs_tier1_x": 51.2,
            "burst_p99_ttft_ms": 540.0,
            "tco_savings_vs_tier1_pct": 64.1,
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
