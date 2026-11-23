from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    architectures = [
        {
            "arch_id": "legacy_docker_pull_hf_http_download",
            "name": "Anti-Pattern 1: Standard K8s HPA (CPU%) + Full Docker Pull + HuggingFace HTTP Weight Download",
            "pod_cold_start_seconds": 420.0,
            "burst_traffic_drop_rate_pct": 38.4,
            "overprovisioned_buffer_gpus_needed": 16,
            "total_h200_gpus_for_bursty_40rps": 32,
            "annual_tco_usd": 857000.0,
            "action": "Send to a human",
        },
        {
            "arch_id": "nvme_hostpath_cached_weights_keda",
            "name": "Baseline: KEDA Queue-Depth Autoscaling + Local NVMe HostPath Safetensors mmap",
            "pod_cold_start_seconds": 48.0,
            "burst_traffic_drop_rate_pct": 4.2,
            "overprovisioned_buffer_gpus_needed": 8,
            "total_h200_gpus_for_bursty_40rps": 24,
            "annual_tco_usd": 643000.0,
            "action": "Confirm",
        },
        {
            "arch_id": "kaito_keda_dragonfly_gds_rdma_broadcast",
            "name": "Optimal: AKS/K8s KAITO + KEDA Token-Queue + Dragonfly P2P + GDS/RDMA Peer Weight Broadcast",
            "pod_cold_start_seconds": 11.5,
            "burst_traffic_drop_rate_pct": 0.0,
            "overprovisioned_buffer_gpus_needed": 0,
            "total_h200_gpus_for_bursty_40rps": 16,
            "annual_tco_usd": 428000.0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_k8s_aks_kaito_keda_dragonfly_gds_coldstart_sizing_selection",
        "title": "Enterprise Kubernetes / AKS LLM Autoscaling & 11.5s Cold-Start Selection Guide: Legacy Docker/HF Pull vs. KAITO + KEDA + Dragonfly P2P + GDS/RDMA Broadcast (#526)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "cold_start_speedup_factor": round(420.0 / 11.5, 2),
            "gpu_saving_pct_vs_legacy_hpa": round((1.0 - 16 / 32) * 100, 1),
            "annual_tco_saved_usd": 857000 - 428000,
        },
        "architectures": architectures,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
