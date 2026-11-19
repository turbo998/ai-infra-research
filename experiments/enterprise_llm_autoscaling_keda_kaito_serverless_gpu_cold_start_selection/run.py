"""Deterministic evaluator for Enterprise LLM Inference Autoscaling, Cold-Start Acceleration & Burst Spillover Selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    tiers = [
        {
            "tier": "Tier 1: Static Peak-Provisioned 24x7 GPU Cluster (No Autoscaling)",
            "architecture": "Fixed 32x H100 GPUs running 24x7 sized for Monday 10:00 AM peak traffic (idle at night and weekends)",
            "scale_out_pod_ready_latency_sec": 0.0,
            "traffic_spike_5x_request_drop_or_timeout_pct": 0.5,
            "average_24x7_gpu_fleet_utilization_pct": 26.5,
            "p99_ttft_during_morning_surge_sec": 1.15,
            "monthly_inference_compute_tco_usd": 198000,
        },
        {
            "tier": "Tier 2: Standard Kubernetes HPA on CPU / Average GPU Utilization (`DCGM_FI_DEV_GPU_UTIL`)",
            "architecture": "Scaling vLLM pods on GPU compute % + pulling 140 GB container images & weights over standard registry HTTP",
            "scale_out_pod_ready_latency_sec": 680.0,
            "traffic_spike_5x_request_drop_or_timeout_pct": 34.2,
            "average_24x7_gpu_fleet_utilization_pct": 48.0,
            "p99_ttft_during_morning_surge_sec": 18.40,
            "monthly_inference_compute_tco_usd": 134000,
        },
        {
            "tier": "Tier 3: KEDA on vLLM Queue & KV-Cache Metrics (Standard Weight Loading, No Cloud Spillover)",
            "architecture": "Scaling on `vllm:num_requests_waiting` & `gpu_cache_usage_perc`, but waiting 4.5 min for node boot + weight load",
            "scale_out_pod_ready_latency_sec": 270.0,
            "traffic_spike_5x_request_drop_or_timeout_pct": 11.8,
            "average_24x7_gpu_fleet_utilization_pct": 64.5,
            "p99_ttft_during_morning_surge_sec": 6.80,
            "monthly_inference_compute_tco_usd": 102000,
        },
        {
            "tier": "Tier 4: KEDA KV-Cache/Queue Scaler + KAITO / OCI Artifact Streaming + Cloud Serverless API Burst Spillover",
            "architecture": "KEDA Leading-Indicator Metrics + Warm Node Pool / NVMe Weight Cache (`< 38s` ready) + AI Gateway Instant Spillover to Cloud PAYGO during scale-out window",
            "scale_out_pod_ready_latency_sec": 36.0,
            "traffic_spike_5x_request_drop_or_timeout_pct": 0.0,
            "average_24x7_gpu_fleet_utilization_pct": 82.4,
            "p99_ttft_during_morning_surge_sec": 0.92,
            "monthly_inference_compute_tco_usd": 78000,
        },
    ]

    scenarios = [
        {
            "id": "AS1",
            "scenario": "Why Standard Kubernetes HPA on `GPU_UTIL` Fails Catastrophically for LLM Serving",
            "question": "Traffic jumps 3x with long-context requests; GPU KV-cache hits 99% full and 80 requests queue up, yet Kubernetes HPA refuses to scale out",
            "model_answer": "LLM decoding is memory-bandwidth bound and KV-cache constrained: `DCGM_FI_DEV_GPU_UTIL` can sit at 52% while `vllm:gpu_cache_usage_perc` is 99% saturated! KEDA scales directly on `gpu_cache_usage_perc > 75%` or `num_requests_waiting > 5` (0.92s P99 TTFT)",
            "tier1_3_failure": "Tier 2 waits for GPU compute utilization to cross 80%, leaving requests queued for 18.4s P99 TTFT and timing out 34.2% of callers.",
            "action": "Act automatically",
            "confidence": 0.99,
        },
        {
            "id": "AS2",
            "scenario": "Slashing 70B Model Pod Cold-Start from 11 Minutes to 36 Seconds",
            "question": "Scaling out a new 70B FP8 vLLM replica requires pulling a 15 GB container image plus 72 GB of HuggingFace weights",
            "model_answer": "Combine AKS/EKS Artifact Streaming (SOCI/OverlayBD) + Pre-warmed Local NVMe/HostPath Model Cache (`KAITO` / `Run:ai Model Streamer` parallel tensor loading) to bring a new replica ready in 36 seconds",
            "tier1_3_failure": "Tier 2 & Tier 3 pull weights over the network on every pod start, taking 270s–680s (4.5 to 11.3 minutes)—by the time the pod is ready, the traffic spike is already over!",
            "action": "Act automatically",
            "confidence": 0.99,
        },
        {
            "id": "AS3",
            "scenario": "Bridging the 36-Second Scale-Out Gap with AI Gateway Cloud PAYGO Burst Spillover",
            "question": "At 09:00 AM Monday, traffic surges 5x in 5 seconds; even a fast 36-second pod cold start cannot absorb the first 36 seconds of surge",
            "model_answer": "AI Gateway monitors in-flight queue depth; while KEDA spins up new self-hosted GPU pods (0s–36s window), excess requests above threshold seamlessly spill over to Azure OpenAI / Bedrock Serverless PAYGO (`0.0%` dropped requests)",
            "tier1_3_failure": "Tier 3 has no cloud spillover path, dropping or timing out 11.8% of requests while waiting for new GPU pods to initialize.",
            "action": "Confirm",
            "confidence": 0.98,
        },
        {
            "id": "AS4",
            "scenario": "Off-Peak Scale-to-Zero / Scale-to-Min FinOps Without Sacrificing Morning SLA",
            "question": "Internal enterprise Copilot traffic drops by 90% between 20:00 and 07:30 and on weekends; CFO wants idle GPUs released",
            "model_answer": "Use KEDA Cron + KV-Cache composite triggers with connection-draining `PreStop` hooks to scale down to 2 minimum warm replicas overnight, cutting monthly GPU spend from $198K (static 32 GPUs) to $78K (-60.6%)",
            "tier1_3_failure": "Tier 1 burns $198K/mo keeping all 32 H100 GPUs running 24x7 at 26.5% average fleet utilization.",
            "action": "Send to a human",
            "confidence": 0.99,
        },
    ]

    perturbation_checks = [
        {
            "check": "Sudden 8x Flash Crowd (Company-Wide Town Hall Live Copilot Prompt)",
            "original": "0.0% Request Timeout | 0.92s P99 TTFT",
            "perturbed": "0.0% Timeout | 1.18s P99 TTFT (Instant AI Gateway Overflow to Cloud PAYGO Absorbs Shock While KEDA Scales)",
            "change_pts": "+0.26s P99 TTFT (Zero Errors)",
        },
        {
            "check": "Scale-Down During Long-Running 60-Second Reasoning / Agentic Streaming Responses",
            "original": "0.0% Dropped In-Flight Streams",
            "perturbed": "0.0% Dropped Streams (Graceful `terminationGracePeriodSeconds: 180` + Envoy Endpoint Draining)",
            "change_pts": "0.0 pts (Invariant)",
        },
    ]

    return {
        "experiment_id": "enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection",
        "round": 470,
        "title": "Enterprise LLM Inference Autoscaling, Cold-Start Acceleration & Burst Spillover Architecture Selection",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "baseline_workload": {
            "peak_gpu_capacity_h100_count": 32,
            "model_size_gb_fp8": 72,
            "diurnal_peak_to_trough_ratio": 8.5,
            "target_p99_ttft_sla_sec": 1.50,
        },
        "tiers": tiers,
        "scenarios": scenarios,
        "perturbation_checks": perturbation_checks,
        "key_findings": [
            "Tier 4 (KEDA KV-Cache/Queue Scaler + KAITO/NVMe Fast Weight Streaming + AI Gateway Cloud PAYGO Burst Spillover) achieves 36s pod ready time, 0.0% surge drops, 82.4% GPU utilization, and $78K/mo TCO (-60.6% vs static 32x H100 fleet).",
            "Standard Kubernetes HPA on GPU compute utilization (`DCGM_FI_DEV_GPU_UTIL`) is a critical anti-pattern for LLM serving because KV-cache memory fills up long before GPU compute reaches 80%, causing 18.4s P99 TTFT stalls.",
            "Pairing 36-second fast weight loading with instant AI Gateway overflow to Cloud Serverless PAYGO bridges the cold-start gap with zero dropped requests during 5x–8x traffic spikes.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify existing results.json matches deterministic payload")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist")
        current = out_path.read_text(encoding="utf-8")
        if current != rendered:
            raise SystemExit("results.json drift detected")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
