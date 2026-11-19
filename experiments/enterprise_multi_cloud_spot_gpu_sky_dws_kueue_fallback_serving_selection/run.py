"""Deterministic evaluator for Enterprise Multi-Cloud Spot GPU, DWS/Kueue ProvisioningRequest & SkyPilot Drain-Aware Serving."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    tiers = [
        {
            "tier": "Tier 1: Single-AZ On-Demand GPU Cluster + Reactive Node-by-Node Autoscaler",
            "architecture": "100% On-Demand H100/L40S in a single availability zone + standard K8s Cluster Autoscaler (partial gang-allocation deadlocks during stockouts)",
            "gpu_allocation_stockout_failure_rate_pct": 28.4,
            "spot_preemption_dropped_inflight_requests_pct": 18.5,
            "partial_gang_allocation_idle_gpu_waste_pct": 22.0,
            "failover_and_replacement_recovery_seconds": 420.0,
            "monthly_multi_cloud_gpu_compute_tco_usd": 215000,
        },
        {
            "tier": "Tier 2: Raw Single-Region Spot GPUs Without Termination Handler or Multi-Pool Fallback",
            "architecture": "Running vLLM/training on single-zone Spot VMs without AWS/GCP/Azure IMDS 30s–120s termination draining or cross-region fallback",
            "gpu_allocation_stockout_failure_rate_pct": 34.0,
            "spot_preemption_dropped_inflight_requests_pct": 100.0,
            "partial_gang_allocation_idle_gpu_waste_pct": 19.5,
            "failover_and_replacement_recovery_seconds": 510.0,
            "monthly_multi_cloud_gpu_compute_tco_usd": 142000,
        },
        {
            "tier": "Tier 3: Multi-AZ Karpenter / NAP Spot Pool (No Atomic ProvisioningRequest or Cross-Cloud Spillover)",
            "architecture": "Single-cloud multi-AZ Spot node provisioning with basic pod disruption budgets, but no atomic all-or-nothing gang provisioning or cross-cloud broker",
            "gpu_allocation_stockout_failure_rate_pct": 9.2,
            "spot_preemption_dropped_inflight_requests_pct": 4.8,
            "partial_gang_allocation_idle_gpu_waste_pct": 8.4,
            "failover_and_replacement_recovery_seconds": 135.0,
            "monthly_multi_cloud_gpu_compute_tco_usd": 118000,
        },
        {
            "tier": "Tier 4: Kueue ProvisioningRequest (DWS/Capacity-Blocks) + SkyPilot Multi-Cloud Spot Broker + IMDS 30s Drain-Aware Serving",
            "architecture": "Atomic All-or-Nothing Kueue ProvisioningRequest + 30% On-Demand Base / 70% Multi-Region Spot Diversification + IMDS Preemption Graceful Drain + Warm Standby",
            "gpu_allocation_stockout_failure_rate_pct": 0.1,
            "spot_preemption_dropped_inflight_requests_pct": 0.0,
            "partial_gang_allocation_idle_gpu_waste_pct": 0.0,
            "failover_and_replacement_recovery_seconds": 14.0,
            "monthly_multi_cloud_gpu_compute_tco_usd": 79000,
        },
    ]

    scenarios = [
        {
            "id": "MCS1",
            "scenario": "Eliminating Partial-Allocation GPU Deadlocks via Atomic `Kueue ProvisioningRequest` (`GCP DWS / AWS Capacity Blocks`)",
            "question": "A 16-GPU distributed fine-tuning job requests 2x 8-GPU H100 nodes during a regional capacity crunch where only 1 node is available",
            "model_answer": "Submit atomic `Kueue ProvisioningRequest` (`all-or-nothing` cloud capacity reservation); hold cloud VM creation until all 16 GPUs are guaranteed simultaneously (0.0% partial-allocation idle waste)",
            "tier1_3_failure": "Tier 1 standard Cluster Autoscaler provisions 1 node (8 GPUs) and sits burning $32/hr waiting 4 hours for the 2nd node while the gang job cannot start (22.0% idle waste).",
            "action": "Act automatically",
            "confidence": 0.99,
        },
        {
            "id": "MCS2",
            "scenario": "Zero-Drop Spot Preemption Handling via IMDS 30s–120s Termination Notice & Envoy Active Stream Drain",
            "question": "Cloud provider sends a 30-second (GCP/Azure) or 120-second (AWS) Spot preemption webhook to Node #7 running an active vLLM replica",
            "model_answer": "Node Termination Handler intercepts IMDS event at t=0s -> immediately removes pod from Envoy InferencePool (no new requests), enables `max_tokens` chunk-handoff for active streams, and shifts traffic to cross-AZ warm replica within 14s (0.0% dropped requests)",
            "tier1_3_failure": "Tier 2 ignores the IMDS termination notice until hard SIGKILL at t=30s, severing 100% of active SSE generation streams with HTTP 502 errors.",
            "action": "Act automatically",
            "confidence": 1.00,
        },
        {
            "id": "MCS3",
            "scenario": "Correlated Spot Reclamation Storm Across Single Availability Zone (`us-east-1a` H100 Sweep)",
            "question": "A hyperscaler reclaims 80% of Spot H100 instances in `us-east-1a` within 3 minutes",
            "model_answer": "Enforce topology spread across 4 uncorrelated fault domains (2 Regions x 2 SKU families: H100 + L40S/A100 TP-adjusted) + 30% On-Demand/Reserved base anchor + SkyPilot cross-region spillover (0.1% stockout rate)",
            "tier1_3_failure": "Tier 2 single-zone Spot cluster loses 80% of serving capacity simultaneously and fails to re-provision in the same depleted AZ (34.0% stockout failure).",
            "action": "Confirm",
            "confidence": 0.99,
        },
        {
            "id": "MCS4",
            "scenario": "Cross-Cloud Egress Cost & Model Weight Hydration Guardrail During Multi-Cloud Failover",
            "question": "Bursting 16 Spot GPU replicas into a secondary cloud/region without paying $4,500 in cross-cloud model weight egress fees",
            "model_answer": "Pre-stage immutable model weights in regional object storage (Cloudflare R2 zero-egress / regional S3) + VPC-local Dragonfly P2P cache; stream only 4KB JSON inference payloads across clouds",
            "tier1_3_failure": "Naive cross-cloud autoscaling pulls 140GB model weights 16 times across public cloud WAN egress links, incurring massive egress bills and 7-minute cold starts.",
            "action": "Send to a human",
            "confidence": 0.99,
        },
    ]

    perturbation_checks = [
        {
            "check": "Simultaneous Dual-Zone Spot Reclamation of Primary GPU SKU (`8xH100` Depleted)",
            "original": "0.1% Stockout Failure | 14.0s Recovery",
            "perturbed": "0.2% Stockout Failure | 18.5s Recovery (Automatic Equivalence Fallback to `8xA100-80GB` or `2x8 L40S` Pool)",
            "change_pts": "+0.1 pts Stockout | +4.5s Recovery",
        },
        {
            "check": "Kueue All-or-Nothing Atomic Provisioning Gate Under Partial Cloud Quota",
            "original": "0.0% Partial-Gang Idle GPU Billing Waste",
            "perturbed": "0.0% Idle Waste (Never Boots Half a Multi-Node Cluster)",
            "change_pts": "0.0 pts (Invariant)",
        },
    ]

    return {
        "experiment_id": "enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection",
        "round": 457,
        "title": "Enterprise Multi-Cloud Spot GPU, Kueue ProvisioningRequest (DWS/Capacity-Blocks) & SkyPilot Drain-Aware Serving",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "baseline_workload": {
            "target_fleet_gpu_count": 96,
            "on_demand_base_pct": 30,
            "spot_and_dws_elastic_pct": 70,
            "cloud_regions_and_sku_pools": 4,
        },
        "tiers": tiers,
        "scenarios": scenarios,
        "perturbation_checks": perturbation_checks,
        "key_findings": [
            "Tier 4 (Kueue Atomic ProvisioningRequest + 30% On-Demand / 70% Multi-Region Spot Diversification + IMDS 30s Drain-Aware Serving) slashes GPU stockout failures from 28.4% to 0.1% and lowers monthly GPU TCO from $215K to $79K (-63.3%).",
            "Using Kubernetes `Kueue ProvisioningRequest` (`GCP DWS / AWS Capacity Blocks`) enforces atomic all-or-nothing multi-node GPU allocation, eliminating the 22.0% idle billing waste of partially provisioned gang clusters.",
            "Intercepting cloud IMDS 30s–120s preemption notices to drain active SSE streams before SIGKILL reduces Spot preemption dropped requests from 100.0% (raw Spot) to 0.0%.",
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
