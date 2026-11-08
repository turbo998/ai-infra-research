#!/usr/bin/env python3
"""Deterministic analytical evaluator for Multi-Tenant GPU Fractional Sharing & Hardware Isolation Selection.

Compares 4 GPU sharing and multi-tenant isolation architectures on NVIDIA H100 SXM 80GB:
  1. tier1_os_k8s_time_slicing (OS / Kubernetes Device-Plugin Round-Robin Time-Slicing)
  2. tier2_cuda_mps_spatial_sharing (CUDA Multi-Process Service with shared CUDA context & fault domain)
  3. tier3_nvidia_hardware_mig (NVIDIA Hardware Multi-Instance GPU: 7x 1g.10gb / mixed slices with silicon isolation)
  4. tier4_green_ctx_and_multi_lora (CUDA 12.4+ Green Contexts Intra-Process SM Partitioning + SGLang/vLLM Multi-LoRA)

Provenance classification:
  - Hardware & official mechanism specifications (H100 132 SMs, 7 MIG instances, MPS shared fatal fault domain,
    CUDA 12.4+ cudaGreenCtxCreate SM resource partitioning): `anonymized_reported` (`independently_reproduced=false`).
  - Latency, SM utilization, noisy-neighbor tail inflation, and monthly TCO consolidation curves: `synthetic`
    deterministic queuing & resource partitioning model (`independently_reproduced=false`).
  - Live multi-tenant GPU cluster fault-injection measurement: `"not_measured"`.
  - Local CPU deterministic harness & unit test verification: `"measured"`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def compute_model() -> Dict[str, Any]:
    gpu_hourly_usd = 2.50
    hours_per_month = 730.0
    single_gpu_monthly_usd = round(gpu_hourly_usd * hours_per_month, 2)  # $1,825.00/mo

    # Scenario A: 7 small-model / microservice tenants (e.g., 3B-8B FP8 / Embedding / Reranker / OCR),
    # each needing ~9.5 GB VRAM and 15.0% average SM duty cycle.
    tenants_per_gpu = 7
    dedicated_gpus_for_7_tenants = 7
    dedicated_monthly_cost_usd = round(dedicated_gpus_for_7_tenants * single_gpu_monthly_usd, 2)  # $12,775.00/mo
    isolated_baseline_p99_ms = 24.0

    consolidated_gpus = 1
    consolidated_monthly_cost_usd = single_gpu_monthly_usd  # $1,825.00/mo
    monthly_tco_savings_7_to_1_usd = round(dedicated_monthly_cost_usd - consolidated_monthly_cost_usd, 2)  # $10,950.00/mo
    tco_reduction_7_to_1_pct = round(
        (monthly_tco_savings_7_to_1_usd / dedicated_monthly_cost_usd) * 100.0, 1
    )  # 85.7%

    # Scenario B: 28 department LoRA adapters (Rank-16, 160 MB each = 4.48 GB total) on shared 70B FP8 base model (70 GB)
    lora_tenant_count = 28
    dedicated_70b_gpus = 28  # 1 H100 80GB per dedicated 70B FP8 deployment
    dedicated_lora_monthly_usd = round(dedicated_70b_gpus * single_gpu_monthly_usd, 2)  # $51,100.00/mo
    multi_lora_shared_gpus = 2  # 2x H100 80GB for HA / high-QPS shared Punica/S-LoRA pool
    multi_lora_monthly_usd = round(multi_lora_shared_gpus * single_gpu_monthly_usd, 2)  # $3,650.00/mo
    multi_lora_monthly_savings_usd = round(dedicated_lora_monthly_usd - multi_lora_monthly_usd, 2)  # $47,450.00/mo
    multi_lora_gpu_reduction_x = round(dedicated_70b_gpus / multi_lora_shared_gpus, 1)  # 14.0x

    tiers = {
        "dedicated_full_gpu_baseline": {
            "label": "Baseline: 1 Dedicated Full GPU per Tenant (7 × H100 80GB)",
            "gpus_required_for_7_microservices": dedicated_gpus_for_7_tenants,
            "avg_sm_utilization_pct": 15.0,
            "aggregate_throughput_req_per_s": 1680.0,
            "p50_latency_ms": 18.5,
            "p99_latency_ms": isolated_baseline_p99_ms,
            "noisy_neighbor_p99_inflation_pct": 0.0,
            "fault_blast_radius_pct": 0.0,
            "hbm_oom_isolation": "physical_separate_gpu",
            "monthly_cost_7_tenants_usd": dedicated_monthly_cost_usd,
            "monthly_savings_vs_dedicated_usd": 0.0,
        },
        "tier1_os_k8s_time_slicing": {
            "label": "Tier 1: OS / Kubernetes Device-Plugin Time-Slicing (7 Pods on 1 × H100)",
            "gpus_required_for_7_microservices": consolidated_gpus,
            "context_switch_overhead_us": 220.0,
            "avg_sm_utilization_pct": 52.0,
            "aggregate_throughput_req_per_s": 1180.0,
            "p50_latency_ms": 31.5,
            "p99_latency_ms": 82.0,
            "noisy_neighbor_p99_inflation_pct": round(((82.0 - isolated_baseline_p99_ms) / isolated_baseline_p99_ms) * 100.0, 1),  # 241.7%
            "p99_degradation_vs_dedicated_x": round(82.0 / isolated_baseline_p99_ms, 2),  # 3.42x
            "fault_blast_radius_pct": 100.0,
            "hbm_oom_isolation": "none_shared_global_hbm_oom_kills_co_tenants",
            "monthly_cost_7_tenants_usd": consolidated_monthly_cost_usd,
            "monthly_savings_vs_dedicated_usd": monthly_tco_savings_7_to_1_usd,
            "api_semantics_note": "Time-slicing multiplexes the entire GPU via temporal context switching without SM or HBM memory protection.",
        },
        "tier2_cuda_mps_spatial_sharing": {
            "label": "Tier 2: CUDA MPS (Multi-Process Service — Concurrent SM Sharing on 1 × H100)",
            "gpus_required_for_7_microservices": consolidated_gpus,
            "context_switch_overhead_us": 0.0,
            "avg_sm_utilization_pct": 78.0,
            "aggregate_throughput_req_per_s": 1840.0,
            "throughput_gain_vs_time_slicing_x": round(1840.0 / 1180.0, 2),  # 1.56x
            "p50_latency_ms": 25.2,
            "p99_latency_ms": 34.5,
            "noisy_neighbor_p99_inflation_pct": round(((34.5 - isolated_baseline_p99_ms) / isolated_baseline_p99_ms) * 100.0, 1),  # 43.8%
            "p99_improvement_vs_time_slicing_x": round(82.0 / 34.5, 2),  # 2.38x
            "fault_blast_radius_pct": 100.0,
            "hbm_oom_isolation": "soft_per_client_limit_but_shared_fatal_xid_domain",
            "monthly_cost_7_tenants_usd": consolidated_monthly_cost_usd,
            "monthly_savings_vs_dedicated_usd": monthly_tco_savings_7_to_1_usd,
            "api_semantics_note": "Per official NVIDIA MPS docs, all clients share one GPU virtual address space and fatal error domain: an unrecoverable GPU fault (Xid 31 / illegal address) in one client terminates all active MPS clients.",
        },
        "tier3_nvidia_hardware_mig": {
            "label": "Tier 3: NVIDIA Hardware MIG (7 × 1g.10gb / Mixed Silicon Partitions on 1 × H100)",
            "gpus_required_for_7_microservices": consolidated_gpus,
            "context_switch_overhead_us": 0.0,
            "avg_sm_utilization_pct": 68.5,
            "aggregate_throughput_req_per_s": 1620.0,
            "throughput_vs_mps_pct": round(((1620.0 - 1840.0) / 1840.0) * 100.0, 1),  # -12.0%
            "p50_latency_ms": 24.5,
            "p99_latency_ms": 25.0,
            "noisy_neighbor_p99_inflation_pct": round(((25.0 - isolated_baseline_p99_ms) / isolated_baseline_p99_ms) * 100.0, 1),  # 4.2%
            "p99_improvement_vs_time_slicing_x": round(82.0 / 25.0, 2),  # 3.28x
            "fault_blast_radius_pct": 0.0,
            "hbm_oom_isolation": "hardware_dedicated_hbm_controllers_l2_crossbar_and_page_tables",
            "monthly_cost_7_tenants_usd": consolidated_monthly_cost_usd,
            "monthly_savings_vs_dedicated_usd": monthly_tco_savings_7_to_1_usd,
            "tco_reduction_pct": tco_reduction_7_to_1_pct,
            "api_semantics_note": "Hardware MIG physically partitions SMs, HBM memory controllers, L2 cache banks, and page tables (0% cross-instance fault propagation), but slices are static and cannot borrow idle SMs across MIG boundaries.",
        },
        "tier4_green_ctx_and_multi_lora": {
            "label": "Tier 4: CUDA 12.4+ Green Contexts (Intra-Process SM Split) + Punica/S-LoRA Multi-LoRA Serving",
            "gpus_required_for_7_microservices": consolidated_gpus,
            "green_ctx_sm_split": "108 SMs Prefill/Compute + 24 SMs Decode/Comm (132 SMs total on H100)",
            "avg_sm_utilization_pct": 86.5,
            "aggregate_throughput_req_per_s": 2010.0,
            "p50_latency_ms": 21.4,
            "p99_latency_ms": 24.8,
            "colocated_decode_p99_before_green_ctx_ms": 68.4,
            "colocated_decode_p99_after_green_ctx_ms": 24.8,
            "green_ctx_decode_p99_reduction_x": round(68.4 / 24.8, 2),  # 2.76x
            "noisy_neighbor_p99_inflation_pct": round(((24.8 - isolated_baseline_p99_ms) / isolated_baseline_p99_ms) * 100.0, 1),  # 3.3%
            "fault_blast_radius_pct": 100.0,
            "hbm_oom_isolation": "shared_single_engine_process_managed_paged_kv_allocator",
            "lora_consolidation_scenario": {
                "lora_tenant_count": lora_tenant_count,
                "adapter_vram_each_mb": 160.0,
                "total_adapter_pool_vram_gb": 4.48,
                "shared_base_model_vram_gb": 70.0,
                "dedicated_gpus_without_multi_lora": dedicated_70b_gpus,
                "shared_gpus_with_multi_lora": multi_lora_shared_gpus,
                "gpu_count_reduction_x": multi_lora_gpu_reduction_x,
                "dedicated_monthly_cost_usd": dedicated_lora_monthly_usd,
                "multi_lora_monthly_cost_usd": multi_lora_monthly_usd,
                "monthly_savings_usd": multi_lora_monthly_savings_usd,
                "multi_lora_p99_tpot_ms": 26.8,
            },
            "api_semantics_note": "CUDA 12.4+ Green Contexts (cudaGreenCtxCreate / cudaDevResourceTypeSm) partition SMs inside one process/address space without GPU reset, eliminating Prefill-vs-Decode SM starvation while sharing unified HBM; Multi-LoRA serves N adapters inside one trusted inference engine.",
        },
    }

    return {
        "experiment_id": "gpu_sharing_isolation_selection",
        "title": "Multi-Tenant GPU Fractional Sharing & Hardware Isolation Selection: OS Time-Slicing vs. CUDA MPS vs. NVIDIA Hardware MIG vs. CUDA Green Contexts / Multi-LoRA",
        "provenance": {
            "model_type": "synthetic_queuing_and_partition_model_calibrated_to_anonymized_reported_specs",
            "independently_reproduced": False,
            "gpu_cluster_hardware_measured": "not_measured",
            "local_cpu_validator_measured": "measured",
            "official_references": [
                "NVIDIA Multi-Instance GPU (MIG) User Guide: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html (Hardware partitioning of SMs, memory controllers, L2 cache crossbar, and error isolation)",
                "NVIDIA Multi-Process Service (MPS) Docs: https://docs.nvidia.com/deploy/mps/index.html (Shared GPU virtual address space and fatal fault domain across MPS clients)",
                "NVIDIA CUDA 12.4+ Driver API Green Contexts: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html (Dynamic SM resource partitioning via cudaDevResourceTypeSm)",
                "S-LoRA / Punica Segmented Gather-Scatter Multi-LoRA Serving (SGLang / vLLM production multi-adapter batching)",
            ],
        },
        "baseline_parameters": {
            "gpu_sku": "NVIDIA H100 SXM 80GB (132 SMs, 3,350 GB/s HBM3, up to 7 MIG instances)",
            "gpu_hourly_cost_usd": gpu_hourly_usd,
            "hours_per_month": hours_per_month,
            "single_gpu_monthly_cost_usd": single_gpu_monthly_usd,
            "microservice_tenant_count": tenants_per_gpu,
            "dedicated_monthly_cost_7_tenants_usd": dedicated_monthly_cost_usd,
            "consolidated_1gpu_monthly_cost_usd": consolidated_monthly_cost_usd,
            "monthly_tco_savings_7_to_1_usd": monthly_tco_savings_7_to_1_usd,
            "tco_reduction_7_to_1_pct": tco_reduction_7_to_1_pct,
        },
        "tiers": tiers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify results.json matches deterministic model.")
    args = parser.parse_args()

    expected = compute_model()
    if args.check:
        if not RESULTS_PATH.exists():
            print(f"ERROR: {RESULTS_PATH} does not exist.", file=sys.stderr)
            return 1
        actual = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if actual != expected:
            print("ERROR: results.json drifted from deterministic model.", file=sys.stderr)
            return 1
        print("OK: experiments/gpu_sharing_isolation_selection/results.json matches deterministic model.")
        return 0

    RESULTS_PATH.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
