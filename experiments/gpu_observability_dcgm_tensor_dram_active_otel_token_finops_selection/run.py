from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "gpu_observability_dcgm_tensor_dram_active_otel_token_finops_selection",
        "title": "GPU Observability & Token FinOps Architecture: DCGM Profiling (TENSOR_ACTIVE vs. DRAM_ACTIVE) & OpenTelemetry Chargeback",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "cluster_gpus": 256,
            "gpu_sku": "NVIDIA H200 SXM 141GB",
            "enterprise_business_units": 45,
        },
        "tiers": [
            {
                "tier_id": "tier1_nvidia_smi_gpu_util_only_and_flat_gpu_hour_chargeback",
                "name": "Tier 1: Coarse `GPU_UTIL` (Field 203) Monitoring Only + Flat GPU-Hour Chargeback (Blind to Fake-Busy GPUs)",
                "true_hardware_efficiency_mbu_mfu_pct": 24.8,
                "unattributed_or_wasted_gpu_spend_pct": 52.4,
                "bottleneck_root_cause_mttr_min": 185.0,
                "monthly_256gpu_cluster_net_tco_usd": 640000,
                "verdict": "Anti-Pattern: `GPU_UTIL=96%` hides `TENSOR_ACTIVE=14%` & `DRAM_ACTIVE=21%` (NCCL/CPU stalls!); 52.4% wasted spend",
            },
            {
                "tier_id": "tier2_dcgm_basic_counters_without_profiling_or_token_otel",
                "name": "Tier 2: Standard DCGM Exporter (Temp/Power/VRAM Only, No `1004/1005` Profiling Metrics or Token Tracing)",
                "true_hardware_efficiency_mbu_mfu_pct": 41.2,
                "unattributed_or_wasted_gpu_spend_pct": 34.0,
                "bottleneck_root_cause_mttr_min": 68.0,
                "monthly_256gpu_cluster_net_tco_usd": 485000,
                "verdict": "Catches OOM & thermal throttling, but still cannot distinguish Compute-Bound vs. Bandwidth-Bound vs. Comm-Stalled",
            },
            {
                "tier_id": "tier3_dcgm_profiling_metrics_without_tenant_token_finops_attribution",
                "name": "Tier 3: DCGM Profiling (`TENSOR_ACTIVE` + `DRAM_ACTIVE` + `NVLINK/PCIE`) Without Per-Tenant Token Chargeback",
                "true_hardware_efficiency_mbu_mfu_pct": 68.4,
                "unattributed_or_wasted_gpu_spend_pct": 18.5,
                "bottleneck_root_cause_mttr_min": 11.0,
                "monthly_256gpu_cluster_net_tco_usd": 360000,
                "verdict": "Pinpoints Roofline bottlenecks in 11 minutes, but shared multi-tenantMaaS pools lack fair unit-token billing",
            },
            {
                "tier_id": "tier4_dcgm_4quadrant_profiling_plus_otel_genai_weighted_token_finops",
                "name": "Tier 4: 4-Quadrant DCGM Profiling (`1002-1012`) + OpenTelemetry GenAI Spans + Weighted Token FinOps Showback",
                "true_hardware_efficiency_mbu_mfu_pct": 76.8,
                "unattributed_or_wasted_gpu_spend_pct": 1.8,
                "bottleneck_root_cause_mttr_min": 3.5,
                "monthly_256gpu_cluster_net_tco_usd": 258000,
                "verdict": "Recommended: 76.8% true hardware efficiency (3.10x), 3.5m MTTR, 98.2% FinOps attribution & -59.7% net TCO",
            },
        ],
        "headline_findings": {
            "true_hardware_efficiency_pct": 76.8,
            "efficiency_gain_vs_tier1_x": 3.10,
            "root_cause_mttr_min": 3.5,
            "tco_savings_vs_tier1_pct": 59.7,
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
