from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "google_cloud_tpu_v5p_v6e_ironwood_vs_nvidia_gpu_selection",
        "title": "Google Cloud TPU (v5p / v6e Trillium / TPU7x Ironwood / 8t & 8i) vs. NVIDIA GPU (H200 / B200 / GB200 NVL72) Architecture & TCO Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "cluster_accelerator_scale_chips": 2048,
            "tpu7x_ironwood_superpod_max_chips": 9216,
            "tpu7x_fp8_tflops_per_chip": 4614,
            "tpu7x_hbm3e_gb_per_chip": 192,
            "tpu7x_hbm3e_bw_tb_s": 7.37,
        },
        "tiers": [
            {
                "tier_id": "tier1_unmodified_dynamic_shape_pytorch_cuda_on_tpu_nonwrap",
                "name": "Tier 1: Lift-and-Shift Dynamic-Shape PyTorch/CUDA on Non-Wrap TPU Slice (Anti-Pattern)",
                "sustained_cluster_mfu_pct": 14.2,
                "cluster_ocs_or_fabric_goodput_pct": 64.0,
                "serving_cost_per_1m_tokens_usd": 1.48,
                "monthly_2048_chip_equivalent_tco_usd": 4850000,
                "verdict": "Anti-Pattern: Dynamic shapes trigger constant XLA recompilation (14.2% MFU) & non-wrap mesh stalls",
            },
            {
                "tier_id": "tier2_homogeneous_h200_rocev2_for_all_workloads",
                "name": "Tier 2: Homogeneous NVIDIA H200 (8x NVLink + 400G RoCEv2 Clos) Baseline",
                "sustained_cluster_mfu_pct": 44.5,
                "cluster_ocs_or_fabric_goodput_pct": 89.5,
                "serving_cost_per_1m_tokens_usd": 0.62,
                "monthly_2048_chip_equivalent_tco_usd": 3620000,
                "verdict": "Strong CUDA compatibility, but 8-GPU NVLink boundary & Clos Spine optics inflate 2,048-chip TCO",
            },
            {
                "tier_id": "tier3_gb200_nvl72_rack_scale_nvlink_ib_ndr",
                "name": "Tier 3: NVIDIA GB200 NVL72 (72-GPU Copper NVLink Domain + InfiniBand XDR/NDR)",
                "sustained_cluster_mfu_pct": 54.8,
                "cluster_ocs_or_fabric_goodput_pct": 93.4,
                "serving_cost_per_1m_tokens_usd": 0.34,
                "monthly_2048_chip_equivalent_tco_usd": 2980000,
                "verdict": "Best-in-class for CUDA-locked MoE & ultra-long-context decode, but requires 120kW+ DLC racks",
            },
            {
                "tier_id": "tier4_workload_aligned_tpu7x_ironwood_v6e_8t_8i_plus_gpu_hybrid",
                "name": "Tier 4: Workload-Aligned TPU7x Ironwood/8t (4x4x4 Twisted Torus + OCS + Shardy) & v6e/8i + GPU",
                "sustained_cluster_mfu_pct": 58.4,
                "cluster_ocs_or_fabric_goodput_pct": 99.2,
                "serving_cost_per_1m_tokens_usd": 0.24,
                "monthly_2048_chip_equivalent_tco_usd": 2120000,
                "verdict": "Recommended: 58.4% MFU, 99.2% OCS Goodput, $0.24/1M tokens & -41.4% TCO vs. Tier 2",
            },
        ],
        "headline_findings": {
            "sustained_mfu_pct": 58.4,
            "ocs_goodput_pct": 99.2,
            "serving_cost_per_1m_usd": 0.24,
            "tco_savings_vs_tier2_pct": 41.4,
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
