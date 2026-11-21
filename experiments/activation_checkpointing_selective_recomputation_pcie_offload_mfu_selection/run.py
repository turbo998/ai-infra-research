from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "activation_checkpointing_selective_recomputation_pcie_offload_mfu_selection",
        "title": "Activation Checkpointing & Offloading Selection: Selective Recomputation & PCIe Gen5 Async Offload vs. Full Checkpointing",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "model": "Llama-3-70B Training at 16K Context Length (BF16 + FlashAttention-3)",
            "gpu_sku": "NVIDIA H100 80GB / H200 141GB (PCIe Gen5 x16 56 GB/s DMA)",
            "baseline_weights_and_states_gb_per_gpu": 46.0,
        },
        "tiers": [
            {
                "tier": "Tier 1: No Activation Checkpointing (Store All Intermediate Activations in HBM)",
                "activation_hbm_gb_per_gpu": 98.0,
                "extra_recompute_flops_pct": 0.0,
                "fits_80gb_h100": False,
                "effective_model_mfu_pct": 0.0,
                "recommended_scenario": "Fails with CUDA OOM on 80GB H100 (46GB states + 98GB activations = 144GB)",
            },
            {
                "tier": "Tier 2: Full Layer Activation Checkpointing (--recompute-granularity full)",
                "activation_hbm_gb_per_gpu": 11.5,
                "extra_recompute_flops_pct": 33.3,
                "fits_80gb_h100": True,
                "effective_model_mfu_pct": 42.5,
                "recommended_scenario": "Emergency memory fallback (wastes +33.3% FLOPs recomputing all Linear GEMMs)",
            },
            {
                "tier": "Tier 3: Selective Activation Checkpointing (Save GEMMs, Recompute Attention/SwiGLU Only)",
                "activation_hbm_gb_per_gpu": 21.0,
                "extra_recompute_flops_pct": 4.5,
                "fits_80gb_h100": True,
                "effective_model_mfu_pct": 53.6,
                "recommended_scenario": "Standard 8K-32K context training when HBM has >=25GB headroom",
            },
            {
                "tier": "Tier 4 (Recommended): Selective Op Checkpointing + Async PCIe Gen5 Pinned-DRAM Offload",
                "activation_hbm_gb_per_gpu": 7.8,
                "extra_recompute_flops_pct": 3.2,
                "fits_80gb_h100": True,
                "effective_model_mfu_pct": 57.8,
                "recommended_scenario": "Long-context 16K-128K training maximizing both batch size and effective MFU (+36% vs Full)",
            },
        ],
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
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
