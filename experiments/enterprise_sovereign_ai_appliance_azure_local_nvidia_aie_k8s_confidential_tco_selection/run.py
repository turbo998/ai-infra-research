from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    h200_gpus = 32
    tiers = [
        {
            "tier": "Tier 1: Ad-Hoc Manual Bare-Metal Assembly (Custom Scripts + Legacy Single-GPU PCIe Bounce TEE)",
            "air_gapped_deployment_hours": 504.0,
            "confidential_tee_throughput_tax_pct": 53.5,
            "three_year_software_license_usd": 0,
            "three_year_total_tco_usd": 2980000,
        },
        {
            "tier": "Tier 2: Turnkey Commercial Appliance (NVIDIA DGX BasePOD + AI Enterprise $4,500/GPU/yr + Run:ai)",
            "air_gapped_deployment_hours": 48.0,
            "confidential_tee_throughput_tax_pct": 6.2,
            "three_year_software_license_usd": 432000,
            "three_year_total_tco_usd": 2840000,
        },
        {
            "tier": "Tier 3: Cloud-Connected Sovereign Stack (Azure Local [Stack HCI] + AKS Arc + Foundry Local)",
            "air_gapped_deployment_hours": 36.0,
            "confidential_tee_throughput_tax_pct": 5.5,
            "three_year_software_license_usd": 192000,
            "three_year_total_tco_usd": 2360000,
        },
        {
            "tier": "Tier 4: Open K8s + Zarf Immutable Air-Gapped Bundle + Kueue + H200 PPCIE / B200 Confidential TEE",
            "air_gapped_deployment_hours": 4.5,
            "confidential_tee_throughput_tax_pct": 4.8,
            "three_year_software_license_usd": 0,
            "three_year_total_tco_usd": 1920000,
        },
    ]
    return {
        "experiment_id": "enterprise_sovereign_ai_appliance_azure_local_nvidia_aie_k8s_confidential_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "cluster_h200_gpus": h200_gpus,
            "nvaie_annual_license_per_gpu_usd": 4500,
            "horizon_years": 3,
        },
        "tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_sovereign_ai_appliance_azure_local_nvidia_aie_k8s_confidential_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
