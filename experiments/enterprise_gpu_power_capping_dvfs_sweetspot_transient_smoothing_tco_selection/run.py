from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    power_profiles = [
        {
            "profile": "Tier 1: Uncapped 700W TDP on All GPUs (Prefill & Decode Both at 1980 MHz Boost, High Thermal Failure Rate)",
            "gpu_power_limit_watts": 700,
            "prefill_throughput_retained_pct": 100.0,
            "decode_throughput_retained_pct": 100.0,
            "tokens_per_joule_efficiency_index": 1.00,
            "gpus_deployable_under_1500kw_it_cap": 1024,
        },
        {
            "profile": "Tier 2: Aggressive 400W Power Cap Across All Nodes (Severe Prefill Clock Throttling)",
            "gpu_power_limit_watts": 400,
            "prefill_throughput_retained_pct": 74.5,
            "decode_throughput_retained_pct": 92.0,
            "tokens_per_joule_efficiency_index": 1.48,
            "gpus_deployable_under_1500kw_it_cap": 1536,
        },
        {
            "profile": "Tier 3: Static 550W Cubic V-f Sweet-Spot Cap (21.4% Power Reduction, 97.2% Decode Speed Retained)",
            "gpu_power_limit_watts": 550,
            "prefill_throughput_retained_pct": 92.8,
            "decode_throughput_retained_pct": 97.2,
            "tokens_per_joule_efficiency_index": 1.24,
            "gpus_deployable_under_1500kw_it_cap": 1248,
        },
        {
            "profile": "Tier 4: Role-Aware PD Power Capping (Prefill Pool at 650W Compute-Peak + Decode Pool at 500W HBM-Bound + Dummy-GEMM Ramp Smoothing)",
            "gpu_power_limit_watts": 535,
            "prefill_throughput_retained_pct": 98.4,
            "decode_throughput_retained_pct": 98.8,
            "tokens_per_joule_efficiency_index": 1.31,
            "gpus_deployable_under_1500kw_it_cap": 1280,
        },
    ]
    return {
        "experiment_id": "enterprise_gpu_power_capping_dvfs_sweetspot_transient_smoothing_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "facility_it_power_cap_kw": 1500,
        "power_profiles": power_profiles,
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
        print("OK: enterprise_gpu_power_capping_dvfs_sweetspot_transient_smoothing_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
