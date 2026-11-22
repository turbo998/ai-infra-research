from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Single-Vendor Lock-In on Legacy 80GB H100 Only (Forced 2-Node Split for 671B)",
            "hbm_per_8chip_node_gb": 640,
            "hbm_bandwidth_per_chip_tbps": 3.35,
            "p99_tpot_ms": 16.4,
            "cost_per_1m_goodput_tokens_usd": 0.31,
        },
        {
            "tier": "Tier 2: Google Cloud TPU v6e (Trillium) / v5p Pod with vLLM-TPU & Persistent XLA Cache",
            "hbm_per_8chip_node_gb": 768,
            "hbm_bandwidth_per_chip_tbps": 1.60,
            "p99_tpot_ms": 14.8,
            "cost_per_1m_goodput_tokens_usd": 0.19,
        },
        {
            "tier": "Tier 3: AMD Instinct MI300X / MI325X (192–256GB HBM3e, 6.0 TB/s · Giant Single-Node KV Pool)",
            "hbm_per_8chip_node_gb": 2048,
            "hbm_bandwidth_per_chip_tbps": 6.00,
            "p99_tpot_ms": 10.6,
            "cost_per_1m_goodput_tokens_usd": 0.14,
        },
        {
            "tier": "Tier 4: Workload-Matched Multi-Silicon Pool (NVIDIA H200/B200 Low-Latency + AMD MI325X Giant-KV Decode + TPU v6e Batch/Prefill)",
            "hbm_per_8chip_node_gb": 1440,
            "hbm_bandwidth_per_chip_tbps": 8.00,
            "p99_tpot_ms": 8.2,
            "cost_per_1m_goodput_tokens_usd": 0.12,
        },
    ]
    return {
        "experiment_id": "enterprise_multi_silicon_gpu_tpu_h200_b200_mi325x_tpu_v6e_tco_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "silicon_specs": {
            "nvidia_h200_sxm": {"hbm_gb": 141, "bw_tbps": 4.8, "fp8_tflops": 1979},
            "nvidia_dgx_b200": {"hbm_gb": 180, "bw_tbps": 8.0, "fp8_tflops": 4500},
            "amd_mi325x": {"hbm_gb": 256, "bw_tbps": 6.0, "fp8_tflops": 2610},
            "google_tpu_v6e": {"hbm_gb": 32, "bw_tbps": 1.6, "bf16_int8_tflops": 918},
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
        print("OK: enterprise_multi_silicon_gpu_tpu_h200_b200_mi325x_tpu_v6e_tco_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
