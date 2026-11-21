from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "single_node_hgx_vs_multinode_rdma_vs_rack_nvl72_tco_goodput_selection",
        "title": "Single-Node 8-GPU HGX (H200/B200) vs. Multi-Node 400G/800G RDMA vs. Rack-Scale Supernode (GB200/GB300 NVL72) Selection, Liquid Cooling Boundaries & 3-Year TCO Guide",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "target_model": "DeepSeek-V3/R1 671B & Frontier 1T+ Reasoning MoE",
            "cooling_threshold_kw_per_rack": 45.0,
        },
        "tiers": [
            {
                "tier": "Tier 1: Dual-Node 16x H100 (80GB) over Legacy Unoptimized Network (Cross-Node Bottleneck)",
                "rack_power_kw": 22.0,
                "p99_tpot_ms": 24.5,
                "goodput_tok_s_per_gpu": 920,
                "cost_per_1m_tokens_usd": 0.58,
                "recommended_scenario": "Legacy 80GB HBM clusters unable to fit 671B FP8 inside a single 8-GPU node",
            },
            {
                "tier": "Tier 2: Single-Node 8x H200 (141GB, 1.13TB HBM3e) Compact Appliance (Zero Cross-Node Traffic)",
                "rack_power_kw": 10.5,
                "p99_tpot_ms": 14.6,
                "goodput_tok_s_per_gpu": 1850,
                "cost_per_1m_tokens_usd": 0.29,
                "recommended_scenario": "Best fit for existing air-cooled enterprise datacenters (<20 kW/rack) deploying private 671B FP8",
            },
            {
                "tier": "Tier 3: Multi-Node 32x HGX B200 + 400G/800G RDMA (PD Disaggregation + DeepEP Dual-Batch Overlap)",
                "rack_power_kw": 48.0,
                "p99_tpot_ms": 8.4,
                "goodput_tok_s_per_gpu": 4120,
                "cost_per_1m_tokens_usd": 0.19,
                "recommended_scenario": "Modular high-throughput scale-out cluster for standard enterprise AI factories",
            },
            {
                "tier": "Tier 4 (Flagship): GB200 / GB300 NVL72 Rack Supernode (72-GPU 1,800 GB/s NVLink 5 Copper Backplane EP=72 + NVFP4)",
                "rack_power_kw": 132.0,
                "p99_tpot_ms": 4.8,
                "goodput_tok_s_per_gpu": 7420,
                "cost_per_1m_tokens_usd": 0.12,
                "recommended_scenario": "DLC liquid-cooled (>120 kW/rack) AI factory flagship delivering 8.07x Goodput/GPU and $0.12/1M tokens",
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
