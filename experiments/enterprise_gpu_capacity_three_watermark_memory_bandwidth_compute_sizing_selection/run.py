from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Weight-Only Naive Sizing (2x H100 80GB for 70B BF16 · Ignores KV & Bandwidth)",
            "gpus_provisioned": 2,
            "max_concurrent_16k_sessions": 4,
            "p99_tpot_ms": 48.2,
            "sla_compliance_pct": 18.5,
            "cost_per_1m_goodput_tokens_usd": 1.18,
        },
        {
            "tier": "Tier 2: 4x L40S 48GB GDDR6 (864 GB/s BW, No NVLink · Memory Fits but BW Strangled)",
            "gpus_provisioned": 4,
            "max_concurrent_16k_sessions": 28,
            "p99_tpot_ms": 34.6,
            "sla_compliance_pct": 46.0,
            "cost_per_1m_goodput_tokens_usd": 0.64,
        },
        {
            "tier": "Tier 3: 16x H100 80GB SXM (2 Nodes for DeepSeek 671B FP8 + 14B MTP)",
            "gpus_provisioned": 16,
            "max_concurrent_16k_sessions": 160,
            "p99_tpot_ms": 15.8,
            "sla_compliance_pct": 99.1,
            "cost_per_1m_goodput_tokens_usd": 0.29,
        },
        {
            "tier": "Tier 4: 3-Watermark Balanced Single-Node 8x H200 141GB / 8x B200 180GB (671B FP8 + 14B MTP)",
            "gpus_provisioned": 8,
            "max_concurrent_16k_sessions": 192,
            "p99_tpot_ms": 11.4,
            "sla_compliance_pct": 99.8,
            "cost_per_1m_goodput_tokens_usd": 0.16,
        },
    ]
    return {
        "experiment_id": "enterprise_gpu_capacity_three_watermark_memory_bandwidth_compute_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "watermark_equations": {
            "w1_memory_gpus": "ceil((M_weights + M_kv + M_act_graph) / (0.85 * M_HBM_per_gpu))",
            "w2_bandwidth_gpus": "ceil((M_active_weights + M_kv_read) / (TPOT_target_sec * BW_eff_per_gpu))",
            "w3_compute_gpus": "ceil((RPS_peak * S_in * (1-eta) * 2 * P_active) / (TFLOPS_eff_per_gpu * 1e12))",
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
        print("OK: enterprise_gpu_capacity_three_watermark_memory_bandwidth_compute_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
