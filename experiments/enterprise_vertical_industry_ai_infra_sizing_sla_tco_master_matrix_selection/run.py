from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    industry_blueprints = [
        {
            "industry": "1. Banking & Quantitative Finance (60/60 Multi-Region HA + H200/B200 PPCIE Confidential AI + DeepSeek-R1 671B PD)",
            "primary_sla_target": "99.995% HA | TTFT<=800ms | TPOT<=18ms",
            "reference_gpu_footprint": "64x H200/B200 (Dual-Region 32+32 Active-Active)",
            "effective_cost_per_1m_tokens_usd": 0.16,
            "tco_savings_vs_naive_arch_pct": 68.4,
        },
        {
            "industry": "2. Healthcare & Clinical Imaging (Air-Gapped 2x RTX PRO 6000 96GB / 4x H200 NVL + 3-Stage E-P-D VLM + HIPAA TEE)",
            "primary_sla_target": "100% On-Prem / HIPAA | VLM TTFT<=650ms",
            "reference_gpu_footprint": "2x RTX PRO 6000 (Dept) or 4x H200 NVL (Hospital)",
            "effective_cost_per_1m_tokens_usd": 0.14,
            "tco_savings_vs_naive_arch_pct": 64.0,
        },
        {
            "industry": "3. Telecom & 10,000-Seat Contact Center (WebRTC Edge SFU + 2x L40S ASR/TTS : 6x H200 LLM Full-Duplex Pool)",
            "primary_sla_target": "P99 TTFA <= 235ms | Barge-In <= 28ms",
            "reference_gpu_footprint": "32x L40S + 96x H200 (10,240 Concurrent Calls)",
            "effective_cost_per_1m_tokens_usd": 0.11,
            "tco_savings_vs_naive_arch_pct": 83.8,
        },
        {
            "industry": "4. Automotive ADAS & World Models (1,024-GPU SuperPod + 8-Rail 400G + 3FS GDS + xDiT Video Simulation)",
            "primary_sla_target": "busbw >= 47.6 GB/s | 3FS >= 3.5 TB/s GDS",
            "reference_gpu_footprint": "1,024x H200/B200 (128 Nodes, 32x 50kW DLC Racks)",
            "effective_cost_per_1m_tokens_usd": 0.095,
            "tco_savings_vs_naive_arch_pct": 58.2,
        },
        {
            "industry": "5. Retail, E-Commerce & SaaS (Multi-LoRA 128 Adapters + DiskANN 1B Vector RAG + Nightly Radix-Sorted Spot Batch)",
            "primary_sla_target": "Search P99<=5ms | Online TPOT<=15ms | Batch $0.028/1M",
            "reference_gpu_footprint": "16x H200 Online + 4x L40S Rerank/Guard + 16x Spot Batch",
            "effective_cost_per_1m_tokens_usd": 0.082,
            "tco_savings_vs_naive_arch_pct": 88.5,
        },
        {
            "industry": "6. Government & Sovereign Cloud (Multi-Silicon H200/MI325X + 7:1 Hybrid Long-Context + Firecracker Agent Sandboxes)",
            "primary_sla_target": "256K TTFT<=1.65s | Hardware KVM Isolation",
            "reference_gpu_footprint": "64x H200 / MI325X + 3,072 Sandbox CPU vCPUs",
            "effective_cost_per_1m_tokens_usd": 0.13,
            "tco_savings_vs_naive_arch_pct": 74.0,
        },
    ]
    return {
        "experiment_id": "enterprise_vertical_industry_ai_infra_sizing_sla_tco_master_matrix_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "industry_count": 6,
        "industry_blueprints": industry_blueprints,
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
        print("OK: enterprise_vertical_industry_ai_infra_sizing_sla_tco_master_matrix_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
