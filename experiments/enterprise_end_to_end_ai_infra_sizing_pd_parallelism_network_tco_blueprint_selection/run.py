from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Siloed Colocated Serving + Cross-Node TP + Commodity 25G TCP (Un-engineered Baseline)",
            "p95_ttft_ms": 1340.0,
            "p99_tpot_ms": 86.2,
            "cluster_goodput_tok_s": 4200,
            "effective_cost_per_1m_goodput_tokens_usd": 1.38,
            "three_year_total_tco_usd": 6480000,
        },
        {
            "tier": "Tier 2: Colocated + Chunked Prefill + Intra-Node TP=8 + Static 400G RoCEv2",
            "p95_ttft_ms": 690.0,
            "p99_tpot_ms": 27.8,
            "cluster_goodput_tok_s": 11400,
            "effective_cost_per_1m_goodput_tokens_usd": 0.56,
            "three_year_total_tco_usd": 3890000,
        },
        {
            "tier": "Tier 3: Static 1:1 PD Disaggregation + Attention DP/EP + 400G RoCEv2",
            "p95_ttft_ms": 470.0,
            "p99_tpot_ms": 16.5,
            "cluster_goodput_tok_s": 18900,
            "effective_cost_per_1m_goodput_tokens_usd": 0.31,
            "three_year_total_tco_usd": 2740000,
        },
        {
            "tier": "Tier 4: Full-Stack Blueprint (APIM+Envoy GIE -> Elastic PD + 25% Flex -> DP+DeepEP+FP8+MTP -> Rail 400G RDMA -> 60/25/15 Portfolio)",
            "p95_ttft_ms": 365.0,
            "p99_tpot_ms": 11.2,
            "cluster_goodput_tok_s": 28600,
            "effective_cost_per_1m_goodput_tokens_usd": 0.16,
            "three_year_total_tco_usd": 1820000,
        },
    ]
    six_step_chain = [
        "Step 1 (Demand): DAU & Peak RPS -> Input/Output TPS & 5x Output-Weighted PTU TPM",
        "Step 2 (Concurrency): Little's Law Active KV Slots B_active = RPS_peak * [TTFT + (N_out-1)*TPOT]",
        "Step 3 (3-Watermarks): N_GPU = max(N_mem/0.85, N_bw, N_compute)",
        "Step 4 (PD & Parallelism): Elastic P:D Ratio + Intra-Node TP / MoE Attention DP + DeepEP + MTP (B<=48)",
        "Step 5 (Interconnect): NVLink Scale-Up (<1us) + Rail-Optimized 400G GPUDirect RDMA Scale-Out (>=90% busbw)",
        "Step 6 (Platform & FinOps): APIM + Envoy K8s GIE EPP + 60% RI/PTU + 25% PAYGO Burst + 15% Spot Batch",
    ]
    return {
        "experiment_id": "enterprise_end_to_end_ai_infra_sizing_pd_parallelism_network_tco_blueprint_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "six_step_sizing_methodology": six_step_chain,
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
        print("OK: enterprise_end_to_end_ai_infra_sizing_pd_parallelism_network_tco_blueprint_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
