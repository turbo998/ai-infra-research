from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Raw Input+Output TPM Sum Sizing (1:1 Weighting + Daily Average RPS · Anti-Pattern)",
            "peak_http_429_throttling_pct": 64.2,
            "p99_itl_ms": 92.0,
            "goodput_sla_attainment_pct": 35.8,
            "monthly_cluster_bill_usd": 218000,
        },
        {
            "tier": "Tier 2: Peak 100% PTU Over-Provisioning (Sizing All Peak Traffic on Dedicated PTU)",
            "peak_http_429_throttling_pct": 0.4,
            "p99_itl_ms": 21.0,
            "goodput_sla_attainment_pct": 98.4,
            "monthly_cluster_bill_usd": 312000,
        },
        {
            "tier": "Tier 3: Little's Law Active-Slot Sizing + Output-Weighted TPM (1 Out = 5 In)",
            "peak_http_429_throttling_pct": 1.2,
            "p99_itl_ms": 17.5,
            "goodput_sla_attainment_pct": 98.8,
            "monthly_cluster_bill_usd": 146000,
        },
        {
            "tier": "Tier 4: Little's Law + P80 Base PTU/GPU (Prefix Cache Deducted) + P20 Serverless Spillover + Dual-SLA Goodput Gate",
            "peak_http_429_throttling_pct": 0.0,
            "p99_itl_ms": 14.2,
            "goodput_sla_attainment_pct": 99.7,
            "monthly_cluster_bill_usd": 98400,
        },
    ]
    scenario_slas = {
        "realtime_voice_agent": {"p95_ttft_ms": 250, "p99_itl_ms": 18, "min_user_tps": 55},
        "ide_code_completion_fim": {"p95_ttft_ms": 150, "p99_itl_ms": 12, "min_user_tps": 80},
        "enterprise_rag_chat": {"p95_ttft_ms": 800, "p99_itl_ms": 25, "min_user_tps": 40},
        "deep_reasoning_cot_agent": {"p95_ttft_ms": 1500, "p99_itl_ms": 20, "min_user_tps": 50},
        "offline_batch_pipeline": {"p95_ttft_ms": 30000, "p99_itl_ms": 100, "min_user_tps": 10},
    }
    return {
        "experiment_id": "enterprise_tpm_tps_littles_law_ttft_tpot_goodput_sla_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "core_formulas": {
            "e2el_sec": "TTFT_sec + (N_out - 1) * TPOT_sec",
            "littles_law_active_concurrency": "B_active = RPS_peak * E2EL_sec",
            "weighted_ptu_tpm": "TPM_in * (1 - eta_cache) + 5 * TPM_out",
        },
        "scenario_sla_benchmarks": scenario_slas,
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
        print("OK: enterprise_tpm_tps_littles_law_ttft_tpot_goodput_sla_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
