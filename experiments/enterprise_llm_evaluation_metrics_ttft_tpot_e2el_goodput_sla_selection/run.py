from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_llm_evaluation_metrics_ttft_tpot_e2el_goodput_sla_selection",
        "title": "Enterprise AI Evaluation Metrics (TTFT, TPOT/ITL, E2EL, Goodput & Effective $/1M Tokens): 4-Tier Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "cluster_gpus": 8,
            "gpu_sku": "NVIDIA H200 SXM 141GB",
            "model": "Llama-3.3-70B-FP8",
            "ttft_sla_ms": 800.0,
            "tpot_sla_ms": 25.0,
        },
        "tiers": [
            {
                "tier_id": "tier1_max_batch_raw_tps_tuning_ignoring_p99_ttft_and_itl_sla",
                "name": "Tier 1: Raw-TPS Maximization (B=256 Saturation, Ignores P99 TTFT Queueing & P99 ITL Stall Spikes)",
                "raw_system_tps_per_gpu": 2650,
                "sla_compliant_goodput_tps_per_gpu": 410,
                "p99_ttft_ms": 4850.0,
                "p99_itl_ms": 92.4,
                "effective_cost_per_1m_goodput_tokens_usd": 2.41,
                "verdict": "Anti-Pattern: High raw TPS (2,650), but 84.5% of requests breach SLA -> only 410 Goodput tok/s/GPU!",
            },
            {
                "tier_id": "tier2_p50_mean_tpot_tuning_without_p99_itl_chunk_or_queue_guard",
                "name": "Tier 2: P50 Mean-Latency Tuning (Meets Average TPOT, Fails P99 ITL Under Long-Prompt Bursts)",
                "raw_system_tps_per_gpu": 1980,
                "sla_compliant_goodput_tps_per_gpu": 1120,
                "p99_ttft_ms": 1420.0,
                "p99_itl_ms": 46.8,
                "effective_cost_per_1m_goodput_tokens_usd": 0.88,
                "verdict": "43.4% of requests suffer mid-stream P99 ITL freezes (46.8ms) during RAG prompt bursts",
            },
            {
                "tier_id": "tier3_overthrottled_low_batch_for_ultra_low_latency",
                "name": "Tier 3: Over-Throttled Low Concurrency (Caps B=16 to Guarantee Latency, Wastes 62% GPU Bandwidth)",
                "raw_system_tps_per_gpu": 890,
                "sla_compliant_goodput_tps_per_gpu": 885,
                "p99_ttft_ms": 380.0,
                "p99_itl_ms": 13.8,
                "effective_cost_per_1m_goodput_tokens_usd": 1.12,
                "verdict": "99.4% SLA pass rate, but severe GPU under-utilization inflates cost to $1.12/1M Goodput tokens",
            },
            {
                "tier_id": "tier4_dual_sla_goodput_pareto_knee_controller_pd_and_admission_gate",
                "name": "Tier 4: Dual-SLA Goodput Pareto-Knee Controller (P95 TTFT <=800ms ∧ P99 ITL <=25ms + Admission Gate)",
                "raw_system_tps_per_gpu": 2180,
                "sla_compliant_goodput_tps_per_gpu": 2115,
                "p99_ttft_ms": 590.0,
                "p99_itl_ms": 16.8,
                "effective_cost_per_1m_goodput_tokens_usd": 0.47,
                "verdict": "Recommended: 2,115 Goodput tok/s/GPU (5.16x vs. Tier 1), 97.0% SLA pass & $0.47/1M Goodput tokens (-80.5%)",
            },
        ],
        "headline_findings": {
            "goodput_tps_per_gpu": 2115,
            "goodput_speedup_vs_tier1_x": 5.16,
            "p99_itl_ms": 16.8,
            "effective_cost_reduction_vs_tier1_pct": 80.5,
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
