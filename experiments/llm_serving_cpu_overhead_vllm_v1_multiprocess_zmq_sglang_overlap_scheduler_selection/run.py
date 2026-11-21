from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "llm_serving_cpu_overhead_vllm_v1_multiprocess_zmq_sglang_overlap_scheduler_selection",
        "title": "LLM Serving CPU Overhead & Zero-Bubble Scheduling: vLLM V1 Multiprocess ZMQ IPC & SGLang Overlap Scheduler vs. Monolithic GIL Loop",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "model": "Llama-3.3-70B-FP8 on 2x H100 / 1x B200",
            "active_batch_size": 128,
            "pure_gpu_forward_step_ms": 10.0,
        },
        "tiers": [
            {
                "tier": "Tier 1: Monolithic Single-Process (FastAPI + Coupled Scheduler + Sync D2H Detokenizer)",
                "exposed_cpu_bubble_ms_per_step": 7.0,
                "effective_step_tpot_ms": 17.0,
                "gpu_active_duty_cycle_pct": 58.8,
                "system_output_tps": 7529,
                "recommended_scenario": "Legacy v0 single-process anti-pattern (GIL & sync D2H stall GPU 41% of the time)",
            },
            {
                "tier": "Tier 2: Multi-Step Coupled Scheduling (--num-scheduler-steps 8)",
                "exposed_cpu_bubble_ms_per_step": 2.4,
                "effective_step_tpot_ms": 12.4,
                "gpu_active_duty_cycle_pct": 80.6,
                "system_output_tps": 10322,
                "recommended_scenario": "Transitional workaround (inflates newly arrived prefill TTFT by up to 8 steps)",
            },
            {
                "tier": "Tier 3: Multiprocess ZMQ IPC Decoupling (Frontend API <-> Tokenizer <-> EngineCore)",
                "exposed_cpu_bubble_ms_per_step": 1.1,
                "effective_step_tpot_ms": 11.1,
                "gpu_active_duty_cycle_pct": 90.1,
                "system_output_tps": 11531,
                "recommended_scenario": "Multi-core CPU isolation eliminating FastAPI/Tokenizer Python GIL stalls",
            },
            {
                "tier": "Tier 4 (Recommended): vLLM V1 ZMQ Multiprocess + SGLang Async Overlap Scheduler (Step T+1 CPU Overlapped With Step T GPU)",
                "exposed_cpu_bubble_ms_per_step": 0.08,
                "effective_step_tpot_ms": 10.08,
                "gpu_active_duty_cycle_pct": 99.2,
                "system_output_tps": 12698,
                "recommended_scenario": "Production H100/H200/B200 high-throughput serving with 99.2% GPU duty cycle",
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
