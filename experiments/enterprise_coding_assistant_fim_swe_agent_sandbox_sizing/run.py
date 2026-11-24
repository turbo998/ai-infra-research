"""Reproducible model for #582 Enterprise AI Coding Assistant & SWE Agent Sandbox Sizer."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_coding_platform(developers: int = 1000) -> dict:
    active_devs = developers * 0.42
    raw_fim_qps = round((active_devs * 11.0) / 60.0, 2)
    debounced_fim_qps = round((active_devs * 4.0) / 60.0, 2)
    debounce_reduction_pct = round((1.0 - debounced_fim_qps / raw_fim_qps) * 100, 1)

    fim_h100_gpus = max(2, math.ceil(debounced_fim_qps / 7.2))
    swe_turn_qps = round((developers / 1000.0) * 43.2, 2)
    swe_h200_gpus = max(4, math.ceil(swe_turn_qps / 5.4))
    kata_sandbox_nodes = max(2, math.ceil(developers / 250.0))

    self_hosted_monthly_usd = fim_h100_gpus * 2300 + swe_h200_gpus * 3400 + kata_sandbox_nodes * 800
    copilot_enterprise_monthly_usd = developers * 39

    return {
        "topic_id": 582,
        "developers": developers,
        "raw_fim_qps": raw_fim_qps,
        "debounced_fim_qps": debounced_fim_qps,
        "debounce_reduction_pct": debounce_reduction_pct,
        "fim_h100_gpus": fim_h100_gpus,
        "swe_turn_qps": swe_turn_qps,
        "swe_h200_gpus": swe_h200_gpus,
        "kata_sandbox_nodes": kata_sandbox_nodes,
        "fim_p95_ttft_isolated_ms": 88,
        "fim_p99_ttft_shared_pool_ms": 1820,
        "self_hosted_monthly_usd": self_hosted_monthly_usd,
        "copilot_enterprise_monthly_usd": copilot_enterprise_monthly_usd,
    }


def main() -> None:
    res = evaluate_coding_platform()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
