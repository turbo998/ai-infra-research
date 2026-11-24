"""Reproducible model for #594 Enterprise GPU Cold-Start Acceleration & Elastic Autoscaling Buffer Sizer."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_cold_start_autoscaling(model_gb: float = 70.0, base_h100_gpus: int = 32) -> dict:
    legacy_pull_sec = 180.0
    legacy_weight_load_sec = 285.0
    legacy_compile_sec = 45.0
    legacy_total_sec = round(legacy_pull_sec + legacy_weight_load_sec + legacy_compile_sec, 1)

    fast_nydus_p2p_sec = 3.5
    fast_streamer_load_sec = round(model_gb / 6.25, 1)
    fast_warm_cache_sec = 2.8
    fast_total_sec = round(fast_nydus_p2p_sec + fast_streamer_load_sec + fast_warm_cache_sec, 1)

    speedup_ratio = round(legacy_total_sec / fast_total_sec, 1)

    legacy_buffer_gpus = math.ceil(base_h100_gpus * 0.60)
    fast_buffer_gpus = math.ceil(base_h100_gpus * 0.125)

    legacy_fleet_gpus = base_h100_gpus + legacy_buffer_gpus
    fast_fleet_gpus = base_h100_gpus + fast_buffer_gpus
    monthly_savings_usd = (legacy_fleet_gpus - fast_fleet_gpus) * 2300

    return {
        "topic_id": 594,
        "model_gb": model_gb,
        "base_h100_gpus": base_h100_gpus,
        "legacy_total_sec": legacy_total_sec,
        "fast_total_sec": fast_total_sec,
        "speedup_ratio": speedup_ratio,
        "legacy_fleet_gpus": legacy_fleet_gpus,
        "fast_fleet_gpus": fast_fleet_gpus,
        "saved_buffer_gpus": legacy_fleet_gpus - fast_fleet_gpus,
        "monthly_savings_usd": monthly_savings_usd,
        "cost_reduction_pct": round((1.0 - fast_fleet_gpus / legacy_fleet_gpus) * 100, 1),
    }


def main() -> None:
    res = evaluate_cold_start_autoscaling()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
