"""Reproducible model for #588 Enterprise Batch LLM Inference & Nightly Tidal GPU Backfill."""
from __future__ import annotations
import json
from pathlib import Path


def evaluate_batch_tidal_backfill(nightly_million_tokens: int = 500, idle_h200_gpus: int = 6) -> dict:
    online_tps_per_h200 = 1150
    offline_sorted_tps_per_h200 = 4850
    speedup_ratio = round(offline_sorted_tps_per_h200 / online_tps_per_h200, 2)

    total_offline_tps = idle_h200_gpus * offline_sorted_tps_per_h200
    completion_hours = round((nightly_million_tokens * 1_000_000) / (total_offline_tps * 3600), 2)
    max_8h_capacity_million_tokens = round((total_offline_tps * 8 * 3600) / 1_000_000)

    monthly_million_tokens = nightly_million_tokens * 30
    cloud_batch_api_monthly_usd = round(monthly_million_tokens * 0.75)
    dedicated_spot_monthly_usd = round(monthly_million_tokens * 0.207)
    tidal_backfill_monthly_usd = round(monthly_million_tokens * 0.019)

    return {
        "topic_id": 588,
        "nightly_million_tokens": nightly_million_tokens,
        "idle_h200_gpus": idle_h200_gpus,
        "online_tps_per_h200": online_tps_per_h200,
        "offline_sorted_tps_per_h200": offline_sorted_tps_per_h200,
        "speedup_ratio": speedup_ratio,
        "completion_hours": completion_hours,
        "max_8h_capacity_million_tokens": max_8h_capacity_million_tokens,
        "cloud_batch_api_monthly_usd": cloud_batch_api_monthly_usd,
        "dedicated_spot_monthly_usd": dedicated_spot_monthly_usd,
        "tidal_backfill_monthly_usd": tidal_backfill_monthly_usd,
        "savings_vs_cloud_batch_pct": round(
            (1.0 - tidal_backfill_monthly_usd / cloud_batch_api_monthly_usd) * 100, 1
        ),
    }


def main() -> None:
    res = evaluate_batch_tidal_backfill()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
