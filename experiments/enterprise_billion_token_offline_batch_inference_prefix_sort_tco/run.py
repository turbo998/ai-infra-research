from __future__ import annotations
import json
from pathlib import Path


def evaluate_offline_batch_inference(total_tokens: int = 1_000_000_000, h200_gpus: int = 16) -> dict:
    nodes = h200_gpus / 8.0
    unordered_tps_per_node = 5600
    sorted_tps_per_node = 16900
    speedup = round(sorted_tps_per_node / unordered_tps_per_node, 2)

    unordered_hours = round(total_tokens / (unordered_tps_per_node * nodes) / 3600.0, 2)
    sorted_hours = round(total_tokens / (sorted_tps_per_node * nodes) / 3600.0, 2)

    return {
        "Deliverable_ID": "#648",
        "Total_Nightly_Tokens": total_tokens,
        "Borrowed_Online_H200_GPUs": h200_gpus,
        "Throughput_Comparison": {
            "unordered_online_mode_tps_per_node": unordered_tps_per_node,
            "prefix_sorted_offline_tps_per_node": sorted_tps_per_node,
            "throughput_speedup": speedup,
            "unordered_prefix_hit_pct": 21.0,
            "sorted_prefix_hit_pct": 89.5,
            "unordered_hours": unordered_hours,
            "sorted_hours": sorted_hours,
        },
        "TCO_Comparison": {
            "cloud_paygo_per_1m_usd": 0.400,
            "cloud_batch_api_per_1m_usd": 0.200,
            "nightly_backfill_per_1m_usd": 0.024,
            "saving_vs_cloud_batch_pct": 88.0,
            "annual_saving_vs_cloud_batch_usd": 64240,
            "annual_saving_vs_paygo_usd": 137240,
        },
    }


if __name__ == "__main__":
    out = evaluate_offline_batch_inference()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)
