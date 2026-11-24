"""Reproducible model for #603 Enterprise E-Commerce Visual Search, Shopping Agent & Nightly Catalog Sizer."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_ecommerce_ai(search_qps: int = 1500, chat_qps: int = 200, nightly_million_skus: float = 2.0) -> dict:
    search_l40s_gpus = max(2, math.ceil(search_qps / 190.0))
    chat_8b_l40s_gpus = max(2, math.ceil((chat_qps * 0.82) / 45.0))
    total_l40s_gpus = search_l40s_gpus + chat_8b_l40s_gpus
    chat_72b_h200_gpus = max(2, math.ceil((chat_qps * 0.18) / 6.0))

    vector_db_monthly_usd = 500
    unified_monthly_usd = total_l40s_gpus * 950 + chat_72b_h200_gpus * 3400 + vector_db_monthly_usd
    siloed_baseline_monthly_usd = round(search_qps * 120 + chat_qps * 850 + nightly_million_skus * 39000)

    return {
        "topic_id": 603,
        "search_qps": search_qps,
        "chat_qps": chat_qps,
        "nightly_million_skus": nightly_million_skus,
        "search_p95_ms": 24.0,
        "search_l40s_gpus": search_l40s_gpus,
        "chat_8b_l40s_gpus": chat_8b_l40s_gpus,
        "total_l40s_gpus": total_l40s_gpus,
        "chat_72b_h200_gpus": chat_72b_h200_gpus,
        "unified_monthly_usd": unified_monthly_usd,
        "siloed_baseline_monthly_usd": siloed_baseline_monthly_usd,
        "savings_pct": round((1.0 - unified_monthly_usd / siloed_baseline_monthly_usd) * 100, 1),
    }


def main() -> None:
    res = evaluate_ecommerce_ai()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
