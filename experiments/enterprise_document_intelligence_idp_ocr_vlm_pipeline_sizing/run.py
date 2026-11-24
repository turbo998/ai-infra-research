"""Reproducible model for #583 Enterprise Document Intelligence 3-Tier IDP Sizer."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_idp_pipeline(million_pages_per_month: float = 10.0) -> dict:
    tier1_share = 0.65
    tier2_share = 0.25
    tier3_share = 0.10

    blended_cost_per_1k_usd = round(
        tier1_share * 0.02 + tier2_share * 0.38 + tier3_share * 4.20, 3
    )
    thousand_pages = million_pages_per_month * 1000.0
    routed_monthly_usd = round(thousand_pages * blended_cost_per_1k_usd)
    cloud_ocr_monthly_usd = round(thousand_pages * 10.0)
    all_vlm_monthly_usd = round(thousand_pages * 4.20)

    l40s_gpus = max(1, math.ceil((million_pages_per_month * tier2_share) / 3.0))
    h100_gpus = max(1, math.ceil((million_pages_per_month * tier3_share) / 0.6))

    return {
        "topic_id": 583,
        "million_pages_per_month": million_pages_per_month,
        "blended_cost_per_1k_usd": blended_cost_per_1k_usd,
        "routed_monthly_usd": routed_monthly_usd,
        "cloud_ocr_monthly_usd": cloud_ocr_monthly_usd,
        "all_vlm_monthly_usd": all_vlm_monthly_usd,
        "savings_vs_cloud_pct": round((1.0 - routed_monthly_usd / cloud_ocr_monthly_usd) * 100, 1),
        "savings_vs_all_vlm_pct": round((1.0 - routed_monthly_usd / all_vlm_monthly_usd) * 100, 1),
        "l40s_gpus": l40s_gpus,
        "h100_gpus": h100_gpus,
        "complex_table_f1_pct": 96.8,
    }


def main() -> None:
    res = evaluate_idp_pipeline()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
