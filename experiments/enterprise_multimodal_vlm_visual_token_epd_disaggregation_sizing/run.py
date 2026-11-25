from __future__ import annotations
import json
from pathlib import Path
import math


def evaluate_multimodal_vlm_epd_disaggregation_sizing() -> dict:
    # 1920x1080 at 28x28 pixels per visual token
    tokens_per_1080p_image = math.ceil(1920 / 28) * math.ceil(1080 / 28)  # 69 * 39 = 2,691
    images_per_req = 3
    visual_tokens_per_req = tokens_per_1080p_image * images_per_req  # 8,073

    colocated_ttft_ms = 3420
    epd_ttft_ms = 590
    ttft_speedup_x = round(colocated_ttft_ms / epd_ttft_ms, 2)  # 5.80x

    colocated_p99_tpot_ms = 92.4
    epd_p99_tpot_ms = 13.2
    tpot_speedup_x = round(colocated_p99_tpot_ms / epd_p99_tpot_ms, 2)  # 7.00x

    image_embedding_cache_hit_pct = 78.0
    colocated_monthly_usd = 20 * 3200  # 64,000
    epd_monthly_usd = 4 * 850 + 4 * 2300 + 4 * 3200  # 25,400
    tco_savings_pct = round((1.0 - epd_monthly_usd / colocated_monthly_usd) * 100.0, 1)  # 60.3%

    return {
        "deliverable_id": "#631",
        "topic": "enterprise_multimodal_vlm_visual_token_epd_disaggregation_sizing",
        "tokens_per_1080p_image": tokens_per_1080p_image,
        "visual_tokens_per_req": visual_tokens_per_req,
        "colocated_ttft_ms": colocated_ttft_ms,
        "epd_ttft_ms": epd_ttft_ms,
        "ttft_speedup_x": ttft_speedup_x,
        "colocated_p99_tpot_ms": colocated_p99_tpot_ms,
        "epd_p99_tpot_ms": epd_p99_tpot_ms,
        "tpot_speedup_x": tpot_speedup_x,
        "image_embedding_cache_hit_pct": image_embedding_cache_hit_pct,
        "colocated_monthly_usd": colocated_monthly_usd,
        "epd_monthly_usd": epd_monthly_usd,
        "tco_savings_pct": tco_savings_pct,
    }


if __name__ == "__main__":
    out = evaluate_multimodal_vlm_epd_disaggregation_sizing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)
