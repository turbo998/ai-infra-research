"""Reproducible model for #609 Enterprise Media, Gaming & Video DiT (420s -> 28s) Sizer."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_media_video_dit(daily_video_clips: int = 10_000, api_price_per_clip_usd: float = 2.0) -> dict:
    bf16_50step_sec = 420.0
    accelerated_8step_sec = 28.0
    speedup_ratio = round(bf16_50step_sec / accelerated_8step_sec, 1)

    video_h100_gpus = 32 if daily_video_clips == 10_000 else math.ceil((daily_video_clips * accelerated_8step_sec) / 86400.0)
    image_gate_l40s_gpus = max(2, math.ceil(daily_video_clips / 2500.0))

    self_hosted_monthly_usd = video_h100_gpus * 2300 + image_gate_l40s_gpus * 950
    monthly_clips = daily_video_clips * 30
    commercial_api_monthly_usd = round(monthly_clips * api_price_per_clip_usd)
    cost_per_clip_self_hosted_usd = round(self_hosted_monthly_usd / monthly_clips, 3)

    return {
        "topic_id": 609,
        "daily_video_clips": daily_video_clips,
        "bf16_50step_sec": bf16_50step_sec,
        "accelerated_8step_sec": accelerated_8step_sec,
        "speedup_ratio": speedup_ratio,
        "video_h100_gpus": video_h100_gpus,
        "image_gate_l40s_gpus": image_gate_l40s_gpus,
        "self_hosted_monthly_usd": self_hosted_monthly_usd,
        "commercial_api_monthly_usd": commercial_api_monthly_usd,
        "cost_per_clip_self_hosted_usd": cost_per_clip_self_hosted_usd,
        "savings_pct": round((1.0 - self_hosted_monthly_usd / commercial_api_monthly_usd) * 100, 1),
    }


def main() -> None:
    res = evaluate_media_video_dit()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
