"""Reproducible model for #581 Real-Time Voice Agent 290ms Latency & GPU Sizer."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_voice_agent(concurrent_calls: int = 500, monthly_minutes: int = 1_500_000) -> dict:
    latency_breakdown_ms = {
        "vad_semantic_eou_ms": 80,
        "streaming_stt_tail_ms": 55,
        "llm_prefix_first_clause_ms": 65,
        "chunked_tts_first_frame_ms": 65,
        "webrtc_rtp_ms": 25,
    }
    p95_voice_to_voice_ms = sum(latency_breakdown_ms.values())

    active_stt_streams = concurrent_calls * 0.38
    active_tts_streams = concurrent_calls * 0.42
    llm_qps = round(concurrent_calls / 12.0, 2)

    stt_l40s = math.ceil(active_stt_streams / 45.0)
    tts_l40s = math.ceil(active_tts_streams / 38.0)
    total_l40s = stt_l40s + tts_l40s
    llm_h100_replicas = math.ceil(llm_qps / 11.5)
    total_h100 = llm_h100_replicas * 2

    sfu_monthly_usd = math.ceil(concurrent_calls / 100.0) * 910
    self_hosted_monthly_usd = total_l40s * 950 + total_h100 * 2300 + sfu_monthly_usd
    managed_realtime_api_monthly_usd = round(monthly_minutes * 0.18)
    crossover_minutes = round(self_hosted_monthly_usd / 0.18)

    return {
        "topic_id": 581,
        "concurrent_calls": concurrent_calls,
        "monthly_minutes": monthly_minutes,
        "latency_breakdown_ms": latency_breakdown_ms,
        "p95_voice_to_voice_ms": p95_voice_to_voice_ms,
        "stt_l40s": stt_l40s,
        "tts_l40s": tts_l40s,
        "total_l40s": total_l40s,
        "total_h100": total_h100,
        "self_hosted_monthly_usd": self_hosted_monthly_usd,
        "managed_realtime_api_monthly_usd": managed_realtime_api_monthly_usd,
        "cost_per_minute_self_hosted_usd": round(self_hosted_monthly_usd / monthly_minutes, 4),
        "savings_pct": round((1 - self_hosted_monthly_usd / managed_realtime_api_monthly_usd) * 100, 1),
        "crossover_minutes": crossover_minutes,
    }


def main() -> None:
    res = evaluate_voice_agent()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
