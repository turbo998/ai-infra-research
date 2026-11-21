from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "enterprise_voice_contact_center_realtime_api_vs_self_hosted_webrtc_tco_selection",
        "title": "Enterprise AI Voice Contact Center Architecture Selection Guide: Azure OpenAI Realtime API (GPT-4o-Audio) vs. Self-Hosted WebRTC + Fast-ASR + SGLang + Streaming TTS",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "monthly_call_minutes_millions": 10.0,
            "peak_concurrent_voice_streams": 1000,
            "human_conversational_m2e_sla_ms": 500.0,
        },
        "tiers": [
            {
                "tier": "Tier 1: Legacy HTTP REST Turn-by-Turn (Wait for Full LLM Completion Before TTS Synthesis)",
                "p95_mouth_to_ear_latency_ms": 2850.0,
                "pre_tts_text_guardrail_coverage_pct": 100.0,
                "cost_per_call_minute_usd": 0.085,
                "monthly_tco_usd": 850000,
                "recommended_scenario": "Unusable for natural phone conversations due to 2.85s awkward silence gaps",
            },
            {
                "tier": "Tier 2: 100% Commercial Native Speech-to-Speech API (GPT-4o-Realtime Audio)",
                "p95_mouth_to_ear_latency_ms": 320.0,
                "pre_tts_text_guardrail_coverage_pct": 0.0,
                "cost_per_call_minute_usd": 0.220,
                "monthly_tco_usd": 2200000,
                "recommended_scenario": "Ideal for low-volume VIP emotional concierge, prohibitive ($2.2M/mo) at 10M minutes/month",
            },
            {
                "tier": "Tier 3: Streaming WebRTC (LiveKit/Pipecat) + Semantic VAD + Fast-ASR + SGLang (6-Word Chunk) + Streaming TTS",
                "p95_mouth_to_ear_latency_ms": 435.0,
                "pre_tts_text_guardrail_coverage_pct": 100.0,
                "cost_per_call_minute_usd": 0.032,
                "monthly_tco_usd": 320000,
                "recommended_scenario": "High-volume regulated banking/telecom contact center standard with 100% pre-TTS text compliance",
            },
            {
                "tier": "Tier 4 (Recommended): Hybrid 90% Streaming WebRTC Pipeline + 10% Thinker-Talker Native S2S Escalation",
                "p95_mouth_to_ear_latency_ms": 410.0,
                "pre_tts_text_guardrail_coverage_pct": 100.0,
                "cost_per_call_minute_usd": 0.038,
                "monthly_tco_usd": 380000,
                "recommended_scenario": "Flagship contact center architecture hitting 410ms M2E and saving $1.82M/month (-82.7%) vs. Tier 2",
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
