from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "realtime_voice_agent_webrtc_semantic_vad_streaming_tts_latency_selection",
        "title": "Real-Time Voice & Multimodal Agent Architecture: WebRTC, Semantic VAD, Streaming STT-LLM-TTS vs. S2S Omni Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "concurrent_voice_calls": 1000,
            "avg_call_duration_min": 4.5,
            "target_mouth_to_ear_sla_ms": 500.0,
        },
        "tiers": [
            {
                "tier_id": "tier1_tcp_websocket_batch_stt_full_sentence_llm_and_batch_tts",
                "name": "Tier 1: TCP WebSocket + Silence-Only VAD (800ms) + Batch Whisper STT + Full-Response LLM + Batch TTS",
                "p95_mouth_to_ear_latency_ms": 3420.0,
                "false_barge_in_or_ awkward_cutoff_rate_pct": 38.4,
                "audio_glitch_or_underrun_rate_pct": 14.2,
                "cost_per_voice_minute_usd": 0.145,
                "verdict": "Anti-Pattern: Waiting for full LLM response before TTS synthesis creates a 3.42-second awkward dead silence!",
            },
            {
                "tier_id": "tier2_webrtc_silero_vad_plus_sentence_chunked_streaming_cascade",
                "name": "Tier 2: WebRTC UDP + Acoustic VAD (500ms) + Streaming STT + Sentence-Chunked LLM-to-TTS",
                "p95_mouth_to_ear_latency_ms": 980.0,
                "false_barge_in_or_ awkward_cutoff_rate_pct": 21.0,
                "audio_glitch_or_underrun_rate_pct": 4.8,
                "cost_per_voice_minute_usd": 0.068,
                "verdict": "Sub-second latency (980ms), but pure acoustic VAD cuts off users mid-thought (21% false barge-in)",
            },
            {
                "tier_id": "tier3_pure_end_to_end_speech_to_speech_omni_audio_tokens",
                "name": "Tier 3: 100% Native Speech-to-Speech (`S2S Omni` 12.5Hz Audio Codec Tokens on Dedicated H200)",
                "p95_mouth_to_ear_latency_ms": 310.0,
                "false_barge_in_or_ awkward_cutoff_rate_pct": 4.2,
                "audio_glitch_or_underrun_rate_pct": 1.1,
                "cost_per_voice_minute_usd": 0.195,
                "verdict": "Fastest emotion-preserving response (310ms), but 5.7x more expensive ($0.195/min) and harder to audit tool calls",
            },
            {
                "tier_id": "tier4_webrtc_semantic_turn_detector_speculative_clause_tts_and_pd_llm",
                "name": "Tier 4: WebRTC + Semantic Turn-Detector VAD + Speculative Pre-Warm LLM (PD Split + MTP) + 6-Token Clause TTS",
                "p95_mouth_to_ear_latency_ms": 420.0,
                "false_barge_in_or_ awkward_cutoff_rate_pct": 1.8,
                "audio_glitch_or_underrun_rate_pct": 0.2,
                "cost_per_voice_minute_usd": 0.034,
                "verdict": "Recommended Enterprise Standard: 420ms P95 M2E (-8.1x vs Tier 1), 1.8% false cutoff, 100% text audit & $0.034/min",
            },
        ],
        "headline_findings": {
            "p95_mouth_to_ear_ms": 420.0,
            "latency_speedup_vs_tier1_x": 8.14,
            "false_barge_in_pct": 1.8,
            "cost_per_minute_usd": 0.034,
        },
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
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
