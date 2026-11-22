from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    architectures = [
        {
            "architecture": "Tier 1: Legacy REST/WebSocket Turn-Based Cascade (Batch Whisper ASR + Full-Sentence LLM + Batch TTS)",
            "vad_endpointing_ms": 650.0,
            "asr_p99_ms": 420.0,
            "llm_ttft_p99_ms": 380.0,
            "tts_first_chunk_p99_ms": 490.0,
            "webrtc_network_jitter_ms": 95.0,
            "end_to_end_ttfa_p99_ms": 2035.0,
            "barge_in_cancel_latency_ms": 680.0,
            "concurrent_streams_per_8gpu_node": 96,
            "cost_per_hour_active_call_usd": 0.42,
        },
        {
            "architecture": "Tier 2: Streaming WebSocket Cascade (Streaming Conformer ASR + Clause-Chunked LLM + Streaming TTS)",
            "vad_endpointing_ms": 320.0,
            "asr_p99_ms": 140.0,
            "llm_ttft_p99_ms": 165.0,
            "tts_first_chunk_p99_ms": 150.0,
            "webrtc_network_jitter_ms": 65.0,
            "end_to_end_ttfa_p99_ms": 840.0,
            "barge_in_cancel_latency_ms": 210.0,
            "concurrent_streams_per_8gpu_node": 240,
            "cost_per_hour_active_call_usd": 0.18,
        },
        {
            "architecture": "Tier 3: WebRTC Full-Duplex Disaggregated Pool (2x L40S ASR/TTS + 6x H200 LLM + Semantic VAD + 4-Token Micro-Chunk TTS)",
            "vad_endpointing_ms": 85.0,
            "asr_p99_ms": 38.0,
            "llm_ttft_p99_ms": 54.0,
            "tts_first_chunk_p99_ms": 36.0,
            "webrtc_network_jitter_ms": 22.0,
            "end_to_end_ttfa_p99_ms": 235.0,
            "barge_in_cancel_latency_ms": 28.0,
            "concurrent_streams_per_8gpu_node": 640,
            "cost_per_hour_active_call_usd": 0.068,
        },
        {
            "architecture": "Tier 4: Native Speech-to-Speech Omni Audio-Token Engine (12.5 Hz Acoustic/Semantic Codec + WebRTC Edge SFU)",
            "vad_endpointing_ms": 40.0,
            "asr_p99_ms": 0.0,
            "llm_ttft_p99_ms": 92.0,
            "tts_first_chunk_p99_ms": 26.0,
            "webrtc_network_jitter_ms": 20.0,
            "end_to_end_ttfa_p99_ms": 178.0,
            "barge_in_cancel_latency_ms": 18.0,
            "concurrent_streams_per_8gpu_node": 480,
            "cost_per_hour_active_call_usd": 0.092,
        },
    ]
    return {
        "experiment_id": "enterprise_realtime_voice_agent_webrtc_asr_llm_tts_omni_latency_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "human_conversational_ttfa_sla_ms": 250.0,
        "audio_codec_token_rate_hz": 12.5,
        "architectures": architectures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_realtime_voice_agent_webrtc_asr_llm_tts_omni_latency_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
