from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "native_speech_to_speech_moshi_qwen_omni_vs_cascaded_asr_llm_tts_selection",
        "title": "Native Speech-to-Speech (Moshi / Qwen2.5-Omni / GLM-4-Voice) vs. Cascaded ASR + LLM + TTS Selection Guide",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "concurrent_voice_sessions": 200,
            "avg_session_duration_sec": 180,
            "audio_codec_frame_rate_hz": 12.5,
            "gpu_sku": "NVIDIA H100 80GB",
        },
        "tiers": [
            {
                "tier": "Tier 1: Non-Streaming Turn-Based Cascade (Batch Whisper-v3 -> 70B LLM -> Sentence TTS)",
                "e2e_voice_response_latency_ms": 2150.0,
                "enterprise_rag_tool_accuracy_pct": 96.5,
                "paralinguistic_emotion_retention_pct": 5.0,
                "concurrent_sessions_per_h100": 18,
                "recommended_scenario": "Legacy offline voice transcription & asynchronous voice notes",
            },
            {
                "tier": "Tier 2: Streaming Clause-Pipelined Cascade (Streaming ASR + vLLM 70B + CosyVoice2 Streaming)",
                "e2e_voice_response_latency_ms": 480.0,
                "enterprise_rag_tool_accuracy_pct": 96.8,
                "paralinguistic_emotion_retention_pct": 18.0,
                "concurrent_sessions_per_h100": 42,
                "recommended_scenario": "Regulated banking/healthcare IVR requiring strict text audit logs & hotword dicts",
            },
            {
                "tier": "Tier 3: Pure Native Speech-to-Speech (Moshi / GLM-4-Voice with Mimi 12.5Hz RVQ Audio Tokens)",
                "e2e_voice_response_latency_ms": 195.0,
                "enterprise_rag_tool_accuracy_pct": 78.5,
                "paralinguistic_emotion_retention_pct": 95.0,
                "concurrent_sessions_per_h100": 36,
                "recommended_scenario": "Emotional companionship, roleplay & ultra-low-latency full-duplex chat",
            },
            {
                "tier": "Tier 4 (Recommended): Hybrid Thinker-Talker Omni Architecture (Qwen2.5-Omni + Grounded Tool/RAG Router)",
                "e2e_voice_response_latency_ms": 235.0,
                "enterprise_rag_tool_accuracy_pct": 96.4,
                "paralinguistic_emotion_retention_pct": 92.0,
                "concurrent_sessions_per_h100": 48,
                "recommended_scenario": "Enterprise real-time multimodal voice assistants needing both <250ms prosody & 96%+ tool accuracy",
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
