from __future__ import annotations
import json
from pathlib import Path


def evaluate_voice_agent_vad_semantic_eou() -> dict:
    # Workload: Real-time conversational Voice Agent evaluating turn-taking latency (ms) and mid-sentence false interruption rate (%)
    tiers = [
        {
            "tier_id": "tier1_aggressive_300ms_fixed_acoustic_vad_silence_threshold",
            "name": "Aggressive `300 ms` Fixed Acoustic VAD Silence Threshold (`Silero-VAD` Only)",
            "turn_taking_response_latency_ms": 540.0,
            "mid_sentence_false_interruption_pct": 34.5,
            "wasted_speculative_gpu_pct": 28.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Triggers LLM generation whenever the user pauses for 300ms to think ('I want to book a flight... um... to Tokyo'), interrupting 34.5% of turns!"
        },
        {
            "tier_id": "tier2_conservative_1100ms_fixed_acoustic_vad_silence_threshold",
            "name": "Conservative `1,100 ms` Fixed Acoustic VAD Silence Threshold",
            "turn_taking_response_latency_ms": 1380.0,
            "mid_sentence_false_interruption_pct": 4.8,
            "wasted_speculative_gpu_pct": 0.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Reduces false interruptions, but forces a sluggish 1.1s dead-air silence wait before even starting LLM prefill (1,380ms total response lag)!"
        },
        {
            "tier_id": "tier3_acoustic_vad_plus_135m_semantic_eou_dynamic_timeout",
            "name": "Acoustic VAD + `135M` Semantic End-of-Utterance (`EoU`) Classifier (`150 ms` Fast vs `1,200 ms` Hold)",
            "turn_taking_response_latency_ms": 420.0,
            "mid_sentence_false_interruption_pct": 2.1,
            "wasted_speculative_gpu_pct": 0.0,
            "action": "Confirm",
            "status": "pass",
            "note": "Evaluates transcript syntax + prosody at 150ms silence: fires immediately if complete (`P(EoU)>0.85`) or extends timeout to 1,200ms if mid-clause!"
        },
        {
            "tier_id": "tier4_semantic_eou_plus_speculative_prefill_on_pause_with_prefix_cache",
            "name": "Acoustic VAD + Semantic `EoU` + Speculative LLM Prefill During Silence (`APC` KV Commit)",
            "turn_taking_response_latency_ms": 235.0,
            "mid_sentence_false_interruption_pct": 2.1,
            "wasted_speculative_gpu_pct": 3.2,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Pre-computes LLM prefill KV into Prefix Cache during the 150ms pause and releases first audio token the instant EoU confirms -> 235ms response & 2.1% false cut!"
        },
    ]
    return {
        "experiment_id": "voice_agent_streaming_vad_semantic_eou_turn_detection_speculative_prefill_selection",
        "round": 318,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "pipeline": "Streaming ASR / Audio-LLM + Turn Detector + TTS",
            "human_natural_turn_gap_ms": 250.0
        },
        "tiers": tiers,
        "summary": {
            "latency_speedup_vs_conservative_x": round(1380.0 / 235.0, 2),
            "false_interruption_reduction_x": round(34.5 / 2.1, 2),
            "recommended_latency_ms": 235.0,
            "recommended_tier": "tier4_semantic_eou_plus_speculative_prefill_on_pause_with_prefix_cache"
        }
    }


if __name__ == "__main__":
    data = evaluate_voice_agent_vad_semantic_eou()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
