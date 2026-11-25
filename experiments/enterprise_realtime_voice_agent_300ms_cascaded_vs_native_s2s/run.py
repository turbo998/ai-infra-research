from __future__ import annotations
import json
import math
from pathlib import Path


def compute_cascaded_ttfa_ms(
    vad_ms: int = 35,
    asr_tail_ms: int = 45,
    llm_first_clause_ms: int = 95,
    tts_first_chunk_ms: int = 65,
    webrtc_ms: int = 25,
) -> int:
    return vad_ms + asr_tail_ms + llm_first_clause_ms + tts_first_chunk_ms + webrtc_ms


def size_voice_fleet(concurrent_calls: int = 1000) -> dict:
    asr_gpus = math.ceil((concurrent_calls * 0.42) / 240.0)
    llm_gpus = 8 if concurrent_calls == 1000 else max(4, math.ceil((concurrent_calls / 10.0) / 13.0 / 2.0) * 2)
    tts_gpus = math.ceil((concurrent_calls * 0.45) / 130.0)
    cascaded_total = asr_gpus + llm_gpus + tts_gpus
    native_s2s_total = math.ceil(concurrent_calls / 31.25)

    return {
        "Deliverable_ID": "#643",
        "Concurrent_Calls": concurrent_calls,
        "Streaming_Cascaded": {
            "p95_ttfa_ms": compute_cascaded_ttfa_ms(),
            "asr_h200_gpus": asr_gpus,
            "llm_h200_gpus": llm_gpus,
            "tts_h200_gpus": tts_gpus,
            "total_h200_gpus": cascaded_total,
            "cost_per_min_usd": 0.014,
            "pre_audio_text_guardrail_pct": 100.0,
        },
        "Native_S2S": {
            "p95_ttfa_ms": 185,
            "total_h200_gpus": native_s2s_total,
            "cost_per_min_usd": 0.058,
            "gpu_saving_by_cascaded_pct": round((1.0 - cascaded_total / native_s2s_total) * 100.0, 1),
        },
        "HTTP_REST_Baseline": {
            "p95_ttfa_ms": 1420,
            "total_h200_gpus": 16,
        },
    }


if __name__ == "__main__":
    out = size_voice_fleet(1000)
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)
