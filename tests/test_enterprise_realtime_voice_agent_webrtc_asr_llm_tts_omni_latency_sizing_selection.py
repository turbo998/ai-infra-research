from pathlib import Path
from experiments.enterprise_realtime_voice_agent_webrtc_asr_llm_tts_omni_latency_sizing_selection.run import build_results


def test_enterprise_realtime_voice_agent_webrtc_asr_llm_tts_omni_latency_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_realtime_voice_agent_webrtc_asr_llm_tts_omni_latency_sizing_selection"
    archs = res["architectures"]
    assert len(archs) == 4
    assert archs[2]["end_to_end_ttfa_p99_ms"] <= res["human_conversational_ttfa_sla_ms"]
    assert archs[3]["end_to_end_ttfa_p99_ms"] < archs[2]["end_to_end_ttfa_p99_ms"]
    assert archs[2]["cost_per_hour_active_call_usd"] < archs[0]["cost_per_hour_active_call_usd"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-realtime-voice-agent-webrtc-asr-llm-tts-omni-latency-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-realtime-voice-agent-webrtc-asr-llm-tts-omni-latency-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/realtime-voice-agent-webrtc-omni-ttfa-sizer-lab.html").exists()
