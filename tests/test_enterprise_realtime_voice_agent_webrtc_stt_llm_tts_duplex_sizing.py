from experiments.enterprise_realtime_voice_agent_webrtc_stt_llm_tts_duplex_sizing.run import evaluate_voice_agent


def test_evaluate_voice_agent():
    r = evaluate_voice_agent(500, 1_500_000)
    assert r["p95_voice_to_voice_ms"] == 290
    assert r["total_l40s"] == 11
    assert r["total_h100"] == 8
    assert r["self_hosted_monthly_usd"] == 33400
    assert r["savings_pct"] > 85.0
