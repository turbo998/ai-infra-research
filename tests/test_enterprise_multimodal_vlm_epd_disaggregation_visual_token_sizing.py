from experiments.enterprise_multimodal_vlm_epd_disaggregation_visual_token_sizing.run import build_summary


def test_enterprise_multimodal_vlm_epd_disaggregation_visual_token_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["heterogeneous_epd_3pool_4l40s_4h100_8h200"]
    assert s["visual_token_math"]["single_4k_12tile_plus_thumb_tokens"] == 3328
    assert opt["p99_tpot_ms"] == 12.4
    assert opt["sla_goodput_pct"] == 99.3
    assert opt["cost_per_1k_requests_usd"] == 0.81
