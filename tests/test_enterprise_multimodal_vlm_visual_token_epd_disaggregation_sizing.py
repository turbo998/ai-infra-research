from experiments.enterprise_multimodal_vlm_visual_token_epd_disaggregation_sizing.run import (
    evaluate_multimodal_vlm_epd_disaggregation_sizing,
)


def test_multimodal_vlm_epd_disaggregation_sizing():
    r = evaluate_multimodal_vlm_epd_disaggregation_sizing()
    assert r["deliverable_id"] == "#631"
    assert r["tokens_per_1080p_image"] == 2691
    assert r["visual_tokens_per_req"] == 8073
    assert r["ttft_speedup_x"] == 5.8
    assert r["tpot_speedup_x"] == 7.0
    assert r["epd_monthly_usd"] == 25400
    assert r["tco_savings_pct"] == 60.3
