from experiments.enterprise_media_gaming_video_gen_brand_safety_npc_sizing.run import evaluate_media_video_dit


def test_evaluate_media_video_dit():
    r = evaluate_media_video_dit(10_000, 2.0)
    assert r["speedup_ratio"] == 15.0
    assert r["video_h100_gpus"] == 32
    assert r["image_gate_l40s_gpus"] == 4
    assert r["self_hosted_monthly_usd"] == 77400
    assert r["cost_per_clip_self_hosted_usd"] == 0.258
    assert r["savings_pct"] == 87.1
