from experiments.enterprise_media_live_sports_highlight_vlm_autoreframe_sizing.run import build_summary


def test_enterprise_media_live_sports_highlight_vlm_autoreframe_sizing() -> None:
    s = build_summary()
    assert s["publish_speedup_vs_manual"] == 317.65
    assert s["gpu_reduction_pct_vs_dense_72b"] == 96.0
    assert s["schemes"]["audio_motion_trigger_plus_7b_vlm_plus_yolo_reframe"]["publish_latency_seconds"] == 8.5
    assert s["schemes"]["audio_motion_trigger_plus_7b_vlm_plus_yolo_reframe"]["required_h200_gpus"] == 16
