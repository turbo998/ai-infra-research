from experiments.enterprise_video_vlm_rtsp_keyframe_3d_token_pooling_sizer.run import (
    compute_compressed_video_tokens,
    evaluate_video_vlm_rtsp_sizer,
)


def test_enterprise_video_vlm_rtsp_keyframe_3d_token_pooling_sizer() -> None:
    res = evaluate_video_vlm_rtsp_sizer()
    assert res["Deliverable_ID"] == "#661"
    assert compute_compressed_video_tokens(3600, 1.0, 0.88, 144) == 62208
    v1h = res["One_Hour_1080p_Video"]
    assert v1h["token_reduction_pct"] == 99.36
    rtsp = res["RTSP_500_Streams_Sizing"]
    assert rtsp["optimized_edge_l40s_gpus"] == 8
    assert rtsp["optimized_core_h200_gpus"] == 8
    assert rtsp["cost_saving_pct"] == 91.8
