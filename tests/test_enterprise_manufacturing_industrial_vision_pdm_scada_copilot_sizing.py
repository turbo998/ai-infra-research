from experiments.enterprise_manufacturing_industrial_vision_pdm_scada_copilot_sizing.run import evaluate_manufacturing_ai


def test_evaluate_manufacturing_ai():
    r = evaluate_manufacturing_ai(20, 80)
    assert r["total_lines"] == 1600
    assert r["l1_yolo_latency_ms"] == 8.5
    assert r["l2_plant_l40s_gpus"] == 40
    assert r["l3_central_h200_gpus"] == 4
    assert r["unified_l2_l3_monthly_usd"] == 51600
    assert r["savings_vs_cloud_stream_pct"] == 86.4
