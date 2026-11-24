from experiments.enterprise_document_intelligence_idp_ocr_vlm_pipeline_sizing.run import evaluate_idp_pipeline


def test_evaluate_idp_pipeline():
    r = evaluate_idp_pipeline(10.0)
    assert r["blended_cost_per_1k_usd"] == 0.528
    assert r["routed_monthly_usd"] == 5280
    assert r["savings_vs_cloud_pct"] == 94.7
    assert r["l40s_gpus"] == 1
    assert r["h100_gpus"] == 2
