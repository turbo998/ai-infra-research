from experiments.enterprise_insurance_claims_multimodal_fraud_stp_adjudication_sizing.run import build_summary


def test_enterprise_insurance_claims_multimodal_fraud_stp_adjudication_sizing() -> None:
    s = build_summary()
    assert s["gpu_reduction_pct_vs_monolithic_72b"] == 87.5
    assert s["cost_reduction_pct_vs_monolithic_72b"] == 93.7
    assert s["schemes"]["dinov2_fraud_plus_yolo_damage_plus_policy_70b_stp"]["fraud_detection_recall_pct"] == 98.8
    assert s["schemes"]["dinov2_fraud_plus_yolo_damage_plus_policy_70b_stp"]["stp_auto_adjudication_pct"] == 82.0
