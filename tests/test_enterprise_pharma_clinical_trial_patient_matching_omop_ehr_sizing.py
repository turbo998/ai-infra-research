from experiments.enterprise_pharma_clinical_trial_patient_matching_omop_ehr_sizing.run import build_summary


def test_enterprise_pharma_clinical_trial_patient_matching_omop_ehr_sizing() -> None:
    s = build_summary()
    assert s["gpu_reduction_pct_vs_brute_70b"] == 97.5
    assert s["recall_gain_pts_vs_sql_only"] == 58.8
    assert s["precision_gain_pts_vs_brute_70b"] == 21.8
    assert s["schemes"]["omop_sql_plus_8b_negation_feature_store_plus_70b_verifier"]["required_h200_gpus"] == 12
