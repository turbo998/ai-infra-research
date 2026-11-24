from experiments.enterprise_healthcare_clinical_scribe_fhir_pacs_vlm_sizing.run import evaluate_clinical_ai


def test_evaluate_clinical_ai():
    r = evaluate_clinical_ai(2000, 450)
    assert r["concurrent_exam_rooms"] == 650
    assert r["asr_l40s_gpus"] == 6
    assert r["soap_fhir_h200_gpus"] == 4
    assert r["self_hosted_monthly_usd"] == 19300
    assert r["cost_per_physician_usd"] == 9.65
    assert r["savings_pct"] == 97.9
