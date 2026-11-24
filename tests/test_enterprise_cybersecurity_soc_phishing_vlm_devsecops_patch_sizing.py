from experiments.enterprise_cybersecurity_soc_phishing_vlm_devsecops_patch_sizing.run import evaluate_cybersecurity_devsecops


def test_evaluate_cybersecurity_devsecops():
    r = evaluate_cybersecurity_devsecops(50_000, 15_000)
    assert r["quishing_l40s_gpus"] == 2
    assert r["devsecops_h200_gpus"] == 4
    assert r["self_hosted_monthly_usd"] == 15500
    assert r["savings_pct"] == 89.3
    assert r["visual_quishing_detection_pct"] == 98.8
    assert r["verified_pr_merge_rate_pct"] == 91.2
