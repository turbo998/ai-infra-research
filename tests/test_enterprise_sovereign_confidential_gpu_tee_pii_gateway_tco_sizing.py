from experiments.enterprise_sovereign_confidential_gpu_tee_pii_gateway_tco_sizing.run import build_summary


def test_enterprise_sovereign_confidential_gpu_tee_pii_gateway_tco_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["reversible_pii_vault_75pct_plus_cloud_h200_tee_25pct"]
    assert opt["multi_entity_reasoning_accuracy_pct"] == 96.4
    assert opt["tee_batch16_throughput_overhead_pct"] == 5.8
    assert opt["three_year_tco_usd"] == 580000
    assert s["cost_reduction_factor_vs_air_gapped"] == 4.32
