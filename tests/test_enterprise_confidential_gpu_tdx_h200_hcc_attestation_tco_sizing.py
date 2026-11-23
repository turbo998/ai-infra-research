from experiments.enterprise_confidential_gpu_tdx_h200_hcc_attestation_tco_sizing.run import build_summary


def test_enterprise_confidential_gpu_tdx_h200_hcc_attestation_tco_sizing() -> None:
    s = build_summary()
    assert s["token_cost_reduction_pct_vs_airgapped"] == 81.3
    assert s["three_year_tco_reduction_pct_vs_airgapped"] == 70.7
    assert s["schemes"]["cloud_confidential_gpu_tdx_h200_hcc_skr"]["throughput_overhead_pct"] == 6.8
    assert s["decode_70b_hcc_overhead_pct"] == 3.4
