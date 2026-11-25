from experiments.enterprise_gb200_nvl72_supernode_vs_hgx_moe_ep72_sizing.run import (
    evaluate_gb200_nvl72_vs_hgx_moe_sizing,
)


def test_gb200_nvl72_vs_hgx_moe_sizing():
    r = evaluate_gb200_nvl72_vs_hgx_moe_sizing()
    assert r["deliverable_id"] == "#632"
    assert r["gb200_nvl72_goodput_tok_s_gpu"] == 7240
    assert r["throughput_gain_vs_h200_x"] == 2.92
    assert r["gb200_nvl72_tpot_ms"] == 4.8
    assert r["grace_c2c_128k_restore_ms"] == 14
    assert r["unit_cost_reduction_pct"] == 55.7
