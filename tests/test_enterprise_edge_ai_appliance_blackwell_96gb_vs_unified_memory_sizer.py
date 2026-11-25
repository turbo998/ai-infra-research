from experiments.enterprise_edge_ai_appliance_blackwell_96gb_vs_unified_memory_sizer.run import (
    evaluate_edge_ai_appliances,
)


def test_enterprise_edge_ai_appliance_blackwell_96gb_vs_unified_memory_sizer() -> None:
    res = evaluate_edge_ai_appliances(500)
    assert res["Deliverable_ID"] == "#664"
    blk = res["Appliance_Specs"]["RTX_PRO_6000_Blackwell_96GB"]
    assert blk["bandwidth_gbs"] == 1792
    assert blk["b16_4k_ttft_ms"] == 240
    assert blk["b16_ttft_speedup_vs_unified"] == 20.2
    tco = res["Cloud_Edge_500_Branches_TCO"]
    assert tco["tco_saving_pct"] == 71.2
