from experiments.enterprise_milestone_635_1024gpu_ai_factory_bom_3fs_dlc_tco.run import (
    evaluate_milestone_635_1024gpu_ai_factory_tco,
)


def test_milestone_635_1024gpu_ai_factory_tco():
    r = evaluate_milestone_635_1024gpu_ai_factory_tco()
    assert r["deliverable_id"] == "#635"
    assert r["rail_switches_400g"] == 48
    assert r["async_dcp_checkpoint_stall_sec"] == 2.4
    assert r["checkpoint_speedup_x"] == 100.0
    assert r["dlc_liquid_pue"] == 1.11
    assert r["net_3yr_savings_usd"] == 28400000
    assert r["tco_reduction_pct"] == 38.3
