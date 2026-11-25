from experiments.enterprise_realtime_voice_agent_300ms_cascaded_vs_native_s2s.run import (
    compute_cascaded_ttfa_ms,
    size_voice_fleet,
)


def test_enterprise_realtime_voice_agent_300ms_cascaded_vs_native_s2s() -> None:
    res = size_voice_fleet(1000)
    assert res["Deliverable_ID"] == "#643"
    assert compute_cascaded_ttfa_ms() == 265
    cas = res["Streaming_Cascaded"]
    assert cas["total_h200_gpus"] == 14
    assert cas["cost_per_min_usd"] == 0.014
    assert res["Native_S2S"]["total_h200_gpus"] == 32
    assert res["Native_S2S"]["gpu_saving_by_cascaded_pct"] == 56.2
