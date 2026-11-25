from experiments.enterprise_llm_observability_otel_dcgm_itl_stall_tree.run import (
    compute_masked_tpot_avg,
    evaluate_llm_observability,
)


def test_enterprise_llm_observability_otel_dcgm_itl_stall_tree() -> None:
    res = evaluate_llm_observability()
    assert res["Deliverable_ID"] == "#653"
    assert compute_masked_tpot_avg(5.8, 825.0, 100) == 13.99
    assert sum(res["Root_Cause_Distribution_Pct"].values()) == 100.0
    ts = res["Telemetry_Storage_100RPS"]
    assert ts["storage_reduction_pct"] == 84.6
    assert ts["anomaly_catch_rate_pct"] == 100.0
