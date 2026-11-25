from experiments.enterprise_parallelism_tp_pp_dp_ep_dense_moe_acceleration_sizing.run import (
    evaluate_parallelism_dense_moe_acceleration_sizing,
)


def test_parallelism_dense_moe_acceleration_sizing():
    r = evaluate_parallelism_dense_moe_acceleration_sizing()
    assert r["deliverable_id"] == "#624"
    assert r["dp16_ep16_deepep_mtp_tpot_ms"] == 11.4
    assert r["tpot_speedup_x"] == 3.04
    assert r["throughput_gain_x"] == 4.77
    assert r["mtp_auto_bypass_batch_threshold"] == 32
    assert r["deepep_exposed_comm_pct"] == 1.9
