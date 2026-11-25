from experiments.enterprise_billion_token_offline_batch_inference_prefix_sort_tco.run import (
    evaluate_offline_batch_inference,
)


def test_enterprise_billion_token_offline_batch_inference_prefix_sort_tco() -> None:
    res = evaluate_offline_batch_inference()
    assert res["Deliverable_ID"] == "#648"
    tp = res["Throughput_Comparison"]
    assert tp["throughput_speedup"] == 3.02
    assert tp["sorted_prefix_hit_pct"] == 89.5
    assert tp["sorted_hours"] == 8.22 or tp["sorted_hours"] < 8.5
    tco = res["TCO_Comparison"]
    assert tco["nightly_backfill_per_1m_usd"] == 0.024
    assert tco["saving_vs_cloud_batch_pct"] == 88.0
