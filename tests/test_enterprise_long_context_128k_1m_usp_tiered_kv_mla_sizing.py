from experiments.enterprise_long_context_128k_1m_usp_tiered_kv_mla_sizing.run import (
    evaluate_long_context_128k_1m_usp_tiered_kv_sizing,
)


def test_long_context_128k_1m_usp_tiered_kv_sizing():
    r = evaluate_long_context_128k_1m_usp_tiered_kv_sizing()
    assert r["deliverable_id"] == "#628"
    assert r["usp_8x2_256k_ttft_sec"] == 1.82
    assert r["ttft_speedup_x"] == 10.66
    assert r["l2_ddr5_restore_128k_ms"] == 97
    assert r["restore_speedup_x"] == 99.0
    assert r["tiered_kv_gpus"] == 12
    assert r["gpu_savings_pct"] == 62.5
