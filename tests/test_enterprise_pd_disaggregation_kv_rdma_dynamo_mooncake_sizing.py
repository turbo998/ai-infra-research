from experiments.enterprise_pd_disaggregation_kv_rdma_dynamo_mooncake_sizing.run import build_summary


def test_enterprise_pd_disaggregation_kv_rdma_dynamo_mooncake_sizing() -> None:
    s = build_summary()
    pd = s["schemes"]["pd_disaggregation_dynamo_mooncake_400g_rdma"]
    col = s["schemes"]["colocated_continuous_batching"]
    assert pd["p99_tpot_ms"] < 15.0
    assert pd["sla_goodput_pct"] > 99.0
    assert pd["required_h200_gpus"] == 16
    assert col["required_h200_gpus"] == 32
    assert s["kv_cache_kb_per_token"]["llama70b_gqa8_fp8"] == 163.84
    assert s["kv_cache_kb_per_token"]["deepseek671b_mla_fp8"] == 35.14
