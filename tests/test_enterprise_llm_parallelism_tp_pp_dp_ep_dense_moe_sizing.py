from experiments.enterprise_llm_parallelism_tp_pp_dp_ep_dense_moe_sizing.run import build_summary


def test_enterprise_llm_parallelism_tp_pp_dp_ep_dense_moe_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["attention_dp16_expert_ep16_deepep_fp8_mtp"]
    bad = s["schemes"]["cross_node_pure_tp16_bf16"]
    assert opt["goodput_tokens_per_sec_per_gpu"] == 3890
    assert opt["effective_p99_tpot_ms"] == 6.2
    assert opt["mla_kv_duplication_factor"] == 1
    assert bad["mla_kv_duplication_factor"] == 16
    assert s["throughput_gain_vs_cross_node_tp16"] == 5.48
