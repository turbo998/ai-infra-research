from experiments.enterprise_conversational_bi_fabric_genie_cortex_mdl_ast_sizing.run import build_summary


def test_enterprise_conversational_bi_fabric_genie_cortex_mdl_ast_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["mdl_semantic_layer_golden_sql_sqlglot_ast_gate"]
    assert opt["execution_accuracy_pct"] == 96.8
    assert opt["runaway_query_rate_pct"] == 0.0
    assert opt["p95_latency_sec"] == 1.45
    assert opt["required_h200_gpus"] == 8
    assert s["accuracy_gain_pts_vs_raw_ddl"] == 42.6
