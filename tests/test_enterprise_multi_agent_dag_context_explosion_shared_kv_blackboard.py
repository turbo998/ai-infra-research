from experiments.enterprise_multi_agent_dag_context_explosion_shared_kv_blackboard.run import (
    evaluate_multi_agent_dag_governance,
)


def test_enterprise_multi_agent_dag_context_explosion_shared_kv_blackboard() -> None:
    res = evaluate_multi_agent_dag_governance()
    assert res["Deliverable_ID"] == "#652"
    opt = res["Optimized_Blackboard_RoleAtTail_DAG"]
    assert opt["uncached_token_reduction_pct"] == 86.0
    assert opt["cross_step_kv_hit_pct"] == 91.5
    assert opt["p95_e2el_sec"] == 4.6
    assert opt["latency_reduction_pct"] == 75.0
    assert opt["cost_saving_pct"] == 82.9
