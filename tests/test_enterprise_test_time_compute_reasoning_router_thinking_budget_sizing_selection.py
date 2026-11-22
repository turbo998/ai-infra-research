from pathlib import Path
from experiments.enterprise_test_time_compute_reasoning_router_thinking_budget_sizing_selection.run import build_results


def test_enterprise_test_time_compute_reasoning_router_thinking_budget_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_test_time_compute_reasoning_router_thinking_budget_sizing_selection"
    strats = res["ttc_strategies"]
    assert len(strats) == 4
    assert strats[3]["gpus_required_at_20_rps"] <= strats[0]["gpus_required_at_20_rps"] * 0.25
    assert strats[3]["complex_task_accuracy_pct"] > 93.0

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-test-time-compute-reasoning-router-thinking-budget-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-test-time-compute-reasoning-router-thinking-budget-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/test-time-compute-reasoning-router-budget-sizer-lab.html").exists()
