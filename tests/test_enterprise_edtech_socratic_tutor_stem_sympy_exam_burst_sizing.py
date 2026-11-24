from experiments.enterprise_edtech_socratic_tutor_stem_sympy_exam_burst_sizing.run import (
    evaluate_edtech_socratic_sizing,
)


def test_edtech_socratic_sizing():
    r = evaluate_edtech_socratic_sizing()
    assert r["deliverable_id"] == "#616"
    assert r["effective_prefill_tokens"] == 800
    assert r["sympy_cas_step_accuracy_pct"] == 99.4
    assert r["socratic_fsm_answer_leakage_pct"] == 0.0
    assert r["hybrid_monthly_usd"] == 30300
    assert r["tco_savings_pct"] == 79.4
