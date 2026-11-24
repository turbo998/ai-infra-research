from experiments.enterprise_aviation_airline_irops_solver_apron_vision_mro_sizing.run import (
    evaluate_aviation_airline_irops_sizing,
)


def test_aviation_airline_irops_sizing():
    r = evaluate_aviation_airline_irops_sizing()
    assert r["deliverable_id"] == "#621"
    assert r["irops_recovery_seconds"] == 4.2
    assert r["column_gen_mip_violation_pct"] == 0.0
    assert r["annual_delay_and_aog_savings_usd"] == 19400000
    assert r["self_hosted_monthly_usd"] == 32400
    assert r["compute_tco_savings_pct"] == 87.5
