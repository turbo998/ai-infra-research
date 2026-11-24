from experiments.enterprise_aec_bim_blueprint_vlm_smart_building_mpc_sizing.run import (
    evaluate_aec_bim_smart_building_sizing,
)


def test_aec_bim_smart_building_sizing():
    r = evaluate_aec_bim_smart_building_sizing()
    assert r["deliverable_id"] == "#619"
    assert r["vector_ifc_vlm_code_accuracy_pct"] == 99.6
    assert r["annual_hvac_savings_usd"] == 4144000
    assert r["ashrae_55_comfort_violation_pct"] == 0.0
    assert r["self_hosted_monthly_usd"] == 14300
    assert r["compute_tco_savings_pct"] == 88.1
